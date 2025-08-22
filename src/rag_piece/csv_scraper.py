"""
CSV scraper for One Piece Wiki content.
Extracts tabular data and converts it to CSV files.
"""

import requests
import re
import time
import logging
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from bs4 import BeautifulSoup
import unicodedata

from .utils import slugify, safe_file_operation


class CSVWikiScraper:
    """CSV scraper for One Piece Wiki content"""
    
    def __init__(self, request_delay: float = 1.0):
        self.request_delay = request_delay
        self.logger = logging.getLogger("rag_piece.csv_scraper")
        
        # API configuration
        self.api_base = "https://onepiece.fandom.com/api.php"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RAG-Piece/1.0 (Educational Research Tool)'
        })
    
    def scrape_article_to_csv(self, article_name: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        Scrape a single article and convert tabular data to CSV files.
        
        Args:
            article_name: Name of the article to scrape
        
        Returns:
            Tuple of (csv_files_created, scraping_metadata)
        """
        try:
            self.logger.info(f"Scraping article for CSV: {article_name}")
            
            # Create CSV files folder
            csv_folder = self._create_csv_folder(article_name)
            
            # Find and scrape sub-articles
            sub_articles = self._find_sub_articles(article_name)
            
            # Process main article and sub-articles
            all_tables, all_metadata = self._scrape_all_tables(article_name, sub_articles)
            
            # Convert tables to CSV files
            csv_files_created = self._convert_tables_to_csv(all_tables, csv_folder, article_name)
            
            # Create metadata
            metadata = self._create_scraping_metadata(
                article_name, sub_articles, all_tables, csv_files_created
            )
            
            self.logger.info(f"Completed CSV scraping: {article_name} ({len(csv_files_created)} CSV files created)")
            return csv_files_created, metadata
        
        except Exception as e:
            self.logger.error(f"Error scraping article {article_name} for CSV: {e}", exc_info=True)
            return [], {}
    
    def _create_csv_folder(self, article_name: str) -> Path:
        """Create CSV files folder for the article."""
        csv_folder = Path("csv_files") / slugify(article_name)
        csv_folder.mkdir(parents=True, exist_ok=True)
        return csv_folder
    
    def _find_sub_articles(self, article_name: str) -> List[str]:
        """Find sub-articles for the main article."""
        try:
            self.logger.info(f"Finding sub-articles for: {article_name}")
            
            # Search for sub-articles using API
            search_params = {
                'action': 'query',
                'format': 'json',
                'list': 'search',
                'srsearch': f'"{article_name}/"',
                'srnamespace': 0,
                'srlimit': 50
            }
            
            response = self._make_api_request(search_params)
            if not response:
                return []
            
            sub_articles = []
            search_results = response.get('query', {}).get('search', [])
            
            for result in search_results:
                title = result['title']
                if title.startswith(f"{article_name}/"):
                    sub_articles.append(title)
            
            # Add common sub-article patterns
            common_patterns = ['Gallery', 'Images', 'Pictures', 'History']
            for pattern in common_patterns:
                potential_sub = f"{article_name}/{pattern}"
                if potential_sub not in sub_articles:
                    # Check if it exists
                    if self._article_exists(potential_sub):
                        sub_articles.append(potential_sub)
            
            self.logger.info(f"Found {len(sub_articles)} sub-articles")
            return sub_articles
        
        except Exception as e:
            self.logger.error(f"Error finding sub-articles: {e}", exc_info=True)
            return []
    
    def _scrape_all_tables(self, main_article: str, sub_articles: List[str]) -> Tuple[List[Dict], List[str]]:
        """Scrape tables from main article and all sub-articles."""
        all_tables = []
        processed_content = set()
        
        # Process main article first
        tables = self._scrape_single_article_tables(main_article)
        all_tables.extend(tables)
        
        # Track processed content to avoid duplicates
        for table in tables:
            content_key = self._generate_content_key(str(table.get('data', '')))
            processed_content.add(content_key)
        
        # Process sub-articles
        for sub_article in sub_articles:
            self.logger.info(f"Processing sub-article tables: {sub_article}")
            
            tables = self._scrape_single_article_tables(sub_article)
            
            # Filter out duplicate content
            unique_tables = []
            for table in tables:
                content_key = self._generate_content_key(str(table.get('data', '')))
                if content_key not in processed_content:
                    unique_tables.append(table)
                    processed_content.add(content_key)
            
            all_tables.extend(unique_tables)
            
            time.sleep(self.request_delay)  # Rate limiting
        
        return all_tables, sub_articles
    
    def _scrape_single_article_tables(self, article_name: str) -> List[Dict]:
        """Scrape tables from a single article."""
        try:
            # Get article content
            parse_params = {
                'action': 'parse',
                'format': 'json',
                'page': article_name,
                'prop': 'text'
            }
            
            response = self._make_api_request(parse_params)
            if not response or 'parse' not in response:
                return []
            
            parse_data = response['parse']
            html_content = parse_data.get('text', {}).get('*', '')
            
            if not html_content:
                return []
            
            # Extract tables
            tables = self._extract_tables(html_content, article_name)
            
            return tables
        
        except Exception as e:
            self.logger.error(f"Error scraping article tables {article_name}: {e}", exc_info=True)
            return []
    
    def _extract_tables(self, html_content: str, article_name: str) -> List[Dict[str, Any]]:
        """Extract tables from HTML content."""
        try:
            # Clean the HTML content
            cleaned_content = self._clean_html_content(html_content)
            
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(cleaned_content, 'html.parser')
            
            # Find all tables
            tables = soup.find_all('table')
            self.logger.info(f"Found {len(tables)} total tables in {article_name}")
            
            extracted_tables = []
            for i, table in enumerate(tables):
                self.logger.debug(f"Processing table {i+1}: classes={table.get('class', [])}, id={table.get('id', 'none')}")
                
                # Debug: show first few cells of first few tables
                if i < 5:
                    rows = table.find_all('tr')
                    if rows:
                        first_row = rows[0]
                        cells = first_row.find_all(['td', 'th'])
                        cell_texts = [cell.get_text().strip()[:50] for cell in cells[:3]]
                        self.logger.debug(f"  Table {i+1} first row cells: {cell_texts}")
                
                if self._is_valid_table(table):
                    table_data = self._extract_table_data(table, i, article_name)
                    if table_data:
                        extracted_tables.append(table_data)
                        self.logger.info(f"Extracted table {i+1}: {table_data.get('table_title', 'Unknown')} ({table_data.get('row_count', 0)} rows, {table_data.get('column_count', 0)} columns)")
                else:
                    self.logger.debug(f"Skipped table {i+1}: not valid")
            
            return extracted_tables
        
        except Exception as e:
            self.logger.error(f"Error extracting tables: {e}", exc_info=True)
            return []
    
    def _clean_html_content(self, html_content: str) -> str:
        """Clean HTML content by removing unwanted elements."""
        # Remove script and style tags
        html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove navigation and reference sections - more aggressive patterns
        skip_patterns = [
            r'<div[^>]*class="[^"]*navbox[^"]*"[^>]*>.*?</div>',
            r'<div[^>]*class="[^"]*references[^"]*"[^>]*>.*?</div>',
            r'<div[^>]*id="[^"]*references[^"]*"[^>]*>.*?</div>',
            r'<div[^>]*class="[^"]*site-navigation[^"]*"[^>]*>.*?</div>',
            r'<div[^>]*class="[^"]*navigation[^"]*"[^>]*>.*?</div>',
            r'<div[^>]*class="[^"]*sidebar[^"]*"[^>]*>.*?</div>',
            r'<div[^>]*class="[^"]*infobox[^"]*"[^>]*>.*?</div>',
            r'<div[^>]*class="[^"]*toc[^"]*"[^>]*>.*?</div>',
            r'<div[^>]*class="[^"]*mw-editsection[^"]*"[^>]*>.*?</div>',
            r'<div[^>]*class="[^"]*mw-indicators[^"]*"[^>]*>.*?</div>',
            r'<div[^>]*class="[^"]*mw-body-header[^"]*"[^>]*>.*?</div>',
            r'<div[^>]*class="[^"]*mw-body-footer[^"]*"[^>]*>.*?</div>',
            r'<div[^>]*class="[^"]*mw-customtoggle[^"]*"[^>]*>.*?</div>',
            r'<div[^>]*class="[^"]*mw-collapsible[^"]*"[^>]*>.*?</div>',
            r'<div[^>]*class="[^"]*arc-navigation[^"]*"[^>]*>.*?</div>',
            r'<div[^>]*class="[^"]*arcnav[^"]*"[^>]*>.*?</div>',
            r'<div[^>]*class="[^"]*story-arc[^"]*"[^"]*navigation[^"]*"[^>]*>.*?</div>'
        ]
        
        for pattern in skip_patterns:
            html_content = re.sub(pattern, '', html_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove bibliography labels
        html_content = re.sub(r'\[[\d\s,]+\]', '', html_content)
        
        return html_content
    
    def _is_valid_table(self, table) -> bool:
        """Check if table is valid for processing."""
        if not table:
            return False
        
        # Check if table has rows
        rows = table.find_all('tr')
        if len(rows) < 2:  # Need at least header + 1 data row
            return False
        
        # Check if table has meaningful content
        first_row = rows[0]
        cells = first_row.find_all(['td', 'th'])
        
        # Allow portrait gallery tables with single column
        table_classes = table.get('class', [])
        if 'portrait-gallery' in table_classes:
            if len(cells) < 1:  # Portrait galleries can have just 1 column
                self.logger.debug(f"Portrait gallery table skipped: not enough columns ({len(cells)})")
                return False
        elif len(cells) < 2:  # Other tables need at least 2 columns
            self.logger.debug(f"Table skipped: not enough columns ({len(cells)})")
            return False
        
        # Skip navigation and utility tables
        skip_classes = ['nav', 'navigation', 'sidebar', 'infobox', 'toc', 'mw-editsection', 
                       'mw-indicators', 'mw-body-header', 'mw-body-footer', 'mw-customtoggle', 
                       'mw-collapsible', 'site-navigation', 'collapsible', 'navibox', 'toccolours',
                       'arc-navigation', 'arcnav', 'story-arc']
        
        # Allow portrait gallery tables (they contain character data)
        if 'portrait-gallery' in table_classes:
            self.logger.debug(f"Allowing portrait gallery table: {table_classes}")
        elif any(skip in str(cls).lower() for cls in table_classes for skip in skip_classes):
            self.logger.debug(f"Table skipped due to class: {table_classes}")
            return False
        
        # Skip tables with very short content (likely navigation)
        total_content_length = 0
        for row in rows[:3]:  # Check first 3 rows
            cells = row.find_all(['td', 'th'])
            for cell in cells:
                cell_text = cell.get_text().strip()
                total_content_length += len(cell_text)
        
        if total_content_length < 30:  # Reduced threshold to be less strict
            self.logger.debug(f"Table skipped due to short content: {total_content_length} chars")
            return False
        
        return True
    
    def _extract_table_data(self, table, table_index: int, article_name: str) -> Optional[Dict[str, Any]]:
        """Extract data from a table."""
        try:
            rows = table.find_all('tr')
            if not rows:
                return None
            
            # Extract headers from first row
            headers = []
            first_row = rows[0]
            header_cells = first_row.find_all(['th', 'td'])
            
            for cell in header_cells:
                header_text = self._clean_cell_text(cell.get_text())
                if header_text:
                    headers.append(header_text)
                else:
                    headers.append(f"Column_{len(headers) + 1}")
            
            # Extract data rows
            data_rows = []
            for row in rows[1:]:  # Skip header row
                cells = row.find_all(['td', 'th'])
                row_data = []
                
                for cell in cells:
                    cell_text = self._clean_cell_text(cell.get_text())
                    row_data.append(cell_text)
                
                # Pad row if it has fewer cells than headers
                while len(row_data) < len(headers):
                    row_data.append("")
                
                # Truncate row if it has more cells than headers
                row_data = row_data[:len(headers)]
                
                if any(cell.strip() for cell in row_data):  # Only add non-empty rows
                    data_rows.append(row_data)
            
            if not data_rows:
                return None
            
            # Create table data structure
            table_data = {
                'article_source': article_name,
                'table_index': table_index,
                'table_title': self._extract_table_title(table, article_name, table_index),
                'headers': headers,
                'data': data_rows,
                'row_count': len(data_rows),
                'column_count': len(headers)
            }
            
            return table_data
        
        except Exception as e:
            self.logger.warning(f"Error extracting table data: {e}")
            return None
    
    def _clean_cell_text(self, text: str) -> str:
        """Clean text from table cells."""
        if not text:
            return ""
        
        # Remove navigation symbols and brackets
        text = re.sub(r'\[v\s*·\s*e\s*·\s*\?\]', '', text)  # Remove [v · e · ?]
        text = re.sub(r'\[.*?\]', '', text)  # Remove any remaining brackets
        
        # Clean up special status markers
        # Replace symbols with readable text
        text = text.replace('†', '')  # Remove deceased symbol (often redundant)
        text = text.replace('≠', '')  # Remove non-canon symbol (often redundant) 
        text = text.replace('‡', '')  # Remove former symbol (often redundant)
        text = text.replace('*', '')  # Remove special symbol (often redundant)
        
        # Clean up status text patterns that might be redundant
        text = re.sub(r'\(Unknown status\)\s*\(Unknown Status\)', '(Unknown Status)', text)
        text = re.sub(r'\(Hysteria only\)', '(Special)', text)
        text = re.sub(r'\(Memoria only\)', '(Special)', text)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Try to separate concatenated names (when multiple capitalized words run together)
        # This regex finds patterns like "NameAnotherName" and adds spaces
        text = re.sub(r'([a-z])([A-Z])', r'\1, \2', text)
        
        # Clean up duplicate status markers
        text = re.sub(r'\(Deceased\)\s*\(Deceased\)', '(Deceased)', text)
        text = re.sub(r'\(Non-Canon\)\s*\(Non-Canon\)', '(Non-Canon)', text)
        text = re.sub(r'\(Former\)\s*\(Former\)', '(Former)', text)
        text = re.sub(r'\(Special\)\s*\(Special\)', '(Special)', text)
        text = re.sub(r'\(Unknown Status\)\s*\(Unknown Status\)', '(Unknown Status)', text)
        
        # Clean up multiple spaces and commas
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r',\s*,', ',', text)  # Remove duplicate commas
        text = re.sub(r',\s*$', '', text)  # Remove trailing comma
        
        # Remove special characters that might cause CSV issues
        text = text.replace('"', '""')  # Escape quotes for CSV
        
        return text
    
    def _extract_table_title(self, table, article_name: str, table_index: int) -> str:
        """Extract a meaningful title for the table."""
        # Try to find a caption
        caption = table.find('caption')
        if caption:
            caption_text = caption.get_text().strip()
            if caption_text:
                return caption_text
        
        # Try to find a heading before the table
        prev_element = table.find_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if prev_element:
            heading_text = prev_element.get_text().strip()
            if heading_text:
                return heading_text
        
        # Try to find a paragraph before the table
        prev_para = table.find_previous('p')
        if prev_para:
            para_text = prev_para.get_text().strip()
            if para_text and len(para_text) < 100:  # Short paragraph might be a title
                return para_text
        
        # Try to find a span or div with class that might indicate content
        prev_span = table.find_previous(['span', 'div'])
        if prev_span:
            span_text = prev_span.get_text().strip()
            if span_text and len(span_text) < 100 and not any(skip in span_text.lower() for skip in ['navigation', 'site', 'nav']):
                return span_text
        
        # Fallback to article name + table number
        return f"{article_name}_Table_{table_index + 1}"
    
    def _convert_tables_to_csv(self, tables: List[Dict], csv_folder: Path, article_name: str) -> List[str]:
        """Convert extracted tables to CSV files."""
        csv_files_created = []
        
        for table in tables:
            try:
                csv_filename = self._create_csv_filename(table, article_name)
                csv_filepath = csv_folder / csv_filename
                
                if self._write_table_to_csv(table, csv_filepath):
                    csv_files_created.append(str(csv_filepath))
                    self.logger.info(f"Created CSV: {csv_filename}")
                
            except Exception as e:
                self.logger.error(f"Error creating CSV for table: {e}")
        
        return csv_files_created
    
    def _create_csv_filename(self, table: Dict, article_name: str) -> str:
        """Create a filename for the CSV file."""
        table_title = table.get('table_title', 'Unknown')
        
        # Clean the title for filename
        clean_title = re.sub(r'[^\w\s-]', '', table_title)
        clean_title = re.sub(r'[-\s]+', '_', clean_title)
        clean_title = clean_title.strip('_')
        
        # Limit length and add article name
        if len(clean_title) > 50:
            clean_title = clean_title[:50]
        
        filename = f"{slugify(article_name)}_{clean_title}.csv"
        return filename
    
    def _write_table_to_csv(self, table: Dict, csv_filepath: Path) -> bool:
        """Write table data to a CSV file."""
        try:
            headers = table.get('headers', [])
            data = table.get('data', [])
            
            if not headers or not data:
                return False
            
            with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write headers
                writer.writerow(headers)
                
                # Write data rows
                for row in data:
                    writer.writerow(row)
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error writing CSV file: {e}")
            return False
    
    def _make_api_request(self, params: Dict) -> Optional[Dict]:
        """Make API request with error handling."""
        try:
            response = self.session.get(self.api_base, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"API request failed: {e}")
            return None
    
    def _article_exists(self, article_name: str) -> bool:
        """Check if an article exists."""
        params = {
            'action': 'query',
            'format': 'json',
            'titles': article_name
        }
        
        response = self._make_api_request(params)
        if not response:
            return False
        
        pages = response.get('query', {}).get('pages', {})
        return not any('-1' in str(page_id) for page_id in pages.keys())
    
    def _generate_content_key(self, content: str) -> str:
        """Generate a key for content deduplication."""
        import hashlib
        # Use first 200 characters for deduplication
        key_content = content.strip()[:200]
        return hashlib.md5(key_content.encode()).hexdigest()
    
    def _create_scraping_metadata(self, article_name: str, sub_articles: List[str],
                                 tables: List[Dict], csv_files: List[str]) -> Dict[str, Any]:
        """Create metadata about the CSV scraping process."""
        return {
            'article_name': article_name,
            'sub_articles': sub_articles,
            'scraping_timestamp': datetime.now().isoformat(),
            'total_tables_found': len(tables),
            'total_csv_files_created': len(csv_files),
            'csv_folder': f'csv_files/{slugify(article_name)}',
            'table_summary': [
                {
                    'table_title': table.get('table_title', 'Unknown'),
                    'rows': table.get('row_count', 0),
                    'columns': table.get('column_count', 0)
                }
                for table in tables
            ]
        }
