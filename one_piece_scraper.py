#!/usr/bin/env python3
"""
One Piece Wiki Scraper
Fetches articles from One Piece Wiki and saves content, tables, and images locally.
"""

import requests
import json
import csv
import os
import re
import time
from urllib.parse import urljoin, urlparse
from pathlib import Path
from datetime import datetime
import unicodedata


def slugify(text):
    """Convert text to filesystem-safe string."""
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    # Convert to lowercase and replace spaces/special chars with underscores
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '_', text)
    return text.strip('_')


def fetch_wiki_content(article_name):
    """Fetch article content from One Piece Wiki using MediaWiki API."""
    base_url = "https://onepiece.fandom.com/api.php"
    
    # Get article content - remove section=0 to get the entire article
    params = {
        'action': 'parse',
        'page': article_name,
        'format': 'json',
        'prop': 'sections|text|images'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'error' in data:
            print(f"Error fetching {article_name}: {data['error']['info']}")
            return None
            
        return data['parse']
    except Exception as e:
        print(f"Failed to fetch {article_name}: {e}")
        return None


def extract_sections(content_html):
    """Extract subsections from HTML content."""
    # Find subsection headers (h3) instead of main sections (h2)
    section_pattern = r'<h3[^>]*>(.*?)</h3>'
    sections = re.findall(section_pattern, content_html, re.DOTALL)
    
    # Extract content between subsections
    section_content = []
    for i, section_title in enumerate(sections):
        # Skip references and navigation sections
        title_lower = section_title.lower().strip()
        if any(skip_word in title_lower for skip_word in ['references', 'site navigation', 'external links', 'notes']):
            continue
            
        # Find the start of this subsection
        start_tag = f'<h3[^>]*>{re.escape(section_title)}</h3>'
        start_match = re.search(start_tag, content_html)
        
        if start_match:
            start_pos = start_match.end()  # Start after the header
            
            # Find the end (next h3 subsection or end of content)
            if i + 1 < len(sections):
                next_section_title = sections[i + 1]
                next_start_tag = f'<h3[^>]*>{re.escape(next_section_title)}</h3>'
                end_match = re.search(next_start_tag, content_html[start_pos:])
                if end_match:
                    end_pos = start_pos + end_match.start()
                else:
                    end_pos = len(content_html)
            else:
                end_pos = len(content_html)
            
            # Extract subsection content
            section_html = content_html[start_pos:end_pos]
            
            # Clean HTML tags but preserve line breaks for readability
            clean_text = re.sub(r'<br\s*/?>', '\n', section_html)
            clean_text = re.sub(r'<[^>]+>', '', section_html)
            clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text)  # Clean up multiple line breaks
            clean_text = clean_text.strip()
            
            if clean_text:
                section_content.append({
                    'id': f'subsection_{i}',
                    'title': section_title.strip(),
                    'content': clean_text
                })
    
    # If no h3 subsections found, try to find h2 sections as fallback
    if not section_content:
        h2_pattern = r'<h2[^>]*>(.*?)</h2>'
        h2_sections = re.findall(h2_pattern, content_html, re.DOTALL)
        
        for i, section_title in enumerate(h2_sections):
            # Skip references and navigation sections
            title_lower = section_title.lower().strip()
            if any(skip_word in title_lower for skip_word in ['references', 'site navigation', 'external links', 'notes']):
                continue
                
            # Find the start of this section
            start_tag = f'<h2[^>]*>{re.escape(section_title)}</h2>'
            start_match = re.search(start_tag, content_html)
            
            if start_match:
                start_pos = start_match.end()
                
                # Find the end (next h2 section or end of content)
                if i + 1 < len(h2_sections):
                    next_section_title = h2_sections[i + 1]
                    next_start_tag = f'<h2[^>]*>{re.escape(next_section_title)}</h2>'
                    end_match = re.search(next_start_tag, content_html[start_pos:])
                    if end_match:
                        end_pos = start_pos + end_match.start()
                    else:
                        end_pos = len(content_html)
                else:
                    end_pos = len(content_html)
                
                section_html = content_html[start_pos:end_pos]
                clean_text = re.sub(r'<br\s*/?>', '\n', section_html)
                clean_text = re.sub(r'<[^>]+>', '', section_html)
                clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text)
                clean_text = clean_text.strip()
                
                if clean_text:
                    section_content.append({
                        'id': f'section_{i}',
                        'title': section_title.strip(),
                        'content': clean_text
                    })
    
    # If still no sections found, treat the entire content as one section
    if not section_content:
        clean_text = re.sub(r'<br\s*/?>', '\n', content_html)
        clean_text = re.sub(r'<[^>]+>', '', clean_text)
        clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text)
        clean_text = clean_text.strip()
        
        if clean_text:
            section_content.append({
                'id': 'content_0',
                'title': 'Article Content',
                'content': clean_text
            })
    
    return section_content


def extract_tables(content_html):
    """Extract tables from HTML content."""
    # Look for tables with captions or headers that can serve as labels
    table_pattern = r'(?:<caption[^>]*>(.*?)</caption>)?.*?<table[^>]*>(.*?)</table>'
    tables = re.findall(table_pattern, content_html, re.DOTALL)
    
    extracted_tables = []
    for i, (caption, table_html) in enumerate(tables):
        # Extract rows
        row_pattern = r'<tr[^>]*>(.*?)</tr>'
        rows = re.findall(row_pattern, table_html, re.DOTALL)
        
        table_data = []
        for row in rows:
            # Extract cells
            cell_pattern = r'<t[dh][^>]*>(.*?)</t[dh]>'
            cells = re.findall(cell_pattern, row, re.DOTALL)
            
            if cells:
                # Clean cell content
                clean_cells = []
                for cell in cells:
                    clean_cell = re.sub(r'<[^>]+>', '', cell)
                    clean_cell = re.sub(r'\s+', ' ', clean_cell).strip()
                    clean_cells.append(clean_cell)
                table_data.append(clean_cells)
        
        if table_data:
            # Try to find a meaningful label for the table
            label = None
            
            # Use caption if available
            if caption:
                label = re.sub(r'<[^>]+>', '', caption).strip()
            
            # If no caption, try to find a header above the table
            if not label:
                # Look for headers (h1-h6) or strong text before the table
                before_table = content_html[:content_html.find(table_html)]
                header_pattern = r'<(?:h[1-6]|strong)[^>]*>(.*?)</(?:h[1-6]|strong)>'
                headers = re.findall(header_pattern, before_table, re.DOTALL)
                if headers:
                    # Use the last header found before the table
                    label = re.sub(r'<[^>]+>', '', headers[-1]).strip()
            
            # If still no label, use the first row as a potential label
            if not label and table_data and table_data[0]:
                first_row_text = ' '.join(table_data[0])
                if len(first_row_text) < 100:  # Only use if it's not too long
                    label = first_row_text
            
            # Clean up the label
            if label:
                label = re.sub(r'\s+', ' ', label).strip()
                # Remove common wiki formatting
                label = re.sub(r'\[.*?\]', '', label)  # Remove wiki links
                label = re.sub(r'\{.*?\}', '', label)  # Remove wiki templates
                label = re.sub(r'[^\w\s\-_]', '', label)  # Keep only alphanumeric, spaces, hyphens, underscores
                label = label.strip()
            
            extracted_tables.append({
                'data': table_data,
                'label': label or f'Table_{i+1}'
            })
    
    return extracted_tables


def extract_images(content_html):
    """Extract image URLs from HTML content."""
    img_pattern = r'<img[^>]*src="([^"]*)"[^>]*alt="([^"]*)"[^>]*width="([^"]*)"[^>]*height="([^"]*)"[^>]*>'
    images = re.findall(img_pattern, content_html)
    
    # Filter for One Piece Wiki images with size > 20x20
    wiki_images = []
    for i, (img_url, alt_text, width, height) in enumerate(images):
        if 'static.wikia.nocookie.net' in img_url or 'vignette.wikia.nocookie.net' in img_url:
            # Skip the first image of the article
            if i == 0:
                continue
                
            # Check if dimensions are available and > 20x20
            try:
                w = int(width) if width else 0
                h = int(height) if height else 0
                if w > 20 and h > 20:
                    # Use alt text as label if available, otherwise use URL
                    label = alt_text.strip() if alt_text.strip() else f"Image_{len(wiki_images)+1}"
                    wiki_images.append({
                        'url': img_url,
                        'label': label,
                        'width': w,
                        'height': h
                    })
            except (ValueError, TypeError):
                # If dimensions can't be parsed, assume it's large enough
                label = alt_text.strip() if alt_text.strip() else f"Image_{len(wiki_images)+1}"
                wiki_images.append({
                    'url': img_url,
                    'label': label,
                    'width': 0,
                    'height': 0
                })
    
    return wiki_images


def download_image(img_url, save_path):
    """Download image from URL."""
    try:
        response = requests.get(img_url, timeout=30)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"Failed to download image {img_url}: {e}")
        return False


def save_csv(data, file_path):
    """Save table data as CSV."""
    try:
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(data)
        return True
    except Exception as e:
        print(f"Failed to save CSV {file_path}: {e}")
        return False


def save_text(content, file_path):
    """Save text content to file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Failed to save text {file_path}: {e}")
        return False


def scrape_article(article_name):
    """Main function to scrape a single article."""
    print(f"Scraping article: {article_name}")
    
    # Fetch content
    content_data = fetch_wiki_content(article_name)
    if not content_data:
        return False
    
    # Create data folder and article folder
    data_folder = Path("data")
    data_folder.mkdir(exist_ok=True)
    safe_name = slugify(article_name)
    article_folder = data_folder / safe_name
    article_folder.mkdir(exist_ok=True)
    
    # Extract content
    sections = extract_sections(content_data['text']['*'])
    tables = extract_tables(content_data['text']['*'])
    images = extract_images(content_data['text']['*'])
    
    created_files = []
    
    # Save sections
    for i, section in enumerate(sections, 1):
        filename = f"{i:02d}_{slugify(section['title'])}.txt"
        file_path = article_folder / filename
        
        if save_text(section['content'], file_path):
            created_files.append(filename)
            print(f"  Saved section: {filename}")
    
    # Save tables
    for i, table_info in enumerate(tables, 1):
        # Create filename from label, sanitized for filesystem
        safe_label = slugify(table_info['label'])
        filename = f"{safe_label}.csv"
        file_path = article_folder / filename
        
        if save_csv(table_info['data'], file_path):
            created_files.append(filename)
            print(f"  Saved table: {filename}")
    
    # Download images
    for i, img_data in enumerate(images, 1):
        # Create filename from label, sanitized for filesystem
        safe_label = slugify(img_data['label'])
        filename = f"{safe_label}.png"
        file_path = article_folder / filename
        
        if download_image(img_data['url'], file_path):
            created_files.append(filename)
            print(f"  Downloaded image: {filename} ({img_data['width']}x{img_data['height']})")
        
        # Rate limiting
        time.sleep(1)
    
    # Save metadata
    metadata = {
        'article_name': article_name,
        'article_url': f"https://onepiece.fandom.com/wiki/{article_name.replace(' ', '_')}",
        'download_timestamp': datetime.now().isoformat(),
        'created_files': created_files
    }
    
    metadata_path = article_folder / 'metadata.json'
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"  Saved metadata: metadata.json")
    except Exception as e:
        print(f"Failed to save metadata: {e}")
    
    print(f"Completed scraping: {article_name}")
    return True


def main():
    """Main function to scrape all articles."""
    # Hard-coded list of One Piece Wiki articles
    articles = [
        "Monkey D. Luffy",
        "Roronoa Zoro",
        "Nami"
    ]
    
    print("One Piece Wiki Scraper")
    print("=" * 30)
    
    successful = 0
    failed = 0
    
    for article in articles:
        try:
            if scrape_article(article):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error processing {article}: {e}")
            failed += 1
        
        # Rate limiting between articles
        time.sleep(1)
    
    print("\n" + "=" * 30)
    print(f"Scraping completed!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
