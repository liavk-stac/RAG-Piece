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
from pathlib import Path
from datetime import datetime
import unicodedata


def slugify(text):
    """Convert text to filesystem-safe string."""
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '_', text)
    return text.strip('_')


def fetch_wiki_content(article_name):
    """Fetch article content from One Piece Wiki using MediaWiki API."""
    params = {
        'action': 'parse',
        'page': article_name,
        'format': 'json',
        'prop': 'text'
    }
    
    try:
        response = requests.get("https://onepiece.fandom.com/api.php", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data['parse'] if 'parse' in data else None
    except Exception as e:
        print(f"Failed to fetch {article_name}: {e}")
        return None


def extract_anchors(content_html):
    """Extract all anchor links to see what sections are available."""
    # Look for both named anchors and section links
    anchor_pattern = r'<a[^>]*name="([^"]*)"[^>]*>|<a[^>]*href="#([^"]*)"[^>]*>([^<]*)</a>|<span[^>]*id="([^"]*)"[^>]*>([^<]*)</span>'
    anchors = re.findall(anchor_pattern, content_html)
    
    # Also look for any div or span with id attributes
    id_pattern = r'<(?:div|span)[^>]*id="([^"]*)"[^>]*>([^<]*)</(?:div|span)>'
    id_elements = re.findall(id_pattern, content_html)
    
    # Look for section headers of any level
    header_pattern = r'<h([1-6])[^>]*>(.*?)</h[1-6]>'
    headers = re.findall(header_pattern, content_html, re.DOTALL)
    
    return {
        'anchors': anchors,
        'id_elements': id_elements,
        'headers': headers
    }


def clean_text_content(text):
    """Clean text content by removing wiki artifacts and formatting."""
    # Remove HTML tags
    clean = re.sub(r'<[^>]+>', '', text)
    
    # Remove bibliography labels and reference citations [1], [2], [42], etc.
    clean = re.sub(r'\[\d+\]', '', clean)
    
    # Remove other common wiki artifacts
    clean = re.sub(r'\[edit\]', '', clean)  # Remove edit links
    clean = re.sub(r'\[citation needed\]', '', clean)  # Remove citation needed tags
    clean = re.sub(r'\[who\?\]', '', clean)  # Remove who tags
    clean = re.sub(r'\[when\?\]', '', clean)  # Remove when tags
    clean = re.sub(r'\[where\?\]', '', clean)  # Remove where tags
    clean = re.sub(r'\[clarification needed\]', '', clean)  # Remove clarification tags
    
    # Remove MediaWiki templates and functions
    clean = re.sub(r'\{\{[^}]+\}\}', '', clean)  # Remove double-brace templates
    clean = re.sub(r'\{\|[^}]+\|\}', '', clean)  # Remove table templates
    
    # Remove external link formatting
    clean = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean)  # Convert [text](url) to just text
    
    # Remove multiple spaces and normalize whitespace
    clean = re.sub(r'\s+', ' ', clean)
    
    # Remove leading/trailing whitespace
    clean = clean.strip()
    
    return clean


def extract_sections(content_html):
    """Extract sections with hierarchical structure (h2 only) and MediaWiki sections."""
    # First, let's see what's actually in the HTML
    debug_info = extract_anchors(content_html)
    print(f"Debug: Found {len(debug_info['headers'])} headers, {len(debug_info['anchors'])} anchors, {len(debug_info['id_elements'])} id elements")
    
    # Show what headers we found
    for level, title in debug_info['headers'][:10]:  # Show first 10
        print(f"  H{level}: {title.strip()}")
    
    section_content = []
    skip_words = ['references', 'site navigation', 'external links', 'notes']
    seen_titles = set()  # Track seen titles to avoid duplicates
    
    def clean_title(title):
        """Clean HTML tags and extra content from section titles."""
        # Remove HTML tags
        clean = re.sub(r'<[^>]+>', '', title)
        # Remove extra whitespace and normalize
        clean = re.sub(r'\s+', ' ', clean).strip()
        # Remove common MediaWiki artifacts
        clean = re.sub(r'\[.*?\]', '', clean)  # Remove edit section brackets
        clean = re.sub(r'\{.*?\}', '', clean)  # Remove templates
        clean = re.sub(r'[^\w\s\-_]', '', clean)  # Keep only alphanumeric, spaces, hyphens, underscores
        return clean.strip()
    
    def is_valid_title(title):
        """Check if a title is valid for use as a filename."""
        clean = clean_title(title)
        # Skip if title is too short, too long, or contains obvious HTML artifacts
        if len(clean) < 2 or len(clean) > 100:
            return False
        if any(word in clean.lower() for word in ['span', 'class', 'id', 'href', 'title']):
            return False
        if clean in seen_titles:
            return False
        return True
    
    # Find MediaWiki sections (span class="mw-headline")
    mw_section_pattern = r'<span[^>]*class="mw-headline"[^>]*id="([^"]*)"[^>]*>([^<]*)</span>'
    mw_sections = re.findall(mw_section_pattern, content_html)
    
    if mw_sections:
        print(f"Found {len(mw_sections)} MediaWiki sections:")
        for section_id, section_title in mw_sections[:5]:  # Show first 5
            print(f"  MW: {section_title.strip()} (id: {section_id})")
        
        # Process MediaWiki sections
        for i, (section_id, section_title) in enumerate(mw_sections):
            section_title_lower = section_title.lower().strip()
            if any(skip_word in section_title_lower for skip_word in skip_words):
                continue
            
            # Clean and validate the title
            clean_section_title = clean_title(section_title)
            if not is_valid_title(clean_section_title):
                continue
            
            # Find the start of this section
            section_start_tag = f'<span[^>]*class="mw-headline"[^>]*id="{re.escape(section_id)}"[^>]*>{re.escape(section_title)}</span>'
            section_start_match = re.search(section_start_tag, content_html)
            if not section_start_match:
                continue
                
            section_start_pos = section_start_match.end()
            section_end_pos = len(content_html)
            
            # Find the end (next MediaWiki section or end of content)
            if i + 1 < len(mw_sections):
                next_section_id, next_section_title = mw_sections[i + 1]
                next_section_start_tag = f'<span[^>]*class="mw-headline"[^>]*id="{re.escape(next_section_id)}"[^>]*>{re.escape(next_section_title)}</span>'
                section_end_match = re.search(next_section_start_tag, content_html[section_start_pos:])
                if section_end_match:
                    section_end_pos = section_start_pos + section_end_match.start()
            
            section_html = content_html[section_start_pos:section_end_pos]
            clean_text = clean_text_content(section_html)
            
            if clean_text:
                seen_titles.add(clean_section_title)
                section_content.append({
                    'combined_title': clean_section_title,
                    'content': clean_text
                })
    
    # Also capture h2 sections (don't skip if MediaWiki sections were found)
    h2_pattern = r'<h2[^>]*>(.*?)</h2>'
    h2_sections = re.findall(h2_pattern, content_html, re.DOTALL)
    
    if h2_sections:
        print(f"Found {len(h2_sections)} H2 sections:")
        for h2_title in h2_sections[:5]:  # Show first 5
            print(f"  H2: {h2_title.strip()}")
        
        for h2_idx, h2_title in enumerate(h2_sections):
            # Clean and validate the title
            clean_h2_title = clean_title(h2_title)
            if not is_valid_title(clean_h2_title):
                continue
            
            h2_start_tag = f'<h2[^>]*>{re.escape(h2_title)}</h2>'
            h2_start_match = re.search(h2_start_tag, content_html)
            if not h2_start_match:
                continue
                
            h2_start_pos = h2_start_match.end()
            h2_end_pos = len(content_html)
            
            if h2_idx + 1 < len(h2_sections):
                next_h2_title = h2_sections[h2_idx + 1]
                next_h2_start_tag = f'<h2[^>]*>{re.escape(next_h2_title)}</h2>'
                h2_end_match = re.search(next_h2_start_tag, content_html[h2_start_pos:])
                if h2_end_match:
                    h2_end_pos = h2_start_pos + h2_end_match.start()
            
            # Extract the entire h2 section content (including any h3 subsections within it)
            h2_section_html = content_html[h2_start_pos:h2_end_pos]
            clean_text = clean_text_content(h2_section_html)
            
            if clean_text:
                seen_titles.add(clean_h2_title)
                section_content.append({
                    'combined_title': clean_h2_title,
                    'content': clean_text
                })
    
    if not section_content:
        clean_text = clean_text_content(content_html)
        if clean_text:
            section_content.append({
                'combined_title': 'Article Content',
                'content': clean_text
            })
    
    return section_content


def extract_tables(content_html):
    """Extract tables from HTML content."""
    table_pattern = r'(?:<caption[^>]*>(.*?)</caption>)?.*?<table[^>]*>(.*?)</table>'
    tables = re.findall(table_pattern, content_html, re.DOTALL)
    
    extracted_tables = []
    for caption, table_html in tables:
        row_pattern = r'<tr[^>]*>(.*?)</tr>'
        rows = re.findall(row_pattern, table_html, re.DOTALL)
        
        table_data = []
        for row in rows:
            cell_pattern = r'<t[dh][^>]*>(.*?)</t[dh]>'
            cells = re.findall(cell_pattern, row, re.DOTALL)
            if cells:
                clean_cells = [clean_text_content(cell) for cell in cells]
                table_data.append(clean_cells)
        
        if table_data:
            label = None
            if caption:
                label = clean_text_content(caption)
            
            if not label:
                before_table = content_html[:content_html.find(table_html)]
                header_pattern = r'<(?:h[1-6]|strong)[^>]*>(.*?)</(?:h[1-6]|strong)>'
                headers = re.findall(header_pattern, before_table, re.DOTALL)
                if headers:
                    label = clean_text_content(headers[-1])
            
            if not label and table_data and table_data[0]:
                first_row_text = ' '.join(table_data[0])
                if len(first_row_text) < 100:
                    label = first_row_text
            
            if label:
                label = re.sub(r'[^\w\s\-_]', '', re.sub(r'\{.*?\}', '', re.sub(r'\[.*?\]', '', label))).strip()
            
            extracted_tables.append({
                'data': table_data,
                'label': label or 'Table'
            })
    
    return extracted_tables


def extract_images(content_html):
    """Extract image URLs from HTML content."""
    img_pattern = r'<img[^>]*src="([^"]*)"[^>]*alt="([^"]*)"[^>]*width="([^"]*)"[^>]*height="([^"]*)"[^>]*>'
    images = re.findall(img_pattern, content_html)
    
    wiki_images = []
    for i, (img_url, alt_text, width, height) in enumerate(images):
        if ('static.wikia.nocookie.net' in img_url or 'vignette.wikia.nocookie.net' in img_url) and i > 0:
            try:
                w = int(width) if width else 0
                h = int(height) if height else 0
                if w > 20 and h > 20:
                    label = alt_text.strip() if alt_text.strip() else f"Image_{len(wiki_images)+1}"
                    wiki_images.append({
                        'url': img_url,
                        'label': label,
                        'width': w,
                        'height': h
                    })
            except (ValueError, TypeError):
                label = alt_text.strip() if alt_text.strip() else f"Image_{len(wiki_images)+1}"
                wiki_images.append({
                    'url': img_url,
                    'label': label,
                    'width': 0,
                    'height': 0
                })
    
    return wiki_images


def scrape_article(article_name):
    """Main function to scrape a single article."""
    print(f"Scraping article: {article_name}")
    
    content_data = fetch_wiki_content(article_name)
    if not content_data:
        return False
    
    data_folder = Path("data")
    data_folder.mkdir(exist_ok=True)
    article_folder = data_folder / slugify(article_name)
    article_folder.mkdir(exist_ok=True)
    
    sections = extract_sections(content_data['text']['*'])
    tables = extract_tables(content_data['text']['*'])
    images = extract_images(content_data['text']['*'])
    
    created_files = []
    
    # Save sections
    for i, section in enumerate(sections, 1):
        filename = f"{i:02d}_{slugify(section['combined_title'])}.txt"
        file_path = article_folder / filename
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(section['content'])
            created_files.append(filename)
            print(f"  Saved section: {filename}")
        except Exception as e:
            print(f"Failed to save {filename}: {e}")
    
    # Save tables
    for table_info in tables:
        safe_label = slugify(table_info['label'])
        filename = f"{safe_label}.csv"
        file_path = article_folder / filename
        
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(table_info['data'])
            created_files.append(filename)
            print(f"  Saved table: {filename}")
        except Exception as e:
            print(f"Failed to save {filename}: {e}")
    
    # Download images
    for img_data in images:
        safe_label = slugify(img_data['label'])
        filename = f"{safe_label}.png"
        file_path = article_folder / filename
        
        try:
            response = requests.get(img_data['url'], timeout=30)
            response.raise_for_status()
            with open(file_path, 'wb') as f:
                f.write(response.content)
            created_files.append(filename)
            print(f"  Downloaded image: {filename} ({img_data['width']}x{img_data['height']})")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
        
        time.sleep(1)
    
    # Save metadata
    metadata = {
        'article_name': article_name,
        'article_url': f"https://onepiece.fandom.com/wiki/{article_name.replace(' ', '_')}",
        'download_timestamp': datetime.now().isoformat(),
        'created_files': created_files
    }
    
    try:
        with open(article_folder / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"  Saved metadata: metadata.json")
    except Exception as e:
        print(f"Failed to save metadata: {e}")
    
    print(f"Completed scraping: {article_name}")
    return True


def clear_data_folder():
    """Clear the data folder before starting fresh scraping."""
    data_folder = Path("data")
    if data_folder.exists():
        print("Clearing previous data folder...")
        try:
            import shutil
            shutil.rmtree(data_folder)
            print("  Previous data folder cleared successfully")
        except Exception as e:
            print(f"  Warning: Could not clear data folder: {e}")
    else:
        print("No previous data folder found, starting fresh")


def main():
    """Main function to scrape all articles."""
    articles = ["Monkey D. Luffy", "Roronoa Zoro", "Nami"]
    
    print("One Piece Wiki Scraper")
    print("=" * 30)
    
    # Clear previous data before starting
    clear_data_folder()
    print()
    
    successful = failed = 0
    
    for article in articles:
        try:
            if scrape_article(article):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error processing {article}: {e}")
            failed += 1
        time.sleep(1)
    
    print("\n" + "=" * 30)
    print(f"Scraping completed! Successful: {successful}, Failed: {failed}")


if __name__ == "__main__":
    main()
