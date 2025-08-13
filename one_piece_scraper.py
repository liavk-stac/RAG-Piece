#!/usr/bin/env python3
"""One Piece Wiki Scraper - Fetches articles from One Piece Wiki and saves content locally."""

import requests, json, csv, re, time, shutil
from pathlib import Path
from datetime import datetime
import unicodedata


def slugify(text):
    """Convert text to filesystem-safe string."""
    text = unicodedata.normalize('NFKD', text)
    return re.sub(r'[-\s]+', '_', re.sub(r'[^\w\s-]', '', text)).strip('_')


def fetch_wiki_content(article_name):
    """Fetch article content from One Piece Wiki using MediaWiki API."""
    try:
        response = requests.get("https://onepiece.fandom.com/api.php", 
                              params={'action': 'parse', 'page': article_name, 'format': 'json', 'prop': 'text'}, 
                              timeout=30)
        response.raise_for_status()
        return response.json().get('parse')
    except Exception as e:
        print(f"Failed to fetch {article_name}: {e}")
        return None


def clean_text(text):
    """Clean text by removing wiki artifacts and formatting."""
    # Remove HTML tags
    clean = re.sub(r'<[^>]+>', '', text)
    
    # Remove bibliography labels and reference citations [1], [2], [42], etc.
    clean = re.sub(r'\[\d+\]', '', clean)
    
    # Remove other common wiki artifacts
    clean = re.sub(r'\[edit\]', '', clean)
    clean = re.sub(r'\[citation needed\]', '', clean)
    clean = re.sub(r'\[who\?\]', '', clean)
    clean = re.sub(r'\[when\?\]', '', clean)
    clean = re.sub(r'\[where\?\]', '', clean)
    clean = re.sub(r'\[clarification needed\]', '', clean)
    
    # Remove MediaWiki templates and functions
    clean = re.sub(r'\{\{[^}]+\}\}', '', clean)
    clean = re.sub(r'\{\|[^}]+\|\}', '', clean)
    
    # Remove external link formatting
    clean = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean)
    
    # Remove multiple spaces and normalize whitespace
    clean = re.sub(r'\s+', ' ', clean)
    
    return clean.strip()


def extract_sections(content_html):
    """Extract sections with hierarchical structure (h2 only) and MediaWiki sections."""
    print(f"Debug: Found {len(re.findall(r'<h([1-6])[^>]*>', content_html))} headers")
    
    section_content = []
    seen_titles = set()
    
    def clean_title(title):
        """Clean and validate section title."""
        clean = re.sub(r'<[^>]+>|\[.*?\]|\{.*?\}', '', title)
        clean = re.sub(r'\s+', ' ', clean).strip()
        return re.sub(r'[^\w\s\-_]', '', clean).strip()
    
    def is_valid_title(title):
        """Check if title is valid for filename."""
        clean = clean_title(title)
        return (2 <= len(clean) <= 100 and 
                clean not in seen_titles and 
                not any(word in clean.lower() for word in ['span', 'class', 'id', 'href', 'title']))
    
    # Process MediaWiki sections
    mw_sections = re.findall(r'<span[^>]*class="mw-headline"[^>]*id="([^"]*)"[^>]*>([^<]*)</span>', content_html)
    if mw_sections:
        print(f"Found {len(mw_sections)} MediaWiki sections")
        for i, (section_id, section_title) in enumerate(mw_sections):
            if any(word in section_title.lower() for word in ['references', 'site navigation', 'external links', 'notes']):
                continue
            
            clean_title_text = clean_title(section_title)
            if not is_valid_title(clean_title_text):
                continue
            
            # Find section boundaries
            start_tag = f'<span[^>]*class="mw-headline"[^>]*id="{re.escape(section_id)}"[^>]*>{re.escape(section_title)}</span>'
            start_match = re.search(start_tag, content_html)
            if not start_match:
                continue
            
            start_pos = start_match.end()
            end_pos = len(content_html)
            
            if i + 1 < len(mw_sections):
                next_section_id, next_section_title = mw_sections[i + 1]
                next_start_tag = f'<span[^>]*class="mw-headline"[^>]*id="{re.escape(next_section_id)}"[^>]*>{re.escape(next_section_title)}</span>'
                end_match = re.search(next_start_tag, content_html[start_pos:])
                if end_match:
                    end_pos = start_pos + end_match.start()
            
            section_html = content_html[start_pos:end_pos]
            clean_text_content = clean_text(section_html)
            
            if clean_text_content:
                seen_titles.add(clean_title_text)
                section_content.append({'combined_title': clean_title_text, 'content': clean_text_content})
    
    # Process H2 sections
    h2_sections = re.findall(r'<h2[^>]*>(.*?)</h2>', content_html, re.DOTALL)
    if h2_sections:
        print(f"Found {len(h2_sections)} H2 sections")
        for h2_idx, h2_title in enumerate(h2_sections):
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
            
            h2_section_html = content_html[h2_start_pos:h2_end_pos]
            clean_text_content = clean_text(h2_section_html)
            
            if clean_text_content:
                seen_titles.add(clean_h2_title)
                section_content.append({'combined_title': clean_h2_title, 'content': clean_text_content})
    
    if not section_content:
        section_content.append({'combined_title': 'Article Content', 'content': clean_text(content_html)})
    
    return section_content


def extract_tables(content_html):
    """Extract tables from HTML content."""
    tables = re.findall(r'(?:<caption[^>]*>(.*?)</caption>)?.*?<table[^>]*>(.*?)</table>', content_html, re.DOTALL)
    extracted_tables = []
    
    for caption, table_html in tables:
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL)
        table_data = []
        
        for row in rows:
            cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, re.DOTALL)
            if cells:
                table_data.append([clean_text(cell) for cell in cells])
        
        if table_data:
            label = None
            if caption:
                label = clean_text(caption)
            elif table_data and table_data[0]:
                first_row_text = ' '.join(table_data[0])
                if len(first_row_text) < 100:
                    label = first_row_text
            
            if label:
                label = re.sub(r'[^\w\s\-_]', '', re.sub(r'\{.*?\}', '', re.sub(r'\[.*?\]', '', label))).strip()
            
            extracted_tables.append({'data': table_data, 'label': label or 'Table'})
    
    return extracted_tables


def extract_images(content_html):
    """Extract image URLs from HTML content."""
    images = re.findall(r'<img[^>]*src="([^"]*)"[^>]*alt="([^"]*)"[^>]*width="([^"]*)"[^>]*height="([^"]*)"[^>]*>', content_html)
    wiki_images = []
    
    for i, (img_url, alt_text, width, height) in enumerate(images):
        if ('static.wikia.nocookie.net' in img_url or 'vignette.wikia.nocookie.net' in img_url) and i > 0:
            try:
                w, h = int(width) if width else 0, int(height) if height else 0
                if w > 20 and h > 20:
                    label = alt_text.strip() if alt_text.strip() else f"Image_{len(wiki_images)+1}"
                    wiki_images.append({'url': img_url, 'label': label, 'width': w, 'height': h})
            except (ValueError, TypeError):
                label = alt_text.strip() if alt_text.strip() else f"Image_{len(wiki_images)+1}"
                wiki_images.append({'url': img_url, 'label': label, 'width': 0, 'height': 0})
    
    return wiki_images


def scrape_article(article_name):
    """Main function to scrape a single article."""
    print(f"Scraping article: {article_name}")
    
    content_data = fetch_wiki_content(article_name)
    if not content_data:
        return False
    
    # Create folders
    article_folder = Path("data") / slugify(article_name)
    article_folder.mkdir(parents=True, exist_ok=True)
    
    # Extract content
    sections = extract_sections(content_data['text']['*'])
    tables = extract_tables(content_data['text']['*'])
    images = extract_images(content_data['text']['*'])
    
    created_files = []
    
    # Save sections
    for i, section in enumerate(sections, 1):
        filename = f"{i:02d}_{slugify(section['combined_title'])}.txt"
        try:
            with open(article_folder / filename, 'w', encoding='utf-8') as f:
                f.write(section['content'])
            created_files.append(filename)
            print(f"  Saved section: {filename}")
        except Exception as e:
            print(f"Failed to save {filename}: {e}")
    
    # Save tables
    for table_info in tables:
        filename = f"{slugify(table_info['label'])}.csv"
        try:
            with open(article_folder / filename, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerows(table_info['data'])
            created_files.append(filename)
            print(f"  Saved table: {filename}")
        except Exception as e:
            print(f"Failed to save {filename}: {e}")
    
    # Download images
    for img_data in images:
        filename = f"{slugify(img_data['label'])}.png"
        try:
            response = requests.get(img_data['url'], timeout=30)
            response.raise_for_status()
            with open(article_folder / filename, 'wb') as f:
                f.write(response.content)
            created_files.append(filename)
            print(f"  Downloaded image: {filename} ({img_data['width']}x{img_data['height']})")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
        time.sleep(1)
    
    # Save metadata
    try:
        metadata = {
            'article_name': article_name,
            'article_url': f"https://onepiece.fandom.com/wiki/{article_name.replace(' ', '_')}",
            'download_timestamp': datetime.now().isoformat(),
            'created_files': created_files
        }
        with open(article_folder / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"  Saved metadata: metadata.json")
    except Exception as e:
        print(f"Failed to save metadata: {e}")
    
    print(f"Completed scraping: {article_name}")
    return True


def main():
    """Main function to scrape all articles."""
    articles = ["Monkey D. Luffy", "Roronoa Zoro", "Nami"]
    
    print("One Piece Wiki Scraper")
    print("=" * 30)
    
    # Clear previous data
    data_folder = Path("data")
    if data_folder.exists():
        print("Clearing previous data folder...")
        try:
            shutil.rmtree(data_folder)
            print("  Previous data folder cleared successfully")
        except Exception as e:
            print(f"  Warning: Could not clear data folder: {e}")
    else:
        print("No previous data folder found, starting fresh")
    print()
    
    # Scrape articles
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
