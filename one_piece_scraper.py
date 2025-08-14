#!/usr/bin/env python3
"""One Piece Wiki Scraper - Fetches articles from One Piece Wiki and saves content locally."""

import requests, json, csv, re, time, shutil
from pathlib import Path
from datetime import datetime
import unicodedata
from PIL import Image
import io


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
    # Remove HTML tags, citations, and wiki artifacts
    clean = re.sub(r'<[^>]+>|\[\d+\]|\[edit\]|\[citation needed\]|\[who\?\]|\[when\?\]|\[where\?\]|\[clarification needed\]', '', text)
    clean = re.sub(r'\{\{[^}]+\}\}|\{\|[^}]+\|\}', '', clean)  # Remove MediaWiki templates
    clean = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean)  # Remove external link formatting
    clean = re.sub(r'\s+', ' ', clean)  # Normalize whitespace
    return clean.strip()


def extract_sections(content_html):
    """Extract sections with hierarchical structure."""
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
    for h2_idx, h2_title in enumerate(h2_sections):
        if any(word in h2_title.lower() for word in ['references', 'site navigation', 'external links', 'notes']):
            continue
            
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
                label = clean_text(label)
                label = re.sub(r'[^\w\s\-_]', '', label).strip()
                if len(label) > 50:
                    label = label[:50].strip()
            
            extracted_tables.append({'data': table_data, 'label': label or 'Table'})
    
    return extracted_tables


def extract_images(content_html, max_img=20):
    """Extract image URLs from HTML content."""
    img_tags = re.findall(r'<img[^>]+>', content_html)
    wiki_images = []
    
    for img_tag in img_tags:
        if len(wiki_images) >= max_img:
            break
            
        # Extract attributes
        src_match = re.search(r'src="([^"]*)"', img_tag)
        alt_match = re.search(r'alt="([^"]*)"', img_tag)
        
        if not src_match:
            continue
            
        img_url = src_match.group(1)
        alt_text = alt_match.group(1) if alt_match else None
        
        # Support multiple wiki image domains
        if not any(domain in img_url for domain in [
            'static.wikia.nocookie.net', 'vignette.wikia.nocookie.net',
            'static.wikimedia.org', 'upload.wikimedia.org'
        ]):
            continue
        
        # Create meaningful label
        if alt_text and alt_text.strip():
            label = clean_text(alt_text.strip())
            label = re.sub(r'[^\w\s\-_]', '', label).strip()
            if len(label) > 50:
                label = label[:50].strip()
            if len(label) < 3:
                label = f"Image_{len(wiki_images)+1}"
        else:
            url_parts = img_url.split('/')
            if len(url_parts) > 2:
                filename = url_parts[-2]
                label = f"Image_{filename}_{len(wiki_images)+1}"
            else:
                label = f"Image_{len(wiki_images)+1}"
        
        wiki_images.append({'url': img_url, 'label': label, 'width': None, 'height': None})
    
    return wiki_images


def find_sub_articles(article_name):
    """Find all sub-articles related to the main article."""
    sub_articles = []
    
    try:
        # Search for sub-articles using the MediaWiki API
        search_params = {
            'action': 'query', 'list': 'search', 'srsearch': f'"{article_name}/"',
            'srnamespace': 0, 'srlimit': 50, 'format': 'json'
        }
        
        response = requests.get("https://onepiece.fandom.com/api.php", params=search_params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'query' in data and 'search' in data['query']:
            for result in data['query']['search']:
                title = result['title']
                if title.startswith(f"{article_name}/") and title != article_name:
                    sub_articles.append(title)
        
        # Check for common gallery and image sub-articles
        common_sub_articles = [
            f"{article_name}/Gallery", f"{article_name}/Images", 
            f"{article_name}/Pictures", f"{article_name}/Screenshots", f"{article_name}/Artwork"
        ]
        
        for common_sub in common_sub_articles:
            if common_sub not in sub_articles:
                test_content = fetch_wiki_content(common_sub)
                if test_content:
                    sub_articles.append(common_sub)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_sub_articles = []
        for article in sub_articles:
            if article not in seen:
                seen.add(article)
                unique_sub_articles.append(article)
        
        return unique_sub_articles
        
    except Exception as e:
        print(f"Warning: Could not search for sub-articles: {e}")
        return []


def scrape_article(article_name, scrape_csv_files=True, max_img=20):
    """Main function to scrape a single article and its sub-articles."""
    print(f"Scraping article: {article_name}")
    
    # Create main article folder
    article_folder = Path("data") / slugify(article_name)
    article_folder.mkdir(parents=True, exist_ok=True)
    
    # Find and scrape sub-articles first
    sub_articles = find_sub_articles(article_name)
    all_sections, all_tables, all_images, created_files = [], [], [], []
    processed_sections, processed_tables, processed_images = set(), set(), set()
    
    # Scrape main article
    main_content = fetch_wiki_content(article_name)
    if main_content:
        print(f"  Processing main article content...")
        main_sections = extract_sections(main_content['text']['*'])
        main_tables = extract_tables(main_content['text']['*']) if scrape_csv_files else []
        main_images = extract_images(main_content['text']['*'], max_img)
        
        # Add main article content to collections
        for section in main_sections:
            section_key = f"main_{section['combined_title']}"
            if section_key not in processed_sections:
                processed_sections.add(section_key)
                all_sections.append(section)
        
        for table in main_tables:
            table_key = f"main_{table['label']}"
            if table_key not in processed_tables:
                processed_tables.add(table_key)
                all_tables.append(table)
        
        for image in main_images:
            image_key = f"main_{image['label']}"
            if image_key not in processed_images:
                processed_images.add(image_key)
                all_images.append(image)
        
        table_info = f"{len(main_tables)} tables" if scrape_csv_files else "tables disabled"
        print(f"  Main article: {len(main_sections)} sections, {table_info}, {len(main_images)} images (limited to {max_img}, min size 100x100)")
    
    # Scrape sub-articles
    for sub_article in sub_articles:
        print(f"  Processing sub-article: {sub_article}")
        sub_content = fetch_wiki_content(sub_article)
        if sub_content:
            sub_sections = extract_sections(sub_content['text']['*'])
            sub_tables = extract_tables(sub_content['text']['*']) if scrape_csv_files else []
            sub_images = extract_images(sub_content['text']['*'], max_img)
            
            # Add sub-article content to collections (avoiding duplicates)
            for section in sub_sections:
                section_key = f"sub_{sub_article}_{section['combined_title']}"
                if section_key not in processed_sections:
                    processed_sections.add(section_key)
                    all_sections.append(section)
            
            for table in sub_tables:
                table_key = f"sub_{sub_article}_{table['label']}"
                if table_key not in processed_tables:
                    processed_tables.add(table_key)
                    all_tables.append(table)
            
            for image in sub_images:
                image_key = f"sub_{sub_article}_{image['label']}"
                if image_key not in processed_images:
                    processed_images.add(image_key)
                    all_images.append(image)
            
            table_info = f"{len(sub_tables)} tables" if scrape_csv_files else "tables disabled"
            print(f"    Sub-article: {len(sub_sections)} sections, {table_info}, {len(sub_images)} images (limited to {max_img}, min size 100x100)")
            time.sleep(1)  # Rate limiting
    
    # Save all sections
    for i, section in enumerate(all_sections, 1):
        filename = f"{i:02d}_{slugify(section['combined_title'])}.txt"
        try:
            with open(article_folder / filename, 'w', encoding='utf-8') as f:
                f.write(section['content'])
            created_files.append(filename)
            print(f"  Saved section: {filename}")
        except Exception as e:
            print(f"Failed to save {filename}: {e}")
    
    # Save all tables (only if CSV scraping is enabled)
    if scrape_csv_files and all_tables:
        for table_info in all_tables:
            filename = f"{slugify(table_info['label'])}.csv"
            try:
                with open(article_folder / filename, 'w', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerows(table_info['data'])
                created_files.append(filename)
                print(f"  Saved table: {filename}")
            except Exception as e:
                print(f"Failed to save {filename}: {e}")
    elif not scrape_csv_files:
        print("  CSV table extraction disabled - skipping table saving")
    
    # Download all images
    print(f"  Downloading {len(all_images)} images (limited to {max_img} per article, min size 100x100)...")
    downloaded_images = []
    for img_data in all_images:
        filename = f"{slugify(img_data['label'])}.png"
        try:
            response = requests.get(img_data['url'], timeout=30)
            response.raise_for_status()
            
            # Check image dimensions before saving
            try:
                img = Image.open(io.BytesIO(response.content))
                width, height = img.size
                
                # Only save images that are at least 100x100 pixels
                if width >= 100 and height >= 100:
                    with open(article_folder / filename, 'wb') as f:
                        f.write(response.content)
                    created_files.append(filename)
                    downloaded_images.append(filename)
                    print(f"    Downloaded: {filename} ({width}x{height}) - Label: {img_data['label']}")
                else:
                    print(f"    Skipped: {filename} ({width}x{height}) - too small, min size 100x100")
            except Exception as img_error:
                print(f"    Failed to check dimensions for {filename}: {img_error}")
                # If we can't check dimensions, save anyway
                with open(article_folder / filename, 'wb') as f:
                    f.write(response.content)
                created_files.append(filename)
                downloaded_images.append(filename)
                print(f"    Downloaded: {filename} (dimensions unknown) - Label: {img_data['label']}")
                
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
        time.sleep(1)
    
    print(f"    Successfully downloaded {len(downloaded_images)} images out of {len(all_images)} found")
    
    # Save metadata
    try:
        metadata = {
            'article_name': article_name,
            'article_url': f"https://onepiece.fandom.com/wiki/{article_name.replace(' ', '_')}",
            'sub_articles': sub_articles,
            'download_timestamp': datetime.now().isoformat(),
            'created_files': created_files,
            'csv_extraction_enabled': scrape_csv_files,
            'total_sections': len(all_sections),
            'total_tables': len(all_tables) if scrape_csv_files else 0,
            'total_images_found': len(all_images),
            'total_images_downloaded': len(downloaded_images)
        }
        with open(article_folder / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"  Saved metadata: metadata.json")
    except Exception as e:
        print(f"Failed to save metadata: {e}")
    
    print(f"Completed scraping: {article_name} (with {len(sub_articles)} sub-articles)")
    return True


def main():
    """Main function to scrape all articles."""
    # Configuration flags
    SCRAPE_CSV_FILES = False  # Set to False to skip CSV table extraction
    MAX_IMAGES = 20  # Maximum number of images to scrape per article
    
    articles = ["Arabasta Kingdom"]
    
    print("One Piece Wiki Scraper")
    print("=" * 30)
    print(f"Configuration:")
    print(f"  - Scrape CSV files (tables): {SCRAPE_CSV_FILES}")
    print(f"  - Maximum images per article: {MAX_IMAGES}")
    print()
    
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
            if scrape_article(article, scrape_csv_files=SCRAPE_CSV_FILES, max_img=MAX_IMAGES):
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
