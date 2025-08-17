#!/usr/bin/env python3
"""One Piece Wiki Scraper - Fetches articles from One Piece Wiki and saves content locally."""

import requests
import re
import time
import shutil
from pathlib import Path
from datetime import datetime
import unicodedata
from PIL import Image
import io

# Import RAG database components
from db_creator import RAGDatabase, RAGConfig


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


def scrape_article(article_name, rag_database, max_img=20):
    """Main function to scrape a single article and process directly into RAG database."""
    print(f"Scraping article: {article_name}")
    
    # Create images folder only (no text files saved)
    images_folder = Path("data") / "images" / slugify(article_name)
    images_folder.mkdir(parents=True, exist_ok=True)
    
    # Find and scrape sub-articles first
    sub_articles = find_sub_articles(article_name)
    all_sections, all_images, created_files = [], [], []
    processed_sections, processed_images = set(), set()
    all_chunks = []  # Collect all chunks for database
    
    # Scrape main article
    main_content = fetch_wiki_content(article_name)
    if main_content:
        print("  Processing main article content...")
        main_sections = extract_sections(main_content['text']['*'])
        main_images = extract_images(main_content['text']['*'], max_img)
        
        # Process sections directly into chunks (no file saving)
        main_chunks = rag_database.process_sections_directly(main_sections, article_name)
        all_chunks.extend(main_chunks)
        
        # Add main article content to collections
        for section in main_sections:
            section_key = f"main_{section['combined_title']}"
            if section_key not in processed_sections:
                processed_sections.add(section_key)
                all_sections.append(section)
        
        for image in main_images:
            image_key = f"main_{image['label']}"
            if image_key not in processed_images:
                processed_images.add(image_key)
                all_images.append(image)
        
        print(f"  Main article: {len(main_sections)} sections, {len(main_chunks)} chunks, {len(main_images)} images (limited to {max_img}, min size 100x100)")
    
    # Scrape sub-articles
    for sub_article in sub_articles:
        print(f"  Processing sub-article: {sub_article}")
        sub_content = fetch_wiki_content(sub_article)
        if sub_content:
            sub_sections = extract_sections(sub_content['text']['*'])
            sub_images = extract_images(sub_content['text']['*'], max_img)
            
            # Extract sub-article name (e.g., "Gallery" from "Arabasta Kingdom/Gallery")
            sub_article_name = sub_article.split('/')[-1] if '/' in sub_article else None
            
            # Process sections directly into chunks
            sub_chunks = rag_database.process_sections_directly(sub_sections, article_name, sub_article_name)
            all_chunks.extend(sub_chunks)
            
            # Add sub-article content to collections (avoiding duplicates)
            for section in sub_sections:
                section_key = f"sub_{sub_article}_{section['combined_title']}"
                if section_key not in processed_sections:
                    processed_sections.add(section_key)
                    all_sections.append(section)
            
            for image in sub_images:
                image_key = f"sub_{sub_article}_{image['label']}"
                if image_key not in processed_images:
                    processed_images.add(image_key)
                    all_images.append(image)
            
            print(f"    Sub-article: {len(sub_sections)} sections, {len(sub_chunks)} chunks, {len(sub_images)} images (limited to {max_img}, min size 100x100)")
            time.sleep(1)  # Rate limiting
    
    # Skip saving text files and CSV tables - content processed directly into chunks
    print(f"  Sections processed directly into {len(all_chunks)} chunks (no text files saved)")
    
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
                    with open(images_folder / filename, 'wb') as f:
                        f.write(response.content)
                    created_files.append(f"images/{slugify(article_name)}/{filename}")
                    downloaded_images.append(filename)
                    print(f"    Downloaded: {filename} ({width}x{height}) - Label: {img_data['label']}")
                else:
                    print(f"    Skipped: {filename} ({width}x{height}) - too small, min size 100x100")
            except Exception as img_error:
                print(f"    Failed to check dimensions for {filename}: {img_error}")
                # If we can't check dimensions, save anyway
                with open(images_folder / filename, 'wb') as f:
                    f.write(response.content)
                created_files.append(f"images/{slugify(article_name)}/{filename}")
                downloaded_images.append(filename)
                print(f"    Downloaded: {filename} (dimensions unknown) - Label: {img_data['label']}")
                
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
        time.sleep(1)
    
    print(f"    Successfully downloaded {len(downloaded_images)} images out of {len(all_images)} found")
    
    # Return chunks and metadata for database building
    scraping_metadata = {
        'article_name': article_name,
        'article_url': f"https://onepiece.fandom.com/wiki/{article_name.replace(' ', '_')}",
        'sub_articles': sub_articles,
        'download_timestamp': datetime.now().isoformat(),
        'total_sections': len(all_sections),
        'total_chunks': len(all_chunks),
        'total_images_found': len(all_images),
        'total_images_downloaded': len(downloaded_images),
        'images_folder': f'data/images/{slugify(article_name)}'
    }
    
    print(f"Completed scraping: {article_name} (with {len(sub_articles)} sub-articles, {len(all_chunks)} chunks)")
    return all_chunks, scraping_metadata



def main():
    """Main function to scrape articles and build RAG database."""
    # Configuration flags
    MAX_IMAGES = 20  # Maximum number of images to scrape per article
    
    articles = ["Arabasta Kingdom"]
    
    print("One Piece Wiki Scraper + RAG Database Builder")
    print("=" * 50)
    print("Configuration:")
    print(f"  - Maximum images per article: {MAX_IMAGES}")
    print("  - Direct processing: text → chunks → database (no intermediate files)")
    print()
    
    # Initialize RAG database
    rag_config = RAGConfig()
    rag_db = RAGDatabase(rag_config)
    
    print("RAG Configuration:")
    print(f"  - Chunking: {rag_config.MIN_CHUNK_SIZE}-{rag_config.MAX_CHUNK_SIZE} tokens (target: {rag_config.TARGET_CHUNK_SIZE})")
    print(f"  - Keywords: {rag_config.KEYWORDS_PER_CHUNK} per chunk using BM25 scoring")
    print(f"  - Embedding model: {rag_config.EMBEDDING_MODEL}")
    print()
    
    # Clear previous data (keep existing images if desired)
    data_folder = Path("data")
    if data_folder.exists():
        print("Clearing previous data folder...")
        try:
            # Remove everything except images folder if it exists
            for item in data_folder.iterdir():
                if item.name != "images":
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
            print("  Previous data cleared (kept images folder)")
        except Exception as e:
            print(f"  Warning: Could not clear data folder: {e}")
    else:
        print("No previous data folder found, starting fresh")
    print()
    
    # Scrape articles and collect all chunks
    all_chunks = []
    all_metadata = []
    successful = failed = 0
    
    for article in articles:
        try:
            chunks, metadata = scrape_article(article, rag_db, max_img=MAX_IMAGES)
            if chunks:
                all_chunks.extend(chunks)
                all_metadata.append(metadata)
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error processing {article}: {e}")
            failed += 1
        time.sleep(1)
    
    print("\n" + "=" * 50)
    print(f"Scraping completed! Successful: {successful}, Failed: {failed}")
    print(f"Total chunks collected: {len(all_chunks)}")
    
    # Build RAG database from collected chunks
    if all_chunks:
        print("\nBuilding RAG database from scraped content...")
        try:
            chunk_count = rag_db.build_indices_from_chunks(all_chunks)
            print(f"\n✓ RAG Database built successfully!")
            print(f"  - {chunk_count} chunks indexed")
            print(f"  - Whoosh index: {rag_db.db_path}/whoosh_index/")
            print(f"  - FAISS index: {rag_db.db_path}/faiss_index.bin")
            print("  - Images saved to: data/images/")
            
            # Test search functionality
            print("\n" + "=" * 50)
            print("Testing search functionality...")
            
            test_queries = [
                "What is Arabasta Kingdom?",
                "Tell me about the desert in Arabasta",
                "Who are the main characters in Arabasta?"
            ]
            
            for query in test_queries:
                print(f"\nTest query: '{query}'")
                try:
                    results = rag_db.search(query, top_k=3)
                    print(f"  Found {len(results)} results")
                    
                    for i, result in enumerate(results, 1):
                        print(f"    {i}. {result['search_metadata']['section_name']} "
                              f"(BM25: {result['bm25_score']:.3f}, "
                              f"Semantic: {result['semantic_score']:.3f}, "
                              f"Combined: {result['combined_score']:.3f})")
                        print(f"       {result['content'][:100]}...")
                
                except Exception as e:
                    print(f"  Search error: {e}")
                    
        except Exception as e:
            print(f"Error building RAG database: {e}")
            return False
    else:
        print("\nNo chunks collected - database not built")
        return False
    
    return True


if __name__ == "__main__":
    main()
