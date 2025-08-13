#!/usr/bin/env python3
"""
One Piece Wiki Article Structure Analyzer
Analyzes the structure of a One Piece Wiki article to understand its layout.
"""

import requests
import json
import re
from pathlib import Path


def fetch_article_structure(article_name):
    """Fetch article structure from One Piece Wiki using MediaWiki API."""
    base_url = "https://onepiece.fandom.com/api.php"
    
    # Get article sections
    sections_params = {
        'action': 'parse',
        'page': article_name,
        'format': 'json',
        'prop': 'sections'
    }
    
    # Get article content
    content_params = {
        'action': 'parse',
        'page': article_name,
        'format': 'json',
        'prop': 'text|images'
    }
    
    try:
        # Fetch sections
        print(f"Fetching sections for: {article_name}")
        sections_response = requests.get(base_url, params=sections_params, timeout=30)
        sections_response.raise_for_status()
        sections_data = sections_response.json()
        
        # Fetch content
        print(f"Fetching content for: {article_name}")
        content_response = requests.get(base_url, params=content_params, timeout=30)
        content_response.raise_for_status()
        content_data = content_response.json()
        
        if 'error' in sections_data:
            print(f"Error fetching sections: {sections_data['error']['info']}")
            return None
            
        if 'error' in content_data:
            print(f"Error fetching content: {content_data['error']['info']}")
            return None
            
        return {
            'sections': sections_data['parse']['sections'],
            'content': content_data['parse']['text']['*'],
            'images': content_data['parse']['images']
        }
        
    except Exception as e:
        print(f"Failed to fetch {article_name}: {e}")
        return None


def analyze_sections(sections):
    """Analyze the section structure."""
    print("\n" + "="*60)
    print("SECTION STRUCTURE ANALYSIS")
    print("="*60)
    
    for i, section in enumerate(sections):
        level = section.get('level', 'N/A')
        title = section.get('title', 'N/A')
        anchor = section.get('anchor', 'N/A')
        
        # Handle level properly - convert to int if possible
        try:
            level_int = int(level) if level != 'N/A' else 0
            indent = "  " * (level_int - 1) if level_int > 0 else ""
        except (ValueError, TypeError):
            indent = ""
            level_int = level
        
        print(f"{indent}Level {level}: {title}")
        if anchor and anchor != title:
            print(f"{indent}  Anchor: {anchor}")


def analyze_content_structure(content_html):
    """Analyze the HTML content structure."""
    print("\n" + "="*60)
    print("CONTENT STRUCTURE ANALYSIS")
    print("="*60)
    
    # Find all headers
    h1_pattern = r'<h1[^>]*>(.*?)</h1>'
    h2_pattern = r'<h2[^>]*>(.*?)</h2>'
    h3_pattern = r'<h3[^>]*>(.*?)</h3>'
    h4_pattern = r'<h4[^>]*>(.*?)</h4>'
    
    h1_headers = re.findall(h1_pattern, content_html, re.DOTALL)
    h2_headers = re.findall(h2_pattern, content_html, re.DOTALL)
    h3_headers = re.findall(h3_pattern, content_html, re.DOTALL)
    h4_headers = re.findall(h4_pattern, content_html, re.DOTALL)
    
    print(f"H1 Headers ({len(h1_headers)}):")
    for header in h1_headers:
        clean_header = re.sub(r'<[^>]+>', '', header).strip()
        print(f"  - {clean_header}")
    
    print(f"\nH2 Headers ({len(h2_headers)}):")
    for header in h2_headers:
        clean_header = re.sub(r'<[^>]+>', '', header).strip()
        print(f"  - {clean_header}")
    
    print(f"\nH3 Headers ({len(h3_headers)}):")
    for header in h3_headers:
        clean_header = re.sub(r'<[^>]+>', '', header).strip()
        print(f"  - {clean_header}")
    
    print(f"\nH4 Headers ({len(h4_headers)}):")
    for header in h4_headers:
        clean_header = re.sub(r'<[^>]+>', '', header).strip()
        print(f"  - {clean_header}")


def analyze_tables(content_html):
    """Analyze table structure in the content."""
    print("\n" + "="*60)
    print("TABLE STRUCTURE ANALYSIS")
    print("="*60)
    
    # Find tables with captions
    table_caption_pattern = r'<caption[^>]*>(.*?)</caption>'
    table_pattern = r'<table[^>]*>'
    
    captions = re.findall(table_caption_pattern, content_html, re.DOTALL)
    tables = re.findall(table_pattern, content_html)
    
    print(f"Total Tables Found: {len(tables)}")
    print(f"Tables with Captions: {len(captions)}")
    
    if captions:
        print("\nTable Captions:")
        for i, caption in enumerate(captions, 1):
            clean_caption = re.sub(r'<[^>]+>', '', caption).strip()
            print(f"  {i}. {clean_caption}")
    
    # Look for tables near headers
    print("\nTables near headers:")
    for i, table_match in enumerate(re.finditer(table_pattern, content_html)):
        table_start = table_match.start()
        
        # Look for headers before this table
        before_table = content_html[:table_start]
        header_pattern = r'<(?:h[1-6]|strong)[^>]*>(.*?)</(?:h[1-6]|strong)>'
        headers = re.findall(header_pattern, before_table, re.DOTALL)
        
        if headers:
            last_header = re.sub(r'<[^>]+>', '', headers[-1]).strip()
            print(f"  Table {i+1}: Near header '{last_header}'")


def analyze_images(content_html, api_images):
    """Analyze image structure in the content."""
    print("\n" + "="*60)
    print("IMAGE STRUCTURE ANALYSIS")
    print("="*60)
    
    # API images
    print(f"API Images ({len(api_images)}):")
    for i, img in enumerate(api_images[:10]):  # Show first 10
        print(f"  {i+1}. {img}")
    
    if len(api_images) > 10:
        print(f"  ... and {len(api_images) - 10} more")
    
    # HTML images
    img_pattern = r'<img[^>]*src="([^"]*)"[^>]*alt="([^"]*)"[^>]*width="([^"]*)"[^>]*height="([^"]*)"[^>]*>'
    html_images = re.findall(img_pattern, content_html)
    
    print(f"\nHTML Images with attributes ({len(html_images)}):")
    for i, (src, alt, width, height) in enumerate(html_images[:10]):  # Show first 10
        print(f"  {i+1}. Src: {src}")
        print(f"     Alt: {alt}")
        print(f"     Size: {width}x{height}")
        print()
    
    if len(html_images) > 10:
        print(f"  ... and {len(html_images) - 10} more")


def save_sample_content(content_html, article_name):
    """Save a sample of the HTML content for manual inspection."""
    sample_file = Path(f"{article_name.replace(' ', '_')}_sample.html")
    
    # Get first 5000 characters as sample
    sample_content = content_html[:5000]
    
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(f"<!-- Sample HTML content from {article_name} -->\n")
        f.write(f"<!-- Total length: {len(content_html)} characters -->\n")
        f.write("<!-- First 5000 characters: -->\n\n")
        f.write(sample_content)
        f.write("\n\n<!-- ... end of sample ... -->")
    
    print(f"\nSample HTML content saved to: {sample_file}")


def main():
    """Main function to analyze article structure."""
    article_name = "Monkey D. Luffy"
    
    print("One Piece Wiki Article Structure Analyzer")
    print("="*60)
    print(f"Analyzing article: {article_name}")
    
    # Fetch article data
    article_data = fetch_article_structure(article_name)
    if not article_data:
        print("Failed to fetch article data")
        return
    
    # Analyze different aspects
    analyze_sections(article_data['sections'])
    analyze_content_structure(article_data['content'])
    analyze_tables(article_data['content'])
    analyze_images(article_data['content'], article_data['images'])
    
    # Save sample content for manual inspection
    save_sample_content(article_data['content'], article_name)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Use this information to decide on the best scraping strategy:")
    print("- Which header levels to use for sectioning")
    print("- How to identify and name tables")
    print("- How to filter and name images")
    print("- What content to exclude")


if __name__ == "__main__":
    main()
