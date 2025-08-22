#!/usr/bin/env python3
"""
Standalone test script for the CSV scraper.
This script tests the CSV scraper independently of the main RAG system.
"""

import sys
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_piece.csv_scraper import CSVWikiScraper
from rag_piece.utils import setup_logging


def main():
    """Test the CSV scraper with Arabasta Kingdom article."""
    print("CSV Scraper Test - One Piece Wiki")
    print("=" * 50)
    
    # Setup logging
    logger = setup_logging("INFO")
    logger.info("Starting CSV scraper test")
    
    try:
        # Initialize the CSV scraper
        csv_scraper = CSVWikiScraper(request_delay=1.0)
        
        # Test with Conomi Islands article
        article_name = "Conomi Islands"
        print(f"\nScraping article: {article_name}")
        print("This will extract all tables and convert them to CSV files...")
        
        # Run the scraper
        csv_files_created, metadata = csv_scraper.scrape_article_to_csv(article_name)
        
        # Display results
        print("\n" + "=" * 50)
        print("SCRAPING RESULTS")
        print("=" * 50)
        
        if csv_files_created:
            print(f"✓ Successfully created {len(csv_files_created)} CSV files!")
            print(f"  CSV files saved to: csv_files/{metadata.get('csv_folder', 'unknown')}")
            
            print(f"\nTable Summary:")
            for table_info in metadata.get('table_summary', []):
                print(f"  - {table_info['table_title']}: {table_info['rows']} rows × {table_info['columns']} columns")
            
            print(f"\nCSV Files Created:")
            for csv_file in csv_files_created:
                print(f"  ✓ {Path(csv_file).name}")
                
        else:
            print("❌ No CSV files were created.")
            print("This might mean:")
            print("  - No tables were found in the article")
            print("  - Tables were filtered out as invalid")
            print("  - There was an error during scraping")
        
        # Display metadata
        print(f"\nScraping Metadata:")
        print(f"  - Article: {metadata.get('article_name', 'Unknown')}")
        print(f"  - Sub-articles found: {len(metadata.get('sub_articles', []))}")
        print(f"  - Total tables found: {metadata.get('total_tables_found', 0)}")
        print(f"  - Timestamp: {metadata.get('scraping_timestamp', 'Unknown')}")
        
        if metadata.get('sub_articles'):
            print(f"  - Sub-articles: {', '.join(metadata.get('sub_articles', []))}")
        
    except Exception as e:
        logger.error(f"Error during CSV scraping test: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1
    
    print("\n" + "=" * 50)
    print("Test completed!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
