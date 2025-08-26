#!/usr/bin/env python3
"""
Comprehensive test script for the entire RAG Piece system.

This script tests the complete pipeline:
1. Scraping "Straw Hat Pirates" article (text, images, tables)
2. Generating summaries
3. Converting CSV tables to text
4. Saving all outputs (summaries, CSV files, CSV-to-text)
5. Building the RAG database
6. Testing search functionality with various queries

Usage:
    python test/test_complete_rag_system.py
"""

import os
import sys
import logging
import shutil
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_piece.config import RAGConfig
from rag_piece.database import RAGDatabase
from rag_piece.scraper import OneWikiScraper
from rag_piece.csv_scraper import CSVWikiScraper
from rag_piece.utils import setup_logging

# Import summarizer for optional integration
try:
    from rag_piece.summarizer import ArticleSummarizer
    SUMMARIZER_AVAILABLE = True
except ImportError:
    SUMMARIZER_AVAILABLE = False

# Import CSV converter
try:
    from rag_piece.csv_to_text import CSVToTextConverter
    CSV_CONVERTER_AVAILABLE = True
except ImportError:
    CSV_CONVERTER_AVAILABLE = False


def setup_test_environment():
    """Set up the test environment and configuration."""
    print("🚀 Setting up test environment for Straw Hat Pirates RAG system...")
    
    # Create test configuration
    config = RAGConfig()
    
    # Configure for testing
    config.ARTICLES_TO_SCRAPE = ["Straw Hat Pirates"]
    config.ENABLE_SUMMARIZATION = True
    config.ENABLE_CSV_SCRAPING = True
    config.ENABLE_CSV_TO_TEXT = True
    config.SAVE_SUMMARIES_TO_FILES = True
    config.SAVE_CSV_FILES_FOR_DEBUG = True
    config.SAVE_CSV_TO_TEXT_FILES = True
    config.MAX_INPUT_TEXT_TOKENS = 8000  # Limit input text for faster summarization
    config.LOG_LEVEL = "INFO"
    
    # Setup logging
    logger = setup_logging(config.LOG_LEVEL)
    logger.info("Test environment configured for Straw Hat Pirates article")
    
    return config, logger


def clear_test_data(config, logger):
    """Clear any existing test data."""
    print("🧹 Clearing previous test data...")
    
    # Clear data directories
    data_dirs = [
        "data/images/Straw_Hat_Pirates",
        "data/debug/csv_files/straw_hat_pirates",
        "data/debug/csv2text/straw_hat_pirates",
        "summaries/straw_hat_pirates",
        "data/rag_db"
    ]
    
    for dir_path in data_dirs:
        if Path(dir_path).exists():
            shutil.rmtree(dir_path)
            logger.info(f"Cleared: {dir_path}")
    
    print("✅ Test data cleared")


def test_scraping(config, logger):
    """Test the scraping functionality."""
    print("\n📡 Testing scraping functionality...")
    
    try:
        # Initialize scrapers
        text_scraper = OneWikiScraper(max_images=15)
        csv_scraper = CSVWikiScraper(
            request_delay=config.CSV_REQUEST_DELAY,
            save_to_files=config.SAVE_CSV_FILES_FOR_DEBUG
        )
        
        article_name = "Straw Hat Pirates"
        logger.info(f"Scraping article: {article_name}")
        
        # Scrape text and images
        print("  - Scraping text and images...")
        sections, text_metadata = text_scraper.scrape_article(article_name)
        
        if not sections:
            raise Exception("No sections scraped")
        
        print(f"    ✓ Scraped {len(sections)} text sections")
        print(f"    ✓ Downloaded {text_metadata.get('images_downloaded', 0)} images")
        
        # Scrape CSV tables
        print("  - Scraping CSV tables...")
        if config.ENABLE_CSV_SCRAPING:
            if config.SAVE_CSV_FILES_FOR_DEBUG:
                csv_files, csv_metadata = csv_scraper.scrape_article_to_csv(article_name)
                print(f"    ✓ Created {len(csv_files)} CSV files")
                print(f"    ✓ Found {csv_metadata.get('total_tables_found', 0)} tables")
            else:
                dataframes, csv_metadata = csv_scraper.extract_tables_in_memory(article_name)
                print(f"    ✓ Extracted {len(dataframes)} tables to DataFrames")
                print(f"    ✓ Found {csv_metadata.get('total_tables_found', 0)} tables")
                csv_files = []
        else:
            csv_files, csv_metadata = [], {}
        
        return sections, text_metadata, csv_files, csv_metadata
        
    except Exception as e:
        logger.error(f"Scraping failed: {e}", exc_info=True)
        raise


def test_summarization(config, logger, sections, article_name):
    """Test the summarization functionality."""
    print("\n📝 Testing summarization functionality...")
    
    if not SUMMARIZER_AVAILABLE:
        print("  ⚠️  Summarizer not available, skipping...")
        return [], {}
    
    try:
        # Initialize summarizer
        summarizer = ArticleSummarizer(
            max_chunk_size=config.MAX_CHUNK_SIZE,
            save_to_files=config.SAVE_SUMMARIES_TO_FILES,
            max_input_tokens=config.MAX_INPUT_TEXT_TOKENS
        )
        
        print("  - Generating article summaries...")
        summary_chunks = summarizer.create_summary_chunks(article_name)
        
        if summary_chunks:
            print(f"    ✓ Created {len(summary_chunks)} summary chunks")
            
            # Show summary types
            summary_types = set()
            for chunk in summary_chunks:
                summary_type = chunk.get('summary_metadata', {}).get('summary_type', 'unknown')
                summary_types.add(summary_type)
            
            print(f"    ✓ Summary types: {', '.join(summary_types)}")
            
            # Get main article summary for CSV conversion
            main_summary = ""
            for chunk in summary_chunks:
                if chunk.get('summary_metadata', {}).get('summary_type') == 'main_article':
                    main_summary = chunk.get('content', '')
                    break
            
            return summary_chunks, {'main_summary': main_summary}
        else:
            print("    ⚠️  No summaries created")
            return [], {}
            
    except Exception as e:
        logger.error(f"Summarization failed: {e}", exc_info=True)
        print(f"    ❌ Summarization failed: {e}")
        return [], {}


def test_csv_conversion(config, logger, csv_files, article_name, main_summary):
    """Test the CSV to text conversion functionality."""
    print("\n🔄 Testing CSV to text conversion...")
    
    if not CSV_CONVERTER_AVAILABLE:
        print("  ⚠️  CSV converter not available, skipping...")
        return []
    
    if not csv_files:
        print("  ⚠️  No CSV files to convert, skipping...")
        return []
    
    try:
        # Initialize converter
        csv_converter = CSVToTextConverter(config)
        
        print(f"  - Converting {len(csv_files)} CSV files to text...")
        csv_text_chunks = []
        
        for csv_file in csv_files:
            try:
                print(f"    - Converting: {Path(csv_file).name}")
                result = csv_converter.convert_csv_to_text(csv_file, article_name, main_summary)
                
                if result.get('success') and result.get('converted_text'):
                    # Create chunks from the converted CSV text
                    csv_text_chunks.append({
                        'combined_title': f"CSV Data: {Path(csv_file).stem}",
                        'content': result['converted_text'],
                        'article_source': article_name,
                        'section_index': 0
                    })
                    print(f"      ✓ Successfully converted")
                else:
                    print(f"      ❌ Conversion failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"Error converting CSV file {csv_file}: {e}")
                print(f"      ❌ Error: {e}")
        
        print(f"    ✓ Created {len(csv_text_chunks)} CSV text chunks")
        return csv_text_chunks
        
    except Exception as e:
        logger.error(f"CSV conversion failed: {e}", exc_info=True)
        print(f"    ❌ CSV conversion failed: {e}")
        return []


def test_database_building(config, logger, sections, summary_chunks, csv_text_chunks, article_name):
    """Test the database building functionality."""
    print("\n🏗️  Testing database building...")
    
    try:
        # Initialize RAG database
        rag_db = RAGDatabase(config)
        
        # Process all content into chunks
        print("  - Processing content into chunks...")
        
        # Process main content sections
        content_chunks = rag_db.process_sections_directly(sections, article_name)
        print(f"    ✓ Created {len(content_chunks)} content chunks")
        
        # Process CSV text chunks
        csv_chunks = []
        if csv_text_chunks:
            csv_chunks = rag_db.process_sections_directly(csv_text_chunks, article_name)
            print(f"    ✓ Created {len(csv_chunks)} CSV chunks")
        
        # Combine all chunks
        all_chunks = content_chunks + summary_chunks + csv_chunks
        print(f"    ✓ Total chunks: {len(all_chunks)}")
        
        if not all_chunks:
            raise Exception("No chunks created")
        
        # Build indices
        print("  - Building search indices...")
        chunk_count = rag_db.build_indices_from_chunks(all_chunks)
        
        if chunk_count > 0:
            print(f"    ✓ Successfully indexed {chunk_count} chunks")
            print(f"    ✓ Whoosh index: {rag_db.db_path}/whoosh_index/")
            print(f"    ✓ FAISS index: {rag_db.db_path}/faiss_index.bin")
        else:
            raise Exception("Failed to build indices")
        
        return rag_db, all_chunks
        
    except Exception as e:
        logger.error(f"Database building failed: {e}", exc_info=True)
        raise


def test_search_functionality(rag_db, logger):
    """Test the search functionality with various queries."""
    print("\n🔍 Testing search functionality...")
    
    try:
        # Load indices for testing
        if not rag_db.load_indices():
            raise Exception("Could not load indices for testing")
        
        # Test queries
        test_queries = [
            "What are the Straw Hat Pirates?",
            "Who is the captain of the crew?",
            "Tell me about Luffy's devil fruit",
            "What is the Thousand Sunny?",
            "Who are the main crew members?",
            "What happened at Enies Lobby?",
            "Tell me about the crew's bounties",
            "What is the crew's ship called?",
            "Who is the navigator?",
            "What is the crew's goal?"
        ]
        
        print(f"  - Testing {len(test_queries)} queries...")
        
        successful_searches = 0
        for i, query in enumerate(test_queries, 1):
            try:
                print(f"    {i:2d}. Query: '{query}'")
                results = rag_db.search(query, top_k=3)
                
                if results:
                    print(f"         ✓ Found {len(results)} results")
                    successful_searches += 1
                    
                    # Show top result
                    top_result = results[0]
                    section = top_result['search_metadata']['section_name']
                    content_preview = top_result['content'][:100].replace('\n', ' ')
                    print(f"         Top: [{section}] {content_preview}...")
                else:
                    print(f"         ❌ No results found")
                    
            except Exception as e:
                logger.error(f"Search query '{query}' failed: {e}")
                print(f"         ❌ Search failed: {e}")
        
        print(f"    ✓ Successful searches: {successful_searches}/{len(test_queries)}")
        return successful_searches == len(test_queries)
        
    except Exception as e:
        logger.error(f"Search testing failed: {e}", exc_info=True)
        return False


def verify_outputs(config, logger):
    """Verify that all expected outputs were created."""
    print("\n📁 Verifying output files...")
    
    article_slug = "straw_hat_pirates"
    outputs_verified = 0
    total_outputs = 0
    
    # Check summaries
    if config.SAVE_SUMMARIES_TO_FILES:
        summary_dir = Path(f"summaries/{article_slug}")
        if summary_dir.exists():
            summary_files = list(summary_dir.glob("*.txt"))
            print(f"  - Summaries: {len(summary_files)} files in {summary_dir}")
            outputs_verified += len(summary_files)
        total_outputs += 1
    
    # Check CSV files
    if config.SAVE_CSV_FILES_FOR_DEBUG:
        csv_dir = Path(f"data/debug/csv_files/{article_slug}")
        if csv_dir.exists():
            csv_files = list(csv_dir.glob("*.csv"))
            print(f"  - CSV files: {len(csv_files)} files in {csv_dir}")
            outputs_verified += len(csv_files)
        total_outputs += 1
    
    # Check CSV-to-text files
    if config.SAVE_CSV_TO_TEXT_FILES:
        csv2text_dir = Path(f"data/debug/csv2text/{article_slug}")
        if csv2text_dir.exists():
            csv2text_files = list(csv2text_dir.glob("*.txt"))
            print(f"  - CSV-to-text: {len(csv2text_files)} files in {csv2text_dir}")
            outputs_verified += len(csv2text_files)
        total_outputs += 1
    
    # Check images
    images_dir = Path(f"data/images/Straw_Hat_Pirates")
    if images_dir.exists():
        image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
        print(f"  - Images: {len(image_files)} files in {images_dir}")
        outputs_verified += len(image_files)
    total_outputs += 1
    
    # Check database
    db_dir = Path("data/rag_db")
    if db_dir.exists():
        db_files = list(db_dir.rglob("*"))
        db_files = [f for f in db_files if f.is_file()]
        print(f"  - Database: {len(db_files)} files in {db_dir}")
        outputs_verified += len(db_files)
    total_outputs += 1
    
    print(f"  ✓ Total outputs verified: {outputs_verified}")
    return outputs_verified > 0


def main():
    """Main test function."""
    print("=" * 70)
    print("🧪 COMPREHENSIVE RAG SYSTEM TEST - Straw Hat Pirates")
    print("=" * 70)
    
    start_time = datetime.now()
    
    try:
        # Setup
        config, logger = setup_test_environment()
        clear_test_data(config, logger)
        
        # Test 1: Scraping
        sections, text_metadata, csv_files, csv_metadata = test_scraping(config, logger)
        
        # Test 2: Summarization
        summary_chunks, summary_metadata = test_summarization(config, logger, sections, "Straw Hat Pirates")
        main_summary = summary_metadata.get('main_summary', '')
        
        # Test 3: CSV Conversion
        csv_text_chunks = test_csv_conversion(config, logger, csv_files, "Straw Hat Pirates", main_summary)
        
        # Test 4: Database Building
        rag_db, all_chunks = test_database_building(config, logger, sections, summary_chunks, csv_text_chunks, "Straw Hat Pirates")
        
        # Test 5: Search Functionality
        search_success = test_search_functionality(rag_db, logger)
        
        # Test 6: Verify Outputs
        outputs_verified = verify_outputs(config, logger)
        
        # Final Results
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 70)
        print("🎯 TEST RESULTS SUMMARY")
        print("=" * 70)
        
        print(f"✅ Scraping: {len(sections)} sections, {text_metadata.get('images_downloaded', 0)} images")
        print(f"✅ CSV Tables: {len(csv_files)} files created")
        print(f"✅ Summarization: {len(summary_chunks)} summary chunks")
        print(f"✅ CSV Conversion: {len(csv_text_chunks)} text chunks")
        print(f"✅ Database: {len(all_chunks)} total chunks indexed")
        print(f"✅ Search: {'PASSED' if search_success else 'FAILED'}")
        print(f"✅ Outputs: {'VERIFIED' if outputs_verified else 'MISSING'}")
        print(f"⏱️  Total Duration: {duration:.1f} seconds")
        
        if search_success and outputs_verified:
            print("\n🎉 ALL TESTS PASSED! The RAG system is working correctly.")
            return True
        else:
            print("\n⚠️  Some tests failed. Check the logs above for details.")
            return False
            
    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n❌ TEST FAILED after {duration:.1f} seconds")
        print(f"Error: {e}")
        logger.error("Test failed", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
