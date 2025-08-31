"""
Main application entry point for the RAG Piece system.
"""

import logging
import shutil
from pathlib import Path


from .config import RAGConfig
from .database import RAGDatabase
from .scraper import OneWikiScraper
from .csv_scraper import CSVWikiScraper
from .utils import setup_logging

# Import summarizer for optional integration
try:
    from .summarizer import ArticleSummarizer
    SUMMARIZER_AVAILABLE = True
except ImportError:
    SUMMARIZER_AVAILABLE = False


def main():
    """Main function to run the integrated scraper + RAG system."""
    # Initialize configuration
    config = RAGConfig()
    
    # Setup logging
    logger = setup_logging(config.LOG_LEVEL)
    logger.info("Starting One Piece Wiki RAG Database Builder")
    
    try:
        # Display configuration
        _display_configuration(config, logger)
        
        # Clear previous data
        _clear_previous_data(logger)
        
        # Initialize components
        text_scraper = OneWikiScraper(max_images=20)
        csv_scraper = CSVWikiScraper(
            request_delay=config.CSV_REQUEST_DELAY,
            save_to_files=config.SAVE_CSV_FILES_FOR_DEBUG
        )
        rag_db = RAGDatabase(config)
        
        # Scrape articles and build database
        success_count = _process_articles(text_scraper, csv_scraper, rag_db, logger)
        
        # Test search functionality
        if success_count > 0:
            _test_search_functionality(rag_db, logger)
        
        logger.info(f"Process completed successfully! Processed {success_count} articles.")
    
    except Exception as e:
        logger.error(f"Fatal error in main process: {e}", exc_info=True)
        raise


def _display_configuration(config: RAGConfig, logger: logging.Logger) -> None:
    """Display system configuration."""
    print("\nOne Piece Wiki Scraper + RAG Database Builder")
    print("=" * 50)
    print("Configuration:")
    print("  - Maximum images per article: 20")
    print("  - Direct processing: text → chunks → database (no intermediate files)")
    print(f"  - Chunking: {config.MIN_CHUNK_SIZE}-{config.MAX_CHUNK_SIZE} tokens (target: {config.TARGET_CHUNK_SIZE}, overlap: {config.CHUNK_OVERLAP})")
    print("  - CSV extraction: Tables → CSV files")
    print("  - Parallel processing: Content chunking + Summarization + CSV conversion")
    
    # Show articles to be scraped
    print(f"  - Articles to scrape: {', '.join(config.ARTICLES_TO_SCRAPE)}")
    
    # Show CSV processing status
    if config.ENABLE_CSV_SCRAPING:
        print("  - CSV processing: ENABLED")
        if config.SAVE_CSV_FILES_FOR_DEBUG:
            print("    - CSV files: Will be saved to data/debug/csv_files/ folder (debug mode)")
        else:
            print("    - CSV files: Will NOT be saved (in-memory processing)")
        print(f"    - Request delay: {config.CSV_REQUEST_DELAY}s")
    else:
        print("  - CSV processing: DISABLED")
    
    # Show summarization status
    if config.ENABLE_SUMMARIZATION:
        if SUMMARIZER_AVAILABLE:
            print(f"  - Article summarization: ENABLED (using OpenAI)")
            print(f"    - Summary input chunks: {config.SUMMARY_INPUT_CHUNK_SIZE} tokens")
            print(f"    - Summary chunk overlap: {config.SUMMARY_CHUNK_OVERLAP} tokens")
        else:
            print("  - Article summarization: ENABLED but summarizer not available")
            print("    (article_summarizer.py not found in root directory)")
    else:
        print("  - Article summarization: DISABLED")
    
    # Show CSV to text conversion status
    if config.ENABLE_CSV_TO_TEXT:
        print("  - CSV to text conversion: ENABLED (using OpenAI)")
    else:
        print("  - CSV to text conversion: DISABLED")
    
    print()
    
    logger.info("RAG Configuration:")
    logger.info(f"  - Chunking: {config.MIN_CHUNK_SIZE}-{config.MAX_CHUNK_SIZE} tokens (target: {config.TARGET_CHUNK_SIZE}, overlap: {config.CHUNK_OVERLAP})")
    logger.info(f"  - Keywords: {config.KEYWORDS_PER_CHUNK} per chunk using BM25 scoring")
    logger.info(f"  - Embedding model: {config.EMBEDDING_MODEL}")
    
    if config.ENABLE_SUMMARIZATION and SUMMARIZER_AVAILABLE:
        logger.info(f"  - Summarization: {config.SUMMARY_MODEL} (temperature: {config.SUMMARY_TEMPERATURE})")
        if config.SAVE_SUMMARIES_TO_FILES:
            logger.info("  - Summary files: Will be saved to data/debug/summaries/ folder")
        else:
            logger.info("  - Summary files: Will NOT be saved to data/debug/summaries/ folder")
    
    if config.ENABLE_CSV_TO_TEXT:
        logger.info(f"  - CSV conversion: {config.CSV_TO_TEXT_MODEL} (temperature: {config.CSV_TO_TEXT_TEMPERATURE})")
        if config.SAVE_CSV_TO_TEXT_FILES:
            logger.info("  - CSV conversion files: Will be saved to data/debug/csv2text/ folder")
        else:
            logger.info("  - CSV conversion files: Will NOT be saved to debug folder")


def _clear_previous_data(logger: logging.Logger) -> None:
    """Clear previous data folder."""
    data_folder = Path("data")
    if data_folder.exists():
        logger.info("Clearing previous data folder...")
        try:
            shutil.rmtree(data_folder)
            print("  Previous data folder cleared")
        except Exception as e:
            logger.warning(f"Could not clear data folder: {e}")
    else:
        print("No previous data folder found, starting fresh")
    
    print("  Images are stored in data/images/ folder (not cleared automatically)")
    print("  CSV files are stored in data/debug/csv_files/ folder (not cleared automatically)")
    print()


def _process_articles(text_scraper: OneWikiScraper, csv_scraper: CSVWikiScraper, rag_db: RAGDatabase, logger: logging.Logger) -> int:
    """Process all articles and build the database."""
    # Get articles to scrape from config
    articles = rag_db.config.ARTICLES_TO_SCRAPE
    
    all_chunks = []
    all_metadata = []
    success_count = 0
    
    # Initialize summarizer if enabled
    summarizer = None
    if rag_db.config.ENABLE_SUMMARIZATION and SUMMARIZER_AVAILABLE:
        try:
            summarizer = ArticleSummarizer(
                max_chunk_size=rag_db.config.MAX_CHUNK_SIZE,
                save_to_files=rag_db.config.SAVE_SUMMARIES_TO_FILES,
                max_input_tokens=rag_db.config.MAX_INPUT_TEXT_TOKENS
            )
            logger.info("Article summarizer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize summarizer: {e}")
            summarizer = None
    elif rag_db.config.ENABLE_SUMMARIZATION and not SUMMARIZER_AVAILABLE:
        logger.warning("Summarization enabled but summarizer not available")
    
    # Initialize CSV to text converter if enabled
    csv_converter = None
    if rag_db.config.ENABLE_CSV_TO_TEXT:
        try:
            from .csv_to_text import CSVToTextConverter
            csv_converter = CSVToTextConverter(rag_db.config)
            logger.info("CSV to text converter initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize CSV converter: {e}")
            logger.warning("Continuing without CSV to text conversion")
            csv_converter = None
    
    logger.info(f"Processing {len(articles)} articles")
    
    for article_name in articles:
        try:
            logger.info(f"Processing article: {article_name}")
            
            # Initialize CSV variables with safe defaults first
            dataframes = []
            csv_files = []
            csv_metadata = {}
            
            # Run both scrapers in parallel
            logger.info("Running text and image scraper...")
            sections, text_metadata = text_scraper.scrape_article(article_name)
            
            logger.info("Running CSV scraper...")
            try:
                if rag_db.config.ENABLE_CSV_SCRAPING:
                    if rag_db.config.SAVE_CSV_FILES_FOR_DEBUG:
                        # Use traditional CSV file creation for debugging
                        csv_files, csv_metadata = csv_scraper.scrape_article_to_csv(article_name)
                    else:
                        # Use new in-memory approach (preferred)
                        dataframes, csv_metadata = csv_scraper.extract_tables_in_memory(article_name)
                        csv_files = []  # No CSV files created
                else:
                    csv_files, csv_metadata = [], {}
                    dataframes = []
            except Exception as e:
                logger.error(f"CSV processing failed for {article_name}: {e}")
                # Keep safe defaults if CSV processing fails
                csv_files, csv_metadata, dataframes = [], {}, []
                
            if sections:
                # PARALLEL PROCESSING: Content chunking and summarization happen simultaneously
                logger.info("Processing content and generating summaries in parallel...")
                
                # Process sections into chunks
                chunks = rag_db.process_sections_directly(sections, article_name)
                
                # Generate summaries immediately after scraping (parallel to chunking)
                summary_chunks = []
                if summarizer:
                    try:
                        logger.info("Generating article summaries...")
                        summary_chunks = summarizer.create_summary_chunks(article_name)
                        logger.info(f"Generated {len(summary_chunks)} summary chunks")
                    except Exception as e:
                        logger.error(f"Error generating summaries: {e}")
                
                # Process CSV files to text chunks using the freshly generated summaries
                csv_chunks = []
                if csv_converter and (csv_files or dataframes):
                    try:
                        logger.info("Converting CSV data to text chunks...")
                        # Get the main article summary for CSV conversion context
                        main_summary = ""
                        if summary_chunks:
                            # Find the main article summary
                            for summary_chunk in summary_chunks:
                                if summary_chunk.get('summary_metadata', {}).get('summary_type') == 'main_article':
                                    main_summary = summary_chunk.get('content', '')
                                    break
                        
                        # Process CSV data (either files or in-memory DataFrames)
                        if dataframes:
                            # Process in-memory DataFrames directly
                            for i, df in enumerate(dataframes):
                                try:
                                    # Convert DataFrame to text using the converter
                                    table_name = csv_metadata.get('table_details', [{}])[i].get('table_name', f'Table_{i+1}')
                                    result = csv_converter.convert_dataframe_to_text(df, article_name, main_summary, table_name)
                                    if result.get('success') and result.get('converted_text'):
                                        # Create chunks from the converted CSV text
                                        csv_text_chunks = rag_db.process_sections_directly([
                                            {
                                                'combined_title': f"CSV Data: {table_name}",
                                                'content': result['converted_text'],
                                                'article_source': article_name,
                                                'section_index': 0
                                            }
                                        ], article_name)
                                        csv_chunks.extend(csv_text_chunks)
                                        logger.info(f"Created {len(csv_text_chunks)} chunks from DataFrame: {table_name}")
                                except Exception as e:
                                    logger.error(f"Error processing DataFrame {i}: {e}")
                        elif csv_files:
                            # Process CSV files (for debugging mode)
                            for csv_file in csv_files:
                                try:
                                    result = csv_converter.convert_csv_to_text(csv_file, article_name, main_summary)
                                    if result.get('success') and result.get('converted_text'):
                                        # Create chunks from the converted CSV text
                                        csv_text_chunks = rag_db.process_sections_directly([
                                            {
                                                'combined_title': f"CSV Data: {Path(csv_file).stem}",
                                                'content': result['converted_text'],
                                                'article_source': article_name,
                                                'section_index': 0
                                            }
                                        ], article_name)
                                        csv_chunks.extend(csv_text_chunks)
                                        logger.info(f"Created {len(csv_text_chunks)} chunks from CSV: {Path(csv_file).name}")
                                except Exception as e:
                                    logger.error(f"Error processing CSV file {csv_file}: {e}")
                        
                        logger.info(f"Created {len(csv_chunks)} total CSV text chunks")
                    except Exception as e:
                        logger.error(f"Error in CSV to text conversion: {e}")
                
                # Combine all chunks (content + summaries + CSV text)
                all_chunks.extend(chunks)
                all_chunks.extend(summary_chunks)
                all_chunks.extend(csv_chunks)
                
                # Combine metadata from both scrapers
                combined_metadata = {
                    **text_metadata,
                    'csv_files_created': csv_files,
                    'csv_metadata': csv_metadata,
                    'summary_chunks_created': len(summary_chunks),
                    'content_chunks_created': len(chunks),
                    'csv_text_chunks_created': len(csv_chunks)
                }
                all_metadata.append(combined_metadata)
                success_count += 1
                
                logger.info(f"Successfully processed: {article_name}")
                logger.info(f"  - Text sections: {len(sections)}")
                logger.info(f"  - CSV files: {len(csv_files)}")
                logger.info(f"  - Content chunks: {len(chunks)}")
                logger.info(f"  - Summary chunks: {len(summary_chunks)}")
                logger.info(f"  - CSV text chunks: {len(csv_chunks)}")
            else:
                logger.warning(f"No content found for article: {article_name}")
        
        except Exception as e:
            logger.error(f"Failed to process article {article_name}: {e}", exc_info=True)
    
    # Build database if we have chunks
    if all_chunks:
        logger.info(f"Building RAG database from {len(all_chunks)} total chunks...")
        chunk_count = rag_db.build_indices_from_chunks(all_chunks)
        
        if chunk_count > 0:
            print("\n✓ RAG Database built successfully!")
            print(f"  - {chunk_count} chunks indexed")
            print(f"  - Whoosh index: {rag_db.db_path}/whoosh_index/")
            print(f"  - FAISS index: {rag_db.db_path}/faiss_index.bin")
            print("  - Images saved to: data/images/")
            print("  - CSV files saved to: data/debug/csv_files/")
            
            # Show chunk breakdown
            content_chunk_count = sum(meta.get('content_chunks_created', 0) for meta in all_metadata)
            summary_chunk_count = sum(meta.get('summary_chunks_created', 0) for meta in all_metadata)
            csv_chunk_count = sum(meta.get('csv_text_chunks_created', 0) for meta in all_metadata)
            print(f"  - Content chunks: {content_chunk_count}")
            print(f"  - Summary chunks: {summary_chunk_count}")
            print(f"  - CSV text chunks: {csv_chunk_count}")
        else:
            logger.error("Failed to build database indices")
    else:
        logger.error("No chunks created, cannot build database")
    
    return success_count


def _test_search_functionality(rag_db: RAGDatabase, logger: logging.Logger) -> None:
    """Test the search functionality with sample queries."""
    print("\n" + "=" * 50)
    print("Testing search functionality...")
    print()
    
    # Load indices for testing
    if not rag_db.load_indices():
        logger.error("Could not load indices for testing")
        return
    
    # Test queries
    test_queries = [
        "What is Arabasta Kingdom?",
        "Tell me about the desert in Arabasta",
        "Who are the main characters in Arabasta?"
    ]
    
    for query in test_queries:
        try:
            print(f"Test query: '{query}'")
            results = rag_db.search(query, top_k=3)
            
            if results:
                print(f"  Found {len(results)} results")
                for i, result in enumerate(results, 1):
                    section = result['search_metadata']['section_name']
                    bm25_score = result.get('bm25_score', 0)
                    semantic_score = result.get('semantic_score', 0)
                    combined_score = result.get('combined_score', 0)
                    content_preview = result['content'][:100] + "..."
                    
                    # Show chunk type if it's a summary
                    chunk_type = ""
                    if 'summary_metadata' in result:
                        chunk_type = f" [SUMMARY: {result['summary_metadata']['summary_type']}]"
                    
                    print(f"    {i}. {section}{chunk_type} (BM25: {bm25_score:.3f}, Semantic: {semantic_score:.3f}, Combined: {combined_score:.3f})")
                    print(f"       {content_preview}")
            else:
                print("  No results found")
            
            print()
        
        except Exception as e:
            logger.error(f"Error testing query '{query}': {e}", exc_info=True)


if __name__ == "__main__":
    main()
