"""
Main application entry point for the RAG Piece system.
"""

import logging
import shutil
from pathlib import Path


from .config import RAGConfig
from .database import RAGDatabase
from .scraper import OneWikiScraper
from .utils import setup_logging


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
        scraper = OneWikiScraper(max_images=20)
        rag_db = RAGDatabase(config)
        
        # Scrape articles and build database
        success_count = _process_articles(scraper, rag_db, logger)
        
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
    print()
    
    logger.info("RAG Configuration:")
    logger.info(f"  - Chunking: {config.MIN_CHUNK_SIZE}-{config.MAX_CHUNK_SIZE} tokens (target: {config.TARGET_CHUNK_SIZE})")
    logger.info(f"  - Keywords: {config.KEYWORDS_PER_CHUNK} per chunk using BM25 scoring")
    logger.info(f"  - Embedding model: {config.EMBEDDING_MODEL}")


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
    
    print("  Images are stored in separate images/ folder (not cleared automatically)")
    print()


def _process_articles(scraper: OneWikiScraper, rag_db: RAGDatabase, logger: logging.Logger) -> int:
    """Process all articles and build the database."""
    # Define articles to scrape
    articles = ["Arabasta Kingdom"]
    
    all_chunks = []
    all_metadata = []
    success_count = 0
    
    logger.info(f"Processing {len(articles)} articles")
    
    for article_name in articles:
        try:
            # Scrape article
            sections, metadata = scraper.scrape_article(article_name)
            
            if sections:
                # Process sections into chunks
                chunks = rag_db.process_sections_directly(sections, article_name)
                all_chunks.extend(chunks)
                all_metadata.append(metadata)
                success_count += 1
                
                logger.info(f"Successfully processed: {article_name} ({len(chunks)} chunks)")
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
            print("  - Images saved to: images/")
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
                    
                    print(f"    {i}. {section} (BM25: {bm25_score:.3f}, Semantic: {semantic_score:.3f}, Combined: {combined_score:.3f})")
                    print(f"       {content_preview}")
            else:
                print("  No results found")
            
            print()
        
        except Exception as e:
            logger.error(f"Error testing query '{query}': {e}", exc_info=True)


if __name__ == "__main__":
    main()
