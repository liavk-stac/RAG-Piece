#!/usr/bin/env python3
"""
Standalone article summarizer tool for One Piece Wiki content.
Uses the integrated package summarizer module.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_piece.summarizer import ArticleSummarizer
from rag_piece.utils import setup_logging, slugify

def main():
    """Test the article summarizer."""
    print("Article Summarizer Test - One Piece Wiki")
    print("=" * 50)
    
    # Setup logging
    logger = setup_logging("INFO")
    logger.info("Starting article summarizer test")
    
    try:
        # Initialize the summarizer
        summarizer = ArticleSummarizer(max_chunk_size=400, save_to_files=True, max_input_tokens=8000)
        
        # Test with Arabasta Kingdom article
        article_name = "Arabasta Kingdom"
        print(f"\nSummarizing article: {article_name}")
        print("This will create summaries for the main article and all sub-articles...")
        
        # Run the summarizer
        summary_chunks = summarizer.create_summary_chunks(article_name)
        
        # Display results
        print("\n" + "=" * 50)
        print("SUMMARIZATION RESULTS")
        print("=" * 50)
        
        if summary_chunks:
            print(f"✓ Successfully created {len(summary_chunks)} summary chunks!")
            
            for i, chunk in enumerate(summary_chunks, 1):
                chunk_type = chunk['summary_metadata']['summary_type']
                token_count = chunk['debug_metadata']['chunk_size']
                content_preview = chunk['content'][:100] + "..."
                
                print(f"\n{i}. {chunk_type.upper()} Summary ({token_count} tokens):")
                print(f"   Content: {content_preview}")
                print(f"   Chunk ID: {chunk['chunk_id']}")
            
            print(f"\nSummary chunks are ready for RAG database integration!")
            print(f"Each chunk follows the proper metadata structure.")
        else:
            print("❌ No summary chunks were created.")
        
        print("\n" + "=" * 50)
        print("Test completed!")
        return 0
        
    except Exception as e:
        logger.error(f"Error during summarization test: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
