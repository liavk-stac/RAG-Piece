#!/usr/bin/env python3
"""
Simple test file for the search function in search.py
Tests the SearchEngine with queries and displays results
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from rag_piece.search import SearchEngine
from rag_piece.config import RAGConfig


def test_search(query: str):
    """Test the search function with a given query."""
    print(f"🔍 Testing Search Function")
    print("=" * 50)
    print(f"Query: '{query}'")
    print()
    
    try:
        # Initialize configuration and search engine
        print("📚 Loading search indices...")
        config = RAGConfig()
        search_engine = SearchEngine(config)
        
        # Load existing indices
        if not search_engine.load_indices():
            print("❌ Failed to load search indices!")
            print("Make sure you have run the main system first to build the database.")
            return
        
        print("✅ Indices loaded successfully!")
        print()
        
        # Perform search
        print("🔎 Performing search...")
        results = search_engine.search(query, top_k=10)
        
        if not results:
            print("❌ No results found for your query.")
            return
        
        print(f"✅ Found {len(results)} results!")
        print()
        
        # Display top result in detail
        print("🏆 TOP RESULT:")
        print("=" * 50)
        top_result = results[0]
        
        # Content
        print("📝 CONTENT:")
        print("-" * 20)
        print(top_result['content'])
        print()
        
        # Scores
        print("📊 SCORES:")
        print("-" * 20)
        print(f"BM25 Score: {top_result.get('bm25_score', 'N/A'):.3f}")
        print(f"Semantic Score: {top_result.get('semantic_score', 'N/A'):.3f}")
        print(f"Combined Score: {top_result.get('combined_score', 'N/A'):.3f}")
        print()
        
        # Search metadata
        print("🏷️  SEARCH METADATA:")
        print("-" * 20)
        search_meta = top_result['search_metadata']
        print(f"Article: {search_meta.get('article_name', 'N/A')}")
        print(f"Sub-Article: {search_meta.get('sub_article_name', 'N/A')}")
        print(f"Section: {search_meta.get('section_name', 'N/A')}")
        print(f"Sub-Section: {search_meta.get('sub_section_name', 'N/A')}")
        print(f"Keywords: {', '.join(search_meta.get('keywords', []))}")
        print()
        
        # Debug metadata
        print("🔧 DEBUG METADATA:")
        print("-" * 20)
        debug_meta = top_result.get('debug_metadata', {})
        for key, value in debug_meta.items():
            print(f"{key}: {value}")
        print()
        
        # Chunk ID
        print("🆔 CHUNK ID:")
        print("-" * 20)
        print(f"ID: {top_result.get('chunk_id', 'N/A')}")
        print()
        
        # Show all other results in detail
        if len(results) > 1:
            print("📋 ALL RESULTS (Top 10):")
            print("=" * 50)
            for i, result in enumerate(results, 1):
                print(f"🏆 RESULT #{i} (Score: {result.get('combined_score', 0):.3f})")
                print("-" * 40)
                
                # Full content
                print(f"📝 Content:")
                print(result['content'])
                
                # Key metadata
                search_meta = result['search_metadata']
                print(f"📚 Article: {search_meta.get('article_name', 'N/A')}")
                print(f"📖 Section: {search_meta.get('section_name', 'N/A')}")
                print(f"🔑 Keywords: {', '.join(search_meta.get('keywords', []))}")  # Show first 5 keywords
                
                # Scores
                print(f"📊 Scores - BM25: {result.get('bm25_score', 'N/A'):.3f}, Semantic: {result.get('semantic_score', 'N/A'):.3f}")
                print()
        
    except Exception as e:
        print(f"❌ Error during search: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function to run the search test."""
    if len(sys.argv) > 1:
        # Query provided as command line argument
        query = " ".join(sys.argv[1:])
    else:
        # Interactive query input
        print("🔍 One Piece Wiki Search Test")
        print("=" * 50)
        print("Enter your search query (or provide it as command line argument)")
        print("Examples:")
        print("  - What is the Straw Hat Pirates?")
        print("  - Tell me about Luffy's Devil Fruit")
        print("  - Who are the main characters?")
        print()
        query = input("Query: ").strip()
        
        if not query:
            print("❌ No query provided. Exiting.")
            return
    
    test_search(query)


if __name__ == "__main__":
    main()
