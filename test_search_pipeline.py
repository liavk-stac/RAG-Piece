#!/usr/bin/env python3
"""
Search Engine Pipeline Test
This file tests each step of the SearchEngine individually to find where truncation occurs.
"""

import sys
import pickle
import json
from pathlib import Path
from typing import Dict, Any, List

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from rag_piece.config import RAGConfig
from rag_piece.utils import count_tokens


def load_database_chunk(chunk_id: str) -> Dict[str, Any]:
    """Load a specific chunk directly from the database for comparison."""
    mapping_path = Path("data/rag_db/chunk_mapping.pkl")
    
    if not mapping_path.exists():
        print(f"❌ Chunk mapping file not found: {mapping_path}")
        return {}
    
    try:
        with open(mapping_path, 'rb') as f:
            chunk_mapping = pickle.load(f)
        
        # Find chunk by ID
        for db_id, chunk in chunk_mapping.items():
            if chunk.get('chunk_id') == chunk_id:
                return chunk
        
        print(f"❌ Chunk ID '{chunk_id}' not found in database")
        return {}
    except Exception as e:
        print(f"❌ Error loading chunk mapping: {e}")
        return {}


def test_whoosh_direct_access(query: str) -> List[Dict[str, Any]]:
    """Test Whoosh index access directly without SearchEngine wrapper."""
    print(f"🔍 TESTING WHOOSH DIRECT ACCESS")
    print("=" * 60)
    
    try:
        from whoosh import index
        from whoosh.qparser import MultifieldParser
        from whoosh.scoring import BM25F
        
        # Open Whoosh index directly
        whoosh_dir = "data/rag_db/whoosh_index"
        ix = index.open_dir(whoosh_dir)
        
        print(f"✅ Whoosh index opened successfully")
        print(f"📊 Index schema fields: {list(ix.schema.names())}")
        
        with ix.searcher(weighting=BM25F()) as searcher:
            # Create parser
            parser = MultifieldParser(['content', 'article_name'], ix.schema)
            parsed_query = parser.parse(query)
            
            # Search
            results = searcher.search(parsed_query, limit=2)  # Limit to 2 for readability
            print(f"🔎 Found {len(results)} results from Whoosh")
            
            whoosh_results = []
            for i, hit in enumerate(results):
                content = hit['content']
                content_tokens = count_tokens(content)
                
                print(f"\n📋 WHOOSH RESULT #{i+1}:")
                print(f"   Chunk ID: {hit['chunk_id']}")
                print(f"   Content Length: {len(content):,} characters")
                print(f"   Token Count: {content_tokens} tokens")
                print(f"   Score: {hit.score:.3f}")
                
                # Check for truncation
                if content.endswith('...') or content.endswith('…'):
                    print("   ⚠️  CONTENT APPEARS TRUNCATED")
                elif content.endswith('.') or content.endswith('!') or content.endswith('?'):
                    print("   ✅ Content appears complete")
                else:
                    print("   ❓ Content ends abruptly")
                
                # Show search metadata
                print(f"   SEARCH METADATA:")
                print("-" * 20)
                print(f"   Article Name: {hit.get('article_name', 'N/A')}")
                print(f"   Sub-Article Name: {hit.get('sub_article_name', 'N/A')}")
                print(f"   Section Name: {hit.get('section_name', 'N/A')}")
                print(f"   Sub-Section Name: {hit.get('sub_section_name', 'N/A')}")
                print(f"   Keywords: {hit.get('keywords', 'N/A')}")
                print(f"   Debug Metadata: {hit.get('debug_metadata', 'N/A')}")
                
                # Show full content
                print(f"   FULL CONTENT:")
                print("-" * 40)
                print(content)
                print("-" * 40)
                
                whoosh_results.append({
                    'chunk_id': hit['chunk_id'],
                    'content': content,
                    'score': hit.score,
                    'source': 'whoosh_direct'
                })
            
            return whoosh_results
            
    except Exception as e:
        print(f"❌ Error in Whoosh direct access: {e}")
        return []


def test_search_engine_bm25(query: str) -> List[Dict[str, Any]]:
    """Test SearchEngine BM25 method specifically."""
    print(f"\n🔍 TESTING SEARCHENGINE BM25 METHOD")
    print("=" * 60)
    
    try:
        from rag_piece.search import SearchEngine
        
        config = RAGConfig()
        search_engine = SearchEngine(config)
        
        # Load indices
        if not search_engine.load_indices():
            print("❌ Failed to load search indices")
            return []
        
        print("✅ SearchEngine loaded successfully")
        
        # Test BM25 search directly
        bm25_results = search_engine._bm25_search(query, limit=2)  # Limit to 2 for readability
        print(f"🔎 BM25 found {len(bm25_results)} results")
        
        for i, result in enumerate(bm25_results):
            content = result.get('content', '')
            content_tokens = count_tokens(content)
            
            print(f"\n📋 BM25 RESULT #{i+1}:")
            print(f"   Chunk ID: {result.get('chunk_id', 'N/A')}")
            print(f"   Content Length: {len(content):,} characters")
            print(f"   Token Count: {content_tokens} tokens")
            print(f"   BM25 Score: {result.get('bm25_score', 'N/A'):.3f}")
            
            # Check for truncation
            if content.endswith('...') or content.endswith('…'):
                print("   ⚠️  CONTENT APPEARS TRUNCATED")
            elif content.endswith('.') or content.endswith('!') or content.endswith('?'):
                print("   ✅ Content appears complete")
            else:
                print("   ❓ Content ends abruptly")
            
            # Show search metadata
            search_meta = result.get('search_metadata', {})
            print(f"   SEARCH METADATA:")
            print("-" * 20)
            print(f"   Article Name: {search_meta.get('article_name', 'N/A')}")
            print(f"   Sub-Article Name: {search_meta.get('sub_article_name', 'N/A')}")
            print(f"   Section Name: {search_meta.get('section_name', 'N/A')}")
            print(f"   Sub-Section Name: {search_meta.get('sub_section_name', 'N/A')}")
            print(f"   Keywords: {search_meta.get('keywords', 'N/A')}")
            
            # Show debug metadata if available
            debug_meta = result.get('debug_metadata', {})
            if debug_meta:
                print(f"   DEBUG METADATA:")
                print("-" * 20)
                for key, value in debug_meta.items():
                    print(f"   {key}: {value}")
            
            # Show full content
            print(f"   FULL CONTENT:")
            print("-" * 40)
            print(content)
            print("-" * 40)
            
            # Mark source
            result['source'] = 'searchengine_bm25'
        
        return bm25_results
        
    except Exception as e:
        print(f"❌ Error in SearchEngine BM25 test: {e}")
        return []


def test_search_engine_full(query: str) -> List[Dict[str, Any]]:
    """Test full SearchEngine search method."""
    print(f"\n🔍 TESTING FULL SEARCHENGINE SEARCH")
    print("=" * 60)
    
    try:
        from rag_piece.search import SearchEngine
        
        config = RAGConfig()
        search_engine = SearchEngine(config)
        
        # Load indices
        if not search_engine.load_indices():
            print("❌ Failed to load search indices")
            return []
        
        print("✅ SearchEngine loaded successfully")
        
        # Test full search
        search_results = search_engine.search(query, top_k=2)  # Limit to 2 for readability
        print(f"🔎 Full search found {len(search_results)} results")
        
        for i, result in enumerate(search_results):
            content = result.get('content', '')
            content_tokens = count_tokens(content)
            
            print(f"\n📋 FULL SEARCH RESULT #{i+1}:")
            print(f"   Chunk ID: {result.get('chunk_id', 'N/A')}")
            print(f"   Content Length: {len(content):,} characters")
            print(f"   Token Count: {content_tokens} tokens")
            print(f"   BM25 Score: {result.get('bm25_score', 'N/A'):.3f}")
            print(f"   Semantic Score: {result.get('semantic_score', 'N/A'):.3f}")
            print(f"   Combined Score: {result.get('combined_score', 'N/A'):.3f}")
            
            # Check for truncation
            if content.endswith('...') or content.endswith('…'):
                print("   ⚠️  CONTENT APPEARS TRUNCATED")
            elif content.endswith('.') or content.endswith('!') or content.endswith('?'):
                print("   ✅ Content appears complete")
            else:
                print("   ❓ Content ends abruptly")
            
            # Show search metadata
            search_meta = result.get('search_metadata', {})
            print(f"   SEARCH METADATA:")
            print("-" * 20)
            print(f"   Article Name: {search_meta.get('article_name', 'N/A')}")
            print(f"   Sub-Article Name: {search_meta.get('sub_article_name', 'N/A')}")
            print(f"   Section Name: {search_meta.get('section_name', 'N/A')}")
            print(f"   Sub-Section Name: {search_meta.get('sub_section_name', 'N/A')}")
            print(f"   Keywords: {search_meta.get('keywords', 'N/A')}")
            
            # Show debug metadata if available
            debug_meta = result.get('debug_metadata', {})
            if debug_meta:
                print(f"   DEBUG METADATA:")
                print("-" * 20)
                for key, value in debug_meta.items():
                    print(f"   {key}: {value}")
            
            # Show summary metadata if available
            summary_meta = result.get('summary_metadata', {})
            if summary_meta:
                print(f"   SUMMARY METADATA:")
                print("-" * 20)
                for key, value in summary_meta.items():
                    print(f"   {key}: {value}")
            
            # Show full content
            print(f"   FULL CONTENT:")
            print("-" * 40)
            print(content)
            print("-" * 40)
            
            # Mark source
            result['source'] = 'searchengine_full'
        
        return search_results
        
    except Exception as e:
        print(f"❌ Error in full SearchEngine test: {e}")
        return []


def compare_results(database_chunk: Dict[str, Any], search_results: List[Dict[str, Any]]) -> None:
    """Compare database content with search results to find truncation point."""
    print(f"\n🔍 COMPARISON ANALYSIS")
    print("=" * 60)
    
    if not database_chunk:
        print("❌ No database chunk to compare with")
        return
    
    db_content = database_chunk.get('content', '')
    db_tokens = count_tokens(db_content)
    
    print(f"📚 DATABASE CHUNK:")
    print(f"   Content Length: {len(db_content):,} characters")
    print(f"   Token Count: {db_tokens} tokens")
    print(f"   Chunk ID: {database_chunk.get('chunk_id', 'N/A')}")
    
    for result in search_results:
        if result.get('chunk_id') == database_chunk.get('chunk_id'):
            search_content = result.get('content', '')
            search_tokens = count_tokens(search_content)
            source = result.get('source', 'unknown')
            
            print(f"\n📋 {source.upper()} RESULT:")
            print(f"   Content Length: {len(search_content):,} characters")
            print(f"   Token Count: {search_tokens} tokens")
            
            # Compare
            if len(search_content) == len(db_content):
                print("   ✅ CONTENT LENGTH MATCHES DATABASE")
            else:
                print(f"   ⚠️  CONTENT TRUNCATED: {len(db_content) - len(search_content):,} characters lost")
                print(f"   📉 Token difference: {db_tokens - search_tokens} tokens")
            
            # Check if content matches
            if search_content == db_content:
                print("   ✅ CONTENT IDENTICAL TO DATABASE")
            else:
                print("   ⚠️  CONTENT DIFFERS FROM DATABASE")
                # Show where they differ
                for i, (db_char, search_char) in enumerate(zip(db_content, search_content)):
                    if db_char != search_char:
                        print(f"   📍 First difference at character {i}")
                        print(f"   DB: '{db_content[max(0, i-20):i+20]}'")
                        print(f"   Search: '{search_content[max(0, i-20):i+20]}'")
                        break
            
            break
    else:
        print("❌ Chunk not found in search results")


def main():
    """Main function to test the search pipeline step by step."""
    print("🔍 SEARCH ENGINE PIPELINE TEST")
    print("=" * 60)
    print("Testing each step of the search pipeline to find truncation source.")
    print()
    
    try:
        # Test query
        query = "Roronoa Zoro personality"
        print(f"🔎 Test Query: '{query}'")
        print()
        
        # Step 1: Test Whoosh direct access
        print("Starting Whoosh direct access test...")
        whoosh_results = test_whoosh_direct_access(query)
        
        # Step 2: Test SearchEngine BM25
        print("\nStarting SearchEngine BM25 test...")
        bm25_results = test_search_engine_bm25(query)
        
        # Step 3: Test full SearchEngine
        print("\nStarting full SearchEngine test...")
        full_results = test_search_engine_full(query)
        
        # Step 4: Compare with database
        if whoosh_results:
            # Get the first result's chunk ID and compare with database
            first_chunk_id = whoosh_results[0].get('chunk_id')
            if first_chunk_id:
                print(f"\n🔍 LOADING DATABASE CHUNK FOR COMPARISON")
                print("=" * 60)
                database_chunk = load_database_chunk(first_chunk_id)
                
                # Compare all results with database
                all_results = whoosh_results + bm25_results + full_results
                compare_results(database_chunk, all_results)
        
        print(f"\n🎯 PIPELINE TEST COMPLETE")
        print("=" * 60)
        print("Check the results above to see where truncation occurs in the pipeline.")
        
    except Exception as e:
        print(f"❌ Error in main test function: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
