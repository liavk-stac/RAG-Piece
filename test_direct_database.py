#!/usr/bin/env python3
"""
Direct Database Access Test
This file bypasses all search functions and accesses the database directly
to see exactly what chunks are stored and their content.
"""

import sys
import pickle
import json
from pathlib import Path
from typing import Dict, Any, List

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from rag_piece.utils import count_tokens


def load_chunk_mapping() -> Dict[int, Dict[str, Any]]:
    """Load the raw chunk mapping directly from the pickle file."""
    mapping_path = Path("data/rag_db/chunk_mapping.pkl")
    
    if not mapping_path.exists():
        print(f"❌ Chunk mapping file not found: {mapping_path}")
        return {}
    
    try:
        with open(mapping_path, 'rb') as f:
            chunk_mapping = pickle.load(f)
        print(f"✅ Loaded chunk mapping with {len(chunk_mapping)} chunks")
        return chunk_mapping
    except Exception as e:
        print(f"❌ Error loading chunk mapping: {e}")
        return {}


def load_whoosh_index_info() -> Dict[str, Any]:
    """Get basic information about the Whoosh index without using SearchEngine."""
    whoosh_dir = Path("data/rag_db/whoosh_index")
    
    if not whoosh_dir.exists():
        print(f"❌ Whoosh index directory not found: {whoosh_dir}")
        return {}
    
    try:
        # Try to get basic file info
        index_files = list(whoosh_dir.rglob("*"))
        print(f"✅ Whoosh index directory found with {len(index_files)} files")
        
        # Look for schema files or other metadata
        schema_files = [f for f in index_files if "schema" in f.name.lower() or f.suffix in ['.toc', '.idx']]
        print(f"   Index files: {[f.name for f in schema_files[:5]]}")
        
        return {"index_files": len(index_files), "schema_files": len(schema_files)}
    except Exception as e:
        print(f"❌ Error examining Whoosh index: {e}")
        return {}


def load_faiss_index_info() -> Dict[str, Any]:
    """Get basic information about the FAISS index without using SearchEngine."""
    faiss_path = Path("data/rag_db/faiss_index.bin")
    
    if not faiss_path.exists():
        print(f"❌ FAISS index file not found: {faiss_path}")
        return {}
    
    try:
        file_size = faiss_path.stat().st_size
        print(f"✅ FAISS index file found: {file_size:,} bytes")
        return {"file_size": file_size}
    except Exception as e:
        print(f"❌ Error examining FAISS index: {e}")
        return {}


def analyze_chunk_content(chunk: Dict[str, Any], chunk_id: int) -> None:
    """Analyze a single chunk's content in detail."""
    print(f"\n🔍 CHUNK #{chunk_id} ANALYSIS:")
    print("=" * 60)
    
    # Basic chunk info
    print(f"📋 Chunk ID: {chunk.get('chunk_id', 'N/A')}")
    print(f"📚 Article: {chunk.get('search_metadata', {}).get('article_name', 'N/A')}")
    print(f"📖 Section: {chunk.get('search_metadata', {}).get('section_name', 'N/A')}")
    
    # Content analysis
    content = chunk.get('content', '')
    content_length = len(content)
    content_tokens = count_tokens(content)
    
    print(f"📏 Content Length: {content_length:,} characters")
    print(f"🎯 Token Count: {content_tokens} tokens")
    
    # Check if content appears truncated
    if content.endswith('...') or content.endswith('…'):
        print("⚠️  CONTENT APPEARS TRUNCATED (ends with ...)")
    elif content.endswith('.') or content.endswith('!') or content.endswith('?'):
        print("✅ Content appears complete (ends with sentence terminator)")
    else:
        print("❓ Content ends abruptly (no clear ending)")
    
    # Show content preview
    print(f"\n📝 CONTENT PREVIEW (first 200 chars):")
    print("-" * 40)
    preview = content[:200] + "..." if len(content) > 200 else content
    print(preview)
    
    # Show content ending
    if len(content) > 200:
        print(f"\n📝 CONTENT ENDING (last 200 chars):")
        print("-" * 40)
        ending = "..." + content[-200:] if len(content) > 200 else content
        print(ending)
    
    # Metadata analysis
    if 'summary_metadata' in chunk:
        print(f"\n🏷️  SUMMARY METADATA:")
        print("-" * 40)
        summary_meta = chunk['summary_metadata']
        print(f"   Chunk Type: {summary_meta.get('chunk_type', 'N/A')}")
        print(f"   Summary Type: {summary_meta.get('summary_type', 'N/A')}")
        print(f"   Generation Method: {summary_meta.get('generation_method', 'N/A')}")
        print(f"   Model Used: {summary_meta.get('model_used', 'N/A')}")
    
    if 'debug_metadata' in chunk:
        print(f"\n🔧 DEBUG METADATA:")
        print("-" * 40)
        debug_meta = chunk['debug_metadata']
        print(f"   Target Token Limit: {debug_meta.get('target_token_limit', 'N/A')}")
        token_efficiency = debug_meta.get('token_efficiency', 'N/A')
        if isinstance(token_efficiency, (int, float)):
            print(f"   Token Efficiency: {token_efficiency:.1f}%")
        else:
            print(f"   Token Efficiency: {token_efficiency}")
        print(f"   Processing Timestamp: {debug_meta.get('processing_timestamp', 'N/A')}")


def main():
    """Main function to test direct database access."""
    print("🔍 DIRECT DATABASE ACCESS TEST")
    print("=" * 60)
    print("This test bypasses all search functions to see raw database content.")
    print()
    
    # Check database structure
    print("📁 DATABASE STRUCTURE CHECK:")
    print("-" * 40)
    
    db_path = Path("data/rag_db")
    if not db_path.exists():
        print(f"❌ Database directory not found: {db_path}")
        print("Make sure you have run the main system first to build the database.")
        return
    
    print(f"✅ Database directory found: {db_path}")
    
    # Load chunk mapping
    print(f"\n📊 LOADING CHUNK DATA:")
    print("-" * 40)
    chunk_mapping = load_chunk_mapping()
    
    if not chunk_mapping:
        print("❌ No chunks found in database!")
        return
    
    # Load index information
    print(f"\n🔍 INDEX INFORMATION:")
    print("-" * 40)
    whoosh_info = load_whoosh_index_info()
    faiss_info = load_faiss_index_info()
    
    # Analyze chunks
    print(f"\n📋 CHUNK ANALYSIS:")
    print("-" * 40)
    print(f"Found {len(chunk_mapping)} chunks in database")
    
    # Show chunk types breakdown
    chunk_types = {}
    summary_chunks = []
    content_chunks = []
    
    for chunk_id, chunk in chunk_mapping.items():
        chunk_type = chunk.get('summary_metadata', {}).get('chunk_type', 'content')
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        if chunk_type == 'summary':
            summary_chunks.append((chunk_id, chunk))
        else:
            content_chunks.append((chunk_id, chunk))
    
    print(f"\n📊 CHUNK BREAKDOWN:")
    for chunk_type, count in chunk_types.items():
        print(f"   {chunk_type.title()} chunks: {count}")
    
    # Analyze a few chunks of each type
    print(f"\n🔍 ANALYZING SAMPLE CHUNKS:")
    print("-" * 40)
    
    # Show first few content chunks
    if content_chunks:
        print(f"\n📚 CONTENT CHUNKS (showing first 3):")
        for i, (chunk_id, chunk) in enumerate(content_chunks[:3]):
            analyze_chunk_content(chunk, chunk_id)
    
    # Show first few summary chunks
    if summary_chunks:
        print(f"\n📝 SUMMARY CHUNKS (showing first 3):")
        for i, (chunk_id, chunk) in enumerate(summary_chunks[:3]):
            analyze_chunk_content(chunk, chunk_id)
    
    # Summary
    print(f"\n🎯 ANALYSIS COMPLETE:")
    print("-" * 40)
    print(f"Total chunks analyzed: {len(chunk_mapping)}")
    print(f"Content chunks: {len(content_chunks)}")
    print(f"Summary chunks: {len(summary_chunks)}")
    print(f"\nThis shows exactly what's stored in the database.")
    print(f"Compare this with what the search functions return to identify truncation sources.")


if __name__ == "__main__":
    main()
