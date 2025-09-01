#!/usr/bin/env python3
"""
Test script to print all chunks in the RAG database with complete metadata.
No truncation - shows full content and all metadata fields.
Reads directly from database files, bypassing search engine.
"""

import sys
import pickle
import json
from pathlib import Path
from typing import Dict, Any

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


def print_separator(title="", char="=", width=120):
    """Print a separator line with optional title."""
    if title:
        title = f" {title} "
        padding = (width - len(title)) // 2
        line = char * padding + title + char * (width - padding - len(title))
    else:
        line = char * width
    print(line)


def print_chunk_complete(chunk_id: int, chunk_data: Dict[str, Any], index: int):
    """Print complete information about a single chunk with NO truncation."""
    print_separator(f"CHUNK {index + 1} (ID: {chunk_id})", "=", 120)
    
    # Basic chunk information
    print(f"📋 CHUNK ID: {chunk_id}")
    print(f"🏷️  CHUNK_ID (internal): {chunk_data.get('chunk_id', 'N/A')}")
    
    # Content analysis
    content = chunk_data.get('content', '')
    content_length = len(content)
    try:
        content_tokens = count_tokens(content)
    except:
        content_tokens = len(content.split())  # Fallback to word count
    
    print(f"📏 CONTENT LENGTH: {content_length:,} characters")
    print(f"🎯 TOKEN COUNT: {content_tokens} tokens")
    
    # Metadata sections
    print_separator("ALL METADATA FIELDS", "-", 120)
    
    # Print ALL top-level fields
    for key, value in chunk_data.items():
        if key == 'content':
            continue  # We'll handle content separately
        
        print(f"\n🏷️  {key.upper()}:")
        if isinstance(value, dict):
            print(json.dumps(value, indent=4, ensure_ascii=False))
        elif isinstance(value, list):
            if len(value) == 0:
                print("   []")
            else:
                print(json.dumps(value, indent=4, ensure_ascii=False))
        else:
            print(f"   {value}")
    
    # Keywords section (if exists)
    keywords = chunk_data.get('keywords', [])
    if keywords:
        print_separator("KEYWORDS", "-", 120)
        print(f"🔑 TOTAL KEYWORDS: {len(keywords)}")
        print(f"🔑 KEYWORDS LIST: {', '.join(keywords)}")
    
    # COMPLETE content section - NO TRUNCATION
    print_separator("COMPLETE CONTENT (NO TRUNCATION)", "-", 120)
    print("📝 FULL CONTENT:")
    print(content)
    
    # Content analysis
    print_separator("CONTENT ANALYSIS", "-", 120)
    if content.endswith('...') or content.endswith('…'):
        print("⚠️  CONTENT APPEARS TRUNCATED (ends with ...)")
    elif content.endswith('.') or content.endswith('!') or content.endswith('?'):
        print("✅ Content appears complete (ends with sentence terminator)")
    else:
        print("❓ Content ends abruptly (no clear ending)")
    
    # Show first and last 100 characters for quick reference
    if len(content) > 200:
        print(f"\n📖 FIRST 100 CHARS: {repr(content[:100])}")
        print(f"📖 LAST 100 CHARS: {repr(content[-100:])}")
    
    print("\n" + "="*120 + "\n")  # End of chunk separator


def main():
    """Main function to load and display all database chunks directly from files."""
    print_separator("RAG DATABASE DIRECT CHUNK READER", "=", 120)
    print("🔍 Reading all chunks directly from database files (NO SEARCH ENGINE)")
    print("📄 This shows the RAW content exactly as stored in the database")
    print()
    
    # Check database structure
    db_path = Path("data/rag_db")
    if not db_path.exists():
        print(f"❌ Database directory not found: {db_path}")
        print("Please run main.py first to build the database.")
        return
    
    print(f"📁 Database path: {db_path}")
    
    # Load chunks directly from pickle file
    print("📖 Loading chunks directly from chunk_mapping.pkl...")
    chunk_mapping = load_chunk_mapping()
    
    if not chunk_mapping:
        print("❌ No chunks found in the database.")
        return
    
    print(f"✅ Found {len(chunk_mapping)} chunks in the database.")
    print()
    
    # Print summary statistics
    print_separator("DATABASE SUMMARY", "=", 120)
    
    # Analyze all chunks
    total_content_length = 0
    total_keywords = 0
    chunk_types = {}
    summary_chunks = []
    content_chunks = []
    
    for chunk_id, chunk_data in chunk_mapping.items():
        # Count content
        content = chunk_data.get('content', '')
        total_content_length += len(content)
        
        # Count keywords
        keywords = chunk_data.get('keywords', [])
        total_keywords += len(keywords)
        
        # Categorize chunks
        chunk_type = chunk_data.get('summary_metadata', {}).get('chunk_type', 'content')
        chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
        
        if chunk_type == 'summary':
            summary_chunks.append((chunk_id, chunk_data))
        else:
            content_chunks.append((chunk_id, chunk_data))
    
    print(f"📊 TOTAL CHUNKS: {len(chunk_mapping)}")
    print(f"📊 TOTAL CONTENT LENGTH: {total_content_length:,} characters")
    print(f"📊 TOTAL KEYWORDS: {total_keywords}")
    print(f"📊 AVERAGE CONTENT LENGTH: {total_content_length // len(chunk_mapping):,} characters per chunk")
    print(f"📊 AVERAGE KEYWORDS: {total_keywords / len(chunk_mapping):.1f} keywords per chunk")
    
    print(f"\n📊 CHUNK TYPES:")
    for chunk_type, count in chunk_types.items():
        print(f"   • {chunk_type.title()}: {count} chunks")
    
    print()
    print_separator("ALL CHUNKS - COMPLETE DATA", "=", 120)
    print()
    
    # Print ALL chunks with complete data
    all_chunks = list(chunk_mapping.items())
    all_chunks.sort(key=lambda x: x[0])  # Sort by chunk ID
    
    for index, (chunk_id, chunk_data) in enumerate(all_chunks):
        print_chunk_complete(chunk_id, chunk_data, index)
    
    print_separator("ANALYSIS COMPLETE", "=", 120)
    print(f"✅ Successfully displayed {len(chunk_mapping)} chunks with COMPLETE content and metadata.")
    print("📄 This is the RAW data exactly as stored in the database pickle file.")


if __name__ == "__main__":
    main()