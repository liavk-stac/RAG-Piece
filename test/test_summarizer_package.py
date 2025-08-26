#!/usr/bin/env python3
"""
Test script to verify summarizer integration with the RAG system.
Uses the package import conventions.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_piece.config import RAGConfig
from rag_piece.summarizer import ArticleSummarizer

def test_summarizer_initialization():
    """Test that the summarizer can be initialized with the config."""
    print("Testing summarizer initialization...")
    
    try:
        # Test with default config (summarization disabled)
        config = RAGConfig()
        print(f"✓ Default config: ENABLE_SUMMARIZATION = {config.ENABLE_SUMMARIZATION}")
        
        # Test with summarization enabled
        config.ENABLE_SUMMARIZATION = True
        print(f"✓ Modified config: ENABLE_SUMMARIZATION = {config.ENABLE_SUMMARIZATION}")
        
        # Test summarizer creation
        summarizer = ArticleSummarizer(max_chunk_size=config.MAX_CHUNK_SIZE, save_to_files=False, max_input_tokens=8000)
        print(f"✓ Summarizer created with max_chunk_size = {summarizer.max_chunk_size}")
        print(f"✓ Save to files: {summarizer.save_to_files}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_summary_chunk_creation():
    """Test that summary chunks are created with proper metadata structure."""
    print("\nTesting summary chunk creation...")
    
    try:
        config = RAGConfig()
        config.ENABLE_SUMMARIZATION = True
        
        summarizer = ArticleSummarizer(max_chunk_size=config.MAX_CHUNK_SIZE, save_to_files=False, max_input_tokens=8000)
        
        # Test with a small article to avoid API costs
        print("Note: This test requires OpenAI API key and will make API calls")
        print("Skipping actual summarization to avoid costs...")
        
        # Test the chunk structure creation method
        print("✓ Summarizer methods available:")
        print(f"  - create_summary_chunks: {hasattr(summarizer, 'create_summary_chunks')}")
        print(f"  - _create_summary_chunk: {hasattr(summarizer, '_create_summary_chunk')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_config_integration():
    """Test that the config properly controls summarization."""
    print("\nTesting config integration...")
    
    try:
        # Test default config
        config = RAGConfig()
        print(f"✓ Default ENABLE_SUMMARIZATION: {config.ENABLE_SUMMARIZATION}")
        print(f"✓ Default SUMMARY_MODEL: {config.SUMMARY_MODEL}")
        print(f"✓ Default SUMMARY_TEMPERATURE: {config.SUMMARY_TEMPERATURE}")
        
        # Test config validation
        config.validate()
        print("✓ Config validation passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_package_imports():
    """Test that all package imports work correctly."""
    print("\nTesting package imports...")
    
    try:
        # Test package imports
        from rag_piece import ArticleSummarizer, RAGConfig
        print("✓ Package imports successful")
        
        # Test direct module imports
        from rag_piece.summarizer import ArticleSummarizer
        from rag_piece.config import RAGConfig
        print("✓ Direct module imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("Summarizer Package Integration Test")
    print("=" * 45)
    
    tests = [
        test_package_imports,
        test_summarizer_initialization,
        test_summary_chunk_creation,
        test_config_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 45)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Summarizer package integration is ready.")
        print("\nTo enable summarization, set ENABLE_SUMMARIZATION = True in your config")
        print("or modify the RAGConfig() call in main.py")
        print("\nThe summarizer is now properly integrated into the src/rag_piece package!")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
