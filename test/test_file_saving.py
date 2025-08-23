#!/usr/bin/env python3
"""
Test script to verify that the summarizer can save summaries to files.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_piece.summarizer import ArticleSummarizer

def test_file_saving():
    """Test that the summarizer can save summaries to files."""
    print("Testing File Saving Functionality")
    print("=" * 40)
    
    try:
        # Test with file saving enabled
        summarizer = ArticleSummarizer(max_chunk_size=400, save_to_files=True)
        print(f"✓ Summarizer created with save_to_files = {summarizer.save_to_files}")
        
        # Test the file saving method directly
        test_summary = "This is a test summary for testing file saving functionality."
        test_article = "Test Article"
        test_sub_article = "Test Sub Article"
        test_type = "test_summary"
        
        # Test saving main article summary
        print("\nTesting main article summary saving...")
        file_path = summarizer._save_summary_to_file(
            test_summary, test_article, None, test_type
        )
        if file_path:
            print(f"✓ Main summary saved to: {file_path}")
        else:
            print("❌ Failed to save main summary")
        
        # Test saving sub-article summary
        print("\nTesting sub-article summary saving...")
        file_path = summarizer._save_summary_to_file(
            test_summary, test_article, test_sub_article, test_type
        )
        if file_path:
            print(f"✓ Sub-article summary saved to: {file_path}")
        else:
            print("❌ Failed to save sub-article summary")
        
        # Check if files were actually created
        summaries_dir = Path("summaries")
        if summaries_dir.exists():
            print(f"\n✓ Summaries directory created: {summaries_dir}")
            
            test_article_dir = summaries_dir / "Test_Article"
            if test_article_dir.exists():
                print(f"✓ Article directory created: {test_article_dir}")
                
                files = list(test_article_dir.glob("*.txt"))
                print(f"✓ Found {len(files)} summary files:")
                for file in files:
                    print(f"  - {file.name}")
            else:
                print("❌ Article directory not created")
        else:
            print("❌ Summaries directory not created")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run the file saving test."""
    success = test_file_saving()
    
    print("\n" + "=" * 40)
    if success:
        print("✓ File saving test completed successfully!")
        print("Check the summaries/ folder for test files.")
    else:
        print("❌ File saving test failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
