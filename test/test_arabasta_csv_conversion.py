"""
Test script for converting the Arabasta Kingdom Citizens CSV file to text.
This script demonstrates the CSV to text conversion capabilities with a real One Piece CSV file.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_piece.config import RAGConfig
from rag_piece.csv_to_text import CSVToTextConverter


def main():
    """Main function to test CSV to text conversion with Arabasta Kingdom data."""
    print("=" * 60)
    print("Testing CSV to Text Conversion: Arabasta Kingdom Citizens")
    print("=" * 60)
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✓ Environment variables loaded")
    except ImportError:
        print("⚠️  python-dotenv not installed, using system environment")
    
    # Check for OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        print("Please set your OpenAI API key in a .env file or environment variable")
        return
    
    print("✓ OpenAI API key found")
    
    # Create configuration with debug saving enabled
    config = RAGConfig()
    config.ENABLE_CSV_TO_TEXT = True
    config.SAVE_CSV_TO_TEXT_FILES = True  # Always save debug files for testing
    config.ENABLE_SUMMARIZATION = True
    
    print("✓ Configuration created with debug file saving enabled")
    
    try:
        # Initialize CSV to text converter
        print("\nInitializing CSV to Text converter...")
        converter = CSVToTextConverter(config)
        print("✓ CSV to Text converter initialized")
        
        # Define the CSV file path and article name
        csv_file_path = "csv_files/Arabasta_Kingdom/Arabasta_Kingdom_Citizens.csv"
        article_name = "Arabasta Kingdom"
        
        # Create a sample article summary for context
        existing_summary = """Arabasta Kingdom is a desert kingdom located in the Grand Line, known for its vast deserts and ancient civilization. 
        It was ruled by the Nefertari family for generations and faced a civil war orchestrated by the Baroque Works organization. 
        The kingdom is home to diverse citizens including royalty, royal guards, desert tribes, and various factions. 
        Notable features include the capital city Alubarna, the Sandora Desert, and the Super Spot-Billed Duck Troops who serve as royal messengers."""
        
        # Check if CSV file exists
        if not os.path.exists(csv_file_path):
            print(f"\n❌ CSV file not found: {csv_file_path}")
            print("Please ensure the file exists in the correct location")
            return
        
        print(f"\n📁 CSV file found: {csv_file_path}")
        print(f"📖 Article: {article_name}")
        print("🔍 Using existing summary to avoid duplicate summarizer runs")
        
        # Convert CSV to text
        print("\n🔄 Converting CSV to text...")
        result = converter.convert_csv_to_text(csv_file_path, article_name, existing_summary)
        
        if result['conversion_metadata']['conversion_success']:
            print("✅ CSV conversion successful!")
            print(f"📊 Converted text length: {result['conversion_metadata']['converted_text_length']} characters")
            print(f"🔢 Converted text tokens: {result['conversion_metadata']['converted_text_tokens']}")
            print(f"📝 Article summary length: {result['conversion_metadata']['article_summary_length']} characters")
            
            # Show preview of converted text
            print("\n" + "=" * 60)
            print("CONVERTED TEXT PREVIEW (First 500 characters):")
            print("=" * 60)
            preview = result['converted_text'][:500] + "..." if len(result['converted_text']) > 500 else result['converted_text']
            print(preview)
            
            # Show debug file information
            if config.SAVE_CSV_TO_TEXT_FILES:
                print("\n" + "=" * 60)
                print("DEBUG FILE INFORMATION:")
                print("=" * 60)
                print("✓ Debug file saved to: data/debug/csv2text/")
                print("📁 Check the folder for the complete converted text")
                print("🔍 File naming format: arabasta-kingdom_arabasta-kingdom-citizens_TIMESTAMP.txt")
                
                # List debug files
                debug_dir = Path("data/debug/csv2text")
                if debug_dir.exists():
                    debug_files = list(debug_dir.glob("*.txt"))
                    if debug_files:
                        print(f"\n📋 Found {len(debug_files)} debug file(s):")
                        for debug_file in debug_files:
                            print(f"   - {debug_file.name}")
                    else:
                        print("\n⚠️  No debug files found in the directory")
                else:
                    print("\n⚠️  Debug directory not found")
            
            # Show conversion metadata
            print("\n" + "=" * 60)
            print("CONVERSION METADATA:")
            print("=" * 60)
            metadata = result['conversion_metadata']
            print(f"🕒 Conversion Timestamp: {metadata['conversion_timestamp']}")
            print(f"🤖 Model Used: {metadata['model_used']}")
            print(f"🌡️  Temperature: {metadata['temperature']}")
            print(f"📊 Original CSV Shape: {metadata['original_csv_shape']}")
            print(f"✅ Conversion Success: {metadata['conversion_success']}")
            
        else:
            print("❌ CSV conversion failed!")
            error_msg = result['conversion_metadata'].get('error_message', 'Unknown error')
            print(f"Error: {error_msg}")
            
            # Show error result metadata
            print("\n" + "=" * 60)
            print("ERROR METADATA:")
            print("=" * 60)
            metadata = result['conversion_metadata']
            print(f"🕒 Timestamp: {metadata['conversion_timestamp']}")
            print(f"❌ Success: {metadata['conversion_success']}")
            print(f"🚨 Error: {error_msg}")
        
        print("\n" + "=" * 60)
        print("TEST COMPLETED!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Check the data/debug/csv2text/ folder for the debug file")
        print("2. Review the converted text quality")
        print("3. Verify that data relationships are maintained")
        print("4. Check that the text is suitable for chunking and embedding")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
