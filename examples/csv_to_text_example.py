"""
Example script demonstrating how to use the CSV to Text converter.
This script shows how to convert CSV files to structured text and integrate them into the RAG system.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_piece.config import RAGConfig
from rag_piece.csv_to_text import CSVToTextConverter
from rag_piece.database import RAGDatabase


def main():
    """Main function demonstrating CSV to text conversion."""
    print("CSV to Text Converter Example")
    print("=" * 40)
    
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
    
    # Create configuration
    config = RAGConfig()
    config.ENABLE_CSV_TO_TEXT = True
    config.SAVE_CSV_TO_TEXT_FILES = True  # Enable debug file saving
    config.ENABLE_SUMMARIZATION = True     # Enable summarization for article summaries
    
    print("✓ Configuration created")
    
    try:
        # Initialize CSV to text converter
        print("\nInitializing CSV to Text converter...")
        converter = CSVToTextConverter(config)
        print("✓ CSV to Text converter initialized")
        
        # Example: Convert a CSV file
        # Replace with actual CSV file path and article name
        csv_file_path = "csv_files/example/example_data.csv"
        article_name = "Example Article"
        
        # If you already have a summary from the database creation process, use it:
        existing_summary = "This is an existing summary about the article that was created during database setup."
        
        if not os.path.exists(csv_file_path):
            print(f"\n⚠️  CSV file not found: {csv_file_path}")
            print("Please create a CSV file or update the path in this script")
            print("\nExample CSV structure:")
            print("Character Name,Devil Fruit,Bounty,Crew Position")
            print("Monkey D. Luffy,Gomu Gomu no Mi,3000000000,Captain")
            print("Roronoa Zoro,None,1111000000,Swordsman")
            print("Note: All CSV rows will be included in the conversion")
            return
        
        print(f"\nConverting CSV file: {csv_file_path}")
        print(f"Article: {article_name}")
        print("Using existing summary to avoid duplicate summarizer runs")
        
        # Convert CSV to text using existing summary
        result = converter.convert_csv_to_text(csv_file_path, article_name, existing_summary)
        
        if result['conversion_metadata']['conversion_success']:
            print("✓ CSV conversion successful!")
            print(f"Converted text length: {result['conversion_metadata']['converted_text_length']} characters")
            print(f"Converted text tokens: {result['conversion_metadata']['converted_text_tokens']}")
            
            # Show first 200 characters of converted text
            preview = result['converted_text'][:200] + "..." if len(result['converted_text']) > 200 else result['converted_text']
            print(f"\nConverted text preview:\n{preview}")
            
            # Save debug file if enabled
            if config.SAVE_CSV_TO_TEXT_FILES:
                debug_path = result.get('debug_file_path', '')
                if debug_path:
                    print(f"\n✓ Debug file saved to: {debug_path}")
            
        else:
            print("❌ CSV conversion failed!")
            error_msg = result['conversion_metadata'].get('error_message', 'Unknown error')
            print(f"Error: {error_msg}")
        
        # Example: Convert multiple CSV files
        print("\n" + "=" * 40)
        print("Example: Converting multiple CSV files")
        
        # Find CSV files in the csv_files directory
        csv_dir = Path("csv_files")
        if csv_dir.exists():
            csv_files = list(csv_dir.rglob("*.csv"))
            if csv_files:
                print(f"Found {len(csv_files)} CSV files:")
                for csv_file in csv_files[:3]:  # Show first 3
                    print(f"  - {csv_file}")
                
                # Convert first CSV file as example
                if csv_files:
                    first_csv = str(csv_files[0])
                    article_from_path = csv_files[0].parent.name
                    
                    print(f"\nConverting first CSV file: {first_csv}")
                    result = converter.convert_csv_to_text(first_csv, article_from_path)
                    
                    if result['conversion_metadata']['conversion_success']:
                        print("✓ Multiple CSV conversion successful!")
                    else:
                        print("❌ Multiple CSV conversion failed!")
            else:
                print("No CSV files found in csv_files directory")
        else:
            print("csv_files directory not found")
        
        # Example: Integration with RAG database
        print("\n" + "=" * 40)
        print("Example: Integration with RAG database")
        
        try:
            # Initialize RAG database
            rag_db = RAGDatabase(config)
            print("✓ RAG database initialized")
            
            # Here you would integrate the converted text into your database
            # For example, you could create chunks from the converted text
            # and add them to your search indices
            
            print("✓ Ready for database integration")
            
        except Exception as e:
            print(f"⚠️  RAG database initialization failed: {e}")
            print("This is expected if you haven't set up the full RAG system yet")
        
        print("\n" + "=" * 40)
        print("Example completed successfully!")
        print("\nNext steps:")
        print("1. Update the csv_file_path and article_name variables")
        print("2. Run the script with your actual CSV files")
        print("3. Check the data/debug/csv2text/ folder for debug files")
        print("4. Integrate the converted text into your RAG database")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
