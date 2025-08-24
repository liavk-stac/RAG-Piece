"""
Test file for CSV to Text converter functionality.
Tests the conversion of CSV files to structured text using OpenAI GPT-4o-mini.
"""

import os
import sys
import tempfile
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_piece.config import RAGConfig
from rag_piece.csv_to_text import CSVToTextConverter


def create_test_csv():
    """Create a test CSV file for testing."""
    # Create test data that represents One Piece character information
    test_data = {
        'Character Name': ['Monkey D. Luffy', 'Roronoa Zoro', 'Nami', 'Usopp'],
        'Devil Fruit': ['Gomu Gomu no Mi', 'None', 'None', 'None'],
        'Bounty': ['3,000,000,000', '1,111,000,000', '366,000,000', '500,000,000'],
        'Crew Position': ['Captain', 'Swordsman', 'Navigator', 'Sniper'],
        'First Appearance': ['Chapter 1', 'Chapter 3', 'Chapter 8', 'Chapter 23']
    }
    
    df = pd.DataFrame(test_data)
    
    # Create temporary CSV file
    temp_dir = Path("test_data")
    temp_dir.mkdir(exist_ok=True)
    
    csv_path = temp_dir / "test_characters.csv"
    df.to_csv(csv_path, index=False)
    
    return str(csv_path), df


def test_csv_to_text_converter_initialization():
    """Test CSV to Text converter initialization."""
    print("Testing CSV to Text converter initialization...")
    
    # Create config with CSV to text enabled
    config = RAGConfig()
    config.ENABLE_CSV_TO_TEXT = True
    config.SAVE_CSV_TO_TEXT_FILES = True
    
    try:
        # Mock OpenAI API key
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            # Mock the summarizer to avoid actual API calls
            with patch('rag_piece.csv_to_text.ArticleSummarizer'):
                converter = CSVToTextConverter(config)
                
                # Check that converter was initialized correctly
                assert converter.config == config
                assert converter.config.ENABLE_CSV_TO_TEXT is True
                assert converter.config.SAVE_CSV_TO_TEXT_FILES is True
                assert converter.config.CSV_TO_TEXT_MODEL == "gpt-4o-mini"
                assert converter.config.CSV_TO_TEXT_TEMPERATURE == 0.2
                
                print("✓ CSV to Text converter initialization successful")
                
    except Exception as e:
        print(f"✗ CSV to Text converter initialization failed: {e}")
        raise


def test_csv_file_reading():
    """Test CSV file reading functionality."""
    print("Testing CSV file reading...")
    
    # Create test CSV
    csv_path, expected_df = create_test_csv()
    
    try:
        # Create config
        config = RAGConfig()
        config.ENABLE_CSV_TO_TEXT = True
        
        # Mock OpenAI API key
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            # Mock the summarizer
            with patch('rag_piece.csv_to_text.ArticleSummarizer'):
                converter = CSVToTextConverter(config)
                
                # Test CSV reading
                df = converter._read_csv_file(csv_path)
                
                # Verify the data was read correctly
                assert df is not None
                assert df.shape == expected_df.shape
                assert list(df.columns) == list(expected_df.columns)
                assert df.iloc[0]['Character Name'] == 'Monkey D. Luffy'
                
                print("✓ CSV file reading successful")
                
    except Exception as e:
        print(f"✗ CSV file reading failed: {e}")
        raise
    finally:
        # Clean up test file
        if os.path.exists(csv_path):
            os.remove(csv_path)
        if os.path.exists("test_data"):
            os.rmdir("test_data")


def test_csv_structure_description():
    """Test CSV structure description generation."""
    print("Testing CSV structure description...")
    
    # Create test CSV
    csv_path, test_df = create_test_csv()
    
    try:
        # Create config
        config = RAGConfig()
        config.ENABLE_CSV_TO_TEXT = True
        
        # Mock OpenAI API key
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            # Mock the summarizer
            with patch('rag_piece.csv_to_text.ArticleSummarizer'):
                converter = CSVToTextConverter(config)
                
                # Test structure description
                structure_desc = converter._describe_csv_structure(test_df)
                
                # Verify structure description contains expected information
                assert "Total rows: 4" in structure_desc
                assert "Total columns: 5" in structure_desc
                assert "Character Name" in structure_desc
                assert "Devil Fruit" in structure_desc
                assert "Bounty" in structure_desc
                assert "Crew Position" in structure_desc
                assert "First Appearance" in structure_desc
                
                print("✓ CSV structure description successful")
                
    except Exception as e:
        print(f"✗ CSV structure description failed: {e}")
        raise
    finally:
        # Clean up test file
        if os.path.exists(csv_path):
            os.remove(csv_path)
        if os.path.exists("test_data"):
            os.rmdir("test_data")


def test_csv_content_formatting():
    """Test CSV content formatting for prompts."""
    print("Testing CSV content formatting...")
    
    # Create test CSV
    csv_path, test_df = create_test_csv()
    
    try:
        # Create config
        config = RAGConfig()
        config.ENABLE_CSV_TO_TEXT = True
        
        # Mock OpenAI API key
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            # Mock the summarizer
            with patch('rag_piece.csv_to_text.ArticleSummarizer'):
                converter = CSVToTextConverter(config)
                
                # Test content formatting
                formatted_content = converter._format_csv_content(test_df)
                
                # Verify formatted content contains expected information
                assert "All 4 rows:" in formatted_content
                assert "Monkey D. Luffy" in formatted_content
                assert "Gomu Gomu no Mi" in formatted_content
                assert "3,000,000,000" in formatted_content
                assert "Captain" in formatted_content
                # Verify all rows are included (no truncation)
                assert "Usopp" in formatted_content  # Last row should be included
                
                print("✓ CSV content formatting successful")
                
    except Exception as e:
        print(f"✗ CSV content formatting failed: {e}")
        raise
    finally:
        # Clean up test file
        if os.path.exists(csv_path):
            os.remove(csv_path)
        if os.path.exists("test_data"):
            os.rmdir("test_data")


def test_article_summary_retrieval():
    """Test article summary retrieval from summarizer."""
    print("Testing article summary retrieval...")
    
    try:
        # Create config
        config = RAGConfig()
        config.ENABLE_CSV_TO_TEXT = True
        
        # Mock OpenAI API key
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            # Mock the summarizer with test data
            mock_summarizer = MagicMock()
            mock_summary_chunks = [
                {
                    'content': 'Monkey D. Luffy is the main protagonist of One Piece. He is the captain of the Straw Hat Pirates and possesses the Gomu Gomu no Mi devil fruit.'
                }
            ]
            mock_summarizer.create_summary_chunks.return_value = mock_summary_chunks
            
            with patch('rag_piece.csv_to_text.ArticleSummarizer', return_value=mock_summarizer):
                converter = CSVToTextConverter(config)
                
                # Test summary retrieval
                summary = converter._get_article_summary("Monkey D. Luffy")
                
                # Verify summary was retrieved correctly
                assert "Monkey D. Luffy" in summary
                assert "Straw Hat Pirates" in summary
                assert "Gomu Gomu no Mi" in summary
                
                print("✓ Article summary retrieval successful")
                
    except Exception as e:
        print(f"✗ Article summary retrieval failed: {e}")
        raise


def test_conversion_result_creation():
    """Test conversion result object creation."""
    print("Testing conversion result creation...")
    
    # Create test CSV
    csv_path, test_df = create_test_csv()
    
    try:
        # Create config
        config = RAGConfig()
        config.ENABLE_CSV_TO_TEXT = True
        
        # Mock OpenAI API key
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            # Mock the summarizer
            with patch('rag_piece.csv_to_text.ArticleSummarizer'):
                converter = CSVToTextConverter(config)
                
                # Test result creation
                converted_text = "This is a test converted text about One Piece characters."
                article_summary = "Test article summary about One Piece."
                
                result = converter._create_conversion_result(
                    csv_path, "One Piece Characters", converted_text, test_df, article_summary
                )
                
                # Verify result structure
                assert result['csv_file_path'] == csv_path
                assert result['csv_filename'] == "test_characters.csv"
                assert result['article_name'] == "One Piece Characters"
                assert result['converted_text'] == converted_text
                assert result['conversion_metadata']['conversion_success'] is True
                assert result['conversion_metadata']['model_used'] == "gpt-4o-mini"
                assert result['conversion_metadata']['temperature'] == 0.2
                assert result['conversion_metadata']['original_csv_shape'] == (4, 5)
                assert result['conversion_metadata']['converted_text_length'] == len(converted_text)
                assert result['conversion_metadata']['article_summary_length'] == len(article_summary)
                
                print("✓ Conversion result creation successful")
                
    except Exception as e:
        print(f"✗ Conversion result creation failed: {e}")
        raise
    finally:
        # Clean up test file
        if os.path.exists(csv_path):
            os.remove(csv_path)
        if os.path.exists("test_data"):
            os.rmdir("test_data")


def test_error_result_creation():
    """Test error result object creation."""
    print("Testing error result creation...")
    
    try:
        # Create config
        config = RAGConfig()
        config.ENABLE_CSV_TO_TEXT = True
        
        # Mock OpenAI API key
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            # Mock the summarizer
            with patch('rag_piece.csv_to_text.ArticleSummarizer'):
                converter = CSVToTextConverter(config)
                
                # Test error result creation
                error_message = "Test error message"
                result = converter._create_error_result(error_message, "test.csv")
                
                # Verify error result structure
                assert result['csv_file_path'] == "test.csv"
                assert result['csv_filename'] == "test.csv"
                assert result['article_name'] == "unknown"
                assert "Test error message" in result['converted_text']
                assert result['conversion_metadata']['conversion_success'] is False
                assert result['conversion_metadata']['error_message'] == error_message
                
                print("✓ Error result creation successful")
                
    except Exception as e:
        print(f"✗ Error result creation failed: {e}")
        raise


def test_debug_file_saving():
    """Test debug file saving functionality."""
    print("Testing debug file saving...")
    
    # Create test CSV
    csv_path, test_df = create_test_csv()
    
    try:
        # Create config with debug saving enabled
        config = RAGConfig()
        config.ENABLE_CSV_TO_TEXT = True
        config.SAVE_CSV_TO_TEXT_FILES = True  # Always save debug files during testing
        
        # Mock OpenAI API key
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            # Mock the summarizer
            with patch('rag_piece.csv_to_text.ArticleSummarizer'):
                converter = CSVToTextConverter(config)
                
                # Create a test result
                converted_text = "Test converted text about One Piece characters."
                article_summary = "Test article summary about One Piece."
                
                result = converter._create_conversion_result(
                    csv_path, "One Piece Characters", converted_text, test_df, article_summary
                )
                
                # Test debug file saving
                debug_file_path = converter._save_debug_file(result)
                
                # Verify debug file was created
                assert debug_file_path != ""
                assert os.path.exists(debug_file_path)
                
                # Verify debug file content
                with open(debug_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    assert "CSV to Text Conversion Debug File" in content
                    assert "test_characters.csv" in content
                    assert "One Piece Characters" in content
                    assert "Test converted text about One Piece characters." in content
                
                print("✓ Debug file saving successful")
                
                # Clean up debug file
                if os.path.exists(debug_file_path):
                    os.remove(debug_file_path)
                
    except Exception as e:
        print(f"✗ Debug file saving failed: {e}")
        raise
    finally:
        # Clean up test files
        if os.path.exists(csv_path):
            os.remove(csv_path)
        if os.path.exists("test_data"):
            os.rmdir("test_data")
        
        # Clean up debug directory
        debug_dir = Path("data/debug/csv2text")
        if debug_dir.exists():
            for file in debug_dir.glob("*"):
                file.unlink()
            debug_dir.rmdir()
        if Path("data/debug").exists():
            Path("data/debug").rmdir()
        if Path("data").exists():
            Path("data").rmdir()


def run_all_tests():
    """Run all tests for the CSV to Text converter."""
    print("=" * 60)
    print("Running CSV to Text Converter Tests")
    print("=" * 60)
    
    tests = [
        test_csv_to_text_converter_initialization,
        test_csv_file_reading,
        test_csv_structure_description,
        test_csv_content_formatting,
        test_article_summary_retrieval,
        test_conversion_result_creation,
        test_error_result_creation,
        test_debug_file_saving
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"Test {test.__name__} failed with error: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 All tests passed! CSV to Text converter is working correctly.")
    else:
        print(f"❌ {failed} test(s) failed. Please check the implementation.")
    
    return failed == 0


if __name__ == "__main__":
    # Run all tests
    success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
