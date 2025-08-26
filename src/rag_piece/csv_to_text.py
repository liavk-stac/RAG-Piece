"""
CSV to Text converter module for the RAG Piece system.
Uses OpenAI GPT-4o-mini to convert CSV files to structured text while maintaining data relationships.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

try:
    from langchain_community.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except ImportError as e:
    print(f"Error: Required LangChain packages not installed: {e}")
    print("Please install: pip install langchain langchain-community openai python-dotenv")
    raise

from .config import RAGConfig
from .summarizer import ArticleSummarizer
from .utils import slugify, count_tokens


class CSVToTextConverter:
    """Converts CSV files to structured text using OpenAI GPT-4o-mini."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = logging.getLogger("rag_piece.csv_to_text")
        
        # Validate configuration
        if not config.ENABLE_CSV_TO_TEXT:
            raise ValueError("CSV to text conversion is disabled in config")
        
        # Initialize OpenAI LLM
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        
        self.llm = ChatOpenAI(
            api_key=api_key,
            temperature=config.CSV_TO_TEXT_TEMPERATURE,
            model_name=config.CSV_TO_TEXT_MODEL
        )
        
        # Initialize summarizer to get article summaries
        self.summarizer = ArticleSummarizer(
            max_chunk_size=config.MAX_CHUNK_SIZE,
            save_to_files=config.SAVE_SUMMARIES_TO_FILES,
            max_input_tokens=config.MAX_INPUT_TEXT_TOKENS
        )
        
        # Custom prompt for CSV to text conversion
        self.conversion_prompt = PromptTemplate(
            input_variables=["csv_filename", "article_summary", "csv_content", "csv_structure"],
            template="""You are an expert data analyst specializing in One Piece Wiki content. Your task is to convert CSV data into well-structured, readable text while maintaining all data relationships and context.

CSV Filename: {csv_filename}
Article Summary: {article_summary}

CSV Structure:
{csv_structure}

CSV Content:
{csv_content}

Instructions:
1. Convert the CSV data into structured, narrative text
2. Maintain ALL data relationships between columns and rows
3. Organize information logically and coherently
4. Use clear, descriptive language that flows naturally
5. Preserve the context and meaning from the original data
6. Structure the output with appropriate sections and formatting
7. Ensure the text is comprehensive and includes all important data points
8. Make the text suitable for vector embedding and search

Output Format:
- Use clear headings and subheadings
- Group related information together
- Use bullet points or numbered lists where appropriate
- Maintain chronological or logical order
- Include all relevant data relationships

Converted Text:"""
        )
        
        # Initialize the conversion chain
        self.conversion_chain = self.conversion_prompt | self.llm | StrOutputParser()
        
        # Create debug directory if saving is enabled
        if config.SAVE_CSV_TO_TEXT_FILES:
            self.debug_dir = Path("data/debug/csv2text")
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Set save_to_files flag
        self.save_to_files = config.SAVE_CSV_TO_TEXT_FILES
    
    def convert_csv_to_text(self, csv_file_path: str, article_name: str, 
                          existing_summary: str = None) -> Dict[str, Any]:
        """
        Convert a CSV file to structured text using the LLM.
        
        Args:
            csv_file_path: Path to the CSV file
            article_name: Name of the article the CSV is from
            existing_summary: Optional pre-existing article summary to avoid re-running summarizer
        
        Returns:
            Dictionary containing the converted text and metadata
        """
        try:
            self.logger.info(f"Converting CSV to text: {csv_file_path}")
            
            # Read CSV file
            csv_data = self._read_csv_file(csv_file_path)
            if csv_data is None:
                return self._create_error_result("Failed to read CSV file")
            
            # Get article summary (use existing if provided, otherwise create new)
            if existing_summary:
                article_summary = existing_summary
                self.logger.info(f"Using existing summary for {article_name}")
            else:
                article_summary = self._get_article_summary(article_name)
                self.logger.info(f"Created new summary for {article_name}")
            
            # Prepare CSV content for the prompt
            csv_filename = Path(csv_file_path).name
            csv_structure = self._describe_csv_structure(csv_data)
            csv_content = self._format_csv_content(csv_data)
            
            # Convert using LLM
            converted_text = self.conversion_chain.invoke({
                "csv_filename": csv_filename,
                "article_summary": article_summary,
                "csv_content": csv_content,
                "csv_structure": csv_structure
            })
            
            # Create result object
            result = self._create_conversion_result(
                csv_file_path, article_name, converted_text, csv_data, article_summary
            )
            
            # Save debug file if enabled
            if self.config.SAVE_CSV_TO_TEXT_FILES:
                self._save_debug_file(result)
            
            self.logger.info(f"Successfully converted CSV to text: {csv_file_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error converting CSV to text {csv_file_path}: {e}", exc_info=True)
            return self._create_error_result(str(e))
    
    def convert_multiple_csvs(self, csv_files: List[str], article_name: str, 
                            existing_summary: str = None) -> List[Dict[str, Any]]:
        """
        Convert multiple CSV files to text.
        
        Args:
            csv_files: List of CSV file paths
            article_name: Name of the article the CSVs are from
            existing_summary: Optional pre-existing article summary to avoid re-running summarizer
        
        Returns:
            List of conversion results
        """
        results = []
        
        for csv_file in csv_files:
            try:
                result = self.convert_csv_to_text(csv_file, article_name, existing_summary)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing CSV file {csv_file}: {e}")
                results.append(self._create_error_result(str(e), csv_file))
        
        return results
    
    def convert_dataframe_to_text(self, df: pd.DataFrame, article_name: str, 
                                 existing_summary: str = "", table_name: str = "Unknown") -> Dict[str, Any]:
        """
        Convert a pandas DataFrame directly to text using LLM processing.
        This method does not require CSV files and works with in-memory DataFrames.
        
        Args:
            df: pandas DataFrame to convert
            article_name: Name of the article the table belongs to
            existing_summary: Optional existing summary for context
            table_name: Name/description of the table
            
        Returns:
            Dictionary with conversion results
        """
        try:
            self.logger.info(f"Converting DataFrame to text: {table_name} ({len(df)} rows, {len(df.columns)} columns)")
            
            # Convert DataFrame to a readable format
            table_text = self._dataframe_to_text(df, table_name)
            
            # Create the prompt for conversion
            prompt = self._create_dataframe_conversion_prompt(table_text, article_name, existing_summary, table_name)
            
            # Convert using LLM
            try:
                converted_text = self.llm.invoke(prompt)
                self.logger.info(f"Successfully converted DataFrame: {table_name}")
                
                # Save to file if enabled
                if self.save_to_files:
                    self._save_converted_text(converted_text, article_name, table_name, "dataframe")
                
                return {
                    'success': True,
                    'converted_text': converted_text,
                    'table_name': table_name,
                    'rows_processed': len(df),
                    'columns_processed': len(df.columns),
                    'conversion_method': 'dataframe_direct'
                }
                
            except Exception as e:
                self.logger.error(f"LLM conversion failed for DataFrame {table_name}: {e}")
                return {
                    'success': False,
                    'error': f"LLM conversion failed: {str(e)}",
                    'table_name': table_name
                }
                
        except Exception as e:
            self.logger.error(f"Error converting DataFrame to text: {e}")
            return {
                'success': False,
                'error': f"DataFrame conversion error: {str(e)}",
                'table_name': table_name
            }
    
    def _dataframe_to_text(self, df: pd.DataFrame, table_name: str) -> str:
        """Convert a DataFrame to a readable text format."""
        try:
            # Get basic table info
            rows, cols = df.shape
            
            # Create header section
            text = f"Table: {table_name}\n"
            text += f"Dimensions: {rows} rows × {cols} columns\n\n"
            
            # Add column names
            text += "Columns:\n"
            for i, col in enumerate(df.columns, 1):
                text += f"  {i}. {col}\n"
            text += "\n"
            
            # Add sample data (first 5 rows)
            text += "Sample Data:\n"
            sample_rows = min(5, rows)
            for i in range(sample_rows):
                row_data = df.iloc[i]
                text += f"Row {i+1}: "
                row_text = []
                for col, val in row_data.items():
                    if pd.notna(val) and str(val).strip():
                        row_text.append(f"{col}: {val}")
                text += "; ".join(row_text) + "\n"
            
            # Add summary statistics if numeric columns exist
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                text += "\nNumeric Column Statistics:\n"
                for col in numeric_cols:
                    if df[col].notna().any():
                        text += f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}\n"
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error formatting DataFrame: {e}")
            return f"Error formatting table {table_name}: {str(e)}"
    
    def _read_csv_file(self, csv_file_path: str) -> Optional[pd.DataFrame]:
        """Read and validate CSV file."""
        try:
            if not os.path.exists(csv_file_path):
                self.logger.error(f"CSV file not found: {csv_file_path}")
                return None
            
            # Read CSV with pandas
            df = pd.read_csv(csv_file_path, encoding='utf-8')
            
            if df.empty:
                self.logger.warning(f"CSV file is empty: {csv_file_path}")
                return None
            
            self.logger.debug(f"Read CSV file: {csv_file_path} - Shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading CSV file {csv_file_path}: {e}")
            return None
    
    def _get_article_summary(self, article_name: str) -> str:
        """Get article summary from the summarizer."""
        try:
            # Create summary chunks for the article
            summary_chunks = self.summarizer.create_summary_chunks(article_name)
            
            if not summary_chunks:
                return f"No summary available for article: {article_name}"
            
            # Get the main article summary (first chunk)
            main_summary = summary_chunks[0].get('content', '')
            
            if not main_summary:
                return f"No summary content available for article: {article_name}"
            
            # Truncate if too long for the prompt
            max_summary_length = 1000  # characters
            if len(main_summary) > max_summary_length:
                main_summary = main_summary[:max_summary_length] + "..."
            
            return main_summary
            
        except Exception as e:
            self.logger.warning(f"Error getting article summary for {article_name}: {e}")
            return f"Article summary unavailable: {article_name}"
    
    def _describe_csv_structure(self, df: pd.DataFrame) -> str:
        """Create a description of the CSV structure for the prompt."""
        try:
            structure_info = []
            structure_info.append(f"Total rows: {len(df)}")
            structure_info.append(f"Total columns: {len(df.columns)}")
            structure_info.append("\nColumns:")
            
            for i, col in enumerate(df.columns):
                # Get sample values for the column
                sample_values = df[col].dropna().head(3).tolist()
                sample_str = ", ".join([str(v)[:50] for v in sample_values if str(v).strip()])
                
                # Get data type info
                dtype = str(df[col].dtype)
                non_null_count = df[col].count()
                
                structure_info.append(f"  {i+1}. {col}")
                structure_info.append(f"     - Type: {dtype}")
                structure_info.append(f"     - Non-null values: {non_null_count}")
                if sample_str:
                    structure_info.append(f"     - Sample values: {sample_str}")
                structure_info.append("")
            
            return "\n".join(structure_info)
            
        except Exception as e:
            self.logger.error(f"Error describing CSV structure: {e}")
            return f"Error describing structure: {str(e)}"
    
    def _format_csv_content(self, df: pd.DataFrame) -> str:
        """Format CSV content for the prompt."""
        try:
            # Include all rows without limitations
            content_info = f"All {len(df)} rows:\n\n"
            
            # Convert to string representation
            csv_string = df.to_string(index=False, max_cols=None, max_colwidth=50)
            
            return content_info + csv_string
            
        except Exception as e:
            self.logger.error(f"Error formatting CSV content: {e}")
            return f"Error formatting content: {str(e)}"
    
    def _create_conversion_result(self, csv_file_path: str, article_name: str, 
                                converted_text: str, csv_data: pd.DataFrame, 
                                article_summary: str) -> Dict[str, Any]:
        """Create a result object for the conversion."""
        csv_filename = Path(csv_file_path).name
        
        result = {
            'success': True,  # Add missing success field for proper success detection
            'csv_file_path': csv_file_path,
            'csv_filename': csv_filename,
            'article_name': article_name,
            'converted_text': converted_text,
            'conversion_metadata': {
                'conversion_timestamp': datetime.now().isoformat(),
                'model_used': self.llm.model_name,
                'temperature': self.config.CSV_TO_TEXT_TEMPERATURE,
                'original_csv_shape': csv_data.shape if csv_data is not None else None,
                'converted_text_length': len(converted_text),
                'converted_text_tokens': count_tokens(converted_text),
                'article_summary_length': len(article_summary),
                'conversion_success': True
            },
            'debug_metadata': {
                'processing_timestamp': datetime.now().isoformat(),
                'csv_file_size': os.path.getsize(csv_file_path) if os.path.exists(csv_file_path) else 0
            }
        }
        
        return result
    
    def _create_error_result(self, error_message: str, csv_file_path: str = "unknown") -> Dict[str, Any]:
        """Create an error result object."""
        return {
            'success': False,  # Add missing success field for proper error detection
            'error': error_message,  # Add error field for proper error reporting
            'csv_file_path': csv_file_path,
            'csv_filename': Path(csv_file_path).name if csv_file_path != "unknown" else "unknown",
            'article_name': "unknown",
            'converted_text': f"Error during conversion: {error_message}",
            'conversion_metadata': {
                'conversion_timestamp': datetime.now().isoformat(),
                'model_used': self.llm.model_name if hasattr(self, 'llm') else "unknown",
                'temperature': self.config.CSV_TO_TEXT_TEMPERATURE if hasattr(self, 'config') else 0.0,
                'original_csv_shape': None,
                'converted_text_length': 0,
                'converted_text_tokens': 0,
                'article_summary_length': 0,
                'conversion_success': False,
                'error_message': error_message
            },
            'debug_metadata': {
                'processing_timestamp': datetime.now().isoformat(),
                'csv_file_size': 0
            }
        }
    
    def _save_debug_file(self, result: Dict[str, Any]) -> str:
        """Save the conversion result as a debug text file."""
        try:
            csv_filename = result['csv_filename']
            article_name = result['article_name']
            
            # Create filename for debug file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_filename = f"{slugify(article_name)}_{slugify(csv_filename.replace('.csv', ''))}_{timestamp}.txt"
            debug_file_path = self.debug_dir / debug_filename
            
            # Write debug file
            with open(debug_file_path, 'w', encoding='utf-8') as f:
                f.write(f"CSV to Text Conversion Debug File\n")
                f.write(f"================================\n\n")
                f.write(f"CSV File: {result['csv_file_path']}\n")
                f.write(f"Article: {result['article_name']}\n")
                f.write(f"Conversion Timestamp: {result['conversion_metadata']['conversion_timestamp']}\n")
                f.write(f"Model Used: {result['conversion_metadata']['model_used']}\n")
                f.write(f"Temperature: {result['conversion_metadata']['temperature']}\n")
                f.write(f"Original CSV Shape: {result['conversion_metadata']['original_csv_shape']}\n")
                f.write(f"Converted Text Length: {result['conversion_metadata']['converted_text_length']} characters\n")
                f.write(f"Converted Text Tokens: {result['conversion_metadata']['converted_text_tokens']}\n")
                f.write(f"Article Summary Length: {result['conversion_metadata']['article_summary_length']} characters\n")
                f.write(f"Conversion Success: {result['conversion_metadata']['conversion_success']}\n\n")
                
                f.write(f"Converted Text:\n")
                f.write(f"==============\n\n")
                f.write(result['converted_text'])
            
            self.logger.info(f"Saved debug file: {debug_file_path}")
            return str(debug_file_path)
            
        except Exception as e:
            self.logger.error(f"Error saving debug file: {e}")
            return ""
    
    def _create_dataframe_conversion_prompt(self, table_text: str, article_name: str, 
                                          existing_summary: str, table_name: str) -> str:
        """Create a prompt for converting DataFrame to text."""
        prompt = f"""You are an expert data analyst specializing in One Piece Wiki content. Your task is to convert table data into well-structured, readable text while maintaining all data relationships and context.

Article: {article_name}
Table: {table_name}
Article Summary: {existing_summary}

Table Data:
{table_text}

Instructions:
1. Convert the table data into structured, narrative text
2. Maintain ALL data relationships between columns and rows
3. Organize information logically and coherently
4. Use clear, descriptive language that flows naturally
5. Preserve the context and meaning from the original data
6. Structure the output with appropriate sections and formatting
7. Ensure the text is comprehensive and includes all important data points
8. Make the text suitable for vector embedding and search

Output Format:
- Use clear headings and subheadings
- Group related information together
- Use bullet points or numbered lists where appropriate
- Maintain chronological or logical order
- Include all relevant data relationships

Converted Text:"""
        return prompt
    
    def _save_converted_text(self, converted_text: str, article_name: str, 
                            table_name: str, conversion_type: str) -> str:
        """Save converted text to a debug file."""
        try:
            # Create debug directory if it doesn't exist
            debug_dir = Path("data/debug/csv2text")
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Create article subdirectory
            article_dir = debug_dir / slugify(article_name)
            article_dir.mkdir(exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{slugify(table_name)}_{conversion_type}_{timestamp}.txt"
            file_path = article_dir / filename
            
            # Write converted text to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# {article_name} - {table_name}\n")
                f.write(f"Conversion Type: {conversion_type}\n")
                f.write(f"Timestamp: {timestamp}\n\n")
                f.write(converted_text)
            
            self.logger.info(f"Saved converted text to file: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Error saving converted text to file: {e}")
            return ""
