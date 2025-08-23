"""
Article summarizer module for the RAG Piece system.
Uses LangChain's refine method with custom prompts to create summaries
that fit within the maximum chunk size for RAG database integration.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from .scraper import OneWikiScraper
from .utils import slugify, count_tokens

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from langchain_community.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
except ImportError as e:
    print(f"Error: Required LangChain packages not installed: {e}")
    print("Please install: pip install langchain langchain-community openai python-dotenv")
    raise


class ArticleSummarizer:
    """Article summarizer using LangChain's refine method with custom prompts."""
    
    def __init__(self, max_chunk_size: int = 400, save_to_files: bool = False):
        self.max_chunk_size = max_chunk_size
        self.save_to_files = save_to_files
        self.logger = logging.getLogger("rag_piece.summarizer")
        
        # Initialize OpenAI LLM
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        
        self.llm = ChatOpenAI(
            api_key=api_key,
            temperature=0.3,  # Lower temperature for more consistent summaries
            max_tokens=max_chunk_size * 2,  # Allow some buffer for the LLM
            model_name="gpt-4o-mini"  # Good balance of quality and cost
        )
        
        # Custom prompts for the refine method
        self.initial_prompt = PromptTemplate(
            input_variables=["context_str", "question"],
            template="""You are an expert One Piece Wiki summarizer. Your task is to create a comprehensive but concise summary of the given article content.

Article Content:
{context_str}

Question: {question}

Instructions:
1. Create a well-structured summary that captures the key information
2. Focus on the most important facts, characters, locations, and events
3. Use clear, organized formatting with bullet points or sections
4. Ensure the summary is comprehensive but concise
5. Do NOT exceed {max_tokens} tokens in your response
6. Maintain factual accuracy and One Piece lore consistency

Summary:"""
        )
        
        self.refine_prompt = PromptTemplate(
            input_variables=["question", "existing_answer", "context_str"],
            template="""You are an expert One Piece Wiki summarizer. You have an existing summary, and now you have additional content to incorporate.

Question: {question}

Existing Summary:
{existing_answer}

Additional Content:
{context_str}

Instructions:
1. Refine and improve the existing summary by incorporating the new content
2. Maintain the existing structure and key points
3. Add new important information from the additional content
4. Remove any redundant or less important details to stay within {max_tokens} tokens
5. Ensure the final summary is comprehensive, well-organized, and concise
6. Maintain factual accuracy and One Piece lore consistency

Refined Summary:"""
        )
        
        # Initialize the refine chain using the new RunnableSequence approach
        # Create the initial chain
        self.initial_chain = self.initial_prompt | self.llm | StrOutputParser()
        
        # Create the refine chain
        self.refine_chain = self.refine_prompt | self.llm | StrOutputParser()
        
        # Text splitter for breaking content into manageable chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size * 2,  # Larger chunks for summarization
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def create_summary_chunks(self, article_name: str) -> List[Dict[str, Any]]:
        """
        Create summary chunks that can be inserted into the RAG database.
        
        Args:
            article_name: Name of the article to summarize
        
        Returns:
            List of chunk objects with proper metadata structure
        """
        try:
            self.logger.info(f"Creating summary chunks for article: {article_name}")
            
            # Initialize the scraper to get content
            scraper = OneWikiScraper(max_images=0)  # No images needed for summarization
            
            # Get article content
            sections, metadata = scraper.scrape_article(article_name)
            
            if not sections:
                self.logger.warning(f"No content found for article: {article_name}")
                return []
            
            summary_chunks = []
            
            # Create main article summary chunk
            main_summary = self._create_summary(sections, f"{article_name} - Main Article")
            
            # Save to file if enabled
            if self.save_to_files:
                self._save_summary_to_file(main_summary, article_name, None, "main_article")
            
            main_chunk = self._create_summary_chunk(
                main_summary, 
                article_name, 
                None,  # No sub-article for main
                "Article Summary",
                "main_article"
            )
            summary_chunks.append(main_chunk)
            
            # Find and create sub-article summary chunks
            sub_articles = self._find_sub_articles(article_name)
            
            for sub_article in sub_articles:
                try:
                    self.logger.info(f"Creating summary chunk for sub-article: {sub_article}")
                    sub_sections, _ = scraper.scrape_article(sub_article)
                    
                    if sub_sections:
                        sub_summary = self._create_summary(sub_sections, sub_article)
                        
                        # Save to file if enabled
                        if self.save_to_files:
                            self._save_summary_to_file(sub_summary, article_name, sub_article, "sub_article")
                        
                        sub_chunk = self._create_summary_chunk(
                            sub_summary,
                            article_name,
                            sub_article,
                            "Article Summary", 
                            "sub_article"
                        )
                        summary_chunks.append(sub_chunk)
                    
                except Exception as e:
                    self.logger.error(f"Error creating summary chunk for sub-article {sub_article}: {e}")
            
            self.logger.info(f"Created {len(summary_chunks)} summary chunks for {article_name}")
            return summary_chunks
            
        except Exception as e:
            self.logger.error(f"Error creating summary chunks for article {article_name}: {e}", exc_info=True)
            return []
    
    def _create_summary(self, sections: List[Dict], article_name: str) -> str:
        """Create a summary from article sections using the refine method."""
        try:
            # Combine all sections into a single text
            full_text = self._combine_sections(sections)
            
            # Split text into manageable chunks for the refine method
            text_chunks = self.text_splitter.split_text(full_text)
            
            # Create the question for summarization
            question = f"Create a comprehensive summary of the {article_name} article, focusing on key information, characters, locations, and events."
            
            # Start with the first chunk
            if not text_chunks:
                return "No content to summarize"
            
            # Create initial summary from first chunk
            initial_summary = self.initial_chain.invoke({
                "context_str": text_chunks[0],
                "question": question,
                "max_tokens": self.max_chunk_size
            })
            
            # Refine with remaining chunks
            current_summary = initial_summary
            for chunk in text_chunks[1:]:
                current_summary = self.refine_chain.invoke({
                    "question": question,
                    "existing_answer": current_summary,
                    "context_str": chunk,
                    "max_tokens": self.max_chunk_size
                })
            
            # Ensure the summary doesn't exceed max chunk size
            summary = self._truncate_summary(current_summary, self.max_chunk_size)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error creating summary for {article_name}: {e}")
            return f"Error creating summary: {str(e)}"
    
    def _combine_sections(self, sections: List[Dict]) -> str:
        """Combine all sections into a single text for summarization."""
        combined_text = ""
        
        for section in sections:
            section_name = section.get('section_name', 'Unknown')
            section_content = section.get('content', '')
            
            # Clean HTML tags and normalize text
            cleaned_content = self._clean_html_content(section_content)
            
            if cleaned_content.strip():
                combined_text += f"\n\n## {section_name}\n{cleaned_content}"
        
        return combined_text.strip()
    
    def _clean_html_content(self, html_content: str) -> str:
        """Clean HTML content for summarization."""
        import re
        
        # Remove HTML tags
        clean_text = re.sub(r'<[^>]+>', '', html_content)
        
        # Remove extra whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        # Remove navigation symbols
        clean_text = re.sub(r'\[v\s*·\s*e\s*·\s*\?\]', '', clean_text)
        
        return clean_text.strip()
    
    def _truncate_summary(self, summary: str, max_tokens: int) -> str:
        """Ensure summary doesn't exceed maximum token count."""
        current_tokens = count_tokens(summary)
        
        if current_tokens <= max_tokens:
            return summary
        
        # Simple truncation strategy - could be improved with more sophisticated approaches
        self.logger.warning(f"Summary exceeds {max_tokens} tokens ({current_tokens}), truncating...")
        
        # Estimate characters per token (rough approximation)
        chars_per_token = len(summary) / current_tokens
        target_chars = int(max_tokens * chars_per_token * 0.9)  # 90% to be safe
        
        truncated = summary[:target_chars]
        
        # Try to end at a complete sentence
        last_period = truncated.rfind('.')
        if last_period > target_chars * 0.8:  # If we can find a period in the last 20%
            truncated = truncated[:last_period + 1]
        
        return truncated
    
    def _create_summary_chunk(self, summary_text: str, article_name: str, 
                             sub_article_name: Optional[str], section_name: str, 
                             summary_type: str) -> Dict[str, Any]:
        """Create a chunk object for a summary that matches the existing chunking system."""
        # Generate chunk ID
        chunk_id = f"{slugify(article_name)}_summary_{slugify(sub_article_name or 'main')}"
        
        # Create chunk object matching the existing structure
        chunk_obj = {
            'chunk_id': chunk_id,
            'content': summary_text,
            'search_metadata': {
                'article_name': article_name,
                'sub_article_name': sub_article_name,
                'section_name': section_name,
                'sub_section_name': None,  # Summaries don't have sub-sections
                'keywords': []  # Will be filled by keyword extractor
            },
            'summary_metadata': {
                'chunk_type': 'summary',
                'summary_type': summary_type,
                'source_sections_count': 0,  # Could be enhanced to track this
                'compression_ratio': 0.0,    # Could be enhanced to calculate this
                'generation_method': 'langchain_refine',
                'model_used': self.llm.model_name,
                'generation_timestamp': datetime.now().isoformat()
            },
            'debug_metadata': {
                'chunk_size': count_tokens(summary_text),
                'target_token_limit': self.max_chunk_size,
                'token_efficiency': (count_tokens(summary_text) / self.max_chunk_size) * 100,
                'processing_timestamp': datetime.now().isoformat()
            }
        }
        
        return chunk_obj
    
    def _find_sub_articles(self, article_name: str) -> List[str]:
        """Find sub-articles for the main article."""
        try:
            # Use the same logic as the scraper
            scraper = OneWikiScraper(max_images=0)
            return scraper._find_sub_articles(article_name)
        except Exception as e:
            self.logger.error(f"Error finding sub-articles: {e}")
            return []
    
    def _save_summary_to_file(self, summary_text: str, article_name: str, 
                             sub_article_name: Optional[str], summary_type: str) -> str:
        """Save a summary to a text file in the summaries folder."""
        try:
            # Create summaries directory if it doesn't exist
            summaries_dir = Path("summaries")
            summaries_dir.mkdir(exist_ok=True)
            
            # Create article subdirectory
            article_dir = summaries_dir / slugify(article_name)
            article_dir.mkdir(exist_ok=True)
            
            # Generate filename
            if sub_article_name:
                filename = f"{slugify(sub_article_name)}_{summary_type}_summary.txt"
            else:
                filename = f"main_{summary_type}_summary.txt"
            
            file_path = article_dir / filename
            
            # Write summary to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# {article_name}")
                if sub_article_name:
                    f.write(f" - {sub_article_name}")
                f.write(f" ({summary_type.replace('_', ' ').title()} Summary)\n\n")
                f.write(summary_text)
            
            self.logger.info(f"Saved summary to file: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Error saving summary to file: {e}")
            return ""
