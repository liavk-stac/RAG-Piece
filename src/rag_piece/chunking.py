"""
Text chunking functionality for the RAG system.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .config import RAGConfig
from .utils import count_tokens


class TextChunker:
    """Handles hybrid chunking strategy with paragraph-based splitting"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = logging.getLogger("rag_piece.chunking")
    
    def chunk_section_content(self, content: str, section_name: str, article_name: str, 
                            sub_article_name: Optional[str] = None, section_index: int = 0) -> List[Dict[str, Any]]:
        """
        Chunk section content using hybrid approach.
        
        Args:
            content: Raw HTML content to chunk
            section_name: Name of the section
            article_name: Name of the article
            sub_article_name: Optional sub-article name
            section_index: Index of the section
        
        Returns:
            List of chunk objects with metadata
        """
        try:
            self._log_chunking_start(content, section_name)
            
            # Step 1: Split into paragraphs
            paragraphs = self._split_into_paragraphs(content)
            if not paragraphs:
                return []
            
            # Step 2: Merge short paragraphs
            merged_chunks = self._merge_short_chunks(paragraphs)
            
            # Step 3: Split long chunks
            final_chunks = self._split_long_chunks(merged_chunks)
            
            # Step 4: Create chunk objects with metadata
            return self._create_chunk_objects(final_chunks, section_name, article_name, 
                                            sub_article_name, section_index)
        
        except Exception as e:
            self.logger.error(f"Error chunking section {section_name}: {e}", exc_info=True)
            return []
    
    def _split_into_paragraphs(self, content: str) -> List[str]:
        """Split content into paragraphs."""
        if not content:
            return []
        
        paragraphs = content.split(self.config.PARAGRAPH_SEPARATOR)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _merge_short_chunks(self, paragraphs: List[str]) -> List[str]:
        """Merge paragraphs that are too short."""
        if not paragraphs:
            return []
        
        merged = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph_tokens = count_tokens(paragraph)
            current_tokens = count_tokens(current_chunk)
            
            if (current_tokens + paragraph_tokens <= self.config.TARGET_CHUNK_SIZE and 
                current_tokens > 0):
                # Merge with current chunk
                current_chunk += self.config.PARAGRAPH_SEPARATOR + paragraph
            else:
                # Start new chunk
                if current_chunk:
                    merged.append(current_chunk)
                current_chunk = paragraph
        
        # Add the last chunk
        if current_chunk:
            merged.append(current_chunk)
        
        return merged
    
    def _split_long_chunks(self, chunks: List[str]) -> List[str]:
        """Split chunks that are too long."""
        final_chunks = []
        
        for chunk in chunks:
            if count_tokens(chunk) <= self.config.MAX_CHUNK_SIZE:
                final_chunks.append(chunk)
            else:
                final_chunks.extend(self._split_long_chunk(chunk))
        
        return final_chunks
    
    def _split_long_chunk(self, chunk: str) -> List[str]:
        """Split a single long chunk into smaller pieces."""
        sentences = self._split_into_sentences(chunk)
        if len(sentences) <= 1:
            # Can't split further, return as is
            return [chunk]
        
        result = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence_tokens = count_tokens(sentence)
            current_tokens = count_tokens(current_chunk)
            
            if (current_tokens + sentence_tokens <= self.config.MAX_CHUNK_SIZE and 
                current_tokens > 0):
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    result.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            result.append(current_chunk.strip())
        
        return result
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using configured separators."""
        sentences = [text]
        
        for separator in self.config.SENTENCE_SEPARATORS:
            new_sentences = []
            for sentence in sentences:
                parts = sentence.split(separator)
                for i, part in enumerate(parts):
                    if i < len(parts) - 1:
                        new_sentences.append(part + separator.strip())
                    else:
                        new_sentences.append(part)
            sentences = [s.strip() for s in new_sentences if s.strip()]
        
        return sentences
    
    def _create_chunk_objects(self, final_chunks: List[str], section_name: str, 
                            article_name: str, sub_article_name: Optional[str], 
                            section_index: int) -> List[Dict[str, Any]]:
        """Create chunk objects with metadata."""
        chunk_objects = []
        
        for i, chunk_text in enumerate(final_chunks):
            if count_tokens(chunk_text) < self.config.MIN_CHUNK_SIZE:
                continue  # Skip chunks that are still too small
            
            # Extract sub-section names from this chunk
            sub_section_names = self._extract_h3_titles(chunk_text)
            sub_section_name = '; '.join(sub_section_names) if sub_section_names else None
            
            # Create chunk ID
            chunk_id = self._generate_chunk_id(article_name, section_name, i + 1)
            
            chunk_obj = {
                'chunk_id': chunk_id,
                'content': chunk_text,
                'search_metadata': {
                    'article_name': article_name,
                    'sub_article_name': sub_article_name,
                    'section_name': section_name,
                    'sub_section_name': sub_section_name,
                    'keywords': []  # Will be filled by keyword extractor
                },
                'debug_metadata': self._create_debug_metadata(
                    section_index, i + 1, chunk_text, sub_section_names
                ) if self.config.ENABLE_DEBUG_METADATA else {}
            }
            
            chunk_objects.append(chunk_obj)
        
        return chunk_objects
    
    def _extract_h3_titles(self, content: str) -> List[str]:
        """Extract <h3> titles from content."""
        h3_pattern = r'<h3[^>]*>(.*?)</h3>'
        matches = re.findall(h3_pattern, content, re.IGNORECASE | re.DOTALL)
        
        # Clean up the titles
        titles = []
        for match in matches:
            clean_title = re.sub(r'<[^>]+>', '', match).strip()
            if clean_title:
                titles.append(clean_title)
        
        return titles
    
    def _generate_chunk_id(self, article_name: str, section_name: str, chunk_index: int) -> str:
        """Generate unique chunk ID."""
        from .utils import slugify
        return f"{slugify(article_name)}_{slugify(section_name)}_{chunk_index:03d}"
    
    def _create_debug_metadata(self, section_index: int, chunk_index: int, 
                             chunk_text: str, sub_section_names: List[str]) -> Dict[str, Any]:
        """Create debug metadata for chunk."""
        return {
            'section_index': section_index,
            'chunk_index': chunk_index,
            'chunk_size': count_tokens(chunk_text),
            'has_h3_tags': len(sub_section_names) > 0,
            'processing_timestamp': datetime.now().isoformat()
        }
    
    def _log_chunking_start(self, content: str, section_name: str) -> None:
        """Log the start of chunking process."""
        if self.config.VERBOSE_LOGGING:
            content_length = len(content)
            token_count = count_tokens(content)
            self.logger.info(f"Chunking section: {section_name} ({content_length} chars, {token_count} tokens)")
