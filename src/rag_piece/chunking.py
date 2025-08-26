"""
Text chunking functionality for the RAG system using recursive approach.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .config import RAGConfig
from .utils import count_tokens


class TextChunker:
    """Handles recursive chunking strategy for optimal semantic boundaries"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = logging.getLogger("rag_piece.chunking")
    
    def chunk_section_content(self, content: str, section_name: str, article_name: str, 
                            sub_article_name: Optional[str] = None, section_index: int = 0) -> List[Dict[str, Any]]:
        """
        Chunk section content using recursive approach for optimal semantic boundaries.
        
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
            
            # Step 1: Recursively chunk content at optimal boundaries
            chunks = self._recursive_chunk_text(content, self.config.MAX_CHUNK_SIZE)
            
            # Step 2: Create overlapping chunks for better context preservation
            overlapping_chunks = self._create_overlapping_chunks(chunks)
            
            # Step 3: Create chunk objects with metadata
            return self._create_chunk_objects(overlapping_chunks, section_name, article_name, 
                                            sub_article_name, section_index)
        
        except Exception as e:
            self.logger.error(f"Error chunking section {section_name}: {e}", exc_info=True)
            return []
    
    def _recursive_chunk_text(self, text: str, max_size: int) -> List[str]:
        """
        Recursively chunk text at the best possible semantic boundaries.
        
        Args:
            text: Text to chunk
            max_size: Maximum chunk size in tokens
            
        Returns:
            List of optimally sized chunks
        """
        # Base case: text is small enough
        if count_tokens(text) <= max_size:
            return [text] if text.strip() else []
        
        # Try different splitting strategies in order of preference
        strategies = [
            (self._split_at_paragraphs, "paragraphs"),
            (self._split_at_sentences, "sentences"),
            (self._split_at_commas, "commas"),
            (self._split_at_words, "words")
        ]
        
        for strategy, strategy_name in strategies:
            parts = strategy(text)
            if len(parts) > 1:  # Successfully split
                if self.config.VERBOSE_LOGGING:
                    self.logger.debug(f"Split using {strategy_name} strategy")
                
                chunks = []
                for part in parts:
                    if part.strip():  # Only process non-empty parts
                        chunks.extend(self._recursive_chunk_text(part, max_size))
                return chunks
        
        # Fallback: force split at word boundary
        return self._force_split_at_words(text, max_size)
    
    def _split_at_paragraphs(self, text: str) -> List[str]:
        """Split text at paragraph boundaries."""
        if not text:
            return []
        
        paragraphs = text.split(self.config.PARAGRAPH_SEPARATOR)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_at_sentences(self, text: str) -> List[str]:
        """Split text at sentence boundaries."""
        if not text:
            return []
        
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
    
    def _split_at_commas(self, text: str) -> List[str]:
        """Split text at comma boundaries for better semantic breaks."""
        if not text:
            return []
        
        # Split at commas, but be smart about it
        parts = text.split(',')
        if len(parts) <= 1:
            return [text]
        
        # Only split if the resulting parts would be reasonable
        min_part_size = self.config.MAX_CHUNK_SIZE // 4  # Don't create tiny parts
        
        if all(count_tokens(part.strip()) >= min_part_size for part in parts if part.strip()):
            return [part.strip() + ',' for part in parts[:-1]] + [parts[-1].strip()]
        
        return [text]
    
    def _split_at_words(self, text: str) -> List[str]:
        """Split text at word boundaries."""
        if not text:
            return []
        
        words = text.split()
        if len(words) <= 1:
            return [text]
        
        # Try to split at word boundaries while maintaining reasonable chunk sizes
        mid_point = len(words) // 2
        left_text = ' '.join(words[:mid_point])
        right_text = ' '.join(words[mid_point:])
        
        # Only split if both parts would be reasonable
        if (count_tokens(left_text) >= self.config.MIN_CHUNK_SIZE and 
            count_tokens(right_text) >= self.config.MIN_CHUNK_SIZE):
            return [left_text, right_text]
        
        return [text]
    
    def _force_split_at_words(self, text: str, max_size: int) -> List[str]:
        """Force split text at word boundaries when other strategies fail."""
        if not text:
            return []
        
        words = text.split()
        if len(words) <= 1:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        for word in words:
            word_with_space = word + " "
            current_tokens = count_tokens(current_chunk)
            word_tokens = count_tokens(word_with_space)
            
            if current_tokens + word_tokens <= max_size:
                current_chunk += word_with_space
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = word_with_space
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _create_overlapping_chunks(self, chunks: List[str]) -> List[str]:
        """
        Create overlapping chunks to improve context preservation and search quality.
        
        Args:
            chunks: List of non-overlapping chunks
            
        Returns:
            List of overlapping chunks
        """
        if not chunks or self.config.CHUNK_OVERLAP == 0:
            return chunks
        
        overlapping_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Add the original chunk
            overlapping_chunks.append(chunk)
            
            # If this isn't the last chunk, create an overlapping version
            if i < len(chunks) - 1:
                # Calculate how much to overlap
                overlap_tokens = min(self.config.CHUNK_OVERLAP, count_tokens(chunk))
                
                if overlap_tokens > 0:
                    # Find the best split point for overlap (prefer sentence boundaries)
                    overlap_text = self._extract_overlap_text(chunk, overlap_tokens)
                    
                    if overlap_text:
                        # Create overlapping chunk with next chunk
                        next_chunk = chunks[i + 1]
                        overlapping_chunk = overlap_text + " " + next_chunk
                        
                        # Only add if it's not too long and not too short
                        chunk_tokens = count_tokens(overlapping_chunk)
                        if (self.config.MIN_CHUNK_SIZE <= chunk_tokens <= self.config.MAX_CHUNK_SIZE):
                            overlapping_chunks.append(overlapping_chunk)
        
        return overlapping_chunks
    
    def _extract_overlap_text(self, chunk: str, target_overlap_tokens: int) -> str:
        """
        Extract the end portion of a chunk to use as overlap.
        Tries to break at sentence boundaries for better coherence.
        """
        if target_overlap_tokens <= 0:
            return ""
        
        # Start from the end and work backwards
        words = chunk.split()
        current_tokens = 0
        overlap_words = []
        
        for word in reversed(words):
            word_tokens = count_tokens(word + " ")
            if current_tokens + word_tokens <= target_overlap_tokens:
                overlap_words.insert(0, word)
                current_tokens += word_tokens
            else:
                break
        
        overlap_text = " ".join(overlap_words)
        
        # Try to find a better break point at sentence boundaries
        sentences = self._split_into_sentences(overlap_text)
        if len(sentences) > 1:
            # Remove incomplete sentences from the beginning
            while sentences and count_tokens(sentences[0]) < target_overlap_tokens * 0.3:
                sentences.pop(0)
            if sentences:
                overlap_text = " ".join(sentences)
        
        return overlap_text
    
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
