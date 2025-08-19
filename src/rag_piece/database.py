"""
Main RAG database class that coordinates all components.
"""

import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.analysis import StandardAnalyzer, StemmingAnalyzer
from whoosh.index import create_in

from .config import RAGConfig
from .chunking import TextChunker
from .keywords import KeywordExtractor
from .search import SearchEngine
from .utils import setup_logging, safe_file_operation


class RAGDatabase:
    """Main RAG database class that handles indexing and search"""
    
    def __init__(self, config: RAGConfig, db_path: str = "data/rag_db"):
        # Validate configuration
        config.validate()
        
        self.config = config
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = setup_logging(config.LOG_LEVEL)
        self.logger.info(f"Initializing RAG Database at {self.db_path}")
        
        # Initialize components
        self.chunker = TextChunker(config)
        self.keyword_extractor = KeywordExtractor(config)
        self.search_engine = SearchEngine(config, str(self.db_path))
    
    def process_sections_directly(self, sections: List[Dict[str, str]], article_name: str, 
                                sub_article_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process scraped sections directly into chunks without file I/O.
        
        Args:
            sections: List of section dictionaries with content
            article_name: Name of the article
            sub_article_name: Optional sub-article name
        
        Returns:
            List of processed chunks
        """
        try:
            self.logger.info(f"Processing {len(sections)} sections for article: {article_name}")
            
            all_chunks = []
            
            for i, section in enumerate(sections):
                section_name = self._extract_section_name(section)
                content = section.get('content', '')
                
                if not content.strip():
                    continue
                
                chunks = self.chunker.chunk_section_content(
                    content, section_name, article_name, sub_article_name, i
                )
                all_chunks.extend(chunks)
            
            self.logger.info(f"Created {len(all_chunks)} chunks from {len(sections)} sections")
            return all_chunks
        
        except Exception as e:
            self.logger.error(f"Error processing sections: {e}", exc_info=True)
            return []
    
    def build_indices_from_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Build search indices from processed chunks.
        
        Args:
            chunks: List of chunk objects
        
        Returns:
            Number of chunks indexed
        """
        try:
            if not chunks:
                self.logger.warning("No chunks provided for indexing")
                return 0
            
            self.logger.info(f"Building indices for {len(chunks)} chunks")
            
            # Extract keywords
            chunks = self.keyword_extractor.extract_keywords(chunks)
            
            # Build search indices
            self.search_engine.build_whoosh_index(chunks)
            self.search_engine.build_faiss_index(chunks)
            
            # Save metadata
            self._save_database_metadata(chunks)
            
            self.logger.info(f"Successfully built indices for {len(chunks)} chunks")
            return len(chunks)
        
        except Exception as e:
            self.logger.error(f"Error building indices: {e}", exc_info=True)
            return 0
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search the database using two-step retrieval.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of search results
        """
        return self.search_engine.search(query, top_k)
    
    def load_indices(self) -> bool:
        """Load existing search indices."""
        return self.search_engine.load_indices()
    
    def _extract_section_name(self, section: Dict[str, str]) -> str:
        """Extract clean section name from section data."""
        # Try different possible keys for section name
        for key in ['combined_title', 'title', 'section_name', 'name']:
            if key in section and section[key]:
                title = section[key].strip()
                if title:
                    return self._clean_section_title(title)
        
        # Fallback to a default name
        return "Unknown Section"
    
    def _clean_section_title(self, title: str) -> str:
        """Clean section title by removing file extensions and numbers."""
        import re
        
        # Remove file extensions
        title = re.sub(r'\.(txt|html?)$', '', title, flags=re.IGNORECASE)
        
        # Remove leading numbers and underscores (e.g., "01_General_Information" -> "General Information")
        title = re.sub(r'^\d+_?', '', title)
        
        # Replace underscores with spaces
        title = title.replace('_', ' ')
        
        # Clean up extra spaces
        title = ' '.join(title.split())
        
        return title
    

    
    def _save_database_metadata(self, chunks: List[Dict[str, Any]]) -> None:
        """Save database metadata and statistics."""
        try:
            metadata = {
                'creation_timestamp': str(Path().resolve()),
                'total_chunks': len(chunks),
                'config': {
                    'embedding_model': self.config.EMBEDDING_MODEL,
                    'chunk_size_range': f"{self.config.MIN_CHUNK_SIZE}-{self.config.MAX_CHUNK_SIZE}",
                    'keywords_per_chunk': self.config.KEYWORDS_PER_CHUNK,
                    'bm25_candidates': self.config.BM25_CANDIDATES,
                    'final_results': self.config.FINAL_RESULTS
                },
                'statistics': self._calculate_statistics(chunks)
            }
            
            metadata_path = self.db_path / "database_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Database metadata saved to {metadata_path}")
        
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}", exc_info=True)
    
    def _calculate_statistics(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate database statistics."""
        from .utils import count_tokens
        
        if not chunks:
            return {}
        
        # Calculate chunk size statistics
        chunk_sizes = [count_tokens(chunk['content']) for chunk in chunks]
        
        # Count articles and sections
        articles = set()
        sections = set()
        
        for chunk in chunks:
            meta = chunk['search_metadata']
            if meta.get('article_name'):
                articles.add(meta['article_name'])
            if meta.get('section_name'):
                sections.add(meta['section_name'])
        
        return {
            'total_articles': len(articles),
            'total_sections': len(sections),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_tokens': sum(chunk_sizes)
        }
