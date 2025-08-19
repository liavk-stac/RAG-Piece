"""
Keyword extraction functionality using BM25-style scoring.
"""

import re
import logging
from typing import List, Dict, Any
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from .config import RAGConfig
from .utils import count_tokens


class KeywordExtractor:
    """Extracts keywords using BM25-style scoring"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = logging.getLogger("rag_piece.keywords")
        self._ensure_nltk_data()
        self._setup_stopwords()
        self.document_frequencies = {}
        self.total_documents = 0
    
    def extract_keywords(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract keywords for all chunks using BM25-style scoring.
        
        Args:
            chunks: List of chunk objects
        
        Returns:
            Updated chunks with keywords added to metadata
        """
        try:
            self.logger.info(f"Extracting keywords for {len(chunks)} chunks")
            
            if not chunks:
                return chunks
            
            # Build corpus statistics
            self._build_corpus_statistics(chunks)
            
            # Calculate average document length
            total_tokens = sum(count_tokens(chunk['content']) for chunk in chunks)
            avg_doc_length = total_tokens / len(chunks) if chunks else 1
            
            # Extract keywords for each chunk
            for chunk in chunks:
                keywords = self._extract_chunk_keywords(chunk, avg_doc_length)
                chunk['search_metadata']['keywords'] = keywords
            
            self.logger.info("Keyword extraction completed")
            return chunks
        
        except Exception as e:
            self.logger.error(f"Error during keyword extraction: {e}", exc_info=True)
            return chunks
    
    def _build_corpus_statistics(self, chunks: List[Dict[str, Any]]) -> None:
        """Build document frequency statistics for the corpus."""
        self.logger.info("Building corpus statistics for keyword extraction")
        
        self.document_frequencies = Counter()
        self.total_documents = len(chunks)
        
        for chunk in chunks:
            tokens = self._preprocess_text(chunk['content'])
            unique_tokens = set(tokens)
            
            for token in unique_tokens:
                self.document_frequencies[token] += 1
    
    def _extract_chunk_keywords(self, chunk: Dict[str, Any], avg_doc_length: float) -> List[str]:
        """Extract keywords for a single chunk."""
        content = chunk['content']
        tokens = self._preprocess_text(content)
        
        if not tokens:
            return []
        
        # Calculate term frequencies in this chunk
        chunk_tf = Counter(tokens)
        doc_length = len(tokens)
        
        # Calculate BM25 scores for all terms
        term_scores = {}
        for term, tf in chunk_tf.items():
            if self._is_valid_keyword(term):
                score = self._calculate_bm25_score(term, tf, doc_length, avg_doc_length)
                term_scores[term] = score
        
        # Get top keywords
        top_keywords = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
        return [term for term, score in top_keywords[:self.config.KEYWORDS_PER_CHUNK]]
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for keyword extraction."""
        if not text:
            return []
        
        try:
            # Tokenize
            tokens = word_tokenize(text.lower())
            
            # Filter tokens
            filtered_tokens = []
            for token in tokens:
                if self._is_valid_token(token):
                    filtered_tokens.append(token)
            
            return filtered_tokens
        
        except Exception as e:
            self.logger.warning(f"Error preprocessing text: {e}")
            return text.lower().split()
    
    def _is_valid_token(self, token: str) -> bool:
        """Check if token is valid for keyword extraction."""
        # Length check
        if len(token) < self.config.MIN_KEYWORD_LENGTH or len(token) > self.config.MAX_KEYWORD_LENGTH:
            return False
        
        # Alphabetic check (preserve proper nouns)
        if not token.isalpha():
            return False
        
        # Stopwords check
        if self.config.STOPWORDS_ENABLED and token in self.stopwords:
            # Preserve proper nouns even if they're stopwords
            if self.config.PRESERVE_PROPER_NOUNS and token[0].isupper():
                return True
            return False
        
        return True
    
    def _is_valid_keyword(self, term: str) -> bool:
        """Check if term meets keyword criteria."""
        df = self.document_frequencies.get(term, 0)
        df_ratio = df / self.total_documents if self.total_documents > 0 else 0
        
        return (df >= self.config.MIN_KEYWORD_DF and 
                df_ratio <= self.config.MAX_KEYWORD_DF_RATIO)
    
    def _calculate_bm25_score(self, term: str, tf: int, doc_length: int, avg_doc_length: float) -> float:
        """Calculate BM25 score for a term."""
        df = self.document_frequencies.get(term, 0)
        if df == 0:
            return 0.0
        
        # IDF component
        idf = self._calculate_idf(df)
        
        # TF component with BM25 normalization
        tf_component = (tf * (self.config.BM25_K1 + 1)) / (
            tf + self.config.BM25_K1 * (
                1 - self.config.BM25_B + 
                self.config.BM25_B * (doc_length / avg_doc_length)
            )
        )
        
        return idf * tf_component
    
    def _calculate_idf(self, df: int) -> float:
        """Calculate inverse document frequency."""
        import math
        return math.log((self.total_documents - df + 0.5) / (df + 0.5))
    
    def _ensure_nltk_data(self) -> None:
        """Ensure required NLTK data is downloaded."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            self.logger.info("Downloading NLTK punkt tokenizer")
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            self.logger.info("Downloading NLTK stopwords")
            nltk.download('stopwords')
    
    def _setup_stopwords(self) -> None:
        """Setup stopwords for keyword filtering."""
        try:
            self.stopwords = set(stopwords.words('english'))
        except Exception as e:
            self.logger.warning(f"Could not load stopwords: {e}")
            self.stopwords = set()
        
        # Add common web/wiki stopwords
        web_stopwords = {
            'wiki', 'wikipedia', 'page', 'article', 'section', 'edit', 'source',
            'reference', 'citation', 'link', 'url', 'http', 'https', 'www'
        }
        self.stopwords.update(web_stopwords)
