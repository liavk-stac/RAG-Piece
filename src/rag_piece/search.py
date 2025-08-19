"""
Search functionality for the RAG system using BM25 and semantic search.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.analysis import StandardAnalyzer, StemmingAnalyzer
from whoosh.index import create_in, open_dir
from whoosh.qparser import MultifieldParser
from whoosh.scoring import BM25F
import faiss
from sentence_transformers import SentenceTransformer

from .config import RAGConfig
from .utils import safe_file_operation


class SearchEngine:
    """Handles BM25 and semantic search functionality"""
    
    def __init__(self, config: RAGConfig, db_path: str = "data/rag_db"):
        self.config = config
        self.db_path = db_path
        self.logger = logging.getLogger("rag_piece.search")
        
        self.whoosh_index = None
        self.faiss_index = None
        self.embedding_model = None
        self.chunk_mapping = {}
        
        self._load_embedding_model()
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Perform two-step search: BM25 + semantic reranking.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of search results with combined scores
        """
        if top_k is None:
            top_k = self.config.FINAL_RESULTS
        
        try:
            self.logger.info(f"Searching for: '{query}' (top_k={top_k})")
            
            # Step 1: BM25 search
            bm25_results = self._bm25_search(query, self.config.BM25_CANDIDATES)
            
            if not bm25_results:
                self.logger.warning("No BM25 results found")
                return []
            
            self.logger.info(f"BM25 found {len(bm25_results)} candidates")
            
            # Step 2: Semantic reranking
            final_results = self._semantic_rerank(query, bm25_results, top_k)
            
            self.logger.info(f"Returning {len(final_results)} final results")
            return final_results
        
        except Exception as e:
            self.logger.error(f"Error during search: {e}", exc_info=True)
            return []
    
    def _bm25_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Perform BM25 search using Whoosh with fallback strategies."""
        if not self.whoosh_index:
            self.logger.error("Whoosh index not loaded")
            return []
        
        with self.whoosh_index.searcher(weighting=BM25F()) as searcher:
            # Create multi-field parser with field boosting
            field_boosts = self._get_field_boosts()
            parser = MultifieldParser(
                ['content', 'article_name', 'sub_article_name', 'section_name', 'sub_section_name', 'keywords'],
                self.whoosh_index.schema,
                fieldboosts=field_boosts
            )
            
            # Try multiple search strategies
            strategies = [
                lambda: self._try_direct_query(parser, searcher, query, limit),
                lambda: self._try_cleaned_query(parser, searcher, query, limit),
                lambda: self._try_or_query(parser, searcher, query, limit)
            ]
            
            for strategy in strategies:
                results = strategy()
                if results:
                    return results
            
            self.logger.warning(f"No results found for query: {query}")
            return []
    
    def _try_direct_query(self, parser, searcher, query: str, limit: int) -> List[Dict[str, Any]]:
        """Try the original query directly."""
        try:
            parsed_query = parser.parse(query)
            results = searcher.search(parsed_query, limit=limit)
            if len(results) > 0:
                return self._convert_whoosh_results(results)
        except Exception as e:
            self.logger.debug(f"Direct query failed: {e}")
        return []
    
    def _try_cleaned_query(self, parser, searcher, query: str, limit: int) -> List[Dict[str, Any]]:
        """Try query with stop words and question words removed."""
        try:
            clean_query = self._clean_query(query)
            if clean_query and clean_query != query:
                parsed_query = parser.parse(clean_query)
                results = searcher.search(parsed_query, limit=limit)
                if len(results) > 0:
                    return self._convert_whoosh_results(results)
        except Exception as e:
            self.logger.debug(f"Cleaned query failed: {e}")
        return []
    
    def _try_or_query(self, parser, searcher, query: str, limit: int) -> List[Dict[str, Any]]:
        """Try OR query with important terms."""
        try:
            important_terms = self._extract_important_terms(query)
            if len(important_terms) > 1:
                or_query = " OR ".join(important_terms)
                parsed_query = parser.parse(or_query)
                results = searcher.search(parsed_query, limit=limit)
                if len(results) > 0:
                    return self._convert_whoosh_results(results)
        except Exception as e:
            self.logger.debug(f"OR query failed: {e}")
        return []
    
    def _semantic_rerank(self, query: str, bm25_results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Rerank BM25 results using semantic similarity."""
        if not self.embedding_model or not bm25_results:
            return bm25_results[:top_k]
        
        try:
            # Get query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Get candidate embeddings
            candidate_texts = [result['content'] for result in bm25_results]
            candidate_embeddings = self.embedding_model.encode(candidate_texts)
            
            # Calculate similarities
            similarities = np.dot(query_embedding, candidate_embeddings.T)[0]
            
            # Combine BM25 and semantic scores
            for i, result in enumerate(bm25_results):
                bm25_score = result.get('bm25_score', 0.0)
                semantic_score = float(similarities[i])
                
                # Normalize scores to [0, 1] range
                normalized_bm25 = self._normalize_score(bm25_score, 0, 20)  # Typical BM25 range
                normalized_semantic = max(0, semantic_score)  # Cosine similarity range [-1, 1]
                
                # Combine scores
                combined_score = (
                    self.config.RERANK_WEIGHT_BM25 * normalized_bm25 +
                    self.config.RERANK_WEIGHT_SEMANTIC * normalized_semantic
                )
                
                result['semantic_score'] = semantic_score
                result['combined_score'] = combined_score
            
            # Sort by combined score and return top_k
            reranked = sorted(bm25_results, key=lambda x: x['combined_score'], reverse=True)
            return reranked[:top_k]
        
        except Exception as e:
            self.logger.error(f"Error during semantic reranking: {e}", exc_info=True)
            return bm25_results[:top_k]
    
    def _get_field_boosts(self) -> Dict[str, float]:
        """Get field boost weights from config."""
        return {
            'content': self.config.CONTENT_BOOST,
            'article_name': self.config.ARTICLE_NAME_BOOST,
            'sub_article_name': self.config.SUB_ARTICLE_NAME_BOOST,
            'section_name': self.config.SECTION_NAME_BOOST,
            'sub_section_name': self.config.SUB_SECTION_BOOST,
            'keywords': self.config.KEYWORDS_BOOST
        }
    
    def _clean_query(self, query: str) -> str:
        """Clean query by removing stop words and question words."""
        clean_query = re.sub(r'\b(what|is|are|how|where|when|who|why|the|a|an|about|tell|me)\b', '', query.lower())
        clean_query = re.sub(r'[^\w\s]', ' ', clean_query)
        return ' '.join(clean_query.split())
    
    def _extract_important_terms(self, query: str) -> List[str]:
        """Extract important terms from query."""
        # Simple extraction - split and filter short words
        terms = query.lower().split()
        important = [term for term in terms if len(term) > 2 and term.isalpha()]
        return important[:5]  # Limit to avoid overly broad queries
    
    def _convert_whoosh_results(self, results) -> List[Dict[str, Any]]:
        """Convert Whoosh results to standardized format."""
        converted = []
        
        for hit in results:
            # Parse stored metadata
            debug_metadata = {}
            if hit.get('debug_metadata'):
                try:
                    debug_metadata = json.loads(hit['debug_metadata'])
                except:
                    pass
            
            # Parse keywords
            keywords = []
            if hit.get('keywords'):
                keywords = hit['keywords'].split()
            
            result = {
                'chunk_id': hit['chunk_id'],
                'content': hit['content'],
                'search_metadata': {
                    'article_name': hit.get('article_name', ''),
                    'sub_article_name': hit.get('sub_article_name', ''),
                    'section_name': hit.get('section_name', ''),
                    'sub_section_name': hit.get('sub_section_name', ''),
                    'keywords': keywords
                },
                'debug_metadata': debug_metadata,
                'bm25_score': hit.score
            }
            converted.append(result)
        
        return converted
    
    def _normalize_score(self, score: float, min_val: float, max_val: float) -> float:
        """Normalize score to [0, 1] range."""
        if max_val <= min_val:
            return 0.0
        return max(0.0, min(1.0, (score - min_val) / (max_val - min_val)))
    
    def _load_embedding_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            self.logger.info(f"Loading embedding model: {self.config.EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        except Exception as e:
            self.logger.error(f"Error loading embedding model: {e}", exc_info=True)
            self.embedding_model = None
    
    def load_indices(self) -> bool:
        """Load existing search indices."""
        try:
            # Load Whoosh index
            whoosh_path = f"{self.db_path}/whoosh_index"
            self.whoosh_index = safe_file_operation(
                lambda: open_dir(whoosh_path),
                f"Failed to load Whoosh index from {whoosh_path}",
                self.logger
            )
            
            if not self.whoosh_index:
                return False
            
            # Load FAISS index
            faiss_path = f"{self.db_path}/faiss_index.bin"
            self.faiss_index = safe_file_operation(
                lambda: faiss.read_index(faiss_path),
                f"Failed to load FAISS index from {faiss_path}",
                self.logger
            )
            
            # Load chunk mapping
            mapping_path = f"{self.db_path}/chunk_mapping.pkl"
            import pickle
            self.chunk_mapping = safe_file_operation(
                lambda: pickle.load(open(mapping_path, 'rb')),
                f"Failed to load chunk mapping from {mapping_path}",
                self.logger
            ) or {}
            
            self.logger.info("Search indices loaded successfully")
            return True
        
        except Exception as e:
            self.logger.error(f"Error loading indices: {e}", exc_info=True)
            return False
    
    def build_whoosh_index(self, chunks: List[Dict[str, Any]]) -> None:
        """Build Whoosh index for BM25 search."""
        try:
            self.logger.info("Building Whoosh index...")
            
            # Create index directory
            whoosh_dir = Path(self.db_path) / "whoosh_index"
            whoosh_dir.mkdir(parents=True, exist_ok=True)
            
            # Create schema and index
            schema = self._create_whoosh_schema()
            self.whoosh_index = create_in(str(whoosh_dir), schema)
            
            # Add documents to index
            self._add_documents_to_index(chunks)
            
            self.logger.info(f"Whoosh index built with {len(chunks)} documents")
        
        except Exception as e:
            self.logger.error(f"Error building Whoosh index: {e}", exc_info=True)
            raise
    
    def build_faiss_index(self, chunks: List[Dict[str, Any]]) -> None:
        """Build FAISS index for semantic search."""
        try:
            self.logger.info("Building FAISS index...")
            
            if not self.embedding_model:
                self.logger.warning("No embedding model available, skipping FAISS index")
                return
            
            # Extract content for embedding
            texts = [chunk['content'] for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            embeddings = np.array(embeddings).astype('float32')
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            if self.config.FAISS_INDEX_TYPE == "IndexFlatIP":
                self.faiss_index = faiss.IndexFlatIP(dimension)
            else:
                self.faiss_index = faiss.IndexFlatL2(dimension)
            
            # Add embeddings to index
            self.faiss_index.add(embeddings)
            
            # Save index and mapping
            self._save_faiss_index(chunks)
            
            self.logger.info(f"FAISS index built with {len(chunks)} embeddings ({dimension}D)")
        
        except Exception as e:
            self.logger.error(f"Error building FAISS index: {e}", exc_info=True)
            raise
    
    def _create_whoosh_schema(self) -> Schema:
        """Create Whoosh schema for indexing."""
        return Schema(
            chunk_id=ID(stored=True, unique=True),
            content=TEXT(analyzer=StemmingAnalyzer(), stored=True),
            article_name=TEXT(analyzer=StandardAnalyzer(), stored=True),
            sub_article_name=TEXT(analyzer=StandardAnalyzer(), stored=True),
            section_name=TEXT(analyzer=StandardAnalyzer(), stored=True),
            sub_section_name=TEXT(analyzer=StandardAnalyzer(), stored=True),
            keywords=TEXT(analyzer=StandardAnalyzer(), stored=True),
            debug_metadata=STORED
        )
    
    def _add_documents_to_index(self, chunks: List[Dict[str, Any]]) -> None:
        """Add documents to Whoosh index."""
        writer = self.whoosh_index.writer()
        
        for chunk in chunks:
            search_meta = chunk['search_metadata']
            debug_meta = chunk.get('debug_metadata', {})
            
            writer.add_document(
                chunk_id=chunk['chunk_id'],
                content=chunk['content'],
                article_name=search_meta['article_name'],
                sub_article_name=search_meta['sub_article_name'] or "",
                section_name=search_meta['section_name'] or "",
                sub_section_name=search_meta['sub_section_name'] or "",
                keywords=' '.join(search_meta['keywords']),
                debug_metadata=json.dumps(debug_meta)
            )
        
        writer.commit()
    
    def _save_faiss_index(self, chunks: List[Dict[str, Any]]) -> None:
        """Save FAISS index and chunk mapping."""
        # Save FAISS index
        faiss_path = Path(self.db_path) / "faiss_index.bin"
        faiss.write_index(self.faiss_index, str(faiss_path))
        
        # Save chunk mapping
        chunk_mapping = {i: chunk for i, chunk in enumerate(chunks)}
        mapping_path = Path(self.db_path) / "chunk_mapping.pkl"
        
        import pickle
        with open(mapping_path, 'wb') as f:
            pickle.dump(chunk_mapping, f)
