#!/usr/bin/env python3
"""
RAG Database Creator - Two-Step Retrieval System
Creates a hybrid search database using BM25 (Whoosh) and semantic search (FAISS)
"""

import re
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter
from datetime import datetime

# Core libraries
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.analysis import StandardAnalyzer, StemmingAnalyzer
from whoosh.index import create_in, open_dir
from whoosh.qparser import MultifieldParser
from whoosh.scoring import BM25F

import faiss
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class RAGConfig:
    """Global configuration for RAG system - all parameters in one place for easy tuning"""
    
    # === CHUNKING PARAMETERS ===
    MIN_CHUNK_SIZE = 100            # tokens - merge if below this
    MAX_CHUNK_SIZE = 400            # tokens - split if above this
    OVERLAP_SIZE = 50               # tokens - overlap when splitting
    TARGET_CHUNK_SIZE = 250         # tokens - ideal chunk size
    PARAGRAPH_SEPARATOR = "\n\n"    # how to split into paragraphs
    SENTENCE_SEPARATORS = [". ", "! ", "? ", ".\n", "!\n", "?\n"]
    
    # === BM25 PARAMETERS ===
    BM25_K1 = 1.2                  # term frequency saturation
    BM25_B = 0.75                  # length normalization
    BM25_CANDIDATES = 100           # candidates from first step
    
    # === KEYWORD EXTRACTION ===
    KEYWORDS_PER_CHUNK = 10         # top keywords to extract per chunk
    MIN_KEYWORD_LENGTH = 3          # minimum keyword character length
    MAX_KEYWORD_LENGTH = 30         # maximum keyword character length
    MIN_KEYWORD_DF = 1              # must appear in at least N chunks
    MAX_KEYWORD_DF_RATIO = 0.8      # must appear in <80% of chunks
    STOPWORDS_ENABLED = True        # remove common stopwords
    PRESERVE_PROPER_NOUNS = True    # keep character/location names
    
    # === FIELD BOOSTING ===
    CONTENT_BOOST = 1.0             # base content weight
    ARTICLE_NAME_BOOST = 3.0        # article name mentions
    SUB_ARTICLE_NAME_BOOST = 2.8    # sub-article name relevance
    SECTION_NAME_BOOST = 2.5        # section title relevance
    SUB_SECTION_BOOST = 2.0         # sub-section title relevance
    KEYWORDS_BOOST = 1.8            # BM25-extracted keywords
    
    # === SEMANTIC SEARCH ===
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence transformer model
    FAISS_INDEX_TYPE = "IndexFlatIP"       # FAISS index type
    SIMILARITY_THRESHOLD = 0.3             # minimum similarity score
    
    # === RESULT FUSION ===
    FINAL_RESULTS = 10              # final results after reranking
    RERANK_WEIGHT_BM25 = 0.4        # BM25 weight in final score
    RERANK_WEIGHT_SEMANTIC = 0.6    # semantic weight in final score
    
    # === DEBUG OPTIONS ===
    ENABLE_DEBUG_METADATA = True    # include debug info in chunks
    VERBOSE_LOGGING = True          # detailed processing logs
    SAVE_INTERMEDIATE_RESULTS = True # save processing artifacts


class TextChunker:
    """Handles hybrid chunking strategy with paragraph-based splitting"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.stopwords = set(stopwords.words('english')) if config.STOPWORDS_ENABLED else set()
    
    def count_tokens(self, text: str) -> int:
        """Simple token counting using word tokenization"""
        return len(word_tokenize(text))
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs using configured separator"""
        paragraphs = text.split(self.config.PARAGRAPH_SEPARATOR)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def split_long_paragraph(self, paragraph: str) -> List[str]:
        """Split oversized paragraph at sentence boundaries with overlap"""
        sentences = sent_tokenize(paragraph)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence exceeds max size, finalize current chunk
            if current_tokens + sentence_tokens > self.config.MAX_CHUNK_SIZE and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_tokens = sum(self.count_tokens(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk if it has content
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def merge_short_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """Merge consecutive short paragraphs until reaching minimum size"""
        merged = []
        current_group = []
        current_tokens = 0
        
        for paragraph in paragraphs:
            para_tokens = self.count_tokens(paragraph)
            
            # If adding this paragraph exceeds target size, finalize current group
            if current_tokens + para_tokens > self.config.TARGET_CHUNK_SIZE and current_group:
                merged.append(' '.join(current_group))
                current_group = [paragraph]
                current_tokens = para_tokens
            else:
                current_group.append(paragraph)
                current_tokens += para_tokens
        
        # Add final group
        if current_group:
            merged.append(' '.join(current_group))
        
        return merged
    
    def extract_h3_titles(self, text: str) -> List[str]:
        """Extract all <h3> titles from chunk content"""
        h3_pattern = r'<h3[^>]*>(.*?)</h3>'
        matches = re.findall(h3_pattern, text, re.IGNORECASE | re.DOTALL)
        
        titles = []
        for match in matches:
            # Clean HTML tags and normalize
            clean_title = re.sub(r'<[^>]+>', '', match).strip()
            clean_title = re.sub(r'\s+', ' ', clean_title)
            if clean_title and len(clean_title) > 2:
                titles.append(clean_title)
        
        return titles
    
    def chunk_section_content(self, content: str, section_name: str, article_name: str, 
                            sub_article_name: Optional[str] = None, section_index: int = 0) -> List[Dict[str, Any]]:
        """
        Chunk a single section's content directly (no file I/O)
        Returns list of chunk dictionaries with content and metadata
        """
        if self.config.VERBOSE_LOGGING:
            print(f"    Chunking section: {section_name}")
        
        if not content or not content.strip():
            return []
        
        # Step 1: Split into paragraphs
        paragraphs = self.split_into_paragraphs(content)
        if not paragraphs:
            return []
        
        # Step 2: Merge short paragraphs
        merged_paragraphs = self.merge_short_paragraphs(paragraphs)
        
        # Step 3: Split long paragraphs
        final_chunks = []
        for merged_para in merged_paragraphs:
            para_tokens = self.count_tokens(merged_para)
            
            if para_tokens > self.config.MAX_CHUNK_SIZE:
                # Split long paragraph
                split_chunks = self.split_long_paragraph(merged_para)
                final_chunks.extend(split_chunks)
            else:
                final_chunks.append(merged_para)
        
        # Step 4: Create chunk objects with metadata
        chunk_objects = []
        for i, chunk_text in enumerate(final_chunks):
            if self.count_tokens(chunk_text) < self.config.MIN_CHUNK_SIZE:
                continue  # Skip chunks that are still too small
            
            # Extract sub-section names from this chunk
            sub_section_names = self.extract_h3_titles(chunk_text)
            sub_section_name = '; '.join(sub_section_names) if sub_section_names else None
            
            # Create chunk ID based on article and section names
            chunk_id = f"{article_name.lower().replace(' ', '_')}_{section_name.lower().replace(' ', '_')}_{i+1:03d}"
            
            chunk_obj = {
                'chunk_id': chunk_id,
                'content': chunk_text,
                'search_metadata': {
                    'article_name': article_name,
                    'sub_article_name': sub_article_name,
                    'section_name': section_name,  # Direct from scraper
                    'sub_section_name': sub_section_name,
                    'keywords': []  # Will be filled by keyword extractor
                },
                'debug_metadata': {
                    'section_index': section_index,
                    'chunk_index': i + 1,
                    'chunk_size': self.count_tokens(chunk_text),
                    'original_paragraph_count': len([p for p in paragraphs if chunk_text.find(p) != -1]),
                    'has_h3_tags': len(sub_section_names) > 0,
                    'processing_timestamp': datetime.now().isoformat()
                } if self.config.ENABLE_DEBUG_METADATA else {}
            }
            
            chunk_objects.append(chunk_obj)
        
        if self.config.VERBOSE_LOGGING:
            print(f"      Created {len(chunk_objects)} chunks from {len(paragraphs)} paragraphs")
        
        return chunk_objects
    

    



class KeywordExtractor:
    """Extracts keywords using BM25-style scoring"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.stopwords = set(stopwords.words('english')) if config.STOPWORDS_ENABLED else set()
        self.term_frequencies = Counter()  # Global term frequencies
        self.document_frequencies = Counter()  # Document frequencies
        self.total_documents = 0
    
    def preprocess_text(self, text: str) -> List[str]:
        """Tokenize and preprocess text for keyword extraction"""
        # Convert to lowercase and tokenize
        tokens = word_tokenize(text.lower())
        
        # Filter tokens
        filtered_tokens = []
        for token in tokens:
            # Remove non-alphabetic tokens and stopwords
            if (token.isalpha() and 
                len(token) >= self.config.MIN_KEYWORD_LENGTH and 
                len(token) <= self.config.MAX_KEYWORD_LENGTH and
                token not in self.stopwords):
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    def build_corpus_statistics(self, chunks: List[Dict[str, Any]]):
        """Build global term and document frequency statistics"""
        if self.config.VERBOSE_LOGGING:
            print("  Building corpus statistics for keyword extraction...")
        
        self.total_documents = len(chunks)
        
        for chunk in chunks:
            content = chunk['content']
            tokens = self.preprocess_text(content)
            
            # Update global term frequencies
            self.term_frequencies.update(tokens)
            
            # Update document frequencies (unique terms per document)
            unique_tokens = set(tokens)
            self.document_frequencies.update(unique_tokens)
    
    def calculate_bm25_score(self, term: str, term_freq: int, doc_length: int, avg_doc_length: float) -> float:
        """Calculate BM25 score for a term in a document"""
        # Get document frequency
        df = self.document_frequencies.get(term, 0)
        if df == 0:
            return 0.0
        
        # Calculate IDF
        idf = np.log((self.total_documents - df + 0.5) / (df + 0.5))
        
        # Calculate BM25 score
        k1 = self.config.BM25_K1
        b = self.config.BM25_B
        
        numerator = term_freq * (k1 + 1)
        denominator = term_freq + k1 * (1 - b + b * (doc_length / avg_doc_length))
        
        return idf * (numerator / denominator)
    
    def extract_keywords(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract top keywords for each chunk using BM25 scoring"""
        if self.config.VERBOSE_LOGGING:
            print("  Extracting keywords using BM25 scoring...")
        
        # Calculate average document length
        total_tokens = sum(len(self.preprocess_text(chunk['content'])) for chunk in chunks)
        avg_doc_length = total_tokens / len(chunks) if chunks else 1
        
        # Extract keywords for each chunk
        for chunk in chunks:
            content = chunk['content']
            tokens = self.preprocess_text(content)
            
            if not tokens:
                chunk['search_metadata']['keywords'] = []
                continue
            
            # Calculate term frequencies in this chunk
            chunk_tf = Counter(tokens)
            doc_length = len(tokens)
            
            # Calculate BM25 scores for all terms
            term_scores = {}
            for term, tf in chunk_tf.items():
                # Apply document frequency filters
                df = self.document_frequencies.get(term, 0)
                df_ratio = df / self.total_documents
                
                if (df >= self.config.MIN_KEYWORD_DF and 
                    df_ratio <= self.config.MAX_KEYWORD_DF_RATIO):
                    
                    score = self.calculate_bm25_score(term, tf, doc_length, avg_doc_length)
                    term_scores[term] = score
            
            # Get top keywords
            top_keywords = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
            keywords = [term for term, score in top_keywords[:self.config.KEYWORDS_PER_CHUNK]]
            
            chunk['search_metadata']['keywords'] = keywords
        
        return chunks


class RAGDatabase:
    """Main RAG database class that handles indexing and search"""
    
    def __init__(self, config: RAGConfig, db_path: str = "rag_db"):
        self.config = config
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.chunker = TextChunker(config)
        self.keyword_extractor = KeywordExtractor(config)
        self.embedding_model = None
        self.whoosh_index = None
        self.faiss_index = None
        self.chunk_mapping = {}  # Maps FAISS indices to chunk data
        
        # Load or initialize embedding model
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load the sentence transformer model"""
        if self.config.VERBOSE_LOGGING:
            print(f"Loading embedding model: {self.config.EMBEDDING_MODEL}")
        
        try:
            self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise
    
    def _create_whoosh_schema(self) -> Schema:
        """Create Whoosh schema (field boosting will be applied at query time)"""
        return Schema(
            # Unique identifier
            chunk_id=ID(stored=True, unique=True),
            
            # Searchable fields (boosting applied at query time)
            content=TEXT(analyzer=StemmingAnalyzer(), stored=True),
            article_name=TEXT(analyzer=StandardAnalyzer(), stored=True),
            sub_article_name=TEXT(analyzer=StandardAnalyzer(), stored=True),
            section_name=TEXT(analyzer=StandardAnalyzer(), stored=True),
            sub_section_name=TEXT(analyzer=StandardAnalyzer(), stored=True),
            keywords=TEXT(analyzer=StandardAnalyzer(), stored=True),
            
            # Stored-only fields (metadata)
            debug_metadata=STORED  # JSON string with debug info
        )
    
    def process_sections_directly(self, sections: List[Dict[str, str]], article_name: str, 
                                sub_article_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Process scraped sections directly into chunks without file I/O"""
        if self.config.VERBOSE_LOGGING:
            print(f"  Processing {len(sections)} sections for article: {article_name}")
        
        all_chunks = []
        
        for section_index, section in enumerate(sections):
            section_name = section['combined_title']  # This becomes our section_name metadata
            section_content = section['content']
            
            # Chunk this section's content
            section_chunks = self.chunker.chunk_section_content(
                content=section_content,
                section_name=section_name,
                article_name=article_name,
                sub_article_name=sub_article_name,
                section_index=section_index
            )
            
            all_chunks.extend(section_chunks)
        
        if self.config.VERBOSE_LOGGING:
            print(f"  Total chunks created: {len(all_chunks)}")
        
        return all_chunks
    

    

    
    def build_indices_from_chunks(self, chunks: List[Dict[str, Any]]):
        """Build both Whoosh and FAISS indices from pre-processed chunks"""
        print("Building RAG database indices...")
        
        if not chunks:
            raise ValueError("No chunks provided for indexing")
        
        # Step 1: Build corpus statistics and extract keywords
        self.keyword_extractor.build_corpus_statistics(chunks)
        chunks = self.keyword_extractor.extract_keywords(chunks)
        
        # Step 2: Build Whoosh index
        self._build_whoosh_index(chunks)
        
        # Step 3: Build FAISS index
        self._build_faiss_index(chunks)
        
        # Step 4: Save chunk mapping and metadata
        self._save_database_metadata(chunks)
        
        print(f"Database built successfully with {len(chunks)} chunks")
        return len(chunks)
    

    
    def _build_whoosh_index(self, chunks: List[Dict[str, Any]]):
        """Build Whoosh index for BM25 search"""
        if self.config.VERBOSE_LOGGING:
            print("  Building Whoosh index...")
        
        # Create index directory
        whoosh_dir = self.db_path / "whoosh_index"
        whoosh_dir.mkdir(exist_ok=True)
        
        # Create schema and index
        schema = self._create_whoosh_schema()
        self.whoosh_index = create_in(str(whoosh_dir), schema)
        
        # Add documents to index
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
        
        if self.config.VERBOSE_LOGGING:
            print(f"    Whoosh index built with {len(chunks)} documents")
    
    def _build_faiss_index(self, chunks: List[Dict[str, Any]]):
        """Build FAISS index for semantic search"""
        if self.config.VERBOSE_LOGGING:
            print("  Building FAISS index...")
        
        # Generate embeddings for all chunks
        chunk_texts = [chunk['content'] for chunk in chunks]
        embeddings = self.embedding_model.encode(chunk_texts, show_progress_bar=True)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        if self.config.FAISS_INDEX_TYPE == "IndexFlatIP":
            self.faiss_index = faiss.IndexFlatIP(dimension)
        elif self.config.FAISS_INDEX_TYPE == "IndexIVFFlat":
            # For larger datasets
            nlist = min(100, len(chunks) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatIP(dimension)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.faiss_index.train(embeddings)
        else:
            raise ValueError(f"Unsupported FAISS index type: {self.config.FAISS_INDEX_TYPE}")
        
        # Add embeddings to index
        self.faiss_index.add(embeddings)
        
        # Create chunk mapping (FAISS index → chunk data)
        self.chunk_mapping = {i: chunk for i, chunk in enumerate(chunks)}
        
        # Save FAISS index
        faiss_path = self.db_path / "faiss_index.bin"
        faiss.write_index(self.faiss_index, str(faiss_path))
        
        # Save chunk mapping
        mapping_path = self.db_path / "chunk_mapping.pkl"
        with open(mapping_path, 'wb') as f:
            pickle.dump(self.chunk_mapping, f)
        
        if self.config.VERBOSE_LOGGING:
            print(f"    FAISS index built with {len(chunks)} embeddings ({dimension}D)")
    
    def _save_database_metadata(self, chunks: List[Dict[str, Any]]):
        """Save database metadata and statistics"""
        metadata = {
            'creation_timestamp': datetime.now().isoformat(),
            'total_chunks': len(chunks),
            'config': {
                'min_chunk_size': self.config.MIN_CHUNK_SIZE,
                'max_chunk_size': self.config.MAX_CHUNK_SIZE,
                'target_chunk_size': self.config.TARGET_CHUNK_SIZE,
                'overlap_size': self.config.OVERLAP_SIZE,
                'embedding_model': self.config.EMBEDDING_MODEL,
                'keywords_per_chunk': self.config.KEYWORDS_PER_CHUNK
            },
            'statistics': {
                'avg_chunk_size': np.mean([len(word_tokenize(c['content'])) for c in chunks]),
                'total_keywords': sum(len(c['search_metadata']['keywords']) for c in chunks),
                'articles_processed': len(set(c['search_metadata']['article_name'] for c in chunks)),
                'sections_processed': len(set(c['search_metadata']['section_name'] for c in chunks))
            }
        }
        
        with open(self.db_path / "database_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        if self.config.VERBOSE_LOGGING:
            print("  Database metadata saved")
    
    def load_indices(self):
        """Load existing Whoosh and FAISS indices"""
        # Load Whoosh index
        whoosh_dir = self.db_path / "whoosh_index"
        if whoosh_dir.exists():
            self.whoosh_index = open_dir(str(whoosh_dir))
        
        # Load FAISS index
        faiss_path = self.db_path / "faiss_index.bin"
        if faiss_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_path))
        
        # Load chunk mapping
        mapping_path = self.db_path / "chunk_mapping.pkl"
        if mapping_path.exists():
            with open(mapping_path, 'rb') as f:
                self.chunk_mapping = pickle.load(f)
        
        return self.whoosh_index is not None and self.faiss_index is not None
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Two-step hybrid search: BM25 → Semantic reranking
        """
        if top_k is None:
            top_k = self.config.FINAL_RESULTS
        
        if not self.whoosh_index or not self.faiss_index:
            raise ValueError("Indices not loaded. Call build_indices() or load_indices() first.")
        
        # Step 1: BM25 search with Whoosh
        bm25_results = self._bm25_search(query, limit=self.config.BM25_CANDIDATES)
        
        if not bm25_results:
            return []
        
        # Step 2: Semantic reranking with FAISS
        final_results = self._semantic_rerank(query, bm25_results, top_k)
        
        return final_results
    
    def _bm25_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Perform BM25 search using Whoosh with fallback strategies"""
        with self.whoosh_index.searcher(weighting=BM25F()) as searcher:
            # Create multi-field parser with field boosting
            field_boosts = {
                'content': self.config.CONTENT_BOOST,
                'article_name': self.config.ARTICLE_NAME_BOOST,
                'sub_article_name': self.config.SUB_ARTICLE_NAME_BOOST,
                'section_name': self.config.SECTION_NAME_BOOST,
                'sub_section_name': self.config.SUB_SECTION_BOOST,
                'keywords': self.config.KEYWORDS_BOOST
            }
            
            parser = MultifieldParser(
                ['content', 'article_name', 'sub_article_name', 'section_name', 'sub_section_name', 'keywords'],
                self.whoosh_index.schema,
                fieldboosts=field_boosts
            )
            
            # Strategy 1: Try original query
            try:
                parsed_query = parser.parse(query)
                results = searcher.search(parsed_query, limit=limit)
                if len(results) > 0:
                    return self._convert_whoosh_results(results)
            except:
                pass
            
            # Strategy 2: Extract key terms (remove stop words and question words)
            import re
            # Remove common question words and punctuation
            clean_query = re.sub(r'\b(what|is|are|how|where|when|who|why|the|a|an|about|tell|me)\b', '', query.lower())
            clean_query = re.sub(r'[^\w\s]', ' ', clean_query)
            clean_query = ' '.join(clean_query.split())  # Remove extra spaces
            
            if clean_query and clean_query != query.lower():
                try:
                    parsed_query = parser.parse(clean_query)
                    results = searcher.search(parsed_query, limit=limit)
                    if len(results) > 0:
                        return self._convert_whoosh_results(results)
                except:
                    pass
            
            # Strategy 3: Try individual important words with OR
            words = clean_query.split() if clean_query else query.split()
            important_words = [w for w in words if len(w) > 2]  # Skip very short words
            
            if important_words:
                try:
                    or_query = ' OR '.join(important_words)
                    parsed_query = parser.parse(or_query)
                    results = searcher.search(parsed_query, limit=limit)
                    return self._convert_whoosh_results(results)
                except:
                    pass
            
            # If all strategies fail, return empty results
            return []
    
    def _convert_whoosh_results(self, results) -> List[Dict[str, Any]]:
        """Convert Whoosh results to our standard format"""
        bm25_results = []
        for i, hit in enumerate(results):
            result = {
                'chunk_id': hit['chunk_id'],
                'content': hit['content'],
                'bm25_score': hit.score,
                'bm25_rank': i + 1,
                'search_metadata': {
                    'article_name': hit['article_name'],
                    'sub_article_name': hit['sub_article_name'],
                    'section_name': hit['section_name'],
                    'sub_section_name': hit['sub_section_name'],
                    'keywords': hit['keywords'].split() if hit['keywords'] else []
                }
            }
            
            # Add debug metadata if available
            if hit['debug_metadata']:
                try:
                    result['debug_metadata'] = json.loads(hit['debug_metadata'])
                except json.JSONDecodeError:
                    result['debug_metadata'] = {}
            
            bm25_results.append(result)
        
        return bm25_results
    
    def _semantic_rerank(self, query: str, bm25_results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Rerank BM25 results using semantic similarity"""
        if not bm25_results:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Generate embeddings for candidate chunks
        candidate_texts = [result['content'] for result in bm25_results]
        candidate_embeddings = self.embedding_model.encode(candidate_texts)
        faiss.normalize_L2(candidate_embeddings)
        
        # Calculate semantic similarities
        similarities = np.dot(query_embedding, candidate_embeddings.T)[0]
        
        # Combine BM25 and semantic scores
        for i, result in enumerate(bm25_results):
            bm25_score = result['bm25_score']
            semantic_score = similarities[i]
            
            # Normalize BM25 score (simple min-max normalization)
            max_bm25 = max(r['bm25_score'] for r in bm25_results)
            min_bm25 = min(r['bm25_score'] for r in bm25_results)
            normalized_bm25 = (bm25_score - min_bm25) / (max_bm25 - min_bm25) if max_bm25 > min_bm25 else 0
            
            # Combine scores
            combined_score = (
                self.config.RERANK_WEIGHT_BM25 * normalized_bm25 + 
                self.config.RERANK_WEIGHT_SEMANTIC * semantic_score
            )
            
            result['semantic_score'] = semantic_score
            result['combined_score'] = combined_score
        
        # Sort by combined score and return top-k
        reranked_results = sorted(bm25_results, key=lambda x: x['combined_score'], reverse=True)
        return reranked_results[:top_k]


def main():
    """Main function - use one_piece_scraper.py instead for integrated workflow"""
    print("RAG Database Creator")
    print("=" * 50)
    print("This module is now integrated with the scraper.")
    print("Run 'python one_piece_scraper.py' to scrape and build the database.")
    print("\nFor direct database operations, use the RAGDatabase class programmatically.")
    return True


if __name__ == "__main__":
    main()
