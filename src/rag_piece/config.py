"""
Configuration settings for the RAG Piece system.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class RAGConfig:
    """Global configuration for RAG system - all parameters in one place for easy tuning"""
    
    # === CHUNKING PARAMETERS ===
    MIN_CHUNK_SIZE: int = 100            # tokens - merge if below this
    MAX_CHUNK_SIZE: int = 400            # tokens - split if above this
    OVERLAP_SIZE: int = 50               # tokens - overlap when splitting
    TARGET_CHUNK_SIZE: int = 250         # tokens - ideal chunk size
    PARAGRAPH_SEPARATOR: str = "\n\n"    # how to split into paragraphs
    SENTENCE_SEPARATORS: List[str] = None
    
    # === BM25 PARAMETERS ===
    BM25_K1: float = 1.2                # term frequency saturation
    BM25_B: float = 0.75                # length normalization
    BM25_CANDIDATES: int = 100           # candidates from first step
    
    # === KEYWORD EXTRACTION ===
    KEYWORDS_PER_CHUNK: int = 10         # top keywords to extract per chunk
    MIN_KEYWORD_LENGTH: int = 3          # minimum keyword character length
    MAX_KEYWORD_LENGTH: int = 30         # maximum keyword character length
    MIN_KEYWORD_DF: int = 1              # must appear in at least N chunks
    MAX_KEYWORD_DF_RATIO: float = 0.8    # must appear in <80% of chunks
    STOPWORDS_ENABLED: bool = True       # remove common stopwords
    PRESERVE_PROPER_NOUNS: bool = True   # keep character/location names
    
    # === FIELD BOOSTING ===
    CONTENT_BOOST: float = 1.0             # base content weight
    ARTICLE_NAME_BOOST: float = 3.0        # article name mentions
    SUB_ARTICLE_NAME_BOOST: float = 2.8    # sub-article name relevance
    SECTION_NAME_BOOST: float = 2.5        # section title relevance
    SUB_SECTION_BOOST: float = 2.0         # sub-section title relevance
    KEYWORDS_BOOST: float = 1.8            # BM25-extracted keywords
    
    # === SEMANTIC SEARCH ===
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # sentence transformer model
    FAISS_INDEX_TYPE: str = "IndexFlatIP"       # FAISS index type
    SIMILARITY_THRESHOLD: float = 0.3           # minimum similarity score
    
    # === RESULT FUSION ===
    FINAL_RESULTS: int = 10              # final results after reranking
    RERANK_WEIGHT_BM25: float = 0.4      # BM25 weight in fusion
    RERANK_WEIGHT_SEMANTIC: float = 0.6  # semantic weight in fusion
    
    # === LOGGING AND DEBUG ===
    VERBOSE_LOGGING: bool = False        # detailed progress logs
    ENABLE_DEBUG_METADATA: bool = False  # include debug info in chunks
    LOG_LEVEL: str = "INFO"              # logging level
    
    def __post_init__(self):
        """Initialize default values that can't be set in dataclass"""
        if self.SENTENCE_SEPARATORS is None:
            self.SENTENCE_SEPARATORS = [". ", "! ", "? ", ".\n", "!\n", "?\n"]
    
    def validate(self) -> None:
        """Validate configuration parameters"""
        from .utils import validate_input
        
        # Validate chunk sizes
        validate_input(self.MIN_CHUNK_SIZE, int, "MIN_CHUNK_SIZE", min_val=1)
        validate_input(self.MAX_CHUNK_SIZE, int, "MAX_CHUNK_SIZE", min_val=self.MIN_CHUNK_SIZE)
        validate_input(self.TARGET_CHUNK_SIZE, int, "TARGET_CHUNK_SIZE", 
                      min_val=self.MIN_CHUNK_SIZE, max_val=self.MAX_CHUNK_SIZE)
        validate_input(self.OVERLAP_SIZE, int, "OVERLAP_SIZE", min_val=0)
        
        # Validate BM25 parameters
        validate_input(self.BM25_K1, float, "BM25_K1", min_val=0.0)
        validate_input(self.BM25_B, float, "BM25_B", min_val=0.0, max_val=1.0)
        validate_input(self.BM25_CANDIDATES, int, "BM25_CANDIDATES", min_val=1)
        
        # Validate keyword extraction
        validate_input(self.KEYWORDS_PER_CHUNK, int, "KEYWORDS_PER_CHUNK", min_val=1)
        validate_input(self.MIN_KEYWORD_LENGTH, int, "MIN_KEYWORD_LENGTH", min_val=1)
        validate_input(self.MAX_KEYWORD_LENGTH, int, "MAX_KEYWORD_LENGTH", 
                      min_val=self.MIN_KEYWORD_LENGTH)
        validate_input(self.MAX_KEYWORD_DF_RATIO, float, "MAX_KEYWORD_DF_RATIO", 
                      min_val=0.0, max_val=1.0)
        
        # Validate boost weights
        for attr in ["CONTENT_BOOST", "ARTICLE_NAME_BOOST", "SUB_ARTICLE_NAME_BOOST",
                     "SECTION_NAME_BOOST", "SUB_SECTION_BOOST", "KEYWORDS_BOOST"]:
            validate_input(getattr(self, attr), float, attr, min_val=0.0)
        
        # Validate fusion weights
        validate_input(self.RERANK_WEIGHT_BM25, float, "RERANK_WEIGHT_BM25", min_val=0.0, max_val=1.0)
        validate_input(self.RERANK_WEIGHT_SEMANTIC, float, "RERANK_WEIGHT_SEMANTIC", min_val=0.0, max_val=1.0)
        
        # Check that fusion weights sum to 1.0 (approximately)
        total_weight = self.RERANK_WEIGHT_BM25 + self.RERANK_WEIGHT_SEMANTIC
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Fusion weights must sum to 1.0, got {total_weight}")
        
        # Validate other parameters
        validate_input(self.SIMILARITY_THRESHOLD, float, "SIMILARITY_THRESHOLD", min_val=0.0, max_val=1.0)
        validate_input(self.FINAL_RESULTS, int, "FINAL_RESULTS", min_val=1)
