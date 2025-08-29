# One Piece Wiki RAG Database System

A comprehensive **Retrieval-Augmented Generation (RAG)** system that scrapes One Piece Wiki content and creates an intelligent, searchable database using both keyword (BM25) and semantic search technologies.

> **🆕 Latest Update**: Fixed image download paths, added API retry logic with caching, and improved system reliability. All images now download to `data/images/[article_name]/` as expected!

## 🚀 Key Features

### **Complete Database Construction Pipeline**
- **End-to-End Processing**: Scraping → Summarization → CSV Conversion → Chunking → Database → Search Ready
- **Intelligent Content Processing**: Hybrid chunking with **overlapping chunks** for better context preservation
- **CSV Integration**: Automatic conversion of wiki tables to searchable text using LLM (with optional in-memory processing)
- **Sequential Processing**: Content chunking, summarization, and CSV conversion run in sequence for reliability
- **Duplicate Prevention**: Smart summarization that runs once and reuses summaries across components
- **Automatic Image Management**: High-quality image downloads with basic filtering and organization
- **API Reliability**: Built-in retry logic with exponential backoff and shared caching to prevent duplicate API calls
- **Basic Validation**: Input validation and error handling for extracted content

### **Advanced Search Capabilities**
- **Two-Step Retrieval**: BM25 keyword search + semantic similarity ranking
- **Natural Language Queries**: Ask questions like "What is Arabasta Kingdom?"
- **Field Boosting**: Weighted search across titles, sections, and content
- **Query Fallback**: Robust handling of complex questions with multiple strategies
- **Context Preservation**: Overlapping chunks ensure important information isn't lost at boundaries
- **Smart Query Processing**: Multiple fallback strategies for complex queries with automatic query cleaning and expansion

### **Intelligent Content Organization**
- **Metadata-Rich Chunks**: Article names, sections, sub-sections, and keywords
- **Hierarchical Search**: Article titles > Sub-articles > Sections > Sub-sections > Keywords > Content
- **Context Preservation**: Maintains document structure and relationships with smart overlap
- **CSV Data Integration**: Wiki tables automatically converted to structured text while preserving data relationships
- **Summary Integration**: Optional AI-generated article summaries for enhanced search context

## 🛠️ Technology Stack

- **Web Scraping**: MediaWiki API integration with retry logic and caching
- **Keyword Search**: Whoosh with BM25F scoring and field boosting
- **Semantic Search**: FAISS with sentence-transformers embeddings
- **Text Processing**: NLTK for tokenization, keyword extraction, and analysis
- **Image Processing**: PIL for validation and quality filtering
- **LLM Integration**: OpenAI GPT-4o-mini for summarization and CSV conversion
- **Data Processing**: Pandas for CSV handling and data manipulation
- **AI Framework**: LangChain for LLM orchestration and prompt management

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd RAG-Piece
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API key** (for summarization and CSV conversion)
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

4. **Run the system**
   ```bash
   python -m src.rag_piece.main
   ```

## 🎯 Usage

### **Complete Database Construction Process**
```bash
python -m src.rag_piece.main
```

This command orchestrates the entire pipeline:

#### **Phase 1: Data Collection & Scraping**
1. **Web Scraping**: Scrapes One Piece Wiki articles using MediaWiki API
2. **Content Extraction**: Extracts HTML sections, headings, and content
3. **Image Download**: Downloads high-quality images to `data/images/[article_name]/`
4. **CSV Generation**: Converts wiki tables to CSV files or in-memory DataFrames

#### **Phase 2: Content Processing & Summarization (Sequential)**
5. **Article Summarization**: Creates comprehensive summaries using OpenAI GPT-4o-mini (optional, configurable)
6. **CSV to Text Conversion**: Converts CSV data to structured, searchable text using LLM while preserving data relationships
7. **Summary Reuse**: Avoids duplicate summarizer runs by sharing summaries across components
8. **Sequential Processing**: Operations run in sequence for reliability and resource management

#### **Phase 3: Content Chunking & Organization**
8. **Hybrid Chunking**: Breaks content into optimized chunks using paragraph-based splitting with intelligent merging
9. **Overlapping Chunks**: Creates overlapping chunks for better context preservation and search quality
10. **Metadata Enrichment**: Adds rich metadata (article, section, sub-sections, keywords) with BM25-style scoring
11. **Context Preservation**: Maintains relationships between data points and chunk boundaries
12. **Smart Chunking**: Recursive splitting strategy that preserves semantic boundaries

#### **Phase 4: Database Construction & Storage**
13. **Index Building**: Creates BM25 keyword search and semantic search indices
14. **Vector Embeddings**: Generates embeddings for semantic similarity search
15. **Database Assembly**: Combines all processed content into searchable database
16. **Dual Search Engine**: BM25 for fast keyword retrieval, FAISS for semantic similarity

#### **Phase 5: Testing & Validation**
17. **Search Testing**: Validates database functionality with sample queries
18. **Logging**: Records all operations with detailed progress tracking
19. **Performance Monitoring**: Tracks processing time and resource usage for each component

### **Advanced Features**

#### **CSV to Text Conversion**
The system automatically converts wiki tables to searchable text using LLM:
- **Data Relationship Preservation**: Maintains logical connections between columns and rows
- **Structured Output**: Creates well-organized text with clear headings and sections
- **Context Integration**: Uses article summaries to provide background context
- **Flexible Processing**: Choose between in-memory processing or file-based debug mode

#### **Article Summarization**
Optional AI-powered summarization for enhanced search:
- **LangChain Integration**: Uses the refine method for high-quality summaries
- **Token Compliance**: Ensures summaries fit within chunk size limits
- **Sub-article Support**: Creates summaries for both main articles and sub-articles
- **Cost Control**: Can be disabled to avoid API costs

#### **API Reliability & Caching**
Built-in mechanisms for robust operation:
- **Retry Logic**: 3-attempt retry with exponential backoff (2s, 4s, 8s delays)
- **Shared Caching**: Eliminates duplicate API calls between scraper instances
- **Rate Limiting**: Built-in delays to respect MediaWiki API limits
- **Error Recovery**: Graceful handling of temporary failures and network issues
- **Fallback Strategies**: Multiple approaches for handling different types of failures
- **Comprehensive Logging**: Full audit trail of all retry attempts and error conditions

### **Search the Database**
```python
from src.rag_piece import RAGDatabase, RAGConfig

# Initialize database
config = RAGConfig()
db = RAGDatabase(config)
db.load_indices()

# Natural language search
results = db.search("What is Arabasta Kingdom?", top_k=5)
results = db.search("Tell me about the desert in Arabasta", top_k=3)
results = db.search("Who are the main characters?", top_k=10)

# Print results
for i, result in enumerate(results, 1):
    print(f"{i}. {result['search_metadata']['section_name']}")
    print(f"   Score: {result['combined_score']:.3f}")
    print(f"   Content: {result['content'][:100]}...")
```

### **Chunking Strategy & Metadata**

The system uses a sophisticated hybrid chunking approach:

#### **Recursive Chunking Strategy**
1. **Paragraph Splitting**: Content split by paragraph separators (`\n\n`)
2. **Smart Merging**: Short chunks merged until they reach target size
3. **Intelligent Splitting**: Long chunks split at sentence boundaries
4. **Overlap Creation**: 50-token overlap between chunks for context continuity

#### **Rich Metadata System**
Each chunk includes comprehensive metadata:
- **Search Metadata**: Article names, sections, sub-sections, and keywords
- **Debug Metadata**: Processing timestamps, chunk sizes, and structural information
- **Hierarchical Organization**: Article → Sub-article → Section → Sub-section → Keywords → Content

#### **Field Boosting Hierarchy**
Different metadata fields have different search weights:
- **Article Names**: 3.0x boost (highest priority)
- **Sub-Article Names**: 2.8x boost
- **Section Names**: 2.5x boost
- **Sub-Section Names**: 2.0x boost
- **Keywords**: 1.8x boost
- **Content**: 1.0x boost (base weight)

### **Using Individual Components**
```python
from src.rag_piece import OneWikiScraper, TextChunker, RAGConfig

# Use just the scraper
scraper = OneWikiScraper(max_images=10)
sections, metadata = scraper.scrape_article("Arabasta Kingdom")

# Use just the chunker
config = RAGConfig()
chunker = TextChunker(config)
chunks = chunker.chunk_section_content(content, "Section Name", "Article Name")
```

## 🗂️ Output Structure

```
RAG-Piece/
├── src/                          # Source code
│   └── rag_piece/                # Main package
│       ├── __init__.py           # Package initialization
│       ├── main.py               # Application main function
│       ├── config.py             # Centralized configuration settings
│       ├── database.py           # RAG database coordination
│       ├── chunking.py           # Text chunking with overlap support
│       ├── keywords.py           # Keyword extraction
│       ├── search.py             # BM25 and semantic search
│       ├── scraper.py            # Wiki scraping functionality
│       ├── summarizer.py         # Article summarization with progress tracking
│       ├── csv_scraper.py        # CSV extraction (file-based or in-memory)
│       ├── csv_to_text.py        # CSV to text conversion using LLM
│       └── utils.py              # Utilities and logging
├── logs/                         # Application logs
├── requirements.txt              # Python dependencies
├── docs/                         # Comprehensive documentation
├── data/                         # Processed data
│   ├── images/                   # Downloaded images organized by article
│   │   └── [article_name]/       # Article-specific image folders
├── data/                         # Processed data
│   ├── debug/
│   │   └── summaries/            # Article summaries (if enabled)
│   ├── debug/                    # Debug files
│   │   ├── csv_files/            # CSV data from wiki tables (optional)
│   │   │   └── [article_name]/   # Organized by article
│   │   └── csv2text/             # CSV conversion debug files
│   └── rag_db/                   # RAG database files
│       ├── whoosh_index/         # BM25 keyword search index
│       ├── faiss_index.bin       # Semantic search index
│       ├── chunk_mapping.pkl     # Mapping between indices
│       └── database_metadata.json # Database statistics and info
└── [legacy files for compatibility]
```

## ⚙️ Configuration

All parameters are centralized in the `RAGConfig` class for easy tuning. The system provides comprehensive configuration options for every aspect of operation:

### **Chunking Configuration**
```python
# === CHUNKING ===
MAX_CHUNK_SIZE: int = 400          # maximum chunk size in tokens
MIN_CHUNK_SIZE: int = 50           # minimum chunk size in tokens
TARGET_CHUNK_SIZE: int = 300       # target chunk size in tokens
CHUNK_OVERLAP: int = 50            # overlap between chunks in tokens (NEW!)
PARAGRAPH_SEPARATOR: str = "\n\n"  # separator for paragraph splitting
SENTENCE_SEPARATORS: List[str] = [". ", "! ", "? ", "\n"]  # sentence boundaries

# === SUMMARIZATION CHUNKING ===
SUMMARY_INPUT_CHUNK_SIZE: int = 800      # input chunk size for summarization
SUMMARY_CHUNK_OVERLAP: int = 100        # overlap between summarization chunks
```

### **CSV Processing Configuration**
```python
# === CSV SCRAPING ===
ENABLE_CSV_SCRAPING: bool = True      # enable table extraction
SAVE_CSV_FILES_FOR_DEBUG: bool = False # save CSV files (optional, for debugging)
CSV_REQUEST_DELAY: float = 1.0         # delay between requests

# === CSV TO TEXT CONVERSION ===
ENABLE_CSV_TO_TEXT: bool = True     # enable CSV to text conversion
CSV_TO_TEXT_MODEL: str = "gpt-4o-mini"  # OpenAI model for conversion
CSV_TO_TEXT_TEMPERATURE: float = 0.2     # temperature for consistency
SAVE_CSV_TO_TEXT_FILES: bool = False     # save conversion results for debugging
```

### **Summarization Configuration**
```python
# === SUMMARIZATION ===
ENABLE_SUMMARIZATION: bool = True   # enable article summarization
SUMMARY_MODEL: str = "gpt-4o-mini"  # OpenAI model for summarization
SUMMARY_TEMPERATURE: float = 0.3     # temperature for summary generation
SAVE_SUMMARIES_TO_FILES: bool = False # save summaries as text files
MAX_INPUT_TEXT_TOKENS: int = 8000    # maximum input text length before summarization
```

### **Search Engine Configuration**
```python
# === BM25 PARAMETERS ===
BM25_K1: float = 1.2                # term frequency saturation
BM25_B: float = 0.75                # length normalization
BM25_CANDIDATES: int = 100           # candidates from first step

# === SEMANTIC SEARCH ===
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # sentence transformer model
FAISS_INDEX_TYPE: str = "IndexFlatIP"       # FAISS index type
SIMILARITY_THRESHOLD: float = 0.3           # minimum similarity score

# === RESULT FUSION ===
FINAL_RESULTS: int = 10              # final results after reranking
RERANK_WEIGHT_BM25: float = 0.4      # BM25 weight in fusion
RERANK_WEIGHT_SEMANTIC: float = 0.6  # semantic weight in fusion
```

### **API Reliability Configuration**
```python
# === API RETRY SETTINGS ===
API_MAX_RETRIES: int = 3               # maximum API retry attempts
API_BASE_DELAY: float = 2.0            # base delay between retries (seconds)
API_RETRY_EXPONENTIAL: bool = True     # use exponential backoff for retries
```

### **Logging & Debug Configuration**
```python
# === LOGGING AND DEBUG ===
VERBOSE_LOGGING: bool = False        # detailed progress logs
ENABLE_DEBUG_METADATA: bool = False  # include debug info in chunks
LOG_LEVEL: str = "INFO"              # logging level (DEBUG, INFO, WARNING, ERROR)
```

### **Articles Configuration**
```python
# === ARTICLES TO SCRAPE ===
ARTICLES_TO_SCRAPE: List[str] = ["Arabasta Kingdom"]  # configurable article list
```

### **Field Boosting Hierarchy**
The system uses a sophisticated field boosting system to ensure the most relevant results appear first:

1. **Article Names** (3.0x boost) - Main article titles (highest priority)
2. **Sub-Article Names** (2.8x boost) - Sub-article/redirect names
3. **Section Names** (2.5x boost) - Section titles from original structure
4. **Sub-Section Names** (2.0x boost) - `<h3>` headings within content
5. **Keywords** (1.8x boost) - BM25-extracted important terms
6. **Content** (1.0x boost) - Main text content (base weight)

This hierarchy ensures that:
- **Title matches** get the highest priority
- **Structural information** is weighted appropriately
- **Content relevance** is balanced with metadata importance
- **Search results** are ranked by overall relevance, not just text matching

## 🔍 Search Examples

### **Search Engine Capabilities**

#### **Two-Step Search Process**
1. **BM25 Keyword Search**: Fast retrieval of 100 candidates using field boosting
2. **Semantic Reranking**: FAISS similarity scoring with configurable weights
3. **Result Fusion**: Combines both approaches with configurable balance
4. **Final Ranking**: Top results with combined BM25 + semantic scores

#### **Query Processing Strategies**
The system uses multiple fallback strategies for complex queries:
- **Strategy 1**: Direct query processing
- **Strategy 2**: Query cleaning (removes stop words)
- **Strategy 3**: OR expansion for important terms
- **Strategy 4**: Semantic similarity fallback

### **Natural Language Queries**
```python
# Character information
results = db.search("What is Arabasta Kingdom?")

# Specific topics
results = db.search("Tell me about the desert in Arabasta")

# Character relationships
results = db.search("Who are the main characters in Arabasta?")
```

### **Keyword Searches**
```python
# Simple keywords
results = db.search("Arabasta")
results = db.search("Citizens")

# Multiple keywords
results = db.search("kingdom desert citizens")
```

## 🆕 Recent Improvements

### **Image Download System (FIXED!)**
- **Correct Path Structure**: Images now download to `data/images/[article_name]/` as expected
- **Metadata Consistency**: Fixed metadata key mismatch (`images_downloaded` vs `total_images_downloaded`)
- **Quality Filtering**: Downloads only high-quality images (100x100 minimum resolution)
- **Organized Storage**: Each article gets its own image folder with proper naming

### **API Reliability & Caching (NEW!)**
- **Retry Logic**: 3-attempt retry with exponential backoff (2s, 4s, 8s delays)
- **Shared Sub-Article Cache**: Eliminates duplicate API calls between scraper instances
- **Configurable Retry Settings**: `API_MAX_RETRIES` and `API_BASE_DELAY` in config
- **Consistent Results**: Both text and CSV scrapers now return identical sub-article counts
- **Rate Limiting**: Built-in delays between API requests to respect MediaWiki limits
- **Error Recovery**: Graceful handling of temporary API failures and network issues

### **Overlapping Chunks (NEW!)**
- **Better Context Preservation**: 50-token overlap between chunks prevents information loss
- **Improved Search Quality**: Queries can find content that spans chunk boundaries
- **Semantic Coherence**: Chunks maintain better meaning across boundaries
- **Configurable**: Can be disabled by setting `CHUNK_OVERLAP = 0`

### **In-Memory CSV Processing**
- **Eliminates File I/O**: CSV data processed directly in memory for efficiency
- **Debug Mode**: Optional file saving for troubleshooting
- **Parallel Processing**: CSV conversion runs alongside content processing

### **Enhanced Progress Tracking & Monitoring**
- **Detailed Logging**: Comprehensive logging with timing information for operations
- **Better Error Handling**: Robust retry mechanisms and fallback strategies
- **Basic Performance Monitoring**: Track processing time for each component
- **Comprehensive Logging**: Both console and file output with timestamps
- **Log Management**: Automatic log file creation and organization
- **Debug Mode**: Configurable logging levels for troubleshooting

### **Centralized Configuration System**
- **Single Source of Truth**: All parameters in `RAGConfig` class
- **Easy Tuning**: Modify chunk sizes, overlap, and processing options
- **Validation**: Automatic validation of configuration parameters
- **Type Safety**: Strong typing with dataclass validation
- **Default Values**: Sensible defaults for all parameters

## 📚 Documentation & Resources

### **Comprehensive Documentation**
The system includes extensive documentation covering all aspects:
- **Code Comments**: Detailed inline documentation throughout the codebase
- **Configuration Guide**: All parameters explained in `RAGConfig` class
- **API Documentation**: Comprehensive function and class documentation
- **Usage Examples**: Practical examples for common use cases

### **Development Resources**
- **Test Suite**: Comprehensive testing framework for validation
- **Debug Tools**: Built-in debugging and monitoring capabilities
- **Logging System**: Full audit trail for troubleshooting
- **Configuration Examples**: Sample configurations for different scenarios

### **Community & Support**
- **Issue Tracking**: Report bugs and request features
- **Contributions**: Guidelines for contributing to the project
- **Examples**: Sample code and configuration files
- **Troubleshooting**: Common issues and solutions

## 🔄 Complete Database Construction Process

### **Comprehensive Data Flow Overview**
The system implements a sophisticated multi-stage processing pipeline:

```
One Piece Wiki → Scraping → HTML Content + Images + CSV Tables
     ↓
HTML Content → Summarization → Article Summaries (with progress tracking)
     ↓
CSV Tables → LLM Conversion → Structured Text (in-memory or file-based)
     ↓
All Text → Hybrid Chunking → Optimized Chunks + Overlap + Rich Metadata
     ↓
Chunks → Database → BM25 + Semantic Search Indices
     ↓
Ready for Natural Language Queries
```

**Key Processing Stages:**
1. **Content Extraction**: MediaWiki API integration with retry logic
2. **Content Processing**: Sequential summarization and CSV conversion
3. **Intelligent Chunking**: Recursive splitting with overlap preservation
4. **Metadata Enrichment**: BM25 keyword extraction and structural analysis
5. **Index Construction**: Dual search engine setup (BM25 + FAISS)
6. **Search Optimization**: Field boosting and result fusion

### **Key Components & Their Roles**

#### **1. OneWikiScraper**
- **Purpose**: Extracts content from One Piece Wiki
- **Output**: HTML sections, images, CSV tables
- **Organization**: Creates structured folders for each article

#### **2. ArticleSummarizer**
- **Purpose**: Creates comprehensive article summaries using GPT-4o-mini
- **Output**: Summary chunks with context and progress tracking
- **Efficiency**: Runs once per article, summaries reused across components

#### **3. CSVWikiScraper**
- **Purpose**: Extracts wiki tables as CSV data
- **Flexibility**: File-based or in-memory processing
- **Debug Support**: Optional file saving for troubleshooting

#### **4. CSVToTextConverter**
- **Purpose**: Converts CSV data to searchable text using LLM
- **Input**: CSV files/DataFrames + existing article summaries
- **Output**: Structured text maintaining data relationships
- **Debug**: Saves conversion results for quality verification

#### **5. TextChunker**
- **Purpose**: Breaks content into optimal chunks for vector embedding
- **Strategy**: Hybrid approach (paragraph → merge → split → overlap)
- **Metadata**: Enriches each chunk with search context
- **Context**: Maintains relationships between chunks with smart overlap

#### **6. RAGDatabase**
- **Purpose**: Orchestrates the entire pipeline
- **Indices**: BM25 keyword search + FAISS semantic search
- **Integration**: Combines all processed content
- **Search**: Provides unified search interface
- **Management**: Handles index loading, saving, and maintenance

#### **7. SearchEngine**
- **Purpose**: Provides dual search capabilities
- **BM25 Search**: Fast keyword-based retrieval with field boosting
- **Semantic Search**: Vector similarity using sentence transformers
- **Result Fusion**: Intelligent combination of both search approaches
- **Query Processing**: Multiple fallback strategies for complex queries

#### **8. KeywordExtractor**
- **Purpose**: Extracts meaningful keywords from content
- **BM25 Scoring**: Uses document frequency and term frequency analysis
- **Quality Filtering**: Removes stopwords and low-quality terms
- **Metadata Enhancement**: Adds keywords to chunk metadata for search

## 🧪 Testing & Validation

### **Comprehensive Test Suite**
The system includes extensive testing capabilities:
- **End-to-End Testing**: `test_complete_rag_system.py` validates entire pipeline
- **Component Testing**: Individual module testing for each component
- **Integration Testing**: Cross-component functionality validation
- **Basic Performance Testing**: Processing time monitoring

### **Test Coverage**
- **Scraping Validation**: Verifies content extraction and image downloads
- **Processing Validation**: Tests chunking, summarization, and CSV conversion
- **Database Validation**: Confirms index building and search functionality
- **Error Handling**: Tests retry logic and fallback strategies
- **Basic Performance Validation**: Monitors processing time
- **Integration Testing**: Validates cross-component functionality
- **Edge Case Testing**: Handles unusual content and error conditions
- **Regression Testing**: Ensures new changes don't break existing functionality

### **Debug & Monitoring**
- **Detailed Logging**: Full audit trail with timestamps and severity levels
- **Progress Tracking**: Comprehensive logging with timing information
- **Performance Metrics**: Basic processing time tracking for each component
- **Debug Files**: Optional file saving for troubleshooting and analysis
- **Log Management**: Automatic log file creation and organization
- **Error Tracking**: Comprehensive error reporting with stack traces
- **Basic Resource Monitoring**: Processing time tracking
- **Debug Metadata**: Optional detailed metadata for development

## 🚀 Getting Started

1. **Quick Start**:
   ```bash
   git clone <repository-url>
   cd RAG-Piece
   pip install -r requirements.txt
   export OPENAI_API_KEY="your-api-key-here"
   python -m src.rag_piece.main
   ```

## 🔧 Troubleshooting & Support

### **Common Issues & Solutions**

#### **Image Download Problems**
- **Issue**: Images not downloading or wrong location
- **Solution**: Ensure `data/images/` directory exists and has write permissions
- **Check**: Verify images are in `data/images/[article_name]/` folders
- **Debug**: Check logs for download errors and retry attempts

#### **API Rate Limiting**
- **Issue**: "API request failed" errors
- **Solution**: Increase `API_BASE_DELAY` in config (default: 2.0 seconds)
- **Check**: Look for retry attempts in logs
- **Debug**: Monitor retry patterns and adjust delays accordingly

#### **Sub-Article Inconsistencies**
- **Issue**: Different sub-article counts between runs
- **Solution**: The new caching system should eliminate this (check logs for "Using cached sub-articles")
- **Check**: Verify both scrapers show identical counts in logs
- **Debug**: Enable verbose logging to see caching behavior

#### **Memory Issues with Large Articles**
- **Issue**: Out of memory during processing
- **Solution**: Reduce `MAX_CHUNK_SIZE` and `TARGET_CHUNK_SIZE` in config
- **Check**: Monitor memory usage in logs
- **Debug**: Enable debug metadata to track chunk sizes and memory usage

2. **Test Search**:
   ```python
   from src.rag_piece import RAGDatabase, RAGConfig
   config = RAGConfig()
   db = RAGDatabase(config)
   db.load_indices()
   results = db.search("What is Arabasta Kingdom?")
   print(f"Found {len(results)} results")
   ```

#### **Additional Troubleshooting Tips**

##### **OpenAI API Issues**
- **Issue**: Summarization or CSV conversion failures
- **Solution**: Verify `OPENAI_API_KEY` is set correctly
- **Check**: Test API key with simple OpenAI call
- **Debug**: Check API rate limits and usage quotas

##### **Search Quality Issues**
- **Issue**: Poor search results or irrelevant matches
- **Solution**: Adjust field boosting weights in configuration
- **Check**: Verify chunk overlap and metadata quality
- **Debug**: Enable debug logging to see search process

##### **Performance Issues**
- **Issue**: Slow processing or search times
- **Solution**: Reduce chunk sizes and adjust search parameters
- **Check**: Monitor processing time in logs
- **Debug**: Enable debug logging to see processing details

3. **Check Logs**:
   ```bash
   # View detailed logs
   ls logs/
   cat logs/rag_piece_*.log
   ```

4. **Advanced Configuration**:
   ```python
   from src.rag_piece import RAGConfig
   
   config = RAGConfig()
   
   # Chunking & Processing
   config.CHUNK_OVERLAP = 100                    # Increase overlap for better context
   config.MAX_CHUNK_SIZE = 500                    # Larger chunks for more context
   config.ENABLE_CSV_SCRAPING = True              # Enable CSV table extraction
   config.ENABLE_SUMMARIZATION = True             # Enable AI summarization
   config.SAVE_CSV_FILES_FOR_DEBUG = False       # Use in-memory processing
   
   # Search & Performance
   config.BM25_CANDIDATES = 150                   # More candidates for better recall
   config.FINAL_RESULTS = 15                      # More final results
   config.API_MAX_RETRIES = 5                     # More retry attempts
   config.API_BASE_DELAY = 3.0                    # Longer delays between retries
   
   # Logging & Debug
   config.LOG_LEVEL = "DEBUG"                     # Detailed logging
   config.VERBOSE_LOGGING = True                  # Progress tracking
   ```

## 🏗️ Architecture & Design

### **Modular Package Structure**
The system follows professional Python package conventions:
```
src/rag_piece/
├── __init__.py           # Package exports and initialization
├── main.py               # Application entry point and orchestration
├── config.py             # Centralized configuration with validation
├── database.py           # RAG database coordination and management
├── chunking.py           # Intelligent text chunking with overlap
├── keywords.py           # BM25-style keyword extraction
├── search.py             # Dual search engine (BM25 + semantic)
├── scraper.py            # Wiki content extraction and image management
├── summarizer.py         # AI-powered article summarization
├── csv_scraper.py        # Table extraction and CSV processing
├── csv_to_text.py        # LLM-powered CSV to text conversion
└── utils.py              # Shared utilities and logging setup
```

### **Design Principles**
- **Single Responsibility**: Each module has one clear purpose
- **Dependency Injection**: Configuration passed to components
- **Error Isolation**: Failures in one component don't affect others
- **Extensible Design**: Easy to add new features without modifying existing code
- **Clean Interfaces**: Well-defined APIs between components
- **Sequential Processing**: Operations run in sequence for reliability

## 🎉 What Makes This Special

### **Complete RAG Pipeline**
- **🔥 Zero Configuration**: Works out of the box with sensible defaults
- **🔄 End-to-End Processing**: From wiki scraping to search-ready database
- **🧠 Intelligent Content Processing**: LLM-powered summarization and CSV conversion
- **📊 Rich Data Integration**: Combines articles, tables, and images seamlessly
- **🔄 Sequential Processing**: Multiple operations run in sequence for reliability

### **Professional Architecture**
- **🏗️ Modular Design**: Clean separation of concerns with single-responsibility modules
- **📦 Package Structure**: Proper Python package organization in `src/rag_piece/`
- **🔧 Configuration Management**: Centralized configuration with validation
- **📝 Comprehensive Logging**: Full audit trail with both console and file output
- **🛡️ Error Handling**: Robust error handling with graceful fallbacks
- **🔄 Sequential Processing**: Reliable sequential execution of operations

### **Robust Testing & Validation**
- **🧪 Comprehensive Test Suite**: `test_complete_rag_system.py` validates entire pipeline
- **✅ Image Download Verification**: Confirms images are saved to correct locations
- **📊 Metadata Validation**: Ensures consistent data structure across components
- **🔄 End-to-End Testing**: Tests complete workflow from scraping to search
- **📝 Detailed Logging**: Full audit trail for debugging and monitoring
- **🔍 Component Testing**: Individual module testing for each component
- **⚡ Basic Performance Testing**: Processing time validation
- **🛡️ Error Testing**: Comprehensive error handling and recovery validation

### **Advanced Search Capabilities**
- **🧠 Intelligent Search**: Understands context and meaning, not just keywords
- **⚡ Fast Performance**: Sub-second search on comprehensive content
- **🎯 High Relevance**: Field boosting ensures the most relevant results first
- **🔍 Dual Search Strategy**: BM25 keyword + semantic similarity for optimal results
- **🔗 Context Preservation**: Overlapping chunks ensure no information is lost
- **🔄 Query Fallback**: Multiple strategies for handling complex queries
- **📊 Result Fusion**: Intelligent combination of keyword and semantic scores
- **🎯 Context-Aware Ranking**: Results ranked based on multiple relevance factors

### **Production-Ready Features**
- **🔧 Fully Configurable**: Every parameter can be tuned for your needs
- **📊 Rich Metadata**: Preserves document structure and relationships
- **🚀 Production Ready**: Robust error handling and fallback strategies
- **📝 Comprehensive Logging**: Full audit trail with progress tracking
- **🔄 Efficient Processing**: Avoids duplicate work through smart summary reuse
- **💾 Flexible Storage**: In-memory or file-based processing options
- **⚡ Performance Optimized**: Sequential processing and smart chunking
- **🛡️ Error Resilience**: Graceful degradation and error recovery
- **📈 Scalability**: Handles large articles and complex content efficiently
- **🔒 Data Integrity**: Basic validation and error handling throughout the pipeline

## 🚀 Future Roadmap & Extensibility

### **Planned Enhancements**
- **Multi-Wiki Support**: Extend to other MediaWiki sites beyond One Piece
- **Incremental Updates**: Add new articles without rebuilding entire database
- **Advanced AI Models**: Support for additional LLM providers and models
- **Enhanced Search**: More sophisticated ranking algorithms and filters
- **API Endpoints**: RESTful API for external integration
- **Web Interface**: User-friendly web-based search interface

### **Extensibility Features**
- **Modular Architecture**: Easy addition of new content processors
- **Configurable Chunking**: Adjustable chunking strategies via configuration
- **Extensible Metadata**: Custom metadata extraction and processing
- **Search Extensions**: Custom search algorithms and ranking methods
- **Export Formats**: Multiple output formats (JSON, XML, etc.)

### **Contributing**
The system is designed to be easily extensible:
- **Clean Architecture**: Well-defined interfaces between components
- **Configuration-Driven**: Most behavior controlled via configuration
- **Modular Design**: Easy to add new features without breaking existing code
- **Comprehensive Testing**: Test suite ensures changes don't break functionality
- **Sequential Processing**: Reliable execution model for new features

---

Transform your One Piece research with the power of modern RAG technology! 🏴‍☠️

**Ready to build your own RAG system?** Start with this comprehensive foundation and extend it to meet your specific needs!
