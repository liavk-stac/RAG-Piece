# One Piece Wiki RAG Database System

A comprehensive **Retrieval-Augmented Generation (RAG)** system that scrapes One Piece Wiki content and creates an intelligent, searchable database using both keyword (BM25) and semantic search technologies.

## 🚀 Key Features

### **Complete Database Construction Pipeline**
- **End-to-End Processing**: Scraping → Summarization → CSV Conversion → Chunking → Database → Search Ready
- **Intelligent Content Processing**: Hybrid chunking with **overlapping chunks** for better context preservation
- **CSV Integration**: Automatic conversion of wiki tables to searchable text using LLM (with optional in-memory processing)
- **Parallel Processing**: Content chunking, summarization, and CSV conversion run simultaneously for efficiency
- **Duplicate Prevention**: Smart summarization that runs once and reuses summaries across components
- **Automatic Image Management**: High-quality image downloads with filtering and organization

### **Advanced Search Capabilities**
- **Two-Step Retrieval**: BM25 keyword search + semantic similarity ranking
- **Natural Language Queries**: Ask questions like "What is Arabasta Kingdom?"
- **Field Boosting**: Weighted search across titles, sections, and content
- **Query Fallback**: Robust handling of complex questions with multiple strategies
- **Context Preservation**: Overlapping chunks ensure important information isn't lost at boundaries

### **Intelligent Content Organization**
- **Metadata-Rich Chunks**: Article names, sections, sub-sections, and keywords
- **Hierarchical Search**: Article titles > Sub-articles > Sections > Sub-sections > Keywords > Content
- **Context Preservation**: Maintains document structure and relationships with smart overlap

## 🛠️ Technology Stack

- **Web Scraping**: MediaWiki API integration
- **Keyword Search**: Whoosh with BM25F scoring and field boosting
- **Semantic Search**: FAISS with sentence-transformers embeddings
- **Text Processing**: NLTK for tokenization, keyword extraction, and analysis
- **Image Processing**: PIL for validation and quality filtering
- **LLM Integration**: OpenAI GPT-4o-mini for summarization and CSV conversion

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
3. **Image Download**: Downloads high-quality images to `images/[article_name]/`
4. **CSV Generation**: Converts wiki tables to CSV files or in-memory DataFrames

#### **Phase 2: Content Processing & Summarization (Parallel)**
5. **Article Summarization**: Creates comprehensive summaries using OpenAI GPT-4o-mini
6. **CSV to Text Conversion**: Converts CSV data to structured, searchable text using LLM
7. **Summary Reuse**: Avoids duplicate summarizer runs by sharing summaries across components

#### **Phase 3: Content Chunking & Organization**
8. **Hybrid Chunking**: Breaks content into optimized chunks using paragraph-based splitting
9. **Overlapping Chunks**: Creates overlapping chunks for better context preservation and search quality
10. **Metadata Enrichment**: Adds rich metadata (article, section, sub-sections, keywords)
11. **Context Preservation**: Maintains relationships between data points

#### **Phase 4: Database Construction & Storage**
12. **Index Building**: Creates BM25 keyword search and semantic search indices
13. **Vector Embeddings**: Generates embeddings for semantic similarity search
14. **Database Assembly**: Combines all processed content into searchable database

#### **Phase 5: Testing & Validation**
15. **Search Testing**: Validates database functionality with sample queries
16. **Logging**: Records all operations with detailed progress tracking

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
├── images/                       # Downloaded images organized by article
│   └── [article_name]/           # Article-specific image folders
├── csv_files/                    # CSV data from wiki tables (optional)
│   └── [article_name]/           # Organized by article
├── data/                         # Processed data
│   ├── summaries/                # Article summaries (if enabled)
│   ├── debug/                    # Debug files
│   │   └── csv2text/             # CSV conversion debug files
│   └── rag_db/                   # RAG database files
│       ├── whoosh_index/         # BM25 keyword search index
│       ├── faiss_index.bin       # Semantic search index
│       ├── chunk_mapping.pkl     # Mapping between indices
│       └── database_metadata.json # Database statistics and info
└── [legacy files for compatibility]
```

## ⚙️ Configuration

All parameters are centralized in the `RAGConfig` class for easy tuning:

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
```

### **Articles Configuration**
```python
# === ARTICLES TO SCRAPE ===
ARTICLES_TO_SCRAPE: List[str] = ["Arabasta Kingdom"]  # configurable article list
```

### **Field Boosting Hierarchy**
1. **Article Names** (3.0x boost) - Main article titles
2. **Sub-Article Names** (2.8x boost) - Sub-article/redirect names
3. **Section Names** (2.5x boost) - Section titles from original structure
4. **Sub-Section Names** (2.0x boost) - `<h3>` headings within content
5. **Keywords** (1.8x boost) - BM25-extracted important terms
6. **Content** (1.0x boost) - Main text content

## 🔍 Search Examples

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

### **Overlapping Chunks (NEW!)**
- **Better Context Preservation**: 50-token overlap between chunks prevents information loss
- **Improved Search Quality**: Queries can find content that spans chunk boundaries
- **Semantic Coherence**: Chunks maintain better meaning across boundaries
- **Configurable**: Can be disabled by setting `CHUNK_OVERLAP = 0`

### **In-Memory CSV Processing**
- **Eliminates File I/O**: CSV data processed directly in memory for efficiency
- **Debug Mode**: Optional file saving for troubleshooting
- **Parallel Processing**: CSV conversion runs alongside content processing

### **Enhanced Progress Tracking**
- **Detailed Logging**: Progress bars and timing information for long operations
- **Better Error Handling**: Robust retry mechanisms and fallback strategies
- **Performance Monitoring**: Track processing time for each component

### **Centralized Configuration**
- **Single Source of Truth**: All parameters in `config.py`
- **Easy Tuning**: Modify chunk sizes, overlap, and processing options
- **Validation**: Automatic validation of configuration parameters

## 📚 Documentation

- **[SCRAPER_DOCUMENTATION.md](docs/SCRAPER_DOCUMENTATION.md)**: Comprehensive technical documentation
- **[CHUNKING_AND_METADATA_EXPLANATION.md](docs/CHUNKING_AND_METADATA_EXPLANATION.md)**: Detailed chunking system guide
- **[INTEGRATION_SUMMARY.md](docs/INTEGRATION_SUMMARY.md)**: System integration overview
- **Code Comments**: Detailed inline documentation
- **Configuration Guide**: All parameters explained in `RAGConfig` class

## 🔄 Complete Database Construction Process

### **Data Flow Overview**
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

## 🚀 Getting Started

1. **Quick Start**:
   ```bash
   git clone <repository-url>
   cd RAG-Piece
   pip install -r requirements.txt
   export OPENAI_API_KEY="your-api-key-here"
   python -m src.rag_piece.main
   ```

2. **Test Search**:
   ```python
   from src.rag_piece import RAGDatabase, RAGConfig
   config = RAGConfig()
   db = RAGDatabase(config)
   db.load_indices()
   results = db.search("What is Arabasta Kingdom?")
   print(f"Found {len(results)} results")
   ```

3. **Check Logs**:
   ```bash
   # View detailed logs
   ls logs/
   cat logs/rag_piece_*.log
   ```

4. **Customize Configuration**:
   ```python
   from src.rag_piece import RAGConfig
   
   config = RAGConfig()
   config.CHUNK_OVERLAP = 100        # Increase overlap for better context
   config.ENABLE_CSV_SCRAPING = True  # Enable CSV processing
   config.SAVE_CSV_FILES_FOR_DEBUG = False  # Use in-memory processing
   ```

## 🎉 What Makes This Special

### **Complete RAG Pipeline**
- **🔥 Zero Configuration**: Works out of the box with sensible defaults
- **🔄 End-to-End Processing**: From wiki scraping to search-ready database
- **🧠 Intelligent Content Processing**: LLM-powered summarization and CSV conversion
- **📊 Rich Data Integration**: Combines articles, tables, and images seamlessly
- **🔄 Parallel Processing**: Multiple operations run simultaneously for efficiency

### **Advanced Search Capabilities**
- **🧠 Intelligent Search**: Understands context and meaning, not just keywords
- **⚡ Fast Performance**: Sub-second search on comprehensive content
- **🎯 High Relevance**: Field boosting ensures the most relevant results first
- **🔍 Dual Search Strategy**: BM25 keyword + semantic similarity for optimal results
- **🔗 Context Preservation**: Overlapping chunks ensure no information is lost

### **Production Features**
- **🔧 Fully Configurable**: Every parameter can be tuned for your needs
- **📊 Rich Metadata**: Preserves document structure and relationships
- **🚀 Production Ready**: Robust error handling and fallback strategies
- **📝 Comprehensive Logging**: Full audit trail with progress tracking
- **🔄 Efficient Processing**: Avoids duplicate work through smart summary reuse
- **💾 Flexible Storage**: In-memory or file-based processing options
- **⚡ Performance Optimized**: Parallel processing and smart chunking

Transform your One Piece research with the power of modern RAG technology! 🏴‍☠️
