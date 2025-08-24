# One Piece Wiki RAG Database System

A comprehensive **Retrieval-Augmented Generation (RAG)** system that scrapes One Piece Wiki content and creates an intelligent, searchable database using both keyword (BM25) and semantic search technologies.

## 🚀 Key Features

### **Complete Database Construction Pipeline**
- **End-to-End Processing**: Scraping → Summarization → CSV Conversion → Chunking → Database → Search Ready
- **Intelligent Content Processing**: Hybrid chunking with paragraph-based splitting and metadata enrichment
- **CSV Integration**: Automatic conversion of wiki tables to searchable text using LLM
- **Duplicate Prevention**: Smart summarization that runs once and reuses summaries across components
- **Automatic Image Management**: High-quality image downloads with filtering and organization

### **Advanced Search Capabilities**
- **Two-Step Retrieval**: BM25 keyword search + semantic similarity ranking
- **Natural Language Queries**: Ask questions like "What is Arabasta Kingdom?"
- **Field Boosting**: Weighted search across titles, sections, and content
- **Query Fallback**: Robust handling of complex questions with multiple strategies

### **Intelligent Content Organization**
- **Metadata-Rich Chunks**: Article names, sections, sub-sections, and keywords
- **Hierarchical Search**: Article titles > Sub-articles > Sections > Sub-sections > Keywords > Content
- **Context Preservation**: Maintains document structure and relationships

## 🛠️ Technology Stack

- **Web Scraping**: MediaWiki API integration
- **Keyword Search**: Whoosh with BM25F scoring and field boosting
- **Semantic Search**: FAISS with sentence-transformers embeddings
- **Text Processing**: NLTK for tokenization, keyword extraction, and analysis
- **Image Processing**: PIL for validation and quality filtering

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

3. **Run the system**
   ```bash
   python main.py
   ```

## 🎯 Usage

### **Complete Database Construction Process**
```bash
python main.py
```

This command orchestrates the entire pipeline:

#### **Phase 1: Data Collection & Scraping**
1. **Web Scraping**: Scrapes One Piece Wiki articles using MediaWiki API
2. **Content Extraction**: Extracts HTML sections, headings, and content
3. **Image Download**: Downloads high-quality images to `images/[article_name]/`
4. **CSV Generation**: Converts wiki tables to CSV files in `csv_files/[article_name]/`

#### **Phase 2: Content Processing & Summarization**
5. **Article Summarization**: Creates comprehensive summaries using OpenAI GPT-4o-mini
6. **CSV to Text Conversion**: Converts CSV data to structured, searchable text
7. **Summary Reuse**: Avoids duplicate summarizer runs by sharing summaries across components

#### **Phase 3: Content Chunking & Organization**
8. **Hybrid Chunking**: Breaks content into optimized chunks using paragraph-based splitting
9. **Metadata Enrichment**: Adds rich metadata (article, section, sub-sections, keywords)
10. **Context Preservation**: Maintains relationships between data points

#### **Phase 4: Database Construction & Storage**
11. **Index Building**: Creates BM25 keyword search and semantic search indices
12. **Vector Embeddings**: Generates embeddings for semantic similarity search
13. **Database Assembly**: Combines all processed content into searchable database

#### **Phase 5: Testing & Validation**
14. **Search Testing**: Validates database functionality with sample queries
15. **Logging**: Records all operations to `logs/` directory

### **Search the Database**
```python
from rag_piece import RAGDatabase, RAGConfig

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
from rag_piece import OneWikiScraper, TextChunker, RAGConfig

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
├── main.py                       # Main entry point
├── src/                          # Source code (follows cursor guidelines)
│   └── rag_piece/                # Main package
│       ├── __init__.py           # Package initialization
│       ├── main.py               # Application main function
│       ├── config.py             # Configuration settings
│       ├── database.py           # RAG database coordination
│       ├── chunking.py           # Text chunking functionality
│       ├── keywords.py           # Keyword extraction
│       ├── search.py             # BM25 and semantic search
│       ├── scraper.py            # Wiki scraping functionality
│       ├── summarizer.py         # Article summarization
│       ├── csv_to_text.py        # CSV to text conversion
│       └── utils.py              # Utilities and logging
├── logs/                         # Application logs
├── requirements.txt              # Python dependencies
├── docs/                         # Comprehensive documentation
│   ├── CSV_TO_TEXT_README.md     # CSV conversion guide
│   ├── CHUNKING_AND_METADATA_EXPLANATION.md # Chunking system guide
│   └── SCRAPER_DOCUMENTATION.md  # Technical documentation
├── images/                       # Downloaded images organized by article
│   └── [article_name]/           # Article-specific image folders
├── csv_files/                    # CSV data from wiki tables
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
├── examples/                     # Usage examples
│   └── csv_to_text_example.py   # CSV conversion example
└── test/                         # Test suite
    └── test_csv_to_text.py      # CSV conversion tests
└── [legacy files for compatibility]
```

## ⚙️ Configuration

All parameters are centralized in the `RAGConfig` class for easy tuning:

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

## 📚 Documentation

- **[SCRAPER_DOCUMENTATION.md](SCRAPER_DOCUMENTATION.md)**: Comprehensive technical documentation
- **Code Comments**: Detailed inline documentation
- **Configuration Guide**: All parameters explained in `RAGConfig` class

## 🔄 Complete Database Construction Process

### **Data Flow Overview**
```
One Piece Wiki → Scraping → HTML Content + Images + CSV Tables
     ↓
HTML Content → Summarization → Article Summaries
     ↓
CSV Tables → LLM Conversion → Structured Text
     ↓
All Text → Hybrid Chunking → Optimized Chunks + Rich Metadata
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
- **Output**: Summary chunks with context
- **Efficiency**: Runs once per article, summaries reused across components

#### **3. CSVToTextConverter**
- **Purpose**: Converts wiki tables to searchable text
- **Input**: CSV files + existing article summaries
- **Output**: Structured text maintaining data relationships
- **Debug**: Saves conversion results for quality verification

#### **4. TextChunker**
- **Purpose**: Breaks content into optimal chunks for vector embedding
- **Strategy**: Hybrid approach (paragraph → merge → split)
- **Metadata**: Enriches each chunk with search context
- **Context**: Maintains relationships between chunks

#### **5. RAGDatabase**
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
   python main.py
   ```

2. **Test Search**:
   ```python
   from rag_piece import RAGDatabase, RAGConfig
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

## 🎉 What Makes This Special

### **Complete RAG Pipeline**
- **🔥 Zero Configuration**: Works out of the box with sensible defaults
- **🔄 End-to-End Processing**: From wiki scraping to search-ready database
- **🧠 Intelligent Content Processing**: LLM-powered summarization and CSV conversion
- **📊 Rich Data Integration**: Combines articles, tables, and images seamlessly

### **Advanced Search Capabilities**
- **🧠 Intelligent Search**: Understands context and meaning, not just keywords
- **⚡ Fast Performance**: Sub-second search on comprehensive content
- **🎯 High Relevance**: Field boosting ensures the most relevant results first
- **🔍 Dual Search Strategy**: BM25 keyword + semantic similarity for optimal results

### **Production Features**
- **🔧 Fully Configurable**: Every parameter can be tuned for your needs
- **📊 Rich Metadata**: Preserves document structure and relationships
- **🚀 Production Ready**: Robust error handling and fallback strategies
- **📝 Comprehensive Logging**: Full audit trail of all operations
- **🔄 Efficient Processing**: Avoids duplicate work through smart summary reuse

Transform your One Piece research with the power of modern RAG technology! 🏴‍☠️
