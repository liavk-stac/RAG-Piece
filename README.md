# One Piece Wiki RAG Database System

A comprehensive **Retrieval-Augmented Generation (RAG)** system that scrapes One Piece Wiki content and creates an intelligent, searchable database using both keyword (BM25) and semantic search technologies.

## 🚀 Key Features

### **Integrated Pipeline**
- **One Command Operation**: Scraping → Chunking → Database Creation → Search Ready
- **No Intermediate Files**: Direct processing from web to searchable database
- **Smart Content Processing**: Hybrid chunking with paragraph-based splitting
- **Automatic Image Management**: High-quality image downloads with filtering

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
   python one_piece_scraper.py
   ```

## 🎯 Usage

### **Single Command Operation**
```bash
python one_piece_scraper.py
```

This command will:
1. **Scrape** One Piece Wiki articles
2. **Process** content into intelligent chunks
3. **Build** BM25 and semantic search indices
4. **Test** the search functionality
5. **Save** high-quality images organized by article

### **Search the Database**
```python
from db_creator import RAGDatabase, RAGConfig

# Initialize database
db = RAGDatabase(RAGConfig())
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

## 🗂️ Output Structure

```
RAG-Piece/
├── one_piece_scraper.py          # Main integrated scraper + RAG system
├── db_creator.py                 # RAG database components and search logic
├── requirements.txt              # Python dependencies
├── SCRAPER_DOCUMENTATION.md      # Detailed technical documentation
├── data/
│   ├── images/                   # Downloaded images organized by article
│   │   └── Arabasta_Kingdom/     # Article-specific image folders
│   └── rag_db/                   # RAG database files
│       ├── whoosh_index/         # BM25 keyword search index
│       ├── faiss_index.bin       # Semantic search index
│       ├── chunk_mapping.pkl     # Mapping between indices
│       └── database_metadata.json # Database statistics and info
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

## 🚀 Getting Started

1. **Quick Start**:
   ```bash
   git clone <repository-url>
   cd RAG-Piece
   pip install -r requirements.txt
   python one_piece_scraper.py
   ```

2. **Test Search**:
   ```python
   from db_creator import RAGDatabase, RAGConfig
   db = RAGDatabase(RAGConfig())
   db.load_indices()
   results = db.search("What is Arabasta Kingdom?")
   print(f"Found {len(results)} results")
   ```

## 🎉 What Makes This Special

- **🔥 Zero Configuration**: Works out of the box with sensible defaults
- **🧠 Intelligent Search**: Understands context and meaning, not just keywords
- **⚡ Fast Performance**: Sub-second search on comprehensive content
- **🎯 High Relevance**: Field boosting ensures the most relevant results first
- **🔧 Fully Configurable**: Every parameter can be tuned for your needs
- **📊 Rich Metadata**: Preserves document structure and relationships
- **🚀 Production Ready**: Robust error handling and fallback strategies

Transform your One Piece research with the power of modern RAG technology! 🏴‍☠️
