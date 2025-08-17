# One Piece Wiki Scraper + RAG Database System - Technical Documentation

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [RAG Database System](#rag-database-system)
5. [Chunking Strategy](#chunking-strategy)
6. [Search Functionality](#search-functionality)
7. [Configuration System](#configuration-system)
8. [File Structure](#file-structure)
9. [API Integration](#api-integration)
10. [Performance Considerations](#performance-considerations)
11. [Usage Examples](#usage-examples)

## Overview

The One Piece Wiki Scraper + RAG Database System is an integrated solution that combines web scraping with advanced Retrieval-Augmented Generation (RAG) capabilities. The system scrapes content from the One Piece Wiki and immediately processes it into a searchable database using both keyword (BM25) and semantic search technologies.

### Key Capabilities
- **Direct Processing Pipeline**: Web scraping → Chunking → Database indexing (no intermediate files)
- **Hybrid Search**: BM25 keyword search + semantic search with score fusion
- **Intelligent Chunking**: Paragraph-based chunking with smart merging/splitting
- **Natural Language Queries**: Robust handling of conversational questions
- **Field Boosting**: Weighted search across content, titles, and metadata
- **Image Management**: Automated image downloading with quality filtering

### Technology Stack
- **Web Scraping**: MediaWiki API integration
- **Keyword Search**: Whoosh with BM25F scoring
- **Semantic Search**: FAISS with sentence-transformers
- **Text Processing**: NLTK for tokenization and analysis
- **Image Processing**: PIL for validation and processing

## System Architecture

### Integrated Workflow
```
Web Scraping → Content Extraction → Hybrid Chunking → RAG Database → Search Interface
```

### File Structure
```
RAG-Piece/
├── one_piece_scraper.py          # Main integrated scraper + RAG system
├── db_creator.py                 # RAG database components and search logic
├── requirements.txt              # Python dependencies
├── SCRAPER_DOCUMENTATION.md      # This technical documentation
├── data/
│   ├── images/                   # Downloaded images organized by article
│   │   └── Arabasta_Kingdom/     # Article-specific image folders
│   └── rag_db/                   # RAG database files
│       ├── whoosh_index/         # BM25 keyword search index
│       ├── faiss_index.bin       # Semantic search index
│       ├── chunk_mapping.pkl     # Mapping between indices
│       └── database_metadata.json # Database statistics and info
```

### Dependencies
```python
# Core dependencies
requests>=2.31.0                  # HTTP requests for API calls and images
Pillow>=10.0.0                   # Image processing and validation
whoosh>=2.7.4                    # BM25 keyword search engine
faiss-cpu>=1.7.4                 # Semantic search with vector similarity
sentence-transformers==2.2.2     # Text embedding generation
huggingface-hub>=0.10.0,<0.16.0  # Model management
numpy>=1.24.0                    # Numerical operations
torch>=2.0.0                     # Deep learning backend
nltk>=3.8.1                      # Natural language processing
```

## Major System Changes (Latest Update)

### 🔄 **Complete Integration Transformation**
The system has been completely transformed from a file-based scraper to an integrated RAG database system:

#### **Previous System** (File-Based)
- ❌ Scraped content → Saved `.txt` files → Read files → Process to database
- ❌ Saved CSV tables separately
- ❌ Basic file organization
- ❌ Manual database creation step

#### **New System** (Integrated RAG)
- ✅ **Direct Pipeline**: Scraped content → Chunks → Database (no intermediate files)
- ✅ **Two-Step Search**: BM25 keyword + Semantic search with score fusion
- ✅ **Smart Chunking**: Paragraph-based with intelligent merging/splitting
- ✅ **Natural Language Queries**: "What is Arabasta Kingdom?" works perfectly
- ✅ **Field Boosting**: Weighted search across content, titles, metadata
- ✅ **Robust Query Handling**: Fallback strategies for complex questions

### 🏗️ **New Architecture Components**

#### **RAG Database System** (`db_creator.py`)
- **RAGConfig**: Centralized configuration for all parameters
- **TextChunker**: Hybrid chunking with paragraph-based splitting
- **KeywordExtractor**: BM25-style keyword extraction
- **RAGDatabase**: Main database class with two-step search

#### **Integration Layer** (`one_piece_scraper.py`)
- **Direct Processing**: `scrape_article()` now processes content directly to chunks
- **No File I/O**: Text content never saved to disk (only images)
- **Immediate Indexing**: Database built immediately after scraping
- **Search Testing**: Automatic validation with sample queries

### 🔍 **Search Capabilities**

#### **Supported Query Types**
```python
# Simple keywords
results = db.search("Arabasta")

# Natural language questions  
results = db.search("What is Arabasta Kingdom?")

# Descriptive queries
results = db.search("Tell me about the desert in Arabasta")

# Character questions
results = db.search("Who are the main characters in Arabasta?")
```

#### **Two-Step Search Process**
1. **BM25 Keyword Search**: Fast retrieval of 100 candidates with field boosting
2. **Semantic Reranking**: FAISS similarity scoring with score fusion
3. **Final Results**: Top 10 results with combined BM25 + semantic scores

#### **Fallback Query Strategies**
- **Strategy 1**: Try original query directly
- **Strategy 2**: Remove stop words ("What is" → "Arabasta Kingdom")  
- **Strategy 3**: OR search with important terms ("Arabasta OR Kingdom")

### 📊 **Metadata Structure**

#### **Search Metadata** (Used in Search)
```python
{
    "article_name": "Arabasta Kingdom",           # From scraper (SEARCHABLE - 3.0x boost)
    "sub_article_name": None,                     # Sub-article if any (SEARCHABLE - 2.8x boost)
    "section_name": "General Information",        # From filename structure (SEARCHABLE - 2.5x boost)
    "sub_section_name": "Geography; Climate",     # <h3> tags in chunk (SEARCHABLE - 2.0x boost)
    "keywords": ["arabasta", "kingdom", "desert"] # BM25-extracted (SEARCHABLE - 1.8x boost)
}
```

**All metadata fields are searchable** and contribute to BM25 scoring with different boost weights. Content text has a 1.0x base boost.

#### **Chunk Processing Flow**
```
Raw HTML → Sections → Paragraphs → Size Check → Merge/Split → Extract H3 → Keywords → Index
```

### ⚙️ **Configuration System**
All parameters centralized in `RAGConfig` class:

```python
# Chunking parameters
MIN_CHUNK_SIZE = 100      # tokens - merge if below
MAX_CHUNK_SIZE = 400      # tokens - split if above  
TARGET_CHUNK_SIZE = 250   # tokens - ideal size
OVERLAP_SIZE = 50         # tokens - overlap when splitting

# Search parameters
BM25_CANDIDATES = 100     # candidates from BM25 step
FINAL_RESULTS = 10        # final results after reranking
RERANK_WEIGHT_BM25 = 0.4  # BM25 weight in fusion
RERANK_WEIGHT_SEMANTIC = 0.6  # semantic weight in fusion

# Field boosting weights
CONTENT_BOOST = 1.0             # base content
ARTICLE_NAME_BOOST = 3.0        # article mentions (highest priority)
SUB_ARTICLE_NAME_BOOST = 2.8    # sub-article names
SECTION_NAME_BOOST = 2.5        # section titles
SUB_SECTION_BOOST = 2.0         # sub-section titles (from <h3> tags)
KEYWORDS_BOOST = 1.8            # extracted keywords
```

### 🎯 **Usage Examples**

#### **Single Command Operation**
```bash
# Complete pipeline: scrape → chunk → index → test
python one_piece_scraper.py
```

#### **Programmatic Usage**
```python
from db_creator import RAGDatabase, RAGConfig

# Load existing database
db = RAGDatabase(RAGConfig())
db.load_indices()

# Search with natural language
results = db.search("What is Arabasta Kingdom?", top_k=5)

# Process results
for result in results:
    print(f"Section: {result['search_metadata']['section_name']}")
    print(f"Combined Score: {result['combined_score']:.3f}")
    print(f"Content: {result['content'][:100]}...")
```

### 📁 **File Structure Changes**
```
# OLD: Many intermediate files
data/Arabasta_Kingdom/
├── 01_General_Information.txt    # ❌ No longer created
├── 02_Kingdom_Information.txt    # ❌ No longer created  
├── Table.csv                     # ❌ No longer created
└── metadata.json                 # ❌ No longer created

# NEW: Clean, organized structure
data/
├── images/Arabasta_Kingdom/      # ✅ Images only
│   ├── Arabasta_Kingdom.png
│   └── Alubarna_Infobox.png
└── rag_db/                       # ✅ Database files
    ├── whoosh_index/             # BM25 search
    ├── faiss_index.bin           # Semantic search
    └── database_metadata.json    # Statistics
```

## Core Functions

### 1. `slugify(text)`
**Purpose**: Converts text to filesystem-safe strings
**Input**: Raw text string
**Output**: Clean, filesystem-safe string
**Process**:
1. Normalizes Unicode characters using NFKD
2. Removes non-alphanumeric characters (except spaces, hyphens, underscores)
3. Replaces multiple spaces/hyphens with single underscores
4. Strips leading/trailing underscores

**Example**:
```python
slugify("Monkey D. Luffy's Devil Fruit") → "Monkey_D_Luffys_Devil_Fruit"
slugify("Roronoa Zoro (三刀流)") → "Roronoa_Zoro"
```

### 2. `fetch_wiki_content(article_name)`
**Purpose**: Fetches article content from One Piece Wiki API
**Input**: Article name string
**Output**: Parsed JSON response or None
**API Endpoint**: `https://onepiece.fandom.com/api.php`
**Parameters**:
- `action=parse`: Parse wiki page
- `page`: Article name
- `format=json`: JSON response format
- `prop=text`: Extract page text content

**Error Handling**: Returns None on network/API failures
**Rate Limiting**: None (handled by caller)

### 3. `clean_text(text)`
**Purpose**: Removes wiki artifacts and HTML formatting
**Input**: Raw HTML text
**Output**: Clean, readable text
**Cleaning Process**:
1. **HTML Tags**: Removes all `<tag>` elements
2. **Citations**: Removes `[1]`, `[2]`, `[42]` patterns
3. **Wiki Artifacts**: Removes `[edit]`, `[citation needed]`, `[who?]`, etc.
4. **MediaWiki Templates**: Removes `{{template}}` and `{|table|}` patterns
5. **External Links**: Converts `[text](url)` to just `text`
6. **Whitespace**: Normalizes multiple spaces to single spaces

**Regex Patterns Used**:
```python
r'<[^>]+>'                    # HTML tags
r'\[\d+\]'                    # Citation numbers
r'\[edit\]|\[citation needed\]|\[who\?\]|\[when\?\]|\[where\?\]|\[clarification needed\]'  # Wiki artifacts
r'\{\{[^}]+\}\}|\{\|[^}]+\|\}'  # MediaWiki templates
r'\[([^\]]+)\]\([^)]+\)'      # External links
r'\s+'                        # Multiple whitespace
```

### 4. `extract_sections(content_html)`
**Purpose**: Extracts hierarchical sections from wiki content
**Input**: HTML content string
**Output**: List of section dictionaries with titles and content
**Section Types Detected**:
1. **MediaWiki Sections**: `<span class="mw-headline">` elements
2. **HTML Sections**: `<h2>` elements

**Section Processing**:
1. **Title Cleaning**: Applies `clean_title()` function
2. **Validation**: Checks title length (2-100 characters) and uniqueness
3. **Boundary Detection**: Finds start/end positions between sections
4. **Content Extraction**: Extracts text between section boundaries
5. **Filtering**: Skips navigation, references, external links sections

**Output Structure**:
```python
[
    {
        'combined_title': 'Introduction',
        'content': 'Clean text content...'
    },
    {
        'combined_title': 'Appearance',
        'content': 'Clean text content...'
    }
]
```

**Navigation Sections Skipped**:
- `references`, `site navigation`, `external links`, `notes`

### 5. `extract_tables(content_html)`
**Purpose**: Extracts HTML tables and converts to CSV format
**Input**: HTML content string
**Output**: List of table dictionaries with data and labels
**Table Detection**:
- **Pattern**: `<table>` elements with optional `<caption>`
- **Row Extraction**: `<tr>` elements
- **Cell Extraction**: `<td>` and `<th>` elements

**Label Generation**:
1. **Caption Priority**: Uses table caption if available
2. **Header Row Fallback**: Uses first row content if caption missing
3. **Default Label**: "Table" if no meaningful label found

**Label Cleaning**:
- Applies `clean_text()` function
- Removes problematic characters
- Limits length to 50 characters
- Ensures minimum 3 characters

**Output Structure**:
```python
[
    {
        'data': [['Header1', 'Header2'], ['Row1Col1', 'Row1Col2']],
        'label': 'Table Name'
    }
]
```

### 6. `extract_images(content_html, max_img=20)`
**Purpose**: Extracts image URLs and metadata from wiki content
**Input**: HTML content string, maximum image count
**Output**: List of image dictionaries with URLs and labels
**Image Detection**:
- **Pattern**: `<img>` tags with `src` attributes
- **Domain Validation**: Only processes wiki image domains
- **Count Limiting**: Stops after reaching `max_img` limit

**Supported Image Domains**:
- `static.wikia.nocookie.net`
- `vignette.wikia.nocookie.net`
- `static.wikimedia.org`
- `upload.wikimedia.org`

**Label Generation**:
1. **Alt Text Priority**: Uses `alt` attribute if available
2. **URL Fallback**: Generates label from URL filename
3. **Numbering**: Adds sequential numbers for uniqueness

**Label Cleaning**:
- Applies `clean_text()` function
- Removes problematic characters
- Limits length to 50 characters
- Ensures minimum 3 characters

**Output Structure**:
```python
[
    {
        'url': 'https://static.wikia.nocookie.net/...',
        'label': 'Monkey_D_Luffy_Profile',
        'width': None,    # Set during download
        'height': None    # Set during download
    }
]
```

### 7. `find_sub_articles(article_name)`
**Purpose**: Discovers related sub-articles for comprehensive coverage
**Input**: Main article name
**Output**: List of sub-article titles
**Discovery Methods**:
1. **API Search**: Queries MediaWiki API for articles starting with "Article_Name/"
2. **Common Patterns**: Checks for standard sub-article names

**Common Sub-Article Patterns**:
- `{Article}/Gallery`
- `{Article}/Images`
- `{Article}/Pictures`
- `{Article}/Screenshots`
- `{Article}/Artwork`

**API Search Parameters**:
- `action=query`
- `list=search`
- `srsearch`: `"{article_name}/"`
- `srnamespace=0` (main namespace only)
- `srlimit=50`

**Deduplication**: Removes duplicate titles while preserving order

### 8. `scrape_article(article_name, scrape_csv_files=True, max_img=20)`
**Purpose**: Main function that orchestrates the entire scraping process
**Input**: Article name, CSV flag, image limit
**Output**: Boolean success indicator
**Process Flow**:
1. **Setup**: Creates output directory, initializes collections
2. **Main Article**: Fetches and processes main article content
3. **Sub-Articles**: Discovers and processes related sub-articles
4. **Content Merging**: Combines all content with duplicate prevention
5. **File Saving**: Saves sections, tables (if enabled), and images
6. **Metadata**: Generates comprehensive metadata file

**Content Collections**:
- `all_sections`: Combined sections from main and sub-articles
- `all_tables`: Combined tables from main and sub-articles
- `all_images`: Combined images from main and sub-articles
- `created_files`: List of all saved files

**Duplicate Prevention**:
- **Section Keys**: `main_{title}` or `sub_{article}_{title}`
- **Table Keys**: `main_{label}` or `sub_{article}_{label}`
- **Image Keys**: `main_{label}` or `sub_{article}_{label}`

**File Saving Process**:
1. **Sections**: Numbered text files (01_Introduction.txt, 02_Appearance.txt)
2. **Tables**: CSV files with descriptive names (if CSV extraction enabled)
3. **Images**: PNG files with meaningful labels and size validation
4. **Metadata**: JSON file with comprehensive scraping summary

### 9. `main()`
**Purpose**: Entry point and configuration management
**Configuration Variables**:
- `SCRAPE_CSV_FILES`: Boolean flag for CSV table extraction
- `MAX_IMAGES`: Integer limit for images per article
- `articles`: List of articles to scrape

**Process Flow**:
1. **Configuration Display**: Shows current settings
2. **Data Cleanup**: Removes previous data folder
3. **Article Processing**: Iterates through article list
4. **Summary**: Displays success/failure counts

## File Naming Conventions

### Section Files
**Format**: `{number:02d}_{slugified_title}.txt`
**Examples**:
- `01_Introduction.txt`
- `02_Appearance.txt`
- `03_Personality.txt`
- `04_Abilities_and_Powers.txt`

**Numbering**: Sequential numbering starting from 01
**Title Processing**: Applied `slugify()` function for filesystem safety

### Table Files
**Format**: `{slugified_label}.csv`
**Examples**:
- `Baratie.csv`
- `Devil_Fruit_Powers.csv`
- `Bounty_History.csv`
- `Table.csv` (default label)

**Label Processing**: Applied `clean_text()` and character filtering
**Length Limit**: Maximum 50 characters

### Image Files
**Format**: `{slugified_label}.png`
**Examples**:
- `Monkey_D_Luffy_Profile.png`
- `Gear_Forms.png`
- `Battle_Scenes.png`
- `Image_1.png` (fallback label)

**Label Sources**:
1. **Alt Text**: Primary source for meaningful names
2. **URL Filename**: Fallback when alt text unavailable
3. **Sequential Numbering**: Ensures uniqueness

**Size Validation**: Only saves images ≥100x100 pixels
**Format**: Always saved as PNG regardless of source format

### Metadata File
**Format**: `metadata.json`
**Content Structure**:
```json
{
    "article_name": "Monkey D. Luffy",
    "article_url": "https://onepiece.fandom.com/wiki/Monkey_D_Luffy",
    "sub_articles": ["Monkey D. Luffy/History", "Monkey D. Luffy/Gallery"],
    "download_timestamp": "2024-01-15T10:30:00",
    "created_files": ["01_Introduction.txt", "Monkey_D_Luffy_Profile.png"],
    "csv_extraction_enabled": false,
    "total_sections": 8,
    "total_tables": 0,
    "total_images_found": 15,
    "total_images_downloaded": 12
}
```

## Data Flow

### 1. Article Discovery
```
User Input → Article List → API Query → Content Fetch
```

### 2. Content Processing
```
Raw HTML → Section Extraction → Text Cleaning → File Saving
Raw HTML → Table Extraction → CSV Conversion → File Saving
Raw HTML → Image Extraction → URL Validation → Download → Size Check → File Saving
```

### 3. Sub-Article Integration
```
Main Article → Sub-Article Discovery → Content Fetch → Deduplication → Merging
```

### 4. Output Generation
```
Processed Content → File Naming → Directory Creation → File Writing → Metadata Generation
```

## Configuration Options

### CSV Extraction Control
```python
SCRAPE_CSV_FILES = False  # Disable table extraction
SCRAPE_CSV_FILES = True   # Enable table extraction
```

**Impact**:
- **False**: Skips table detection and CSV file creation
- **True**: Processes all tables and saves as CSV files

### Image Limit Control
```python
MAX_IMAGES = 5    # Minimal images
MAX_IMAGES = 20   # Default limit
MAX_IMAGES = 50   # Extended limit
MAX_IMAGES = 0    # No images
```

**Impact**:
- Controls maximum images per article
- Applies to both main article and sub-articles
- Stops processing after reaching limit

### Article List Customization
```python
articles = ["Custom Article 1", "Custom Article 2"]
```

**Requirements**:
- Must be valid wiki article names
- Case-sensitive
- Spaces allowed (converted to underscores in URLs)

## Error Handling

### Network Errors
**HTTP Failures**: Graceful degradation with error messages
**Timeout Handling**: 30-second timeout for all requests
**Retry Logic**: None (single attempt per request)

### File System Errors
**Permission Issues**: Continues processing other files
**Disk Space**: Continues until disk full
**Path Creation**: Automatic directory creation with error handling

### Content Processing Errors
**HTML Parsing**: Continues with next element on regex failures
**Image Download**: Skips failed images, continues with others
**Dimension Checking**: Falls back to saving without size validation

### API Errors
**Invalid Articles**: Returns None, continues with next article
**Rate Limiting**: 1-second delays between requests
**Authentication**: Not required for public wiki access

## API Integration

### MediaWiki API Endpoints
**Base URL**: `https://onepiece.fandom.com/api.php`

**Parse Action**:
```
GET /api.php?action=parse&page={article}&format=json&prop=text
```

**Search Action**:
```
GET /api.php?action=query&list=search&srsearch="{article}/"&srnamespace=0&srlimit=50&format=json
```

### Response Processing
**Parse Response**:
```json
{
    "parse": {
        "text": {
            "*": "<html content>"
        }
    }
}
```

**Search Response**:
```json
{
    "query": {
        "search": [
            {"title": "Article/SubArticle"}
        ]
    }
}
```

### Rate Limiting
**Request Delays**: 1 second between requests
**Concurrent Requests**: None (sequential processing)
**API Limits**: Respects MediaWiki's recommended limits

## Performance Considerations

### Memory Management
**Incremental Processing**: Content processed one article at a time
**Streaming Downloads**: Images downloaded and saved individually
**Garbage Collection**: Automatic cleanup of processed content

### Network Efficiency
**Image Validation**: Downloads images before size checking
**Domain Filtering**: Only processes wiki image domains
**Size Filtering**: Skips small images to reduce bandwidth

### Processing Speed
**Regex Optimization**: Efficient patterns for HTML parsing
**Early Termination**: Stops image processing at limits
**Parallel Processing**: None (sequential for stability)

### Scalability
**Article Count**: Limited by available disk space
**Image Count**: Limited by `MAX_IMAGES` setting
**Content Size**: Limited by individual article sizes

## Best Practices

### Configuration
1. **Start Small**: Begin with few articles and low image limits
2. **Monitor Output**: Check generated files for quality
3. **Adjust Limits**: Increase limits based on needs and resources

### Error Handling
1. **Check Logs**: Review console output for errors
2. **Verify Files**: Ensure all expected files are created
3. **Handle Failures**: Individual failures don't stop entire process

### Performance Tuning
1. **Image Limits**: Balance quality vs. download time
2. **CSV Extraction**: Enable only when table data needed
3. **Rate Limiting**: Don't reduce delays (respects API limits)

### Maintenance
1. **Regular Updates**: Check for wiki structure changes
2. **Output Review**: Periodically review generated content quality
3. **Storage Management**: Monitor disk space usage

This documentation provides a comprehensive understanding of the One Piece Wiki Scraper's architecture, functionality, and usage patterns. The scraper is designed to be robust, configurable, and maintainable while providing high-quality content extraction from the One Piece Wiki.
