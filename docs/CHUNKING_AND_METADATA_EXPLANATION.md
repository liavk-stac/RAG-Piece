# Chunking and Metadata System Explanation

## Overview

The RAG Piece system uses a sophisticated chunking strategy to break down large articles into manageable pieces while preserving context and relationships. Each chunk is enriched with comprehensive metadata for optimal search and retrieval.

## Chunking Process

### 1. **Hybrid Chunking Strategy**

The system uses a three-step approach:

#### **Step 1: Paragraph Splitting**
- Content is first split by paragraph separators (`\n\n` by default)
- Each paragraph is cleaned and stripped of whitespace
- Empty paragraphs are filtered out

#### **Step 2: Merge Short Chunks**
- Paragraphs that are too small (below `MIN_CHUNK_SIZE`) are merged together
- Merging continues until chunks reach the `TARGET_CHUNK_SIZE`
- This prevents overly fragmented content

#### **Step 3: Split Long Chunks**
- Chunks that exceed `MAX_CHUNK_SIZE` are split into smaller pieces
- Splitting occurs at sentence boundaries using configurable separators
- Overlap is maintained between chunks for context continuity

### 2. **Configuration Parameters**

```python
# From RAGConfig
MIN_CHUNK_SIZE: int = 100            # Minimum tokens per chunk
MAX_CHUNK_SIZE: int = 400            # Maximum tokens per chunk
TARGET_CHUNK_SIZE: int = 250         # Ideal chunk size
OVERLAP_SIZE: int = 50               # Overlap between chunks
PARAGRAPH_SEPARATOR: str = "\n\n"    # Paragraph separator
SENTENCE_SEPARATORS: List[str] = [". ", "! ", "? ", ".\n", "!\n", "?\n"]
```

### 3. **Chunking Algorithm Flow**

```
Raw Content → Paragraph Split → Merge Short → Split Long → Final Chunks
     ↓              ↓              ↓           ↓           ↓
HTML Content   Paragraphs    Merged      Split      Chunk Objects
              (by \n\n)     Chunks      Chunks     with Metadata
```

## Metadata Structure

Each chunk contains three types of metadata:

### 1. **Search Metadata** (Core Search Information)

```python
'search_metadata': {
    'article_name': str,           # Main article name (e.g., "Monkey D. Luffy")
    'sub_article_name': str,       # Sub-article if applicable (e.g., "Devil Fruit")
    'section_name': str,           # Section title (e.g., "Background")
    'sub_section_name': str,       # H3 sub-sections found in chunk
    'keywords': List[str]          # Extracted keywords (filled by keyword extractor)
}
```

### 2. **Debug Metadata** (Optional Development Information)

```python
'debug_metadata': {
    'section_index': int,          # Index of the section in the article
    'chunk_index': int,            # Index of the chunk within the section
    'chunk_size': int,             # Token count of the chunk
    'has_h3_tags': bool,           # Whether chunk contains H3 headings
    'processing_timestamp': str    # ISO timestamp of processing
}
```

### 3. **Core Chunk Data**

```python
{
    'chunk_id': str,               # Unique identifier (e.g., "monkey-d-luffy_background_001")
    'content': str,                # Actual text content of the chunk
    'search_metadata': {...},      # Search-related metadata
    'debug_metadata': {...}        # Debug information (if enabled)
}
```

## Chunk ID Generation

### **Format**: `{article_slug}_{section_slug}_{chunk_number:03d}`

**Examples:**
- `monkey-d-luffy_background_001`
- `straw-hat-pirates_crew-members_003`
- `devil-fruit_types_002`

**Components:**
- `article_slug`: URL-friendly version of article name
- `section_slug`: URL-friendly version of section name
- `chunk_number`: Sequential number padded to 3 digits

## Sub-Section Detection

The system automatically detects H3 headings within chunks:

### **HTML Pattern Matching**
```python
h3_pattern = r'<h3[^>]*>(.*?)</h3>'
```

### **Extraction Process**
1. Scan chunk content for `<h3>` tags
2. Extract text content from tags
3. Clean HTML markup
4. Join multiple sub-sections with semicolons

### **Example**
```html
<h3>Early Life</h3>
<p>Luffy was born in...</p>
<h3>Devil Fruit</h3>
<p>He ate the Gomu Gomu no Mi...</p>
```

**Result**: `sub_section_name = "Early Life; Devil Fruit"`

## Integration with CSV to Text Converter

When converting CSV files to text, the process integrates seamlessly:

### 1. **CSV Conversion**
- CSV data is converted to structured text using GPT-4o-mini
- Text maintains data relationships and context
- Output is formatted with clear headings and structure

### 2. **Chunking Integration**
```python
# Convert CSV to text
result = converter.convert_csv_to_text("file.csv", "Article Name", existing_summary)

# Create chunks from converted text
chunks = rag_db.chunker.chunk_section_content(
    result['converted_text'],      # The converted CSV text
    "CSV Data",                   # Section name
    result['article_name'],        # Article name
    None                          # No sub-article
)
```

### 3. **Metadata Preservation**
- CSV filename becomes part of the chunk context
- Article summary provides background context
- Data relationships are maintained in the chunked content

## Search Optimization

### **Field Boosting**
Different metadata fields have different search weights:

```python
CONTENT_BOOST: float = 1.0             # Base content weight
ARTICLE_NAME_BOOST: float = 3.0        # Article name mentions (highest)
SUB_ARTICLE_NAME_BOOST: float = 2.8    # Sub-article name relevance
SECTION_NAME_BOOST: float = 2.5        # Section title relevance
SUB_SECTION_BOOST: float = 2.0         # Sub-section title relevance
KEYWORDS_BOOST: float = 1.8            # BM25-extracted keywords
```

### **Search Strategy**
1. **BM25 Search**: Fast keyword-based retrieval
2. **Semantic Search**: Vector similarity using embeddings
3. **Result Fusion**: Combines both approaches with configurable weights
4. **Reranking**: Final relevance scoring using metadata

## Example Chunk Object

```python
{
    'chunk_id': 'monkey-d-luffy_devil-fruit_002',
    'content': 'The Gomu Gomu no Mi is a Paramecia-type Devil Fruit...',
    'search_metadata': {
        'article_name': 'Monkey D. Luffy',
        'sub_article_name': None,
        'section_name': 'Devil Fruit',
        'sub_section_name': 'Powers and Abilities',
        'keywords': ['gomu gomu', 'devil fruit', 'paramecia', 'rubber']
    },
    'debug_metadata': {
        'section_index': 0,
        'chunk_index': 2,
        'chunk_size': 245,
        'has_h3_tags': True,
        'processing_timestamp': '2024-01-15T10:30:00'
    }
}
```

## Benefits of This System

### **1. Context Preservation**
- Chunks maintain logical boundaries
- Overlap ensures context continuity
- Metadata provides rich search context

### **2. Search Efficiency**
- Fast retrieval using chunk IDs
- Semantic search across chunk content
- Metadata-based filtering and boosting

### **3. Scalability**
- Configurable chunk sizes
- Efficient storage and indexing
- Support for large document collections

### **4. Debugging and Monitoring**
- Comprehensive metadata tracking
- Processing timestamps
- Chunk size and structure information

## Configuration Tuning

### **For Small Documents**
```python
MIN_CHUNK_SIZE = 50
TARGET_CHUNK_SIZE = 150
MAX_CHUNK_SIZE = 300
```

### **For Large Documents**
```python
MIN_CHUNK_SIZE = 150
TARGET_CHUNK_SIZE = 300
MAX_CHUNK_SIZE = 500
```

### **For Technical Content**
```python
SENTENCE_SEPARATORS = [". ", "! ", "? ", ".\n", "!\n", "?\n", "; ", ":\n"]
```

This chunking and metadata system ensures that your RAG database maintains context, enables efficient search, and provides comprehensive information for each piece of content, whether it's from scraped articles or converted CSV data.
