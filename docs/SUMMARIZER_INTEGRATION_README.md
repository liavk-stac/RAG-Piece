# Article Summarizer Integration

This document describes how to use the integrated article summarizer in the RAG Piece system.

## Overview

The article summarizer has been integrated into the main RAG system and can create summary chunks alongside regular content chunks. These summaries provide high-level overviews of articles and can improve search results for general queries.

## Features

- **LangChain Refine Method**: Uses the refine method for high-quality summarization
- **OpenAI Integration**: Leverages GPT-4o-mini for consistent summaries
- **Token Limit Compliance**: Ensures summaries fit within the maximum chunk size (400 tokens)
- **Sub-article Support**: Creates summaries for both main articles and sub-articles
- **Configurable**: Can be enabled/disabled via configuration flag
- **Package Integration**: Properly integrated into the `src/rag_piece` package structure

## Package Structure

The summarizer is now properly integrated into the package:

```
src/rag_piece/
├── __init__.py          # Exports ArticleSummarizer
├── summarizer.py        # Main summarizer module
├── config.py            # Configuration with summarization flags
├── main.py              # Main application with summarizer integration
└── ...                  # Other modules
```

## Configuration

### Enable Summarization

To enable article summarization, modify the configuration in `src/rag_piece/main.py`:

```python
def main():
    """Main function to run the integrated scraper + RAG system."""
    # Initialize configuration
    config = RAGConfig()
    config.ENABLE_SUMMARIZATION = True  # Enable summarization
    config.SUMMARY_MODEL = "gpt-4o-mini"  # Optional: change model
    config.SUMMARY_TEMPERATURE = 0.3    # Optional: adjust creativity
    config.SAVE_SUMMARIES_TO_FILES = True  # Optional: save summaries as text files
```

### Configuration Parameters

- **`ENABLE_SUMMARIZATION`**: Boolean flag to enable/disable summarization (default: `False`)
- **`SUMMARY_MODEL`**: OpenAI model to use for summarization (default: `"gpt-4o-mini"`)
- **`SUMMARY_TEMPERATURE`**: Temperature setting for summary generation (default: `0.3`)
- **`SAVE_SUMMARIES_TO_FILES`**: Boolean flag to save summaries as text files in summaries/ folder (default: `False`)

## How It Works

1. **Content Scraping**: Regular content is scraped and chunked as before
2. **Summary Generation**: If enabled, the summarizer creates summaries for each article
3. **Chunk Creation**: Summaries are converted to chunk objects with proper metadata
4. **File Saving**: If enabled, summaries are also saved as text files in the summaries/ folder
5. **Database Integration**: Summary chunks are indexed alongside content chunks
6. **Search Enhancement**: Summaries can appear in search results with special labeling

## Summary Chunk Structure

Summary chunks follow the same metadata structure as regular chunks but include additional summary-specific information:

## File Output Structure

When `SAVE_SUMMARIES_TO_FILES` is enabled, summaries are saved as text files in the following structure:

```
summaries/
└── arabasta_kingdom/
    ├── main_main_article_summary.txt
    ├── nefertari_cobra_sub_article_summary.txt
    ├── pell_sub_article_summary.txt
    └── ...
```

Each summary file contains:
- A header with the article name and summary type
- The full summary text
- UTF-8 encoding for proper character support

```python
{
    'chunk_id': 'arabasta_kingdom_summary_main',
    'content': 'Summary text...',
    'search_metadata': {
        'article_name': 'Arabasta Kingdom',
        'sub_article_name': None,
        'section_name': 'Article Summary',
        'sub_section_name': None,
        'keywords': []  # Filled by keyword extractor
    },
    'summary_metadata': {
        'chunk_type': 'summary',
        'summary_type': 'main_article',
        'generation_method': 'langchain_refine',
        'model_used': 'gpt-4o-mini',
        'generation_timestamp': '2024-01-01T12:00:00'
    },
    'debug_metadata': {
        'chunk_size': 350,
        'target_token_limit': 400,
        'token_efficiency': 87.5
    }
}
```

## Usage Examples

### Basic Usage (Disabled by Default)

```bash
# Run without summarization (faster, no API costs)
python -m src.rag_piece.main
```

### With Summarization Enabled

```bash
# First, enable summarization in main.py, then run:
python -m src.rag_piece.main
```

### Standalone Testing

```bash
# Test the summarizer independently using the package
python test/standalone_summarizer.py

# Test package integration
python test/test_summarizer_package.py
```

### Package Import Usage

```python
# Import from the package
from rag_piece import ArticleSummarizer, RAGConfig

# Or import directly from modules
from rag_piece.summarizer import ArticleSummarizer
from rag_piece.config import RAGConfig

# Use the summarizer
config = RAGConfig()
config.ENABLE_SUMMARIZATION = True
summarizer = ArticleSummarizer(max_chunk_size=config.MAX_CHUNK_SIZE)
```

## Search Results

When searching the database, summary chunks will be clearly labeled:

```
Test query: 'What is Arabasta Kingdom?'
  Found 3 results
    1. Article Summary [SUMMARY: main_article] (BM25: 0.856, Semantic: 0.923, Combined: 0.901)
        Arabasta Kingdom is a desert kingdom located in the Grand Line...
    2. Geography [CONTENT] (BM25: 0.723, Semantic: 0.789, Combined: 0.756)
        The kingdom is characterized by its vast desert landscape...
```

## Performance Considerations

- **API Costs**: Summarization requires OpenAI API calls (expensive for large datasets)
- **Processing Time**: Adds significant time to database construction
- **Memory Usage**: Summary chunks increase database size
- **Search Quality**: Improves results for general queries but may reduce specificity

## Best Practices

1. **Use Sparingly**: Only enable for articles where summaries add significant value
2. **Monitor Costs**: Track OpenAI API usage when summarization is enabled
3. **Test First**: Use standalone testing before full integration
4. **Cache Results**: Consider saving summaries to avoid regeneration
5. **Package Structure**: Use the integrated package instead of standalone files

## Troubleshooting

### Common Issues

1. **Import Error**: The summarizer is now properly integrated into the package
2. **API Key Missing**: Check that `.env` file contains `OPENAI_API_KEY`
3. **Memory Issues**: Large articles may cause memory problems during summarization
4. **Rate Limiting**: OpenAI API rate limits may affect processing speed

### Debug Mode

Enable verbose logging in the configuration:

```python
config.VERBOSE_LOGGING = True
config.LOG_LEVEL = "DEBUG"
```

## Migration from Standalone

If you were using the old standalone `article_summarizer.py`:

1. **Remove**: Delete the old `article_summarizer.py` from the root directory
2. **Update Imports**: Use `from rag_piece.summarizer import ArticleSummarizer`
3. **Test**: Run `python test/test_summarizer_package.py` to verify integration
4. **Clean Up**: Remove any old test files that import from the root

## Future Enhancements

- **Batch Processing**: Process multiple articles simultaneously
- **Caching**: Save and reuse summaries across runs
- **Quality Metrics**: Track summary quality and compression ratios
- **Custom Prompts**: Allow user-defined summarization prompts
- **Alternative Models**: Support for other LLM providers
