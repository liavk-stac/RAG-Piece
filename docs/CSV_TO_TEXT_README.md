# CSV to Text Converter

The CSV to Text Converter is a powerful module that converts CSV files to structured, readable text using OpenAI's GPT-4o-mini model. It maintains data relationships while creating text suitable for vector embedding and search.

## Features

- **Intelligent Conversion**: Uses GPT-4o-mini to convert CSV data to structured text
- **Metadata Integration**: Leverages CSV filename and article summaries for context-aware conversion
- **Data Relationship Preservation**: Maintains logical connections between columns and rows
- **Debug Support**: Optional saving of conversion results as text files
- **Batch Processing**: Convert multiple CSV files at once
- **Error Handling**: Comprehensive error handling and logging

## Configuration

Enable the CSV to text converter in your `RAGConfig`:

```python
from rag_piece.config import RAGConfig

config = RAGConfig()
config.ENABLE_CSV_TO_TEXT = True                    # Enable CSV to text conversion
config.CSV_TO_TEXT_MODEL = "gpt-4o-mini"           # OpenAI model to use
config.CSV_TO_TEXT_TEMPERATURE = 0.2                # Temperature for consistency
config.SAVE_CSV_TO_TEXT_FILES = True                # Save debug files
```

## Basic Usage

### Single CSV Conversion

```python
from rag_piece.csv_to_text import CSVToTextConverter

# Initialize converter
converter = CSVToTextConverter(config)

# Convert a single CSV file (with optional existing summary)
existing_summary = "Article summary from database creation process"
result = converter.convert_csv_to_text("path/to/file.csv", "Article Name", existing_summary)

if result['conversion_metadata']['conversion_success']:
    converted_text = result['converted_text']
    print(f"Conversion successful! Text length: {len(converted_text)}")
else:
    print(f"Conversion failed: {result['conversion_metadata'].get('error_message')}")
```

### Multiple CSV Conversion

```python
# Convert multiple CSV files
csv_files = ["file1.csv", "file2.csv", "file3.csv"]
results = converter.convert_multiple_csvs(csv_files, "Article Name")

for result in results:
    if result['conversion_metadata']['conversion_success']:
        print(f"✓ {result['csv_filename']}: {len(result['converted_text'])} characters")
    else:
        print(f"❌ {result['csv_filename']}: Failed")
```

## How It Works

### 1. CSV Analysis
The converter analyzes the CSV structure:
- Reads the file using pandas
- Identifies columns, data types, and sample values
- Creates a comprehensive structure description

### 2. Context Retrieval
Retrieves article context:
- Gets article summary from the existing summarizer system
- Uses the summary to provide context for the conversion

### 3. LLM Conversion
Sends structured prompt to GPT-4o-mini:
- CSV filename and structure
- Article summary for context
- CSV content (all rows included)
- Instructions for maintaining data relationships

### 4. Result Processing
Creates comprehensive result object:
- Converted text
- Conversion metadata (timestamps, model info, statistics)
- Debug information

## Custom Prompts

The converter uses a carefully crafted prompt that:

1. **Maintains Data Relationships**: Ensures logical connections between columns are preserved
2. **Structures Output**: Creates well-organized text with headings and sections
3. **Preserves Context**: Uses article summaries to maintain One Piece lore consistency
4. **Optimizes for Embedding**: Produces text suitable for vector search

Example prompt structure:
```
You are an expert data analyst specializing in One Piece Wiki content. 
Your task is to convert CSV data into well-structured, readable text 
while maintaining all data relationships and context.

CSV Filename: {csv_filename}
Article Summary: {article_summary}

CSV Structure: {csv_structure}
CSV Content: {csv_content}

Instructions:
1. Convert the CSV data into structured, narrative text
2. Maintain ALL data relationships between columns and rows
3. Organize information logically and coherently
4. Use clear, descriptive language that flows naturally
5. Preserve the context and meaning from the original data
6. Structure the output with appropriate sections and formatting
7. Ensure the text is comprehensive and includes all important data points
8. Make the text suitable for vector embedding and search
```

## Output Format

The converted text is structured for optimal embedding:

- **Clear Headings**: Section titles that describe the content
- **Logical Grouping**: Related information grouped together
- **Bullet Points**: Lists where appropriate for easy scanning
- **Chronological Order**: Maintains logical sequence of information
- **Data Relationships**: Preserves connections between different data points

## Debug Files

When `SAVE_CSV_TO_TEXT_FILES` is enabled, debug files are saved to `data/debug/csv2text/`:

```
CSV to Text Conversion Debug File
================================

CSV File: /path/to/file.csv
Article: Article Name
Conversion Timestamp: 2024-01-15T10:30:00
Model Used: gpt-4o-mini
Temperature: 0.2
Original CSV Shape: (50, 6)
Converted Text Length: 2500 characters
Converted Text Tokens: 625
Article Summary Length: 800 characters
Conversion Success: True

Converted Text:
==============

[The actual converted text content...]
```

## Integration with RAG System

The converted text can be integrated into your RAG database:

```python
from rag_piece.database import RAGDatabase

# Initialize RAG database
rag_db = RAGDatabase(config)

# Convert CSV to text
result = converter.convert_csv_to_text("file.csv", "Article Name")

if result['conversion_metadata']['conversion_success']:
    # Create chunks from converted text
    chunks = rag_db.chunker.chunk_section_content(
        result['converted_text'],
        "CSV Data",
        result['article_name'],
        None  # No sub-article
    )
    
    # Build indices
    rag_db.build_indices_from_chunks(chunks)
```

## Error Handling

The converter provides comprehensive error handling:

- **File Not Found**: Graceful handling of missing CSV files
- **Empty Files**: Detection and reporting of empty CSV files
- **API Errors**: Handling of OpenAI API failures
- **Summary Errors**: Fallback when article summaries are unavailable

## Performance Considerations

- **No Size Limitations**: All CSV rows are included in the conversion
- **Batch Processing**: Process multiple files efficiently
- **Avoid Duplicate Summarizer Runs**: Use existing summaries from database creation
- **Rate Limiting**: Respects OpenAI API rate limits

### Avoiding Duplicate Summarizer Runs

Since the database creation process already runs the summarizer to create article summaries, you can pass these existing summaries to the CSV converter:

```python
# During database creation
summary_chunks = summarizer.create_summary_chunks("Article Name")
existing_summary = summary_chunks[0]['content']

# Later, when converting CSV files
result = converter.convert_csv_to_text("file.csv", "Article Name", existing_summary)
```

This prevents the CSV converter from re-running the summarizer and saves both time and API costs.

## Testing

Run the comprehensive test suite:

```bash
cd test
python test_csv_to_text.py
```

The tests cover:
- Initialization and configuration
- CSV file reading and parsing
- Structure description generation
- Content formatting
- Article summary retrieval
- Result object creation
- Error handling
- Debug file saving

## Example Output

**Input CSV:**
```csv
Character Name,Devil Fruit,Bounty,Crew Position
Monkey D. Luffy,Gomu Gomu no Mi,3000000000,Captain
Roronoa Zoro,None,1111000000,Swordsman
Nami,None,366000000,Navigator
```

**Converted Text:**
```
# Straw Hat Pirates - Character Information

## Main Crew Members

### Monkey D. Luffy
- **Devil Fruit**: Gomu Gomu no Mi (Rubber Rubber Fruit)
- **Bounty**: 3,000,000,000 Berries
- **Crew Position**: Captain
- **Role**: Leader of the Straw Hat Pirates

### Roronoa Zoro
- **Devil Fruit**: None (Non-Devil Fruit User)
- **Bounty**: 1,111,000,000 Berries
- **Crew Position**: Swordsman
- **Role**: First mate and primary combatant

### Nami
- **Devil Fruit**: None (Non-Devil Fruit User)
- **Bounty**: 366,000,000 Berries
- **Crew Position**: Navigator
- **Role**: Expert navigator and weather specialist

## Crew Overview
The Straw Hat Pirates consist of diverse individuals with unique abilities and roles. The crew is led by Monkey D. Luffy, who possesses the powerful Gomu Gomu no Mi devil fruit, making him a rubber human. Roronoa Zoro serves as the crew's swordsman and first mate, while Nami provides essential navigation skills.
```

## Requirements

- OpenAI API key
- LangChain packages: `langchain`, `langchain-community`, `openai`
- pandas for CSV processing
- python-dotenv for environment variables

## Troubleshooting

### Common Issues

1. **API Key Missing**: Ensure `OPENAI_API_KEY` is set in environment
2. **CSV File Not Found**: Check file paths and permissions
3. **Empty CSV Files**: Verify CSV files contain data
4. **Large Files**: All CSV data is included regardless of file size

### Debug Steps

1. Enable debug file saving: `config.SAVE_CSV_TO_TEXT_FILES = True`
2. Check the `data/debug/csv2text/` folder for detailed logs
3. Verify CSV file format and encoding
4. Check article summary availability

## Future Enhancements

- **Custom Prompt Templates**: Allow users to define their own conversion prompts
- **Batch Size Control**: Configurable batch sizes for large CSV collections
- **Output Format Options**: Multiple output formats (markdown, HTML, etc.)
- **Progress Tracking**: Real-time progress updates for large conversions
- **Caching**: Cache article summaries to reduce API calls
