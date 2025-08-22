# CSV Scraper for One Piece Wiki

This is a standalone CSV scraper that extracts tabular data from One Piece Wiki articles and converts them to CSV files.

## Features

- **Table Extraction**: Automatically finds and extracts all tables from wiki articles
- **CSV Conversion**: Converts HTML tables to clean, structured CSV files
- **Sub-article Support**: Scrapes both main articles and their sub-articles
- **Content Filtering**: Excludes navigation, references, and site navigation sections
- **Duplicate Prevention**: Avoids processing duplicate content across articles
- **Smart Naming**: Generates meaningful filenames based on table content

## How It Works

1. **Article Discovery**: Finds the main article and any sub-articles (e.g., "Arabasta Kingdom/Gallery")
2. **HTML Parsing**: Uses the One Piece Wiki API to get article content
3. **Table Detection**: Identifies valid HTML tables using BeautifulSoup
4. **Data Extraction**: Extracts headers and data rows from each table
5. **CSV Generation**: Creates properly formatted CSV files with UTF-8 encoding
6. **File Organization**: Saves CSV files to `csv_files/[article_name]/` folder

## Usage

### Standalone Testing

```bash
# Run the test script
python csv_scraper_test.py
```

This will:
- Scrape the "Arabasta Kingdom" article
- Extract all tables
- Convert them to CSV files
- Save files to `csv_files/arabasta_kingdom/`
- Display a summary of what was created

### Programmatic Usage

```python
from rag_piece.csv_scraper import CSVWikiScraper

# Initialize the scraper
scraper = CSVWikiScraper(request_delay=1.0)

# Scrape an article
csv_files, metadata = scraper.scrape_article_to_csv("Arabasta Kingdom")

# Check results
print(f"Created {len(csv_files)} CSV files")
print(f"Tables found: {metadata['total_tables_found']}")
```

## Output Structure

```
csv_files/
└── arabasta_kingdom/
    ├── arabasta_kingdom_Table_1.csv
    ├── arabasta_kingdom_Table_2.csv
    └── ...
```

## Table Validation

The scraper only processes tables that:
- Have at least 2 rows (header + data)
- Have at least 2 columns
- Are not navigation tables
- Contain meaningful content

## Content Filtering

The scraper automatically excludes:
- Navigation boxes (`navbox` class)
- Reference sections (`references` class)
- Site navigation sections
- Script and style tags
- Bibliography citations

## Dependencies

- `requests`: HTTP requests to the wiki API
- `beautifulsoup4`: HTML parsing and table extraction
- `csv`: CSV file writing
- `pandas`: Data manipulation (imported but not currently used)

## Rate Limiting

The scraper includes a 1-second delay between requests to be respectful to the wiki servers.

## Error Handling

- Graceful handling of missing articles
- Logging of all operations and errors
- Continues processing even if individual tables fail
- Comprehensive error reporting

## Future Integration

This scraper is designed to be integrated with the main RAG system later, allowing it to:
- Work alongside the existing text scraper
- Feed CSV data into the RAG database
- Enable hybrid search across both text and tabular data
