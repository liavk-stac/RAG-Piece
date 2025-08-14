# One Piece Wiki Scraper

A comprehensive Python script that scrapes articles from the One Piece Wiki and saves content, tables, and images locally. The scraper automatically detects and includes sub-articles for complete character coverage.

## Features

### Core Functionality
- **Main Article Scraping**: Fetches complete article content using the MediaWiki API
- **Sub-Article Discovery**: Automatically finds and scrapes related sub-articles (e.g., "Monkey D. Luffy/History")
- **Comprehensive Content Extraction**: Captures sections, tables, and images from both main articles and sub-articles
- **Duplicate Prevention**: Intelligent deduplication to avoid content overlap between main and sub-articles
- **Configurable Scraping**: Control which content types to extract (CSV tables, images, sections)

### Content Processing
- **Dual Section Detection**: Extracts both MediaWiki sections (`<span class="mw-headline">`) and traditional HTML sections (`<h2>`)
- **Smart Text Cleaning**: Removes HTML tags, bibliography labels `[1][2][3]`, MediaWiki templates, and other artifacts
- **Navigation Filtering**: Automatically skips non-content sections like "Site Navigation", "References", "External Links"
- **Clean Filenames**: Generates filesystem-safe names without HTML artifacts or special characters

### Data Organization
- **Structured Output**: Creates organized folder structure for each character
- **CSV Tables**: Saves tables with clean, descriptive filenames (configurable)
- **High-Quality Images**: Downloads images with size filtering and configurable limits
- **Comprehensive Metadata**: Tracks sub-articles, file counts, and timestamps

### Technical Features
- **Rate Limiting**: Implements delays to respect API limits
- **Error Handling**: Graceful failure handling with detailed logging
- **Clean Data Folder**: Automatically clears previous runs for fresh data
- **Unicode Support**: Proper handling of special characters and accents
- **Image Dimension Validation**: Checks image sizes before downloading

## Configuration Options

### CSV Table Extraction
- **Control Flag**: `SCRAPE_CSV_FILES` - Set to `False` to skip CSV table extraction
- **Default**: `False` (tables are not extracted by default)
- **Use Case**: Useful when you only need text content and images

### Image Download Settings
- **Maximum Images**: `MAX_IMAGES` - Controls how many images to download per article
- **Default**: 20 images per article
- **Size Filtering**: Only downloads images that are at least 100x100 pixels
- **Quality Control**: Automatically skips small or low-quality images

### Content Selection
- **Sections**: Always extracted (text content)
- **Tables**: Optional (controlled by CSV flag)
- **Images**: Always extracted (with size and count limits)

## Installation

1. Install Python 3.7 or higher
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage
Run the scraper from the project root:
```bash
python one_piece_scraper.py
```

### Configuration
Edit the configuration section in `main()` function:
```python
# Configuration flags
SCRAPE_CSV_FILES = False  # Set to True to enable CSV table extraction
MAX_IMAGES = 20          # Maximum images per article (1-100)
```

### Custom Article List
Modify the articles list in the `main()` function:
```python
articles = ["Monkey D. Luffy", "Roronoa Zoro", "Nami", "Your Custom Article"]
```

## Output Structure

For each character, the scraper creates a comprehensive folder structure:

```
data/
└── Monkey_D_Luffy/
    ├── 01_Introduction.txt
    ├── 02_Appearance.txt
    ├── 03_Personality.txt
    ├── 04_Abilities_and_Powers.txt
    ├── 05_History.txt
    ├── 06_Major_Battles.txt
    ├── 07_Relationships.txt
    ├── 08_Trivia.txt
    ├── Baratie.csv                    # Only if CSV extraction enabled
    ├── Devil_Fruit_Powers.csv         # Only if CSV extraction enabled
    ├── Bounty_History.csv             # Only if CSV extraction enabled
    ├── Monkey_D_Luffy_Profile.png    # Up to MAX_IMAGES count
    ├── Gear_Forms.png                 # Minimum 100x100 pixels
    ├── Battle_Scenes.png              # Size-validated images only
    └── metadata.json                  # Complete scraping summary
```

## Articles Included

The scraper processes these One Piece characters with their complete sub-article coverage:

- **Monkey D. Luffy** - Main article + sub-articles (History, Abilities, etc.)
- **Roronoa Zoro** - Main article + sub-articles (Swords, Training, etc.)
- **Nami** - Main article + sub-articles (Navigation, Weather, etc.)
- **Arabasta Kingdom** - Main article + sub-articles (Geography, Culture, etc.)

## Content Quality Improvements

### Text Cleaning
- ✅ Removes bibliography labels `[1][2][3][42]`
- ✅ Strips HTML tags and MediaWiki templates
- ✅ Cleans citation placeholders `[citation needed]`
- ✅ Normalizes whitespace and formatting

### Section Extraction
- ✅ **MediaWiki Sections**: Captures `<span class="mw-headline">` sections
- ✅ **HTML Sections**: Captures traditional `<h2>` sections
- ✅ **Navigation Filtering**: Skips "Site Navigation", "References", etc.
- ✅ **Duplicate Prevention**: Tracks seen titles to avoid repetition

### File Naming
- ✅ **Clean Section Names**: No more HTML artifacts in filenames
- ✅ **Descriptive Table Names**: Meaningful CSV filenames
- ✅ **Length Limits**: Prevents extremely long filenames
- ✅ **Filesystem Safe**: Only alphanumeric characters, spaces, hyphens, underscores

### Image Processing
- ✅ **Size Validation**: Only downloads images ≥100x100 pixels
- ✅ **Count Limiting**: Configurable maximum images per article
- ✅ **Quality Control**: Skips small, low-quality images
- ✅ **Smart Labeling**: Creates meaningful filenames from alt text or URLs

## Sub-Article Discovery

The scraper automatically finds sub-articles using two methods:

1. **API Search**: Queries MediaWiki API for articles starting with "Character_Name/"
2. **Common Patterns**: Checks for standard sub-articles like "Gallery", "Images", "Pictures"

This ensures comprehensive coverage without manual article list maintenance.

## Technical Details

### API Endpoints Used
- `action=parse`: Fetches main article content
- `action=query&list=search`: Discovers sub-articles
- Rate limiting: 1-second delays between requests

### Content Merging Strategy
- **Main Article**: Processed first, establishes baseline content
- **Sub-Articles**: Processed sequentially, content merged with deduplication
- **Unique Keys**: Generated for each content piece to prevent duplicates
- **Metadata Tracking**: Comprehensive logging of all discovered content

### Error Handling
- **Network Failures**: Graceful degradation with detailed error messages
- **File System Issues**: Continues processing even if individual files fail
- **API Limits**: Respects rate limits and handles timeouts
- **Data Validation**: Ensures content quality before saving

### Image Download Process
- **URL Validation**: Checks if image is from supported wiki domains
- **Dimension Checking**: Downloads image first, then validates size
- **Fallback Handling**: Saves images even if dimension checking fails
- **Progress Tracking**: Shows download progress and success rates

## Configuration Examples

### Text-Only Scraping
```python
SCRAPE_CSV_FILES = False  # Skip tables
MAX_IMAGES = 0            # Skip images
```

### Full Content Scraping
```python
SCRAPE_CSV_FILES = True   # Include tables
MAX_IMAGES = 50           # More images per article
```

### Minimal Image Scraping
```python
SCRAPE_CSV_FILES = False  # Skip tables
MAX_IMAGES = 5            # Just a few key images
```

## Notes

- **Automatic Cleanup**: Previous data folders are automatically cleared before each run
- **Comprehensive Logging**: Detailed progress information and error reporting
- **Unicode Support**: Proper handling of Japanese names and special characters
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **No Manual Configuration**: Automatically adapts to wiki structure changes
- **Memory Efficient**: Processes content incrementally without loading everything into memory

## Recent Updates

### Latest Version Features
- **Configurable CSV Extraction**: Control whether to extract and save tables
- **Maximum Images Parameter**: Set custom limits for images per article
- **Image Size Filtering**: Only downloads images ≥100x100 pixels
- **Simplified Code Structure**: Cleaner, more maintainable codebase
- **Enhanced Configuration**: Easy-to-modify settings for different use cases
- **Improved Image Processing**: Better quality control and error handling
- **Sub-Article Scraping**: Complete coverage of character-related content
- **Enhanced Text Cleaning**: Bibliography labels and HTML artifacts removed
- **Navigation Filtering**: Automatic skipping of non-content sections
- **Improved File Naming**: Clean, descriptive filenames without artifacts
- **Duplicate Prevention**: Intelligent content merging from multiple sources

## Performance Considerations

- **Image Downloads**: Each image requires a separate HTTP request
- **Rate Limiting**: 1-second delays between requests to respect API limits
- **Memory Usage**: Images are processed one at a time to minimize memory footprint
- **Network Efficiency**: Only downloads images that meet size requirements

The scraper now provides the most comprehensive and configurable One Piece character data available, automatically discovering and including all related content while maintaining clean, organized output and giving you full control over what gets extracted.
