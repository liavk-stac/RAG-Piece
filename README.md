# One Piece Wiki Scraper

A Python script that scrapes articles from the One Piece Wiki and saves the content, tables, and images locally.

## Features

- Fetches article content using the MediaWiki API
- Extracts sections and preserves their order
- Saves tables as CSV files
- Downloads images in original resolution
- Creates organized folder structure for each article
- Includes metadata for each scraped article
- Implements rate limiting to avoid API restrictions

## Installation

1. Install Python 3.7 or higher
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the scraper from the project root:
```bash
python src/one_piece_scraper.py
```

## Output Structure

For each article, the scraper creates a folder with the following structure:
```
Article_Name/
├── 01_Intro.txt
├── 02_History.txt
├── 03_Abilities.txt
├── Table_1.csv
├── Table_2.csv
├── Image_1.png
├── Image_2.png
└── metadata.json
```

## Articles Included

The scraper processes these One Piece characters:
- Monkey D. Luffy
- Roronoa Zoro
- Nami
- Usopp
- Sanji
- Tony Tony Chopper
- Nico Robin
- Franky
- Brook
- Jinbe

## Notes

- The script includes a 1-second delay between API calls to avoid rate limiting
- Failed downloads are logged but don't stop the scraping process
- All file names are automatically sanitized for filesystem compatibility
