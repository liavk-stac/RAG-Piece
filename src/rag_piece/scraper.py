"""
One Piece Wiki scraper functionality.
"""

import requests
import re
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from PIL import Image
import io

from .utils import slugify


class OneWikiScraper:
    """Scraper for One Piece Wiki content"""
    
    # Class-level cache for sub-articles to avoid duplicate API calls
    _sub_articles_cache = {}
    
    def __init__(self, max_images: int = 6, request_delay: float = 1.0):
        self.max_images = max_images
        self.request_delay = request_delay
        self.logger = logging.getLogger("rag_piece.scraper")
        
        # API configuration
        self.api_base = "https://onepiece.fandom.com/api.php"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RAG-Piece/1.0 (Educational Research Tool)'
        })
    
    def scrape_article(self, article_name: str) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Scrape a single article and return sections and metadata.
        
        Args:
            article_name: Name of the article to scrape
        
        Returns:
            Tuple of (sections_list, scraping_metadata)
        """
        try:
            self.logger.info(f"Scraping article: {article_name}")
            
            # Create images folder
            images_folder = self._create_images_folder(article_name)
            
            # Find and scrape sub-articles
            sub_articles = self._find_sub_articles(article_name)
            
            # Process main article and sub-articles
            all_sections, all_images = self._scrape_all_content(article_name, sub_articles)
            
            # Download images
            downloaded_images = self._download_images(all_images, images_folder)
            
            # Create metadata
            metadata = self._create_scraping_metadata(
                article_name, sub_articles, all_sections, all_images, downloaded_images
            )
            
            self.logger.info(f"Completed scraping: {article_name} ({len(all_sections)} sections, {len(downloaded_images)} images)")
            return all_sections, metadata
        
        except Exception as e:
            self.logger.error(f"Error scraping article {article_name}: {e}", exc_info=True)
            return [], {}
    
    def _create_images_folder(self, article_name: str) -> Path:
        """Create images folder for the article."""
        images_folder = Path("data/images") / slugify(article_name)
        images_folder.mkdir(parents=True, exist_ok=True)
        return images_folder
    
    def _find_sub_articles(self, article_name: str) -> List[str]:
        """Find sub-articles for the main article with retry logic and caching."""
        # Check cache first to avoid duplicate API calls
        if article_name in self._sub_articles_cache:
            cached_result = self._sub_articles_cache[article_name]
            self.logger.info(f"Using cached sub-articles for {article_name}: {len(cached_result)} found")
            return cached_result
        
        # Use configuration values if available, otherwise use defaults
        try:
            max_retries = getattr(self, 'config', None) and getattr(self.config, 'API_MAX_RETRIES', 3) or 3
            base_delay = getattr(self, 'config', None) and getattr(self.config, 'API_BASE_DELAY', 2.0) or 2.0
        except AttributeError:
            max_retries = 3
            base_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Finding sub-articles for: {article_name} (attempt {attempt + 1}/{max_retries})")
                
                # Search for sub-articles using API
                search_params = {
                    'action': 'query',
                    'format': 'json',
                    'list': 'search',
                    'srsearch': f'"{article_name}/"',
                    'srnamespace': 0,
                    'srlimit': 50
                }
                
                response = self._make_api_request(search_params)
                if not response:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        self.logger.warning(f"API request failed, retrying in {delay}s...")
                        time.sleep(delay)
                        continue
                    return []
                
                sub_articles = []
                search_results = response.get('query', {}).get('search', [])
                
                for result in search_results:
                    title = result['title']
                    if title.startswith(f"{article_name}/"):
                        sub_articles.append(title)
                
                # Add common sub-article patterns
                common_patterns = ['Gallery', 'Images', 'Pictures', 'History']
                for pattern in common_patterns:
                    potential_sub = f"{article_name}/{pattern}"
                    if potential_sub not in sub_articles:
                        # Check if it exists
                        if self._article_exists(potential_sub):
                            sub_articles.append(potential_sub)
                
                # Cache the result for future use
                self._sub_articles_cache[article_name] = sub_articles
                
                self.logger.info(f"Found {len(sub_articles)} sub-articles")
                return sub_articles
            
            except Exception as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    self.logger.warning(f"Error finding sub-articles (attempt {attempt + 1}): {e}, retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"Error finding sub-articles after {max_retries} attempts: {e}", exc_info=True)
                    return []
        
        return []
    
    def _scrape_all_content(self, main_article: str, sub_articles: List[str]) -> tuple[List[Dict], List[Dict]]:
        """Scrape content from main article and all sub-articles."""
        all_sections = []
        all_images = []
        processed_content = set()
        
        # Process main article first
        sections, images = self._scrape_single_article(main_article)
        all_sections.extend(sections)
        all_images.extend(images)
        
        # Track processed content to avoid duplicates
        for section in sections:
            content_key = self._generate_content_key(section.get('content', ''))
            processed_content.add(content_key)
        
        # Process sub-articles
        for sub_article in sub_articles:
            self.logger.info(f"Processing sub-article: {sub_article}")
            
            sections, images = self._scrape_single_article(sub_article)
            
            # Filter out duplicate content
            unique_sections = []
            for section in sections:
                content_key = self._generate_content_key(section.get('content', ''))
                if content_key not in processed_content:
                    unique_sections.append(section)
                    processed_content.add(content_key)
            
            all_sections.extend(unique_sections)
            all_images.extend(images)
            
            time.sleep(self.request_delay)  # Rate limiting
        
        return all_sections, all_images
    
    def _scrape_single_article(self, article_name: str) -> tuple[List[Dict], List[Dict]]:
        """Scrape a single article."""
        try:
            # Get article content with redirect following
            parse_params = {
                'action': 'parse',
                'format': 'json',
                'page': article_name,
                'prop': 'text|images|sections',
                'redirects': 1  # Follow redirects automatically
            }
            
            response = self._make_api_request(parse_params)
            if not response or 'parse' not in response:
                return [], []
            
            parse_data = response['parse']
            html_content = parse_data.get('text', {}).get('*', '')
            
            if not html_content:
                return [], []
            
            # Check if this is still a redirect page (some redirects don't get followed)
            if self._is_redirect_page(html_content):
                self.logger.warning(f"Article {article_name} appears to be a redirect page - trying to extract redirect target")
                redirect_target = self._extract_redirect_target(html_content)
                if redirect_target and redirect_target != article_name:
                    self.logger.info(f"Following redirect from {article_name} to {redirect_target}")
                    return self._scrape_single_article(redirect_target)
            
            # Extract sections and images
            sections = self._extract_sections(html_content, article_name)
            images = self._extract_images(parse_data.get('images', []))
            
            return sections, images
        
        except Exception as e:
            self.logger.error(f"Error scraping article {article_name}: {e}", exc_info=True)
            return [], []
    
    def _extract_sections(self, html_content: str, article_name: str) -> List[Dict[str, str]]:
        """Extract sections from HTML content."""
        try:
            # First, extract the main article content and filter out unwanted elements
            filtered_html = self._filter_main_content(html_content)
            
            # Split into sections based on headings BEFORE cleaning HTML
            sections = self._split_into_sections(filtered_html)
            
            # Process each section
            processed_sections = []
            for i, (title, content) in enumerate(sections):
                # Clean the content for this section
                cleaned_content = self._clean_html_content(content)
                
                if self._is_valid_section(title, cleaned_content):
                    section_data = {
                        'combined_title': title or f"Section {i+1}",
                        'content': cleaned_content,
                        'article_source': article_name,
                        'section_index': i
                    }
                    processed_sections.append(section_data)
            
            return processed_sections
        
        except Exception as e:
            self.logger.error(f"Error extracting sections: {e}", exc_info=True)
            return []
    
    def _filter_main_content(self, html_content: str) -> str:
        """Filter HTML to extract only main article content, removing navigation and unwanted elements."""
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements that contain navigation, character lists, etc.
            unwanted_selectors = [
                # Navigation and site elements
                '.navbox', '.navigation-box', '.nav', '.site-navigation',
                '.mw-navigation', '.portal', '.portlet',
                
                # References and citations
                '.references', '.citation', '.reference', '.mw-references-wrap',
                '.reflist', '.mw-cite-backlink',
                
                # Images and media (we handle these separately)
                '.thumb', '.thumbinner', '.thumbcaption', 
                
                # Infoboxes and tables (we might want these, but they often contain character lists)
                '.infobox', '.wikitable', '.character-list', '.bounty-list',
                
                # Edit sections and tools
                '.mw-editsection', '.edit-section',
                
                # Categories and templates
                '.category-links', '.template',
                
                # Specific One Piece Wiki elements
                '.weaponed-fighters', '.individuals-with-bounties',
                
                # IDs
                '#references', '#References', '#External_links', '#See_also',
                '#Navigation', '#Site_Navigation'
            ]
            
            # Remove unwanted elements
            for selector in unwanted_selectors:
                for element in soup.select(selector):
                    element.decompose()
            
            # Try to find the main content area
            main_content = None
            
            # Look for common main content containers
            main_selectors = [
                '.mw-parser-output',  # MediaWiki main content
                '.article-content',
                '.page-content',
                '#content',
                '#mw-content-text'
            ]
            
            for selector in main_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            # If no main content container found, use the entire soup
            if not main_content:
                main_content = soup
            
            return str(main_content)
            
        except ImportError:
            # Fallback: remove common unwanted patterns with regex
            self.logger.warning("BeautifulSoup not available, using regex fallback for content filtering")
            return self._filter_main_content_regex(html_content)
        except Exception as e:
            self.logger.error(f"Error filtering main content: {e}")
            return html_content  # Return original if filtering fails
    
    def _filter_main_content_regex(self, html_content: str) -> str:
        """Regex-based fallback for filtering main content."""
        # Remove navigation boxes
        html_content = re.sub(r'<div[^>]*class="[^"]*navbox[^"]*"[^>]*>.*?</div>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove reference sections
        html_content = re.sub(r'<div[^>]*class="[^"]*references[^"]*"[^>]*>.*?</div>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove infoboxes (they often contain character lists)
        html_content = re.sub(r'<table[^>]*class="[^"]*infobox[^"]*"[^>]*>.*?</table>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        
        return html_content
    
    def _clean_html_content(self, html_content: str) -> str:
        """Clean HTML content and convert to plain text."""
        try:
            from bs4 import BeautifulSoup
            
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            unwanted_tags = [
                'script', 'style', 'noscript', 'nav', 'footer', 'header',
                'aside', 'figure', 'figcaption', 'sup', 'sub'
            ]
            for tag in unwanted_tags:
                for element in soup.find_all(tag):
                    element.decompose()
            
            # Remove elements with specific classes/ids
            unwanted_selectors = [
                '.navbox', '.references', '.citation', '.reference',
                '.mw-references-wrap', '.reflist', '.thumb', '.thumbinner',
                '.thumbcaption', '.infobox', '.wikitable', '.mw-editsection',
                '#references', '#References', '.mw-cite-backlink'
            ]
            for selector in unwanted_selectors:
                for element in soup.select(selector):
                    element.decompose()
            
            # Convert to text
            text = soup.get_text()
            
            # Clean up text
            # Remove multiple whitespaces and newlines
            text = re.sub(r'\s+', ' ', text)
            
            # Remove bibliography references like [1], [2], [1-5], etc.
            text = re.sub(r'\[[\d\s,\-]+\]', '', text)
            
            # Remove extra spacing around punctuation
            text = re.sub(r'\s+([.,:;!?])', r'\1', text)
            
            # Clean up multiple spaces
            text = ' '.join(text.split())
            
            return text.strip()
            
        except ImportError:
            # Fallback to regex-based cleaning if BeautifulSoup not available
            return self._clean_html_content_regex(html_content)
        except Exception as e:
            self.logger.error(f"Error cleaning HTML content: {e}")
            return self._clean_html_content_regex(html_content)
    
    def _clean_html_content_regex(self, html_content: str) -> str:
        """Fallback regex-based HTML cleaning."""
        # Remove script and style tags
        html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove all HTML tags
        html_content = re.sub(r'<[^>]+>', '', html_content)
        
        # Remove bibliography labels
        html_content = re.sub(r'\[[\d\s,\-]+\]', '', html_content)
        
        # Clean up whitespace
        html_content = re.sub(r'\s+', ' ', html_content)
        
        return html_content.strip()
    
    def _split_into_sections(self, content: str) -> List[tuple[str, str]]:
        """Split content into sections based on headings."""
        # Find all headings (h2, h3, h4)
        heading_pattern = r'<h([2-4])[^>]*?id="([^"]*)"[^>]*>(.*?)</h\1>|<h([2-4])[^>]*>(.*?)</h\4>'
        headings = list(re.finditer(heading_pattern, content, re.IGNORECASE | re.DOTALL))
        
        if not headings:
            # No headings found, return entire content as one section
            return [("Main Content", content)]
        
        sections = []
        
        for i, heading_match in enumerate(headings):
            # Extract heading title (handle both patterns: with and without id)
            if heading_match.group(3):  # Pattern with id
                heading_html = heading_match.group(3)
            else:  # Pattern without id
                heading_html = heading_match.group(5)
            
            # Clean heading title
            heading_title = re.sub(r'<[^>]+>', '', heading_html).strip()
            
            # Skip edit links and other unwanted headings
            if '[edit]' in heading_title.lower() or 'edit' in heading_title.lower():
                heading_title = re.sub(r'\[edit\]', '', heading_title, flags=re.IGNORECASE).strip()
            
            # Extract content between this heading and the next
            start_pos = heading_match.end()
            end_pos = headings[i + 1].start() if i + 1 < len(headings) else len(content)
            
            section_content = content[start_pos:end_pos].strip()
            
            if section_content and heading_title:
                sections.append((heading_title, section_content))
        
        return sections
    
    def _is_valid_section(self, title: str, content: str) -> bool:
        """Check if section is valid for processing."""
        if not content or not content.strip():
            return False
        
        # Skip navigation, reference, and list sections
        skip_titles = [
            'site navigation', 'references', 'external links', 'see also',
            'navigation', 'navbox', 'categories', 'gallery', 'trivia',
            'weaponed fighters', 'individuals with bounties', 'bounties',
            'related articles', 'template', 'templates'
        ]
        
        if title and any(skip in title.lower() for skip in skip_titles):
            return False
        
        # Skip sections that are mostly character lists (lots of names separated by spaces)
        words = content.split()
        if len(words) > 20:  # Only check longer content
            # Count capitalized words (likely character names)
            capitalized_words = [w for w in words if w and w[0].isupper() and len(w) > 2]
            capitalized_ratio = len(capitalized_words) / len(words)
            
            # If more than 60% of words are capitalized, it's likely a character list
            if capitalized_ratio > 0.6:
                self.logger.debug(f"Skipping section '{title}' - appears to be a character list (capitalized ratio: {capitalized_ratio:.2f})")
                return False
        
        # Must have minimum content length
        if len(content.strip()) < 100:  # Increased minimum length
            return False
        
        # Skip sections with very short sentences (likely lists)
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if len(sentences) > 3:  # Only check if we have multiple sentences
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if avg_sentence_length < 5:  # Very short sentences suggest lists
                self.logger.debug(f"Skipping section '{title}' - appears to be a list (avg sentence length: {avg_sentence_length:.1f} words)")
                return False
        
        return True
    
    def _extract_images(self, image_list: List[str]) -> List[Dict[str, str]]:
        """Extract image information from API response."""
        images = []
        
        for image_name in image_list:
            try:
                # Get image info
                image_info = self._get_image_info(image_name)
                if image_info:
                    images.append(image_info)
            except Exception as e:
                self.logger.warning(f"Error processing image {image_name}: {e}")
        
        return images
    
    def _get_image_info(self, image_name: str) -> Optional[Dict[str, str]]:
        """Get detailed information about an image."""
        try:
            # Filter out "Featured Article" images
            if "Featured_Article" in image_name:
                return None
            
            params = {
                'action': 'query',
                'format': 'json',
                'titles': f'File:{image_name}',
                'prop': 'imageinfo',
                'iiprop': 'url|size'
            }
            
            response = self._make_api_request(params)
            if not response:
                return None
            
            pages = response.get('query', {}).get('pages', {})
            for page_data in pages.values():
                imageinfo = page_data.get('imageinfo', [])
                if imageinfo:
                    info = imageinfo[0]
                    return {
                        'url': info.get('url', ''),
                        'width': info.get('width', 0),
                        'height': info.get('height', 0),
                        'label': image_name.replace('File:', '').replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
                    }
            
            return None
        
        except Exception as e:
            self.logger.warning(f"Error getting image info for {image_name}: {e}")
            return None
    
    def _download_images(self, images: List[Dict], images_folder: Path) -> List[str]:
        """Download images to the specified folder."""
        downloaded = []
        
        for i, img_data in enumerate(images):
            if i >= self.max_images:
                break
            
            try:
                if self._download_single_image(img_data, images_folder):
                    downloaded.append(img_data['label'])
            except Exception as e:
                self.logger.warning(f"Failed to download image {img_data.get('label', 'unknown')}: {e}")
        
        return downloaded
    
    def _download_single_image(self, img_data: Dict, images_folder: Path) -> bool:
        """Download a single image."""
        url = img_data.get('url')
        width = img_data.get('width', 0)
        height = img_data.get('height', 0)
        
        if not url or width < 100 or height < 100:
            return False
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Validate image
            image = Image.open(io.BytesIO(response.content))
            if image.width < 100 or image.height < 100:
                return False
            
            # Save image
            filename = f"{slugify(img_data['label'])}.png"
            filepath = images_folder / filename
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            self.logger.info(f"Downloaded: {filename} ({width}x{height})")
            return True
        
        except Exception as e:
            self.logger.warning(f"Error downloading image: {e}")
            return False
    
    def _make_api_request(self, params: Dict) -> Optional[Dict]:
        """Make API request with error handling."""
        try:
            response = self.session.get(self.api_base, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"API request failed: {e}")
            return None
    
    def _article_exists(self, article_name: str) -> bool:
        """Check if an article exists."""
        params = {
            'action': 'query',
            'format': 'json',
            'titles': article_name
        }
        
        response = self._make_api_request(params)
        if not response:
            return False
        
        pages = response.get('query', {}).get('pages', {})
        return not any('-1' in str(page_id) for page_id in pages.keys())
    
    def _is_redirect_page(self, html_content: str) -> bool:
        """Check if the HTML content represents a redirect page."""
        # Look for redirect indicators in the HTML
        redirect_indicators = [
            'class="redirectMsg"',
            'class="redirectText"',
            'Redirect to:',
            '<p>Redirect to:</p>'
        ]
        return any(indicator in html_content for indicator in redirect_indicators)
    
    def _extract_redirect_target(self, html_content: str) -> Optional[str]:
        """Extract the redirect target from HTML content."""
        try:
            # Look for redirect link in the HTML
            import re
            # Pattern to find redirect links like: <a href="/wiki/TARGET" title="TARGET">TARGET</a>
            redirect_pattern = r'<a href="/wiki/([^"]+)" title="[^"]*">([^<]+)</a>'
            match = re.search(redirect_pattern, html_content)
            if match:
                # Return the link target (group 1) with URL decoding
                target = match.group(1).replace('_', ' ')
                return target
            return None
        except Exception as e:
            self.logger.error(f"Error extracting redirect target: {e}")
            return None
    
    def _generate_content_key(self, content: str) -> str:
        """Generate a key for content deduplication."""
        import hashlib
        # Use first 200 characters for deduplication
        key_content = content.strip()[:200]
        return hashlib.md5(key_content.encode()).hexdigest()
    
    def _create_scraping_metadata(self, article_name: str, sub_articles: List[str],
                                 sections: List[Dict], images: List[Dict], downloaded: List[str]) -> Dict[str, Any]:
        """Create metadata about the scraping process."""
        return {
            'article_name': article_name,
            'sub_articles': sub_articles,
            'scraping_timestamp': datetime.now().isoformat(),
            'total_sections': len(sections),
            'total_images_found': len(images),
            'images_downloaded': len(downloaded),
            'images_folder': f'data/images/{slugify(article_name)}'
        }
