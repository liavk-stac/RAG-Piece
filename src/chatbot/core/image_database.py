"""
Image Database Index

This module provides an in-memory index of available images with metadata
for fast searching and retrieval in the image retrieval system.
"""

import os
import json
import pickle
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import logging
from datetime import datetime

from ..utils.image_metadata_parser import ImageMetadataParser

logger = logging.getLogger(__name__)


class ImageDatabase:
    """
    In-memory database index for image metadata and fast retrieval.
    
    Maintains a searchable index of all available images with their metadata,
    providing fast search capabilities for the image retrieval system.
    """
    
    def __init__(self, images_path: str, config_path: Optional[str] = None):
        """
        Initialize the image database.
        
        Args:
            images_path: Path to the directory containing images
            config_path: Optional path to save/load database configuration
        """
        self.images_path = Path(images_path)
        self.config_path = config_path
        self.logger = logger
        
        # Initialize components
        self.metadata_parser = ImageMetadataParser()
        self.image_index: Dict[str, Dict[str, Any]] = {}
        self.character_index: Dict[str, Set[str]] = {}
        self.type_index: Dict[str, Set[str]] = {}
        self.searchable_terms_index: Dict[str, Set[str]] = {}
        
        # Statistics
        self.total_images = 0
        self.last_updated = None
        
        # Build or load the index
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize the image index by building or loading from disk."""
        if self.config_path and os.path.exists(self.config_path):
            self.logger.info(f"Loading existing image index from {self.config_path}")
            if self._load_index():
                self.logger.info(f"Successfully loaded index with {self.total_images} images")
                return
        
        self.logger.info(f"Building new image index from {self.images_path}")
        self._build_index()
        
        # Save the index if config path is provided
        if self.config_path:
            self._save_index()
    
    def _build_index(self):
        """Build the image index by scanning the images directory."""
        if not self.images_path.exists():
            self.logger.error(f"Images directory does not exist: {self.images_path}")
            return
        
        try:
            # Parse all images in the directory
            metadata_list = self.metadata_parser.parse_directory(str(self.images_path))
            
            # Build the main index
            for metadata in metadata_list:
                if metadata.get('success', True):  # Skip failed parses
                    image_id = self._generate_image_id(metadata)
                    self.image_index[image_id] = metadata
            
            # Build secondary indexes for fast searching
            self._build_secondary_indexes()
            
            # Update statistics
            self.total_images = len(self.image_index)
            self.last_updated = datetime.now()
            
            self.logger.info(f"Successfully built index with {self.total_images} images")
            
        except Exception as e:
            self.logger.error(f"Error building image index: {e}")
            raise
    
    def _build_secondary_indexes(self):
        """Build secondary indexes for fast searching by character, type, and terms."""
        # Character index
        self.character_index.clear()
        for image_id, metadata in self.image_index.items():
            character = metadata.get('character', '').lower()
            if character:
                if character not in self.character_index:
                    self.character_index[character] = set()
                self.character_index[character].add(image_id)
        
        # Type index
        self.type_index.clear()
        for image_id, metadata in self.image_index.items():
            image_type = metadata.get('type', '')
            if image_type:
                if image_type not in self.type_index:
                    self.type_index[image_type] = set()
                self.type_index[image_type].add(image_id)
        
        # Searchable terms index
        self.searchable_terms_index.clear()
        for image_id, metadata in self.image_index.items():
            terms = metadata.get('searchable_terms', [])
            for term in terms:
                term_lower = term.lower()
                if term_lower not in self.searchable_terms_index:
                    self.searchable_terms_index[term_lower] = set()
                self.searchable_terms_index[term_lower].add(image_id)
    
    def _generate_image_id(self, metadata: Dict[str, Any]) -> str:
        """
        Generate a unique ID for an image based on its metadata.
        
        Args:
            metadata: Image metadata dictionary
            
        Returns:
            Unique image ID string
        """
        # Use a combination of character and content for uniqueness
        character = metadata.get('character', 'unknown')
        content = metadata.get('content', 'unknown')
        filename = metadata.get('filename', 'unknown')
        
        # Create a unique identifier
        image_id = f"{character}_{content}_{filename}".lower().replace(' ', '_')
        
        # Ensure uniqueness by adding counter if needed
        counter = 1
        original_id = image_id
        while image_id in self.image_index:
            image_id = f"{original_id}_{counter}"
            counter += 1
        
        return image_id
    
    def _save_index(self):
        """Save the image index to disk for persistence."""
        if not self.config_path:
            return
        
        try:
            # Prepare data for saving
            save_data = {
                'image_index': self.image_index,
                'character_index': {k: list(v) for k, v in self.character_index.items()},
                'type_index': {k: list(v) for k, v in self.type_index.items()},
                'searchable_terms_index': {k: list(v) for k, v in self.searchable_terms_index.items()},
                'total_images': self.total_images,
                'last_updated': self.last_updated.isoformat() if self.last_updated else None
            }
            
            # Save as pickle for efficiency
            with open(self.config_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            self.logger.info(f"Successfully saved image index to {self.config_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving image index: {e}")
    
    def _load_index(self) -> bool:
        """Load the image index from disk."""
        if not self.config_path or not os.path.exists(self.config_path):
            return False
        
        try:
            with open(self.config_path, 'rb') as f:
                save_data = pickle.load(f)
            
            # Restore the index
            self.image_index = save_data.get('image_index', {})
            
            # Restore secondary indexes
            self.character_index = {k: set(v) for k, v in save_data.get('character_index', {}).items()}
            self.type_index = {k: set(v) for k, v in save_data.get('type_index', {}).items()}
            self.searchable_terms_index = {k: set(v) for k, v in save_data.get('searchable_terms_index', {}).items()}
            
            # Restore statistics
            self.total_images = save_data.get('total_images', 0)
            last_updated_str = save_data.get('last_updated')
            if last_updated_str:
                self.last_updated = datetime.fromisoformat(last_updated_str)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading image index: {e}")
            return False
    
    def search_images(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search for images matching query criteria.
        
        Args:
            query: Dictionary containing search criteria
                - character: Character name to search for
                - image_type: Type of image to search for
                - terms: List of search terms
                - limit: Maximum number of results to return
        
        Returns:
            List of matching image metadata dictionaries
        """
        try:
            matching_image_ids = set()
            
            # Search by character
            if 'character' in query:
                character = query['character'].lower()
                if character in self.character_index:
                    matching_image_ids.update(self.character_index[character])
            
            # Search by image type
            if 'image_type' in query:
                image_type = query['image_type']
                if image_type in self.type_index:
                    if matching_image_ids:
                        matching_image_ids.intersection_update(self.type_index[image_type])
                    else:
                        matching_image_ids.update(self.type_index[image_type])
            
            # Search by terms
            if 'terms' in query and query['terms']:
                term_matches = set()
                for term in query['terms']:
                    term_lower = term.lower()
                    if term_lower in self.searchable_terms_index:
                        if term_matches:
                            term_matches.intersection_update(self.searchable_terms_index[term_lower])
                        else:
                            term_matches.update(self.searchable_terms_index[term_lower])
                
                if matching_image_ids:
                    matching_image_ids.intersection_update(term_matches)
                else:
                    matching_image_ids.update(term_matches)
            
            # If no specific criteria, return all images
            if not matching_image_ids and not any(key in query for key in ['character', 'image_type', 'terms']):
                matching_image_ids = set(self.image_index.keys())
            
            # Convert to metadata list
            results = []
            for image_id in matching_image_ids:
                if image_id in self.image_index:
                    results.append(self.image_index[image_id])
            
            # Apply limit if specified
            limit = query.get('limit', len(results))
            if limit > 0:
                results = results[:limit]
            
            self.logger.debug(f"Search query '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching images: {e}")
            return []
    
    def get_image_path(self, image_id: str) -> Optional[str]:
        """
        Get the full path to an image by its ID.
        
        Args:
            image_id: Unique identifier for the image
            
        Returns:
            Full path to the image file, or None if not found
        """
        if image_id in self.image_index:
            return self.image_index[image_id].get('full_path')
        return None
    
    def get_image_metadata(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific image by its ID.
        
        Args:
            image_id: Unique identifier for the image
            
        Returns:
            Image metadata dictionary, or None if not found
        """
        return self.image_index.get(image_id)
    
    def get_character_images(self, character: str) -> List[Dict[str, Any]]:
        """
        Get all images for a specific character.
        
        Args:
            character: Character name to search for
            
        Returns:
            List of metadata for images of the specified character
        """
        character_lower = character.lower()
        if character_lower in self.character_index:
            image_ids = self.character_index[character_lower]
            return [self.image_index[image_id] for image_id in image_ids if image_id in self.image_index]
        return []
    
    def get_images_by_type(self, image_type: str) -> List[Dict[str, Any]]:
        """
        Get all images of a specific type.
        
        Args:
            image_type: Type of image to search for
            
        Returns:
            List of metadata for images of the specified type
        """
        if image_type in self.type_index:
            image_ids = self.type_index[image_type]
            return [self.image_index[image_id] for image_id in image_ids if image_id in self.image_index]
        return []
    
    def get_all_images(self) -> List[Dict[str, Any]]:
        """
        Get metadata for all images in the database.
        
        Returns:
            List of all image metadata dictionaries
        """
        return list(self.image_index.values())
    
    def refresh_index(self):
        """Refresh the image index by rescanning the directory."""
        self.logger.info("Refreshing image index...")
        self._build_index()
        
        # Save the updated index
        if self.config_path:
            self._save_index()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics and information.
        
        Returns:
            Dictionary containing database statistics
        """
        return {
            'total_images': self.total_images,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'characters_count': len(self.character_index),
            'types_count': len(self.type_index),
            'searchable_terms_count': len(self.searchable_terms_index),
            'images_path': str(self.images_path),
            'config_path': self.config_path
        }
    
    def validate_image_files(self) -> Dict[str, Any]:
        """
        Validate that all indexed images still exist on disk.
        
        Returns:
            Dictionary containing validation results
        """
        valid_count = 0
        invalid_count = 0
        invalid_images = []
        
        for image_id, metadata in self.image_index.items():
            file_path = metadata.get('full_path')
            if file_path and os.path.exists(file_path):
                valid_count += 1
            else:
                invalid_count += 1
                invalid_images.append({
                    'image_id': image_id,
                    'path': file_path,
                    'metadata': metadata
                })
        
        return {
            'valid_images': valid_count,
            'invalid_images': invalid_count,
            'total_images': self.total_images,
            'invalid_image_details': invalid_images
        }
