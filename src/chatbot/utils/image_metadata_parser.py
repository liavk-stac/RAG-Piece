"""
Image Metadata Parser

This module provides utilities for parsing image metadata from folder structures
and filenames to create searchable metadata for the image retrieval system.
"""

import os
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ImageMetadataParser:
    """
    Parser for extracting metadata from image folder structures and filenames.
    
    Extracts character/entity information from folder names and scene/location
    information from filenames to create searchable metadata.
    """
    
    def __init__(self):
        """Initialize the image metadata parser."""
        self.logger = logger
        
        # Common One Piece character name variations and mappings
        self.character_mappings = {
            'straw_hat_pirates': ['straw hat pirates', 'straw hat crew', 'mugiwara'],
            'monkey_d_luffy': ['monkey d luffy', 'luffy', 'straw hat luffy', 'monkey luffy'],
            'roronoa_zoro': ['roronoa zoro', 'zoro', 'pirate hunter zoro'],
            'nami': ['nami', 'navigator nami'],
            'usopp': ['usopp', 'sogeking', 'god usopp'],
            'sanji': ['sanji', 'black leg sanji', 'vinsmoke sanji'],
            'tony_tony_chopper': ['tony tony chopper', 'chopper', 'doctor chopper'],
            'nico_robin': ['nico robin', 'robin', 'devil child robin'],
            'franky': ['franky', 'cyborg franky', 'cutty flam'],
            'brook': ['brook', 'soul king brook', 'dead bones brook'],
            'jinbe': ['jinbe', 'first son of the sea jinbe', 'knight of the sea']
        }
        
        # Common scene/location patterns
        self.scene_patterns = {
            'crew': ['crew', 'group', 'team', 'together'],
            'ship': ['ship', 'merry', 'sunny', 'thousand sunny', 'going merry'],
            'battle': ['battle', 'fight', 'combat', 'war', 'conflict'],
            'location': ['arabasta', 'water7', 'enies lobby', 'marineford', 'dressrosa'],
            'form': ['gear', 'transformation', 'power', 'ability'],
            'training': ['training', 'practice', 'learning', 'development']
        }
    
    def parse_image_path(self, image_path: str) -> Dict[str, Any]:
        """
        Parse an image path to extract metadata.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing extracted metadata
        """
        try:
            path_obj = Path(image_path)
            
            # Extract folder name (character/entity)
            folder_name = path_obj.parent.name
            filename = path_obj.stem  # filename without extension
            
            # Parse metadata
            metadata = {
                'full_path': str(image_path),
                'filename': filename,
                'folder': folder_name,
                'extension': path_obj.suffix.lower(),
                'character': self._extract_character(folder_name),
                'content': self._extract_content(filename),
                'type': self._determine_image_type(folder_name, filename),
                'searchable_terms': self._generate_searchable_terms(folder_name, filename)
            }
            
            self.logger.debug(f"Parsed metadata for {image_path}: {metadata}")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error parsing image path {image_path}: {e}")
            return {
                'full_path': str(image_path),
                'error': str(e),
                'success': False
            }
    
    def _extract_character(self, folder_name: str) -> str:
        """
        Extract character name from folder name.
        
        Args:
            folder_name: Name of the folder containing the image
            
        Returns:
            Normalized character name
        """
        folder_lower = folder_name.lower().replace('_', ' ')
        
        # Check for exact matches in character mappings
        for key, variations in self.character_mappings.items():
            if folder_lower in variations or folder_lower == key.replace('_', ' '):
                return key.replace('_', ' ').title()
        
        # If no exact match, try to extract from folder name
        # Handle common patterns like "Monkey_D_Luffy" -> "Monkey D Luffy"
        if '_' in folder_name:
            # Split by underscore and capitalize each part
            parts = folder_name.split('_')
            return ' '.join(part.title() for part in parts)
        
        return folder_name.replace('_', ' ').title()
    
    def _extract_content(self, filename: str) -> str:
        """
        Extract content description from filename.
        
        Args:
            filename: Name of the image file (without extension)
            
        Returns:
            Normalized content description
        """
        # Remove common prefixes and normalize
        content = filename.lower()
        
        # Handle common patterns - only remove if it's a clear prefix pattern
        # Don't remove if the remaining content would be too short or unclear
        # Also don't remove if the filename contains meaningful content that should be preserved
        if content.startswith('luffy_') and len(content) > 6:
            remaining = content[6:]
            # Don't remove if the remaining content is too short or if it's a descriptive name
            if len(remaining) > 5 and not any(word in remaining for word in ['and', 'with', 'crew', 'team']):
                content = remaining
        elif content.startswith('zoro_') and len(content) > 5:
            remaining = content[5:]
            # Don't remove if the remaining content is too short or if it's a descriptive name
            if len(remaining) > 5 and not any(word in remaining for word in ['and', 'with', 'crew', 'team']):
                content = remaining
        elif content.startswith('straw_hat_') and len(content) > 11:
            remaining = content[11:]
            # Don't remove if the remaining content is too short or if it's a descriptive name
            if len(remaining) > 5 and not any(word in remaining for word in ['and', 'with', 'crew', 'team']):
                content = remaining
        
        # Replace underscores with spaces and capitalize
        content = content.replace('_', ' ').title()
        
        # Fix common words that should be lowercase
        content = content.replace(' And ', ' and ').replace(' With ', ' with ').replace(' Of ', ' of ').replace(' The ', ' the ')
        
        return content
    
    def _determine_image_type(self, folder_name: str, filename: str) -> str:
        """
        Determine the type of image based on folder and filename.
        
        Args:
            folder_name: Name of the folder
            filename: Name of the image file
            
        Returns:
            Image type classification
        """
        folder_lower = folder_name.lower()
        filename_lower = filename.lower()
        
        # Check for crew/group images
        if 'straw_hat' in folder_lower or 'crew' in filename_lower:
            return 'crew_group'
        
        # Check for ship images
        if any(term in filename_lower for term in ['ship', 'merry', 'sunny']):
            return 'ship'
        
        # Check for battle/combat images
        if any(term in filename_lower for term in ['battle', 'fight', 'combat', 'war']):
            return 'battle_combat'
        
        # Check for location-specific images
        if any(term in filename_lower for term in ['arabasta', 'water7', 'enies', 'marineford', 'dressrosa']):
            return 'location_specific'
        
        # Check for character form/transformation images
        if any(term in filename_lower for term in ['gear', 'transformation', 'power']):
            return 'character_form'
        
        # Check for training/development images
        if any(term in filename_lower for term in ['training', 'practice', 'learning']):
            return 'training_development'
        
        # Default to character image
        return 'character_image'
    
    def _generate_searchable_terms(self, folder_name: str, filename: str) -> List[str]:
        """
        Generate searchable terms from folder and filename.
        
        Args:
            folder_name: Name of the folder
            filename: Name of the image file
            
        Returns:
            List of searchable terms
        """
        terms = []
        
        # Add folder-based terms
        folder_terms = folder_name.lower().replace('_', ' ').split()
        terms.extend(folder_terms)
        
        # Add filename-based terms
        filename_terms = filename.lower().replace('_', ' ').split()
        terms.extend(filename_terms)
        
        # Add character variations
        folder_lower = folder_name.lower()
        for key, variations in self.character_mappings.items():
            if folder_lower == key.replace('_', ' '):
                terms.extend(variations)
                break
        
        # Add scene/location terms
        filename_lower = filename.lower()
        for category, patterns in self.scene_patterns.items():
            if any(pattern in filename_lower for pattern in patterns):
                terms.extend(patterns)
                break
        
        # Remove duplicates and empty strings
        terms = list(set(term for term in terms if term.strip()))
        
        return terms
    
    def parse_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Parse all images in a directory recursively.
        
        Args:
            directory_path: Path to the directory containing images
            
        Returns:
            List of metadata dictionaries for all images
        """
        metadata_list = []
        directory = Path(directory_path)
        
        if not directory.exists():
            self.logger.error(f"Directory does not exist: {directory_path}")
            return []
        
        # Supported image extensions
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
        
        try:
            # Walk through directory recursively
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = Path(root) / file
                    
                    # Check if it's an image file
                    if file_path.suffix.lower() in image_extensions:
                        # Parse metadata for this image
                        metadata = self.parse_image_path(str(file_path))
                        if metadata.get('success', True):  # Skip failed parses
                            metadata_list.append(metadata)
            
            self.logger.info(f"Successfully parsed {len(metadata_list)} images from {directory_path}")
            return metadata_list
            
        except Exception as e:
            self.logger.error(f"Error parsing directory {directory_path}: {e}")
            return []
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Validate parsed metadata for completeness and correctness.
        
        Args:
            metadata: Metadata dictionary to validate
            
        Returns:
            True if metadata is valid, False otherwise
        """
        required_fields = ['full_path', 'character', 'content', 'type']
        
        # Check required fields
        for field in required_fields:
            if field not in metadata or not metadata[field]:
                self.logger.warning(f"Missing required field: {field}")
                return False
        
        # Check if file exists
        if not os.path.exists(metadata['full_path']):
            self.logger.warning(f"Image file does not exist: {metadata['full_path']}")
            return False
        
        # Check if character is reasonable
        if len(metadata['character']) < 2:
            self.logger.warning(f"Character name too short: {metadata['character']}")
            return False
        
        return True
    
    def get_character_images(self, metadata_list: List[Dict[str, Any]], character: str) -> List[Dict[str, Any]]:
        """
        Get all images for a specific character.
        
        Args:
            metadata_list: List of image metadata
            character: Character name to search for
            
        Returns:
            List of metadata for images of the specified character
        """
        character_lower = character.lower()
        
        matching_images = []
        for metadata in metadata_list:
            if metadata.get('character', '').lower() == character_lower:
                matching_images.append(metadata)
        
        return matching_images
    
    def get_images_by_type(self, metadata_list: List[Dict[str, Any]], image_type: str) -> List[Dict[str, Any]]:
        """
        Get all images of a specific type.
        
        Args:
            metadata_list: List of image metadata
            image_type: Type of image to search for
            
        Returns:
            List of metadata for images of the specified type
        """
        matching_images = []
        for metadata in metadata_list:
            if metadata.get('type') == image_type:
                matching_images.append(metadata)
        
        return matching_images
