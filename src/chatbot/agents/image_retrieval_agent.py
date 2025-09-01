"""
Image Retrieval Agent

This agent uses LLM capabilities to intelligently analyze user queries and
select the most relevant image from the image database for display.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from .base_agent import BaseAgent, AgentInput, AgentOutput, AgentType
from ..utils.llm_client import LLMClient
from ..core.image_database import ImageDatabase

logger = logging.getLogger(__name__)


class ImageRetrievalAgent(BaseAgent):
    """
    LLM-powered image retrieval agent for intelligent image selection.
    
    This agent analyzes user queries to determine what type of image would be
    most relevant, searches the image database for potential matches, and uses
    LLM-based scoring to select the best single image for display.
    """
    
    def __init__(self, config):
        """Initialize the image retrieval agent."""
        super().__init__(config, AgentType.IMAGE_RETRIEVAL)
        
        # Initialize LLM client for query analysis and image scoring
        try:
            self.llm_client = LLMClient(config)
            self.logger.info("LLM client initialized successfully for image retrieval")
        except Exception as e:
            self.logger.warning(f"Failed to initialize LLM client: {e}")
            self.llm_client = None
        
        # Initialize image database
        self.image_database = None
        self._initialize_image_database()
    
    def _initialize_image_database(self):
        """Initialize the image database with the configured images path."""
        try:
            images_path = self.config.IMAGES_PATH
            config_path = self.config.IMAGE_INDEX_PATH
            
            # Create images directory if it doesn't exist
            os.makedirs(images_path, exist_ok=True)
            
            # Initialize image database
            self.image_database = ImageDatabase(images_path, config_path)
            self.logger.info(f"Image database initialized with {self.image_database.total_images} images")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize image database: {e}")
            self.image_database = None
    
    def _validate_input(self, input_data: AgentInput) -> bool:
        """Validate input data for the image retrieval agent."""
        # Call parent validation
        if not super()._validate_input(input_data):
            return False
        
        # Image retrieval agent requires a query to analyze
        if not input_data.query or not input_data.query.strip():
            self.logger.warning("Image retrieval agent requires a non-empty query")
            return False
        
        # Check if image database is available
        if not self.image_database:
            self.logger.warning("Image database not available")
            return False
        
        return True
    
    def _execute_agent(self, input_data: AgentInput) -> Dict[str, Any]:
        """
        Execute the image retrieval agent logic.
        
        Args:
            input_data: Input data containing query and context
            
        Returns:
            Dictionary containing image retrieval results
        """
        self.logger.info("Executing image retrieval agent")
        
        try:
            # Step 1: Analyze query intent using LLM
            intent_analysis = self._analyze_query_intent(input_data.query)
            self.logger.info(f"Query intent analyzed: {intent_analysis.get('intent_type', 'unknown')}")
            
            # Step 2: Find relevant images based on intent
            relevant_images = self._find_relevant_images(intent_analysis)
            self.logger.info(f"Found {len(relevant_images)} relevant images")
            
            # Step 3: Score images for relevance using LLM
            scored_images = self._score_image_relevance(relevant_images, intent_analysis)
            
            # Step 4: Select the best image
            best_image = self._select_best_image(scored_images)
            
            # Step 5: Prepare response
            response = self._prepare_response(best_image, intent_analysis, scored_images)
            
            self.logger.info("Image retrieval agent execution completed successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"Error in image retrieval agent execution: {e}")
            return {
                'success': False,
                'error': str(e),
                'image': None,
                'intent_analysis': {},
                'metadata': {
                    'agent_type': 'image_retrieval',
                    'error_occurred': True,
                    'error_message': str(e)
                }
            }
    
    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Use LLM to analyze what type of image would be most relevant for the query.
        
        Args:
            query: User query to analyze
            
        Returns:
            Dictionary containing intent analysis results
        """
        if not self.llm_client:
            self.logger.warning("LLM client not available, using fallback intent analysis")
            return self._fallback_intent_analysis(query)
        
        try:
            system_message = """You are an expert in One Piece knowledge and image analysis.
            Analyze the user's query to determine what type of image would be most relevant.
            
            Consider:
            1. What character(s) are mentioned or implied?
            2. What type of scene or context is being discussed?
            3. What visual elements would best represent the query?
            
            Respond with a JSON object containing:
            {
                "intent_type": "character_focus|crew_focus|location_focus|action_focus|general_info",
                "primary_character": "character name or null",
                "secondary_characters": ["list of other characters or empty"],
                "scene_type": "crew|battle|ship|location|training|form|other",
                "location": "specific location or null",
                "action": "specific action or null",
                "confidence": 0.0-1.0,
                "reasoning": "brief explanation of why this image type is relevant"
            }"""
            
            prompt = f"""User Query: "{query}"
            
            What type of image would be most relevant for this query? Analyze the intent and provide a structured response."""
            
            # Log LLM call for intent analysis
            if hasattr(self, 'pipeline_logger'):
                self.pipeline_logger.log_llm_call(
                    agent_name="IMAGE_RETRIEVAL_AGENT",
                    prompt=prompt,
                    response="",
                    tokens_used=0,
                    system_message=system_message
                )
            
            llm_response = self.llm_client.generate_text(
                prompt, 
                system_message, 
                max_tokens=300, 
                temperature=0.1
            )
            
            # Log LLM response for intent analysis
            if hasattr(self, 'pipeline_logger'):
                self.pipeline_logger.log_llm_call(
                    agent_name="IMAGE_RETRIEVAL_AGENT",
                    prompt=prompt,
                    response=llm_response,
                    tokens_used=300,
                    system_message=system_message
                )
            
            # Parse LLM response
            intent_analysis = self._parse_intent_response(llm_response)
            
            self.logger.debug(f"LLM intent analysis: {intent_analysis}")
            return intent_analysis
            
        except Exception as e:
            self.logger.error(f"LLM intent analysis failed: {e}")
            return self._fallback_intent_analysis(query)
    
    def _fallback_intent_analysis(self, query: str) -> Dict[str, Any]:
        """
        Fallback intent analysis when LLM is not available.
        
        Args:
            query: User query to analyze
            
        Returns:
            Dictionary containing basic intent analysis
        """
        query_lower = query.lower()
        
        # Simple keyword-based analysis
        intent_type = "general_info"
        primary_character = None
        scene_type = "other"
        
        # Check for character mentions
        character_keywords = {
            'luffy': 'Monkey D Luffy',
            'zoro': 'Roronoa Zoro',
            'nami': 'Nami',
            'usopp': 'Usopp',
            'sanji': 'Sanji',
            'chopper': 'Tony Tony Chopper',
            'robin': 'Nico Robin',
            'franky': 'Franky',
            'brook': 'Brook',
            'jinbe': 'Jinbe'
        }
        
        for keyword, character in character_keywords.items():
            if keyword in query_lower:
                primary_character = character
                intent_type = "character_focus"
                break
        
        # Check for crew mentions
        if any(term in query_lower for term in ['crew', 'straw hat', 'pirates', 'team', 'straw hat pirates']):
            intent_type = "crew_focus"
            scene_type = "crew"
        
        # Check for location mentions
        if any(term in query_lower for term in ['arabasta', 'water7', 'enies lobby', 'marineford', 'dressrosa']):
            intent_type = "location_focus"
            scene_type = "location"
        
        return {
            'intent_type': intent_type,
            'primary_character': primary_character,
            'secondary_characters': [],
            'scene_type': scene_type,
            'location': None,
            'action': None,
            'confidence': 0.6,
            'reasoning': 'Fallback keyword-based analysis'
        }
    
    def _parse_intent_response(self, llm_response: str) -> Dict[str, Any]:
        """
        Parse the LLM response for intent analysis.
        
        Args:
            llm_response: Raw LLM response string
            
        Returns:
            Parsed intent analysis dictionary
        """
        try:
            # Try to extract JSON from the response
            import json
            import re
            
            # Look for JSON content in the response
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                # Ensure all required fields are present
                required_fields = ['intent_type', 'confidence', 'reasoning']
                for field in required_fields:
                    if field not in parsed:
                        parsed[field] = 'unknown' if field == 'intent_type' else 0.5 if field == 'confidence' else ''
                
                return parsed
            
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM intent response: {e}")
        
        # Return default structure if parsing fails
        return {
            'intent_type': 'general_info',
            'primary_character': None,
            'secondary_characters': [],
            'scene_type': 'other',
            'location': None,
            'action': None,
            'confidence': 0.5,
            'reasoning': 'Failed to parse LLM response'
        }
    
    def _find_relevant_images(self, intent_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find relevant images based on intent analysis.
        
        Args:
            intent_analysis: Intent analysis results
            
        Returns:
            List of relevant image metadata
        """
        try:
            search_query = {}
            
            # Search by character if specified
            if intent_analysis.get('primary_character'):
                character = intent_analysis['primary_character'].lower()
                search_query['character'] = character
            
            # Search by image type if specified
            scene_type = intent_analysis.get('scene_type')
            if scene_type:
                # Map scene type to image type
                type_mapping = {
                    'crew': 'crew_group',
                    'battle': 'battle_combat',
                    'ship': 'ship',
                    'location': 'location_specific',
                    'form': 'character_form',
                    'training': 'training_development'
                }
                if scene_type in type_mapping:
                    search_query['image_type'] = type_mapping[scene_type]
            
            # Search by terms if available
            if intent_analysis.get('location'):
                search_query['terms'] = [intent_analysis['location']]
            elif intent_analysis.get('action'):
                search_query['terms'] = [intent_analysis['action']]
            
            # Execute search
            relevant_images = self.image_database.search_images(search_query)
            
            # If no specific results, try broader search
            if not relevant_images:
                self.logger.info("No specific results found, trying broader search")
                relevant_images = self.image_database.get_all_images()
            
            return relevant_images
            
        except Exception as e:
            self.logger.error(f"Error finding relevant images: {e}")
            return []
    
    def _score_image_relevance(self, images: List[Dict[str, Any]], intent_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Score images for relevance using LLM-based evaluation.
        
        Args:
            images: List of image metadata to score
            intent_analysis: Intent analysis results
            
        Returns:
            List of images with relevance scores
        """
        if not self.llm_client:
            self.logger.warning("LLM client not available, using fallback scoring")
            return self._fallback_image_scoring(images, intent_analysis)
        
        try:
            scored_images = []
            
            for image in images:
                # Create scoring prompt
                scoring_prompt = self._create_scoring_prompt(image, intent_analysis)
                
                # Get LLM score
                score = self._get_llm_image_score(scoring_prompt)
                
                # Add score to image metadata
                scored_image = image.copy()
                scored_image['relevance_score'] = score
                scored_images.append(scored_image)
            
            # Sort by relevance score (highest first)
            scored_images.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            self.logger.debug(f"Scored {len(scored_images)} images")
            return scored_images
            
        except Exception as e:
            self.logger.error(f"Error scoring images: {e}")
            return self._fallback_image_scoring(images, intent_analysis)
    
    def _create_scoring_prompt(self, image: Dict[str, Any], intent_analysis: Dict[str, Any]) -> str:
        """
        Create a prompt for LLM-based image scoring.
        
        Args:
            image: Image metadata
            intent_analysis: Intent analysis results
            
        Returns:
            Scoring prompt string
        """
        query = intent_analysis.get('reasoning', 'user query')
        character = image.get('character', 'unknown')
        content = image.get('content', 'unknown')
        image_type = image.get('type', 'unknown')
        
        prompt = f"""Rate how well this image matches the user's request.

User Request: {query}

Image Details:
- Character: {character}
- Content: {content}
- Type: {image_type}

Rate from 0.0 to 1.0 where:
0.0 = Completely irrelevant
0.5 = Somewhat relevant
1.0 = Perfectly matches the request

Respond with only the numerical score (e.g., 0.85)."""
        
        return prompt
    
    def _get_llm_image_score(self, prompt: str) -> float:
        """
        Get image relevance score from LLM.
        
        Args:
            prompt: Scoring prompt
            
        Returns:
            Relevance score (0.0-1.0)
        """
        try:
            system_message = "You are an expert at evaluating image relevance. Respond with only a numerical score between 0.0 and 1.0."
            
            # Log LLM call for image scoring
            if hasattr(self, 'pipeline_logger'):
                self.pipeline_logger.log_llm_call(
                    agent_name="IMAGE_RETRIEVAL_AGENT",
                    prompt=prompt,
                    response="",
                    tokens_used=0,
                    system_message=system_message
                )
            
            llm_response = self.llm_client.generate_text(
                prompt, 
                system_message, 
                max_tokens=10, 
                temperature=0.1
            )
            
            # Log LLM response for image scoring
            if hasattr(self, 'pipeline_logger'):
                self.pipeline_logger.log_llm_call(
                    agent_name="IMAGE_RETRIEVAL_AGENT",
                    prompt=prompt,
                    response=llm_response,
                    tokens_used=10,
                    system_message=system_message
                )
            
            # Extract numerical score
            import re
            score_match = re.search(r'0\.\d+|1\.0|0|1', llm_response.strip())
            if score_match:
                score = float(score_match.group())
                return max(0.0, min(1.0, score))  # Clamp to 0.0-1.0
            
            return 0.5  # Default score if parsing fails
            
        except Exception as e:
            self.logger.warning(f"Failed to get LLM image score: {e}")
            return 0.5
    
    def _fallback_image_scoring(self, images: List[Dict[str, Any]], intent_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fallback image scoring when LLM is not available.
        
        Args:
            images: List of image metadata to score
            intent_analysis: Intent analysis results
            
        Returns:
            List of images with basic relevance scores
        """
        scored_images = []
        
        for image in images:
            scored_image = image.copy()
            
            # Start with a higher base score for better matching
            score = 0.6  # Base score above threshold
            
            # Character match bonus
            if intent_analysis.get('primary_character'):
                query_character = intent_analysis['primary_character'].lower()
                image_character = image.get('character', '').lower()
                
                if query_character in image_character or image_character in query_character:
                    score += 0.3  # Character match bonus
                    score = min(1.0, score)
            
            # Type match bonus
            if intent_analysis.get('scene_type') == 'crew' and image.get('type') == 'crew_group':
                score += 0.2  # Crew type match bonus
                score = min(1.0, score)
            
            # Additional bonuses for specific matches
            if intent_analysis.get('intent_type') == 'crew_focus' and 'crew' in image.get('content', '').lower():
                score += 0.1  # Content relevance bonus
            
            if intent_analysis.get('scene_type') == 'ship' and any(term in image.get('content', '').lower() for term in ['ship', 'merry', 'sunny']):
                score += 0.1  # Ship content bonus
            
            scored_image['relevance_score'] = min(1.0, score)
            scored_images.append(scored_image)
        
        # Sort by score
        scored_images.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return scored_images
    
    def _select_best_image(self, scored_images: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Select the best image from scored candidates.
        
        Args:
            scored_images: List of images with relevance scores
            
        Returns:
            Best image metadata or None if no suitable images
        """
        if not scored_images:
            return None
        
        # Get the highest scoring image
        best_image = scored_images[0]
        best_score = best_image.get('relevance_score', 0)
        
        # Only return image if score is above threshold
        if best_score >= self.config.IMAGE_RELEVANCE_THRESHOLD:
            self.logger.info(f"Selected image with score {best_score}: {best_image.get('filename', 'unknown')}")
            return best_image
        else:
            self.logger.info(f"No image met relevance threshold {self.config.IMAGE_RELEVANCE_THRESHOLD}")
            return None
    
    def _prepare_response(self, best_image: Optional[Dict[str, Any]], 
                         intent_analysis: Dict[str, Any], 
                         scored_images: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare the final response with image retrieval results.
        
        Args:
            best_image: Selected best image or None
            intent_analysis: Intent analysis results
            scored_images: All scored images
            
        Returns:
            Complete response dictionary
        """
        response = {
            'success': True,
            'image': None,
            'intent_analysis': intent_analysis,
            'candidates_count': len(scored_images),
            'metadata': {
                'agent_type': 'image_retrieval',
                'total_images_considered': len(scored_images),
                'relevance_threshold': self.config.IMAGE_RELEVANCE_THRESHOLD,
                'database_total_images': self.image_database.total_images if self.image_database else 0
            }
        }
        
        if best_image:
            # Convert full path to relative path for frontend
            full_path = best_image['full_path']
            images_dir = str(self.config.IMAGES_PATH)
            if full_path.startswith(images_dir):
                relative_path = full_path[len(images_dir):].lstrip('\\/')
            else:
                # Fallback: construct relative path from folder and filename
                extension = best_image.get('extension', '')
                if extension and not extension.startswith('.'):
                    extension = '.' + extension
                relative_path = f"{best_image.get('folder', '')}/{best_image['filename']}{extension}"
            
            response['image'] = {
                'path': relative_path,  # Use relative path for frontend
                'full_path': full_path,  # Keep full path for reference
                'filename': best_image['filename'],
                'character': best_image['character'],
                'content': best_image['content'],
                'type': best_image['type'],
                'relevance_score': best_image.get('relevance_score', 0),
                'metadata': {
                    'folder': best_image.get('folder'),
                    'extension': best_image.get('extension'),
                    'searchable_terms': best_image.get('searchable_terms', [])
                }
            }
            
            response['metadata']['selected_image_score'] = best_image.get('relevance_score', 0)
            response['metadata']['selection_success'] = True
        else:
            response['metadata']['selection_success'] = False
            response['metadata']['reason'] = 'No image met relevance threshold'
        
        return response
