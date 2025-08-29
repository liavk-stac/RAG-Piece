"""
Image Analysis Agent

The image analysis agent is responsible for processing uploaded images,
generating detailed descriptions, and integrating visual information with
the RAG database for comprehensive analysis.
"""

from typing import Dict, Any, List, Optional
import base64
import io
from PIL import Image
import logging

from .base_agent import BaseAgent, AgentType, AgentInput
from ..config import ChatbotConfig
from ..utils.llm_client import LLMClient


class ImageAnalysisAgent(BaseAgent):
    """
    Image analysis agent for processing and understanding One Piece images.
    
    This agent:
    - Processes uploaded images and validates them
    - Generates detailed descriptions using vision models
    - Integrates visual information with RAG database
    - Provides comprehensive image analysis
    """
    
    def __init__(self, config: ChatbotConfig):
        """Initialize the image analysis agent."""
        super().__init__(config, AgentType.IMAGE_ANALYSIS)
        
        # Initialize LLM client for vision analysis
        try:
            self.llm_client = LLMClient(config)
            self.logger.info("LLM client initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize LLM client: {e}")
            self.llm_client = None
        
        # Initialize vision model (placeholder for now)
        self.vision_model = None
        self._initialize_vision_model()
    
    def _initialize_vision_model(self):
        """Initialize the vision model for image analysis."""
        try:
            # This would be replaced with actual vision model initialization
            # For now, we'll use a placeholder
            self.vision_model = "gpt-4o"  # Placeholder
            self.logger.info("Vision model initialized (placeholder)")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vision model: {e}")
            self.logger.warning("Image analysis will use fallback methods")
    
    def _execute_agent(self, input_data: AgentInput) -> Dict[str, Any]:
        """
        Execute the image analysis agent logic.
        
        Args:
            input_data: Input data containing image and context
            
        Returns:
            Dictionary containing image analysis results
        """
        self.logger.info("Executing image analysis")
        
        # Validate image input
        if not input_data.image_data:
            return {
                'success': False,
                'error_message': 'No image data provided',
                'confidence_score': 0.0,
                'analysis': None
            }
        
        # Process the image
        image_analysis = self._analyze_image(input_data)
        
        # Generate detailed description
        description = self._generate_image_description(image_analysis, input_data)
        
        # Cross-reference with RAG database
        rag_integration = self._integrate_with_rag(description, input_data)
        
        # Compile analysis results
        analysis_results = {
            'image_analysis': image_analysis,
            'description': description,
            'rag_integration': rag_integration,
            'confidence_score': self._calculate_analysis_confidence(image_analysis, rag_integration),
            'metadata': {
                'image_size': len(input_data.image_data),
                'processing_method': 'vision_model' if self.vision_model else 'fallback',
                'description_length': len(description),
                'rag_results_count': len(rag_integration.get('results', [])),
            }
        }
        
        self.logger.info("Image analysis completed successfully")
        
        return analysis_results
    
    def _analyze_image(self, input_data: AgentInput) -> Dict[str, Any]:
        """Analyze the uploaded image and extract basic information."""
        try:
            # Handle both raw bytes and base64-encoded data
            if isinstance(input_data.image_data, bytes):
                # Raw bytes - use directly
                image_bytes = input_data.image_data
            else:
                # Assume base64 string - decode
                image_bytes = base64.b64decode(input_data.image_data)
            
            image = Image.open(io.BytesIO(image_bytes))
            
            # Basic image analysis
            image_info = {
                'format': image.format,
                'mode': image.mode,
                'size': image.size,
                'width': image.width,
                'height': image.height,
                'aspect_ratio': image.width / image.height if image.height > 0 else 0,
            }
            
            # Validate image quality
            quality_assessment = self._assess_image_quality(image)
            image_info.update(quality_assessment)
            
            # Extract basic visual features
            visual_features = self._extract_visual_features(image)
            image_info.update(visual_features)
            
            return image_info
            
        except Exception as e:
            self.logger.error(f"Image analysis failed: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    def _assess_image_quality(self, image: Image.Image) -> Dict[str, Any]:
        """Assess the quality and suitability of the image for analysis."""
        quality_info = {
            'quality_score': 0.0,
            'quality_issues': [],
            'suitable_for_analysis': True
        }
        
        # Check resolution
        if image.width < self.config.IMAGE_QUALITY_THRESHOLD or image.height < self.config.IMAGE_QUALITY_THRESHOLD:
            quality_info['quality_issues'].append('Low resolution')
            quality_info['suitable_for_analysis'] = False
        
        # Check file size (approximate)
        estimated_size = image.width * image.height * len(image.getbands())
        if estimated_size < 10000:  # Very small image
            quality_info['quality_issues'].append('Very small image')
            quality_info['suitable_for_analysis'] = False
        
        # Check aspect ratio
        aspect_ratio = image.width / image.height
        if aspect_ratio > 3 or aspect_ratio < 0.33:  # Very wide or tall
            quality_info['quality_issues'].append('Extreme aspect ratio')
        
        # Calculate quality score
        if quality_info['suitable_for_analysis']:
            quality_info['quality_score'] = 0.8
            if not quality_info['quality_issues']:
                quality_info['quality_score'] = 1.0
        else:
            quality_info['quality_score'] = 0.3
        
        return quality_info
    
    def _extract_visual_features(self, image: Image.Image) -> Dict[str, Any]:
        """Extract basic visual features from the image."""
        features = {
            'dominant_colors': [],
            'brightness_level': 'medium',
            'contrast_level': 'medium',
            'has_text': False,
            'has_faces': False,
            'has_objects': True,
        }
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Analyze colors (simplified)
        colors = image.getcolors(maxcolors=1000)
        if colors:
            # Sort by frequency and get top colors
            sorted_colors = sorted(colors, key=lambda x: x[0], reverse=True)
            features['dominant_colors'] = [f"RGB{rgb}" for count, rgb in sorted_colors[:5]]
        
        # Analyze brightness (simplified)
        pixels = list(image.getdata())
        if pixels:
            avg_brightness = sum(sum(pixel) / 3 for pixel in pixels) / len(pixels)
            if avg_brightness < 85:
                features['brightness_level'] = 'dark'
            elif avg_brightness > 170:
                features['brightness_level'] = 'bright'
        
        # Simple text detection (placeholder)
        # In a real implementation, you'd use OCR or text detection models
        features['has_text'] = self._detect_text_simple(image)
        
        # Simple face detection (placeholder)
        # In a real implementation, you'd use face detection models
        features['has_faces'] = self._detect_faces_simple(image)
        
        return features
    
    def _detect_text_simple(self, image: Image.Image) -> bool:
        """Simple text detection (placeholder implementation)."""
        # This is a very basic heuristic - in practice, you'd use OCR
        # For now, we'll assume no text unless we have specific indicators
        return False
    
    def _detect_faces_simple(self, image: Image.Image) -> bool:
        """Simple face detection (placeholder implementation)."""
        # This is a very basic heuristic - in practice, you'd use face detection models
        # For now, we'll assume no faces unless we have specific indicators
        return False
    
    def _generate_image_description(self, image_analysis: Dict[str, Any], 
                                  input_data: AgentInput) -> str:
        """Generate a detailed description of the image using LLM vision capabilities."""
        if not image_analysis.get('success', True):
            return "Unable to analyze image due to processing errors."
        
        # Create a detailed prompt for image analysis
        prompt = """Please analyze this One Piece image and provide a detailed description including:
1. Characters visible in the image
2. Location or setting
3. Objects, weapons, or items present
4. Actions or events happening
5. Any text or symbols visible
6. Overall mood or atmosphere
7. Potential One Piece story context

Be specific and detailed in your description."""
        
        # Use LLM for image analysis
        description = self.llm_client.analyze_image(
            image_data=input_data.image_data,
            prompt=prompt
        )
        
        self.logger.info("Generated image description using LLM vision")
        return description
    
    def _enhance_with_one_piece_context(self, description: str, 
                                      image_analysis: Dict[str, Any]) -> str:
        """Enhance the description with One Piece specific context."""
        # This would be enhanced with actual One Piece image recognition
        # For now, we'll add some general context
        
        enhancement = ""
        
        # Check for potential One Piece elements based on visual features
        if 'dominant_colors' in image_analysis:
            colors = image_analysis['dominant_colors']
            if any('RGB(255, 0, 0)' in color for color in colors):  # Red
                enhancement += " Red coloring may indicate fire-based abilities or Marine affiliation. "
            if any('RGB(0, 0, 255)' in color for color in colors):  # Blue
                enhancement += " Blue coloring may indicate water themes or calm environments. "
            if any('RGB(255, 255, 0)' in color for color in colors):  # Yellow
                enhancement += " Yellow coloring may indicate lightning or energy-based elements. "
        
        # Check for potential character elements
        if image_analysis.get('has_faces', False):
            enhancement += " Human characters detected - may include One Piece characters. "
        
        # Check for potential location elements
        if image_analysis.get('brightness_level') == 'bright':
            enhancement += " Bright lighting suggests outdoor or well-lit environments. "
        elif image_analysis.get('brightness_level') == 'dark':
            enhancement += " Dark lighting suggests indoor or shadowy environments. "
        
        return description + enhancement
    
    def _integrate_with_rag(self, description: str, 
                           input_data: AgentInput) -> Dict[str, Any]:
        """Integrate image description with RAG database for enhanced analysis."""
        try:
            # Use the search agent's functionality to search the RAG database
            # with the image description
            search_query = f"Image analysis: {description}"
            
            # For now, we'll return a placeholder integration
            # In practice, this would call the RAG database
            rag_integration = {
                'search_query': search_query,
                'results': [],
                'integration_success': True,
                'metadata': {
                    'description_used': description[:200] + "..." if len(description) > 200 else description,
                    'search_terms_extracted': self._extract_search_terms(description),
                    'integration_method': 'description_based_search'
                }
            }
            
            return rag_integration
            
        except Exception as e:
            self.logger.error(f"RAG integration failed: {e}")
            return {
                'search_query': description,
                'results': [],
                'integration_success': False,
                'error': str(e),
                'metadata': {
                    'description_used': description[:200] + "..." if len(description) > 200 else description,
                    'integration_method': 'failed'
                }
            }
    
    def _extract_search_terms(self, description: str) -> List[str]:
        """Extract potential search terms from the image description."""
        # Simple term extraction - in practice, you'd use more sophisticated NLP
        words = description.lower().split()
        
        # Filter out common words and keep potentially relevant terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        relevant_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Remove duplicates and return
        return list(set(relevant_terms))
    
    def _calculate_analysis_confidence(self, image_analysis: Dict[str, Any], 
                                     rag_integration: Dict[str, Any]) -> float:
        """Calculate confidence score for the image analysis."""
        confidence = 0.5  # Base confidence
        
        # Adjust based on image quality
        if 'quality_score' in image_analysis:
            confidence += image_analysis['quality_score'] * 0.3
        
        # Adjust based on RAG integration success
        if rag_integration.get('integration_success', False):
            confidence += 0.2
        
        # Adjust based on analysis completeness
        if 'visual_features' in image_analysis:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _validate_input(self, input_data: AgentInput) -> bool:
        """Validate input data for the image analysis agent."""
        # Call parent validation
        if not super()._validate_input(input_data):
            return False
        
        # Image analysis agent requires image data
        if not input_data.image_data:
            self.logger.warning("Image analysis agent requires image data")
            return False
        
        # Validate image data format - handle both raw bytes and base64
        try:
            if isinstance(input_data.image_data, bytes):
                # Raw bytes - try to open as image
                Image.open(io.BytesIO(input_data.image_data))
            else:
                # Assume base64 string - try to decode
                base64.b64decode(input_data.image_data)
        except Exception as e:
            self.logger.warning(f"Invalid image data format: {e}")
            return False
        
        return True
