"""
Main Chatbot Class

The main chatbot class that provides a high-level interface for users
to interact with the One Piece chatbot system.
"""

from typing import Dict, Any, Optional, List
import logging
import time

from .orchestrator import ChatbotOrchestrator
from ..config import ChatbotConfig


class OnePieceChatbot:
    """
    Main One Piece chatbot class.
    
    This class provides a high-level interface for users to interact with
    the chatbot system, handling both text and image queries.
    """
    
    def __init__(self, config: Optional[ChatbotConfig] = None):
        """
        Initialize the One Piece chatbot.
        
        Args:
            config: Optional configuration instance. If not provided, uses default config.
        """
        self.config = config or ChatbotConfig()
        self.logger = self._setup_logger()
        
        # Initialize the orchestrator
        self.orchestrator = ChatbotOrchestrator(self.config)
        
        # Chatbot state
        self.is_ready = True
        self.startup_time = time.time()
        
        self.logger.info("One Piece Chatbot initialized successfully")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the chatbot."""
        logger = logging.getLogger("chatbot.main")
        logger.setLevel(self.config.LOG_LEVEL)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create console handler if not exists
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # Add file handler if enabled
            if self.config.LOG_TO_FILE:
                try:
                    file_handler = logging.FileHandler(self.config.LOG_FILE_PATH)
                    file_handler.setFormatter(formatter)
                    logger.addHandler(file_handler)
                except Exception as e:
                    logger.warning(f"Could not set up file logging: {e}")
        
        return logger
    
    def ask(self, question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Ask a text-based question to the chatbot.
        
        Args:
            question: The text question to ask
            session_id: Optional session identifier for conversation continuity
            
        Returns:
            Dictionary containing the response and metadata
        """
        if not self.is_ready:
            return self._get_not_ready_response()
        
        try:
            self.logger.info(f"Processing text question: {question[:100]}...")
            
            # Process the question through the orchestrator
            response = self.orchestrator.process_query(
                query=question,
                image_data=None,
                session_id=session_id
            )
            
            self.logger.info("Text question processed successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to process text question: {e}", exc_info=True)
            return self._get_error_response(str(e), session_id)
    
    def analyze_image(self, image_data: bytes, question: Optional[str] = None, 
                     session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze an image with an optional question.
        
        Args:
            image_data: The image data to analyze
            question: Optional question about the image
            session_id: Optional session identifier for conversation continuity
            
        Returns:
            Dictionary containing the analysis and metadata
        """
        if not self.is_ready:
            return self._get_not_ready_response()
        
        try:
            # Validate image data
            if not self._validate_image_data(image_data):
                return self._get_invalid_image_response(session_id)
            
            # Prepare query
            query = question or "What do you see in this image?"
            
            self.logger.info(f"Processing image analysis with question: {query[:100]}...")
            
            # Process the image analysis through the orchestrator
            response = self.orchestrator.process_query(
                query=query,
                image_data=image_data,
                session_id=session_id
            )
            
            self.logger.info("Image analysis completed successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to analyze image: {e}", exc_info=True)
            return self._get_error_response(str(e), session_id)
    
    def chat(self, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Send a chat message to the chatbot.
        
        This is an alias for the ask method, providing a more conversational interface.
        
        Args:
            message: The chat message to send
            session_id: Optional session identifier for conversation continuity
            
        Returns:
            Dictionary containing the response and metadata
        """
        return self.ask(message, session_id)
    
    def get_conversation_history(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get the conversation history for a session.
        
        Args:
            session_id: Optional session identifier. If not provided, returns current session.
            
        Returns:
            List of conversation turns
        """
        if not self.is_ready:
            return []
        
        try:
            # For now, return the current session history
            # In a full implementation, you'd support multiple sessions
            return self.orchestrator.get_conversation_summary()
            
        except Exception as e:
            self.logger.error(f"Failed to get conversation history: {e}")
            return []
    
    def get_chatbot_status(self) -> Dict[str, Any]:
        """
        Get the current status of the chatbot.
        
        Returns:
            Dictionary containing status information
        """
        try:
            status = {
                'is_ready': self.is_ready,
                'startup_time': self.startup_time,
                'uptime': time.time() - self.startup_time,
                'config': {
                    'log_level': self.config.LOG_LEVEL,
                    'response_time_target': self.config.RESPONSE_TIME_TARGET,
                    'enable_caching': self.config.ENABLE_CACHING,
                },
                'orchestrator_status': 'ready' if self.orchestrator else 'not_initialized',
            }
            
            # Add orchestrator metrics if available
            if self.orchestrator:
                try:
                    conversation_summary = self.orchestrator.get_conversation_summary()
                    agent_performance = self.orchestrator.get_agent_performance()
                    
                    status.update({
                        'conversation_summary': conversation_summary,
                        'agent_performance': agent_performance,
                    })
                except Exception as e:
                    self.logger.warning(f"Could not get orchestrator metrics: {e}")
                    status['orchestrator_metrics'] = 'unavailable'
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get chatbot status: {e}")
            return {
                'is_ready': False,
                'error': str(e),
                'status': 'error'
            }
    
    def reset_conversation(self, session_id: Optional[str] = None):
        """
        Reset the conversation for a session.
        
        Args:
            session_id: Optional session identifier. If not provided, resets current session.
        """
        if not self.is_ready:
            return
        
        try:
            self.orchestrator.reset_conversation()
            self.logger.info("Conversation reset successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to reset conversation: {e}")
    
    def end_session(self, session_id: Optional[str] = None):
        """
        End a conversation session.
        
        Args:
            session_id: Optional session identifier. If not provided, ends current session.
        """
        if not self.is_ready:
            return
        
        try:
            self.orchestrator.end_session()
            self.logger.info("Session ended successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to end session: {e}")
    
    def _validate_image_data(self, image_data: bytes) -> bool:
        """Validate the provided image data."""
        try:
            # Check file size
            if len(image_data) > self.config.MAX_IMAGE_SIZE:
                self.logger.warning(f"Image too large: {len(image_data)} bytes")
                return False
            
            # Check if it's a valid image by trying to open it
            from PIL import Image
            import io
            
            image = Image.open(io.BytesIO(image_data))
            image.verify()  # Verify the image
            
            # Check dimensions
            if image.width < self.config.IMAGE_QUALITY_THRESHOLD or image.height < self.config.IMAGE_QUALITY_THRESHOLD:
                self.logger.warning(f"Image resolution too low: {image.width}x{image.height}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Invalid image data: {e}")
            return False
    
    def _get_not_ready_response(self) -> Dict[str, Any]:
        """Get response when chatbot is not ready."""
        return {
            'response': "I'm sorry, but the chatbot is not ready at the moment. Please try again later.",
            'confidence': 0.0,
            'processing_time': 0.0,
            'session_id': None,
            'conversation_turn': 0,
            'metadata': {
                'agents_executed': [],
                'pipeline_success': False,
                'error': 'Chatbot not ready',
                'modality': 'text',
            }
        }
    
    def _get_invalid_image_response(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get response for invalid image data."""
        return {
            'response': "I'm sorry, but I couldn't process that image. Please make sure it's a valid image file and try again.",
            'confidence': 0.0,
            'processing_time': 0.0,
            'session_id': session_id,
            'conversation_turn': 0,
            'metadata': {
                'agents_executed': [],
                'pipeline_success': False,
                'error': 'Invalid image data',
                'modality': 'image',
            }
        }
    
    def _get_error_response(self, error_message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get response for general errors."""
        return {
            'response': f"I'm sorry, but I encountered an error: {error_message}. Please try again or rephrase your question.",
            'confidence': 0.0,
            'processing_time': 0.0,
            'session_id': session_id,
            'conversation_turn': 0,
            'metadata': {
                'agents_executed': [],
                'pipeline_success': False,
                'error': error_message,
                'modality': 'unknown',
            }
        }
    
    def cleanup(self):
        """Clean up resources and shutdown the chatbot."""
        try:
            if self.orchestrator:
                self.orchestrator.cleanup()
            
            self.is_ready = False
            self.logger.info("One Piece Chatbot cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during destruction
