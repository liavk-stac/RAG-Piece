"""
Chatbot Configuration Module

Centralized configuration management for the One Piece chatbot system.
All parameters are organized in logical categories for easy tuning and maintenance.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Force reload to ensure environment variables are available
    load_dotenv(override=True)
except ImportError:
    # dotenv not available, continue without it
    pass


@dataclass
class ChatbotConfig:
    """
    Centralized configuration for the One Piece chatbot system.
    
    This class contains all configurable parameters organized in logical categories
    for easy tuning and maintenance. All parameters have sensible defaults.
    """
    
    # === AGENT SETTINGS ===
    AGENT_TIMEOUT: int = 120  # Maximum execution time per agent (seconds)
    AGENT_MAX_RETRIES: int = 3  # Maximum retry attempts for agent execution
    PIPELINE_EXECUTION_TIMEOUT: int = 300  # Total pipeline execution timeout (seconds)
    AGENT_SELECTION_THRESHOLD: float = 0.7  # Confidence threshold for agent selection
    ENABLE_IMAGE_RETRIEVAL_AGENT: bool = True  # Enable/disable image retrieval agent (disabled for RAGAS eval)
    
    # === MEMORY SETTINGS ===
    CONVERSATION_MEMORY_WINDOW: int = 10  # Number of conversation turns to remember
    SESSION_TIMEOUT: int = 3600  # Session timeout in seconds (1 hour)
    MEMORY_CLEANUP_INTERVAL: int = 300  # Memory cleanup interval (5 minutes)
    MAX_MEMORY_SIZE: int = 1000  # Maximum memory entries to store
    
    # === TOOL SETTINGS ===
    RAG_SEARCH_LIMIT: int = 4  # Maximum search results from RAG database
    IMAGE_PROCESSING_TIMEOUT: int = 60  # Image processing timeout (seconds)
    TOOL_EXECUTION_TIMEOUT: int = 30  # Individual tool execution timeout (seconds)
    MAX_TOOL_RETRIES: int = 2  # Maximum retry attempts for tools
    
    # === PERFORMANCE SETTINGS ===
    RESPONSE_TIME_TARGET: int = 120  # Target response time in seconds (2 minutes)
    ENABLE_CACHING: bool = True  # Enable response and tool result caching
    CACHE_TTL: int = 3600  # Cache time-to-live in seconds (1 hour)
    MAX_CONCURRENT_REQUESTS: int = 1  # Single user system
    
    # === ONE PIECE SETTINGS ===
    LORE_VALIDATION_ENABLED: bool = True  # Enable One Piece lore validation
    CANONICAL_SOURCES_ONLY: bool = True  # Use only canonical information sources
    WORLD_KNOWLEDGE_CONSTRAINTS: bool = True  # Apply world knowledge constraints
    
    # === IMAGE PROCESSING SETTINGS ===
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # Maximum image size (10MB)
    SUPPORTED_IMAGE_FORMATS: List[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".webp"])
    IMAGE_RESIZE_DIMENSIONS: tuple = (1024, 1024)  # Target image dimensions
    IMAGE_QUALITY_THRESHOLD: int = 100  # Minimum image resolution (pixels)
    
    # === IMAGE RETRIEVAL SETTINGS ===
    IMAGES_PATH: str = "data/images"  # Path to images directory
    IMAGE_INDEX_PATH: str = "data/image_index.pkl"  # Path to image index file
    IMAGE_RELEVANCE_THRESHOLD: float = 0.3  # Minimum relevance score for image selection
    ENABLE_IMAGE_RETRIEVAL: bool = True  # Enable image retrieval functionality
    
    # === LLM MODEL SETTINGS ===
    LLM_MODEL_NAME: str = "gpt-4o-mini"  # Primary LLM model for agents
    LLM_MODEL_TEMPERATURE: float = 0.3  # Temperature for balanced creativity
    MAX_LLM_TOKENS: int = 4000  # Maximum tokens for LLM input/output
    LLM_REQUEST_TIMEOUT: int = 60  # LLM API request timeout (seconds)
    LLM_MAX_RETRIES: int = 3  # Maximum retries for LLM API calls
    
    # === VISION MODEL SETTINGS ===
    VISION_MODEL_NAME: str = "gpt-4o"  # Vision model for image analysis
    VISION_MODEL_TEMPERATURE: float = 0.1  # Temperature for consistent descriptions
    MAX_VISION_TOKENS: int = 4000  # Maximum tokens for vision model input
    VISION_DESCRIPTION_DETAIL: str = "high"  # Detail level: low, medium, high
    
    # === LOGGING SETTINGS ===
    LOG_LEVEL: str = "WARNING"  # Logging level (DEBUG, INFO, WARNING, ERROR) - Reduced for RAGAS eval
    ENABLE_VERBOSE_LOGGING: bool = False  # Enable detailed logging
    LOG_TO_FILE: bool = True  # Log to file in addition to console
    LOG_FILE_PATH: str = "logs/one_piece_pipeline.log"  # Main pipeline log file
    ENABLE_PIPELINE_LOGGING: bool = False  # Enable comprehensive pipeline logging - Disabled for speed
    ENABLE_LLM_CALL_LOGGING: bool = False  # Log all LLM API calls - Disabled for speed
    ENABLE_LANGCHAIN_LOGGING: bool = False  # Enable LangChain verbose logging
    LOG_MAX_SIZE: int = 100 * 1024 * 1024  # Max log file size (100MB)
    LOG_BACKUP_COUNT: int = 5  # Number of backup log files
    
    # === WEB INTERFACE SETTINGS ===
    WEB_HOST: str = "localhost"  # Web interface host
    WEB_PORT: int = 8080  # Web interface port
    WEB_DEBUG: bool = False  # Enable web interface debug mode
    WEB_RELOAD: bool = True  # Enable auto-reload for development
    
    # === RAG INTEGRATION SETTINGS ===
    RAG_DATABASE_PATH: str = "data/rag_db"  # Path to RAG database
    RAG_INDEX_PATH: str = "data/rag_db/whoosh_index"  # Path to search index
    RAG_FAISS_PATH: str = "data/rag_db/faiss_index.bin"  # Path to FAISS index
    RAG_CHUNK_MAPPING_PATH: str = "data/rag_db/chunk_mapping.pkl"  # Path to chunk mapping
    
    # === ERROR HANDLING SETTINGS ===
    ENABLE_FALLBACK_MECHANISMS: bool = True  # Enable fallback mechanisms
    GRACEFUL_DEGRADATION: bool = True  # Enable graceful degradation
    USER_FRIENDLY_ERRORS: bool = True  # Show user-friendly error messages
    ERROR_RETRY_ENABLED: bool = True  # Enable automatic error retry
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        # Ensure environment variables are loaded
        self._ensure_env_loaded()
        self.update_from_env()  # Load environment variables first
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate configuration parameters and set defaults if needed."""
        # Ensure positive values for timeouts
        if self.AGENT_TIMEOUT <= 0:
            self.AGENT_TIMEOUT = 120
        
        if self.PIPELINE_EXECUTION_TIMEOUT <= 0:
            self.PIPELINE_EXECUTION_TIMEOUT = 300
        
        # Ensure memory window is reasonable
        if self.CONVERSATION_MEMORY_WINDOW < 1:
            self.CONVERSATION_MEMORY_WINDOW = 10
        
        if self.CONVERSATION_MEMORY_WINDOW > 100:
            self.CONVERSATION_MEMORY_WINDOW = 100
        
        # Validate image formats
        if not self.SUPPORTED_IMAGE_FORMATS:
            self.SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".webp"]
        
        # Ensure response time target is reasonable
        if self.RESPONSE_TIME_TARGET < 30:
            self.RESPONSE_TIME_TARGET = 30
        
        if self.RESPONSE_TIME_TARGET > 300:
            self.RESPONSE_TIME_TARGET = 300
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_')
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ChatbotConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def _ensure_env_loaded(self):
        """Ensure environment variables are loaded from .env file."""
        try:
            from dotenv import load_dotenv
            # Force reload environment variables
            load_dotenv(override=True)
        except ImportError:
            # dotenv not available, continue without it
            pass
    
    def update_from_env(self):
        """Update configuration from environment variables."""
        env_mapping = {
            'CHATBOT_AGENT_TIMEOUT': 'AGENT_TIMEOUT',
            'CHATBOT_RESPONSE_TIME_TARGET': 'RESPONSE_TIME_TARGET',
            'CHATBOT_LOG_LEVEL': 'LOG_LEVEL',
            'CHATBOT_WEB_PORT': 'WEB_PORT',
            'CHATBOT_VISION_MODEL': 'VISION_MODEL_NAME',
            'CHATBOT_LLM_MODEL': 'LLM_MODEL_NAME',
        }
        
        for env_var, config_attr in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    # Try to convert to appropriate type
                    current_value = getattr(self, config_attr)
                    if isinstance(current_value, bool):
                        setattr(self, config_attr, env_value.lower() in ('true', '1', 'yes'))
                    elif isinstance(current_value, int):
                        setattr(self, config_attr, int(env_value))
                    elif isinstance(current_value, float):
                        setattr(self, config_attr, float(env_value))
                    else:
                        setattr(self, config_attr, env_value)
                except (ValueError, TypeError):
                    # Skip invalid environment variable values
                    continue


# Global configuration instance
default_config = ChatbotConfig()
