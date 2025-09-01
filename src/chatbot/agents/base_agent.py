"""
Base Agent Interface

Defines the common interface and functionality that all chatbot agents must implement.
This provides a consistent structure for agent development and integration.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging
import time
from enum import Enum

from ..config import ChatbotConfig
from ..utils.pipeline_logger import get_pipeline_logger


class AgentType(Enum):
    """Enumeration of available agent types."""
    ROUTER = "router"
    SEARCH = "search"
    REASONING = "reasoning"
    IMAGE_ANALYSIS = "image_analysis"
    IMAGE_RETRIEVAL = "image_retrieval"
    RESPONSE = "response"
    TIMELINE = "timeline"


@dataclass
class AgentInput:
    """Standardized input structure for all agents."""
    query: str
    context: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    modality: str = "text"  # text, image, or multimodal
    image_data: Optional[bytes] = None
    conversation_history: Optional[List[Dict[str, Any]]] = None


@dataclass
class AgentOutput:
    """Standardized output structure for all agents."""
    success: bool
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    confidence_score: float = 0.0
    execution_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class BaseAgent(ABC):
    """
    Abstract base class for all chatbot agents.
    
    This class provides common functionality and enforces a consistent interface
    that all agents must implement. It handles logging, timing, error handling,
    and configuration management.
    """
    
    def __init__(self, config: ChatbotConfig, agent_type: AgentType):
        """
        Initialize the base agent.
        
        Args:
            config: Chatbot configuration instance
            agent_type: Type identifier for this agent
        """
        self.config = config
        self.agent_type = agent_type
        self.logger = self._setup_logger()
        self.pipeline_logger = get_pipeline_logger(config)
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.success_count = 0
        self.error_count = 0
        
        # Performance tracking
        self.performance_metrics = {
            'avg_execution_time': 0.0,
            'success_rate': 1.0,
            'last_execution_time': 0.0,
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for this agent."""
        logger = logging.getLogger(f"chatbot.agent.{self.agent_type.value}")
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
    
    def execute(self, input_data: AgentInput) -> AgentOutput:
        """
        Execute the agent with the given input.
        
        This is the main entry point for agent execution. It handles timing,
        error handling, and performance tracking.
        
        Args:
            input_data: Structured input data for the agent
            
        Returns:
            AgentOutput containing the execution results
        """
        start_time = time.time()
        self.execution_count += 1
        
        # Pipeline logging - agent start
        operation = f"Execute {self.agent_type.value} agent"
        input_dict = {
            'query': input_data.query,
            'modality': input_data.modality,
            'session_id': input_data.session_id,
            'has_image': input_data.image_data is not None,
            'context_keys': list(input_data.context.keys()) if input_data.context else [],
            'full_context': input_data.context,
            'conversation_history': input_data.conversation_history
        }
        self.pipeline_logger.log_agent_start(self.agent_type.value, operation, input_dict)
        
        try:
            self.logger.info(f"Executing {self.agent_type.value} agent")
            self.logger.debug(f"Input: {input_data}")
            
            # Validate input
            if not self._validate_input(input_data):
                execution_time = time.time() - start_time
                error_output = AgentOutput(
                    success=False,
                    error_message="Invalid input data",
                    confidence_score=0.0,
                    execution_time=execution_time
                )
                
                # Pipeline logging - agent end (failure)
                self.pipeline_logger.log_agent_end(
                    self.agent_type.value, operation, 
                    output_data={'error': 'Invalid input data'}, 
                    execution_time=execution_time, success=False
                )
                
                return error_output
            
            # Execute agent-specific logic
            result = self._execute_agent(input_data)
            
            # Update performance metrics
            execution_time = time.time() - start_time
            self._update_performance_metrics(execution_time, True)
            
            self.logger.info(f"Agent execution completed successfully in {execution_time:.2f}s")
            
            # Pipeline logging - agent end (success)
            # Log the complete result data
            self.pipeline_logger.log_agent_end(
                self.agent_type.value, operation, 
                output_data=result, 
                execution_time=execution_time, success=True
            )
            
            return AgentOutput(
                success=True,
                result=result,
                confidence_score=result.get('confidence_score', 0.8),
                execution_time=execution_time,
                metadata=result.get('metadata', {})
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Agent execution failed: {str(e)}"
            
            self.logger.error(error_msg, exc_info=True)
            self._update_performance_metrics(execution_time, False)
            
            # Pipeline logging - agent error
            self.pipeline_logger.log_agent_error(
                self.agent_type.value, operation, e, input_dict
            )
            
            return AgentOutput(
                success=False,
                error_message=error_msg,
                confidence_score=0.0,
                execution_time=execution_time
            )
    
    @abstractmethod
    def _execute_agent(self, input_data: AgentInput) -> Dict[str, Any]:
        """
        Execute the agent-specific logic.
        
        This method must be implemented by all concrete agent classes.
        It contains the actual agent functionality.
        
        Args:
            input_data: Validated input data
            
        Returns:
            Dictionary containing the agent's output and metadata
        """
        pass
    
    def _validate_input(self, input_data: AgentInput) -> bool:
        """
        Validate the input data for this agent.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        # Basic validation - require either query or image data
        if not input_data.query and not input_data.image_data:
            self.logger.warning("No query or image data provided")
            return False
        
        # Agent-specific validation can be implemented in subclasses
        return True
    
    def _update_performance_metrics(self, execution_time: float, success: bool):
        """Update performance tracking metrics."""
        self.total_execution_time += execution_time
        self.performance_metrics['last_execution_time'] = execution_time
        
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        
        # Calculate averages
        if self.execution_count > 0:
            self.performance_metrics['avg_execution_time'] = (
                self.total_execution_time / self.execution_count
            )
            self.performance_metrics['success_rate'] = (
                self.success_count / self.execution_count
            )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of agent performance metrics."""
        return {
            'agent_type': self.agent_type.value,
            'execution_count': self.execution_count,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'total_execution_time': self.total_execution_time,
            'avg_execution_time': self.performance_metrics['avg_execution_time'],
            'success_rate': self.performance_metrics['success_rate'],
            'last_execution_time': self.performance_metrics['last_execution_time'],
        }
    
    def reset_metrics(self):
        """Reset performance metrics (useful for testing)."""
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.success_count = 0
        self.error_count = 0
        self.performance_metrics = {
            'avg_execution_time': 0.0,
            'success_rate': 1.0,
            'last_execution_time': 0.0,
        }
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.agent_type.value.capitalize()}Agent(executions={self.execution_count})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the agent."""
        return f"{self.__class__.__name__}(type={self.agent_type.value}, config={self.config})"
