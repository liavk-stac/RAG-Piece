"""
Chatbot Orchestrator

The main orchestrator that coordinates the agent pipeline and manages
the conversation flow for the One Piece chatbot.
"""

from typing import Dict, Any, List, Optional
import time
import logging
from dataclasses import dataclass

from ..config import ChatbotConfig
from ..agents import (
    RouterAgent, SearchAgent, ReasoningAgent, ImageAnalysisAgent,
    ImageRetrievalAgent, ResponseAgent, TimelineAgent
)
from ..agents.base_agent import AgentInput, AgentOutput


@dataclass
class ConversationTurn:
    """Represents a single conversation turn."""
    query: str
    modality: str
    image_data: Optional[bytes] = None
    timestamp: float = 0.0
    response: Optional[str] = None
    agent_outputs: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    confidence: float = 0.0


class ChatbotOrchestrator:
    """
    Main orchestrator for the One Piece chatbot system.
    
    This class coordinates the execution of all agents in the pipeline,
    manages conversation state, and ensures proper flow between components.
    """
    
    def __init__(self, config: ChatbotConfig):
        """Initialize the chatbot orchestrator."""
        self.config = config
        self.logger = self._setup_logger()
        
        # Initialize all agents
        self.agents = self._initialize_agents()
        
        # Conversation state
        self.conversation_history: List[ConversationTurn] = []
        self.current_session_id: Optional[str] = None
        self.session_start_time: float = 0.0
        
        # Performance tracking
        self.total_queries_processed = 0
        self.average_response_time = 0.0
        self.successful_responses = 0
        self.failed_responses = 0
        
        self.logger.info("Chatbot orchestrator initialized successfully")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the orchestrator."""
        logger = logging.getLogger("chatbot.orchestrator")
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
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all the agents in the system."""
        agents = {}
        
        try:
            agents['router'] = RouterAgent(self.config)
            agents['search'] = SearchAgent(self.config)
            agents['reasoning'] = ReasoningAgent(self.config)
            agents['image_analysis'] = ImageAnalysisAgent(self.config)
            agents['image_retrieval'] = ImageRetrievalAgent(self.config)
            agents['response'] = ResponseAgent(self.config)
            agents['timeline'] = TimelineAgent(self.config)
            
            self.logger.info("All agents initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {e}")
            raise
        
        return agents
    
    def process_query(self, query: str, image_data: Optional[bytes] = None, 
                     session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user query through the agent pipeline.
        
        Args:
            query: User's text query
            image_data: Optional image data for analysis
            session_id: Optional session identifier
            
        Returns:
            Dictionary containing the response and metadata
        """
        start_time = time.time()
        
        try:
            # Start or continue session
            if session_id:
                self.current_session_id = session_id
            elif not self.current_session_id:
                self._start_new_session()
            
            # Create conversation turn
            conversation_turn = ConversationTurn(
                query=query,
                modality='multimodal' if image_data else 'text',
                image_data=image_data,
                timestamp=start_time
            )
            
            self.logger.info(f"Processing query: {query[:100]}...")
            
            # Execute the agent pipeline
            pipeline_result = self._execute_agent_pipeline(conversation_turn)
            
            # Update conversation turn with results
            conversation_turn.response = pipeline_result.get('response', '')
            conversation_turn.agent_outputs = pipeline_result.get('agent_outputs', {})
            conversation_turn.processing_time = time.time() - start_time
            conversation_turn.confidence = pipeline_result.get('confidence_score', 0.0)
            
            # Add to conversation history
            self._add_to_conversation_history(conversation_turn)
            
            # Update performance metrics
            self._update_performance_metrics(conversation_turn)
            
            # Prepare response
            response = {
                'response': conversation_turn.response,
                'confidence': conversation_turn.confidence,
                'processing_time': conversation_turn.processing_time,
                'session_id': self.current_session_id,
                'conversation_turn': len(self.conversation_history),
                'metadata': {
                    'agents_executed': list(pipeline_result.get('agent_outputs', {}).keys()),
                    'pipeline_success': True,
                    'modality': conversation_turn.modality,
                }
            }
            
            self.logger.info(f"Query processed successfully in {conversation_turn.processing_time:.2f}s")
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Query processing failed: {e}", exc_info=True)
            
            # Update failure metrics
            self.failed_responses += 1
            
            # Return error response
            return {
                'response': f"I apologize, but I encountered an error while processing your query: {str(e)}",
                'confidence': 0.0,
                'processing_time': processing_time,
                'session_id': self.current_session_id,
                'conversation_turn': len(self.conversation_history),
                'metadata': {
                    'agents_executed': [],
                    'pipeline_success': False,
                    'error': str(e),
                    'modality': conversation_turn.modality if 'conversation_turn' in locals() else 'unknown',
                }
            }
    
    def _execute_agent_pipeline(self, conversation_turn: ConversationTurn) -> Dict[str, Any]:
        """Execute the agent pipeline for the given conversation turn."""
        pipeline_start_time = time.time()
        agent_outputs = {}
        
        try:
            # Step 1: Router Agent - Determine execution plan
            self.logger.info("Executing router agent")
            router_input = AgentInput(
                query=conversation_turn.query,
                image_data=conversation_turn.image_data,
                conversation_history=self._get_conversation_context(),
                modality=conversation_turn.modality
            )
            
            router_output = self.agents['router'].execute(router_input)
            if not router_output.success:
                raise Exception(f"Router agent failed: {router_output.error_message}")
            
            routing_plan = router_output.result
            agent_outputs['router'] = routing_plan
            
            self.logger.info(f"Routing plan: {routing_plan['intent']} query, "
                           f"{routing_plan['complexity']} complexity")
            
            # Step 2: Execute required agents in order
            execution_order = routing_plan['execution_order']
            
            for agent_name in execution_order:
                if agent_name == 'router':
                    continue  # Already executed
                
                self.logger.info(f"Executing {agent_name} agent")
                
                # Prepare input for the agent
                agent_input = self._prepare_agent_input(agent_name, conversation_turn, agent_outputs)
                
                # Execute the agent
                agent_output = self.agents[agent_name].execute(agent_input)
                
                if agent_output.success:
                    agent_outputs[agent_name] = agent_output.result
                    self.logger.info(f"{agent_name} agent completed successfully")
                else:
                    self.logger.warning(f"{agent_name} agent failed: {agent_output.error_message}")
                    # Continue with other agents, but note the failure
                    agent_outputs[agent_name] = {
                        'success': False,
                        'error': agent_output.error_message,
                        'fallback_data': {}
                    }
            
            # Step 3: Response Agent - Generate final response
            self.logger.info("Executing response agent")
            response_input = AgentInput(
                query=conversation_turn.query,
                context={'agent_outputs': agent_outputs},
                conversation_history=self._get_conversation_context(),
                modality=conversation_turn.modality
            )
            
            response_output = self.agents['response'].execute(response_input)
            if not response_output.success:
                raise Exception(f"Response agent failed: {response_output.error_message}")
            
            agent_outputs['response'] = response_output.result
            
            # Compile pipeline result
            pipeline_result = {
                'response': response_output.result.get('response', ''),
                'confidence_score': response_output.result.get('confidence_score', 0.0),
                'agent_outputs': agent_outputs,
                'pipeline_execution_time': time.time() - pipeline_start_time,
                'routing_plan': routing_plan,
            }
            
            return pipeline_result
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise
    
    def _prepare_agent_input(self, agent_name: str, conversation_turn: ConversationTurn, 
                            agent_outputs: Dict[str, Any]) -> AgentInput:
        """Prepare input for a specific agent."""
        # Base input
        agent_input = AgentInput(
            query=conversation_turn.query,
            image_data=conversation_turn.image_data,
            conversation_history=self._get_conversation_context(),
            modality=conversation_turn.modality
        )
        
        # Add context based on agent type
        if agent_name == 'search':
            # Search agent gets the original query
            pass
        elif agent_name == 'image_analysis':
            # Image analysis agent gets image data and query context
            pass
        elif agent_name == 'image_retrieval':
            # Image retrieval agent gets query context and search results
            if 'search' in agent_outputs:
                agent_input.context = {
                    'search_results': agent_outputs['search'],
                    'query_intent': agent_outputs.get('router', {}).get('intent', '')
                }
        elif agent_name == 'reasoning':
            # Reasoning agent gets search results
            if 'search' in agent_outputs:
                agent_input.context = {'search_results': agent_outputs['search'].get('results', [])}
        elif agent_name == 'timeline':
            # Timeline agent gets search results
            if 'search' in agent_outputs:
                agent_input.context = {'search_results': agent_outputs['search'].get('results', [])}
        elif agent_name == 'response':
            # Response agent gets all agent outputs
            agent_input.context = {'agent_outputs': agent_outputs}
        
        return agent_input
    
    def _get_conversation_context(self) -> List[Dict[str, Any]]:
        """Get conversation context for agents."""
        context = []
        
        # Get recent conversation turns (within memory window)
        recent_turns = self.conversation_history[-self.config.CONVERSATION_MEMORY_WINDOW:]
        
        for turn in recent_turns:
            context.append({
                'query': turn.query,
                'response': turn.response,
                'timestamp': turn.timestamp,
                'modality': turn.modality,
                'confidence': turn.confidence,
            })
        
        return context
    
    def _start_new_session(self):
        """Start a new conversation session."""
        import uuid
        self.current_session_id = str(uuid.uuid4())
        self.session_start_time = time.time()
        self.conversation_history = []
        
        self.logger.info(f"Started new session: {self.current_session_id}")
    
    def _add_to_conversation_history(self, conversation_turn: ConversationTurn):
        """Add a conversation turn to the history."""
        self.conversation_history.append(conversation_turn)
        
        # Maintain memory size limit
        if len(self.conversation_history) > self.config.MAX_MEMORY_SIZE:
            self.conversation_history = self.conversation_history[-self.config.MAX_MEMORY_SIZE:]
        
        self.logger.debug(f"Added turn to conversation history. Total turns: {len(self.conversation_history)}")
    
    def _update_performance_metrics(self, conversation_turn: ConversationTurn):
        """Update performance tracking metrics."""
        self.total_queries_processed += 1
        
        # Update average response time
        if self.total_queries_processed == 1:
            self.average_response_time = conversation_turn.processing_time
        else:
            self.average_response_time = (
                (self.average_response_time * (self.total_queries_processed - 1) + 
                 conversation_turn.processing_time) / self.total_queries_processed
            )
        
        # Update success/failure counts
        if conversation_turn.response and conversation_turn.confidence > 0.3:
            self.successful_responses += 1
        else:
            self.failed_responses += 1
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the current conversation."""
        return {
            'session_id': self.current_session_id,
            'session_duration': time.time() - self.session_start_time if self.session_start_time else 0,
            'total_turns': len(self.conversation_history),
            'total_queries_processed': self.total_queries_processed,
            'successful_responses': self.successful_responses,
            'failed_responses': self.failed_responses,
            'average_response_time': self.average_response_time,
            'recent_queries': [turn.query[:100] + "..." if len(turn.query) > 100 else turn.query 
                              for turn in self.conversation_history[-5:]],
        }
    
    def get_agent_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all agents."""
        agent_performance = {}
        
        for agent_name, agent in self.agents.items():
            try:
                agent_performance[agent_name] = agent.get_performance_summary()
            except Exception as e:
                self.logger.warning(f"Could not get performance for {agent_name}: {e}")
                agent_performance[agent_name] = {'error': str(e)}
        
        return agent_performance
    
    def reset_conversation(self):
        """Reset the current conversation."""
        self.conversation_history = []
        self.logger.info("Conversation reset")
    
    def end_session(self):
        """End the current session."""
        if self.current_session_id:
            session_duration = time.time() - self.session_start_time
            self.logger.info(f"Ending session {self.current_session_id}. Duration: {session_duration:.2f}s")
            
            # Log session summary
            summary = self.get_conversation_summary()
            self.logger.info(f"Session summary: {summary}")
            
            # Reset session
            self.current_session_id = None
            self.session_start_time = 0
            self.conversation_history = []
    
    def cleanup(self):
        """Clean up resources and end session."""
        self.end_session()
        self.logger.info("Chatbot orchestrator cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
