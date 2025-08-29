"""
Router Agent

The router agent is responsible for analyzing user input, determining intent,
and selecting the appropriate agent pipeline for processing the query.
"""

from typing import Dict, Any, List, Optional
import re
from enum import Enum

from .base_agent import BaseAgent, AgentType, AgentInput
from ..config import ChatbotConfig
from ..utils.llm_client import LLMClient


class QueryIntent(Enum):
    """Enumeration of possible query intents."""
    SEARCH = "search"
    ANALYSIS = "analysis"
    CONVERSATION = "conversation"
    IMAGE_ANALYSIS = "image_analysis"
    TIMELINE = "timeline"
    RELATIONSHIP = "relationship"
    CHARACTER = "character"
    LOCATION = "location"
    UNKNOWN = "unknown"


class QueryComplexity(Enum):
    """Enumeration of query complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class Modality(Enum):
    """Enumeration of input modalities."""
    TEXT_ONLY = "text_only"
    IMAGE_ONLY = "image_only"
    MULTIMODAL = "multimodal"


class RouterAgent(BaseAgent):
    """
    Router agent for intelligent query routing and agent selection.
    
    This agent analyzes user input to determine:
    - Query intent and complexity
    - Input modality (text, image, or both)
    - Required agents for processing
    - Optimal execution order
    """
    
    def __init__(self, config: ChatbotConfig):
        """Initialize the router agent."""
        super().__init__(config, AgentType.ROUTER)
        
        # Initialize LLM client for enhanced intent detection
        try:
            self.llm_client = LLMClient(config)
            self.logger.info("LLM client initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize LLM client: {e}")
            self.llm_client = None
        
        # Intent detection patterns
        self.intent_patterns = {
            QueryIntent.SEARCH: [
                r'\b(what|who|where|when|how|why)\b',
                r'\b(tell me about|explain|describe|define)\b',
                r'\b(find|search|look for|locate)\b',
            ],
            QueryIntent.ANALYSIS: [
                r'\b(analyze|examine|investigate|study)\b',
                r'\b(compare|contrast|difference|similarity)\b',
                r'\b(relationship|connection|link|tie)\b',
            ],
            QueryIntent.TIMELINE: [
                r'\b(when|timeline|chronology|history)\b',
                r'\b(before|after|during|while)\b',
                r'\b(era|period|age|time)\b',
            ],
            QueryIntent.CHARACTER: [
                r'\b(character|person|individual|figure)\b',
                r'\b(pirate|marine|revolutionary|villain)\b',
                r'\b(crew|team|group|organization)\b',
            ],
            QueryIntent.LOCATION: [
                r'\b(place|location|island|kingdom)\b',
                r'\b(sea|ocean|world|realm)\b',
                r'\b(city|town|village|settlement)\b',
            ],
            QueryIntent.RELATIONSHIP: [
                r'\b(relationship|connection|bond|tie)\b',
                r'\b(friend|enemy|ally|rival)\b',
                r'\b(family|parent|child|sibling)\b',
            ],
        }
        
        # Complexity indicators
        self.complexity_indicators = {
            QueryComplexity.SIMPLE: [
                r'\b(simple|basic|easy|straightforward)\b',
                r'\b(what is|who is|where is)\b',
                r'\b(definition|meaning|explanation)\b',
            ],
            QueryComplexity.COMPLEX: [
                r'\b(complex|complicated|difficult|challenging)\b',
                r'\b(how does|why does|what causes)\b',
                r'\b(relationship|connection|interaction)\b',
                r'\b(compare|contrast|analyze)\b',
            ]
        }
        
        # One Piece specific keywords for enhanced intent detection
        self.one_piece_keywords = [
            'devil fruit', 'haki', 'nakama', 'pirate king', 'marine',
            'world government', 'revolutionary army', 'yonko', 'shichibukai',
            'grand line', 'new world', 'paradise', 'red line', 'calm belt'
        ]
    
    def _execute_agent(self, input_data: AgentInput) -> Dict[str, Any]:
        """
        Execute the router agent logic.
        
        Args:
            input_data: Input data containing query and context
            
        Returns:
            Dictionary containing routing decisions and agent pipeline
        """
        self.logger.info("Analyzing input for routing decisions")
        
        # Analyze input modality
        modality = self._determine_modality(input_data)
        
        # Detect query intent using LLM (handle empty query for image-only input)
        if modality == Modality.IMAGE_ONLY:
            intent = QueryIntent.IMAGE_ANALYSIS
        else:
            intent = self._detect_intent_with_llm(input_data.query)
        self.logger.info("Intent detected using LLM")
        
        # Assess query complexity (handle empty query for image-only input)
        if modality == Modality.IMAGE_ONLY:
            complexity = QueryComplexity.MODERATE  # Image analysis is typically moderate complexity
        else:
            complexity = self._assess_complexity(input_data.query)
        
        # Select required agents
        required_agents = self._select_agents(intent, complexity, modality)
        
        # Determine execution order
        execution_order = self._determine_execution_order(required_agents, intent)
        
        # Generate routing plan
        routing_plan = {
            'modality': modality.value,
            'intent': intent.value,
            'complexity': complexity.value,
            'required_agents': [agent.value for agent in required_agents],
            'execution_order': [agent.value for agent in execution_order],
            'confidence_score': self._calculate_confidence(intent, complexity, modality),
            'estimated_execution_time': self._estimate_execution_time(complexity, modality),
            'metadata': {
                'one_piece_keywords_detected': self._detect_one_piece_keywords(input_data.query or ""),
                'query_length': len(input_data.query or ""),
                'has_image': input_data.image_data is not None,
                'conversation_context': bool(input_data.conversation_history),
                'llm_used': True,
            }
        }
        
        self.logger.info(f"Routing plan generated: {routing_plan['intent']} query, "
                        f"{routing_plan['complexity']} complexity, "
                        f"{len(required_agents)} agents required")
        
        return routing_plan
    
    def _determine_modality(self, input_data: AgentInput) -> Modality:
        """Determine the input modality (text, image, or multimodal)."""
        has_text = bool(input_data.query and input_data.query.strip())
        has_image = input_data.image_data is not None
        
        if has_text and has_image:
            return Modality.MULTIMODAL
        elif has_image:
            return Modality.IMAGE_ONLY
        else:
            return Modality.TEXT_ONLY
    
    def _detect_intent_with_llm(self, query: str) -> QueryIntent:
        """Detect query intent using LLM for enhanced accuracy."""
        system_message = """You are an intent detection specialist for One Piece knowledge queries. 
        Analyze the user's query and determine the primary intent from these categories:
        - SEARCH: General information seeking (what, who, where, when, how, why)
        - ANALYSIS: Deep analysis, comparison, or investigation
        - CONVERSATION: Casual chat or follow-up questions
        - IMAGE_ANALYSIS: Questions about images or visual content
        - TIMELINE: Questions about chronology, history, or time periods
        - RELATIONSHIP: Questions about connections between characters or events
        - CHARACTER: Specific questions about characters or people
        - LOCATION: Specific questions about places or locations
        
        Respond with only the intent category name."""
        
        prompt = f"""Query: "{query}"
        
        What is the primary intent of this query? Respond with only the intent category."""
        
        llm_response = self.llm_client.generate_text(prompt, system_message, max_tokens=50)
        
        # Parse LLM response
        intent_text = llm_response.strip().upper()
        
        # Map to QueryIntent enum
        intent_mapping = {
            'SEARCH': QueryIntent.SEARCH,
            'ANALYSIS': QueryIntent.ANALYSIS,
            'CONVERSATION': QueryIntent.CONVERSATION,
            'IMAGE_ANALYSIS': QueryIntent.IMAGE_ANALYSIS,
            'TIMELINE': QueryIntent.TIMELINE,
            'RELATIONSHIP': QueryIntent.RELATIONSHIP,
            'CHARACTER': QueryIntent.CHARACTER,
            'LOCATION': QueryIntent.LOCATION,
        }
        
        if intent_text in intent_mapping:
            return intent_mapping[intent_text]
        else:
            # Default to SEARCH if LLM response is unclear
            return QueryIntent.SEARCH
    
    def _detect_intent(self, query: str) -> QueryIntent:
        """Detect the primary intent of the query using rule-based patterns."""
        query_lower = query.lower()
        
        # Check for explicit intent indicators
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        # Check for One Piece specific keywords
        if any(keyword in query_lower for keyword in self.one_piece_keywords):
            return QueryIntent.SEARCH
        
        # Default to search for One Piece related queries
        if self._is_one_piece_related(query):
            return QueryIntent.SEARCH
        
        return QueryIntent.UNKNOWN
    
    def _assess_complexity(self, query: str) -> QueryComplexity:
        """Assess the complexity level of the query."""
        query_lower = query.lower()
        
        # Check for complexity indicators
        for complexity, patterns in self.complexity_indicators.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return complexity
        
        # Heuristic complexity assessment
        word_count = len(query.split())
        question_marks = query.count('?')
        complex_indicators = ['why', 'how', 'explain', 'analyze', 'compare']
        
        if (word_count > 20 or question_marks > 1 or 
            any(indicator in query_lower for indicator in complex_indicators)):
            return QueryComplexity.COMPLEX
        elif word_count > 10 or question_marks > 0:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
    
    def _select_agents(self, intent: QueryIntent, complexity: QueryComplexity, 
                       modality: Modality) -> List[AgentType]:
        """Select the required agents based on intent, complexity, and modality."""
        required_agents = []
        
        # Always start with search agent for information retrieval
        required_agents.append(AgentType.SEARCH)
        
        # Add agents based on intent
        if intent == QueryIntent.IMAGE_ANALYSIS or modality == Modality.IMAGE_ONLY:
            required_agents.append(AgentType.IMAGE_ANALYSIS)
        
        if intent == QueryIntent.TIMELINE:
            required_agents.append(AgentType.TIMELINE)
        
        # Add reasoning agent for complex queries
        if complexity == QueryComplexity.COMPLEX:
            required_agents.append(AgentType.REASONING)
        
        # Always end with response agent
        required_agents.append(AgentType.RESPONSE)
        
        return required_agents
    
    def _determine_execution_order(self, required_agents: List[AgentType], 
                                  intent: QueryIntent) -> List[AgentType]:
        """Determine the optimal execution order for the selected agents."""
        # Start with search agent
        execution_order = [AgentType.SEARCH]
        
        # Add specialized agents in logical order
        if AgentType.IMAGE_ANALYSIS in required_agents:
            execution_order.append(AgentType.IMAGE_ANALYSIS)
        
        if AgentType.TIMELINE in required_agents:
            execution_order.append(AgentType.TIMELINE)
        
        if AgentType.REASONING in required_agents:
            execution_order.append(AgentType.REASONING)
        
        # End with response agent
        execution_order.append(AgentType.RESPONSE)
        
        return execution_order
    
    def _calculate_confidence(self, intent: QueryIntent, complexity: QueryComplexity, 
                             modality: Modality) -> float:
        """Calculate confidence score for the routing decision."""
        confidence = 0.8  # Base confidence
        
        # Adjust based on intent clarity
        if intent != QueryIntent.UNKNOWN:
            confidence += 0.1
        
        # Adjust based on complexity assessment
        if complexity != QueryComplexity.SIMPLE:
            confidence += 0.05
        
        # Adjust based on modality
        if modality == Modality.TEXT_ONLY:
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def _estimate_execution_time(self, complexity: QueryComplexity, 
                                modality: Modality) -> float:
        """Estimate execution time for the selected agent pipeline."""
        base_time = 30.0  # Base time in seconds
        
        # Adjust for complexity
        if complexity == QueryComplexity.COMPLEX:
            base_time *= 2.0
        elif complexity == QueryComplexity.MODERATE:
            base_time *= 1.5
        
        # Adjust for modality
        if modality == Modality.IMAGE_ONLY:
            base_time *= 1.3
        elif modality == Modality.MULTIMODAL:
            base_time *= 1.5
        
        return min(base_time, self.config.RESPONSE_TIME_TARGET)
    
    def _detect_one_piece_keywords(self, query: str) -> List[str]:
        """Detect One Piece specific keywords in the query."""
        query_lower = query.lower()
        detected_keywords = []
        
        for keyword in self.one_piece_keywords:
            if keyword in query_lower:
                detected_keywords.append(keyword)
        
        return detected_keywords
    
    def _is_one_piece_related(self, query: str) -> bool:
        """Check if the query is related to One Piece."""
        # This is a simple heuristic - in a real system, you might use
        # more sophisticated methods like embedding similarity
        one_piece_terms = [
            'one piece', 'luffy', 'zoro', 'nami', 'usopp', 'sanji',
            'chopper', 'robin', 'franky', 'brook', 'jimbe', 'ace',
            'sabo', 'dragon', 'garp', 'whitebeard', 'kaido', 'big mom',
            'shanks', 'blackbeard', 'roger', 'rayleigh'
        ]
        
        query_lower = query.lower()
        return any(term in query_lower for term in one_piece_terms)
    
    def _validate_input(self, input_data: AgentInput) -> bool:
        """Validate input data for the router agent."""
        # Call parent validation
        if not super()._validate_input(input_data):
            return False
        
        # Router agent specific validation can be added here if needed
        return True
