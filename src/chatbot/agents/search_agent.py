"""
Search Agent

The search agent is responsible for retrieving information from the existing RAG database.
It leverages the proven SearchEngine class for robust BM25 + FAISS search infrastructure.
"""

from typing import Dict, Any, List, Optional
import sys
import os

# Add the parent directory to the path to import RAG components
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from .base_agent import BaseAgent, AgentType, AgentInput
from ..config import ChatbotConfig
from ..utils.llm_client import LLMClient


class SearchAgent(BaseAgent):
    """
    Search agent for RAG database information retrieval.
    
    This agent interfaces with the existing One Piece RAG database using the proven SearchEngine:
    - Executes hybrid BM25 + semantic search via SearchEngine.search()
    - Retrieves relevant information and context with rich metadata
    - Provides search results with comprehensive scoring and ranking
    - Handles different types of search queries with intelligent enhancement
    """
    
    def __init__(self, config: ChatbotConfig):
        """Initialize the search agent."""
        super().__init__(config, AgentType.SEARCH)
        
        # Initialize LLM client for query enhancement
        try:
            self.llm_client = LLMClient(config)
            self.logger.info("LLM client initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize LLM client: {e}")
            self.llm_client = None
        
        # Initialize SearchEngine integration
        self.search_engine = None
        self._initialize_search_engine()
    
    def _initialize_search_engine(self):
        """Initialize the proven SearchEngine from the RAG project."""
        try:
            # Import the proven SearchEngine and RAGConfig
            from ...rag_piece.search import SearchEngine
            from ...rag_piece.config import RAGConfig
            
            # Load RAG configuration
            rag_config = RAGConfig()
            rag_config.BM25_CANDIDATES = self.config.RAG_SEARCH_LIMIT
            rag_config.FINAL_RESULTS = min(self.config.RAG_SEARCH_LIMIT, 10)
            
            # Initialize SearchEngine with the existing database
            self.search_engine = SearchEngine(rag_config)
            
            # Load the existing search indices
            success = self.search_engine.load_indices()
            if not success:
                raise Exception("Failed to load search indices")
            
            self.logger.info("SearchEngine integration established successfully")
            
        except ImportError as e:
            self.logger.error(f"Could not import SearchEngine: {e}")
            raise Exception("SearchEngine is required and must be available")
        except Exception as e:
            self.logger.error(f"Failed to initialize SearchEngine: {e}")
            raise Exception("SearchEngine initialization failed")
    
    def _execute_agent(self, input_data: AgentInput) -> Dict[str, Any]:
        """
        Execute the search agent logic using the proven SearchEngine.
        
        Args:
            input_data: Input data containing query and context
            
        Returns:
            Dictionary containing search results and metadata
        """
        self.logger.info(f"Executing search for query: {input_data.query[:100]}...")
        
        # Prepare search query
        search_query = self._prepare_search_query(input_data)
        
        # Execute search using SearchEngine
        search_results = self._execute_search_with_engine(search_query, input_data)
        
        # Process and format results
        processed_results = self._process_search_results(search_results, input_data)
        
        # Log RAG retrieval with complete metadata
        if hasattr(self, 'pipeline_logger'):
            # Extract chunks and scores for logging
            chunks = [result.get('content', '') for result in processed_results]
            scores = [result.get('combined_score', 0.0) for result in processed_results]
            
            # Get search metadata for logging
            search_metadata = {
                'search_parameters': search_query,
                'search_performance': {
                    'query_enhancement': True,
                    'context_integration': bool(input_data.conversation_history),
                    'one_piece_terms_used': len(search_query.get('one_piece_terms', [])),
                    'search_engine_used': 'SearchEngine',
                    'hybrid_search': True,
                },
                'query_modifications': search_query.get('modifications', []),
            }
            
            self.pipeline_logger.log_rag_retrieval(
                input_data.query, 
                processed_results, 
                scores, 
                search_metadata
            )
        
        # Generate search summary
        search_summary = {
            'query': input_data.query,
            'search_strategy': search_query.get('strategy', 'default'),
            'results_count': len(processed_results),
            'total_results_found': len(search_results),
            'search_time': 0,  # SearchEngine handles timing internally
            'confidence_score': self._calculate_search_confidence(search_results),
            'results': processed_results,
            'metadata': {
                'search_parameters': search_query,
                'search_performance': {
                    'query_enhancement': True,
                    'context_integration': bool(input_data.conversation_history),
                    'one_piece_terms_used': len(search_query.get('one_piece_terms', [])),
                    'search_engine_used': 'SearchEngine',
                    'hybrid_search': True,
                },
                'query_modifications': search_query.get('modifications', []),
            }
        }
        
        self.logger.info(f"Search completed: {len(processed_results)} results found")
        
        return search_summary
    
    def _prepare_search_query(self, input_data: AgentInput) -> Dict[str, Any]:
        """Prepare and optimize the search query using LLM if available."""
        query = input_data.query.strip()
        
        # Extract One Piece specific terms
        one_piece_terms = self._extract_one_piece_terms(query)
        
        # Use LLM for query enhancement
        enhanced_query = self._enhance_query_with_llm(query, input_data)
        self.logger.info("Query enhanced using LLM")
        
        # Determine search strategy
        search_strategy = self._determine_search_strategy(query, one_piece_terms)
        
        # Generate search parameters
        search_params = {
            'original_query': query,
            'enhanced_query': enhanced_query,
            'strategy': search_strategy,
            'one_piece_terms': one_piece_terms,
            'search_limit': self.config.RAG_SEARCH_LIMIT,
            'modality': input_data.modality,
            'conversation_context': input_data.conversation_history,
            'llm_enhanced': True,
        }
        
        return search_params
    
    def _extract_one_piece_terms(self, query: str) -> List[str]:
        """Extract One Piece specific terms from the query."""
        # Common One Piece terms and their variations
        one_piece_terms = {
            'luffy': ['monkey d luffy', 'straw hat luffy', 'straw hat', 'luffy'],
            'zoro': ['roronoa zoro', 'pirate hunter zoro', 'zoro'],
            'nami': ['nami', 'cat burglar nami'],
            'usopp': ['usopp', 'god usopp', 'sogeking'],
            'sanji': ['sanji', 'black leg sanji', 'vinsmoke sanji'],
            'chopper': ['tony tony chopper', 'chopper', 'reindeer'],
            'robin': ['nico robin', 'robin', 'devil child'],
            'franky': ['franky', 'cutty flam', 'cyborg'],
            'brook': ['brook', 'soul king', 'skeleton'],
            'jimbe': ['jimbe', 'jinbe', 'first son of the sea'],
            'ace': ['portgas d ace', 'ace', 'fire fist ace'],
            'sabo': ['sabo', 'flame emperor sabo'],
            'whitebeard': ['edward newgate', 'whitebeard', 'yonko'],
            'kaido': ['kaido', 'king of beasts', 'yonko'],
            'big mom': ['charlotte linlin', 'big mom', 'yonko'],
            'shanks': ['red hair shanks', 'shanks', 'yonko'],
            'blackbeard': ['marshall d teach', 'blackbeard', 'yonko'],
            'roger': ['gol d roger', 'gold roger', 'pirate king'],
            'devil fruit': ['devil fruit', 'akuma no mi', 'fruit'],
            'haki': ['haki', 'observation haki', 'armament haki', 'conqueror haki'],
            'nakama': ['nakama', 'crew', 'friends', 'comrades'],
        }
        
        extracted_terms = []
        query_lower = query.lower()
        
        for term, variations in one_piece_terms.items():
            if any(var in query_lower for var in variations):
                extracted_terms.append(term)
        
        return extracted_terms
    
    def _determine_search_strategy(self, query: str, one_piece_terms: List[str]) -> str:
        """Determine the optimal search strategy for the query using LLM."""
        try:
            return self._llm_detect_strategy(query, one_piece_terms)
        except Exception as e:
            self.logger.error(f"LLM strategy detection failed: {e}")
            # Minimal safe default
            return 'general_search'

    def _llm_detect_strategy(self, query: str, one_piece_terms: List[str]) -> str:
        """Use LLM to classify the query into a search strategy label.
        Expected labels: character_search | location_search | timeline_search |
                         explanatory_search | relationship_search | term_specific_search | general_search
        """
        system_message = (
            "You classify One Piece queries into one of these exact labels: "
            "character_search, location_search, timeline_search, explanatory_search, "
            "relationship_search, term_specific_search, general_search. "
            "- character_search: asks about who a person is, identities, or character details. "
            "- location_search: asks about places, islands, or where something is. "
            "- timeline_search: asks when/chronology/history/era/time ordering. "
            "- explanatory_search: asks how something works or asks for explanations. "
            "- relationship_search: asks relationships/connections between entities. "
            "- term_specific_search: contains One Piece specific terms (e.g., devil fruit, haki, named characters). "
            "- general_search: any other general info query not fitting above. "
            "Return only the single label, no extra text."
        )
        prompt = (
            f"Query: {query}\n"
            f"Detected_terms: {', '.join(one_piece_terms) if one_piece_terms else 'none'}\n"
            "Label:"
        )
        # Log LLM call for search strategy detection
        if hasattr(self, 'pipeline_logger'):
            self.pipeline_logger.log_llm_call(
                agent_name="SEARCH_AGENT",
                prompt=prompt,
                response="",
                tokens_used=0,
                system_message=system_message
            )
        
        label = self.llm_client.generate_text(
            prompt,
            system_message,
            max_tokens=10,
            temperature=0.0
        )
        
        # Log LLM response for search strategy detection
        if hasattr(self, 'pipeline_logger'):
            self.pipeline_logger.log_llm_call(
                agent_name="SEARCH_AGENT",
                prompt=prompt,
                response=label,
                tokens_used=10,
                system_message=system_message
            )
        
        label = label.strip().lower()
        # Normalize any unexpected output
        allowed = {
            'character_search','location_search','timeline_search','explanatory_search',
            'relationship_search','term_specific_search','general_search'
        }
        return label if label in allowed else 'general_search'
    
    def _enhance_query_with_llm(self, query: str, input_data: AgentInput) -> str:
        """Enhance the query using LLM for better search results."""
        try:
            system_message = """You are a search query enhancement specialist for One Piece knowledge. 
            Take the user's query and enhance it with relevant One Piece terms, context, and synonyms 
            to improve search results. Keep the enhanced query concise but comprehensive."""
            
            prompt = f"""Original query: "{query}"

Please enhance this query for searching a One Piece knowledge database. Consider:
1. Adding relevant One Piece character names, locations, or terms
2. Including many many synonyms or alternative names
3. Adding context if the query seems generic
4. Maintaining the original intent while improving searchability

Enhanced query:"""
            
            # Log LLM call for query enhancement
            if hasattr(self, 'pipeline_logger'):
                self.pipeline_logger.log_llm_call(
                    agent_name="SEARCH_AGENT",
                    prompt=prompt,
                    response="",
                    tokens_used=0,
                    system_message=system_message
                )
            
            enhanced_query = self.llm_client.generate_text(prompt, system_message, max_tokens=100)
            
            # Log LLM response for query enhancement
            if hasattr(self, 'pipeline_logger'):
                self.pipeline_logger.log_llm_call(
                    agent_name="SEARCH_AGENT",
                    prompt=prompt,
                    response=enhanced_query,
                    tokens_used=100,
                    system_message=system_message
                )
            
            # Clean up the response
            enhanced_query = enhanced_query.strip().strip('"')
            if enhanced_query and enhanced_query != query:
                return enhanced_query
            else:
                return query
                
        except Exception as e:
            self.logger.error(f"LLM query enhancement failed: {e}")
            return query
    
    def _enhance_query_with_context(self, query: str, input_data: AgentInput) -> str:
        """Enhance the query with conversation context and additional information."""
        enhanced_query = query
        
        # Add conversation context if available
        if input_data.conversation_history:
            context_keywords = self._extract_context_keywords(input_data.conversation_history)
            if context_keywords:
                enhanced_query += f" {', '.join(context_keywords)}"
        
        # Add One Piece context if query seems generic
        if not self._is_one_piece_specific(query):
            enhanced_query += " One Piece"
        
        return enhanced_query
    
    def _extract_context_keywords(self, conversation_history: List[Dict[str, Any]]) -> List[str]:
        """Extract relevant keywords from conversation history."""
        # Simple keyword extraction from recent conversation
        keywords = []
        recent_messages = conversation_history[-3:]  # Last 3 messages
        
        for message in recent_messages:
            if 'query' in message:
                # Extract One Piece terms from previous queries
                terms = self._extract_one_piece_terms(message['query'])
                keywords.extend(terms)
        
        return list(set(keywords))  # Remove duplicates
    
    def _is_one_piece_specific(self, query: str) -> bool:
        """Check if the query is specifically about One Piece."""
        one_piece_indicators = [
            'one piece', 'luffy', 'zoro', 'nami', 'pirate', 'marine',
            'devil fruit', 'haki', 'grand line', 'new world', 'yonko'
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in one_piece_indicators)
    
    def _execute_search_with_engine(self, search_params: Dict[str, Any], 
                                  input_data: AgentInput) -> List[Dict[str, Any]]:
        """Execute the search using the proven SearchEngine."""
        # Assert that SearchEngine is available (it should always be)
        assert self.search_engine is not None, "SearchEngine must be available"
        
        try:
            # Execute search with the enhanced query using SearchEngine
            search_results = self.search_engine.search(
                search_params['enhanced_query'],
                top_k=search_params['search_limit']
            )
            
            # Assert that search results are returned (SearchEngine is robust)
            assert isinstance(search_results, list), "SearchEngine should return a list of results"
            
            self.logger.info(f"SearchEngine returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            self.logger.error(f"SearchEngine search failed: {e}")
            # Since SearchEngine is robust, this should not happen
            raise Exception(f"SearchEngine search failed unexpectedly: {e}")
    
    def _process_search_results(self, search_results: List[Dict[str, Any]], 
                              input_data: AgentInput) -> List[Dict[str, Any]]:
        """Process and format search results from SearchEngine."""
        # Assert that we have search results (SearchEngine is reliable)
        assert len(search_results) > 0, "SearchEngine should always return results"
        
        processed_results = []
        
        for i, result in enumerate(search_results):
            # SearchEngine provides rich metadata structure
            processed_result = {
                'rank': i + 1,
                'content': result.get('content', ''),
                'score': result.get('combined_score', result.get('bm25_score', 0.0)),
                'metadata': {
                    'article_name': result.get('search_metadata', {}).get('article_name', 'Unknown'),
                    'section_name': result.get('search_metadata', {}).get('section_name', 'Unknown'),
                    'sub_section_name': result.get('search_metadata', {}).get('sub_section_name', ''),
                    'keywords': result.get('search_metadata', {}).get('keywords', []),
                },
                'search_metadata': result.get('search_metadata', {}),
                'debug_metadata': result.get('debug_metadata', {}),
                'bm25_score': result.get('bm25_score', 0.0),
                'semantic_score': result.get('semantic_score', 0.0),
                'combined_score': result.get('combined_score', 0.0),
                'relevance': self._assess_result_relevance(result, input_data),
                'source_quality': self._assess_source_quality(result),
            }
            
            processed_results.append(processed_result)
        
        # Sort by combined score (SearchEngine provides intelligent ranking)
        processed_results.sort(key=lambda x: x.get('combined_score', 0.0), reverse=True)
        
        return processed_results
    
    def _assess_result_relevance(self, result: Dict[str, Any], 
                                input_data: AgentInput) -> float:
        """Assess the relevance of a search result to the query."""
        relevance = 0.5  # Base relevance
        
        # Check content length
        content = result.get('content', '')
        if len(content) > 100:
            relevance += 0.1
        
        # Check for query terms in content
        query_terms = input_data.query.lower().split()
        content_lower = content.lower()
        
        term_matches = sum(1 for term in query_terms if term in content_lower)
        if term_matches > 0:
            relevance += min(0.3, term_matches * 0.1)
        
        # Check metadata quality
        metadata = result.get('search_metadata', {})
        if metadata.get('article_name') and metadata.get('section_name'):
            relevance += 0.1
        
        return min(relevance, 1.0)
    
    def _assess_source_quality(self, result: Dict[str, Any]) -> str:
        """Assess the quality of the information source."""
        metadata = result.get('search_metadata', {})
        
        # Check if we have comprehensive metadata
        if (metadata.get('article_name') and metadata.get('section_name') and 
            metadata.get('sub_section_name')):
            return 'high'
        elif metadata.get('article_name') and metadata.get('section_name'):
            return 'medium'
        else:
            return 'low'
    
    def _calculate_search_confidence(self, search_results: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the search results."""
        # Base confidence is high since SearchEngine is proven and reliable
        confidence = 0.8
        
        # Adjust based on result count
        result_count = len(search_results)
        if result_count > 5:
            confidence += 0.1
        elif result_count == 0:
            confidence -= 0.1  # Should not happen with SearchEngine
        
        # Adjust based on result quality (SearchEngine provides good scoring)
        if result_count > 0:
            top_result = search_results[0]
            top_score = top_result.get('combined_score', 0.0)
            if top_score > 0.7:
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _validate_input(self, input_data: AgentInput) -> bool:
        """Validate input data for the search agent."""
        # Call parent validation
        if not super()._validate_input(input_data):
            return False
        
        # Search agent requires a text query
        if not input_data.query or not input_data.query.strip():
            self.logger.warning("Search agent requires a text query")
            return False
        
        return True
