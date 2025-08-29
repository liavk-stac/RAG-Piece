"""
Reasoning Agent

The reasoning agent is responsible for logical reasoning, information synthesis,
and connecting related pieces of information from the RAG database.
"""

from typing import Dict, Any, List, Optional
import re

from .base_agent import BaseAgent, AgentType, AgentInput
from ..config import ChatbotConfig
from ..utils.llm_client import LLMClient


class ReasoningAgent(BaseAgent):
    """
    Reasoning agent for logical analysis and information synthesis.
    
    This agent:
    - Connects retrieved information logically
    - Identifies relationships and patterns
    - Synthesizes complex information
    - Handles multi-step reasoning
    """
    
    def __init__(self, config: ChatbotConfig):
        """Initialize the reasoning agent."""
        super().__init__(config, AgentType.REASONING)
        
        # Initialize LLM client
        try:
            self.llm_client = LLMClient(config)
            self.logger.info("LLM client initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize LLM client: {e}")
            self.llm_client = None
        
        # Reasoning patterns and rules
        self.reasoning_patterns = self._initialize_reasoning_patterns()
        self.relationship_patterns = self._initialize_relationship_patterns()
    
    def _initialize_reasoning_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for different types of reasoning."""
        return {
            'causality': [
                r'\b(cause|effect|result|consequence|because|due to|leads to)\b',
                r'\b(if|then|when|while|during|after|before)\b',
            ],
            'comparison': [
                r'\b(compare|contrast|difference|similarity|versus|vs|unlike|like)\b',
                r'\b(both|neither|either|same|different|similar|alike)\b',
            ],
            'inference': [
                r'\b(infer|conclude|imply|suggest|indicate|show|demonstrate)\b',
                r'\b(means|implies|suggests|indicates|shows|demonstrates)\b',
            ],
            'classification': [
                r'\b(type|category|group|class|kind|sort|form|variety)\b',
                r'\b(belongs to|classified as|grouped under|categorized as)\b',
            ],
        }
    
    def _initialize_relationship_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for relationship detection."""
        return {
            'family': [
                r'\b(father|mother|son|daughter|brother|sister|parent|child)\b',
                r'\b(family|ancestor|descendant|relative|kin|blood)\b',
            ],
            'friendship': [
                r'\b(friend|ally|companion|partner|teammate|crew|nakama)\b',
                r'\b(close|trusted|loyal|faithful|supportive)\b',
            ],
            'conflict': [
                r'\b(enemy|rival|opponent|adversary|foe|nemesis|antagonist)\b',
                r'\b(fight|battle|war|conflict|struggle|clash|confrontation)\b',
            ],
            'mentorship': [
                r'\b(teacher|student|mentor|apprentice|pupil|disciple|master)\b',
                r'\b(trained|taught|learned|studied|guided|instructed)\b',
            ],
            'leadership': [
                r'\b(leader|captain|commander|chief|boss|ruler|king|queen)\b',
                r'\b(leads|commands|rules|governs|oversees|manages)\b',
            ],
        }
    
    def _execute_agent(self, input_data: AgentInput) -> Dict[str, Any]:
        """
        Execute the reasoning agent logic.
        
        Args:
            input_data: Input data containing query and context
            
        Returns:
            Dictionary containing reasoning results and logical connections
        """
        self.logger.info("Executing logical reasoning and information synthesis")
        
        # Extract search results from context
        search_results = self._extract_search_results(input_data)
        if not search_results:
            return self._generate_no_data_response()
        
        # Perform logical reasoning using LLM
        reasoning_results = self._perform_llm_reasoning(search_results, input_data)
        
        # Identify relationships
        relationships = self._identify_relationships(search_results, input_data)
        
        # Synthesize information
        synthesis = self._synthesize_information(search_results, reasoning_results, relationships, input_data)
        
        # Generate insights
        insights = self._generate_insights(reasoning_results, relationships, input_data)
        
        # Compile reasoning output
        reasoning_output = {
            'reasoning_type': self._determine_reasoning_type(input_data.query),
            'logical_connections': reasoning_results,
            'relationships': relationships,
            'synthesis': synthesis,
            'insights': insights,
            'confidence_score': self._calculate_reasoning_confidence(reasoning_results, relationships),
            'metadata': {
                'search_results_analyzed': len(search_results),
                'relationships_identified': len(relationships),
                'insights_generated': len(insights),
                'reasoning_complexity': self._assess_reasoning_complexity(input_data.query),
                'llm_used': True,
            }
        }
        
        self.logger.info("Logical reasoning completed successfully")
        
        return reasoning_output
    
    def _extract_search_results(self, input_data: AgentInput) -> List[Dict[str, Any]]:
        """Extract search results from the input context."""
        if not input_data.context:
            return []
        
        # Look for search agent output
        search_output = input_data.context.get('search_agent', {})
        if isinstance(search_output, dict) and 'results' in search_output:
            return search_output['results']
        
        # Look for direct search results
        if 'search_results' in input_data.context:
            return input_data.context['search_results']
        
        return []
    
    def _perform_llm_reasoning(self, search_results: List[Dict[str, Any]], 
                              input_data: AgentInput) -> Dict[str, Any]:
        """Perform logical reasoning using LLM."""
        try:
            if not self.llm_client or not self.llm_client.is_available():
                return self._perform_logical_reasoning(search_results, input_data)
            
            # Use LLM for reasoning
            llm_result = self.llm_client.generate_reasoning(
                context=input_data.context.get('query_context', ''),
                query=input_data.query,
                search_results=search_results
            )
            
            # Convert LLM result to expected format
            return {
                'causal_connections': [llm_result.get('logical_connections', '')],
                'comparative_analysis': [llm_result.get('relationships', '')],
                'inferences': [llm_result.get('reasoning_analysis', '')],
                'classifications': ['LLM-generated classification'],
                'logical_patterns': ['LLM-identified patterns'],
                'llm_confidence': llm_result.get('confidence_score', 0.0),
            }
            
        except Exception as e:
            self.logger.warning(f"LLM reasoning failed, falling back to rule-based: {e}")
            return self._perform_logical_reasoning(search_results, input_data)
    
    def _perform_logical_reasoning(self, search_results: List[Dict[str, Any]], 
                                 input_data: AgentInput) -> Dict[str, Any]:
        """Perform logical reasoning on the search results using rule-based methods."""
        reasoning_results = {
            'causal_connections': [],
            'comparative_analysis': [],
            'inferences': [],
            'classifications': [],
            'logical_patterns': [],
        }
        
        # Analyze each search result for logical patterns
        for result in search_results:
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            
            # Detect causal connections
            causal_connections = self._detect_causal_connections(content, metadata)
            if causal_connections:
                reasoning_results['causal_connections'].extend(causal_connections)
            
            # Detect comparative elements
            comparative_elements = self._detect_comparative_elements(content, metadata)
            if comparative_elements:
                reasoning_results['comparative_analysis'].extend(comparative_elements)
            
            # Generate inferences
            inferences = self._generate_inferences(content, metadata, input_data.query)
            if inferences:
                reasoning_results['inferences'].extend(inferences)
            
            # Perform classifications
            classifications = self._perform_classifications(content, metadata)
            if classifications:
                reasoning_results['classifications'].extend(classifications)
        
        # Identify logical patterns across results
        reasoning_results['logical_patterns'] = self._identify_logical_patterns(search_results)
        
        return reasoning_results
    
    def _detect_causal_connections(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect causal connections in the content."""
        causal_connections = []
        
        # Look for causal language patterns
        causal_patterns = [
            (r'(\w+)\s+(caused|led to|resulted in|brought about)\s+(\w+)', 'cause_effect'),
            (r'(\w+)\s+(because|due to|as a result of|owing to)\s+(\w+)', 'reason_cause'),
            (r'(\w+)\s+(happened|occurred|took place)\s+(after|following|subsequent to)\s+(\w+)', 'temporal_sequence'),
        ]
        
        for pattern, connection_type in causal_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                causal_connections.append({
                    'type': connection_type,
                    'cause': match.group(1).strip(),
                    'effect': match.group(2) if len(match.groups()) == 2 else match.group(3).strip(),
                    'source': metadata.get('article_name', 'Unknown'),
                    'confidence': 0.7,
                })
        
        return causal_connections
    
    def _detect_comparative_elements(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect comparative elements in the content."""
        comparative_elements = []
        
        # Look for comparative language patterns
        comparative_patterns = [
            (r'(\w+)\s+(is|are)\s+(similar to|like|resembles)\s+(\w+)', 'similarity'),
            (r'(\w+)\s+(differs from|unlike|different from)\s+(\w+)', 'difference'),
            (r'(\w+)\s+(compared to|versus|vs)\s+(\w+)', 'comparison'),
        ]
        
        for pattern, comparison_type in comparative_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                comparative_elements.append({
                    'type': comparison_type,
                    'subject': match.group(1).strip(),
                    'object': match.group(2) if len(match.groups()) == 2 else match.group(3).strip(),
                    'source': metadata.get('article_name', 'Unknown'),
                    'confidence': 0.6,
                })
        
        return comparative_elements
    
    def _generate_inferences(self, content: str, metadata: Dict[str, Any], 
                           query: str) -> List[Dict[str, Any]]:
        """Generate logical inferences from the content."""
        inferences = []
        
        # Simple inference generation based on content analysis
        if 'devil fruit' in content.lower() and 'power' in content.lower():
            inferences.append({
                'type': 'power_inference',
                'inference': 'Devil fruit abilities provide supernatural powers',
                'evidence': 'Content mentions devil fruit and power',
                'source': metadata.get('article_name', 'Unknown'),
                'confidence': 0.8,
            })
        
        if 'haki' in content.lower() and 'ability' in content.lower():
            inferences.append({
                'type': 'haki_inference',
                'inference': 'Haki is a special ability that can be developed',
                'evidence': 'Content mentions haki and ability',
                'source': metadata.get('article_name', 'Unknown'),
                'confidence': 0.8,
            })
        
        if 'pirate' in content.lower() and 'crew' in content.lower():
            inferences.append({
                'type': 'crew_inference',
                'inference': 'Pirates typically operate in crews or groups',
                'evidence': 'Content mentions pirate and crew',
                'source': metadata.get('article_name', 'Unknown'),
                'confidence': 0.7,
            })
        
        return inferences
    
    def _perform_classifications(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform classifications of entities in the content."""
        classifications = []
        
        # Classify One Piece entities
        entity_patterns = {
            'devil_fruit': r'\b(devil fruit|akuma no mi)\b',
            'haki_type': r'\b(observation haki|armament haki|conqueror haki)\b',
            'pirate_rank': r'\b(yonko|shichibukai|supernova|rookie)\b',
            'marine_rank': r'\b(admiral|vice admiral|rear admiral|captain)\b',
            'weapon_type': r'\b(sword|gun|cannon|spear|axe|club)\b',
        }
        
        for entity_type, pattern in entity_patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                classifications.append({
                    'type': 'entity_classification',
                    'entity_type': entity_type,
                    'content': content[:100] + "..." if len(content) > 100 else content,
                    'source': metadata.get('article_name', 'Unknown'),
                    'confidence': 0.8,
                })
        
        return classifications
    
    def _identify_logical_patterns(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify logical patterns across multiple search results."""
        patterns = []
        
        # Look for recurring themes
        themes = {}
        for result in search_results:
            content = result.get('content', '').lower()
            metadata = result.get('metadata', {})
            article_name = metadata.get('article_name', 'Unknown')
            
            # Count theme occurrences
            for theme in ['devil fruit', 'haki', 'pirate', 'marine', 'world government']:
                if theme in content:
                    if theme not in themes:
                        themes[theme] = {'count': 0, 'sources': []}
                    themes[theme]['count'] += 1
                    themes[theme]['sources'].append(article_name)
        
        # Convert themes to patterns
        for theme, data in themes.items():
            if data['count'] > 1:
                patterns.append({
                    'type': 'recurring_theme',
                    'theme': theme,
                    'frequency': data['count'],
                    'sources': data['sources'],
                    'confidence': min(0.9, 0.5 + data['count'] * 0.1),
                })
        
        return patterns
    
    def _identify_relationships(self, search_results: List[Dict[str, Any]], 
                              input_data: AgentInput) -> List[Dict[str, Any]]:
        """Identify relationships between entities in the search results."""
        relationships = []
        
        # Extract entities from search results
        entities = self._extract_entities(search_results)
        
        # Analyze relationships between entities
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                relationship = self._analyze_entity_relationship(entity1, entity2, search_results)
                if relationship:
                    relationships.append(relationship)
        
        # Add relationship patterns from content analysis
        content_relationships = self._extract_content_relationships(search_results)
        relationships.extend(content_relationships)
        
        return relationships
    
    def _extract_entities(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract entities from search results."""
        entities = []
        
        # One Piece specific entity patterns
        entity_patterns = {
            'character': r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            'location': r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Kingdom|Island|Sea|Ocean|World))\b',
            'organization': r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Pirates|Marines|Government|Army))\b',
        }
        
        for result in search_results:
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            
            for entity_type, pattern in entity_patterns.items():
                matches = re.finditer(pattern, content)
                for match in matches:
                    entity_name = match.group(1)
                    entities.append({
                        'name': entity_name,
                        'type': entity_type,
                        'source': metadata.get('article_name', 'Unknown'),
                        'context': content[:100] + "..." if len(content) > 100 else content,
                    })
        
        return entities
    
    def _analyze_entity_relationship(self, entity1: Dict[str, Any], entity2: Dict[str, Any], 
                                   search_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Analyze the relationship between two entities."""
        # Look for co-occurrence in search results
        co_occurrences = []
        
        for result in search_results:
            content = result.get('content', '').lower()
            if (entity1['name'].lower() in content and 
                entity2['name'].lower() in content):
                co_occurrences.append(result)
        
        if not co_occurrences:
            return None
        
        # Analyze the nature of the relationship
        relationship_type = self._determine_relationship_type(entity1, entity2, co_occurrences)
        
        return {
            'type': 'entity_relationship',
            'entity1': entity1['name'],
            'entity2': entity2['name'],
            'relationship_type': relationship_type,
            'evidence': [r.get('metadata', {}).get('article_name', 'Unknown') for r in co_occurrences],
            'confidence': min(0.9, 0.5 + len(co_occurrences) * 0.1),
        }
    
    def _determine_relationship_type(self, entity1: Dict[str, Any], entity2: Dict[str, Any], 
                                   co_occurrences: List[Dict[str, Any]]) -> str:
        """Determine the type of relationship between two entities."""
        # Analyze content for relationship indicators
        relationship_indicators = {
            'allies': ['friend', 'ally', 'companion', 'teammate', 'crew', 'nakama'],
            'enemies': ['enemy', 'rival', 'opponent', 'adversary', 'foe'],
            'family': ['father', 'mother', 'son', 'daughter', 'brother', 'sister'],
            'mentorship': ['teacher', 'student', 'mentor', 'apprentice'],
            'leadership': ['leader', 'follower', 'captain', 'crew member'],
        }
        
        for relationship_type, indicators in relationship_indicators.items():
            for occurrence in co_occurrences:
                content = occurrence.get('content', '').lower()
                if any(indicator in content for indicator in indicators):
                    return relationship_type
        
        return 'associated'  # Default relationship type
    
    def _extract_content_relationships(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships mentioned directly in the content."""
        content_relationships = []
        
        for result in search_results:
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            
            # Look for relationship patterns in content
            for relationship_type, patterns in self.relationship_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        content_relationships.append({
                            'type': 'content_relationship',
                            'relationship_type': relationship_type,
                            'content': content[:150] + "..." if len(content) > 150 else content,
                            'source': metadata.get('article_name', 'Unknown'),
                            'confidence': 0.7,
                        })
        
        return content_relationships
    
    def _synthesize_information(self, search_results: List[Dict[str, Any]], 
                                reasoning_results: Dict[str, Any], 
                                relationships: List[Dict[str, Any]],
                                input_data: AgentInput) -> Dict[str, Any]:
        """Synthesize information from multiple sources."""
        synthesis = {
            'main_themes': [],
            'key_insights': [],
            'contradictions': [],
            'gaps': [],
            'summary': '',
        }
        
        # Identify main themes
        themes = reasoning_results.get('logical_patterns', [])
        if isinstance(themes, list):
            synthesis['main_themes'] = [theme['theme'] for theme in themes if isinstance(theme, dict) and theme.get('confidence', 0) > 0.7]
        else:
            synthesis['main_themes'] = []
        
        # Generate key insights
        insights = []
        if reasoning_results.get('causal_connections'):
            insights.append("Causal relationships identified between key events and characters")
        if reasoning_results.get('comparative_analysis'):
            insights.append("Comparative analysis reveals differences and similarities")
        if relationships:
            insights.append(f"Identified {len(relationships)} key relationships between entities")
        
        synthesis['key_insights'] = insights
        
        # Identify potential contradictions
        contradictions = self._identify_contradictions(search_results)
        synthesis['contradictions'] = contradictions
        
        # Identify information gaps
        gaps = self._identify_information_gaps(search_results, input_data.query)
        synthesis['gaps'] = gaps
        
        # Generate summary
        synthesis['summary'] = self._generate_synthesis_summary(synthesis)
        
        return synthesis
    
    def _identify_contradictions(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify potential contradictions in the search results."""
        contradictions = []
        
        # This is a simplified implementation
        # In practice, you'd use more sophisticated contradiction detection
        
        return contradictions
    
    def _identify_information_gaps(self, search_results: List[Dict[str, Any]], query: str) -> List[str]:
        """Identify gaps in the available information."""
        gaps = []
        
        # Check if we have comprehensive coverage
        if len(search_results) < 3:
            gaps.append("Limited information available - may need broader search")
        
        # Check for specific information types
        query_lower = query.lower()
        if 'timeline' in query_lower and not any('time' in r.get('content', '').lower() for r in search_results):
            gaps.append("Timeline information not found in results")
        
        if 'relationship' in query_lower and not any('relationship' in r.get('content', '').lower() for r in search_results):
            gaps.append("Relationship information not found in results")
        
        return gaps
    
    def _generate_synthesis_summary(self, synthesis: Dict[str, Any]) -> str:
        """Generate a summary of the synthesis."""
        summary_parts = []
        
        if synthesis['main_themes']:
            summary_parts.append(f"Main themes identified: {', '.join(synthesis['main_themes'])}")
        
        if synthesis['key_insights']:
            summary_parts.append(f"Key insights: {'; '.join(synthesis['key_insights'])}")
        
        if synthesis['gaps']:
            summary_parts.append(f"Information gaps: {'; '.join(synthesis['gaps'])}")
        
        if summary_parts:
            return ". ".join(summary_parts) + "."
        else:
            return "Information synthesis completed with available data."
    
    def _generate_insights(self, reasoning_results: Dict[str, Any], 
                          relationships: List[Dict[str, Any]], 
                          input_data: AgentInput) -> List[Dict[str, Any]]:
        """Generate insights based on reasoning and relationships."""
        insights = []
        
        # Generate insights from causal connections
        causal_connections = reasoning_results.get('causal_connections', [])
        if isinstance(causal_connections, list):
            for connection in causal_connections:
                if isinstance(connection, dict) and connection.get('confidence', 0) > 0.7:
                    insights.append({
                        'type': 'causal_insight',
                        'insight': f"{connection.get('cause', 'Unknown')} leads to {connection.get('effect', 'Unknown')}",
                        'confidence': connection.get('confidence', 0),
                        'source': connection.get('source', 'Unknown'),
                    })
        
        # Generate insights from relationships
        if isinstance(relationships, list):
            for relationship in relationships:
                if isinstance(relationship, dict) and relationship.get('confidence', 0) > 0.7:
                    insights.append({
                        'type': 'relationship_insight',
                        'insight': f"{relationship.get('entity1', 'Unknown')} has {relationship.get('relationship_type', 'Unknown')} relationship with {relationship.get('entity2', 'Unknown')}",
                        'confidence': relationship.get('confidence', 0),
                        'source': relationship.get('source', 'Unknown'),
                    })
        
        # Generate query-specific insights
        query_insights = self._generate_query_specific_insights(input_data.query, reasoning_results)
        insights.extend(query_insights)
        
        return insights
    
    def _generate_query_specific_insights(self, query: str, 
                                        reasoning_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights specific to the user's query."""
        insights = []
        query_lower = query.lower()
        
        # Character-related insights
        if any(word in query_lower for word in ['character', 'person', 'who']):
            if reasoning_results.get('causal_connections'):
                insights.append({
                    'type': 'character_insight',
                    'insight': "Character development shows clear progression and growth",
                    'confidence': 0.8,
                    'source': 'reasoning_analysis',
                })
        
        # Location-related insights
        if any(word in query_lower for word in ['location', 'place', 'where']):
            if reasoning_results.get('logical_patterns'):
                insights.append({
                    'type': 'location_insight',
                    'insight': "Geographical patterns reveal strategic importance",
                    'confidence': 0.7,
                    'source': 'reasoning_analysis',
                })
        
        return insights
    
    def _determine_reasoning_type(self, query: str) -> str:
        """Determine the type of reasoning needed for the query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['why', 'cause', 'reason']):
            return 'causal_reasoning'
        elif any(word in query_lower for word in ['compare', 'difference', 'similarity']):
            return 'comparative_reasoning'
        elif any(word in query_lower for word in ['how', 'process', 'method']):
            return 'procedural_reasoning'
        elif any(word in query_lower for word in ['relationship', 'connection', 'bond']):
            return 'relational_reasoning'
        else:
            return 'general_reasoning'
    
    def _assess_reasoning_complexity(self, query: str) -> str:
        """Assess the complexity of reasoning required."""
        query_lower = query.lower()
        
        # Count reasoning indicators
        reasoning_indicators = 0
        
        for pattern_list in self.reasoning_patterns.values():
            for pattern in pattern_list:
                if re.search(pattern, query_lower):
                    reasoning_indicators += 1
        
        if reasoning_indicators > 2:
            return 'high'
        elif reasoning_indicators > 0:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_reasoning_confidence(self, reasoning_results: Dict[str, Any], 
                                      relationships: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the reasoning results."""
        confidence = 0.5  # Base confidence
        
        # Adjust based on reasoning results quality
        if reasoning_results.get('causal_connections'):
            confidence += 0.1
        
        if reasoning_results.get('comparative_analysis'):
            confidence += 0.1
        
        if reasoning_results.get('inferences'):
            confidence += 0.1
        
        if relationships:
            confidence += min(0.2, len(relationships) * 0.05)
        
        return min(confidence, 1.0)
    
    def _generate_no_data_response(self) -> Dict[str, Any]:
        """Generate response when no search data is available."""
        return {
            'reasoning_type': 'no_data',
            'logical_connections': [],
            'relationships': [],
            'synthesis': {
                'main_themes': [],
                'key_insights': [],
                'contradictions': [],
                'gaps': ['No search results available for reasoning'],
                'summary': 'Unable to perform reasoning without search data.',
            },
            'insights': [],
            'confidence_score': 0.0,
            'metadata': {
                'search_results_analyzed': 0,
                'relationships_identified': 0,
                'insights_generated': 0,
                'reasoning_complexity': 'none',
            }
        }
    
    def _validate_input(self, input_data: AgentInput) -> bool:
        """Validate input data for the reasoning agent."""
        # Call parent validation
        if not super()._validate_input(input_data):
            return False
        
        # Reasoning agent requires context with search results
        if not input_data.context:
            self.logger.warning("Reasoning agent requires context with search results")
            return False
        
        return True
