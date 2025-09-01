"""
Response Agent

The response agent is responsible for generating the final, coherent response
by synthesizing outputs from all other agents in the pipeline.
"""

from typing import Dict, Any, List, Optional
import re
import time

from .base_agent import BaseAgent, AgentType, AgentInput
from ..config import ChatbotConfig
from ..utils.llm_client import LLMClient


class ResponseAgent(BaseAgent):
    """
    Response agent for final response generation and formatting.
    
    This agent:
    - Synthesizes outputs from all other agents
    - Generates coherent, comprehensive responses
    - Ensures One Piece lore accuracy
    - Formats responses for optimal user experience
    """
    
    def __init__(self, config: ChatbotConfig):
        """Initialize the response agent."""
        super().__init__(config, AgentType.RESPONSE)
        
        # Initialize LLM client for enhanced response generation
        try:
            self.llm_client = LLMClient(config)
            self.logger.info("LLM client initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize LLM client: {e}")
            self.llm_client = None
        
        # Response templates and formatting rules
        self.response_templates = self._initialize_response_templates()
        self.formatting_rules = self._initialize_formatting_rules()
    
    def _initialize_response_templates(self) -> Dict[str, str]:
        """Initialize response templates for different types of queries."""
        return {
            'character_info': """
Based on my analysis of the One Piece database, here's what I found about {character_name}:

**Character Overview:**
{character_summary}

**Key Information:**
{key_details}

**Notable Relationships:**
{relationships}

**Recent Developments:**
{recent_info}
            """,
            
            'location_info': """
Here's what I discovered about {location_name} in the One Piece world:

**Location Details:**
{location_summary}

**Geographical Information:**
{geography_info}

**Historical Significance:**
{history_info}

**Notable Characters:**
{notable_characters}
            """,
            
            'image_analysis': """
I've analyzed the image you provided and here's what I found:

**Image Analysis:**
{image_description}

**One Piece Context:**
{one_piece_context}

**Related Information:**
{related_info}

**Additional Insights:**
{additional_insights}
            """,
            
            'general_search': """
Based on your query about "{query}", here's what I found in the One Piece database:

**Main Information:**
{main_info}

**Key Details:**
{key_details}

**Related Topics:**
{related_topics}

**Additional Context:**
{additional_context}
            """,
            
            'timeline_info': """
Here's the timeline information for your query about "{query}":

**Chronological Order:**
{timeline_events}

**Key Events:**
{key_events}

**Historical Context:**
{historical_context}

**Related Timeline:**
{related_timeline}
            """,
            
            'relationship_analysis': """
Here's my analysis of the relationships for "{query}":

**Primary Relationships:**
{primary_relationships}

**Connection Details:**
{connection_details}

**Relationship Dynamics:**
{relationship_dynamics}

**Historical Context:**
{historical_context}
            """,
            
            'error_response': """
I apologize, but I encountered some difficulties while processing your request:

**Issue:**
{error_description}

**What I was able to find:**
{partial_results}

**Suggestions:**
{user_suggestions}
            """,
            
            'no_results': """
I searched the One Piece database for information about "{query}", but I couldn't find any relevant results.

**Possible reasons:**
- The topic might not be covered in our current database
- The search terms might need to be more specific
- The information might be referenced under different terms

**Suggestions:**
- Try using different keywords or character names
- Check the spelling of names or terms
- Ask about broader topics that might include this information
            """
        }
    
    def _initialize_formatting_rules(self) -> Dict[str, Any]:
        """Initialize formatting rules for response generation."""
        return {
            'max_section_length': 300,
            'max_total_length': 1500,
            'preferred_format': 'markdown',
            'include_sources': True,
            'include_confidence': True,
            'include_metadata': False,
            'one_piece_style': True,
        }
    
    def _execute_agent(self, input_data: AgentInput) -> Dict[str, Any]:
        """
        Execute the response agent logic.
        
        Args:
            input_data: Input data containing query and context
            
        Returns:
            Dictionary containing the final formatted response
        """
        self.logger.info("Generating final response")
        
        # Extract agent outputs from context
        agent_outputs = input_data.context.get('agent_outputs', {}) if input_data.context else {}
        
        # Generate response based on available information using LLM
        if not agent_outputs:
            response = self._generate_error_response(input_data, "No agent outputs available")
        else:
            response = self._synthesize_response_with_llm(agent_outputs, input_data)
            self.logger.info("Response generated using LLM")
        
        # Format the response
        formatted_response = self._format_response(response, input_data)
        
        # Generate response metadata
        response_metadata = self._generate_response_metadata(response, input_data)
        
        # Compile final output
        final_output = {
            'response': formatted_response,
            'metadata': response_metadata,
            'confidence_score': response.get('confidence_score', 0.0),
            'response_type': response.get('response_type', 'general'),
            'sources_used': response.get('sources_used', []),
            'processing_summary': response.get('processing_summary', {}),
            'llm_used': True,
        }
        
        # Add image information if available from image retrieval agent
        if 'image_retrieval' in agent_outputs:
            image_retrieval_output = agent_outputs['image_retrieval']
            if image_retrieval_output.get('success') and image_retrieval_output.get('image'):
                final_output['image'] = image_retrieval_output['image']
                final_output['image_metadata'] = {
                    'intent_analysis': image_retrieval_output.get('intent_analysis', {}),
                    'candidates_count': image_retrieval_output.get('candidates_count', 0),
                    'relevance_score': image_retrieval_output['image'].get('relevance_score', 0.0)
                }
        
        self.logger.info("Final response generated successfully")
        
        return final_output
    
    def _synthesize_response_with_llm(self, agent_outputs: Dict[str, Any], 
                                     input_data: AgentInput) -> Dict[str, Any]:
        """Synthesize a coherent response using LLM for enhanced quality."""
        # Prepare context for LLM
        context_summary = self._prepare_context_for_llm(agent_outputs, input_data)
        
        system_message = """You are a One Piece knowledge expert. Synthesize information from multiple sources 
        to create a comprehensive, accurate, and engaging response. Be true to One Piece lore and provide 
        detailed, well-structured information. 
        
        IMPORTANT: If the context mentions "Image Display: Showing image", you MUST inform the user that an image 
        is being displayed alongside your response (e.g., "I'm displaying an image of [character/scene] for you" 
        or "Here's an image showing [description]")."""
        
        prompt = f"""User Query: {input_data.query}

Available Information:
{context_summary}

Please synthesize this information into a comprehensive response that:
1. Directly answers the user's question
2. Answer the user's question based only on the information provided above
3. Maintains One Piece lore accuracy
4. Is well-structured and easy to read
5. Includes confidence level in the information provided
6. Try to make the answer concise and to the point (no more then a paragraph)
7. If an image is being displayed (mentioned in context), inform the user about it

Response:"""
        
        # Log the full LLM prompt and system message
        if hasattr(self, 'pipeline_logger'):
            self.pipeline_logger.log_llm_call(
                agent_name="RESPONSE_AGENT",
                prompt=prompt,
                response="",  # Will be updated after call
                tokens_used=0,  # Will be updated after call
                system_message=system_message
            )
        
        # Generate response using LLM
        llm_response = self.llm_client.generate_text(prompt, system_message, max_tokens=800)
        
        # Log the LLM response
        if hasattr(self, 'pipeline_logger'):
            self.pipeline_logger.log_llm_call(
                agent_name="RESPONSE_AGENT",
                prompt=prompt,
                response=llm_response,
                tokens_used=800,  # Approximate, real token count would need API response parsing
                system_message=system_message
            )
        
        # Determine response type
        response_type = self._determine_response_type(agent_outputs, input_data)
        
        # Calculate confidence
        confidence_score = self._calculate_overall_confidence(agent_outputs)
        
        # Extract sources
        sources_used = self._extract_sources(agent_outputs)
        
        # Generate processing summary
        processing_summary = self._generate_processing_summary(agent_outputs)
        
        return {
            'response_type': response_type,
            'content': llm_response,
            'confidence_score': confidence_score,
            'sources_used': sources_used,
            'processing_summary': processing_summary,
            'agent_outputs_summary': {
                'search_results_count': len(agent_outputs.get('search', {}).get('results', [])),
                'image_analysis_success': bool(agent_outputs.get('image_analysis', {}).get('success', False)),
                'reasoning_applied': bool(agent_outputs.get('reasoning')),
                'timeline_info_available': bool(agent_outputs.get('timeline')),
                'llm_generated': True,
            }
        }
    
    def _synthesize_response(self, agent_outputs: Dict[str, Any], 
                            input_data: AgentInput) -> Dict[str, Any]:
        """Synthesize a coherent response from multiple agent outputs using templates."""
        # Determine response type based on agent outputs
        response_type = self._determine_response_type(agent_outputs, input_data)
        
        # Extract relevant information from each agent
        search_info = agent_outputs.get('search_agent', {})
        image_info = agent_outputs.get('image_analysis_agent', {})
        reasoning_info = agent_outputs.get('reasoning_agent', {})
        timeline_info = agent_outputs.get('timeline_agent', {})
        
        # Generate response content based on type
        if response_type == 'image_analysis':
            response_content = self._generate_image_response(
                image_info, search_info, input_data
            )
        elif response_type == 'character_info':
            response_content = self._generate_character_response(
                search_info, reasoning_info, input_data
            )
        elif response_type == 'location_info':
            response_content = self._generate_location_response(
                search_info, reasoning_info, input_data
            )
        elif response_type == 'timeline_info':
            response_content = self._generate_timeline_response(
                timeline_info, search_info, input_data
            )
        elif response_type == 'relationship_analysis':
            response_content = self._generate_relationship_response(
                search_info, reasoning_info, input_data
            )
        else:
            response_content = self._generate_general_response(
                search_info, reasoning_info, input_data
            )
        
        # Calculate overall confidence
        confidence_score = self._calculate_overall_confidence(agent_outputs)
        
        # Compile sources used
        sources_used = self._extract_sources(agent_outputs)
        
        # Generate processing summary
        processing_summary = self._generate_processing_summary(agent_outputs)
        
        return {
            'response_type': response_type,
            'content': response_content,
            'confidence_score': confidence_score,
            'sources_used': sources_used,
            'processing_summary': processing_summary,
            'agent_outputs_summary': {
                'search_results_count': len(search_info.get('results', [])),
                'image_analysis_success': bool(image_info.get('success', False)),
                'reasoning_applied': bool(reasoning_info),
                'timeline_info_available': bool(timeline_info),
                'llm_generated': False,
            }
        }
    
    def _determine_response_type(self, agent_outputs: Dict[str, Any], 
                                input_data: AgentInput) -> str:
        """Determine the type of response to generate."""
        # Check for image analysis
        if agent_outputs.get('image_analysis_agent'):
            return 'image_analysis'
        
        # Check for specific query types
        query = input_data.query.lower()
        
        if any(word in query for word in ['who', 'character', 'person']):
            return 'character_info'
        elif any(word in query for word in ['where', 'location', 'place', 'island']):
            return 'location_info'
        elif any(word in query for word in ['when', 'timeline', 'history', 'era']):
            return 'timeline_info'
        elif any(word in query for word in ['relationship', 'connection', 'bond']):
            return 'relationship_analysis'
        else:
            return 'general_search'
    
    def _generate_image_response(self, image_info: Dict[str, Any], 
                               search_info: Dict[str, Any], 
                               input_data: AgentInput) -> str:
        """Generate response for image analysis queries."""
        template = self.response_templates['image_analysis']
        
        # Extract image description
        image_description = image_info.get('description', 'Image analysis completed')
        
        # Extract One Piece context
        one_piece_context = image_info.get('rag_integration', {}).get('search_query', '')
        
        # Extract related information from search results
        search_results = search_info.get('results', [])
        related_info = self._format_search_results(search_results, max_results=3)
        
        # Generate additional insights
        additional_insights = self._generate_image_insights(image_info, search_info)
        
        # Fill template
        response = template.format(
            image_description=image_description,
            one_piece_context=one_piece_context or "No specific One Piece context found",
            related_info=related_info or "No directly related information found",
            additional_insights=additional_insights or "Analysis completed successfully"
        )
        
        return response
    
    def _generate_character_response(self, search_info: Dict[str, Any], 
                                   reasoning_info: Dict[str, Any], 
                                   input_data: AgentInput) -> str:
        """Generate response for character information queries."""
        template = self.response_templates['character_info']
        
        # Extract character information from search results
        search_results = search_info.get('results', [])
        if not search_results:
            return self.response_templates['no_results'].format(query=input_data.query)
        
        # Get top result for character overview
        top_result = search_results[0]
        character_name = top_result.get('metadata', {}).get('article_name', 'this character')
        character_summary = top_result.get('content', '')[:200] + "..." if len(top_result.get('content', '')) > 200 else top_result.get('content', '')
        
        # Extract key details
        key_details = self._extract_key_details(search_results)
        
        # Extract relationships
        relationships = self._extract_relationships(search_results)
        
        # Extract recent information
        recent_info = self._extract_recent_info(search_results)
        
        # Fill template
        response = template.format(
            character_name=character_name,
            character_summary=character_summary,
            key_details=key_details or "Additional details available in the database",
            relationships=relationships or "Relationship information not available",
            recent_info=recent_info or "Recent developments not specified"
        )
        
        return response
    
    def _generate_location_response(self, search_info: Dict[str, Any], 
                                 reasoning_info: Dict[str, Any], 
                                 input_data: AgentInput) -> str:
        """Generate response for location information queries."""
        template = self.response_templates['location_info']
        
        # Extract location information from search results
        search_results = search_info.get('results', [])
        if not search_results:
            return self.response_templates['no_results'].format(query=input_data.query)
        
        # Get top result for location overview
        top_result = search_results[0]
        location_name = top_result.get('metadata', {}).get('article_name', 'this location')
        location_summary = top_result.get('content', '')[:200] + "..." if len(top_result.get('content', '')) > 200 else top_result.get('content', '')
        
        # Extract key details
        key_details = self._extract_key_details(search_results)
        
        # Extract geography information
        geography_info = self._extract_geography_info(search_results)
        
        # Extract history information
        history_info = self._extract_history_info(search_results)
        
        # Extract notable characters
        notable_characters = self._extract_notable_characters(search_results)
        
        # Fill template
        response = template.format(
            location_name=location_name,
            location_summary=location_summary,
            geography_info=geography_info or "Geographical details not specified",
            history_info=history_info or "Historical information not available",
            notable_characters=notable_characters or "Character associations not specified"
        )
        
        return response
    
    def _generate_timeline_response(self, timeline_info: Dict[str, Any], 
                                 search_info: Dict[str, Any], 
                                 input_data: AgentInput) -> str:
        """Generate response for timeline queries."""
        template = self.response_templates['timeline_info']
        
        # Extract timeline information
        timeline_events = timeline_info.get('timeline_events', 'Timeline information not available')
        key_events = timeline_info.get('key_events', 'Key events not specified')
        historical_context = timeline_info.get('historical_context', 'Historical context not available')
        related_timeline = timeline_info.get('related_timeline', 'Related timeline not available')
        
        # Fill template
        response = template.format(
            query=input_data.query,
            timeline_events=timeline_events,
            key_events=key_events,
            historical_context=historical_context,
            related_timeline=related_timeline
        )
        
        return response
    
    def _generate_relationship_response(self, search_info: Dict[str, Any], 
                                     reasoning_info: Dict[str, Any], 
                                     input_data: AgentInput) -> str:
        """Generate response for relationship analysis queries."""
        template = self.response_templates['relationship_analysis']
        
        # Extract relationship information
        primary_relationships = reasoning_info.get('primary_relationships', 'Relationship information not available')
        connection_details = reasoning_info.get('connection_details', 'Connection details not specified')
        relationship_dynamics = reasoning_info.get('relationship_dynamics', 'Relationship dynamics not available')
        historical_context = reasoning_info.get('historical_context', 'Historical context not available')
        
        # Fill template
        response = template.format(
            query=input_data.query,
            primary_relationships=primary_relationships,
            connection_details=connection_details,
            relationship_dynamics=relationship_dynamics,
            historical_context=historical_context
        )
        
        return response
    
    def _generate_general_response(self, search_info: Dict[str, Any], 
                                 reasoning_info: Dict[str, Any], 
                                 input_data: AgentInput) -> str:
        """Generate response for general search queries."""
        template = self.response_templates['general_search']
        
        # Extract search information
        search_results = search_info.get('results', [])
        if not search_results:
            return self.response_templates['no_results'].format(query=input_data.query)
        
        # Get main information from top results
        main_info = self._extract_main_info(search_results)
        key_details = self._extract_key_details(search_results)
        related_topics = self._extract_related_topics(search_results)
        additional_context = self._extract_additional_context(search_results)
        
        # Fill template
        response = template.format(
            query=input_data.query,
            main_info=main_info or "Information found in the database",
            key_details=key_details or "Additional details available",
            related_topics=related_topics or "Related topics not specified",
            additional_context=additional_context or "Context information not available"
        )
        
        return response
    
    def _generate_error_response(self, input_data: AgentInput, 
                               error_description: str) -> Dict[str, Any]:
        """Generate response for error conditions."""
        template = self.response_templates['error_response']
        
        response = template.format(
            error_description=error_description,
            partial_results="No partial results available",
            user_suggestions="Try rephrasing your question or ask about a different topic"
        )
        
        return {
            'response_type': 'error',
            'content': response,
            'confidence_score': 0.0,
            'sources_used': [],
            'processing_summary': {'error': error_description}
        }
    
    def _extract_key_details(self, search_results: List[Dict[str, Any]]) -> str:
        """Extract key details from search results."""
        if not search_results:
            return ""
        
        details = []
        for result in search_results[:3]:  # Top 3 results
            content = result.get('content', '')
            if len(content) > 50:
                details.append(content[:100] + "...")
        
        return "\n".join(details) if details else ""
    
    def _extract_relationships(self, search_results: List[Dict[str, Any]]) -> str:
        """Extract relationship information from search results."""
        # This would be enhanced with actual relationship extraction logic
        return "Relationship information extracted from search results"
    
    def _extract_recent_info(self, search_results: List[Dict[str, Any]]) -> str:
        """Extract recent information from search results."""
        # This would be enhanced with actual recency analysis
        return "Recent information extracted from search results"
    
    def _extract_geography_info(self, search_results: List[Dict[str, Any]]) -> str:
        """Extract geography information from search results."""
        # This would be enhanced with actual geography extraction
        return "Geographical information extracted from search results"
    
    def _extract_history_info(self, search_results: List[Dict[str, Any]]) -> str:
        """Extract history information from search results."""
        # This would be enhanced with actual history extraction
        return "Historical information extracted from search results"
    
    def _extract_notable_characters(self, search_results: List[Dict[str, Any]]) -> str:
        """Extract notable characters from search results."""
        # This would be enhanced with actual character extraction
        return "Notable characters extracted from search results"
    
    def _extract_main_info(self, search_results: List[Dict[str, Any]]) -> str:
        """Extract main information from search results."""
        if not search_results:
            return ""
        
        main_result = search_results[0]
        content = main_result.get('content', '')
        return content[:300] + "..." if len(content) > 300 else content
    
    def _extract_related_topics(self, search_results: List[Dict[str, Any]]) -> str:
        """Extract related topics from search results."""
        # This would be enhanced with actual topic extraction
        return "Related topics extracted from search results"
    
    def _extract_additional_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Extract additional context from search results."""
        # This would be enhanced with actual context extraction
        return "Additional context extracted from search results"
    
    def _format_search_results(self, search_results: List[Dict[str, Any]], 
                             max_results: int = 3) -> str:
        """Format search results for display."""
        if not search_results:
            return ""
        
        formatted_results = []
        for i, result in enumerate(search_results[:max_results]):
            content = result.get('content', '')
            metadata = result.get('metadata', {})
            article_name = metadata.get('article_name', 'Unknown')
            section_name = metadata.get('section_name', 'Unknown')
            
            formatted_result = f"**{i+1}. {article_name} - {section_name}**\n{content[:150]}..."
            formatted_results.append(formatted_result)
        
        return "\n\n".join(formatted_results)
    
    def _generate_image_insights(self, image_info: Dict[str, Any], 
                               search_info: Dict[str, Any]) -> str:
        """Generate additional insights for image analysis."""
        insights = []
        
        # Image quality insights
        if 'quality_score' in image_info:
            quality_score = image_info['quality_score']
            if quality_score > 0.8:
                insights.append("High quality image provides clear visual information")
            elif quality_score < 0.5:
                insights.append("Image quality may limit analysis accuracy")
        
        # Search result insights
        search_results = search_info.get('results', [])
        if search_results:
            insights.append(f"Found {len(search_results)} related pieces of information")
        else:
            insights.append("No directly related information found in the database")
        
        return ". ".join(insights) if insights else "Analysis completed with available information"
    
    def _calculate_overall_confidence(self, agent_outputs: Dict[str, Any]) -> float:
        """Calculate overall confidence score from all agent outputs."""
        confidence_scores = []
        
        for agent_name, output in agent_outputs.items():
            if isinstance(output, dict) and 'confidence_score' in output:
                confidence_scores.append(output['confidence_score'])
        
        if not confidence_scores:
            return 0.5  # Default confidence
        
        # Weighted average based on agent importance
        weights = {
            'search_agent': 0.4,
            'image_analysis_agent': 0.3,
            'reasoning_agent': 0.2,
            'timeline_agent': 0.1,
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for agent_name, output in agent_outputs.items():
            if agent_name in weights and 'confidence_score' in output:
                weighted_sum += output['confidence_score'] * weights[agent_name]
                total_weight += weights[agent_name]
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return sum(confidence_scores) / len(confidence_scores)
    
    def _extract_sources(self, agent_outputs: Dict[str, Any]) -> List[str]:
        """Extract sources used by all agents."""
        sources = []
        
        for agent_name, output in agent_outputs.items():
            if isinstance(output, dict):
                # Extract sources from search results
                if 'results' in output:
                    for result in output['results']:
                        metadata = result.get('metadata', {})
                        article_name = metadata.get('article_name', 'Unknown')
                        if article_name not in sources:
                            sources.append(article_name)
                
                # Extract sources from other agent outputs
                if 'sources_used' in output:
                    sources.extend(output['sources_used'])
        
        return list(set(sources))  # Remove duplicates
    
    def _generate_processing_summary(self, agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of processing performed by all agents."""
        summary = {
            'agents_executed': list(agent_outputs.keys()),
            'total_processing_time': 0.0,
            'successful_agents': 0,
            'failed_agents': 0,
        }
        
        for agent_name, output in agent_outputs.items():
            if isinstance(output, dict):
                if output.get('success', True):
                    summary['successful_agents'] += 1
                else:
                    summary['failed_agents'] += 1
                
                if 'execution_time' in output:
                    summary['total_processing_time'] += output['execution_time']
        
        return summary
    
    def _format_response(self, response: Dict[str, Any], 
                        input_data: AgentInput) -> str:
        """Format the response according to formatting rules."""
        content = response.get('content', '')
        
        # Apply formatting rules
        if self.formatting_rules['preferred_format'] == 'markdown':
            content = self._apply_markdown_formatting(content)
        
        # Truncate if too long
        if len(content) > self.formatting_rules['max_total_length']:
            content = content[:self.formatting_rules['max_total_length']] + "..."
        
        # Add confidence indicator if enabled
        if self.formatting_rules['include_confidence']:
            confidence = response.get('confidence_score', 0.0)
            confidence_text = f"\n\n*Confidence: {confidence:.1%}*"
            content += confidence_text
        
        return content
    
    def _apply_markdown_formatting(self, content: str) -> str:
        """Apply markdown formatting to the response content."""
        # This is a basic implementation - in practice, you'd use a markdown library
        # For now, we'll just ensure proper spacing and basic formatting
        
        # Clean up extra whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        # Ensure proper paragraph spacing
        content = re.sub(r'([.!?])\s*\n([A-Z])', r'\1\n\n\2', content)
        
        return content.strip()
    
    def _validate_input(self, input_data: AgentInput) -> bool:
        """Validate input data for the response agent."""
        # Call parent validation
        if not super()._validate_input(input_data):
            return False
        
        # Response agent requires either a query or context
        if not input_data.query and not input_data.context:
            self.logger.warning("Response agent requires either a query or context")
            return False
        
        return True
    
    def _prepare_context_for_llm(self, agent_outputs: Dict[str, Any], input_data: AgentInput) -> str:
        """Prepare context summary for LLM response generation."""
        context_parts = []
        
        # Add search results summary
        search_info = agent_outputs.get('search', {})
        if search_info and 'results' in search_info:
            results = search_info['results']
            if results:
                context_parts.append(f"Search Results ({len(results)} found):")
                for i, result in enumerate(results[:3]):  # Top 3 results
                    content = result.get('content', '')
                    context_parts.append(f"  {i+1}. {content}")
        
        # Add image analysis summary
        image_info = agent_outputs.get('image_analysis', {})
        if image_info and 'description' in image_info:
            context_parts.append(f"Image Analysis: {image_info['description']}")
        
        # Add image retrieval summary
        image_retrieval_info = agent_outputs.get('image_retrieval', {})
        if image_retrieval_info and image_retrieval_info.get('success') and image_retrieval_info.get('image'):
            image_data = image_retrieval_info['image']
            filename = image_data.get('filename', 'unknown')
            relevance_score = image_data.get('relevance_score', 0.0)
            context_parts.append(f"Image Display: Showing image '{filename}' (relevance: {relevance_score:.2f}) - inform the user that an image is being displayed")
        
        # Add reasoning summary
        reasoning_info = agent_outputs.get('reasoning', {})
        if reasoning_info and 'logical_connections' in reasoning_info:
            context_parts.append(f"Logical Analysis: {reasoning_info['logical_connections']}")
        
        # Add timeline summary
        timeline_info = agent_outputs.get('timeline', {})
        if timeline_info and 'chronological_context' in timeline_info:
            context_parts.append(f"Timeline Context: {timeline_info['chronological_context']}")
        
        return "\n\n".join(context_parts) if context_parts else "Limited context available."
    
    def _generate_response_metadata(self, response: Dict[str, Any], input_data: AgentInput) -> Dict[str, Any]:
        """Generate metadata for the response."""
        return {
            'response_type': response.get('response_type', 'general'),
            'confidence_score': response.get('confidence_score', 0.0),
            'sources_used': response.get('sources_used', []),
            'processing_summary': response.get('processing_summary', {}),
            'query_length': len(input_data.query) if input_data.query else 0,
            'has_image': input_data.image_data is not None,
            'modality': input_data.modality,
            'timestamp': time.time()
        }
