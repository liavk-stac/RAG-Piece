"""
LLM Client Module

Provides a shared OpenAI client for all agents to use GPT-4o-mini.
Handles API calls, retries, and error management.
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional
from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

from ..config import ChatbotConfig


class LLMClient:
    """Shared LLM client for OpenAI GPT-4o-mini integration."""
    
    def __init__(self, config: ChatbotConfig):
        """Initialize the LLM client."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Get API key from environment
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=self.api_key,
            timeout=self.config.LLM_REQUEST_TIMEOUT
        )
        
        self.logger.info(f"LLM client initialized with model: {self.config.LLM_MODEL_NAME}")
    
    def generate_text(self, 
                     prompt: str, 
                     system_message: str = None,
                     max_tokens: int = None,
                     temperature: float = None) -> str:
        """
        Generate text using GPT-4o-mini.
        
        Args:
            prompt: User prompt/message
            system_message: Optional system message for context
            max_tokens: Maximum tokens to generate
            temperature: Creativity level (0.0-2.0)
            
        Returns:
            Generated text response
        """
        try:
            # Prepare messages
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt})
            
            # Set parameters
            max_tokens = max_tokens or self.config.MAX_LLM_TOKENS
            temperature = temperature or self.config.LLM_MODEL_TEMPERATURE
            
            # Make API call with retries
            for attempt in range(self.config.LLM_MAX_RETRIES):
                try:
                    response = self.client.chat.completions.create(
                        model=self.config.LLM_MODEL_NAME,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        timeout=self.config.LLM_REQUEST_TIMEOUT
                    )
                    
                    # Extract response text
                    if response.choices and len(response.choices) > 0:
                        return response.choices[0].message.content
                    else:
                        raise ValueError("No response content received")
                        
                except Exception as e:
                    if attempt < self.config.LLM_MAX_RETRIES - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        self.logger.warning(f"LLM API call failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise e
                        
        except Exception as e:
            self.logger.error(f"Failed to generate text: {e}")
            raise
    
    def analyze_image(self, 
                     image_data: bytes,
                     prompt: str,
                     max_tokens: int = None) -> str:
        """
        Analyze image using GPT-4o vision capabilities.
        
        Args:
            image_data: Image bytes
            prompt: Analysis prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Image analysis text
        """
        try:
            # Convert bytes to base64 for OpenAI API
            import base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Prepare messages with image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            # Set parameters
            max_tokens = max_tokens or self.config.MAX_VISION_TOKENS
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.config.VISION_MODEL_NAME,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.config.VISION_MODEL_TEMPERATURE,
                timeout=self.config.LLM_REQUEST_TIMEOUT
            )
            
            # Extract response text
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                raise ValueError("No response content received")
                
        except Exception as e:
            self.logger.error(f"Failed to analyze image: {e}")
            raise
    
    def generate_reasoning(self, 
                          context: str,
                          query: str,
                          search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate logical reasoning using LLM.
        
        Args:
            context: Query context
            query: User query
            search_results: Search results to reason about
            
        Returns:
            Reasoning results
        """
        try:
            # Format search results for the prompt
            results_text = self._format_results_for_prompt(search_results)
            
            system_message = """You are a logical reasoning agent for One Piece knowledge. 
            Analyze the provided information and identify logical connections, relationships, 
            and patterns. Be precise and accurate with One Piece lore."""
            
            prompt = f"""Context: {context}
Query: {query}

Search Results:
{results_text}

Please analyze this information and provide:
1. Logical connections between the pieces of information
2. Relationships and patterns you can identify
3. Any causal relationships or implications
4. Classification or categorization of the information
5. Confidence level in your analysis (0.0-1.0)

Format your response as structured analysis."""
            
            response = self.generate_text(prompt, system_message)
            
            # Parse the response (simplified for now)
            return {
                'reasoning_analysis': response,
                'confidence_score': 0.8,  # Default confidence
                'logical_connections': 'Extracted from LLM response',
                'relationships': 'Identified by LLM analysis'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate reasoning: {e}")
            return {
                'reasoning_analysis': 'Reasoning analysis failed',
                'confidence_score': 0.0,
                'logical_connections': 'Unable to analyze',
                'relationships': 'Unable to identify'
            }
    
    def generate_timeline_analysis(self, 
                                  query: str,
                                  search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate timeline analysis using LLM.
        
        Args:
            query: User query
            search_results: Search results to analyze
            
        Returns:
            Timeline analysis results
        """
        try:
            results_text = self._format_results_for_prompt(search_results)
            
            system_message = """You are a timeline analysis agent for One Piece knowledge. 
            Extract chronological information, identify key events, and organize information 
            in temporal order. Be accurate with One Piece timeline and eras."""
            
            prompt = f"""Query: {query}

Search Results:
{results_text}

Please analyze this information and provide:
1. Chronological order of events mentioned
2. Key timeline events and their significance
3. Historical context and era information
4. Related timeline information
5. Confidence level in your analysis (0.0-1.0)

Format your response as structured timeline analysis."""
            
            response = self.generate_text(prompt, system_message)
            
            return {
                'timeline_events': response,
                'key_events': 'Extracted from LLM response',
                'historical_context': 'Identified by LLM analysis',
                'related_timeline': 'Timeline connections found',
                'confidence_score': 0.8
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate timeline analysis: {e}")
            return {
                'timeline_events': 'Timeline analysis failed',
                'key_events': 'Unable to extract',
                'historical_context': 'Unable to identify',
                'related_timeline': 'Unable to analyze',
                'confidence_score': 0.0
            }
    
    def _format_results_for_prompt(self, search_results: List[Dict[str, Any]]) -> str:
        """Format search results for inclusion in LLM prompts."""
        if not search_results:
            return "No search results available."
        
        formatted = []
        for i, result in enumerate(search_results[:5]):  # Limit to top 5 results
            content = result.get('content', '')[:200] + "..." if len(result.get('content', '')) > 200 else result.get('content', '')
            metadata = result.get('metadata', {})
            article_name = metadata.get('article_name', 'Unknown')
            section_name = metadata.get('section_name', 'Unknown')
            
            formatted.append(f"Result {i+1} - {article_name} ({section_name}):\n{content}")
        
        return "\n\n".join(formatted)
    
    def is_available(self) -> bool:
        """Check if the LLM client is available and working."""
        try:
            # Simple test call
            test_response = self.generate_text("Hello", max_tokens=10)
            return bool(test_response and len(test_response.strip()) > 0)
        except Exception as e:
            self.logger.warning(f"LLM client availability check failed: {e}")
            return False
