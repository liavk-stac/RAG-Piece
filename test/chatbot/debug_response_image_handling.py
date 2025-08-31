#!/usr/bin/env python3
"""
Debug script to test specifically how the response agent handles image data.
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('../..'))

from src.chatbot.config import ChatbotConfig
from src.chatbot.agents.response_agent import ResponseAgent
from src.chatbot.agents.base_agent import AgentInput

def test_response_image_handling():
    """Test specifically how the response agent handles image data."""
    print("🔍 Testing Response Agent Image Handling...")
    
    try:
        # Initialize configuration
        config = ChatbotConfig()
        print("✅ Configuration loaded")
        
        # Initialize response agent
        agent = ResponseAgent(config)
        print("✅ Response Agent initialized")
        
        # Create mock agent outputs that exactly match what the orchestrator passes
        mock_agent_outputs = {
            'router': {
                'intent': 'search',
                'complexity': 'simple',
                'modality': 'text_only'
            },
            'search': {
                'results': [
                    {'content': 'Sample search result 1', 'source': 'test1'},
                    {'content': 'Sample search result 2', 'source': 'test2'}
                ],
                'query_enhancement': 'Enhanced query',
                'search_strategy': 'hybrid'
            },
            'image_retrieval': {
                'success': True,
                'image': {
                    'path': 'data/images/Straw_Hat_Pirates/Featured_Article.png',
                    'filename': 'Featured_Article',
                    'character': 'Straw Hat Pirates',
                    'content': 'Featured Article',
                    'type': 'crew_group',
                    'relevance_score': 1.0,
                    'metadata': {
                        'folder': 'Straw_Hat_Pirates',
                        'extension': '.png',
                        'searchable_terms': ['featured', 'article', 'straw', 'hat', 'pirates']
                    }
                },
                'intent_analysis': {
                    'intent_type': 'crew_focus',
                    'confidence': 0.9
                },
                'candidates_count': 14
            }
        }
        
        print(f"📊 Mock agent outputs structure:")
        print(f"   Keys: {list(mock_agent_outputs.keys())}")
        print(f"   Image retrieval present: {'image_retrieval' in mock_agent_outputs}")
        print(f"   Image retrieval success: {mock_agent_outputs['image_retrieval'].get('success')}")
        print(f"   Image data present: {'image' in mock_agent_outputs['image_retrieval']}")
        
        # Create agent input with the mock data
        agent_input = AgentInput(
            query="Tell me about the Straw Hat Pirates",
            image_data=None,
            conversation_history=[],
            context={'agent_outputs': mock_agent_outputs}
        )
        
        # Execute the agent
        print("🚀 Executing response agent...")
        result = agent.execute(agent_input)
        
        print(f"📊 Response agent execution result:")
        print(f"   Success: {result.success}")
        print(f"   Error: {result.error_message if not result.success else 'None'}")
        
        if result.success:
            response_result = result.result
            print(f"   Response keys: {list(response_result.keys())}")
            
            # Check if image data is present in the final output
            if 'image' in response_result:
                print(f"   ✅ Image data found in final output")
                image_data = response_result['image']
                print(f"   Image path: {image_data.get('path', 'None')}")
                print(f"   Image filename: {image_data.get('filename', 'None')}")
                print(f"   Image character: {image_data.get('character', 'None')}")
                print(f"   Image relevance score: {image_data.get('relevance_score', 'None')}")
            else:
                print(f"   ❌ No image data in final output")
                
            # Check if image_metadata is present
            if 'image_metadata' in response_result:
                print(f"   ✅ Image metadata found: {response_result['image_metadata']}")
            else:
                print(f"   ❌ No image metadata in final output")
                
            # Check the response content
            if 'response' in response_result:
                response_text = response_result['response']
                print(f"   Response length: {len(response_text)}")
                print(f"   Response preview: {response_text[:200]}...")
            else:
                print(f"   ❌ No response content found")
        else:
            print(f"   ❌ Response agent execution failed: {result.error_message}")
            
    except Exception as e:
        print(f"❌ Error testing response image handling: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_response_image_handling()
