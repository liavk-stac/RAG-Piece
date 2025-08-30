#!/usr/bin/env python3
"""
Debug script to check the final response structure from the orchestrator.
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from src.chatbot.config import ChatbotConfig
from src.chatbot.core.orchestrator import ChatbotOrchestrator

def test_final_response():
    """Test the final response structure from the orchestrator."""
    print("🔍 Testing Final Response Structure...")
    
    try:
        # Initialize configuration
        config = ChatbotConfig()
        print("✅ Configuration loaded")
        
        # Initialize orchestrator
        orchestrator = ChatbotOrchestrator(config)
        print("✅ Orchestrator initialized")
        
        # Test query that should trigger image retrieval
        test_query = "Tell me about the Straw Hat Pirates"
        print(f"📝 Test query: {test_query}")
        
        # Process the query through the full pipeline
        print("🚀 Processing query through full pipeline...")
        result = orchestrator.process_query(test_query)
        
        print(f"📊 Final response structure:")
        print(f"   Response keys: {list(result.keys())}")
        
        # Check if image data is present in the final response
        if 'image' in result:
            print(f"   ✅ Image data found in final response")
            image_data = result['image']
            if image_data:
                print(f"   Image path: {image_data.get('path', 'None')}")
                print(f"   Image filename: {image_data.get('filename', 'None')}")
                print(f"   Image character: {image_data.get('character', 'None')}")
                print(f"   Image relevance score: {image_data.get('relevance_score', 'None')}")
            else:
                print(f"   Image data is None")
        else:
            print(f"   ❌ No image data in final response")
            
        # Check other response fields
        print(f"   Response text: {result.get('response', '')[:100]}...")
        print(f"   Confidence: {result.get('confidence', 0.0)}")
        print(f"   Processing time: {result.get('processing_time', 0.0):.2f}s")
        print(f"   Session ID: {result.get('session_id', 'None')}")
        print(f"   Conversation turn: {result.get('conversation_turn', 0)}")
        
        # Check metadata
        metadata = result.get('metadata', {})
        print(f"   Agents executed: {metadata.get('agents_executed', [])}")
        print(f"   Pipeline success: {metadata.get('pipeline_success', False)}")
        print(f"   Modality: {metadata.get('modality', 'Unknown')}")
            
    except Exception as e:
        print(f"❌ Error testing final response: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_final_response()
