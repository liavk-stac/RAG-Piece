#!/usr/bin/env python3
"""
Test Router Logic Updates

This script tests the updated router agent logic for smart image retrieval decisions.
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('../..'))

from src.chatbot.config import ChatbotConfig
from src.chatbot.agents.router_agent import RouterAgent, QueryIntent, QueryComplexity, Modality
from src.chatbot.agents.base_agent import AgentInput


def test_router_logic():
    """Test the updated router logic for image retrieval decisions."""
    print("🧪 Testing Updated Router Logic")
    print("=" * 50)
    
    # Initialize configuration and router
    config = ChatbotConfig()
    router = RouterAgent(config)
    
    # Test cases
    test_cases = [
        {
            "query": "What is One Piece?",
            "description": "Primary One Piece content query",
            "expected_image_retrieval": True
        },
        {
            "query": "Tell me about the Straw Hat Pirates",
            "description": "Crew-focused query",
            "expected_image_retrieval": True
        },
        {
            "query": "Who is Monkey D. Luffy?",
            "description": "Character-focused query",
            "expected_image_retrieval": True
        },
        {
            "query": "What happened next?",
            "description": "Follow-up question",
            "expected_image_retrieval": False
        },
        {
            "query": "Tell me more about that",
            "description": "Follow-up indicator",
            "expected_image_retrieval": False
        },
        {
            "query": "How are you?",
            "description": "General conversation",
            "expected_image_retrieval": False
        },
        {
            "query": "What's the weather like?",
            "description": "Non-One Piece query",
            "expected_image_retrieval": False
        },
        {
            "query": "Show me a picture of the crew",
            "description": "Explicit image request",
            "expected_image_retrieval": True
        },
        {
            "query": "What about the Going Merry ship?",
            "description": "Ship-related query",
            "expected_image_retrieval": True
        }
    ]
    
    # Run tests
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['description']}")
        print(f"Query: \"{test_case['query']}\"")
        
        # Create mock input data
        input_data = AgentInput(
            query=test_case['query'],
            image_data=None,
            conversation_history=[],
            session_id="test-session"
        )
        
        # Test the logic
        try:
            # Mock the required parameters
            intent = QueryIntent.SEARCH  # Default intent
            complexity = QueryComplexity.SIMPLE  # Default complexity
            modality = Modality.TEXT_ONLY  # Default modality
            
            # Test the image retrieval decision
            should_include = router._should_include_image_retrieval(intent, complexity, modality, input_data)
            
            # Check result
            status = "✅ PASS" if should_include == test_case['expected_image_retrieval'] else "❌ FAIL"
            print(f"Expected: {test_case['expected_image_retrieval']}, Got: {should_include} {status}")
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print("\n🎉 Router logic testing completed!")


if __name__ == "__main__":
    test_router_logic()
