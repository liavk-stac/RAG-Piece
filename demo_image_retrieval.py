#!/usr/bin/env python3
"""
Demo Script for Image Retrieval Functionality

This script demonstrates the image retrieval system by running
sample queries and showing the results.
"""

import os
import sys
import time

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from src.chatbot.config import ChatbotConfig
from src.chatbot.core.orchestrator import ChatbotOrchestrator


def run_image_retrieval_demo():
    """Run a demonstration of the image retrieval functionality."""
    print("🚀 One Piece Chatbot - Image Retrieval Demo")
    print("=" * 60)
    
    # Initialize configuration
    config = ChatbotConfig()
    config.LOG_LEVEL = "INFO"
    config.LOG_TO_FILE = False
    
    print(f"📁 Images directory: {config.IMAGES_PATH}")
    print(f"🔍 Image index: {config.IMAGE_INDEX_PATH}")
    print(f"📊 Relevance threshold: {config.IMAGE_RELEVANCE_THRESHOLD}")
    print()
    
    # Initialize orchestrator
    try:
        orchestrator = ChatbotOrchestrator(config)
        print("✅ Chatbot orchestrator initialized successfully")
        print()
    except Exception as e:
        print(f"❌ Failed to initialize orchestrator: {e}")
        return False
    
    # Demo queries
    demo_queries = [
        "Tell me about the Straw Hat Pirates crew",
        "Who is Monkey D. Luffy?",
        "What can you tell me about Roronoa Zoro?",
        "Show me information about the Going Merry ship",
        "Tell me about the Thousand Sunny"
    ]
    
    print("🧪 Running Image Retrieval Demo Queries")
    print("-" * 60)
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 40)
        
        try:
            # Process query
            start_time = time.time()
            response = orchestrator.process_query(query)
            processing_time = time.time() - start_time
            
            # Display response
            print(f"⏱️  Processing time: {processing_time:.2f}s")
            print(f"🎯 Confidence: {response.get('confidence', 0):.1%}")
            
            # Check for image
            if 'image' in response:
                image_data = response['image']
                print(f"🖼️  Image retrieved: {image_data.get('filename', 'Unknown')}")
                print(f"👤 Character: {image_data.get('character', 'Unknown')}")
                print(f"📝 Content: {image_data.get('content', 'Unknown')}")
                print(f"⭐ Relevance score: {image_data.get('relevance_score', 0):.1%}")
                print(f"📁 Path: {image_data.get('path', 'Unknown')}")
                
                # Check if image file exists
                if os.path.exists(image_data.get('path', '')):
                    print("✅ Image file exists on disk")
                else:
                    print("⚠️  Image file not found on disk")
            else:
                print("ℹ️  No image retrieved for this query")
            
            # Display response text (truncated)
            response_text = response.get('response', '')
            if len(response_text) > 200:
                response_text = response_text[:200] + "..."
            print(f"💬 Response: {response_text}")
            
            # Display metadata
            metadata = response.get('metadata', {})
            if metadata:
                agents_executed = metadata.get('agents_executed', [])
                if agents_executed:
                    print(f"🤖 Agents executed: {', '.join(agents_executed)}")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
        
        print()
    
    # Display final statistics
    print("📊 Demo Summary")
    print("-" * 60)
    
    try:
        summary = orchestrator.get_conversation_summary()
        print(f"Total queries processed: {summary.get('total_queries_processed', 0)}")
        print(f"Successful responses: {summary.get('successful_responses', 0)}")
        print(f"Failed responses: {summary.get('failed_responses', 0)}")
        print(f"Average response time: {summary.get('average_response_time', 0):.2f}s")
    except Exception as e:
        print(f"Could not get summary: {e}")
    
    print("\n🎉 Image retrieval demo completed!")
    return True


def main():
    """Main function to run the demo."""
    try:
        success = run_image_retrieval_demo()
        if success:
            print("\n✅ Demo completed successfully!")
        else:
            print("\n❌ Demo failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
