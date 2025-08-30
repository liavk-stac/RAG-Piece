#!/usr/bin/env python3
"""
Test Web UI Integration

This script tests the complete Web UI integration with image retrieval.
"""

import os
import sys
import time
import requests

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from src.chatbot.config import ChatbotConfig
from src.chatbot.interfaces.web_interface import ChatbotWebInterface


def test_web_ui_integration():
    """Test the complete Web UI integration."""
    print("🧪 Testing Complete Web UI Integration")
    print("=" * 60)
    
    try:
        # Initialize configuration
        config = ChatbotConfig()
        print("✅ Configuration loaded successfully")
        
        # Initialize web interface
        web_interface = ChatbotWebInterface(config)
        print("✅ Web interface initialized successfully")
        
        # Start the server in a separate thread
        import threading
        server_thread = threading.Thread(
            target=web_interface.run,
            kwargs={'debug': False, 'host': '127.0.0.1', 'port': 5000}
        )
        server_thread.daemon = True
        server_thread.start()
        
        print("🌐 Starting Flask server on port 5000...")
        time.sleep(3)  # Wait for server to start
        
        # Test if server is running
        try:
            response = requests.get('http://127.0.0.1:5000', timeout=5)
            if response.status_code == 200:
                print("✅ Web server is running successfully")
                print(f"📄 Response length: {len(response.text)} characters")
                
                # Check if the response contains expected HTML elements
                if 'chatMessages' in response.text:
                    print("✅ Chat interface HTML found")
                else:
                    print("❌ Chat interface HTML not found")
                
                if 'One Piece Chatbot' in response.text:
                    print("✅ One Piece Chatbot title found")
                else:
                    print("❌ One Piece Chatbot title not found")
                
                # Test the chat API endpoint
                print("\n📡 Testing Chat API...")
                chat_response = requests.post(
                    'http://127.0.0.1:5000/api/chat',
                    json={'message': 'Tell me about the Straw Hat Pirates'},
                    timeout=30
                )
                
                if chat_response.status_code == 200:
                    print("✅ Chat API is working")
                    chat_data = chat_response.json()
                    print(f"📊 Response confidence: {chat_data.get('confidence', 'N/A')}")
                    print(f"📊 Processing time: {chat_data.get('processing_time', 'N/A')}")
                    
                    # Check for image data
                    if 'image' in chat_data and chat_data['image']:
                        print("✅ Image retrieved and included in response")
                        print(f"🖼️ Image: {chat_data['image'].get('filename', 'Unknown')}")
                    else:
                        print("ℹ️ No image in response (this might be expected)")
                        
                else:
                    print(f"❌ Chat API failed with status {chat_response.status_code}")
                    print(f"Error: {chat_response.text}")
                
            else:
                print(f"❌ Server responded with status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to connect to web server: {e}")
        
        print("\n🎉 Web UI integration testing completed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_web_ui_integration()
