#!/usr/bin/env python3
"""
Test Web UI Components Directly

This script tests the Web UI components without starting the Flask server.
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('../..'))

from src.chatbot.config import ChatbotConfig
from src.chatbot.interfaces.web_interface import ChatbotWebInterface


def test_web_ui_components():
    """Test the Web UI components directly."""
    print("🧪 Testing Web UI Components Directly")
    print("=" * 60)
    
    try:
        # Initialize configuration
        config = ChatbotConfig()
        print("✅ Configuration loaded successfully")
        
        # Initialize web interface
        web_interface = ChatbotWebInterface(config)
        print("✅ Web interface initialized successfully")
        
        # Test chatbot initialization
        if hasattr(web_interface, 'chatbot') and web_interface.chatbot:
            print("✅ Chatbot instance created successfully")
        else:
            print("❌ Chatbot instance not created")
            return
        
        # Test Flask app initialization
        if hasattr(web_interface, 'app') and web_interface.app:
            print("✅ Flask app initialized successfully")
        else:
            print("❌ Flask app not initialized")
            return
        
        # Test routes setup
        routes = []
        for rule in web_interface.app.url_map.iter_rules():
            routes.append(f"{rule.methods} {rule.rule}")
        
        print(f"✅ Flask routes configured: {len(routes)} routes found")
        for route in routes:
            print(f"   - {route}")
        
        # Test HTML template generation
        print("\n📄 Testing HTML Template Generation...")
        try:
            # Test if we can get the HTML content directly
            from src.chatbot.interfaces.web_interface import create_html_template
            html_content = create_html_template()
            
            if html_content:
                print("✅ HTML template rendered successfully")
                print(f"📊 Template size: {len(html_content)} characters")
                
                # Check for key elements
                if 'chatMessages' in html_content:
                    print("✅ Chat interface HTML found")
                else:
                    print("❌ Chat interface HTML not found")
                
                if 'One Piece Chatbot' in html_content:
                    print("✅ One Piece Chatbot title found")
                else:
                    print("❌ One Piece Chatbot title not found")
                
                if 'image' in html_content.lower():
                    print("✅ Image-related HTML elements found")
                else:
                    print("ℹ️ No image-related HTML elements found")
                    
            else:
                print("❌ HTML template rendering failed")
                    
        except Exception as e:
            print(f"❌ HTML template test failed: {e}")
        
        # Test image serving route
        print("\n🖼️ Testing Image Serving Route...")
        try:
            # Check if image route exists
            image_routes = [r for r in routes if 'image' in r.lower()]
            if image_routes:
                print("✅ Image serving routes found:")
                for route in image_routes:
                    print(f"   - {route}")
            else:
                print("ℹ️ No image serving routes found")
                
        except Exception as e:
            print(f"❌ Image route test failed: {e}")
        
        # Test chat API endpoint
        print("\n📡 Testing Chat API Endpoint...")
        try:
            # Check if chat route exists
            chat_routes = [r for r in routes if 'chat' in r.lower()]
            if chat_routes:
                print("✅ Chat API routes found:")
                for route in chat_routes:
                    print(f"   - {route}")
            else:
                print("❌ No chat API routes found")
                
        except Exception as e:
            print(f"❌ Chat API test failed: {e}")
        
        print("\n🎉 Web UI components testing completed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_web_ui_components()
