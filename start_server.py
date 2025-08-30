#!/usr/bin/env python3
"""
Start the One Piece Chatbot Web Interface

This script starts the Flask server for Phase 7 testing.
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

def main():
    """Start the Flask web server."""
    try:
        print("🚀 Starting One Piece Chatbot Web Interface...")
        print("=" * 60)
        
        # Import required modules
        from src.chatbot.interfaces.web_interface import ChatbotWebInterface
        from src.chatbot.config import ChatbotConfig
        
        # Initialize configuration
        print("📋 Loading configuration...")
        config = ChatbotConfig()
        print("✅ Configuration loaded successfully")
        
        # Initialize web interface
        print("🌐 Initializing web interface...")
        web_interface = ChatbotWebInterface(config)
        print("✅ Web interface initialized successfully")
        
        # Start the server
        print("🚀 Starting Flask server on http://127.0.0.1:5000")
        print("=" * 60)
        print("🌐 Server is running! Open your browser to: http://127.0.0.1:5000")
        print("⏹️  Press Ctrl+C to stop the server")
        print("=" * 60)
        
        web_interface.run(debug=False, host='127.0.0.1', port=5000)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
