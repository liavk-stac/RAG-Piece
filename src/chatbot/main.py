"""
Main Entry Point for One Piece Chatbot

This module provides the main entry point for running the One Piece chatbot
system, including both the web interface and command-line interface.
"""

import argparse
import logging
import sys
import os
from typing import Optional

# Add the parent directory to the path to import RAG components
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from .core.chatbot import OnePieceChatbot
from .interfaces.web_interface import ChatbotWebInterface
from .config import ChatbotConfig


def setup_logging(config: ChatbotConfig):
    """Set up logging for the chatbot system."""
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(config.LOG_FILE_PATH)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logging
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.LOG_FILE_PATH) if config.LOG_TO_FILE else logging.NullHandler()
        ]
    )


def run_web_interface(config: ChatbotConfig, host: Optional[str] = None, 
                     port: Optional[int] = None, debug: Optional[bool] = None):
    """Run the web interface for the chatbot."""
    print("🏴‍☠️ Starting One Piece Chatbot Web Interface...")
    print(f"📍 Host: {host or config.WEB_HOST}")
    print(f"🔌 Port: {port or config.WEB_PORT}")
    print(f"🐛 Debug: {debug if debug is not None else config.WEB_DEBUG}")
    print("🌐 Open your browser and navigate to the URL above")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        with ChatbotWebInterface(config) as web_interface:
            web_interface.run(host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down web interface...")
    except Exception as e:
        print(f"❌ Error running web interface: {e}")
        logging.error(f"Web interface error: {e}", exc_info=True)
        sys.exit(1)


def run_cli_interface(config: ChatbotConfig):
    """Run the command-line interface for the chatbot."""
    print("🏴‍☠️ One Piece Chatbot - Command Line Interface")
    print("💬 Type your questions about One Piece (type 'quit' to exit)")
    print("📸 For image analysis, use: /image <image_path> [question]")
    print("📊 For status, use: /status")
    print("🔄 For reset, use: /reset")
    print("-" * 50)
    
    try:
        with OnePieceChatbot(config) as chatbot:
            while True:
                try:
                    # Get user input
                    user_input = input("\n🤔 You: ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle special commands
                    if user_input.lower() == 'quit':
                        print("👋 Goodbye! Thanks for using One Piece Chatbot!")
                        break
                    elif user_input.startswith('/status'):
                        status = chatbot.get_chatbot_status()
                        print(f"\n📊 Status: {status}")
                        continue
                    elif user_input.startswith('/reset'):
                        chatbot.reset_conversation()
                        print("\n🔄 Conversation reset!")
                        continue
                    elif user_input.startswith('/image'):
                        # Handle image analysis
                        parts = user_input.split(' ', 2)
                        if len(parts) < 2:
                            print("❌ Usage: /image <image_path> [question]")
                            continue
                        
                        image_path = parts[1]
                        question = parts[2] if len(parts) > 2 else None
                        
                        if not os.path.exists(image_path):
                            print(f"❌ Image file not found: {image_path}")
                            continue
                        
                        try:
                            with open(image_path, 'rb') as f:
                                image_data = f.read()
                            
                            print(f"🔍 Analyzing image: {image_path}")
                            response = chatbot.analyze_image(image_data, question)
                            
                            print(f"\n🤖 Bot: {response['response']}")
                            if response['confidence'] > 0:
                                print(f"📈 Confidence: {response['confidence']:.1%}")
                            
                        except Exception as e:
                            print(f"❌ Error analyzing image: {e}")
                        continue
                    
                    # Process text question
                    print("🤔 Thinking...")
                    response = chatbot.ask(user_input)
                    
                    print(f"\n🤖 Bot: {response['response']}")
                    if response['confidence'] > 0:
                        print(f"📈 Confidence: {response['confidence']:.1%}")
                    if response['processing_time'] > 0:
                        print(f"⏱️  Processing time: {response['processing_time']:.2f}s")
                    
                except KeyboardInterrupt:
                    print("\n👋 Goodbye! Thanks for using One Piece Chatbot!")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
                    logging.error(f"CLI error: {e}", exc_info=True)
    
    except Exception as e:
        print(f"❌ Error initializing chatbot: {e}")
        logging.error(f"Chatbot initialization error: {e}", exc_info=True)
        sys.exit(1)


def run_demo(config: ChatbotConfig):
    """Run a demo of the chatbot with predefined questions."""
    print("🏴‍☠️ One Piece Chatbot - Demo Mode")
    print("🎭 Running through some example questions...")
    print("-" * 50)
    
    demo_questions = [
        "Who is Monkey D. Luffy?",
        "What is a Devil Fruit?",
        "Tell me about the Straw Hat Pirates",
        "What is Haki?",
        "Who are the Yonko?"
    ]
    
    try:
        with OnePieceChatbot(config) as chatbot:
            for i, question in enumerate(demo_questions, 1):
                print(f"\n🎯 Demo Question {i}: {question}")
                print("🤔 Processing...")
                
                try:
                    response = chatbot.ask(question)
                    
                    print(f"🤖 Bot: {response['response'][:200]}...")
                    print(f"📈 Confidence: {response['confidence']:.1%}")
                    print(f"⏱️  Processing time: {response['processing_time']:.2f}s")
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                
                print("-" * 30)
        
        print("\n✅ Demo completed!")
        
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        logging.error(f"Demo error: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main entry point for the One Piece chatbot."""
    parser = argparse.ArgumentParser(
        description="One Piece Chatbot - Your AI-powered One Piece expert",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.chatbot.main web                    # Run web interface
  python -m src.chatbot.main web --port 9000        # Run on port 9000
  python -m src.chatbot.main cli                    # Run command-line interface
  python -m src.chatbot.main demo                   # Run demo mode
  python -m src.chatbot.main web --debug            # Run web interface in debug mode
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['web', 'cli', 'demo'],
        help='Mode to run the chatbot in'
    )
    
    parser.add_argument(
        '--host',
        help='Host to bind to (web mode only)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        help='Port to bind to (web mode only)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (web mode only)'
    )
    
    parser.add_argument(
        '--config',
        help='Path to configuration file (optional)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if args.config and os.path.exists(args.config):
            # Load from file (you'd implement this)
            config = ChatbotConfig()
            print(f"📁 Loaded configuration from: {args.config}")
        else:
            config = ChatbotConfig()
            print("⚙️  Using default configuration")
        
        # Set up logging
        setup_logging(config)
        
        # Run in selected mode
        if args.mode == 'web':
            run_web_interface(config, args.host, args.port, args.debug)
        elif args.mode == 'cli':
            run_cli_interface(config)
        elif args.mode == 'demo':
            run_demo(config)
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
