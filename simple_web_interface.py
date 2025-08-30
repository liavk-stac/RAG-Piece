#!/usr/bin/env python3
"""
Simple One Piece Chatbot Web Interface

This is a simplified version to test basic functionality.
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from flask import Flask, request, jsonify, render_template_string
from src.chatbot.config import ChatbotConfig
from src.chatbot.core.chatbot import OnePieceChatbot

def create_simple_web_interface():
    """Create a simple working web interface."""
    
    # Initialize configuration and chatbot
    config = ChatbotConfig()
    chatbot = OnePieceChatbot(config)
    
    app = Flask(__name__)
    
    # Simple HTML template
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>One Piece Chatbot - Simple</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .chat-container {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .chat-header {
            text-align: center;
            margin-bottom: 20px;
            color: #333;
        }
        .chat-messages {
            height: 400px;
            overflow-y: auto;
            border: 1px solid #ddd;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 5px;
            background-color: #fafafa;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 5px;
        }
        .user-message {
            background-color: #007bff;
            color: white;
            margin-left: 20%;
            text-align: right;
        }
        .bot-message {
            background-color: #e9ecef;
            color: #333;
            margin-right: 20%;
        }
        .input-container {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        .text-input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
        }
        .send-button {
            padding: 10px 20px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .send-button:hover {
            background-color: #0056b3;
        }
        .status {
            text-align: center;
            margin-bottom: 20px;
            padding: 10px;
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 5px;
            color: #155724;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>🏴‍☠️ One Piece Chatbot - Simple</h1>
            <p>Ask me anything about the One Piece world!</p>
        </div>
        
        <div id="status" class="status">
            Chatbot is ready! Ask me a question about One Piece.
        </div>
        
        <div class="chat-messages" id="chatMessages">
            <div class="message bot-message">
                Hello! I'm your One Piece expert. I can answer questions about characters, locations, events, and more!
            </div>
        </div>
        
        <div class="input-container">
            <input type="text" id="messageInput" class="text-input" placeholder="Ask me about One Piece...">
            <button onclick="sendMessage()" class="send-button">Send</button>
        </div>
        
        <div style="text-align: center;">
            <button onclick="resetChat()" style="margin-right: 10px; padding: 8px 16px; background-color: #6c757d; color: white; border: none; border-radius: 5px; cursor: pointer;">Reset Chat</button>
        </div>
    </div>

    <script>
        let sessionId = null;
        
        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            // Add user message to chat
            addMessage(message, 'user');
            input.value = '';
            
            // Show loading message
            const loadingId = addMessage('Thinking...', 'bot');
            
            // Send message to API
            fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    session_id: sessionId
                })
            })
            .then(response => response.json())
            .then(data => {
                // Remove loading message
                removeMessage(loadingId);
                
                if (data.error) {
                    addMessage('Error: ' + data.error, 'bot');
                } else {
                    // Update session ID if provided
                    if (data.session_id) {
                        sessionId = data.session_id;
                    }
                    
                    // Add bot response
                    addMessage(data.response, 'bot');
                    
                    // Update status
                    updateStatus('Response received successfully!');
                }
            })
            .catch(error => {
                removeMessage(loadingId);
                addMessage('Error: Failed to send message', 'bot');
                console.error('Error:', error);
            });
        }
        
        function addMessage(text, sender, className = '') {
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}-message ${className}`;
            messageDiv.textContent = text;
            
            const messageId = 'msg_' + Date.now();
            messageDiv.id = messageId;
            
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            return messageId;
        }
        
        function removeMessage(messageId) {
            const message = document.getElementById(messageId);
            if (message) {
                message.remove();
            }
        }
        
        function resetChat() {
            // Clear chat messages
            const messagesDiv = document.getElementById('chatMessages');
            messagesDiv.innerHTML = '<div class="message bot-message">Hello! I\'m your One Piece expert. I can answer questions about characters, locations, events, and more!</div>';
            
            // Reset session
            sessionId = null;
            updateStatus('Chat reset successfully!');
        }
        
        function updateStatus(message) {
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = message;
        }
        
        // Handle Enter key
        document.getElementById('messageInput').addEventListener('keypress', function(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        });
    </script>
</body>
</html>
    """
    
    @app.route('/')
    def index():
        """Main page of the chatbot interface."""
        return render_template_string(html_template)
    
    @app.route('/api/chat', methods=['POST'])
    def chat():
        """Handle text-based chat messages."""
        try:
            data = request.get_json()
            if not data or 'message' not in data:
                return jsonify({'error': 'Message is required'}), 400
            
            message = data['message']
            session_id = data.get('session_id')
            
            print(f"Received chat message: {message[:100]}...")
            
            # Process the message
            response = chatbot.ask(message, session_id)
            
            return jsonify(response)
            
        except Exception as e:
            print(f"Chat error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/status')
    def status():
        """Get chatbot status information."""
        try:
            return jsonify({
                'status': 'ready',
                'message': 'Chatbot is running successfully'
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return app

def main():
    """Start the simple web interface."""
    try:
        print("🚀 Starting Simple One Piece Chatbot Web Interface...")
        print("=" * 60)
        
        # Create the web interface
        app = create_simple_web_interface()
        
        print("✅ Web interface created successfully")
        print("🌐 Starting Flask server on http://127.0.0.1:5000")
        print("=" * 60)
        print("🌐 Server is running! Open your browser to: http://127.0.0.1:5000")
        print("⏹️  Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Start the server
        app.run(debug=False, host='127.0.0.1', port=5000)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
