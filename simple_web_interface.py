#!/usr/bin/env python3
"""
Simple Web Interface for One Piece Chatbot

This is a clean, simple implementation to fix the JavaScript syntax issues.
"""

import os
import sys
from flask import Flask, request, jsonify, send_from_directory
import logging

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from src.chatbot.core.chatbot import OnePieceChatbot
from src.chatbot.config import ChatbotConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
chatbot = None
config = None

def create_simple_html_template():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>One Piece Chatbot - Simple</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f5f5f5; }
        .chat-container { background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .chat-header { text-align: center; margin-bottom: 20px; color: #333; }
        .chat-messages { height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 5px; background-color: #fafafa; }
        .message { margin-bottom: 15px; padding: 10px; border-radius: 5px; }
        .user-message { background-color: #007bff; color: white; margin-left: 20%; text-align: right; }
        .bot-message { background-color: #e9ecef; color: #333; margin-right: 20%; }
        .input-container { display: flex; gap: 10px; margin-bottom: 20px; }
        .text-input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 16px; }
        .send-button { padding: 10px 20px; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .send-button:hover { background-color: #0056b3; }
        .status { text-align: center; margin-bottom: 20px; padding: 10px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; color: #155724; }
        .error { background-color: #f8d7da; border-color: #f5c6cb; color: #721c24; }
        .loading { text-align: center; color: #666; font-style: italic; }
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
                Hello! I am your One Piece expert. I can answer questions about characters, locations, events, and more!
            </div>
        </div>

        <div class="input-container">
            <input type="text" id="messageInput" class="text-input" placeholder="Ask me about One Piece..." onkeypress="handleKeyPress(event)">
            <button onclick="sendMessage()" class="send-button">Send</button>
        </div>

        <div style="text-align: center;">
            <button onclick="resetConversation()" style="margin-right: 10px; padding: 8px 16px; background-color: #6c757d; color: white; border: none; border-radius: 5px; cursor: pointer;">Reset Chat</button>
            <button onclick="getStatus()" style="padding: 8px 16px; background-color: #17a2b8; color: white; border: none; border-radius: 5px; cursor: pointer;">Status</button>
        </div>
    </div>

    <script>
        let sessionId = null;

        function addMessage(text, sender, className) {
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + sender + '-message ' + (className || '');
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

        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();

            if (!message) return;

            addMessage(message, 'user');
            input.value = '';

            const loadingId = addMessage('Thinking...', 'bot', 'loading');

            fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: message, session_id: sessionId })
            })
            .then(response => response.json())
            .then(data => {
                removeMessage(loadingId);
                if (data.error) {
                    addMessage('Error: ' + data.error, 'bot', 'error');
                } else {
                    if (data.session_id) {
                        sessionId = data.session_id;
                    }
                    addMessage(data.response, 'bot');
                    updateStatus('Response received');
                }
            })
            .catch(error => {
                removeMessage(loadingId);
                addMessage('Error: Failed to send message', 'bot', 'error');
                console.error('Error:', error);
            });
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }

        function resetConversation() {
            fetch('/api/reset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    updateStatus('Error: ' + data.error, 'error');
                } else {
                    const messagesDiv = document.getElementById('chatMessages');
                    messagesDiv.innerHTML = '<div class="message bot-message">Hello! I am your One Piece expert. I can answer questions about characters, locations, events, and more!</div>';
                    sessionId = null;
                    updateStatus('Conversation reset successfully');
                }
            })
            .catch(error => {
                updateStatus('Error: Failed to reset conversation', 'error');
                console.error('Error:', error);
            });
        }

        function getStatus() {
            fetch('/api/status')
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    updateStatus('Error: ' + data.error, 'error');
                } else {
                    const uptime = Math.floor(data.uptime / 60);
                    updateStatus('Chatbot Status: Ready | Uptime: ' + uptime + ' minutes | Total Queries: ' + (data.conversation_summary?.total_queries_processed || 0));
                }
            })
            .catch(error => {
                updateStatus('Error: Failed to get status', 'error');
                console.error('Error:', error);
            });
        }

        function updateStatus(message, className) {
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = message;
            statusDiv.className = 'status ' + (className || '');
        }

        console.log('JavaScript loaded successfully!');
    </script>
</body>
</html>'''

@app.route('/')
def index():
    return create_simple_html_template()

@app.route('/api/chat', methods=['POST'])
def chat():
    global chatbot, config
    if chatbot is None:
        config = ChatbotConfig()
        chatbot = OnePieceChatbot(config)
        logger.info("Chatbot initialized in simple_web_interface")

    data = request.get_json()
    message = data['message']
    session_id = data.get('session_id')
    response = chatbot.ask(message, session_id)
    return jsonify(response)

@app.route('/api/reset', methods=['POST'])
def reset_conversation():
    global chatbot
    if chatbot:
        session_id = request.json.get('session_id') if request.json else None
        chatbot.reset_conversation(session_id)
    return jsonify({'message': 'Conversation reset successfully'})

@app.route('/api/status')
def status():
    global chatbot, config
    if chatbot is None:
        config = ChatbotConfig()
        chatbot = OnePieceChatbot(config)
    
    status_info = chatbot.get_chatbot_status()
    return jsonify(status_info)

@app.route('/api/image/<path:image_path>')
def serve_image(image_path):
    global config
    if config is None:
        config = ChatbotConfig()
    images_dir = config.IMAGES_PATH
    return send_from_directory(images_dir, image_path)

if __name__ == '__main__':
    logger.info("🚀 Starting Simple One Piece Chatbot Web Interface...")
    logger.info("=" * 60)
    try:
        config = ChatbotConfig()
        chatbot = OnePieceChatbot(config)
        logger.info("✅ Web interface created successfully")
        logger.info("🌐 Starting Flask server on http://127.0.0.1:5000")
        logger.info("=" * 60)
        logger.info("🌐 Server is running! Open your browser to: http://127.0.0.1:5000")
        logger.info("⏹️  Press Ctrl+C to stop the server")
        logger.info("=" * 60)
        app.run(debug=False, host='127.0.0.1', port=5000)
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)
