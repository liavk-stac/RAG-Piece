"""
Web Interface for One Piece Chatbot

A basic Flask web interface that provides a user-friendly way to interact
with the One Piece chatbot system.
"""

from flask import Flask, render_template, request, jsonify, session
import base64
import logging
import time
from typing import Dict, Any, Optional

from ..core.chatbot import OnePieceChatbot
from ..config import ChatbotConfig


class ChatbotWebInterface:
    """
    Web interface for the One Piece chatbot.
    
    This class provides a Flask-based web interface that allows users to:
    - Ask text questions
    - Upload and analyze images
    - View conversation history
    - Monitor chatbot status
    """
    
    def __init__(self, config: Optional[ChatbotConfig] = None):
        """Initialize the web interface."""
        self.config = config or ChatbotConfig()
        self.logger = self._setup_logger()
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.secret_key = 'one_piece_chatbot_secret_key'
        
        # Initialize chatbot
        self.chatbot = OnePieceChatbot(self.config)
        
        # Set up routes
        self._setup_routes()
        
        self.logger.info("Web interface initialized successfully")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the web interface."""
        logger = logging.getLogger("chatbot.web_interface")
        logger.setLevel(self.config.LOG_LEVEL)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create console handler if not exists
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # Add file handler if enabled
            if self.config.LOG_TO_FILE:
                try:
                    file_handler = logging.FileHandler(self.config.LOG_FILE_PATH)
                    file_handler.setFormatter(formatter)
                    logger.addHandler(file_handler)
                except Exception as e:
                    logger.warning(f"Could not set up file logging: {e}")
        
        return logger
    
    def _setup_routes(self):
        """Set up Flask routes for the web interface."""
        
        @self.app.route('/')
        def index():
            """Main page of the chatbot interface."""
            return create_html_template()
        
        @self.app.route('/api/chat', methods=['POST'])
        def chat():
            """Handle text-based chat messages."""
            try:
                data = request.get_json()
                if not data or 'message' not in data:
                    return jsonify({'error': 'Message is required'}), 400
                
                message = data['message']
                session_id = data.get('session_id')
                
                self.logger.info(f"Received chat message: {message[:100]}...")
                
                # Process the message
                response = self.chatbot.ask(message, session_id)
                
                return jsonify(response)
                
            except Exception as e:
                self.logger.error(f"Chat error: {e}", exc_info=True)
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/analyze_image', methods=['POST'])
        def analyze_image():
            """Handle image analysis requests."""
            try:
                # Check if image file is present
                if 'image' not in request.files:
                    return jsonify({'error': 'Image file is required'}), 400
                
                image_file = request.files['image']
                if image_file.filename == '':
                    return jsonify({'error': 'No image file selected'}), 400
                
                # Get optional question
                question = request.form.get('question', '')
                session_id = request.form.get('session_id')
                
                # Read image data
                image_data = image_file.read()
                
                self.logger.info(f"Received image analysis request: {image_file.filename}")
                
                # Process the image
                response = self.chatbot.analyze_image(image_data, question, session_id)
                
                return jsonify(response)
                
            except Exception as e:
                self.logger.error(f"Image analysis error: {e}", exc_info=True)
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/status')
        def status():
            """Get chatbot status information."""
            try:
                status_info = self.chatbot.get_chatbot_status()
                return jsonify(status_info)
                
            except Exception as e:
                self.logger.error(f"Status error: {e}", exc_info=True)
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/history')
        def history():
            """Get conversation history."""
            try:
                session_id = request.args.get('session_id')
                history = self.chatbot.get_conversation_history(session_id)
                return jsonify(history)
                
            except Exception as e:
                self.logger.error(f"History error: {e}", exc_info=True)
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/reset', methods=['POST'])
        def reset_conversation():
            """Reset the current conversation."""
            try:
                session_id = request.json.get('session_id') if request.json else None
                self.chatbot.reset_conversation(session_id)
                return jsonify({'message': 'Conversation reset successfully'})
                
            except Exception as e:
                self.logger.error(f"Reset error: {e}", exc_info=True)
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/end_session', methods=['POST'])
        def end_session():
            """End the current session."""
            try:
                session_id = request.json.get('session_id') if request.json else None
                self.chatbot.end_session(session_id)
                return jsonify({'message': 'Session ended successfully'})
                
            except Exception as e:
                self.logger.error(f"End session error: {e}", exc_info=True)
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/image/<path:image_path>')
        def serve_image(image_path):
            """Serve images from the images directory."""
            try:
                import os
                from flask import send_from_directory
                
                # Ensure the path is safe (no directory traversal)
                if '..' in image_path or image_path.startswith('/'):
                    return jsonify({'error': 'Invalid image path'}), 400
                
                # Get the images directory from config
                images_dir = self.config.IMAGES_PATH
                full_path = os.path.join(images_dir, image_path)
                
                # Check if file exists and is within images directory
                if not os.path.exists(full_path) or not os.path.commonpath([images_dir, full_path]) == images_dir:
                    return jsonify({'error': 'Image not found'}), 404
                
                # Serve the image
                return send_from_directory(images_dir, image_path)
                
            except Exception as e:
                self.logger.error(f"Image serving error: {e}", exc_info=True)
                return jsonify({'error': str(e)}), 500
        
        # Error handlers
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({'error': 'Not found'}), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({'error': 'Internal server error'}), 500
    
    def run(self, host: Optional[str] = None, port: Optional[int] = None, 
            debug: Optional[bool] = None):
        """
        Run the web interface.
        
        Args:
            host: Host to bind to (defaults to config)
            port: Port to bind to (defaults to config)
            debug: Debug mode (defaults to config)
        """
        host = host or self.config.WEB_HOST
        port = port or self.config.WEB_PORT
        debug = debug if debug is not None else self.config.WEB_DEBUG
        
        self.logger.info(f"Starting web interface on {host}:{port}")
        
        try:
            self.app.run(
                host=host,
                port=port,
                debug=debug,
                use_reloader=self.config.WEB_RELOAD
            )
        except Exception as e:
            self.logger.error(f"Failed to start web interface: {e}")
            raise
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.chatbot:
                self.chatbot.cleanup()
            
            self.logger.info("Web interface cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during destruction


# Create a simple HTML template for the interface
def create_html_template():
    """Create a basic HTML template for the chatbot interface."""
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>One Piece Chatbot</title>
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
        .image-upload {
            margin-bottom: 20px;
        }
        .file-input {
            margin-right: 10px;
        }
        .upload-button {
            padding: 8px 16px;
            background-color: #28a745;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .upload-button:hover {
            background-color: #1e7e34;
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
        .error {
            background-color: #f8d7da;
            border-color: #f5c6cb;
            color: #721c24;
        }
        .loading {
            text-align: center;
            color: #666;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>🏴‍☠️ One Piece Chatbot</h1>
            <p>Ask me anything about the One Piece world!</p>
        </div>
        
        <div id="status" class="status">
            Chatbot is ready! Ask me a question about One Piece.
        </div>
        
        <div class="image-upload">
            <h3>📸 Image Analysis</h3>
            <input type="file" id="imageFile" class="file-input" accept="image/*">
            <input type="text" id="imageQuestion" placeholder="Ask about this image (optional)" class="text-input">
            <button onclick="analyzeImage()" class="upload-button">Analyze Image</button>
        </div>
        
        <div class="chat-messages" id="chatMessages">
            <div class="message bot-message">
                Hello! I'm your One Piece expert. I can answer questions about characters, locations, events, and more. I can also analyze images related to One Piece!
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
        
        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            // Add user message to chat
            addMessage(message, 'user');
            input.value = '';
            
            // Show loading message
            const loadingId = addMessage('Thinking...', 'bot', 'loading');
            
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
                    addMessage('Error: ' + data.error, 'bot', 'error');
                } else {
                    // Update session ID if provided
                    if (data.session_id) {
                        sessionId = data.session_id;
                    }
                    
                    // Add bot response with image if available
                    const imageData = data.image || null;
                    addMessage(data.response, 'bot', '', imageData);
                    
                    // Update status
                    updateStatus(`Response received in ${data.processing_time.toFixed(2)}s (Confidence: ${(data.confidence * 100).toFixed(1)}%)`);
                    
                    // Log image retrieval info if available
                    if (imageData) {
                        console.log('Image retrieved:', imageData);
                    }
                }
            })
            .catch(error => {
                removeMessage(loadingId);
                addMessage('Error: Failed to send message', 'bot', 'error');
                console.error('Error:', error);
            });
        }
        
        function analyzeImage() {
            const fileInput = document.getElementById('imageFile');
            const questionInput = document.getElementById('imageQuestion');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please select an image file');
                return;
            }
            
            const formData = new FormData();
            formData.append('image', file);
            formData.append('question', questionInput.value);
            if (sessionId) {
                formData.append('session_id', sessionId);
            }
            
            // Add user message to chat
            const message = questionInput.value || 'Analyze this image';
            addMessage(message + ' [Image: ' + file.name + ']', 'user');
            
            // Show loading message
            const loadingId = addMessage('Analyzing image...', 'bot', 'loading');
            
            // Send image to API
            fetch('/api/analyze_image', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Remove loading message
                removeMessage(loadingId);
                
                if (data.error) {
                    addMessage('Error: ' + data.error, 'bot', 'error');
                } else {
                    // Update session ID if provided
                    if (data.session_id) {
                        sessionId = data.session_id;
                    }
                    
                    // Add bot response
                    addMessage(data.response, 'bot');
                    
                    // Update status
                    updateStatus(`Image analyzed in ${data.processing_time.toFixed(2)}s (Confidence: ${(data.confidence * 100).toFixed(1)}%)`);
                }
            })
            .catch(error => {
                removeMessage(loadingId);
                addMessage('Error: Failed to analyze image', 'bot', 'error');
                console.error('Error:', error);
            });
            
            // Clear inputs
            fileInput.value = '';
            questionInput.value = '';
        }
        
        function addMessage(text, sender, className = '', imageData = null) {
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}-message ${className}`;
            
            // Add text content
            const textDiv = document.createElement('div');
            textDiv.textContent = text;
            messageDiv.appendChild(textDiv);
            
            // Add image if available
            if (imageData && imageData.path) {
                const imageDiv = document.createElement('div');
                imageDiv.className = 'message-image';
                
                const img = document.createElement('img');
                img.src = `/api/image/${imageData.path.replace(/^.*[\\\/]/, '')}`; // Extract filename from path
                img.alt = imageData.filename || 'Retrieved image';
                img.style.maxWidth = '100%';
                img.style.maxHeight = '300px';
                img.style.borderRadius = '5px';
                img.style.marginTop = '10px';
                
                // Add image metadata
                const metadataDiv = document.createElement('div');
                metadataDiv.className = 'image-metadata';
                metadataDiv.style.fontSize = '12px';
                metadataDiv.style.color = '#666';
                metadataDiv.style.marginTop = '5px';
                metadataDiv.textContent = `📸 ${imageData.character} - ${imageData.content} (Relevance: ${(imageData.relevance_score * 100).toFixed(0)}%)`;
                
                imageDiv.appendChild(img);
                imageDiv.appendChild(metadataDiv);
                messageDiv.appendChild(imageDiv);
            }
            
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
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        function resetConversation() {
            fetch('/api/reset', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: sessionId
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    updateStatus('Error: ' + data.error, 'error');
                } else {
                    // Clear chat messages
                    const messagesDiv = document.getElementById('chatMessages');
                    messagesDiv.innerHTML = '<div class="message bot-message">Hello! I\'m your One Piece expert. I can answer questions about characters, locations, events, and more. I can also analyze images related to One Piece!</div>';
                    
                    // Reset session
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
                    updateStatus(`Chatbot Status: Ready | Uptime: ${uptime} minutes | Total Queries: ${data.conversation_summary?.total_queries_processed || 0}`);
                }
            })
            .catch(error => {
                updateStatus('Error: Failed to get status', 'error');
                console.error('Error:', error);
            });
        }
        
        function updateStatus(message, className = '') {
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = message;
            statusDiv.className = 'status ' + className;
        }
    </script>
</body>
</html>
    """
    
    return html_template


def create_templates_directory():
    """Create the templates directory and HTML file."""
    import os
    
    # Create templates directory
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create HTML template file
    template_file = os.path.join(templates_dir, 'index.html')
    with open(template_file, 'w', encoding='utf-8') as f:
        f.write(create_html_template())
    
    return template_file


if __name__ == '__main__':
    # Create templates directory and HTML file
    create_templates_directory()
    
    # Create and run web interface
    web_interface = ChatbotWebInterface()
    
    try:
        web_interface.run()
    except KeyboardInterrupt:
        print("\\nShutting down web interface...")
    finally:
        web_interface.cleanup()
