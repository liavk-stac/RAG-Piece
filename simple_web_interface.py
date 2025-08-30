#!/usr/bin/env python3
"""
Simple Web Interface for One Piece Chatbot

This is a clean, simple implementation with image upload and display capabilities.
"""

import os
import sys
import base64
import logging
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

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

# Configure file upload settings
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_simple_html_template():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>One Piece Chatbot - Image Enabled</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f5f5f5; }
        .chat-container { background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .chat-header { text-align: center; margin-bottom: 20px; color: #333; }
        .chat-messages { height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 5px; background-color: #fafafa; }
        .message { margin-bottom: 15px; padding: 10px; border-radius: 5px; }
        .user-message { background-color: #007bff; color: white; margin-left: 20%; text-align: right; }
        .bot-message { background-color: #e9ecef; color: #333; margin-right: 20%; }
        .input-container { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
        .text-input { flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 16px; min-width: 200px; }
        .send-button { padding: 10px 20px; background-color: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .send-button:hover { background-color: #0056b3; }
        .status { text-align: center; margin-bottom: 20px; padding: 10px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; color: #155724; }
        .error { background-color: #f8d7da; border-color: #f5c6cb; color: #721c24; }
        .loading { text-align: center; color: #666; font-style: italic; }
        .image-upload { display: flex; align-items: center; gap: 10px; margin-bottom: 15px; }
        .file-input { padding: 8px; border: 1px solid #ddd; border-radius: 5px; }
        .image-preview { max-width: 100px; max-height: 100px; border-radius: 5px; border: 2px solid #ddd; }
        .remove-image { padding: 5px 10px; background-color: #dc3545; color: white; border: none; border-radius: 3px; cursor: pointer; font-size: 12px; }
        .remove-image:hover { background-color: #c82333; }
        .message-image { margin-top: 10px; text-align: center; }
        .message-image img { max-width: 100%; max-height: 300px; border-radius: 5px; border: 1px solid #ddd; }
        .image-metadata { font-size: 12px; color: #666; margin-top: 5px; font-style: italic; }
        .upload-status { font-size: 12px; color: #666; margin-left: 10px; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>🏴‍☠️ One Piece Chatbot - Image Enabled</h1>
            <p>Ask me anything about One Piece or upload an image for analysis!</p>
        </div>

        <div id="status" class="status">
            Chatbot is ready! Ask me a question about One Piece or upload an image.
        </div>

        <div class="chat-messages" id="chatMessages">
            <div class="message bot-message">
                Hello! I am your One Piece expert. I can answer questions about characters, locations, events, and analyze images too!
            </div>
        </div>

        <div class="image-upload">
            <input type="file" id="imageInput" class="file-input" accept="image/*" onchange="handleImageSelect(event)">
            <img id="imagePreview" class="image-preview" style="display: none;">
            <button id="removeImageBtn" class="remove-image" style="display: none;" onclick="removeImage()">Remove</button>
            <span id="uploadStatus" class="upload-status"></span>
        </div>

        <div class="input-container">
            <input type="text" id="messageInput" class="text-input" placeholder="Ask me about One Piece or describe what you want to know about the image..." onkeypress="handleKeyPress(event)">
            <button onclick="sendMessage()" class="send-button">Send</button>
        </div>

        <div style="text-align: center;">
            <button onclick="resetConversation()" style="margin-right: 10px; padding: 8px 16px; background-color: #6c757d; color: white; border: none; border-radius: 5px; cursor: pointer;">Reset Chat</button>
            <button onclick="getStatus()" style="padding: 8px 16px; background-color: #17a2b8; color: white; border: none; border-radius: 5px; cursor: pointer;">Status</button>
        </div>
    </div>

    <script>
        let sessionId = null;
        let selectedImage = null;

        function handleImageSelect(event) {
            const file = event.target.files[0];
            if (file) {
                if (file.size > 16 * 1024 * 1024) {
                    alert('Image file is too large. Please select an image smaller than 16MB.');
                    event.target.value = '';
                    return;
                }

                const reader = new FileReader();
                reader.onload = function(e) {
                    selectedImage = {
                        file: file,
                        data: e.target.result,
                        name: file.name
                    };
                    
                    document.getElementById('imagePreview').src = e.target.result;
                    document.getElementById('imagePreview').style.display = 'block';
                    document.getElementById('removeImageBtn').style.display = 'block';
                    document.getElementById('uploadStatus').textContent = 'Image selected: ' + file.name;
                };
                reader.readAsDataURL(file);
            }
        }

        function removeImage() {
            selectedImage = null;
            document.getElementById('imageInput').value = '';
            document.getElementById('imagePreview').style.display = 'none';
            document.getElementById('removeImageBtn').style.display = 'none';
            document.getElementById('uploadStatus').textContent = '';
        }

        function addMessage(text, sender, className, imageData) {
            console.log('addMessage called with:', {text, sender, className, imageData});
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + sender + '-message ' + (className || '');
            
            // Add text content
            const textDiv = document.createElement('div');
            textDiv.textContent = text;
            messageDiv.appendChild(textDiv);

            // Add image if available
            if (imageData) {
                console.log('Processing image data:', imageData);
                const imageDiv = document.createElement('div');
                imageDiv.className = 'message-image';

                const img = document.createElement('img');
                if (imageData.data) {
                    // User uploaded image
                    img.src = imageData.data;
                    img.alt = 'Uploaded image';
                } else if (imageData.path) {
                    // Retrieved image from backend
                    // Use the full path if available, otherwise construct from folder and filename
                    let imagePath = imageData.path;
                    if (!imagePath.includes('/') && imageData.metadata && imageData.metadata.folder) {
                        // Construct path from folder and filename if path is just filename
                        imagePath = imageData.metadata.folder + '/' + imageData.filename;
                    }
                    img.src = '/api/image/' + imagePath;
                    img.alt = imageData.filename || 'Retrieved image';
                }

                img.style.maxWidth = '100%';
                img.style.maxHeight = '300px';
                img.style.borderRadius = '5px';
                img.style.marginTop = '10px';

                // Add image metadata if available
                if (imageData.character || imageData.content || imageData.relevance_score !== undefined) {
                    const metadataDiv = document.createElement('div');
                    metadataDiv.className = 'image-metadata';
                    
                    let metadataText = '📸 ';
                    if (imageData.character) metadataText += imageData.character + ' - ';
                    if (imageData.content) metadataText += imageData.content;
                    if (imageData.relevance_score !== undefined) {
                        metadataText += ' (Relevance: ' + (imageData.relevance_score * 100).toFixed(0) + '%)';
                    }
                    metadataDiv.textContent = metadataText;
                    imageDiv.appendChild(metadataDiv);
                }

                imageDiv.appendChild(img);
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

        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();

            if (!message && !selectedImage) {
                alert('Please enter a message or select an image to analyze.');
                return;
            }

            // Add user message with image if available
            addMessage(message || 'Analyzing image...', 'user', '', selectedImage);
            input.value = '';

            const loadingId = addMessage('Processing...', 'bot', 'loading');

            // Prepare form data
            const formData = new FormData();
            formData.append('message', message || 'Analyze this image');
            if (sessionId) formData.append('session_id', sessionId);
            if (selectedImage) {
                formData.append('image', selectedImage.file);
            }

            fetch('/api/chat', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                removeMessage(loadingId);
                if (data.error) {
                    addMessage('Error: ' + data.error, 'bot', 'error');
                    console.error('Backend error:', data.error);
                } else {
                    if (data.session_id) {
                        sessionId = data.session_id;
                    }
                    
                    // Add bot response with retrieved image if available
                    console.log('Bot response data:', data);
                    console.log('Image data from response:', data.image);
                    addMessage(data.response, 'bot', '', data.image);
                    updateStatus('Response received');
                    
                    // Clear selected image from upload interface only (not from chat)
                    if (selectedImage) {
                        // Reset upload interface without affecting chat display
                        document.getElementById('imageInput').value = '';
                        document.getElementById('imagePreview').style.display = 'none';
                        document.getElementById('removeImageBtn').style.display = 'none';
                        document.getElementById('uploadStatus').textContent = '';
                        selectedImage = null;
                    }
                }
            })
            .catch(error => {
                removeMessage(loadingId);
                const errorMsg = 'Error: Failed to send message. Please try again.';
                addMessage(errorMsg, 'bot', 'error');
                console.error('Network error:', error);
                
                // Log error for debugging
                if (selectedImage) {
                    console.error('Image processing failed for:', selectedImage.name);
                }
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
                    messagesDiv.innerHTML = '<div class="message bot-message">Hello! I am your One Piece expert. I can answer questions about characters, locations, events, and analyze images too!</div>';
                    sessionId = null;
                    removeImage();
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

    try:
        message = request.form.get('message', '')
        session_id = request.form.get('session_id')
        
        # Handle image upload
        image_file = None
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file and image_file.filename:
                if not allowed_file(image_file.filename):
                    return jsonify({'error': 'Invalid file type. Please upload an image file (png, jpg, jpeg, gif, bmp, webp).'}), 400
                
                if image_file.content_length and image_file.content_length > 16 * 1024 * 1024:
                    return jsonify({'error': 'Image file is too large. Please upload an image smaller than 16MB.'}), 400
                
                # Convert image to base64 for processing
                try:
                    image_data = image_file.read()
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                    logger.info(f"Image uploaded: {image_file.filename}, size: {len(image_data)} bytes")
                except Exception as e:
                    logger.error(f"Error processing uploaded image: {e}")
                    return jsonify({'error': 'Failed to process uploaded image. Please try again.'}), 500
            else:
                image_file = None

        # Process the query
        if image_file:
            # Image + text query
            try:
                # Use the orchestrator's process_query method directly
                response = chatbot.orchestrator.process_query(
                    query=message or 'Analyze this image',
                    image_data=image_data,  # Pass the raw bytes
                    session_id=session_id
                )
                logger.info(f"Image query processed successfully: {image_file.filename}")
            except Exception as e:
                error_msg = f"Image processing failed: {str(e)}"
                logger.error(f"Image processing error: {e}")
                return jsonify({'error': error_msg}), 500
        else:
            # Text-only query
            try:
                response = chatbot.ask(message, session_id)
                logger.info("Text query processed successfully")
            except Exception as e:
                error_msg = f"Text processing failed: {str(e)}"
                logger.error(f"Text processing error: {e}")
                return jsonify({'error': error_msg}), 500

        # Debug: Log the response structure
        logger.info(f"Response structure: {list(response.keys())}")
        if 'image' in response:
            logger.info(f"Image data present: {response['image'] is not None}")
            if response['image']:
                logger.info(f"Image path: {response['image'].get('path', 'None')}")
        else:
            logger.info("No image key in response")
            
        return jsonify(response)

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"Unexpected error in chat endpoint: {e}")
        return jsonify({'error': error_msg}), 500

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
    logger.info("🚀 Starting Image-Enabled One Piece Chatbot Web Interface...")
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
