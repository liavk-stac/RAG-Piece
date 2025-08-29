# 🏴‍☠️ One Piece Chatbot

An intelligent, multimodal chatbot system that leverages your existing One Piece RAG database to provide comprehensive answers about the One Piece universe. Built with advanced agent architecture using LangChain.

## 🌟 Features

### **🤖 Intelligent Agent Pipeline**
- **Router Agent**: Analyzes queries and determines optimal processing path
- **Search Agent**: Integrates with your existing RAG database for information retrieval
- **Image Analysis Agent**: Processes and analyzes One Piece images
- **Reasoning Agent**: Performs logical analysis and information synthesis
- **Timeline Agent**: Handles chronological and historical queries
- **Response Agent**: Generates coherent, formatted responses

### **📱 Multiple Interfaces**
- **Web Interface**: Beautiful, responsive web UI with real-time chat
- **Command Line Interface**: Full-featured CLI for power users
- **Demo Mode**: Automated demonstration of capabilities

### **🖼️ Multimodal Capabilities**
- **Text Queries**: Ask questions about characters, locations, events, and lore
- **Image Analysis**: Upload One Piece images for detailed analysis
- **Context Awareness**: Maintains conversation history and context

### **🔍 Advanced Search Integration**
- **RAG Database**: Seamlessly integrates with your existing One Piece knowledge base
- **Hybrid Search**: Combines semantic and keyword search for optimal results
- **One Piece Specialization**: Enhanced understanding of One Piece terminology and concepts

## 🚀 Quick Start

### **Prerequisites**
- Python 3.8+
- Your existing One Piece RAG database
- OpenAI API key (for advanced features)

### **Installation**

1. **Clone and navigate to the chatbot directory:**
   ```bash
   cd src/chatbot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables (optional):**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   export LOG_LEVEL="INFO"
   ```

### **Running the Chatbot**

#### **🌐 Web Interface (Recommended)**
```bash
python -m src.chatbot.main web
```
Open your browser to `http://localhost:5000`

#### **💻 Command Line Interface**
```bash
python -m src.chatbot.main cli
```

#### **🎭 Demo Mode**
```bash
python -m src.chatbot.main demo
```

#### **🔧 Custom Configuration**
```bash
# Run on different port
python -m src.chatbot.main web --port 9000

# Enable debug mode
python -m src.chatbot.main web --debug

# Custom host
python -m src.chatbot.main web --host 0.0.0.0
```

## 🏗️ Architecture

### **System Components**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │  CLI Interface  │    │  Demo Mode      │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │   OnePieceChatbot        │
                    │   (Main Interface)       │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │  ChatbotOrchestrator     │
                    │  (Agent Pipeline)        │
                    └─────────────┬─────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
┌─────────▼─────────┐ ┌───────────▼──────────┐ ┌─────────▼─────────┐
│   Router Agent    │ │   Search Agent      │ │ Reasoning Agent   │
│   (Query Routing) │ │   (RAG Integration) │ │   (Logic)         │
└───────────────────┘ └──────────────────────┘ └───────────────────┘
          │                       │                       │
┌─────────▼─────────┐ ┌───────────▼──────────┐ ┌─────────▼─────────┐
│Image Analysis     │ │  Timeline Agent     │ │ Response Agent    │
│   Agent           │ │   (Chronology)      │ │   (Output)        │
└───────────────────┘ └──────────────────────┘ └───────────────────┘
```

### **Agent Pipeline Flow**
1. **Input Processing**: User query/image is received
2. **Routing**: Router agent determines optimal processing path
3. **Information Retrieval**: Search agent queries RAG database
4. **Analysis**: Specialized agents process the information
5. **Synthesis**: Response agent generates final answer
6. **Output**: Formatted response is delivered to user

## 🎯 Usage Examples

### **Text Queries**
```
🤔 You: Who is Monkey D. Luffy?
🤖 Bot: Monkey D. Luffy is the main protagonist of One Piece...

🤔 You: What is a Devil Fruit?
🤖 Bot: Devil Fruits are mysterious fruits that grant supernatural powers...

🤔 You: Tell me about the relationship between Ace and Luffy
🤖 Bot: Portgas D. Ace and Monkey D. Luffy are sworn brothers...
```

### **Image Analysis**
```
📸 Upload: One Piece character image
🤔 Question: Who is this character and what are their abilities?
🤖 Bot: This appears to be [Character Name] from One Piece...
```

### **Timeline Queries**
```
🤔 You: When did the Great Pirate Era begin?
🤖 Bot: The Great Pirate Era began 22 years ago when Gol D. Roger...
```

## ⚙️ Configuration

### **Key Configuration Options**
```python
from src.chatbot import ChatbotConfig

config = ChatbotConfig()

# Agent settings
config.AGENT_TIMEOUT = 120                    # Agent execution timeout
config.AGENT_MAX_RETRIES = 3                  # Maximum retry attempts

# RAG integration
config.RAG_SEARCH_LIMIT = 100                 # Search result limit
config.ENABLE_CACHING = True                  # Enable response caching

# Image processing
config.MAX_IMAGE_SIZE = 10 * 1024 * 1024     # 10MB max image size
config.IMAGE_QUALITY_THRESHOLD = 100          # Minimum image resolution

# Web interface
config.WEB_HOST = "127.0.0.1"                # Web server host
config.WEB_PORT = 5000                       # Web server port
config.WEB_DEBUG = False                      # Debug mode

# Logging
config.LOG_LEVEL = "INFO"                     # Logging level
config.LOG_TO_FILE = True                     # Save logs to file
config.LOG_FILE_PATH = "logs/chatbot.log"     # Log file path
```

## 🔧 Development

### **Project Structure**
```
src/chatbot/
├── __init__.py              # Package initialization
├── main.py                  # Main entry point
├── config.py                # Configuration management
├── requirements.txt          # Dependencies
├── README.md                # This file
├── core/                    # Core chatbot components
│   ├── chatbot.py          # Main chatbot class
│   └── orchestrator.py     # Agent pipeline orchestrator
├── agents/                  # Agent implementations
│   ├── base_agent.py       # Base agent interface
│   ├── router_agent.py     # Query routing agent
│   ├── search_agent.py     # RAG search agent
│   ├── reasoning_agent.py  # Logical reasoning agent
│   ├── image_analysis_agent.py # Image processing agent
│   ├── timeline_agent.py   # Chronology agent
│   └── response_agent.py   # Response generation agent
├── interfaces/              # User interfaces
│   └── web_interface.py    # Flask web interface
├── memory/                  # Conversation memory (future)
├── multimodal/              # Multimodal processing (future)
├── pipeline/                # Pipeline components (future)
├── tools/                   # Specialized tools (future)
├── utils/                   # Utility functions (future)
└── tests/                   # Test suite (future)
```

### **Adding New Agents**
1. Create a new agent class inheriting from `BaseAgent`
2. Implement the `_execute_agent` method
3. Add the agent to the orchestrator's agent initialization
4. Update the routing logic in the router agent

### **Testing**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/chatbot

# Run specific test file
pytest tests/test_agents.py
```

## 🚧 Current Status & Roadmap

### **✅ Implemented**
- Complete agent pipeline architecture
- Router, Search, Reasoning, Image Analysis, Timeline, and Response agents
- Web interface with Flask
- Command-line interface
- Configuration management
- Logging and error handling
- Basic image processing capabilities

### **🔄 In Progress**
- Advanced image analysis with vision models
- Enhanced RAG integration
- Performance optimization

### **📋 Planned Features**
- **Memory Management**: Persistent conversation memory
- **Advanced Vision**: Integration with state-of-the-art vision models
- **Multi-session Support**: Handle multiple concurrent users
- **API Endpoints**: RESTful API for external integration
- **Advanced Analytics**: Detailed usage analytics and insights
- **Plugin System**: Extensible architecture for custom tools

## 🐛 Troubleshooting

### **Common Issues**

#### **RAG Database Connection Failed**
```
Error: Could not import RAG components
```
**Solution**: Ensure your RAG database is properly set up and accessible

#### **Image Analysis Not Working**
```
Error: Invalid image data
```
**Solution**: Check image format (JPEG, PNG supported) and file size limits

#### **Web Interface Won't Start**
```
Error: Port already in use
```
**Solution**: Use a different port with `--port 9000`

#### **Low Confidence Responses**
**Solution**: 
- Check RAG database content quality
- Verify search parameters in configuration
- Ensure proper One Piece terminology in queries

### **Debug Mode**
Enable debug mode for detailed logging:
```bash
python -m src.chatbot.main web --debug
```

### **Log Files**
Check log files for detailed error information:
```bash
tail -f logs/chatbot.log
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### **Development Guidelines**
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is part of your One Piece RAG system. Please ensure compliance with any applicable licenses for the underlying technologies and data sources.

## 🙏 Acknowledgments

- **One Piece Community**: For the rich lore and content
- **LangChain**: For the excellent agent framework
- **OpenAI**: For powerful language models
- **Flask**: For the web framework

## 📞 Support

For questions, issues, or contributions:
- Check the troubleshooting section above
- Review the code and documentation
- Open an issue on the repository

---

**🏴‍☠️ Set sail with the One Piece Chatbot and explore the vast world of One Piece knowledge!**
