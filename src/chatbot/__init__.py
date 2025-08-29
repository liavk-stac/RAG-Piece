"""
One Piece Chatbot Package

A comprehensive, multimodal One Piece chatbot that leverages the existing RAG database
to provide intelligent, context-aware responses to both text and image queries.

This package implements an advanced agent pipeline using LangChain for intelligent
conversation management and One Piece knowledge integration.
"""

from .core.chatbot import OnePieceChatbot
from .core.orchestrator import ChatbotOrchestrator
from .config import ChatbotConfig

__version__ = "1.0.0"
__author__ = "One Piece RAG Project"
__description__ = "Intelligent One Piece chatbot with multimodal capabilities"

# Main exports
__all__ = [
    "OnePieceChatbot",
    "ChatbotOrchestrator", 
    "ChatbotConfig",
]
