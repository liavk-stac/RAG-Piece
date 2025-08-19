"""
RAG Piece - One Piece Wiki RAG Database System

A comprehensive Retrieval-Augmented Generation (RAG) system that scrapes
One Piece Wiki content and creates an intelligent, searchable database.
"""

__version__ = "1.0.0"
__author__ = "RAG-Piece Team"

from .database import RAGDatabase
from .config import RAGConfig
from .scraper import OneWikiScraper
from .main import main

__all__ = ["RAGDatabase", "RAGConfig", "OneWikiScraper", "main"]
