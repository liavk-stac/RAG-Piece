#!/usr/bin/env python3
"""
Entry point for the One Piece Wiki RAG Database System.

This script provides a clean entry point that follows the cursor guidelines
by keeping executable code in the src/ directory structure.
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run main function
from rag_piece.main import main

if __name__ == "__main__":
    main()
