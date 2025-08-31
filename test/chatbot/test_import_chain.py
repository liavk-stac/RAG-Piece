#!/usr/bin/env python3
"""
Test the import chain to see where the environment variable gets lost.
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('../..'))

print("🔍 Testing Import Chain...")
print("=" * 50)

# Test 1: Load dotenv first
print("1. Loading dotenv...")
from dotenv import load_dotenv
load_dotenv()
print(f"   🔑 OPENAI_API_KEY: {'✅ SET' if os.getenv('OPENAI_API_KEY') else '❌ NOT SET'}")

# Test 2: Import config
print("\n2. Importing ChatbotConfig...")
try:
    from src.chatbot.config import ChatbotConfig
    print("   ✅ ChatbotConfig imported successfully")
    print(f"   🔑 OPENAI_API_KEY after config import: {'✅ SET' if os.getenv('OPENAI_API_KEY') else '❌ NOT SET'}")
except Exception as e:
    print(f"   ❌ Failed to import ChatbotConfig: {e}")

# Test 3: Import orchestrator
print("\n3. Importing ChatbotOrchestrator...")
try:
    from src.chatbot.core.orchestrator import ChatbotOrchestrator
    print("   ✅ ChatbotOrchestrator imported successfully")
    print(f"   🔑 OPENAI_API_KEY after orchestrator import: {'✅ SET' if os.getenv('OPENAI_API_KEY') else '❌ NOT SET'}")
except Exception as e:
    print(f"   ❌ Failed to import ChatbotOrchestrator: {e}")

# Test 4: Import agents
print("\n4. Importing agents...")
try:
    from src.chatbot.agents import RouterAgent
    print("   ✅ RouterAgent imported successfully")
    print(f"   🔑 OPENAI_API_KEY after agent import: {'✅ SET' if os.getenv('OPENAI_API_KEY') else '❌ NOT SET'}")
except Exception as e:
    print(f"   ❌ Failed to import RouterAgent: {e}")

# Test 5: Import LLM client
print("\n5. Importing LLM client...")
try:
    from src.chatbot.utils.llm_client import LLMClient
    print("   ✅ LLMClient imported successfully")
    print(f"   🔑 OPENAI_API_KEY after LLM client import: {'✅ SET' if os.getenv('OPENAI_API_KEY') else '❌ NOT SET'}")
except Exception as e:
    print(f"   ❌ Failed to import LLMClient: {e}")

# Test 6: Try to create LLM client
print("\n6. Testing LLM client creation...")
try:
    from src.chatbot.utils.llm_client import LLMClient
    client = LLMClient()
    print("   ✅ LLMClient created successfully")
except Exception as e:
    print(f"   ❌ Failed to create LLMClient: {e}")

print("\n" + "=" * 50)
print("🏁 Import chain test completed!")
