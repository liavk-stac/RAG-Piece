#!/usr/bin/env python3
"""
Test dotenv loading to debug the environment variable issue.
"""

import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

print("🔍 Testing dotenv loading...")
print("=" * 50)

# Test 1: Check if .env file exists
print("1. Checking for .env file...")
if os.path.exists('.env'):
    print("   ✅ .env file found")
    with open('.env', 'r') as f:
        content = f.read()
        print(f"   📄 Content: {content[:50]}...")
else:
    print("   ❌ .env file not found")

# Test 2: Check current working directory
print("\n2. Current working directory:")
print(f"   📁 {os.getcwd()}")

# Test 3: Check if OPENAI_API_KEY is already set
print("\n3. Environment variables before dotenv:")
print(f"   🔑 OPENAI_API_KEY: {'✅ SET' if os.getenv('OPENAI_API_KEY') else '❌ NOT SET'}")

# Test 4: Try to load dotenv manually
print("\n4. Loading dotenv manually...")
try:
    from dotenv import load_dotenv
    print("   ✅ dotenv imported successfully")
    
    # Try to load .env file
    result = load_dotenv()
    print(f"   📥 load_dotenv() result: {result}")
    
    # Check if .env file was found
    if result:
        print("   ✅ .env file loaded successfully")
    else:
        print("   ⚠️  .env file not loaded (might not exist or be empty)")
        
except ImportError as e:
    print(f"   ❌ Failed to import dotenv: {e}")

# Test 5: Check environment variables after dotenv
print("\n5. Environment variables after dotenv:")
print(f"   🔑 OPENAI_API_KEY: {'✅ SET' if os.getenv('OPENAI_API_KEY') else '❌ NOT SET'}")

# Test 6: Try to import and instantiate ChatbotConfig
print("\n6. Testing ChatbotConfig import...")
try:
    from src.chatbot.config import ChatbotConfig
    print("   ✅ ChatbotConfig imported successfully")
    
    # Try to instantiate
    config = ChatbotConfig()
    print("   ✅ ChatbotConfig instantiated successfully")
    
except Exception as e:
    print(f"   ❌ Failed to import/instantiate ChatbotConfig: {e}")

print("\n" + "=" * 50)
print("�� Test completed!")
