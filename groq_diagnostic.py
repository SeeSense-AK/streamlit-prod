#!/usr/bin/env python3
"""
Groq Diagnostic Script - Identify the exact issue
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

print("=== Groq Diagnostic ===")

# Check if API key is available
api_key = os.getenv('GROQ_API_KEY')
print(f"1. API Key found: {api_key is not None}")
if api_key:
    print(f"   API Key starts with: {api_key[:10]}...")

# Check Groq import
print("\n2. Testing Groq imports:")
try:
    import groq
    print(f"   ✅ groq module imported successfully")
    print(f"   Version: {getattr(groq, '__version__', 'unknown')}")
    
    # List available classes/functions
    groq_items = [item for item in dir(groq) if not item.startswith('_')]
    print(f"   Available items: {groq_items}")
    
except ImportError as e:
    print(f"   ❌ groq import failed: {e}")
    sys.exit(1)

# Test different client initialization methods
print("\n3. Testing client initialization methods:")

methods = [
    ("groq.Groq()", lambda: groq.Groq(api_key=api_key)),
    ("groq.Client()", lambda: groq.Client(api_key=api_key)),
    ("Groq from import", lambda: __import__('groq').Groq(api_key=api_key)),
]

for method_name, method_func in methods:
    try:
        print(f"   Testing {method_name}...")
        if api_key:
            client = method_func()
            print(f"   ✅ {method_name} - SUCCESS")
            
            # Test a simple API call
            try:
                response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=1
                )
                print(f"   ✅ {method_name} - API call successful")
                break
            except Exception as e:
                print(f"   ⚠️  {method_name} - API call failed: {e}")
                
        else:
            print(f"   ⚠️  {method_name} - No API key to test")
            
    except Exception as e:
        print(f"   ❌ {method_name} - FAILED: {e}")

print("\n=== Diagnostic Complete ===")
print("\nIf all methods failed, try:")
print("1. pip uninstall groq")
print("2. pip install groq==0.8.0")
print("3. Restart your environment")
