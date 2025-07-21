#!/usr/bin/env python3
"""
Groq Diagnostic Script - Identify the exact issue
"""
import os
import sys
import inspect
from dotenv import load_dotenv

load_dotenv()

print("=== Groq Diagnostic ===")

# Check if API key is available
api_key = os.getenv('GROQ_API_KEY')
print(f"1. API Key found: {api_key is not None}")
if api_key:
    print(f"   API Key starts with: {api_key[:10]}...")

# Check for conflicting libraries
print("\n2. Checking for potential conflicts:")
try:
    import openai
    print(f"   ⚠️  OpenAI library found: {openai.__version__}")
    print("   This might cause conflicts with Groq")
except ImportError:
    print("   ✅ No OpenAI library conflict")

# Check environment variables for proxy settings
print("\n3. Checking environment for proxy settings:")
proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY']
found_proxies = []
for var in proxy_vars:
    if os.getenv(var):
        found_proxies.append(f"{var}={os.getenv(var)}")

if found_proxies:
    print("   ⚠️  Proxy environment variables found:")
    for proxy in found_proxies:
        print(f"      {proxy}")
    print("   These might be causing the 'proxies' parameter issue")
else:
    print("   ✅ No proxy environment variables found")

# Check Groq import
print("\n4. Testing Groq imports:")
try:
    import groq
    print(f"   ✅ groq module imported successfully")
    print(f"   Version: {getattr(groq, '__version__', 'unknown')}")
    
    # Check Groq constructor signature
    try:
        sig = inspect.signature(groq.Groq.__init__)
        params = list(sig.parameters.keys())
        print(f"   Groq.__init__ parameters: {params}")
        
        if 'proxies' in params:
            print("   ✅ 'proxies' parameter is supported")
        else:
            print("   ❌ 'proxies' parameter is NOT supported")
            
    except Exception as e:
        print(f"   ⚠️  Could not inspect Groq constructor: {e}")
    
except ImportError as e:
    print(f"   ❌ groq import failed: {e}")
    sys.exit(1)

# Test minimal client initialization
print("\n5. Testing minimal client initialization:")

if not api_key:
    print("   ⚠️  Cannot test without API key")
else:
    try:
        # Try the most basic initialization possible
        print("   Trying basic Groq(api_key=...) initialization:")
        client = groq.Groq(api_key=api_key)
        print("   ✅ Basic initialization successful!")
        
        # Test API call
        try:
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1
            )
            print("   ✅ API call successful!")
        except Exception as e:
            print(f"   ⚠️  API call failed: {e}")
            
    except Exception as e:
        print(f"   ❌ Basic initialization failed: {e}")
        
        # Try to understand what's happening
        print(f"\n   Detailed error analysis:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        
        # Check if it's specifically about proxies
        if "proxies" in str(e):
            print("\n   🔍 This is definitely a 'proxies' parameter issue")
            print("   Possible causes:")
            print("   1. Another library is monkey-patching the Groq client")
            print("   2. Environment configuration is forcing proxy settings")
            print("   3. Version incompatibility")

print("\n=== Diagnostic Complete ===")
print("\nRecommended fixes:")
print("1. If OpenAI library found: pip uninstall openai")
print("2. If proxy vars found: unset them or restart terminal")
print("3. Try: pip uninstall groq && pip install groq==0.10.0")
print("4. Clear environment: unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy")
