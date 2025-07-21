#!/usr/bin/env python3
"""
Completely isolated Groq test to bypass initialization issues
"""
import os
import sys
import subprocess
import importlib

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def test_groq_in_subprocess():
    """Test Groq in a completely clean subprocess"""
    print("=== Testing Groq in Clean Subprocess ===")
    
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("❌ No API key found")
        return False
    
    # Create a minimal test script
    test_script = '''
import os
import sys

# Set the API key
api_key = "{}"

try:
    from groq import Groq
    
    # Try to create client with no additional parameters
    print("Creating Groq client...")
    client = Groq(api_key=api_key)
    print("SUCCESS: Client created")
    
    # Test API call
    print("Testing API call...")
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{{"role": "user", "content": "hi"}}],
        max_tokens=1
    )
    print("SUCCESS: API call worked")
    print("RESULT: Everything working!")
    
except Exception as e:
    print(f"ERROR: {{e}}")
    import traceback
    traceback.print_exc()
'''.format(api_key)
    
    # Write and run the test script
    with open('temp_groq_test.py', 'w') as f:
        f.write(test_script)
    
    try:
        result = subprocess.run([sys.executable, 'temp_groq_test.py'], 
                              capture_output=True, text=True, timeout=30)
        
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return "SUCCESS: API call worked" in result.stdout
        
    except Exception as e:
        print(f"Subprocess test failed: {e}")
        return False
    finally:
        # Clean up
        try:
            os.remove('temp_groq_test.py')
        except:
            pass

def test_manual_import():
    """Try manual import with different methods"""
    print("\n=== Testing Manual Import Methods ===")
    
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("❌ No API key found")
        return False
    
    # Method 1: Force reload
    try:
        print("Method 1: Force reload groq module")
        if 'groq' in sys.modules:
            importlib.reload(sys.modules['groq'])
        
        import groq
        client = groq.Groq(api_key=api_key)
        print("✅ Method 1 SUCCESS")
        return True
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
    
    # Method 2: Import with __import__
    try:
        print("Method 2: Using __import__")
        groq_module = __import__('groq')
        client = groq_module.Groq(api_key=api_key)
        print("✅ Method 2 SUCCESS")
        return True
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
    
    # Method 3: Try different initialization pattern
    try:
        print("Method 3: Alternative initialization")
        from groq import Groq as GroqClient
        
        # Create with minimal kwargs
        kwargs = {'api_key': api_key}
        client = GroqClient(**kwargs)
        print("✅ Method 3 SUCCESS")
        return True
    except Exception as e:
        print(f"❌ Method 3 failed: {e}")
    
    return False

def main():
    print("🔍 Advanced Groq Debugging")
    print("=" * 50)
    
    # Test in subprocess first
    subprocess_works = test_groq_in_subprocess()
    
    # Test manual imports
    manual_works = test_manual_import()
    
    print("\n" + "=" * 50)
    print("RESULTS:")
    print(f"Subprocess test: {'✅ WORKS' if subprocess_works else '❌ FAILS'}")
    print(f"Manual import test: {'✅ WORKS' if manual_works else '❌ FAILS'}")
    
    if subprocess_works:
        print("\n✅ Groq works in clean environment!")
        print("The issue is likely with module loading in your current environment.")
        print("Recommendation: Restart your Python environment/terminal")
    elif manual_works:
        print("\n✅ Found a working import method!")
    else:
        print("\n❌ All methods failed. This suggests:")
        print("1. Groq version incompatibility")
        print("2. System-level proxy configuration")
        print("3. Corrupted Python environment")
        print("\nTry:")
        print("- pip uninstall groq && pip install groq==0.9.0")
        print("- Check system proxy settings")
        print("- Try in a fresh virtual environment")

if __name__ == "__main__":
    main()
