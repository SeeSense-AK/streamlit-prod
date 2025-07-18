#!/usr/bin/env python3
"""
Comprehensive diagnostic script for Groq AI insights issue
Run this to identify exactly why the AI insights stopped working
"""
import os
import sys
import logging
from pathlib import Path

# Setup logging to see all details
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir
sys.path.insert(0, str(project_root))

def check_groq_import():
    """Check if Groq can be imported"""
    print("🔍 Checking Groq import...")
    try:
        from groq import Groq
        print("✅ Groq library imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Groq import failed: {e}")
        print("🔧 Fix: pip install groq>=0.4.1")
        return False

def check_streamlit_import():
    """Check if Streamlit can be imported"""
    print("\n🔍 Checking Streamlit import...")
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        print("🔧 Fix: pip install streamlit")
        return False

def check_secrets_access():
    """Check different ways to access secrets"""
    print("\n🔍 Checking secrets access methods...")
    
    # Method 1: Environment variable
    env_key = os.getenv('GROQ_API_KEY')
    if env_key:
        print(f"✅ Found in environment: GROQ_API_KEY = {env_key[:10]}...")
    else:
        print("❌ Not found in environment variables")
    
    # Method 2: Streamlit secrets (outside streamlit context)
    try:
        import streamlit as st
        
        # Try accessing secrets in different ways
        try:
            # Method 2a: Direct access
            streamlit_key = st.secrets["GROQ_API_KEY"]
            print(f"✅ Found in Streamlit secrets (direct): {streamlit_key[:10]}...")
        except Exception as e:
            print(f"❌ Direct secrets access failed: {e}")
            
        try:
            # Method 2b: Get method
            streamlit_key = st.secrets.get("GROQ_API_KEY")
            if streamlit_key:
                print(f"✅ Found in Streamlit secrets (get): {streamlit_key[:10]}...")
            else:
                print("❌ secrets.get() returned None")
        except Exception as e:
            print(f"❌ secrets.get() failed: {e}")
            
    except Exception as e:
        print(f"❌ Streamlit secrets check failed: {e}")
    
    # Method 3: Check .streamlit/secrets.toml file directly
    secrets_file = project_root / ".streamlit" / "secrets.toml"
    if secrets_file.exists():
        print(f"✅ Found secrets file: {secrets_file}")
        try:
            with open(secrets_file, 'r') as f:
                content = f.read()
                if 'GROQ_API_KEY' in content:
                    print("✅ GROQ_API_KEY found in secrets.toml")
                else:
                    print("❌ GROQ_API_KEY not found in secrets.toml")
        except Exception as e:
            print(f"❌ Error reading secrets file: {e}")
    else:
        print(f"❌ Secrets file not found: {secrets_file}")
    
    # Method 4: Check .env file
    env_file = project_root / ".env"
    if env_file.exists():
        print(f"✅ Found .env file: {env_file}")
        try:
            with open(env_file, 'r') as f:
                content = f.read()
                if 'GROQ_API_KEY' in content:
                    print("✅ GROQ_API_KEY found in .env file")
                else:
                    print("❌ GROQ_API_KEY not found in .env file")
        except Exception as e:
            print(f"❌ Error reading .env file: {e}")
    else:
        print(f"❌ .env file not found: {env_file}")

def test_groq_insights_generator():
    """Test the actual insights generator"""
    print("\n🔍 Testing GroqInsightsGenerator class...")
    
    try:
        # Import the insights generator
        from app.core.groq_insights_generator import create_insights_generator, GROQ_AVAILABLE
        
        print(f"✅ Insights generator imported successfully")
        print(f"📊 GROQ_AVAILABLE = {GROQ_AVAILABLE}")
        
        # Create insights generator instance
        generator = create_insights_generator()
        
        print(f"✅ Generator created: {type(generator)}")
        print(f"🔑 API Key available: {generator.api_key is not None}")
        print(f"🤖 Client initialized: {generator.client is not None}")
        
        if generator.api_key:
            print(f"🔑 API Key (partial): {generator.api_key[:10]}...")
        
        # Test with sample metrics
        sample_metrics = {
            'safety_score': 7.5,
            'total_routes': 100,
            'avg_daily_rides': 1500,
            'infrastructure_coverage': 65.0
        }
        
        print("\n🧪 Testing insights generation...")
        insights = generator.generate_comprehensive_insights(sample_metrics)
        
        print(f"✅ Generated {len(insights)} insights")
        for i, insight in enumerate(insights[:3]):
            print(f"   {i+1}. {insight.title} ({insight.impact_level} impact)")
        
        # Test executive summary
        print("\n📝 Testing executive summary...")
        summary = generator.generate_executive_summary(insights, sample_metrics)
        print(f"✅ Summary generated: {len(summary)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Insights generator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_streamlit_context():
    """Check if we're running in Streamlit context"""
    print("\n🔍 Checking Streamlit context...")
    
    try:
        import streamlit as st
        
        # Check if we're in Streamlit context
        try:
            # This will work only in Streamlit context
            st.session_state
            print("✅ Running in Streamlit context")
            return True
        except Exception:
            print("❌ Not running in Streamlit context (this is expected for this script)")
            return False
            
    except Exception as e:
        print(f"❌ Streamlit context check failed: {e}")
        return False

def main():
    """Run comprehensive diagnostic"""
    print("🚀 SeeSense Dashboard - Groq AI Diagnostic")
    print("=" * 60)
    
    # Step 1: Check imports
    groq_ok = check_groq_import()
    streamlit_ok = check_streamlit_import()
    
    if not groq_ok:
        print("\n❌ Cannot proceed without Groq library")
        return False
    
    # Step 2: Check secrets access
    check_secrets_access()
    
    # Step 3: Check Streamlit context
    check_streamlit_context()
    
    # Step 4: Test insights generator
    generator_ok = test_groq_insights_generator()
    
    print("\n" + "=" * 60)
    print("📋 DIAGNOSTIC SUMMARY:")
    print(f"   Groq Library: {'✅' if groq_ok else '❌'}")
    print(f"   Streamlit Library: {'✅' if streamlit_ok else '❌'}")
    print(f"   Insights Generator: {'✅' if generator_ok else '❌'}")
    
    if generator_ok:
        print("\n🎉 AI insights should be working!")
        print("▶️  Try running: streamlit run app/main.py")
    else:
        print("\n❌ Issues found that need fixing")
        print("\n🔧 RECOMMENDED FIXES:")
        print("1. Ensure Groq API key is properly set in .streamlit/secrets.toml")
        print("2. Restart your Streamlit application")
        print("3. Clear browser cache and refresh")
        print("4. Check the logs when running the dashboard")
    
    return generator_ok

if __name__ == "__main__":
    main()
