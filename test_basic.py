#!/usr/bin/env python3
"""
Simple test script to verify core functionality
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')

def test_basic_setup():
    """Test basic setup and imports"""
    print("🧪 Testing Basic Setup...")
    
    # Test environment
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        print(f"✅ Gemini API Key loaded: {api_key[:10]}...")
    else:
        print("❌ Gemini API Key not found")
        return False
    
    # Test basic imports
    try:
        import google.generativeai as genai
        print("✅ Google Generative AI imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Google Generative AI: {e}")
        return False
    
    try:
        import langchain
        import langgraph
        print("✅ LangChain and LangGraph imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import LangChain/LangGraph: {e}")
        return False
    
    return True

def test_gemini_api():
    """Test Gemini API connectivity"""
    print("\n🔗 Testing Gemini API...")
    
    try:
        import google.generativeai as genai
        
        # Configure API
        api_key = os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=api_key)
        
        # Test simple generation
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        response = model.generate_content("Hello! Please respond with 'Gemini API is working!'")
        
        print(f"✅ Gemini Response: {response.text}")
        return True
        
    except Exception as e:
        print(f"❌ Gemini API test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Running Basic Tests for Multimodal AI Agent\n")
    
    # Run tests
    tests = [
        ("Basic Setup", test_basic_setup),
        ("Gemini API", test_gemini_api),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
        print()
    
    print(f"📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! Your setup is ready!")
        print("\n🔧 Next steps:")
        print("   1. Run: python test_advanced.py (for advanced features)")
        print("   2. Run: python main.py (for full demo)")
    else:
        print("\n⚠️  Some tests failed. Please check the setup.")

if __name__ == "__main__":
    main() 