#!/usr/bin/env python3
"""
Quick Setup for Multimodal AI Agent

This script quickly sets up the environment with the provided Gemini API key.

Author: Anuj Patel (amp10162@nyu.edu)
Website: panuj.com
"""
import shutil
from pathlib import Path


def main():
    """Quick setup with Gemini API key"""
    print("🚀 Quick Setup - Multimodal AI Agent")
    print("=" * 50)
    print("Created by Anuj Patel (amp10162@nyu.edu)")
    print("Website: panuj.com")
    print("=" * 50)
    
    # Copy template to .env
    template_file = Path("env.template")
    env_file = Path(".env")
    
    if not template_file.exists():
        print("❌ env.template file not found")
        return False
    
    try:
        # Read template
        with open(template_file, 'r') as f:
            content = f.read()
        
        # Replace Gemini API key
        content = content.replace(
            "GEMINI_API_KEY=your_gemini_api_key_here",
            "GEMINI_API_KEY=AIzaSyCtCBbm9TePgnMYJW0byx_gQ7QMBFDyOvI"
        )
        
        # Write to .env
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✅ Created .env file with Gemini API key configured")
        print("🔑 Gemini API key: AIzaSyCtCBbm9TePgnMYJW0byx_gQ7QMBFDyOvI")
        print("\n📝 Next steps:")
        print("1. Add your OpenAI API key to .env file (optional)")
        print("2. Run: python main.py")
        print("3. Or run full setup: python setup.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    main() 