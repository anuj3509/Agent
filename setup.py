#!/usr/bin/env python3
"""
Setup script for the Multimodal AI Agent

This script automates the installation and configuration process for the
multimodal AI agent, making it easy to get started.

Author: Anuj Patel (amp10162@nyu.edu)
Website: panuj.com
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional


def print_banner():
    """Print the setup banner"""
    print("""
🤖 Multimodal AI Agent Setup
═══════════════════════════════════════════════════════════════
Created by Anuj Patel (amp10162@nyu.edu) • Website: panuj.com

Setting up your multimodal AI agent for content creation and analysis...
    """)


def check_python_version():
    """Check if Python version is supported"""
    print("🔍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    
    print(f"✅ Python {sys.version.split()[0]} is supported")
    return True


def check_dependencies():
    """Check for required system dependencies"""
    print("\n🔍 Checking system dependencies...")
    
    dependencies = {
        "ffmpeg": "Required for video processing",
        "git": "Required for version control"
    }
    
    missing = []
    for dep, description in dependencies.items():
        if shutil.which(dep) is None:
            print(f"⚠️  {dep} not found - {description}")
            missing.append(dep)
        else:
            print(f"✅ {dep} found")
    
    if missing:
        print(f"\n💡 Please install missing dependencies: {', '.join(missing)}")
        print("   macOS: brew install ffmpeg git")
        print("   Ubuntu: sudo apt-get install ffmpeg git")
        print("   Windows: Download from official websites")
        return False
    
    return True


def setup_environment():
    """Set up the environment file"""
    print("\n🔧 Setting up environment configuration...")
    
    env_file = Path(".env")
    template_file = Path("env.template")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if not template_file.exists():
        print("❌ env.template file not found")
        return False
    
    try:
        # Copy template to .env
        shutil.copy(template_file, env_file)
        print("✅ Created .env file from template")
        
        # Prompt for API keys
        print("\n🔑 API Key Configuration:")
        print("Please edit the .env file with your API keys:")
        print(f"   nano {env_file}")
        print("\nRequired API keys:")
        print("   - OPENAI_API_KEY: For GPT-4 and vision models")
        print("   - GEMINI_API_KEY: For Google Gemini models (already set)")
        print("\nOptional API keys:")
        print("   - HUGGINGFACE_API_TOKEN: For Hugging Face models")
        print("   - ANTHROPIC_API_KEY: For Claude models")
        
        return True
        
    except Exception as e:
        print(f"❌ Error setting up environment: {e}")
        return False


def install_python_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    try:
        # Check if we're in a virtual environment
        in_venv = (hasattr(sys, 'real_prefix') or 
                  (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
        
        if not in_venv:
            print("💡 Recommendation: Use a virtual environment")
            print("   python -m venv venv")
            print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
            print("   python setup.py")
            
            response = input("\nContinue without virtual environment? (y/N): ")
            if response.lower() != 'y':
                return False
        
        # Install dependencies
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        
        print("✅ Python dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating project directories...")
    
    directories = [
        "data/content",
        "data/output", 
        "data/cache",
        "data/logs",
        "plugins"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True


def test_installation():
    """Test the installation"""
    print("\n🧪 Testing installation...")
    
    try:
        # Test imports
        print("   Testing imports...")
        import multimodal_agent
        from multimodal_agent import MultimodalAgent, settings
        print("   ✅ Core imports successful")
        
        # Test configuration
        print("   Testing configuration...")
        print(f"   📁 Content storage: {settings.content_storage_path}")
        print(f"   📤 Output storage: {settings.output_storage_path}")
        print("   ✅ Configuration loaded successfully")
        
        # Test API key configuration
        if settings.gemini_api_key and settings.gemini_api_key != "your_gemini_api_key_here":
            print("   ✅ Gemini API key configured")
        else:
            print("   ⚠️  Gemini API key needs configuration")
        
        if settings.openai_api_key and settings.openai_api_key != "your_openai_api_key_here":
            print("   ✅ OpenAI API key configured")
        else:
            print("   ⚠️  OpenAI API key needs configuration")
        
        print("\n✅ Installation test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False


def print_next_steps():
    """Print next steps for the user"""
    print("""
🎉 Setup Complete!

Next Steps:
───────────

1. 🔑 Configure API Keys (if not done already):
   Edit .env file with your API keys:
   nano .env

2. 🚀 Run the demo:
   python main.py

3. 📚 Explore examples:
   cd multimodal_agent/examples/
   python video_caption_example.py

4. 🔧 Customize the agent:
   - Add custom processors to extend functionality
   - Create plugins in the plugins/ directory
   - Modify configuration in multimodal_agent/config/settings.py

5. 📖 Documentation:
   Check README.md for detailed usage instructions

6. 🐛 Support:
   - Email: amp10162@nyu.edu
   - Website: panuj.com
   - GitHub Issues: For bug reports and feature requests

Happy content creating! 🎨
    """)


def main():
    """Main setup function"""
    print_banner()
    
    # Check requirements
    if not check_python_version():
        sys.exit(1)
    
    if not check_dependencies():
        response = input("\nContinue setup without all dependencies? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Setup steps
    steps = [
        ("Environment Configuration", setup_environment),
        ("Python Dependencies", install_python_dependencies),
        ("Project Directories", create_directories),
        ("Installation Test", test_installation)
    ]
    
    for step_name, step_function in steps:
        try:
            if not step_function():
                print(f"\n❌ Setup failed at: {step_name}")
                sys.exit(1)
        except KeyboardInterrupt:
            print(f"\n⚠️  Setup interrupted during: {step_name}")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Unexpected error during {step_name}: {e}")
            sys.exit(1)
    
    print_next_steps()
    print("🎉 Setup completed successfully!")


if __name__ == "__main__":
    main() 