#!/usr/bin/env python3
"""
Simple Demo of Multimodal AI Agent
Works without OpenCV - focuses on text and basic image processing
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import io
import base64

# Load environment variables
load_dotenv('.env')

class SimpleMultimodalAgent:
    """Simplified Multimodal AI Agent using Gemini"""
    
    def __init__(self):
        # Configure Gemini API
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
    def analyze_text(self, text: str, task: str = "analyze") -> str:
        """Analyze text content"""
        prompt = f"""
        Task: {task}
        
        Text to analyze: {text}
        
        Please provide a comprehensive analysis focusing on:
        1. Key themes and concepts
        2. Sentiment and tone
        3. Potential use cases
        4. Content structure
        
        Provide a clear, actionable response.
        """
        
        response = self.model.generate_content(prompt)
        return response.text
    
    def analyze_image(self, image_path: str, task: str = "describe") -> str:
        """Analyze image content"""
        try:
            image = Image.open(image_path)
            
            prompt = f"""
            Task: {task}
            
            Please analyze this image and provide:
            1. Detailed description of what you see
            2. Key objects, people, or scenes
            3. Colors, composition, and style
            4. Potential context or story
            5. Suggestions for captions or marketing use
            
            Be specific and creative in your analysis.
            """
            
            response = self.model.generate_content([prompt, image])
            return response.text
            
        except Exception as e:
            return f"Error analyzing image: {str(e)}"
    
    def create_marketing_content(self, description: str, target_audience: str = "general") -> str:
        """Generate marketing content from description"""
        prompt = f"""
        Create compelling marketing content based on this description:
        {description}
        
        Target Audience: {target_audience}
        
        Please provide:
        1. 3 different headline options
        2. A short marketing description (50-100 words)
        3. 3 social media captions with hashtags
        4. A call-to-action suggestion
        
        Make it engaging and persuasive.
        """
        
        response = self.model.generate_content(prompt)
        return response.text
    
    def create_interactive_narrative(self, theme: str, elements: list) -> str:
        """Create an interactive multimedia narrative"""
        elements_str = ", ".join(elements)
        
        prompt = f"""
        Create an interactive multimedia narrative with these specifications:
        
        Theme: {theme}
        Elements to include: {elements_str}
        
        Please provide:
        1. A compelling story outline (5-7 key scenes)
        2. Interactive decision points for audience engagement
        3. Multimedia suggestions (images, sounds, videos) for each scene
        4. Character development and dialogue samples
        5. Call-to-action moments for audience participation
        
        Format as a structured narrative plan ready for production.
        """
        
        response = self.model.generate_content(prompt)
        return response.text

def demonstrate_agent():
    """Demonstrate the multimodal agent capabilities"""
    print("🤖 Initializing Simple Multimodal AI Agent...")
    
    try:
        agent = SimpleMultimodalAgent()
        print("✅ Agent initialized successfully!\n")
        
        # Demo 1: Text Analysis
        print("📝 DEMO 1: Text Content Analysis")
        print("=" * 50)
        sample_text = """
        Artificial Intelligence is revolutionizing the way we create and consume content. 
        From automated video editing to personalized marketing campaigns, AI tools are 
        becoming indispensable for content creators. The future of content creation lies 
        in human-AI collaboration, where creativity meets efficiency.
        """
        
        analysis = agent.analyze_text(sample_text, "content strategy analysis")
        print("Sample Text:", sample_text.strip())
        print("\n🔍 AI Analysis:")
        print(analysis)
        print("\n" + "="*80 + "\n")
        
        # Demo 2: Marketing Content Creation
        print("🎯 DEMO 2: Marketing Content Generation")
        print("=" * 50)
        product_description = "A revolutionary AI-powered content creation platform that helps marketers create engaging multimedia campaigns in minutes instead of hours."
        
        marketing_content = agent.create_marketing_content(product_description, "digital marketers")
        print("Product Description:", product_description)
        print("\n📢 Generated Marketing Content:")
        print(marketing_content)
        print("\n" + "="*80 + "\n")
        
        # Demo 3: Interactive Narrative
        print("📚 DEMO 3: Interactive Multimedia Narrative")
        print("=" * 50)
        theme = "A journey through the future of AI and human creativity"
        elements = ["virtual reality", "AI assistants", "creative collaboration", "digital art", "emotional connection"]
        
        narrative = agent.create_interactive_narrative(theme, elements)
        print(f"Theme: {theme}")
        print(f"Elements: {', '.join(elements)}")
        print("\n🎬 Generated Interactive Narrative:")
        print(narrative)
        print("\n" + "="*80 + "\n")
        
        print("🎉 Demo completed successfully!")
        print("\n💡 Try these advanced features:")
        print("   • Image analysis: Add an image file and call agent.analyze_image()")
        print("   • Custom prompts: Modify the prompts for different use cases")
        print("   • Integration: Connect with your existing content workflow")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("Please check your setup and try again.")

def interactive_mode():
    """Interactive mode for testing different features"""
    print("\n🔧 INTERACTIVE MODE")
    print("=" * 50)
    
    try:
        agent = SimpleMultimodalAgent()
        
        while True:
            print("\nChoose an option:")
            print("1. Analyze text")
            print("2. Create marketing content") 
            print("3. Create interactive narrative")
            print("4. Run full demo")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                text = input("Enter text to analyze: ")
                task = input("Analysis type (or press Enter for default): ").strip() or "analyze"
                result = agent.analyze_text(text, task)
                print(f"\n🔍 Analysis:\n{result}")
                
            elif choice == "2":
                description = input("Enter product/service description: ")
                audience = input("Target audience (or press Enter for general): ").strip() or "general"
                result = agent.create_marketing_content(description, audience)
                print(f"\n📢 Marketing Content:\n{result}")
                
            elif choice == "3":
                theme = input("Enter narrative theme: ")
                elements_input = input("Enter elements (comma-separated): ")
                elements = [e.strip() for e in elements_input.split(",")]
                result = agent.create_interactive_narrative(theme, elements)
                print(f"\n🎬 Interactive Narrative:\n{result}")
                
            elif choice == "4":
                demonstrate_agent()
                
            elif choice == "5":
                print("👋 Goodbye!")
                break
                
            else:
                print("Invalid choice. Please try again.")
                
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    print("🚀 Simple Multimodal AI Agent Demo")
    print("Powered by Gemini 2.5 Flash")
    print("=" * 60)
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_mode()
    else:
        demonstrate_agent()
        
        # Ask if user wants interactive mode
        while True:
            try:
                choice = input("\n🔧 Would you like to try interactive mode? (y/n): ").strip().lower()
                if choice == 'y':
                    interactive_mode()
                    break
                elif choice == 'n':
                    print("👋 Thanks for trying the demo!")
                    break
                else:
                    print("Please enter 'y' or 'n'")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break 