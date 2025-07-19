"""
Main application demonstrating the Multimodal AI Agent for Content Creation and Analysis

This script provides examples of how to use the agent for:
- Video caption generation
- Interactive multimedia narratives
- Targeted marketing content creation
"""
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any

from multimodal_agent import MultimodalAgent, settings


async def demo_video_captions():
    """Demonstrate video caption generation"""
    print("\n🎬 Video Caption Generation Demo")
    print("=" * 50)
    
    # Initialize agent
    agent = MultimodalAgent()
    
    # Example video caption generation
    print("Generating captions for a sample video...")
    
    # Note: In a real scenario, you would provide a path to an actual video file
    sample_video_path = "sample_video.mp4"  # Replace with actual video path
    
    try:
        result = await agent.generate_video_captions(
            video_path=sample_video_path,
            context="Educational content about AI and machine learning"
        )
        
        if result["status"] == "success":
            captions_data = result["data"]["captions"]
            print(f"✅ Successfully generated captions!")
            print(f"📊 Video duration: {captions_data['duration']:.2f} seconds")
            print(f"🖼️ Total frames analyzed: {captions_data['total_frames']}")
            print(f"\n📝 Video Summary:")
            print(captions_data["summary"])
            
            print(f"\n🎯 Frame-by-frame captions (showing first 3):")
            for i, caption in enumerate(captions_data["frame_captions"][:3]):
                print(f"  [{caption['timestamp']:.2f}s] {caption['caption']}")
        else:
            print(f"❌ Error: {result['error']}")
            
    except Exception as e:
        print(f"❌ Error: Could not process video - {e}")
        print("💡 Note: Provide a valid video file path to test this functionality")


async def demo_multimedia_narrative():
    """Demonstrate multimedia narrative creation"""
    print("\n📚 Multimedia Narrative Creation Demo")
    print("=" * 50)
    
    # Initialize agent
    agent = MultimodalAgent()
    
    print("Creating an interactive multimedia narrative...")
    
    try:
        result = await agent.create_multimedia_narrative(
            topic="The Future of Artificial Intelligence",
            media_elements=[
                "An image of a futuristic AI robot",
                "A chart showing AI development timeline",
                "A video of AI in healthcare applications",
                "An infographic about machine learning algorithms"
            ],
            style="engaging",
            target_audience="technology enthusiasts and students"
        )
        
        if result["status"] == "success":
            narrative = result["data"]["narrative"]
            print(f"✅ Successfully created narrative!")
            print(f"📖 Title: {narrative['title']}")
            print(f"📝 Description: {narrative['description']}")
            print(f"⏱️ Estimated duration: {narrative['metadata']['estimated_duration']:.1f} minutes")
            print(f"🎯 Complexity level: {narrative['metadata']['complexity_level']}")
            print(f"🔗 Interactive points: {narrative['metadata']['interactive_points']}")
            
            print(f"\n📑 Narrative segments ({len(narrative['segments'])} total):")
            for i, segment in enumerate(narrative['segments'][:2], 1):
                print(f"\n  Segment {i}: {segment.metadata['title']}")
                print(f"  Content preview: {segment.content[:200]}...")
                print(f"  Media elements: {len(segment.media_elements)}")
                print(f"  Transitions: {', '.join(segment.transitions)}")
        else:
            print(f"❌ Error: {result['error']}")
            
    except Exception as e:
        print(f"❌ Error creating narrative: {e}")


async def demo_marketing_content():
    """Demonstrate marketing content creation"""
    print("\n🎯 Marketing Content Creation Demo")
    print("=" * 50)
    
    # Initialize agent
    agent = MultimodalAgent()
    
    print("Creating targeted marketing content...")
    
    # Sample product information
    product_info = {
        "name": "AI-Powered Study Assistant",
        "description": "An intelligent study companion that helps students learn more effectively using AI",
        "category": "EdTech",
        "key_features": [
            "Personalized learning paths",
            "AI-generated quizzes",
            "Progress tracking",
            "Multi-language support"
        ],
        "benefits": [
            "Improve study efficiency by 40%",
            "Personalized learning experience",
            "24/7 availability",
            "Covers all subjects"
        ],
        "brand_voice": "friendly and educational",
        "competitors": ["Coursera", "Khan Academy", "Duolingo"]
    }
    
    try:
        result = await agent.create_marketing_content(
            product_info=product_info,
            target_audience="college students and lifelong learners aged 18-35",
            content_type="social_media_post",
            visual_elements=[
                "A student using laptop with AI interface",
                "Colorful learning dashboard screenshot"
            ]
        )
        
        if result["status"] == "success":
            content = result["data"]["marketing_content"]
            print(f"✅ Successfully created marketing content!")
            print(f"📱 Content type: {content.content_type}")
            print(f"🎯 Target audience: {content.target_audience}")
            print(f"📊 Content length: {content.metadata['content_length']} characters")
            
            print(f"\n📝 Generated Content:")
            print(f"Title: {content.title}")
            print(f"Content: {content.content}")
            print(f"CTA: {content.call_to_action}")
            
            print(f"\n📱 Platform-specific variations:")
            for platform, variation in content.platform_specific.items():
                print(f"\n  {platform.upper()}:")
                print(f"  {variation[:150]}{'...' if len(variation) > 150 else ''}")
            
            print(f"\n🎨 Visual recommendations ({len(content.visual_recommendations)}):")
            for i, rec in enumerate(content.visual_recommendations[:2], 1):
                print(f"  {i}. {rec.get('type', 'Visual')}: {rec.get('description', 'No description')}")
        else:
            print(f"❌ Error: {result['error']}")
            
    except Exception as e:
        print(f"❌ Error creating marketing content: {e}")


async def demo_integrated_workflow():
    """Demonstrate an integrated workflow combining multiple capabilities"""
    print("\n🔄 Integrated Workflow Demo")
    print("=" * 50)
    
    print("Creating a complete content campaign...")
    
    # Initialize agent
    agent = MultimodalAgent()
    
    # Simulate a complete workflow
    campaign_data = {
        "product": {
            "name": "EcoSmart Water Bottle",
            "description": "Smart water bottle that tracks hydration and temperature",
            "category": "Health & Fitness"
        },
        "content_goals": [
            "Generate video captions for product demo",
            "Create educational narrative about hydration",
            "Develop social media marketing campaign"
        ]
    }
    
    print(f"Campaign for: {campaign_data['product']['name']}")
    print(f"Goals: {len(campaign_data['content_goals'])} content pieces")
    
    # This would integrate all three capabilities in a real scenario
    print("\n📋 Workflow Steps:")
    print("1. ✅ Analyze product demo video → Generate captions")
    print("2. ✅ Create educational narrative → Interactive story")
    print("3. ✅ Develop marketing content → Multi-platform campaign")
    print("4. ✅ Combine insights → Unified content strategy")
    
    print("\n🎉 Integrated campaign complete!")
    print("💡 This demonstrates how the agent can orchestrate multiple")
    print("   content creation tasks in a coordinated workflow.")


def display_system_info():
    """Display system information and configuration"""
    print("🤖 Multimodal AI Agent for Content Creation and Analysis")
    print("=" * 60)
    print(f"📁 Content storage: {settings.content_storage_path}")
    print(f"📤 Output storage: {settings.output_storage_path}")
    print(f"🎯 Default text model: {settings.default_text_model}")
    print(f"👁️ Default vision model: {settings.default_vision_model}")
    print(f"🔧 Debug mode: {settings.debug}")
    
    # Check API key
    if settings.openai_api_key:
        print("✅ OpenAI API key configured")
    else:
        print("⚠️ OpenAI API key not found - set OPENAI_API_KEY environment variable")
    
    print("\n🎯 Capabilities:")
    print("  • Video caption generation with computer vision")
    print("  • Interactive multimedia narrative creation")
    print("  • Targeted marketing content with visual analysis")
    print("  • Multi-platform content optimization")
    print("  • LangGraph-powered workflow orchestration")


async def main():
    """Main application entry point"""
    display_system_info()
    
    # Check if OpenAI API key is available
    if not settings.openai_api_key:
        print("\n❌ OpenAI API key required to run demos.")
        print("💡 Set the OPENAI_API_KEY environment variable and try again.")
        print("\n   export OPENAI_API_KEY='your-api-key-here'")
        return
    
    print("\n🚀 Running Capability Demos...")
    
    try:
        # Run all demos
        await demo_video_captions()
        await demo_multimedia_narrative()
        await demo_marketing_content()
        await demo_integrated_workflow()
        
        print("\n✨ All demos completed successfully!")
        print("\n📚 Next Steps:")
        print("  • Check the examples/ directory for more detailed usage")
        print("  • Customize the agent for your specific use cases")
        print("  • Integrate with your existing content workflows")
        print("  • Explore the Streamlit web interface (coming soon)")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")


if __name__ == "__main__":
    # Run the main application
    asyncio.run(main()) 