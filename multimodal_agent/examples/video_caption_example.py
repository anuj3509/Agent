"""
Example: Video Caption Generation

This script demonstrates how to use the multimodal agent to generate
captions for video files with various customization options.
"""
import asyncio
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from multimodal_agent import MultimodalAgent


async def basic_video_captioning():
    """Basic video captioning example"""
    print("🎬 Basic Video Captioning Example")
    print("-" * 40)
    
    # Initialize the agent
    agent = MultimodalAgent()
    
    # Example video path (replace with your video file)
    video_path = "path/to/your/video.mp4"
    
    try:
        # Generate captions
        result = await agent.generate_video_captions(
            video_path=video_path,
            context="Educational video about renewable energy"
        )
        
        if result["status"] == "success":
            captions = result["data"]["captions"]
            
            print(f"✅ Captions generated successfully!")
            print(f"📊 Video Statistics:")
            print(f"  - Duration: {captions['duration']:.2f} seconds")
            print(f"  - Total frames: {captions['total_frames']}")
            print(f"  - Frames with captions: {len(captions['frame_captions'])}")
            
            print(f"\n📝 Summary:")
            print(captions['summary'])
            
            print(f"\n🎯 Sample Captions:")
            for caption in captions['frame_captions'][:5]:
                timestamp = caption['timestamp']
                text = caption['caption']
                print(f"  [{timestamp:6.2f}s] {text}")
                
        else:
            print(f"❌ Error: {result['error']}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure to provide a valid video file path")


async def advanced_video_captioning():
    """Advanced video captioning with SRT export"""
    print("\n🎬 Advanced Video Captioning with SRT Export")
    print("-" * 50)
    
    agent = MultimodalAgent()
    video_path = "path/to/your/video.mp4"
    
    try:
        # Generate SRT format captions
        from multimodal_agent.modules.video_captions import VideoCaptionProcessor
        from langchain_openai import ChatOpenAI
        
        # Create processor directly for SRT functionality
        vision_llm = ChatOpenAI(model="gpt-4-vision-preview")
        processor = VideoCaptionProcessor(vision_llm)
        
        srt_content = await processor.generate_srt_captions(
            video_path=Path(video_path),
            context="Corporate training video",
            caption_duration=5.0  # 5 seconds per caption
        )
        
        # Save SRT file
        srt_path = Path(video_path).with_suffix('.srt')
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)
        
        print(f"✅ SRT captions saved to: {srt_path}")
        print(f"\n📄 SRT Preview:")
        print(srt_content[:500] + "..." if len(srt_content) > 500 else srt_content)
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def batch_video_processing():
    """Process multiple videos in batch"""
    print("\n🎬 Batch Video Processing Example")
    print("-" * 40)
    
    agent = MultimodalAgent()
    
    # List of video files to process
    video_files = [
        "video1.mp4",
        "video2.mp4", 
        "video3.mp4"
    ]
    
    results = []
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\nProcessing video {i}/{len(video_files)}: {video_path}")
        
        try:
            result = await agent.generate_video_captions(
                video_path=video_path,
                context=f"Video {i} in the series"
            )
            
            if result["status"] == "success":
                captions = result["data"]["captions"]
                print(f"  ✅ Success - {len(captions['frame_captions'])} captions generated")
                results.append({
                    "video": video_path,
                    "status": "success",
                    "caption_count": len(captions['frame_captions']),
                    "duration": captions['duration']
                })
            else:
                print(f"  ❌ Failed: {result['error']}")
                results.append({
                    "video": video_path,
                    "status": "failed",
                    "error": result['error']
                })
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                "video": video_path,
                "status": "error",
                "error": str(e)
            })
    
    # Summary
    print(f"\n📊 Batch Processing Summary:")
    successful = len([r for r in results if r["status"] == "success"])
    print(f"  ✅ Successful: {successful}/{len(video_files)}")
    
    total_captions = sum(r.get("caption_count", 0) for r in results)
    print(f"  📝 Total captions generated: {total_captions}")


async def main():
    """Run all video captioning examples"""
    print("🎥 Video Caption Generation Examples")
    print("=" * 50)
    
    await basic_video_captioning()
    await advanced_video_captioning()
    await batch_video_processing()
    
    print("\n🎉 All examples completed!")
    print("\n💡 Tips:")
    print("  • Use high-quality videos for better caption accuracy")
    print("  • Provide context to improve caption relevance") 
    print("  • Adjust frame sample rate based on video content")
    print("  • Consider video duration limits for processing efficiency")


if __name__ == "__main__":
    asyncio.run(main()) 