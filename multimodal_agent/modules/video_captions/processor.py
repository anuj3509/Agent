"""
Video caption generation processor using computer vision and LLM
"""
import asyncio
import base64
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from ...config.settings import settings
from ...utils.logger import get_logger

logger = get_logger(__name__)


class VideoCaptionProcessor:
    """
    Processes videos to generate captions using computer vision and LLMs
    """
    
    def __init__(self, vision_llm: ChatOpenAI):
        """
        Initialize the video caption processor
        
        Args:
            vision_llm: Vision-enabled language model
        """
        self.vision_llm = vision_llm
        self.max_duration = settings.max_video_duration
        self.frame_sample_rate = settings.video_frame_sample_rate
    
    async def generate_captions(self, video_path: Path, context: str = "") -> Dict[str, Any]:
        """
        Generate captions for a video file
        
        Args:
            video_path: Path to the video file
            context: Additional context for caption generation
            
        Returns:
            Dictionary containing captions and metadata
        """
        try:
            logger.info(f"Processing video: {video_path}")
            
            # Validate video file
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Extract frames from video
            frames = await self._extract_frames(video_path)
            if not frames:
                raise ValueError("No frames could be extracted from the video")
            
            # Generate captions for each frame
            frame_captions = []
            for i, (frame, timestamp) in enumerate(frames):
                logger.info(f"Processing frame {i+1}/{len(frames)} at {timestamp:.2f}s")
                
                caption = await self._generate_frame_caption(frame, timestamp, context)
                frame_captions.append({
                    "timestamp": timestamp,
                    "caption": caption,
                    "frame_index": i
                })
            
            # Generate overall video summary
            summary = await self._generate_video_summary(frame_captions, context)
            
            # Create final output
            result = {
                "video_path": str(video_path),
                "total_frames": len(frames),
                "duration": frames[-1][1] if frames else 0,
                "frame_captions": frame_captions,
                "summary": summary,
                "context": context
            }
            
            logger.info(f"Video caption generation completed for {video_path}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating video captions: {e}")
            raise
    
    async def _extract_frames(self, video_path: Path) -> List[Tuple[np.ndarray, float]]:
        """
        Extract frames from video at specified intervals
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of (frame, timestamp) tuples
        """
        frames = []
        
        try:
            # Open video file
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            logger.info(f"Video properties: {fps:.2f} FPS, {total_frames} frames, {duration:.2f}s duration")
            
            # Check duration limit
            if duration > self.max_duration:
                logger.warning(f"Video duration ({duration:.2f}s) exceeds limit ({self.max_duration}s)")
                duration = self.max_duration
            
            # Calculate frame sampling
            frame_interval = int(fps / self.frame_sample_rate)
            frame_indices = range(0, min(total_frames, int(duration * fps)), frame_interval)
            
            # Extract frames
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    timestamp = frame_idx / fps
                    frames.append((frame_rgb, timestamp))
                    
                    if len(frames) % 10 == 0:
                        logger.info(f"Extracted {len(frames)} frames...")
            
            cap.release()
            logger.info(f"Extracted {len(frames)} frames from video")
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            if 'cap' in locals():
                cap.release()
            raise
        
        return frames
    
    async def _generate_frame_caption(self, frame: np.ndarray, timestamp: float, context: str) -> str:
        """
        Generate caption for a single frame
        
        Args:
            frame: Frame as numpy array
            timestamp: Timestamp in seconds
            context: Additional context
            
        Returns:
            Generated caption
        """
        try:
            # Convert frame to base64 for API
            frame_b64 = self._frame_to_base64(frame)
            
            # Create prompt
            prompt = f"""
            Analyze this video frame captured at {timestamp:.2f} seconds and provide a detailed caption.
            
            Context: {context if context else "No additional context provided"}
            
            Please describe:
            1. Main subjects/objects in the frame
            2. Actions or activities happening
            3. Setting/environment
            4. Any notable details or emotions
            
            Keep the caption concise but informative (2-3 sentences).
            """
            
            # Create message with image
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}
                    }
                ]
            )
            
            # Generate caption
            response = await self.vision_llm.ainvoke([message])
            caption = response.content.strip()
            
            logger.debug(f"Generated caption for frame at {timestamp:.2f}s: {caption[:100]}...")
            return caption
            
        except Exception as e:
            logger.error(f"Error generating frame caption: {e}")
            return f"Error generating caption for frame at {timestamp:.2f}s"
    
    async def _generate_video_summary(self, frame_captions: List[Dict[str, Any]], context: str) -> str:
        """
        Generate overall video summary from frame captions
        
        Args:
            frame_captions: List of frame caption data
            context: Additional context
            
        Returns:
            Video summary
        """
        try:
            # Combine all captions
            captions_text = "\n".join([
                f"[{cap['timestamp']:.2f}s] {cap['caption']}"
                for cap in frame_captions
            ])
            
            prompt = f"""
            Based on the following frame-by-frame captions from a video, create a comprehensive summary:
            
            Context: {context if context else "No additional context provided"}
            
            Frame Captions:
            {captions_text}
            
            Please provide:
            1. A concise overall summary of the video content
            2. Key events or progression throughout the video
            3. Main themes or subjects
            4. Any notable changes or developments
            
            Format as a well-structured paragraph or short sections.
            """
            
            response = await self.vision_llm.ainvoke([HumanMessage(content=prompt)])
            summary = response.content.strip()
            
            logger.info("Generated video summary")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating video summary: {e}")
            return "Error generating video summary"
    
    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """
        Convert frame to base64 encoded string
        
        Args:
            frame: Frame as numpy array
            
        Returns:
            Base64 encoded string
        """
        # Convert to PIL Image
        image = Image.fromarray(frame)
        
        # Resize if too large (for API efficiency)
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to base64
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode()
    
    async def generate_srt_captions(self, video_path: Path, context: str = "", 
                                  caption_duration: float = 3.0) -> str:
        """
        Generate SRT format captions for the video
        
        Args:
            video_path: Path to video file
            context: Additional context
            caption_duration: Duration for each caption in seconds
            
        Returns:
            SRT formatted string
        """
        try:
            # Generate frame captions
            captions_data = await self.generate_captions(video_path, context)
            frame_captions = captions_data["frame_captions"]
            
            # Convert to SRT format
            srt_content = []
            for i, caption_data in enumerate(frame_captions):
                start_time = caption_data["timestamp"]
                end_time = min(start_time + caption_duration, captions_data["duration"])
                
                # Format timestamps for SRT
                start_srt = self._seconds_to_srt_time(start_time)
                end_srt = self._seconds_to_srt_time(end_time)
                
                # Add SRT entry
                srt_content.append(f"{i + 1}")
                srt_content.append(f"{start_srt} --> {end_srt}")
                srt_content.append(caption_data["caption"])
                srt_content.append("")  # Empty line between entries
            
            return "\n".join(srt_content)
            
        except Exception as e:
            logger.error(f"Error generating SRT captions: {e}")
            raise
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """
        Convert seconds to SRT time format (HH:MM:SS,mmm)
        
        Args:
            seconds: Time in seconds
            
        Returns:
            SRT formatted time string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}" 