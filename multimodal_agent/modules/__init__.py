"""
Multimodal AI Agent Modules

This package contains specialized modules for different content creation tasks:
- video_captions: Video analysis and caption generation
- multimedia_narratives: Interactive story and narrative creation
- marketing_content: Targeted marketing content generation
"""

from .video_captions import VideoCaptionProcessor
from .multimedia_narratives import NarrativeGenerator
from .marketing_content import MarketingContentCreator

__all__ = [
    "VideoCaptionProcessor",
    "NarrativeGenerator", 
    "MarketingContentCreator"
] 