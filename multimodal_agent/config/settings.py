"""
Configuration settings for the Multimodal AI Agent
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # API Keys
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    huggingface_api_token: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Model Configuration
    default_vision_model: str = "gpt-4-vision-preview"
    default_text_model: str = "gpt-4-turbo-preview"
    default_embedding_model: str = "text-embedding-ada-002"
    
    # Gemini model configuration
    gemini_model: str = "gemini-2.0-flash-exp"
    gemini_vision_model: str = "gemini-2.0-flash-exp"
    
    # Default AI provider
    default_ai_provider: str = "openai"  # Options: openai, gemini, anthropic
    
    # Application Settings
    debug: bool = True
    log_level: str = "INFO"
    max_content_length: int = 10485760  # 10MB
    
    # Storage Configuration
    content_storage_path: Path = Path("./data/content")
    output_storage_path: Path = Path("./data/output")
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Video Processing Settings
    max_video_duration: int = 300  # 5 minutes
    video_frame_sample_rate: int = 1  # Extract 1 frame per second
    
    # Content Generation Settings
    max_narrative_length: int = 5000
    marketing_content_templates: list = [
        "social_media_post",
        "blog_article",
        "product_description",
        "email_campaign"
    ]
    
    # Feature Flags
    enable_video_processing: bool = True
    enable_narrative_generation: bool = True
    enable_marketing_content: bool = True
    enable_gemini_integration: bool = True
    enable_batch_processing: bool = True
    enable_caching: bool = True
    
    # Concurrency Settings
    max_concurrent_requests: int = 10
    max_concurrent_video_processing: int = 3
    max_concurrent_content_generation: int = 5
    
    # Timeout Settings (in seconds)
    api_timeout: int = 300
    video_processing_timeout: int = 600
    content_generation_timeout: int = 120
    
    class Config:
        env_file = ".env"
        extra = "ignore"


# Create global settings instance
settings = Settings()

# Ensure storage directories exist
settings.content_storage_path.mkdir(parents=True, exist_ok=True)
settings.output_storage_path.mkdir(parents=True, exist_ok=True) 