"""
OpenAI AI Provider implementation

This module provides integration with OpenAI's models for text generation
and vision capabilities.

Author: Anuj Patel (amp10162@nyu.edu)
Website: panuj.com
"""
import base64
import asyncio
from typing import Dict, List, Any, Optional

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..core.base import BaseAIProvider, AIProvider
from ..utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIProvider(BaseAIProvider):
    """
    OpenAI AI provider implementation
    
    Provides integration with OpenAI's models including GPT-4, GPT-4 Vision,
    and other capabilities.
    """
    
    def __init__(self, api_key: str, config: Dict[str, Any] = None):
        """
        Initialize the OpenAI provider
        
        Args:
            api_key: OpenAI API key
            config: Additional configuration options
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI library not available. "
                "Install with: pip install openai"
            )
        
        super().__init__(AIProvider.OPENAI, api_key, config)
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=api_key)
        
        # Default models
        self.default_text_model = config.get("default_text_model", "gpt-4-turbo-preview")
        self.default_vision_model = config.get("default_vision_model", "gpt-4-vision-preview")
        self.default_embedding_model = config.get("default_embedding_model", "text-embedding-ada-002")
        
        # Default parameters
        self.default_params = {
            "temperature": config.get("temperature", 0.7),
            "max_tokens": config.get("max_tokens", 4096),
            "top_p": config.get("top_p", 1.0),
            "frequency_penalty": config.get("frequency_penalty", 0.0),
            "presence_penalty": config.get("presence_penalty", 0.0)
        }
        
        logger.info("OpenAI provider initialized successfully")
    
    async def generate_text(self, prompt: str, model: str = None, **kwargs) -> str:
        """
        Generate text using OpenAI models
        
        Args:
            prompt: Text prompt for generation
            model: Specific model to use (optional)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        try:
            model_name = model or self.default_text_model
            
            # Merge default params with overrides
            params = self.default_params.copy()
            params.update(kwargs)
            
            # Create completion
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                **params
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {e}")
            raise
    
    async def analyze_image(self, image_data: bytes, prompt: str, model: str = None, **kwargs) -> str:
        """
        Analyze image using OpenAI vision models
        
        Args:
            image_data: Image data as bytes
            prompt: Analysis prompt
            model: Specific model to use (optional)
            **kwargs: Additional parameters
            
        Returns:
            Image analysis result
        """
        try:
            model_name = model or self.default_vision_model
            
            # Merge default params with overrides
            params = self.default_params.copy()
            params.update(kwargs)
            
            # Convert image to base64
            image_b64 = base64.b64encode(image_data).decode()
            
            # Create message with image
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                    }
                ]
            }]
            
            # Create completion
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                **params
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error analyzing image with OpenAI: {e}")
            raise
    
    async def generate_embeddings(self, texts: List[str], model: str = None) -> List[List[float]]:
        """
        Generate embeddings for text using OpenAI models
        
        Args:
            texts: List of texts to embed
            model: Embedding model to use (optional)
            
        Returns:
            List of embedding vectors
        """
        try:
            model_name = model or self.default_embedding_model
            
            response = await self.client.embeddings.create(
                model=model_name,
                input=texts
            )
            
            return [data.embedding for data in response.data]
            
        except Exception as e:
            logger.error(f"Error generating embeddings with OpenAI: {e}")
            raise
    
    async def generate_multimodal_content(self, messages: List[Dict[str, Any]], 
                                        model: str = None, **kwargs) -> str:
        """
        Generate content using multimodal messages
        
        Args:
            messages: List of message dictionaries with potentially multimodal content
            model: Specific model to use (optional)
            **kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        try:
            model_name = model or self.default_vision_model
            
            # Merge default params with overrides
            params = self.default_params.copy()
            params.update(kwargs)
            
            # Create completion
            response = await self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                **params
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating multimodal content with OpenAI: {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available OpenAI models
        
        Returns:
            List of available model names
        """
        try:
            # Note: This would need to be async in a real implementation
            # For now, return known models
            return [
                "gpt-4-turbo-preview",
                "gpt-4-vision-preview", 
                "gpt-4",
                "gpt-3.5-turbo",
                "text-embedding-ada-002",
                "text-embedding-3-small",
                "text-embedding-3-large"
            ]
        except Exception as e:
            logger.error(f"Error listing OpenAI models: {e}")
            return []
    
    async def count_tokens(self, text: str, model: str = None) -> int:
        """
        Estimate token count for text
        
        Args:
            text: Text to count tokens for
            model: Model to use for counting (optional)
            
        Returns:
            Estimated number of tokens
        """
        try:
            # Simple estimation: ~4 characters per token for English
            return len(text) // 4
            
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            return 0
    
    async def health_check(self) -> bool:
        """
        Check if OpenAI provider is healthy and accessible
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Simple test generation
            test_response = await self.generate_text(
                "Hello", 
                max_tokens=5
            )
            return len(test_response) > 0
            
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return False
    
    def set_default_params(self, **params):
        """
        Update default parameters
        
        Args:
            **params: Parameters to update
        """
        self.default_params.update(params)
        logger.info("Updated OpenAI default parameters")
    
    def __str__(self) -> str:
        return f"OpenAIProvider(model={self.default_text_model})"
    
    def __repr__(self) -> str:
        return f"<OpenAIProvider(text_model='{self.default_text_model}', vision_model='{self.default_vision_model}')>" 