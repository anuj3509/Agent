"""
Google Gemini AI Provider implementation

This module provides integration with Google's Gemini AI models for text generation
and multimodal capabilities.

Author: Anuj Patel (amp10162@nyu.edu)
Website: panuj.com
"""
import base64
import asyncio
from typing import Dict, List, Any, Optional

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from ..core.base import BaseAIProvider, AIProvider
from ..utils.logger import get_logger

logger = get_logger(__name__)


class GeminiProvider(BaseAIProvider):
    """
    Google Gemini AI provider implementation
    
    Provides integration with Google's Gemini models for text generation,
    image analysis, and multimodal tasks.
    """
    
    def __init__(self, api_key: str, config: Dict[str, Any] = None):
        """
        Initialize the Gemini provider
        
        Args:
            api_key: Google Gemini API key
            config: Additional configuration options
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Google Generative AI library not available. "
                "Install with: pip install google-generativeai"
            )
        
        super().__init__(AIProvider.GEMINI, api_key, config)
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Default models
        self.default_text_model = config.get("default_text_model", "gemini-2.0-flash-exp")
        self.default_vision_model = config.get("default_vision_model", "gemini-2.0-flash-exp")
        
        # Safety settings
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        
        # Generation config
        self.generation_config = {
            "temperature": config.get("temperature", 0.7),
            "top_p": config.get("top_p", 0.8),
            "top_k": config.get("top_k", 40),
            "max_output_tokens": config.get("max_output_tokens", 8192),
        }
        
        logger.info("Gemini provider initialized successfully")
    
    async def generate_text(self, prompt: str, model: str = None, **kwargs) -> str:
        """
        Generate text using Gemini models
        
        Args:
            prompt: Text prompt for generation
            model: Specific model to use (optional)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        try:
            model_name = model or self.default_text_model
            
            # Create generation config with overrides
            gen_config = self.generation_config.copy()
            gen_config.update(kwargs)
            
            # Initialize the model
            gemini_model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=gen_config,
                safety_settings=self.safety_settings
            )
            
            # Generate response
            response = await asyncio.to_thread(
                gemini_model.generate_content,
                prompt
            )
            
            if response.candidates and len(response.candidates) > 0:
                return response.candidates[0].content.parts[0].text
            else:
                raise ValueError("No valid response generated")
                
        except Exception as e:
            logger.error(f"Error generating text with Gemini: {e}")
            raise
    
    async def analyze_image(self, image_data: bytes, prompt: str, model: str = None, **kwargs) -> str:
        """
        Analyze image using Gemini vision models
        
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
            
            # Create generation config with overrides
            gen_config = self.generation_config.copy()
            gen_config.update(kwargs)
            
            # Initialize the model
            gemini_model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=gen_config,
                safety_settings=self.safety_settings
            )
            
            # Prepare image for Gemini
            image_parts = [
                {
                    "mime_type": "image/jpeg",  # Assume JPEG, could be made more flexible
                    "data": base64.b64encode(image_data).decode()
                }
            ]
            
            # Create content with image and prompt
            content = [prompt] + image_parts
            
            # Generate response
            response = await asyncio.to_thread(
                gemini_model.generate_content,
                content
            )
            
            if response.candidates and len(response.candidates) > 0:
                return response.candidates[0].content.parts[0].text
            else:
                raise ValueError("No valid response generated")
                
        except Exception as e:
            logger.error(f"Error analyzing image with Gemini: {e}")
            raise
    
    async def generate_multimodal_content(self, content_parts: List[Dict[str, Any]], 
                                        model: str = None, **kwargs) -> str:
        """
        Generate content using multiple modalities (text, images, etc.)
        
        Args:
            content_parts: List of content parts (text, images, etc.)
            model: Specific model to use (optional)
            **kwargs: Additional parameters
            
        Returns:
            Generated multimodal response
        """
        try:
            model_name = model or self.default_vision_model
            
            # Create generation config with overrides
            gen_config = self.generation_config.copy()
            gen_config.update(kwargs)
            
            # Initialize the model
            gemini_model = genai.GenerativeModel(
                model_name=model_name,
                generation_config=gen_config,
                safety_settings=self.safety_settings
            )
            
            # Process content parts
            processed_content = []
            for part in content_parts:
                if part["type"] == "text":
                    processed_content.append(part["content"])
                elif part["type"] == "image":
                    image_part = {
                        "mime_type": part.get("mime_type", "image/jpeg"),
                        "data": base64.b64encode(part["content"]).decode()
                    }
                    processed_content.append(image_part)
            
            # Generate response
            response = await asyncio.to_thread(
                gemini_model.generate_content,
                processed_content
            )
            
            if response.candidates and len(response.candidates) > 0:
                return response.candidates[0].content.parts[0].text
            else:
                raise ValueError("No valid response generated")
                
        except Exception as e:
            logger.error(f"Error generating multimodal content with Gemini: {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available Gemini models
        
        Returns:
            List of available model names
        """
        try:
            models = []
            for model in genai.list_models():
                if 'generateContent' in model.supported_generation_methods:
                    models.append(model.name.replace('models/', ''))
            return models
        except Exception as e:
            logger.error(f"Error listing Gemini models: {e}")
            return [
                "gemini-2.0-flash-exp",
                "gemini-1.5-pro",
                "gemini-1.5-flash",
                "gemini-pro-vision"
            ]
    
    async def count_tokens(self, content: str, model: str = None) -> int:
        """
        Count tokens in content
        
        Args:
            content: Content to count tokens for
            model: Model to use for counting (optional)
            
        Returns:
            Number of tokens
        """
        try:
            model_name = model or self.default_text_model
            gemini_model = genai.GenerativeModel(model_name)
            
            response = await asyncio.to_thread(
                gemini_model.count_tokens,
                content
            )
            
            return response.total_tokens
            
        except Exception as e:
            logger.error(f"Error counting tokens with Gemini: {e}")
            return 0
    
    async def health_check(self) -> bool:
        """
        Check if Gemini provider is healthy and accessible
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Simple test generation
            test_response = await self.generate_text(
                "Hello", 
                max_output_tokens=10
            )
            return len(test_response) > 0
            
        except Exception as e:
            logger.error(f"Gemini health check failed: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        try:
            model = genai.get_model(f"models/{model_name}")
            return {
                "name": model.name,
                "display_name": model.display_name,
                "description": model.description,
                "input_token_limit": model.input_token_limit,
                "output_token_limit": model.output_token_limit,
                "supported_generation_methods": model.supported_generation_methods,
                "temperature": getattr(model, 'temperature', None),
                "top_p": getattr(model, 'top_p', None),
                "top_k": getattr(model, 'top_k', None)
            }
        except Exception as e:
            logger.error(f"Error getting model info for {model_name}: {e}")
            return {"error": str(e)}
    
    def set_safety_settings(self, safety_settings: Dict[HarmCategory, HarmBlockThreshold]):
        """
        Update safety settings for content generation
        
        Args:
            safety_settings: Dictionary of safety settings
        """
        self.safety_settings = safety_settings
        logger.info("Updated Gemini safety settings")
    
    def set_generation_config(self, **config):
        """
        Update generation configuration
        
        Args:
            **config: Generation configuration parameters
        """
        self.generation_config.update(config)
        logger.info("Updated Gemini generation configuration")
    
    def __str__(self) -> str:
        return f"GeminiProvider(model={self.default_text_model})"
    
    def __repr__(self) -> str:
        return f"<GeminiProvider(text_model='{self.default_text_model}', vision_model='{self.default_vision_model}')>" 