"""
Abstract base classes for multimodal agent processors

This module defines the base interfaces for all processors in the multimodal agent,
making the system modular and extensible.

Author: Anuj Patel (amp10162@nyu.edu)
Website: panuj.com
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum
import asyncio

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Type variables for generic processors
InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')


class ProcessorType(Enum):
    """Types of processors available in the system"""
    VIDEO_CAPTION = "video_caption"
    MULTIMEDIA_NARRATIVE = "multimedia_narrative"
    MARKETING_CONTENT = "marketing_content"
    CUSTOM = "custom"


class AIProvider(Enum):
    """Supported AI providers"""
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


@dataclass
class ProcessingResult:
    """Standard result format for all processors"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None


@dataclass
class ProcessorConfig:
    """Configuration for processors"""
    name: str
    processor_type: ProcessorType
    ai_provider: AIProvider
    model_config: Dict[str, Any]
    settings: Dict[str, Any]
    enabled: bool = True


class BaseProcessor(ABC, Generic[InputType, OutputType]):
    """
    Abstract base class for all processors in the multimodal agent
    
    This class defines the common interface that all processors must implement,
    ensuring consistency and extensibility across the system.
    """
    
    def __init__(self, config: ProcessorConfig):
        """
        Initialize the processor with configuration
        
        Args:
            config: ProcessorConfig object with processor settings
        """
        self.config = config
        self.name = config.name
        self.processor_type = config.processor_type
        self.ai_provider = config.ai_provider
        self.model_config = config.model_config
        self.settings = config.settings
        self.enabled = config.enabled
        
        logger.info(f"Initialized {self.name} processor ({self.processor_type.value})")
    
    @abstractmethod
    async def process(self, input_data: InputType) -> ProcessingResult:
        """
        Process input data and return result
        
        Args:
            input_data: Input data to process
            
        Returns:
            ProcessingResult with processed data or error information
        """
        pass
    
    @abstractmethod
    def validate_input(self, input_data: InputType) -> bool:
        """
        Validate input data before processing
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        pass
    
    def get_capabilities(self) -> List[str]:
        """
        Get list of capabilities this processor supports
        
        Returns:
            List of capability strings
        """
        return []
    
    def get_requirements(self) -> Dict[str, Any]:
        """
        Get requirements for this processor
        
        Returns:
            Dictionary of requirements (dependencies, resources, etc.)
        """
        return {
            "ai_provider": self.ai_provider.value,
            "model_config": self.model_config,
            "dependencies": []
        }
    
    async def health_check(self) -> bool:
        """
        Check if the processor is healthy and ready to process
        
        Returns:
            True if healthy, False otherwise
        """
        return self.enabled
    
    def __str__(self) -> str:
        return f"{self.name} ({self.processor_type.value})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', type='{self.processor_type.value}')>"


class BaseAIProvider(ABC):
    """
    Abstract base class for AI providers
    
    This class defines the interface for different AI providers (OpenAI, Gemini, etc.)
    allowing the system to work with multiple AI services.
    """
    
    def __init__(self, provider_type: AIProvider, api_key: str, config: Dict[str, Any] = None):
        """
        Initialize the AI provider
        
        Args:
            provider_type: Type of AI provider
            api_key: API key for the provider
            config: Additional configuration
        """
        self.provider_type = provider_type
        self.api_key = api_key
        self.config = config or {}
        
        logger.info(f"Initialized {provider_type.value} AI provider")
    
    @abstractmethod
    async def generate_text(self, prompt: str, model: str = None, **kwargs) -> str:
        """
        Generate text using the AI provider
        
        Args:
            prompt: Text prompt
            model: Model to use (optional)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    async def analyze_image(self, image_data: bytes, prompt: str, model: str = None, **kwargs) -> str:
        """
        Analyze image using vision models
        
        Args:
            image_data: Image data as bytes
            prompt: Analysis prompt
            model: Model to use (optional)
            **kwargs: Additional parameters
            
        Returns:
            Analysis result
        """
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """
        Get list of available models for this provider
        
        Returns:
            List of model names
        """
        pass
    
    async def health_check(self) -> bool:
        """
        Check if the AI provider is healthy and accessible
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Simple test to check if the provider is accessible
            models = self.get_available_models()
            return len(models) > 0
        except Exception as e:
            logger.error(f"Health check failed for {self.provider_type.value}: {e}")
            return False


class ProcessorRegistry:
    """
    Registry for managing processors in the multimodal agent
    
    This class provides a centralized way to register, discover, and manage
    different processors, making the system highly extensible.
    """
    
    def __init__(self):
        """Initialize the processor registry"""
        self._processors: Dict[str, BaseProcessor] = {}
        self._processor_types: Dict[ProcessorType, List[str]] = {
            processor_type: [] for processor_type in ProcessorType
        }
        
        logger.info("Initialized processor registry")
    
    def register_processor(self, processor: BaseProcessor) -> bool:
        """
        Register a processor in the registry
        
        Args:
            processor: Processor instance to register
            
        Returns:
            True if registered successfully, False otherwise
        """
        try:
            if processor.name in self._processors:
                logger.warning(f"Processor '{processor.name}' already registered, overwriting")
            
            self._processors[processor.name] = processor
            self._processor_types[processor.processor_type].append(processor.name)
            
            logger.info(f"Registered processor: {processor}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register processor {processor.name}: {e}")
            return False
    
    def unregister_processor(self, name: str) -> bool:
        """
        Unregister a processor from the registry
        
        Args:
            name: Name of the processor to unregister
            
        Returns:
            True if unregistered successfully, False otherwise
        """
        try:
            if name not in self._processors:
                logger.warning(f"Processor '{name}' not found in registry")
                return False
            
            processor = self._processors[name]
            del self._processors[name]
            
            if name in self._processor_types[processor.processor_type]:
                self._processor_types[processor.processor_type].remove(name)
            
            logger.info(f"Unregistered processor: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister processor {name}: {e}")
            return False
    
    def get_processor(self, name: str) -> Optional[BaseProcessor]:
        """
        Get a processor by name
        
        Args:
            name: Name of the processor
            
        Returns:
            Processor instance or None if not found
        """
        return self._processors.get(name)
    
    def get_processors_by_type(self, processor_type: ProcessorType) -> List[BaseProcessor]:
        """
        Get all processors of a specific type
        
        Args:
            processor_type: Type of processors to retrieve
            
        Returns:
            List of processor instances
        """
        processor_names = self._processor_types.get(processor_type, [])
        return [self._processors[name] for name in processor_names if name in self._processors]
    
    def list_processors(self) -> Dict[str, Dict[str, Any]]:
        """
        List all registered processors with their information
        
        Returns:
            Dictionary of processor information
        """
        return {
            name: {
                "type": processor.processor_type.value,
                "ai_provider": processor.ai_provider.value,
                "enabled": processor.enabled,
                "capabilities": processor.get_capabilities(),
                "requirements": processor.get_requirements()
            }
            for name, processor in self._processors.items()
        }
    
    async def health_check_all(self) -> Dict[str, bool]:
        """
        Perform health check on all registered processors
        
        Returns:
            Dictionary with processor names and their health status
        """
        health_results = {}
        
        for name, processor in self._processors.items():
            try:
                health_results[name] = await processor.health_check()
            except Exception as e:
                logger.error(f"Health check failed for processor {name}: {e}")
                health_results[name] = False
        
        return health_results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics
        
        Returns:
            Dictionary with registry statistics
        """
        return {
            "total_processors": len(self._processors),
            "processors_by_type": {
                processor_type.value: len(processors)
                for processor_type, processors in self._processor_types.items()
            },
            "enabled_processors": len([p for p in self._processors.values() if p.enabled]),
            "ai_providers": list(set(p.ai_provider.value for p in self._processors.values()))
        }


class PluginInterface(ABC):
    """
    Interface for plugins that extend the multimodal agent functionality
    
    Plugins can add new processors, AI providers, or other capabilities
    to the system without modifying the core code.
    """
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the plugin name"""
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        """Get the plugin version"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get the plugin description"""
        pass
    
    @abstractmethod
    async def initialize(self, registry: ProcessorRegistry) -> bool:
        """
        Initialize the plugin and register its components
        
        Args:
            registry: Processor registry to register components
            
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """
        Cleanup plugin resources
        
        Returns:
            True if cleanup successful, False otherwise
        """
        pass
    
    def get_dependencies(self) -> List[str]:
        """
        Get list of plugin dependencies
        
        Returns:
            List of required dependencies
        """
        return []
    
    def get_requirements(self) -> Dict[str, Any]:
        """
        Get plugin requirements
        
        Returns:
            Dictionary of requirements
        """
        return {
            "name": self.get_name(),
            "version": self.get_version(),
            "dependencies": self.get_dependencies()
        }


# Global processor registry instance
global_processor_registry = ProcessorRegistry() 