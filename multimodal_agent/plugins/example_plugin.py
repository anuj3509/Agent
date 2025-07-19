"""
Example Plugin for the Multimodal AI Agent

This plugin demonstrates how to create custom processors and extend
the agent's functionality using the plugin system.

Author: Anuj Patel (amp10162@nyu.edu)
Website: panuj.com
"""
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path

from ..core.base import (
    PluginInterface, BaseProcessor, ProcessorRegistry,
    ProcessorType, AIProvider, ProcessingResult, ProcessorConfig
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ExampleProcessor(BaseProcessor[str, str]):
    """
    Example processor that demonstrates the base processor interface
    
    This processor simply echoes the input with some transformation.
    """
    
    def __init__(self, config: ProcessorConfig):
        """Initialize the example processor"""
        super().__init__(config)
        self.transformation_type = config.settings.get("transformation", "uppercase")
    
    async def process(self, input_data: str) -> ProcessingResult:
        """
        Process input data by applying a simple transformation
        
        Args:
            input_data: Input string to process
            
        Returns:
            ProcessingResult with transformed data
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Validate input
            if not self.validate_input(input_data):
                return ProcessingResult(
                    success=False,
                    error="Invalid input data"
                )
            
            # Apply transformation based on configuration
            if self.transformation_type == "uppercase":
                result = input_data.upper()
            elif self.transformation_type == "lowercase":
                result = input_data.lower()
            elif self.transformation_type == "reverse":
                result = input_data[::-1]
            elif self.transformation_type == "word_count":
                word_count = len(input_data.split())
                result = f"Word count: {word_count}"
            else:
                result = input_data  # No transformation
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return ProcessingResult(
                success=True,
                data={"transformed_text": result, "original_text": input_data},
                metadata={
                    "transformation": self.transformation_type,
                    "input_length": len(input_data),
                    "output_length": len(result)
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error in example processor: {e}")
            return ProcessingResult(
                success=False,
                error=str(e)
            )
    
    def validate_input(self, input_data: str) -> bool:
        """
        Validate input data
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        return isinstance(input_data, str) and len(input_data) > 0
    
    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this processor supports"""
        return [
            "text_transformation",
            "uppercase_conversion",
            "lowercase_conversion", 
            "text_reversal",
            "word_counting"
        ]
    
    def get_requirements(self) -> Dict[str, Any]:
        """Get requirements for this processor"""
        requirements = super().get_requirements()
        requirements.update({
            "input_type": "string",
            "output_type": "string",
            "transformation_types": [
                "uppercase", "lowercase", "reverse", "word_count"
            ]
        })
        return requirements


class ExamplePlugin(PluginInterface):
    """
    Example plugin that demonstrates the plugin interface
    
    This plugin registers the ExampleProcessor and shows how to
    extend the agent's functionality.
    """
    
    def get_name(self) -> str:
        """Get the plugin name"""
        return "example_plugin"
    
    def get_version(self) -> str:
        """Get the plugin version"""
        return "1.0.0"
    
    def get_description(self) -> str:
        """Get the plugin description"""
        return "Example plugin demonstrating text transformation capabilities"
    
    async def initialize(self, registry: ProcessorRegistry) -> bool:
        """
        Initialize the plugin and register its components
        
        Args:
            registry: Processor registry to register components
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info(f"Initializing {self.get_name()} v{self.get_version()}")
            
            # Create different configurations for the example processor
            configs = [
                ProcessorConfig(
                    name="example_uppercase",
                    processor_type=ProcessorType.CUSTOM,
                    ai_provider=AIProvider.CUSTOM,
                    model_config={},
                    settings={"transformation": "uppercase"},
                    enabled=True
                ),
                ProcessorConfig(
                    name="example_lowercase",
                    processor_type=ProcessorType.CUSTOM,
                    ai_provider=AIProvider.CUSTOM,
                    model_config={},
                    settings={"transformation": "lowercase"},
                    enabled=True
                ),
                ProcessorConfig(
                    name="example_reverse",
                    processor_type=ProcessorType.CUSTOM,
                    ai_provider=AIProvider.CUSTOM,
                    model_config={},
                    settings={"transformation": "reverse"},
                    enabled=True
                ),
                ProcessorConfig(
                    name="example_word_count",
                    processor_type=ProcessorType.CUSTOM,
                    ai_provider=AIProvider.CUSTOM,
                    model_config={},
                    settings={"transformation": "word_count"},
                    enabled=True
                )
            ]
            
            # Register all configurations
            for config in configs:
                processor = ExampleProcessor(config)
                success = registry.register_processor(processor)
                if not success:
                    logger.error(f"Failed to register processor: {config.name}")
                    return False
            
            logger.info(f"Successfully registered {len(configs)} example processors")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing example plugin: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """
        Cleanup plugin resources
        
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            logger.info(f"Cleaning up {self.get_name()}")
            # In a real plugin, you might need to:
            # - Close database connections
            # - Release file handles
            # - Cancel running tasks
            # - Unregister processors
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up example plugin: {e}")
            return False
    
    def get_dependencies(self) -> List[str]:
        """Get list of plugin dependencies"""
        return []  # This example plugin has no dependencies
    
    async def health_check(self) -> bool:
        """
        Perform health check on the plugin
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Simple health check - verify we can create a processor
            config = ProcessorConfig(
                name="health_check_processor",
                processor_type=ProcessorType.CUSTOM,
                ai_provider=AIProvider.CUSTOM,
                model_config={},
                settings={"transformation": "uppercase"}
            )
            
            processor = ExampleProcessor(config)
            test_result = await processor.process("health check")
            
            return test_result.success
            
        except Exception as e:
            logger.error(f"Health check failed for example plugin: {e}")
            return False 