"""
Core components of the multimodal agent
"""

from .agent import MultimodalAgent, AgentState
from .base import (
    BaseProcessor, BaseAIProvider, ProcessorRegistry, PluginInterface,
    ProcessorType, AIProvider, ProcessingResult, ProcessorConfig,
    global_processor_registry
)

__all__ = [
    "MultimodalAgent", 
    "AgentState",
    "BaseProcessor",
    "BaseAIProvider", 
    "ProcessorRegistry",
    "PluginInterface",
    "ProcessorType",
    "AIProvider",
    "ProcessingResult",
    "ProcessorConfig",
    "global_processor_registry"
] 