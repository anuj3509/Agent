"""
Multimodal AI Agent for Content Creation and Analysis

A comprehensive AI agent that combines LangChain and LangGraph to create
multimodal content including video captions, interactive narratives, and
targeted marketing content.

Author: Anuj Patel (amp10162@nyu.edu)
Website: panuj.com
"""

__version__ = "0.1.0"
__author__ = "Anuj Patel"
__email__ = "amp10162@nyu.edu"
__website__ = "panuj.com"

from .core.agent import MultimodalAgent
from .core.base import (
    BaseProcessor, BaseAIProvider, ProcessorRegistry, PluginInterface,
    ProcessorType, AIProvider, ProcessingResult, ProcessorConfig,
    global_processor_registry
)
from .config.settings import settings
from .providers import OpenAIProvider, GeminiProvider

__all__ = [
    "MultimodalAgent", 
    "settings",
    "BaseProcessor",
    "BaseAIProvider",
    "ProcessorRegistry", 
    "PluginInterface",
    "ProcessorType",
    "AIProvider",
    "ProcessingResult",
    "ProcessorConfig",
    "global_processor_registry",
    "OpenAIProvider",
    "GeminiProvider"
] 