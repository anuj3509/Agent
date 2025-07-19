"""
Multimodal AI Agent for Content Creation and Analysis

A comprehensive AI agent that combines LangChain and LangGraph to create
multimodal content including video captions, interactive narratives, and
targeted marketing content.
"""

__version__ = "0.1.0"
__author__ = "Multimodal AI Agent Team"

from .core.agent import MultimodalAgent
from .config.settings import settings

__all__ = ["MultimodalAgent", "settings"] 