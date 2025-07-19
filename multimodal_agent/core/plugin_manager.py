"""
Plugin Manager for the Multimodal AI Agent

This module provides a plugin management system that allows for dynamic loading
and management of plugins to extend the agent's functionality.

Author: Anuj Patel (amp10162@nyu.edu)
Website: panuj.com
"""
import asyncio
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Type

from .base import PluginInterface, ProcessorRegistry, global_processor_registry
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PluginManager:
    """
    Plugin manager for loading and managing plugins
    
    This class provides functionality to dynamically load, initialize,
    and manage plugins that extend the multimodal agent's capabilities.
    """
    
    def __init__(self, registry: ProcessorRegistry = None):
        """
        Initialize the plugin manager
        
        Args:
            registry: Processor registry to use (defaults to global registry)
        """
        self.registry = registry or global_processor_registry
        self._plugins: Dict[str, PluginInterface] = {}
        self._plugin_modules: Dict[str, Any] = {}
        self._plugin_paths: List[Path] = []
        
        # Default plugin paths
        self._default_plugin_paths = [
            Path(__file__).parent.parent / "plugins",
            Path.cwd() / "plugins",
            Path.home() / ".multimodal_agent" / "plugins"
        ]
        
        self._plugin_paths.extend(self._default_plugin_paths)
        
        logger.info("Plugin manager initialized")
    
    def add_plugin_path(self, path: Path):
        """
        Add a path to search for plugins
        
        Args:
            path: Path to add to plugin search paths
        """
        if path not in self._plugin_paths:
            self._plugin_paths.append(path)
            logger.info(f"Added plugin path: {path}")
    
    def remove_plugin_path(self, path: Path):
        """
        Remove a path from plugin search paths
        
        Args:
            path: Path to remove from plugin search paths
        """
        if path in self._plugin_paths:
            self._plugin_paths.remove(path)
            logger.info(f"Removed plugin path: {path}")
    
    def discover_plugins(self) -> List[str]:
        """
        Discover available plugins in plugin paths
        
        Returns:
            List of discovered plugin names
        """
        discovered_plugins = []
        
        for plugin_path in self._plugin_paths:
            if not plugin_path.exists():
                continue
                
            logger.info(f"Searching for plugins in: {plugin_path}")
            
            # Look for Python files
            for file_path in plugin_path.glob("*.py"):
                if file_path.name.startswith("__"):
                    continue
                    
                plugin_name = file_path.stem
                discovered_plugins.append(plugin_name)
                logger.debug(f"Discovered plugin: {plugin_name}")
            
            # Look for plugin packages (directories with __init__.py)
            for dir_path in plugin_path.iterdir():
                if dir_path.is_dir() and (dir_path / "__init__.py").exists():
                    plugin_name = dir_path.name
                    discovered_plugins.append(plugin_name)
                    logger.debug(f"Discovered plugin package: {plugin_name}")
        
        logger.info(f"Discovered {len(discovered_plugins)} plugins")
        return discovered_plugins
    
    def load_plugin(self, plugin_name: str, plugin_path: Path = None) -> bool:
        """
        Load a plugin by name
        
        Args:
            plugin_name: Name of the plugin to load
            plugin_path: Specific path to the plugin (optional)
            
        Returns:
            True if plugin loaded successfully, False otherwise
        """
        try:
            if plugin_name in self._plugins:
                logger.warning(f"Plugin '{plugin_name}' already loaded")
                return True
            
            # Find plugin file
            plugin_file = None
            if plugin_path:
                plugin_file = plugin_path
            else:
                for path in self._plugin_paths:
                    potential_file = path / f"{plugin_name}.py"
                    potential_package = path / plugin_name / "__init__.py"
                    
                    if potential_file.exists():
                        plugin_file = potential_file
                        break
                    elif potential_package.exists():
                        plugin_file = potential_package
                        break
            
            if not plugin_file:
                logger.error(f"Plugin file not found for: {plugin_name}")
                return False
            
            # Load the module
            spec = importlib.util.spec_from_file_location(plugin_name, plugin_file)
            if not spec or not spec.loader:
                logger.error(f"Could not create spec for plugin: {plugin_name}")
                return False
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find plugin class
            plugin_class = None
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, PluginInterface) and 
                    obj != PluginInterface and 
                    obj.__module__ == module.__name__):
                    plugin_class = obj
                    break
            
            if not plugin_class:
                logger.error(f"No plugin class found in: {plugin_name}")
                return False
            
            # Instantiate and store plugin
            plugin_instance = plugin_class()
            self._plugins[plugin_name] = plugin_instance
            self._plugin_modules[plugin_name] = module
            
            logger.info(f"Loaded plugin: {plugin_name} ({plugin_instance.get_version()})")
            return True
            
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {e}")
            return False
    
    async def initialize_plugin(self, plugin_name: str) -> bool:
        """
        Initialize a loaded plugin
        
        Args:
            plugin_name: Name of the plugin to initialize
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            if plugin_name not in self._plugins:
                logger.error(f"Plugin not loaded: {plugin_name}")
                return False
            
            plugin = self._plugins[plugin_name]
            
            # Check dependencies
            dependencies = plugin.get_dependencies()
            if dependencies:
                logger.info(f"Checking dependencies for {plugin_name}: {dependencies}")
                # Here you could add dependency checking logic
            
            # Initialize the plugin
            success = await plugin.initialize(self.registry)
            
            if success:
                logger.info(f"Initialized plugin: {plugin_name}")
            else:
                logger.error(f"Failed to initialize plugin: {plugin_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error initializing plugin {plugin_name}: {e}")
            return False
    
    async def load_and_initialize_plugin(self, plugin_name: str, plugin_path: Path = None) -> bool:
        """
        Load and initialize a plugin in one step
        
        Args:
            plugin_name: Name of the plugin
            plugin_path: Specific path to the plugin (optional)
            
        Returns:
            True if both loading and initialization successful, False otherwise
        """
        if self.load_plugin(plugin_name, plugin_path):
            return await self.initialize_plugin(plugin_name)
        return False
    
    async def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin
        
        Args:
            plugin_name: Name of the plugin to unload
            
        Returns:
            True if unloading successful, False otherwise
        """
        try:
            if plugin_name not in self._plugins:
                logger.warning(f"Plugin not loaded: {plugin_name}")
                return True
            
            plugin = self._plugins[plugin_name]
            
            # Cleanup plugin
            await plugin.cleanup()
            
            # Remove from registry and cleanup
            del self._plugins[plugin_name]
            
            if plugin_name in self._plugin_modules:
                del self._plugin_modules[plugin_name]
            
            # Remove from sys.modules if it exists
            module_name = f"plugin_{plugin_name}"
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            logger.info(f"Unloaded plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False
    
    async def load_all_plugins(self) -> Dict[str, bool]:
        """
        Load and initialize all discovered plugins
        
        Returns:
            Dictionary with plugin names and their loading status
        """
        results = {}
        discovered = self.discover_plugins()
        
        for plugin_name in discovered:
            try:
                success = await self.load_and_initialize_plugin(plugin_name)
                results[plugin_name] = success
            except Exception as e:
                logger.error(f"Error loading plugin {plugin_name}: {e}")
                results[plugin_name] = False
        
        successful = len([r for r in results.values() if r])
        logger.info(f"Loaded {successful}/{len(discovered)} plugins successfully")
        
        return results
    
    def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        """
        List all loaded plugins with their information
        
        Returns:
            Dictionary with plugin information
        """
        return {
            name: {
                "version": plugin.get_version(),
                "description": plugin.get_description(),
                "dependencies": plugin.get_dependencies(),
                "requirements": plugin.get_requirements()
            }
            for name, plugin in self._plugins.items()
        }
    
    def get_plugin(self, plugin_name: str) -> Optional[PluginInterface]:
        """
        Get a plugin instance by name
        
        Args:
            plugin_name: Name of the plugin
            
        Returns:
            Plugin instance or None if not found
        """
        return self._plugins.get(plugin_name)
    
    async def reload_plugin(self, plugin_name: str) -> bool:
        """
        Reload a plugin
        
        Args:
            plugin_name: Name of the plugin to reload
            
        Returns:
            True if reload successful, False otherwise
        """
        try:
            # Store plugin path before unloading
            plugin_path = None
            if plugin_name in self._plugin_modules:
                module = self._plugin_modules[plugin_name]
                if hasattr(module, '__file__'):
                    plugin_path = Path(module.__file__)
            
            # Unload the plugin
            await self.unload_plugin(plugin_name)
            
            # Reload the plugin
            return await self.load_and_initialize_plugin(plugin_name, plugin_path)
            
        except Exception as e:
            logger.error(f"Error reloading plugin {plugin_name}: {e}")
            return False
    
    async def health_check_plugins(self) -> Dict[str, bool]:
        """
        Perform health check on all loaded plugins
        
        Returns:
            Dictionary with plugin names and their health status
        """
        health_results = {}
        
        for name, plugin in self._plugins.items():
            try:
                # Check if plugin has a health check method
                if hasattr(plugin, 'health_check'):
                    health_results[name] = await plugin.health_check()
                else:
                    # Default health check - just verify the plugin exists
                    health_results[name] = True
            except Exception as e:
                logger.error(f"Health check failed for plugin {name}: {e}")
                health_results[name] = False
        
        return health_results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get plugin manager statistics
        
        Returns:
            Dictionary with plugin manager statistics
        """
        return {
            "total_plugins": len(self._plugins),
            "plugin_paths": [str(p) for p in self._plugin_paths],
            "loaded_plugins": list(self._plugins.keys()),
            "registry_stats": self.registry.get_stats() if self.registry else None
        }


# Global plugin manager instance
global_plugin_manager = PluginManager() 