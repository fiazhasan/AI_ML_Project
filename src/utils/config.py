"""
Configuration management module
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration from YAML file
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_paths()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {self.config_path}")
        return config
    
    def _validate_paths(self):
        """Create necessary directories if they don't exist"""
        paths = self.config.get('paths', {})
        
        for key, path in paths.items():
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # Also create data directories
        Path(self.config['data']['data_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['data']['processed_dir']).mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key: Configuration key (e.g., 'data.batch_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access"""
        return self.config[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists"""
        return key in self.config


# Global config instance
_config_instance = None


def get_config(config_path: str = "config.yaml") -> Config:
    """
    Get global configuration instance
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance
