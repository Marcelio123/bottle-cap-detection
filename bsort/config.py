"""Configuration module for bsort.

This module handles loading and validation of YAML configuration files.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


class Config:
    """Configuration container for bsort settings.

    Attributes:
        data: Dictionary containing all configuration parameters.
        config_path: Path to the configuration file.
    """

    def __init__(self, data: Dict[str, Any], config_path: Path) -> None:
        """Initialize Config object.

        Args:
            data: Configuration dictionary loaded from YAML.
            config_path: Path to the configuration file.
        """
        self.data = data
        self.config_path = config_path

    def __getitem__(self, key: str) -> Any:
        """Get configuration value by key.

        Args:
            key: Configuration key.

        Returns:
            Configuration value.
        """
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default fallback.

        Args:
            key: Configuration key.
            default: Default value if key not found.

        Returns:
            Configuration value or default.
        """
        return self.data.get(key, default)


def load_config(config_path: Path) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Config object containing the loaded configuration.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML file is malformed.
    """
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.info(f"Loading configuration from: {config_path}")

    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML file: {e}")
        raise

    if data is None:
        data = {}

    logger.info("Configuration loaded successfully")
    return Config(data, config_path)
