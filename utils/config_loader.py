#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Config Loader Module
------------------
This module provides utilities for loading configuration files.
"""

import os
import yaml
import json
from typing import Any, Dict, Optional, Union


class ConfigLoader:
    """
    Utility class for loading configuration files.
    """

    @staticmethod
    def load_yaml(path: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file.

        Args:
            path: Path to the YAML file

        Returns:
            Dictionary with configuration values
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    @staticmethod
    def load_json(path: str) -> Dict[str, Any]:
        """
        Load a JSON configuration file.

        Args:
            path: Path to the JSON file

        Returns:
            Dictionary with configuration values
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r') as f:
            config = json.load(f)

        return config

    @staticmethod
    def load(path: str) -> Dict[str, Any]:
        """
        Load a configuration file based on its extension.

        Args:
            path: Path to the configuration file

        Returns:
            Dictionary with configuration values

        Raises:
            ValueError: If the file extension is not supported
        """
        if path.endswith('.yaml') or path.endswith('.yml'):
            return ConfigLoader.load_yaml(path)
        elif path.endswith('.json'):
            return ConfigLoader.load_json(path)
        else:
            raise ValueError(f"Unsupported config file extension: {path}")


def load_config(path: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load a configuration file.
    
    Args:
        path: Path to the configuration file
        default: Default configuration to return if loading fails
        
    Returns:
        Dictionary with configuration values
    """
    try:
        return ConfigLoader.load(path)
    except Exception as e:
        if default is not None:
            return default
        raise e
