"""Tests for configuration module."""

import tempfile
from pathlib import Path

import pytest
import yaml

from bsort.config import load_config


def test_load_config_valid_file():
    """Test loading a valid configuration file."""
    config_data = {
        "model": "yolo11n.pt",
        "data": "sample/data.yaml",
        "epochs": 100,
        "imgsz": 640,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    try:
        config = load_config(Path(config_path))
        assert config["model"] == "yolo11n.pt"
        assert config["epochs"] == 100
        assert config["imgsz"] == 640
    finally:
        Path(config_path).unlink()


def test_load_config_file_not_found():
    """Test loading a non-existent configuration file."""
    with pytest.raises(FileNotFoundError):
        load_config(Path("nonexistent_config.yaml"))


def test_load_config_malformed_yaml():
    """Test loading a malformed YAML file."""
    malformed_yaml = """
    key1: value1
    key2: [unclosed list
    key3: value3
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(malformed_yaml)
        config_path = f.name

    try:
        with pytest.raises(yaml.YAMLError):
            load_config(Path(config_path))
    finally:
        Path(config_path).unlink()


def test_load_config_empty_file():
    """Test loading an empty YAML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        # Write nothing (empty file)
        config_path = f.name

    try:
        config = load_config(Path(config_path))
        # Empty file should return empty dict
        assert config.data == {}
    finally:
        Path(config_path).unlink()


def test_config_get_with_default():
    """Test Config.get() method with default value."""
    config_data = {"existing_key": "value"}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    try:
        config = load_config(Path(config_path))

        # Test getting existing key
        assert config.get("existing_key") == "value"

        # Test getting non-existent key with default
        assert config.get("missing_key", "default_value") == "default_value"

        # Test getting non-existent key without default (should return None)
        assert config.get("missing_key") is None
    finally:
        Path(config_path).unlink()
