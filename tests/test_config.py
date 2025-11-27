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
