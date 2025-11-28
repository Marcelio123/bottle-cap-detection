"""Shared test fixtures for bsort tests."""

import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import Mock

import cv2
import numpy as np
import pytest


@pytest.fixture
def temp_image():
    """Create a temporary test image as numpy array.

    Returns:
        np.ndarray: BGR image (10x10x3) with uint8 dtype
    """
    # Create a small 10x10 test image
    image = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    return image


@pytest.fixture
def temp_image_file(temp_image):
    """Create a temporary image file on disk.

    Args:
        temp_image: Numpy array from temp_image fixture

    Returns:
        Path: Path to temporary image file

    Note:
        File is automatically cleaned up after test
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        temp_path = Path(f.name)

    # Write image to file
    cv2.imwrite(str(temp_path), temp_image)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def sample_config() -> Dict:
    """Create a sample configuration dictionary for testing.

    Returns:
        Dict: Sample configuration with inference parameters
    """
    return {
        "inference": {
            "model_path": "model.onnx",
            "confidence_threshold": 0.25,
            "iou_threshold": 0.45,
            "imgsz": 224,
            "class_names": ["Light Blue", "Dark Blue", "Others"]
        }
    }


@pytest.fixture
def temp_dir():
    """Create a temporary directory for file operations.

    Returns:
        Path: Path to temporary directory

    Note:
        Directory is automatically cleaned up after test
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_onnx_session(mocker):
    """Create a mock ONNX Runtime inference session.

    Args:
        mocker: pytest-mock fixture

    Returns:
        Mock: Mocked ONNX InferenceSession
    """
    # Create mock session
    mock_session = Mock()

    # Mock input details
    mock_input = Mock()
    mock_input.name = "images"
    mock_input.shape = [1, 3, 224, 224]
    mock_session.get_inputs.return_value = [mock_input]

    # Mock providers
    mock_session.get_providers.return_value = ["CPUExecutionProvider"]

    # Mock run method - returns dummy output
    # YOLO output format: (1, 4+num_classes, N_anchors)
    # For 3 classes: (1, 7, 100) where 7 = 4 bbox + 3 classes
    dummy_output = np.random.rand(1, 7, 100).astype(np.float32)
    mock_session.run.return_value = [dummy_output]

    # Mock the InferenceSession constructor
    mock_ort = mocker.patch("onnxruntime.InferenceSession", return_value=mock_session)

    return mock_session
