"""Tests for the inference module."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from bsort.config import Config
from bsort.infer import (
    non_max_suppression,
    postprocess_output,
    preprocess_image,
    run_inference,
    save_annotated_image,
)


# ====================================================================
# Tests for preprocess_image()
# ====================================================================


def test_preprocess_image_correct_shape():
    """Test that preprocess_image returns correct output shape."""
    # Create a 640x480 BGR image
    image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

    # Preprocess
    preprocessed, original_shape = preprocess_image(image, input_size=224)

    # Verify output shape is (1, 3, 224, 224)
    assert preprocessed.shape == (1, 3, 224, 224)
    # Verify dtype is float32
    assert preprocessed.dtype == np.float32


def test_preprocess_image_normalization():
    """Test that preprocess_image normalizes values to [0, 1] range."""
    # Create a test image
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    # Preprocess
    preprocessed, _ = preprocess_image(image, input_size=224)

    # Verify values are in [0, 1] range
    assert preprocessed.min() >= 0.0
    assert preprocessed.max() <= 1.0


def test_preprocess_image_color_conversion():
    """Test that preprocess_image converts BGR to RGB."""
    # Create an image with distinct BGR values
    # Blue channel = 100, Green = 150, Red = 200
    image = np.full((50, 50, 3), [100, 150, 200], dtype=np.uint8)

    # Preprocess
    preprocessed, _ = preprocess_image(image, input_size=224)

    # After BGR->RGB conversion:
    # Channel 0 (R) should have values from original Red channel (200/255 ≈ 0.78)
    # Channel 1 (G) should have values from original Green channel (150/255 ≈ 0.59)
    # Channel 2 (B) should have values from original Blue channel (100/255 ≈ 0.39)

    # Extract a sample pixel
    pixel = preprocessed[0, :, 0, 0]  # First pixel, all channels

    # Verify approximate RGB values (with tolerance for resizing interpolation)
    assert pytest.approx(pixel[0], abs=0.05) == 200 / 255  # R channel
    assert pytest.approx(pixel[1], abs=0.05) == 150 / 255  # G channel
    assert pytest.approx(pixel[2], abs=0.05) == 100 / 255  # B channel


def test_preprocess_image_returns_original_shape():
    """Test that preprocess_image returns original shape tuple."""
    # Create a 480x640 image
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Preprocess
    _, original_shape = preprocess_image(image)

    # Verify second return value is (height, width)
    assert original_shape == (480, 640)
    assert isinstance(original_shape, tuple)


# ====================================================================
# Tests for non_max_suppression()
# ====================================================================


def test_nms_empty_input():
    """Test NMS with empty input arrays."""
    boxes = np.array([])
    scores = np.array([])
    classes = np.array([])

    filtered_boxes, filtered_scores, filtered_classes = non_max_suppression(
        boxes, scores, classes
    )

    # Verify returns empty arrays
    assert len(filtered_boxes) == 0
    assert len(filtered_scores) == 0
    assert len(filtered_classes) == 0


def test_nms_score_threshold_filtering():
    """Test that NMS filters boxes by score threshold."""
    # Create 3 boxes with different confidence scores
    boxes = np.array([[10, 10, 50, 50], [60, 60, 100, 100], [110, 110, 150, 150]])
    scores = np.array([0.9, 0.3, 0.1])  # Only first box above threshold
    classes = np.array([0, 0, 0])

    filtered_boxes, filtered_scores, filtered_classes = non_max_suppression(
        boxes, scores, classes, score_threshold=0.5
    )

    # Only the box with score >= 0.5 should remain
    assert len(filtered_boxes) == 1
    assert len(filtered_scores) == 1
    assert filtered_scores[0] == 0.9


def test_nms_removes_overlapping_boxes():
    """Test that NMS removes overlapping boxes with high IoU."""
    # Create two heavily overlapping boxes (same position, different scores)
    boxes = np.array([[10, 10, 50, 50], [12, 12, 52, 52]])  # Very similar boxes
    scores = np.array([0.9, 0.8])  # First box has higher confidence
    classes = np.array([0, 0])  # Same class

    filtered_boxes, filtered_scores, filtered_classes = non_max_suppression(
        boxes, scores, classes, iou_threshold=0.45, score_threshold=0.25
    )

    # Only highest confidence box should remain
    assert len(filtered_boxes) == 1
    assert filtered_scores[0] == 0.9


def test_nms_per_class_separation():
    """Test that NMS is applied separately per class."""
    # Create two overlapping boxes from different classes
    boxes = np.array([[10, 10, 50, 50], [12, 12, 52, 52]])
    scores = np.array([0.9, 0.8])
    classes = np.array([0, 1])  # Different classes

    filtered_boxes, filtered_scores, filtered_classes = non_max_suppression(
        boxes, scores, classes, iou_threshold=0.45, score_threshold=0.25
    )

    # Both boxes should remain (different classes, NMS applied separately)
    assert len(filtered_boxes) == 2
    assert len(np.unique(filtered_classes)) == 2  # Two different classes


# ====================================================================
# Tests for postprocess_output()
# ====================================================================


def test_postprocess_no_detections():
    """Test postprocess with low confidence scores (no detections)."""
    # Create output with low confidence scores
    # Format: (1, 7, N) where 7 = 4 bbox + 3 class probs
    output = np.zeros((1, 7, 10))  # All zeros = low confidence

    boxes, scores, classes = postprocess_output(
        output, original_shape=(480, 640), conf_threshold=0.25
    )

    # Should return empty arrays
    assert len(boxes) == 0
    assert len(scores) == 0
    assert len(classes) == 0


def test_postprocess_single_detection():
    """Test postprocess with a single valid detection."""
    # Create output with one valid detection
    # Format: (1, 7, N) where 7 = 4 bbox (xywh) + 3 class probabilities
    # Use at least 10 anchors so transpose logic works (7 < 10)
    output = np.zeros((1, 7, 10))

    # Set bbox for first anchor: center at (112, 112), size (50, 50) in 224x224 input
    output[0, 0, 0] = 112  # x center
    output[0, 1, 0] = 112  # y center
    output[0, 2, 0] = 50  # width
    output[0, 3, 0] = 50  # height

    # Set class probabilities (class 0 has highest)
    output[0, 4, 0] = 0.9  # class 0
    output[0, 5, 0] = 0.05  # class 1
    output[0, 6, 0] = 0.05  # class 2

    # Other anchors have low confidence (will be filtered out)
    # Leave them as zeros

    boxes, scores, classes = postprocess_output(
        output, original_shape=(224, 224), input_size=224, conf_threshold=0.25
    )

    # Verify one detection (other anchors filtered by confidence)
    assert len(boxes) == 1
    assert len(scores) == 1
    assert len(classes) == 1

    # Verify class is 0 (highest probability)
    assert classes[0] == 0

    # Verify score is approximately 0.9
    assert pytest.approx(scores[0], abs=0.01) == 0.9


def test_postprocess_bbox_clipping():
    """Test that postprocess clips bboxes to image boundaries."""
    # Create output with bbox extending beyond image bounds
    # Use at least 10 anchors so transpose logic works (7 < 10)
    output = np.zeros((1, 7, 10))

    # Set bbox that extends beyond image (centered at edge)
    output[0, 0, 0] = 10  # x center (near edge)
    output[0, 1, 0] = 10  # y center (near edge)
    output[0, 2, 0] = 40  # width (extends beyond 0)
    output[0, 3, 0] = 40  # height (extends beyond 0)

    # High confidence
    output[0, 4, 0] = 0.95

    # Other anchors have low confidence (will be filtered out)
    # Leave them as zeros

    boxes, scores, classes = postprocess_output(
        output,
        original_shape=(100, 100),  # Small image
        input_size=224,
        conf_threshold=0.25,
    )

    # Verify bbox is clipped to [0, width] and [0, height]
    box = boxes[0]
    assert box[0] >= 0  # x1 >= 0
    assert box[1] >= 0  # y1 >= 0
    assert box[2] <= 100  # x2 <= width
    assert box[3] <= 100  # y2 <= height


# ====================================================================
# Tests for save_annotated_image()
# ====================================================================


def test_save_annotated_image_missing_file():
    """Test save_annotated_image with non-existent image file."""
    non_existent_path = Path("/tmp/nonexistent_image_123456.jpg")
    output_path = Path("/tmp/output_123456.jpg")
    detections = []

    # Should raise ValueError
    with pytest.raises(ValueError, match="Failed to load image"):
        save_annotated_image(non_existent_path, output_path, detections)


def test_save_annotated_image_creates_output(temp_image_file):
    """Test that save_annotated_image creates output file."""
    # Use NamedTemporaryFile instead of deprecated mktemp
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        output_path = Path(f.name)

    try:
        # Create sample detection
        detections = [
            {
                "bbox": [10, 10, 50, 50],
                "class_id": 0,
                "class_name": "Light Blue",
                "confidence": 0.95,
            }
        ]

        # Save annotated image
        save_annotated_image(temp_image_file, output_path, detections)

        # Verify output file exists
        assert output_path.exists()

        # Verify file has content (size > 0)
        assert output_path.stat().st_size > 0

    finally:
        # Cleanup
        if output_path.exists():
            output_path.unlink()


# ====================================================================
# Tests for run_inference()
# ====================================================================


def test_run_inference_missing_image(sample_config):
    """Test run_inference with non-existent image file."""
    cfg = Config(sample_config, Path("/tmp/test_config.yaml"))
    non_existent_image = Path("/tmp/nonexistent_image_987654.jpg")

    # Should raise FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Image file not found"):
        run_inference(cfg, non_existent_image)


def test_run_inference_missing_model(sample_config, temp_image_file):
    """Test run_inference with non-existent model file."""
    cfg = Config(sample_config, Path("/tmp/test_config.yaml"))

    # Ensure model path doesn't exist
    cfg.data["inference"]["model_path"] = "/tmp/nonexistent_model_987654.onnx"

    # Should raise FileNotFoundError with helpful message
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        run_inference(cfg, temp_image_file)
