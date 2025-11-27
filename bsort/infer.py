"""Inference module for bsort.

This module contains functions for running inference with ONNX models.
Supports YOLO object detection models exported to ONNX format.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision

from bsort.config import Config

logger = logging.getLogger(__name__)

# Default class names (Light Blue, Dark Blue, Others)
DEFAULT_CLASS_NAMES = ["Light Blue", "Dark Blue", "Others"]


def preprocess_image(
    image: np.ndarray, input_size: int = 224
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Preprocess image for YOLO inference.

    Args:
        image: Input image in BGR format (OpenCV format)
        input_size: Target input size for the model

    Returns:
        Tuple containing:
        - Preprocessed image tensor (1, 3, H, W) with float32 dtype
        - Original image shape (height, width) before preprocessing
    """
    original_shape = image.shape[:2]  # (height, width)

    # Resize image while maintaining aspect ratio
    # YOLO expects square input, so we'll resize to (input_size, input_size)
    resized = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1] and convert to float32
    normalized = rgb.astype(np.float32) / 255.0

    # Transpose from HWC to CHW format
    chw = np.transpose(normalized, (2, 0, 1))

    # Add batch dimension (1, 3, H, W)
    batched = np.expand_dims(chw, axis=0)

    return batched, original_shape


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_threshold: float = 0.45,
    score_threshold: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply Non-Maximum Suppression to filter overlapping detections.

    Args:
        boxes: Bounding boxes in format [x1, y1, x2, y2] (N, 4)
        scores: Confidence scores (N,)
        classes: Class IDs (N,)
        iou_threshold: IoU threshold for NMS
        score_threshold: Minimum confidence threshold

    Returns:
        Tuple of filtered (boxes, scores, classes)
    """
    # Debug logging
    logger.debug(f"[NMS] Input: {len(boxes)} boxes, score_threshold={score_threshold:.4f}")
    if len(scores) > 0:
        logger.debug(f"[NMS] Scores >= threshold: {np.sum(scores >= score_threshold)}")

    # Filter by score threshold ONCE here
    # NMS will use threshold=0.0 to avoid double filtering
    valid_mask = scores >= score_threshold
    boxes = boxes[valid_mask]
    scores = scores[valid_mask]
    classes = classes[valid_mask]

    if len(boxes) == 0:
        return boxes, scores, classes

    # Sort by scores (highest first)
    sorted_indices = np.argsort(scores)[::-1]
    boxes = boxes[sorted_indices]
    scores = scores[sorted_indices]
    classes = classes[sorted_indices]

    # Apply NMS per class using PyTorch (matches .pt model behavior)
    keep_indices = []
    for class_id in np.unique(classes):
        class_mask = classes == class_id
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]
        class_indices = np.where(class_mask)[0]

        # Convert to PyTorch tensors
        boxes_tensor = torch.from_numpy(class_boxes).float()
        scores_tensor = torch.from_numpy(class_scores).float()

        # Apply NMS using PyTorch (same as Ultralytics .pt model)
        nms_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold)

        if len(nms_indices) > 0:
            logger.debug(
                f"[NMS] Class {class_id}: kept {len(nms_indices)} / {len(class_boxes)} boxes"
            )
            keep_indices.extend(class_indices[nms_indices.numpy()])
        else:
            logger.debug(
                f"[NMS] Class {class_id}: no boxes kept (had {len(class_boxes)} candidates)"
            )

    # Sort keep_indices to maintain order
    keep_indices = sorted(keep_indices)

    logger.debug(f"[NMS] Output: {len(keep_indices)} boxes after NMS")
    return boxes[keep_indices], scores[keep_indices], classes[keep_indices]


def postprocess_output(
    output: np.ndarray,
    original_shape: Tuple[int, int],
    input_size: int = 224,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Postprocess YOLO ONNX model output.

    Args:
        output: Raw model output from ONNX inference
        original_shape: Original image shape (height, width)
        input_size: Model input size used during inference
        conf_threshold: Confidence threshold for filtering detections
        iou_threshold: IoU threshold for NMS

    Returns:
        Tuple containing:
        - boxes: Bounding boxes in format [x1, y1, x2, y2] scaled to original image size
        - scores: Confidence scores
        - classes: Class IDs
    """
    # YOLO11 output format: (1, 4+num_classes, N_anchors)
    # YOLO11 does not use objectness score, only class probabilities
    # For 3 classes: (1, 7, N) where 7 = 4 bbox coords + 3 class probs

    # Squeeze batch dimension
    output = np.squeeze(output)  # Now (7, N) or (N, 7)

    # Handle different output formats
    if output.shape[0] < output.shape[-1]:
        # Format: (4+num_classes, N) -> transpose to (N, 4+num_classes)
        output = output.T

    # Now output is (N, 4+num_classes)
    # Split into box coords and class probabilities
    boxes_xywh = output[:, :4]  # (N, 4) - x, y, w, h

    # YOLO11 doesn't have objectness score, just class probabilities
    if output.shape[1] > 4:
        class_probs = output[:, 4:]  # (N, num_classes) - already probabilities from ONNX model

        # Debug: Check if values are in [0,1] range (probabilities) or raw logits
        min_val, max_val = class_probs.min(), class_probs.max()
        logger.debug(f"Class output range: min={min_val:.4f}, max={max_val:.4f}")

        # If values are outside [0, 1], they are likely logits - apply sigmoid
        if min_val < 0 or max_val > 1:
            logger.debug("Applying sigmoid to convert logits to probabilities")
            class_probs = 1.0 / (1.0 + np.exp(-class_probs))
        else:
            logger.debug("Values already in [0,1] range, treating as probabilities")

        class_ids = np.argmax(class_probs, axis=1)  # (N,)
        scores = np.max(class_probs, axis=1)  # (N,) - use max class prob as confidence

        logger.debug(f"Total anchors before filtering: {len(scores)}")
        logger.debug(f"Score range: min={scores.min():.4f}, max={scores.max():.4f}")
    else:
        # Fallback if format is different
        scores = np.ones(len(boxes_xywh))
        class_ids = np.zeros(len(scores), dtype=int)

    # Filter by confidence threshold
    valid_mask = scores >= conf_threshold
    if not np.any(valid_mask):
        # No valid detections
        return np.array([]), np.array([]), np.array([])

    boxes_xywh = boxes_xywh[valid_mask]
    scores = scores[valid_mask]
    class_ids = class_ids[valid_mask]

    # Convert from xywh to xyxy format
    boxes_xyxy = np.zeros_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1 = x - w/2
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1 = y - h/2
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2 = x + w/2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2 = y + h/2

    # Scale boxes from input_size to original image size
    orig_h, orig_w = original_shape
    scale_x = orig_w / input_size
    scale_y = orig_h / input_size

    boxes_xyxy[:, [0, 2]] *= scale_x
    boxes_xyxy[:, [1, 3]] *= scale_y

    # Clip boxes to image boundaries
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)

    # Apply NMS
    boxes_xyxy, scores, class_ids = non_max_suppression(
        boxes_xyxy, scores, class_ids, iou_threshold, conf_threshold
    )

    return boxes_xyxy, scores, class_ids


def save_annotated_image(
    image_path: Path, output_path: Path, detections: list, class_names: list = None
) -> None:
    """Save an annotated image with bounding boxes drawn.

    Args:
        image_path: Path to the original image
        output_path: Path where to save the annotated image
        detections: List of detections from run_inference
        class_names: List of class names for labels

    Raises:
        ValueError: If image cannot be loaded
        RuntimeError: If saving fails
    """
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    # Load original image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Define colors for each class (BGR format)
    colors = [
        (255, 255, 0),  # Cyan for Light Blue
        (255, 0, 0),  # Blue for Dark Blue
        (0, 165, 255),  # Orange for Others
    ]

    # Draw each detection
    for detection in detections:
        bbox = detection["bbox"]  # [x1, y1, x2, y2]
        class_id = detection["class_id"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]

        x1, y1, x2, y2 = map(int, bbox)

        # Choose color based on class_id
        color = colors[class_id] if class_id < len(colors) else (0, 255, 0)

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw label with confidence
        label = f"{class_name}: {confidence:.2f}"

        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # Draw background rectangle for text
        cv2.rectangle(
            image, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), color, -1
        )

        # Draw text
        cv2.putText(
            image,
            label,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # Save annotated image
    success = cv2.imwrite(str(output_path), image)
    if not success:
        raise RuntimeError(f"Failed to save annotated image to: {output_path}")

    logger.info(f"Annotated image saved to: {output_path}")


def run_inference(cfg: Config, image_path: Path) -> Dict[str, Any]:
    """Run inference on a single image using ONNX model.

    Args:
        cfg: Configuration object containing inference parameters.
        image_path: Path to the input image file.

    Returns:
        Dictionary containing inference results with keys:
        - image_path: Path to the input image
        - detections: List of detections, each containing:
          - bbox: [x1, y1, x2, y2] bounding box coordinates
          - confidence: Detection confidence score
          - class_id: Class ID
          - class_name: Class name
        - num_detections: Number of detections found
        - inference_time_ms: Time taken for model inference (milliseconds)
        - preprocess_time_ms: Time taken for preprocessing (milliseconds)
        - postprocess_time_ms: Time taken for postprocessing (milliseconds)
        - total_time_ms: Total inference time including all steps (milliseconds)

    Raises:
        FileNotFoundError: If the image file or model file doesn't exist.
        ValueError: If required configuration parameters are missing.
        RuntimeError: If inference fails.

    Example:
        >>> from bsort.config import load_config
        >>> cfg = load_config(Path("settings.yaml"))
        >>> results = run_inference(cfg, Path("sample.jpg"))
        >>> print(f"Found {results['num_detections']} objects in {results['total_time_ms']}ms")
    """
    logger.info(f"Starting inference on image: {image_path}")

    # Validate image exists
    if not image_path.exists():
        logger.error(f"Image file not found: {image_path}")
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Get inference configuration
    inference_cfg = cfg.get("inference", {})
    model_path = Path(inference_cfg.get("model_path", "model.onnx"))
    conf_threshold = inference_cfg.get("confidence_threshold", 0.25)
    iou_threshold = inference_cfg.get("iou_threshold", 0.45)
    imgsz = inference_cfg.get("imgsz", 224)
    class_names = inference_cfg.get("class_names", DEFAULT_CLASS_NAMES)

    # Validate model exists
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            f"Please export your trained model using 'bsort export' command."
        )

    logger.info(f"Loading ONNX model from: {model_path}")

    try:
        # Create ONNX Runtime session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Use CPU provider (as per user requirements)
        providers = ["CPUExecutionProvider"]

        session = ort.InferenceSession(
            str(model_path), sess_options=session_options, providers=providers
        )

        logger.info(f"Model loaded successfully with provider: {session.get_providers()}")

        # Get model input details
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        logger.info(f"Model input: {input_name}, shape: {input_shape}")

        # Start total timing
        total_start = time.time()

        # Load and preprocess image
        logger.info("Loading and preprocessing image...")
        preprocess_start = time.time()
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        preprocessed, original_shape = preprocess_image(image, imgsz)
        preprocess_time_ms = (time.time() - preprocess_start) * 1000

        # Run inference
        logger.info("Running inference...")
        inference_start = time.time()
        outputs = session.run(None, {input_name: preprocessed})
        inference_time_ms = (time.time() - inference_start) * 1000

        # Postprocess output
        logger.info("Postprocessing results...")
        postprocess_start = time.time()
        boxes, scores, class_ids = postprocess_output(
            outputs[0],
            original_shape,
            imgsz,
            conf_threshold,
            iou_threshold,
        )
        postprocess_time_ms = (time.time() - postprocess_start) * 1000

        # Calculate total time
        total_time_ms = (time.time() - total_start) * 1000

        # Format results
        detections = []
        for box, score, class_id in zip(boxes, scores, class_ids):
            detection = {
                "bbox": box.tolist(),  # [x1, y1, x2, y2]
                "confidence": float(score),
                "class_id": int(class_id),
                "class_name": (
                    class_names[int(class_id)]
                    if int(class_id) < len(class_names)
                    else f"Class {class_id}"
                ),
            }
            detections.append(detection)

        results = {
            "image_path": str(image_path),
            "detections": detections,
            "num_detections": len(detections),
            "inference_time_ms": round(inference_time_ms, 2),
            "preprocess_time_ms": round(preprocess_time_ms, 2),
            "postprocess_time_ms": round(postprocess_time_ms, 2),
            "total_time_ms": round(total_time_ms, 2),
        }

        logger.info(f"Inference completed successfully - found {len(detections)} detections")
        logger.info(
            f"Timing: inference={inference_time_ms:.2f}ms, preprocess={preprocess_time_ms:.2f}ms, postprocess={postprocess_time_ms:.2f}ms, total={total_time_ms:.2f}ms"
        )
        return results

    except Exception as e:
        logger.error(f"Inference failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"Inference failed: {str(e)}")
