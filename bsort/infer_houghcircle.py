"""HoughCircle-based inference module for bsort.

This module contains functions for detecting and classifying circular objects
using traditional computer vision techniques (HoughCircles + HSV color analysis).
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from bsort.config import Config

logger = logging.getLogger(__name__)

# Class name mapping
CLASS_NAMES = ["Light Blue", "Dark Blue", "Others"]


def fast_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction for image preprocessing.

    Args:
        image: Input image (grayscale)
        gamma: Gamma value for correction

    Returns:
        Gamma-corrected image
    """
    return (cv2.pow(image / 255.0, gamma) * 255).astype(np.uint8)


def classify_blue_color(h: float) -> str:
    """Classify hue value into color categories.

    Args:
        h: Hue value (0-180 in OpenCV HSV)

    Returns:
        Color classification: "light_blue", "dark_blue", or "other"
    """
    if h < 85 or h > 130:
        return "other"
    return "dark_blue" if h > 102.5 else "light_blue"


def extract_circle_pixels(img: np.ndarray, x: int, y: int, r: int) -> Tuple[np.ndarray, np.ndarray]:
    """Extract pixels within a circular region.

    Args:
        img: Input image
        x: Circle center x-coordinate
        y: Circle center y-coordinate
        r: Circle radius

    Returns:
        Tuple of (circle_pixels, mask)
    """
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    circle_pixels = cv2.bitwise_and(img, img, mask=mask)
    return circle_pixels, mask


def detect_and_classify(img: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Detect circles and classify their colors.

    Uses HoughCircles with exact parameters from the notebook code.

    Args:
        img: Input image in BGR format

    Returns:
        Tuple of (annotated_image, detections_list)
        Each detection contains: bbox, center, radius, h_mean, class_label, class_id
    """
    # Resize for speed (exact parameters from notebook)
    img_small = cv2.resize(img, (0, 0), fx=0.4, fy=0.4)
    gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)

    # Gamma + CLAHE + Blur (exact parameters from notebook)
    gray_gamma = fast_gamma(gray, 0.5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray_eq = clahe.apply(gray_gamma)
    gray_blur = cv2.GaussianBlur(gray_eq, (7, 7), 1.5)

    # HoughCircles detection (exact parameters from notebook)
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=25,
        param1=35,
        param2=25,
        minRadius=8,
        maxRadius=16,
    )

    # Prepare output
    output_hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    detections = []

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for x, y, r in circles[0]:
            # Extract pixels inside circle
            circle_pixels, mask = extract_circle_pixels(output_hsv, x, y, r)

            # Mean HSV - calculate mean hue
            h_mean = np.mean(circle_pixels[mask > 0, 0])

            # Classify color
            label = classify_blue_color(h_mean)

            # Map label to class_id
            class_id_map = {"light_blue": 0, "dark_blue": 1, "other": 2}
            class_id = class_id_map[label]

            # Calculate bounding box (x1, y1, x2, y2)
            x1 = max(0, x - r)
            y1 = max(0, y - r)
            x2 = min(img_small.shape[1], x + r)
            y2 = min(img_small.shape[0], y + r)

            # Store detection
            detection = {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "center": [int(x), int(y)],
                "radius": int(r),
                "h_mean": float(h_mean),
                "class_label": label,
                "class_id": class_id,
            }
            detections.append(detection)

    return img_small, detections


def save_annotated_image(image_path: Path, output_path: Path, detections: list) -> None:
    """Save an annotated image with circles and labels drawn.

    Args:
        image_path: Path to the original image
        output_path: Path where to save the annotated image
        detections: List of detections from run_inference_houghcircle

    Raises:
        ValueError: If image cannot be loaded
        RuntimeError: If saving fails
    """
    # Load original image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Resize to match detection scale (0.4x)
    img_small = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)

    # Define colors for each class (BGR format)
    colors = {
        0: (255, 255, 0),  # Cyan for Light Blue
        1: (255, 0, 0),  # Blue for Dark Blue
        2: (0, 165, 255),  # Orange for Others
    }

    # Draw each detection on the small image
    for detection in detections:
        bbox = detection["bbox"]  # [x1, y1, x2, y2] in original image coordinates
        class_id = detection["class_id"]
        class_name = detection["class_name"]

        # Scale bbox back to small image coordinates (0.4x scale)
        scale_factor = 0.4
        x1 = int(bbox[0] * scale_factor)
        y1 = int(bbox[1] * scale_factor)
        x2 = int(bbox[2] * scale_factor)
        y2 = int(bbox[3] * scale_factor)

        # Calculate center and radius from bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        radius = (x2 - x1) // 2

        # Choose color based on class_id
        color = colors.get(class_id, (0, 255, 0))

        # Draw circle with class-specific color
        cv2.circle(img_small, (center_x, center_y), radius, color, 2)

        # Draw label above circle (white text on the image)
        label = class_name
        cv2.putText(
            img_small,
            label,
            (center_x - radius, center_y - radius - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    # Save annotated image (small version, matching notebook output)
    success = cv2.imwrite(str(output_path), img_small)
    if not success:
        raise RuntimeError(f"Failed to save annotated image to: {output_path}")

    logger.info(f"Annotated image saved to: {output_path}")


def run_inference_houghcircle(cfg: Config, image_path: Path) -> Dict[str, Any]:
    """Run HoughCircle-based inference on a single image.

    This function matches the signature and behavior of run_inference() from infer.py.

    Args:
        cfg: Configuration object (passed for consistency, parameters are hardcoded)
        image_path: Path to the input image file

    Returns:
        Dictionary containing inference results with keys:
        - image_path: Path to the input image
        - detections: List of detections, each containing:
          - bbox: [x1, y1, x2, y2] bounding box coordinates
          - confidence: Detection confidence (1.0 for HoughCircles)
          - class_id: Class ID (0=Light Blue, 1=Dark Blue, 2=Others)
          - class_name: Class name string
        - num_detections: Number of detections found
        - preprocess_time_ms: Time for preprocessing (milliseconds)
        - inference_time_ms: Time for HoughCircles detection (milliseconds)
        - postprocess_time_ms: Time for color classification (milliseconds)
        - total_time_ms: Total processing time (milliseconds)

    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If image cannot be loaded
        RuntimeError: If inference fails
    """
    logger.info(f"Starting HoughCircle inference on image: {image_path}")

    # Validate image exists
    if not image_path.exists():
        logger.error(f"Image file not found: {image_path}")
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        # Start total timing
        total_start = time.time()

        # Load image
        logger.info("Loading image...")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        original_shape = image.shape[:2]  # (height, width)

        # Run detection with timing breakdown
        logger.info("Running HoughCircle detection...")

        # Preprocess timing (resize, gamma, CLAHE, blur)
        preprocess_start = time.time()
        img_small = cv2.resize(image, (0, 0), fx=0.4, fy=0.4)
        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        gray_gamma = fast_gamma(gray, 0.5)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray_eq = clahe.apply(gray_gamma)
        gray_blur = cv2.GaussianBlur(gray_eq, (7, 7), 1.5)
        preprocess_time_ms = (time.time() - preprocess_start) * 1000

        # Inference timing (HoughCircles detection)
        inference_start = time.time()
        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=25,
            param1=35,
            param2=25,
            minRadius=8,
            maxRadius=16,
        )
        inference_time_ms = (time.time() - inference_start) * 1000

        # Postprocess timing (color classification and bbox conversion)
        postprocess_start = time.time()
        output_hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
        detections = []

        if circles is not None:
            circles = np.uint16(np.around(circles))

            for x, y, r in circles[0]:
                # Extract pixels inside circle
                circle_pixels, mask = extract_circle_pixels(output_hsv, x, y, r)

                # Mean hue
                h_mean = np.mean(circle_pixels[mask > 0, 0])

                # Classify color
                label = classify_blue_color(h_mean)
                class_id_map = {"light_blue": 0, "dark_blue": 1, "other": 2}
                class_id = class_id_map[label]

                # Calculate bounding box scaled to original image size
                # Coordinates are in small image (0.4x scale), need to scale back
                scale_factor = 2.5  # 1 / 0.4
                x1_orig = max(0, int((x - r) * scale_factor))
                y1_orig = max(0, int((y - r) * scale_factor))
                x2_orig = min(original_shape[1], int((x + r) * scale_factor))
                y2_orig = min(original_shape[0], int((y + r) * scale_factor))

                # Create detection in format matching infer.py
                detection = {
                    "bbox": [x1_orig, y1_orig, x2_orig, y2_orig],
                    "confidence": 1.0,  # HoughCircles doesn't provide confidence
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                }
                detections.append(detection)

        postprocess_time_ms = (time.time() - postprocess_start) * 1000

        # Calculate total time
        total_time_ms = (time.time() - total_start) * 1000

        # Format results matching infer.py
        results = {
            "image_path": str(image_path),
            "detections": detections,
            "num_detections": len(detections),
            "preprocess_time_ms": round(preprocess_time_ms, 2),
            "inference_time_ms": round(inference_time_ms, 2),
            "postprocess_time_ms": round(postprocess_time_ms, 2),
            "total_time_ms": round(total_time_ms, 2),
        }

        logger.info(f"HoughCircle inference completed - found {len(detections)} detections")
        logger.info(
            f"Timing: preprocess={preprocess_time_ms:.2f}ms, inference={inference_time_ms:.2f}ms, postprocess={postprocess_time_ms:.2f}ms, total={total_time_ms:.2f}ms"
        )

        return results

    except Exception as e:
        logger.error(f"HoughCircle inference failed: {str(e)}", exc_info=True)
        raise RuntimeError(f"HoughCircle inference failed: {str(e)}")
