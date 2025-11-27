"""
Data augmentation module for bsort.

This module provides offline data augmentation capabilities for YOLO detection datasets.
It applies various geometric and advanced transformations while automatically adjusting
bounding box annotations to match the transformed images.
"""

import logging
import random
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np

from bsort.config import Config

logger = logging.getLogger(__name__)


def create_augmentation_pipeline(config: Config) -> A.Compose:
    """
    Create an augmentation pipeline using albumentations.

    Args:
        config: Configuration object containing augmentation parameters

    Returns:
        Albumentations Compose object with augmentation transforms
    """
    aug_config = config.get("augmentation_offline", {})

    # Define augmentation transforms (excluding color space transforms as requested)
    transforms = [
        # Geometric transforms
        A.Rotate(
            limit=aug_config.get("rotation_limit", 30), border_mode=cv2.BORDER_CONSTANT, p=0.7
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            rotate=0,  # Already handled by Rotate above
            shear=0,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5,
        ),
        # Advanced transforms
        A.GaussianBlur(blur_limit=(3, 7), p=aug_config.get("blur_prob", 0.3)),
        A.GaussNoise(
            std_range=(0.01, 0.05),  # Standard deviation range for noise
            mean_range=(0.0, 0.0),  # Mean of the noise distribution
            per_channel=True,
            p=aug_config.get("noise_prob", 0.3),
        ),
        # Optional: Slight perspective changes
        A.Perspective(scale=(0.05, 0.1), p=0.2),
    ]

    # Create pipeline with bbox support (YOLO format)
    pipeline = A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"], min_area=0, min_visibility=0.3
        ),
    )

    return pipeline


def load_yolo_annotations(label_path: Path) -> Tuple[List[List[float]], List[int]]:
    """
    Load YOLO format annotations from a text file.

    Args:
        label_path: Path to the YOLO .txt annotation file

    Returns:
        Tuple of (bboxes, class_labels) where bboxes are in YOLO format
        [x_center, y_center, width, height] normalized to [0, 1]
    """
    if not label_path.exists():
        return [], []

    bboxes = []
    class_labels = []

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                logger.warning(f"Invalid annotation line in {label_path}: {line}")
                continue

            class_id = int(parts[0])
            bbox = [float(x) for x in parts[1:5]]

            class_labels.append(class_id)
            bboxes.append(bbox)

    return bboxes, class_labels


def save_yolo_annotations(
    label_path: Path, bboxes: List[List[float]], class_labels: List[int]
) -> None:
    """
    Save YOLO format annotations to a text file.

    Args:
        label_path: Path where the annotation file should be saved
        bboxes: List of bounding boxes in YOLO format
        class_labels: List of class IDs corresponding to each bbox
    """
    with open(label_path, "w") as f:
        for class_id, bbox in zip(class_labels, bboxes):
            # Ensure class_id is integer (albumentations may return as float)
            class_id = int(class_id)
            # Ensure values are within valid range [0, 1]
            bbox = [max(0.0, min(1.0, val)) for val in bbox]
            line = f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
            f.write(line)


def augment_image_with_annotations(
    image_path: Path,
    label_path: Path,
    output_image_path: Path,
    output_label_path: Path,
    pipeline: A.Compose,
) -> bool:
    """
    Augment a single image and its annotations.

    Args:
        image_path: Path to input image
        label_path: Path to input YOLO annotations
        output_image_path: Path to save augmented image
        output_label_path: Path to save augmented annotations
        pipeline: Albumentations augmentation pipeline

    Returns:
        True if augmentation was successful, False otherwise
    """
    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return False

        # Load annotations
        bboxes, class_labels = load_yolo_annotations(label_path)

        # Apply augmentation
        if len(bboxes) > 0:
            augmented = pipeline(image=image, bboxes=bboxes, class_labels=class_labels)

            # Check if any bboxes survived the augmentation
            if len(augmented["bboxes"]) == 0:
                logger.warning(f"All bounding boxes were removed during augmentation: {image_path}")
                return False

            augmented_image = augmented["image"]
            augmented_bboxes = augmented["bboxes"]
            augmented_labels = augmented["class_labels"]
        else:
            # No annotations, just augment the image
            augmented = pipeline(image=image, bboxes=[], class_labels=[])
            augmented_image = augmented["image"]
            augmented_bboxes = []
            augmented_labels = []

        # Save augmented image
        cv2.imwrite(str(output_image_path), augmented_image)

        # Save augmented annotations
        if label_path.exists():
            save_yolo_annotations(output_label_path, augmented_bboxes, augmented_labels)

        return True

    except Exception as e:
        logger.error(f"Error augmenting {image_path}: {e}")
        return False


def run_augmentation(config: Config, input_dir: Path, num_augmentations: int = 7) -> None:
    """
    Run data augmentation on a directory of images.

    Args:
        config: Configuration object
        input_dir: Directory containing images to augment
        num_augmentations: Number of augmented versions to create per image (5-10)
    """
    logger.info("Starting data augmentation...")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Augmentations per image: {num_augmentations}")

    # Validate input directory
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return

    # Ensure num_augmentations is in valid range
    if not (5 <= num_augmentations <= 10):
        logger.warning(
            f"num_augmentations {num_augmentations} is outside recommended range [5, 10]"
        )

    # Find images directory and labels directory
    if input_dir.name == "images":
        images_dir = input_dir
        labels_dir = input_dir.parent / "labels"
    else:
        # Assume input_dir contains both images and labels subdirectories
        images_dir = input_dir / "images"
        labels_dir = input_dir / "labels"

        # If images subdir doesn't exist, assume all images are in input_dir
        if not images_dir.exists():
            images_dir = input_dir
            labels_dir = input_dir  # Labels in same directory

    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Labels directory: {labels_dir}")

    # Get list of image files
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f"*{ext}"))
        image_files.extend(images_dir.glob(f"*{ext.upper()}"))

    if not image_files:
        logger.error(f"No image files found in {images_dir}")
        return

    logger.info(f"Found {len(image_files)} images to augment")

    # Create augmentation pipeline
    pipeline = create_augmentation_pipeline(config)

    # Process each image
    successful = 0
    failed = 0
    total_generated = 0

    for image_path in image_files:
        # Skip already augmented images
        if "_aug" in image_path.stem:
            logger.debug(f"Skipping already augmented image: {image_path.name}")
            continue

        # Find corresponding label file
        label_path = labels_dir / f"{image_path.stem}.txt"

        # Generate multiple augmentations
        for aug_idx in range(1, num_augmentations + 1):
            # Create output paths with suffix
            output_image_path = (
                image_path.parent / f"{image_path.stem}_aug{aug_idx}{image_path.suffix}"
            )
            output_label_path = labels_dir / f"{image_path.stem}_aug{aug_idx}.txt"

            # Perform augmentation
            success = augment_image_with_annotations(
                image_path, label_path, output_image_path, output_label_path, pipeline
            )

            if success:
                successful += 1
                total_generated += 1
                if total_generated % 50 == 0:
                    logger.info(f"Generated {total_generated} augmented images...")
            else:
                failed += 1

    # Summary
    logger.info("=" * 60)
    logger.info("Augmentation complete!")
    logger.info(f"Original images: {len(image_files)}")
    logger.info(f"Augmented images generated: {successful}")
    logger.info(f"Failed augmentations: {failed}")
    logger.info(f"Total dataset size: {len(image_files) + successful}")
    logger.info("=" * 60)
