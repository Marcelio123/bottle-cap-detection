"""Training module for bsort.

This module contains functions for training YOLO models using Ultralytics.

Data Structure Required:
    Before training, organize your data in the following structure:

    dataset/
    ├── train/
    │   ├── images/     # Training images (.jpg, .png, etc.)
    │   └── labels/     # Training labels (.txt in YOLO format)
    ├── val/
    │   ├── images/     # Validation images
    │   └── labels/     # Validation labels
    └── data.yaml       # Dataset configuration file

    The data.yaml file should contain:
        path: /absolute/path/to/dataset
        train: train/images
        val: val/images
        nc: 3
        names: ['Light Blue', 'Dark Blue', 'Others']
"""

import logging
import os
from pathlib import Path

from ultralytics import YOLO

from bsort.config import Config

logger = logging.getLogger(__name__)


def run_training(cfg: Config) -> None:
    """Run the YOLO training pipeline with proper W&B integration."""

    logger.info("Starting YOLO training pipeline")

    # ---------------------------------------------------------------------
    # Validate configuration
    # ---------------------------------------------------------------------
    model_type = cfg.get("model", {}).get("type")
    if not model_type:
        raise ValueError("Model type not specified in config")

    yaml_path = cfg.get("data", {}).get("yaml_path")
    if not yaml_path:
        raise ValueError("Dataset yaml_path not specified in config")

    if not Path(yaml_path).exists():
        raise FileNotFoundError(f"Dataset configuration file not found: {yaml_path}")

    # ---------------------------------------------------------------------
    # Load YOLO model
    # ---------------------------------------------------------------------
    logger.info(f"Loading YOLO model: {model_type}")
    model = YOLO(model_type)

    # ---------------------------------------------------------------------
    # Training parameters
    # ---------------------------------------------------------------------
    training_cfg = cfg.get("training", {})
    epochs = training_cfg.get("epochs", 100)
    imgsz = training_cfg.get("imgsz", 640)
    batch = training_cfg.get("batch", 16)
    device = training_cfg.get("device", "cuda")
    patience = training_cfg.get("patience", 50)

    aug_cfg = training_cfg.get("augmentation", {})

    # ---------------------------------------------------------------------
    # Output configuration
    # ---------------------------------------------------------------------
    output_cfg = cfg.get("output", {})
    project = output_cfg.get("project", "outputs")
    name = output_cfg.get("name", "train")
    save_period = output_cfg.get("save_period", 10)

    # ---------------------------------------------------------------------
    # Logging configuration
    # ---------------------------------------------------------------------
    logging_cfg = cfg.get("logging", {})
    verbose = logging_cfg.get("verbose", True)
    plots = logging_cfg.get("plots", True)

    # ---------------------------------------------------------------------
    # W&B Integration (CORRECT)
    # ---------------------------------------------------------------------
    wandb_cfg = cfg.get("wandb", {})
    wandb_enabled = wandb_cfg.get("enabled", False)

    if wandb_enabled:
        logger.info("Weights & Biases logging ENABLED")

        # These are the ONLY correct ways to configure W&B for YOLO.
        project_name = wandb_cfg.get("project", "bsort-training")
        run_name = wandb_cfg.get("name", None)
        tags = wandb_cfg.get("tags", [])
        notes = wandb_cfg.get("notes", None)

        os.environ["WANDB_PROJECT"] = project_name

        if run_name:
            os.environ["WANDB_NAME"] = run_name

        if tags:
            os.environ["WANDB_TAGS"] = ",".join(tags)

        if notes:
            os.environ["WANDB_NOTES"] = notes

        # Important: DO NOT call wandb.init()
        # YOLO will automatically start the run.

    else:
        logger.info("Weights & Biases logging DISABLED")
        os.environ["WANDB_MODE"] = "disabled"

    # ---------------------------------------------------------------------
    # Start training
    # ---------------------------------------------------------------------
    logger.info(f"Starting training for {epochs} epochs")
    logger.info(f"Image size: {imgsz}, Batch size: {batch}, Device: {device}")

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=patience,
        project=project,
        name=name,
        save_period=save_period,
        verbose=verbose,
        plots=plots,
        # Augmentations
        hsv_h=aug_cfg.get("hsv_h", 0.015),
        hsv_s=aug_cfg.get("hsv_s", 0.7),
        hsv_v=aug_cfg.get("hsv_v", 0.4),
        degrees=aug_cfg.get("degrees", 0.0),
        translate=aug_cfg.get("translate", 0.1),
        scale=aug_cfg.get("scale", 0.5),
        shear=aug_cfg.get("shear", 0.0),
        perspective=aug_cfg.get("perspective", 0.0),
        flipud=aug_cfg.get("flipud", 0.0),
        fliplr=aug_cfg.get("fliplr", 0.5),
        mosaic=aug_cfg.get("mosaic", 1.0),
        mixup=aug_cfg.get("mixup", 0.0),
    )

    # ---------------------------------------------------------------------
    # Training complete
    # ---------------------------------------------------------------------
    logger.info("Training completed successfully")
    logger.info(f"Results saved to: {project}/{name}")
    logger.info(f"Best model saved to: {project}/{name}/weights/best.pt")
