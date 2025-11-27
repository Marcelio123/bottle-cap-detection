"""
ONNX Export Module for YOLO Models.

This module handles the conversion of trained YOLO models (.pt) to ONNX format
for optimized inference with onnxruntime.
"""

import logging
from pathlib import Path
from typing import Optional

from ultralytics import YOLO

logger = logging.getLogger(__name__)


def export_to_onnx(
    model_path: str,
    output_path: Optional[str] = None,
    imgsz: int = 224,
    opset: int = 12,
    simplify: bool = True,
) -> str:
    """
    Export a YOLO model to ONNX format.

    Args:
        model_path: Path to the trained YOLO model (.pt file)
        output_path: Optional output path for the ONNX model. If None, saves alongside .pt file
        imgsz: Input image size for the model (should match training size)
        opset: ONNX opset version to use (default: 12 for compatibility)
        simplify: Whether to simplify the ONNX graph (recommended)

    Returns:
        str: Path to the exported ONNX model

    Raises:
        FileNotFoundError: If the model_path doesn't exist
        RuntimeError: If export fails
    """
    model_path = Path(model_path)

    # Validate input model exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading YOLO model from: {model_path}")

    try:
        # Load the YOLO model
        model = YOLO(str(model_path))

        # Export to ONNX
        logger.info("Exporting model to ONNX format...")
        logger.info(f"  - Image size: {imgsz}")
        logger.info(f"  - Opset version: {opset}")
        logger.info(f"  - Simplify: {simplify}")
        logger.info(f"  - Batch size: 1 (fixed)")

        # Use Ultralytics' built-in export
        # The export method returns the path to the exported model
        exported_path = model.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            simplify=simplify,
            dynamic=False,  # Fixed batch size
        )

        # If user specified a custom output path, move the file
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Move the exported file to the desired location
            exported_path_obj = Path(exported_path)
            if exported_path_obj != output_path:
                import shutil

                shutil.move(str(exported_path_obj), str(output_path))
                exported_path = str(output_path)

        logger.info(f"✓ Model successfully exported to: {exported_path}")
        return exported_path

    except Exception as e:
        logger.error(f"Failed to export model: {str(e)}")
        raise RuntimeError(f"ONNX export failed: {str(e)}")


def validate_onnx_model(onnx_path: str) -> bool:
    """
    Validate an ONNX model to ensure it's properly formatted.

    Args:
        onnx_path: Path to the ONNX model file

    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        import onnx

        logger.info(f"Validating ONNX model: {onnx_path}")
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        logger.info("✓ ONNX model validation passed")
        return True

    except ImportError:
        logger.warning("onnx package not installed, skipping validation")
        return True

    except Exception as e:
        logger.error(f"ONNX validation failed: {str(e)}")
        return False
