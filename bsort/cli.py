"""Command-line interface for bsort.

This module provides the CLI for the bsort machine learning tool.
It exposes five main commands:
    - train: Train a machine learning model using configuration from YAML
    - export: Export trained YOLO models to ONNX format
    - infer: Run inference on an image using a trained model
    - sanity_check: Validate sample data and optionally visualize detections
    - augment: Generate augmented versions of dataset images

The CLI uses Typer for argument parsing and command management.
All configuration is loaded from YAML files specified via the --config flag.

Example usage:
    $ bsort train --config settings.yaml
    $ bsort export --model outputs/train/weights/best.pt --output best.onnx
    $ bsort infer --config settings.yaml --image sample.jpg
    $ bsort sanity_check ./sample --visualize
    $ bsort augment --input-dir ./sample/train/images --num-augmentations 7 --config settings.yaml
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from bsort.augment import run_augmentation
from bsort.config import load_config
from bsort.export import export_to_onnx, validate_onnx_model
from bsort.infer import run_inference
from bsort.infer import save_annotated_image as save_annotated_yolo
from bsort.infer_houghcircle import (
    run_inference_houghcircle,
)
from bsort.infer_houghcircle import save_annotated_image as save_annotated_hough
from bsort.sanity_check import SanityChecker
from bsort.train import run_training

# Create Typer app instance
app = typer.Typer(
    name="bsort",
    help="Machine learning sorting and classification CLI tool",
    add_completion=False,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@app.command()
def train(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to YAML configuration file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
) -> None:
    """Train a machine learning model.

    Loads configuration from the specified YAML file and runs the training pipeline.

    Args:
        config: Path to the YAML configuration file.

    Returns:
        None

    Raises:
        typer.Exit: If an error occurs during training.

    Example:
        $ bsort train --config settings.yaml
    """
    logger.info("=" * 60)
    logger.info("BSORT TRAINING")
    logger.info("=" * 60)

    try:
        # Load configuration
        cfg = load_config(config)
        logger.info(f"Loaded configuration from: {config}")

        # Run training
        run_training(cfg)

        logger.info("=" * 60)
        logger.info("Training completed successfully")
        logger.info("=" * 60)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise typer.Exit(code=1)


@app.command()
def export(
    model: Annotated[
        Path,
        typer.Option(
            "--model",
            "-m",
            help="Path to trained YOLO model (.pt file)",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output path for ONNX model (optional, defaults to same directory as input)",
        ),
    ] = None,
    imgsz: Annotated[
        int,
        typer.Option(
            "--imgsz",
            help="Input image size (should match training size)",
        ),
    ] = 224,
    opset: Annotated[
        int,
        typer.Option(
            "--opset",
            help="ONNX opset version",
        ),
    ] = 12,
    validate: Annotated[
        bool,
        typer.Option(
            "--validate/--no-validate",
            help="Validate ONNX model after export",
        ),
    ] = True,
) -> None:
    """Export trained YOLO model to ONNX format.

    Converts a PyTorch YOLO model (.pt) to ONNX format for optimized inference
    with onnxruntime. The exported model uses fixed batch size (batch=1) and
    simplified ONNX graph for better performance.

    Args:
        model: Path to the trained YOLO model (.pt file).
        output: Optional output path for the ONNX model. If not specified,
                saves alongside the input model with .onnx extension.
        imgsz: Input image size (default: 224, should match training size).
        opset: ONNX opset version (default: 12 for wide compatibility).
        validate: Whether to validate the ONNX model after export (default: True).

    Returns:
        None

    Raises:
        typer.Exit: If an error occurs during export.

    Example:
        $ bsort export --model outputs/train/weights/best.pt
        $ bsort export -m best.pt -o models/detector.onnx --imgsz 224
        $ bsort export --model last.pt --opset 12 --no-validate
    """
    logger.info("=" * 60)
    logger.info("BSORT ONNX EXPORT")
    logger.info("=" * 60)

    try:
        # Export to ONNX
        exported_path = export_to_onnx(
            model_path=str(model),
            output_path=str(output) if output else None,
            imgsz=imgsz,
            opset=opset,
            simplify=True,
        )

        # Validate if requested
        if validate:
            validation_passed = validate_onnx_model(exported_path)
            if not validation_passed:
                logger.warning("ONNX validation failed, but export completed")

        logger.info("=" * 60)
        logger.info("Export completed successfully")
        logger.info(f"ONNX model saved to: {exported_path}")
        logger.info("=" * 60)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        raise typer.Exit(code=1)


@app.command()
def infer(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to YAML configuration file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    image: Annotated[
        Path,
        typer.Option(
            "--image",
            "-i",
            help="Path to input image file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output path for annotated image (optional)",
        ),
    ] = None,
) -> None:
    """Run inference on an image.

    Loads configuration from the specified YAML file and runs inference
    on the provided image.

    Args:
        config: Path to the YAML configuration file.
        image: Path to the input image file.
        output: Optional output path for saving annotated image with bounding boxes.

    Returns:
        None

    Raises:
        typer.Exit: If an error occurs during inference.

    Example:
        $ bsort infer --config settings.yaml --image sample.jpg
        $ bsort infer -c settings.yaml -i sample.jpg -o output.jpg
    """
    logger.info("=" * 60)
    logger.info("BSORT INFERENCE")
    logger.info("=" * 60)

    try:
        # Load configuration
        cfg = load_config(config)
        logger.info(f"Loaded configuration from: {config}")

        # Run inference
        results = run_inference(cfg, image)

        # Save annotated image if output path provided
        if output:
            inference_cfg = cfg.get("inference", {})
            class_names = inference_cfg.get("class_names", ["Light Blue", "Dark Blue", "Others"])
            save_annotated_yolo(image, output, results["detections"], class_names)

        # Display results
        logger.info("=" * 60)
        logger.info("RESULTS")
        logger.info("=" * 60)
        for key, value in results.items():
            logger.info(f"{key}: {value}")
        if output:
            logger.info(f"output_image: {output}")

        logger.info("=" * 60)
        logger.info("Inference completed successfully")
        logger.info("=" * 60)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise typer.Exit(code=1)


@app.command()
def infer_houghcircle(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to YAML configuration file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    image: Annotated[
        Path,
        typer.Option(
            "--image",
            "-i",
            help="Path to input image file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output path for annotated image (optional)",
        ),
    ] = None,
) -> None:
    """Run HoughCircle-based inference on an image.

    Detects circular objects and classifies them by color (light blue, dark blue, other)
    using traditional computer vision techniques (HoughCircles + HSV analysis).

    Uses exact thresholds from the notebook code (hardcoded parameters).

    Args:
        config: Path to the YAML configuration file.
        image: Path to the input image file.
        output: Optional output path for saving annotated image with circles and labels.

    Returns:
        None

    Raises:
        typer.Exit: If an error occurs during inference.

    Example:
        $ bsort infer-houghcircle --config settings.yaml --image sample.jpg
        $ bsort infer-houghcircle -c settings.yaml -i sample.jpg -o output.jpg
    """
    logger.info("=" * 60)
    logger.info("BSORT HOUGHCIRCLE INFERENCE")
    logger.info("=" * 60)

    try:
        # Load configuration
        cfg = load_config(config)
        logger.info(f"Loaded configuration from: {config}")

        # Run HoughCircle inference
        results = run_inference_houghcircle(cfg, image)

        # Save annotated image if output path provided
        if output:
            save_annotated_hough(image, output, results["detections"])

        # Display results
        logger.info("=" * 60)
        logger.info("RESULTS")
        logger.info("=" * 60)
        for key, value in results.items():
            logger.info(f"{key}: {value}")
        if output:
            logger.info(f"output_image: {output}")

        logger.info("=" * 60)
        logger.info("Inference completed successfully")
        logger.info("=" * 60)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise typer.Exit(code=1)


@app.command()
def sanity_check(
    sample_dir: Annotated[
        Path,
        typer.Argument(
            help="Path to sample directory",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
        ),
    ] = Path("./sample"),
    visualize: Annotated[
        bool,
        typer.Option(
            "--visualize",
            "-v",
            help="Generate visualization after validation",
        ),
    ] = False,
    min_samples: Annotated[
        int,
        typer.Option(
            "--min-samples",
            "-n",
            help="Minimum number of samples to visualize",
        ),
    ] = 20,
) -> None:
    """Validate sample data and optionally visualize detections.

    Performs comprehensive validation on sample data including:
    - File pairing (each .jpg has matching .txt)
    - Naming convention validation
    - Data format and completeness checks
    - Value range validation

    Optionally generates visualizations with bounding boxes for detected classes:
    - Light Blue (cyan boxes)
    - Dark Blue (blue boxes)
    - Others (orange boxes)

    Args:
        sample_dir: Path to the sample directory containing .jpg and .txt files.
        visualize: Whether to generate visualizations after validation.
        min_samples: Minimum number of samples to visualize (default: 20).

    Returns:
        None

    Raises:
        typer.Exit: If validation fails.

    Example:
        $ bsort sanity_check
        $ bsort sanity_check ./data/samples --visualize
        $ bsort sanity_check ./data/samples -v -n 10
    """
    logger.info("=" * 60)
    logger.info("BSORT SANITY CHECK")
    logger.info("=" * 60)

    try:
        # Create checker instance
        checker = SanityChecker(str(sample_dir))

        # Run validation
        validation_passed = checker.run_validation()

        # Visualize if requested and validation passed
        if visualize and validation_passed:
            checker.visualize_samples(min_samples=min_samples)

        if not validation_passed:
            logger.error("=" * 60)
            logger.error("Sanity check FAILED")
            logger.error("=" * 60)
            raise typer.Exit(code=1)

        logger.info("=" * 60)
        logger.info("Sanity check completed successfully")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Sanity check failed: {e}", exc_info=True)
        raise typer.Exit(code=1)


@app.command()
def augment(
    input_dir: Annotated[
        Path,
        typer.Option(
            "--input-dir",
            "-i",
            help="Path to directory containing images to augment",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
        ),
    ],
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to YAML configuration file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    num_augmentations: Annotated[
        int,
        typer.Option(
            "--num-augmentations",
            "-n",
            help="Number of augmented versions per image (recommended: 5-10)",
            min=1,
            max=20,
        ),
    ] = 7,
) -> None:
    """Generate augmented versions of dataset images.

    Creates multiple augmented copies of each image in the input directory,
    applying various geometric and advanced transformations. Automatically
    adjusts YOLO bounding box annotations to match the transformations.

    Augmentations applied:
    - Rotation (up to 30 degrees)
    - Horizontal and vertical flips
    - Scale and translation
    - Gaussian blur
    - Gaussian noise
    - Perspective transforms

    Augmented images are saved in the same directory with suffix pattern:
    - Original: raw-250110_dc_s001_b2_1.jpg
    - Augmented: raw-250110_dc_s001_b2_1_aug1.jpg, raw-250110_dc_s001_b2_1_aug2.jpg, etc.

    Args:
        input_dir: Directory containing images to augment. Can be:
                   - Direct path to images/ directory
                   - Parent directory containing images/ and labels/ subdirs
        config: Path to YAML configuration file with augmentation parameters.
        num_augmentations: Number of augmented versions per image (default: 7).

    Returns:
        None

    Raises:
        typer.Exit: If an error occurs during augmentation.

    Example:
        $ bsort augment --input-dir ./sample/train/images --config settings.yaml
        $ bsort augment -i ./sample/train/images -c settings.yaml -n 5
        $ bsort augment --input-dir ./data/train --num-augmentations 10 --config settings.yaml
    """
    logger.info("=" * 60)
    logger.info("BSORT DATA AUGMENTATION")
    logger.info("=" * 60)

    try:
        # Load configuration
        cfg = load_config(config)
        logger.info(f"Loaded configuration from: {config}")

        # Run augmentation
        run_augmentation(cfg, input_dir, num_augmentations)

        logger.info("=" * 60)
        logger.info("Augmentation completed successfully")
        logger.info("=" * 60)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Augmentation failed: {e}", exc_info=True)
        raise typer.Exit(code=1)


def main() -> None:
    """Main entrypoint for the CLI application.

    This function is called when the bsort command is executed.
    It runs the Typer app with the registered commands.

    Returns:
        None
    """
    app()


if __name__ == "__main__":
    main()
