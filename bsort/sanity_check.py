#!/usr/bin/env python3
"""
Sanity check script for sample data validation and visualization.
Validates image-annotation pairs and displays multiple samples with overlays.
"""

import random
import re
import sys
from pathlib import Path
from typing import Dict, List


class ValidationError:
    """Container for validation errors."""

    def __init__(self, file_path: str, line_num: int, error_msg: str):
        self.file_path = file_path
        self.line_num = line_num
        self.error_msg = error_msg

    def __str__(self):
        if self.line_num > 0:
            return f"[{self.file_path}:{self.line_num}] {self.error_msg}"
        return f"[{self.file_path}] {self.error_msg}"


class SanityChecker:
    """Validates sample data directory."""

    def __init__(self, sample_dir: str = "./sample"):
        self.sample_dir = Path(sample_dir)
        self.errors: List[ValidationError] = []
        self.file_pattern = re.compile(
            r"^raw-(\d{6})_([a-z]+)_s(\d{3})_b(\d+)_(\d+)(_aug\d+)?\.(jpg|txt)$"
        )
        self.is_yolo_format = self._detect_yolo_format()

    def _detect_yolo_format(self) -> bool:
        """Detect if directory uses YOLO format (train/val with images/labels subdirs)."""
        train_dir = self.sample_dir / "train"
        val_dir = self.sample_dir / "val"

        # Check if train or val directories exist with images subdirectory
        has_train = (train_dir / "images").exists()
        has_val = (val_dir / "images").exists()

        return has_train or has_val

    def add_error(self, file_path: str, error_msg: str, line_num: int = 0):
        """Add a validation error."""
        self.errors.append(ValidationError(file_path, line_num, error_msg))

    def _get_file_pairs_from_dir(
        self, images_dir: Path, labels_dir: Path, split_name: str
    ) -> Dict[str, List[Path]]:
        """Get file pairs from a specific directory."""
        if not images_dir.exists() or not labels_dir.exists():
            return {"jpg": [], "txt": []}

        jpg_files = set(f.stem for f in images_dir.glob("*.jpg"))
        txt_files = set(f.stem for f in labels_dir.glob("*.txt"))

        # Check for orphaned files
        orphaned_jpgs = jpg_files - txt_files
        orphaned_txts = txt_files - jpg_files

        for stem in orphaned_jpgs:
            self.add_error(f"{split_name}/{stem}.jpg", "Missing corresponding .txt file")

        for stem in orphaned_txts:
            self.add_error(f"{split_name}/{stem}.txt", "Missing corresponding .jpg file")

        # Return properly paired files
        paired_files = jpg_files & txt_files
        return {
            "jpg": [images_dir / f"{stem}.jpg" for stem in paired_files],
            "txt": [labels_dir / f"{stem}.txt" for stem in paired_files],
        }

    def check_file_pairing(self) -> Dict[str, List[Path]]:
        """Verify each .jpg has matching .txt and vice versa."""
        all_jpg_files = []
        all_txt_files = []

        if self.is_yolo_format:
            # Check train and val directories
            for split in ["train", "val"]:
                split_dir = self.sample_dir / split
                if split_dir.exists():
                    images_dir = split_dir / "images"
                    labels_dir = split_dir / "labels"

                    pairs = self._get_file_pairs_from_dir(images_dir, labels_dir, split)
                    all_jpg_files.extend(pairs["jpg"])
                    all_txt_files.extend(pairs["txt"])
        else:
            # Flat structure - check sample_dir directly
            pairs = self._get_file_pairs_from_dir(self.sample_dir, self.sample_dir, "sample")
            all_jpg_files = pairs["jpg"]
            all_txt_files = pairs["txt"]

        return {"jpg": all_jpg_files, "txt": all_txt_files}

    def check_naming_convention(self):
        """Validate file naming follows expected pattern."""
        files_to_check = []

        if self.is_yolo_format:
            # Check files in train/val directories
            for split in ["train", "val"]:
                images_dir = self.sample_dir / split / "images"
                labels_dir = self.sample_dir / split / "labels"

                if images_dir.exists():
                    files_to_check.extend(images_dir.glob("*"))
                if labels_dir.exists():
                    files_to_check.extend(labels_dir.glob("*"))
        else:
            # Check files in sample_dir directly
            files_to_check = list(self.sample_dir.glob("*"))

        for file_path in files_to_check:
            if file_path.is_file() and file_path.suffix in [".jpg", ".txt"]:
                match = self.file_pattern.match(file_path.name)
                if not match:
                    self.add_error(
                        file_path.name,
                        f"Invalid naming convention. Expected: raw-YYYYMMDD_type_sNNN_bN_N(_augN).(jpg|txt)",
                    )

    def check_data_format(self, txt_file: Path):
        """Validate data format in text file."""
        try:
            with open(txt_file, "r") as f:
                lines = f.readlines()

            if not lines:
                self.add_error(txt_file.name, "Empty file")
                return

            for line_num, line in enumerate(lines, start=1):
                line = line.strip()

                # Check for empty lines (incomplete rows)
                if not line:
                    self.add_error(txt_file.name, f"Incomplete/empty row", line_num)
                    continue

                # Split and validate fields
                fields = line.split()
                if len(fields) != 5:
                    self.add_error(
                        txt_file.name,
                        f"Expected 5 fields (CLASS + 4 values), got {len(fields)}",
                        line_num,
                    )
                    continue

                # Validate class (should be integer)
                try:
                    int(fields[0])
                except ValueError:
                    self.add_error(
                        txt_file.name,
                        f"Invalid class value: '{fields[0]}' (expected integer)",
                        line_num,
                    )

                # Validate float values
                for i, val in enumerate(fields[1:], start=1):
                    try:
                        float_val = float(val)
                        # Check if values are in expected range (0-1 for normalized data)
                        if not (0.0 <= float_val <= 1.0):
                            self.add_error(
                                txt_file.name,
                                f"Value {i} out of range [0,1]: {float_val}",
                                line_num,
                            )
                    except ValueError:
                        self.add_error(
                            txt_file.name, f"Invalid float value at position {i}: '{val}'", line_num
                        )

        except Exception as e:
            self.add_error(txt_file.name, f"Failed to read file: {str(e)}")

    def run_validation(self) -> bool:
        """Run all validation checks."""
        if not self.sample_dir.exists():
            print(f"Error: Directory '{self.sample_dir}' does not exist")
            return False

        print(f"Running sanity checks on: {self.sample_dir}")
        print(f"Directory format: {'YOLO (train/val split)' if self.is_yolo_format else 'Flat'}")
        print("=" * 70)

        # Check file pairing
        paired_files = self.check_file_pairing()

        print(f"Found {len(paired_files['jpg'])} image-annotation pairs")

        # Check naming conventions
        self.check_naming_convention()

        # Check data format for each text file
        for txt_file in paired_files["txt"]:
            self.check_data_format(txt_file)

        # Report errors
        if self.errors:
            print(f"\n❌ VALIDATION FAILED: {len(self.errors)} issue(s) found\n")
            for error in self.errors:
                print(f"  {error}")
            return False
        else:
            print("\n✅ VALIDATION PASSED: All checks successful")
            return True

    def visualize_samples(self, min_samples: int = 20):
        """Visualize multiple samples with annotations."""
        try:
            import matplotlib.pyplot as plt
            from PIL import Image, ImageDraw
        except ImportError:
            print("\n⚠️  Visualization skipped: Install PIL and matplotlib")
            print("   pip install pillow matplotlib")
            return

        # Class mapping and colors
        class_labels = {0: "Light Blue", 1: "Dark Blue", 2: "Others"}
        class_colors = {
            0: "cyan",  # Light blue for light blue class
            1: "blue",  # Dark blue for dark blue class
            2: "orange",  # Orange for others
        }

        # Get all text files based on directory format
        txt_files = []
        if self.is_yolo_format:
            # Get from train and val directories
            for split in ["train", "val"]:
                labels_dir = self.sample_dir / split / "labels"
                if labels_dir.exists():
                    txt_files.extend(labels_dir.glob("*.txt"))
        else:
            # Get from sample_dir directly
            txt_files = list(self.sample_dir.glob("*.txt"))

        if not txt_files:
            print("\n⚠️  No text files found for visualization")
            return

        # If we have fewer than min_samples, visualize all; otherwise sample randomly
        if len(txt_files) <= min_samples:
            selected_files = txt_files
        else:
            selected_files = random.sample(txt_files, min_samples)

        print(f"\n📊 Visualizing {len(selected_files)} sample(s)")

        # Calculate grid dimensions
        import math

        cols = min(5, len(selected_files))  # Max 5 columns
        rows = math.ceil(len(selected_files) / cols)

        # Create figure with subplots
        _fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        if len(selected_files) == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

        # Process each sample
        class_counts = {0: 0, 1: 0, 2: 0}
        for idx, txt_file in enumerate(selected_files):
            # Find corresponding jpg file
            if self.is_yolo_format:
                # In YOLO format, replace 'labels' with 'images' in the path
                jpg_file = Path(str(txt_file).replace("/labels/", "/images/")).with_suffix(".jpg")
            else:
                # In flat format, jpg is in same directory
                jpg_file = txt_file.with_suffix(".jpg")

            if not jpg_file.exists():
                print(f"⚠️  Skipping {txt_file.name}: image not found at {jpg_file}")
                continue

            # Load image
            img = Image.open(jpg_file)
            img_width, img_height = img.size

            # Read annotations
            annotations = []
            with open(txt_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        parts = line.split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:])
                            annotations.append((class_id, x_center, y_center, width, height))
                            class_counts[class_id] = class_counts.get(class_id, 0) + 1

            # Draw bounding boxes
            draw = ImageDraw.Draw(img)

            for class_id, x_c, y_c, w, h in annotations:
                # Convert normalized coordinates to pixel coordinates
                x_center_px = x_c * img_width
                y_center_px = y_c * img_height
                box_width = w * img_width
                box_height = h * img_height

                # Calculate box corners
                x1 = x_center_px - box_width / 2
                y1 = y_center_px - box_height / 2
                x2 = x_center_px + box_width / 2
                y2 = y_center_px + box_height / 2

                # Get color and label based on class
                color = class_colors.get(class_id, "red")
                label = class_labels.get(class_id, f"Class {class_id}")

                # Draw rectangle
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                # Draw label with background for better visibility
                draw.text((x1 + 2, max(0, y1 - 15)), label, fill=color)

            # Display in subplot
            ax = axes[idx]
            ax.imshow(img)
            ax.set_title(f"{jpg_file.name}\n{len(annotations)} detection(s)", fontsize=8)
            ax.axis("off")

        # Hide extra subplots if any
        for idx in range(len(selected_files), len(axes)):
            axes[idx].axis("off")

        # Print summary statistics
        print(f"\n📈 Detection Summary:")
        for class_id, count in sorted(class_counts.items()):
            if count > 0:
                print(f"   {class_labels.get(class_id, f'Class {class_id}')}: {count}")

        # Adjust layout and save
        plt.tight_layout()
        output_path = "sample_visualizations.png"
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"✅ Visualizations saved to: {output_path}")

        plt.show()


def main():
    """Main entry point."""
    checker = SanityChecker("./sample")

    # Run validation
    validation_passed = checker.run_validation()

    # Visualize multiple samples (only if validation passed)
    if validation_passed:
        checker.visualize_samples(min_samples=20)

    # Exit with appropriate code
    sys.exit(0 if validation_passed else 1)


if __name__ == "__main__":
    main()
