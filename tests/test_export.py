"""Tests for the export module."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from bsort.export import export_to_onnx, validate_onnx_model


# ====================================================================
# Tests for export_to_onnx()
# ====================================================================


def test_export_missing_model_file():
    """Test export_to_onnx with non-existent .pt file."""
    non_existent_model = "/tmp/nonexistent_model_123456.pt"

    # Should raise FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        export_to_onnx(non_existent_model)


@patch("bsort.export.YOLO")
def test_export_success_default_output(mock_yolo):
    """Test successful export with default output path."""
    # Create a temporary model file
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        model_path = f.name

    try:
        # Mock the YOLO model
        mock_model = Mock()
        mock_model.export.return_value = str(Path(model_path).with_suffix(".onnx"))
        mock_yolo.return_value = mock_model

        # Export
        result = export_to_onnx(model_path, imgsz=224, opset=12, simplify=True)

        # Verify YOLO was instantiated with model path
        mock_yolo.assert_called_once_with(model_path)

        # Verify export was called with correct parameters
        mock_model.export.assert_called_once_with(
            format="onnx", imgsz=224, opset=12, simplify=True, dynamic=False
        )

        # Verify result is the exported path
        assert result == str(Path(model_path).with_suffix(".onnx"))

    finally:
        # Cleanup
        Path(model_path).unlink(missing_ok=True)


@patch("bsort.export.YOLO")
@patch("shutil.move")
def test_export_custom_output_path(mock_move, mock_yolo):
    """Test export with custom output path."""
    # Create a temporary model file
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        model_path = f.name

    try:
        # Mock the YOLO model
        mock_model = Mock()
        default_export_path = str(Path(model_path).with_suffix(".onnx"))
        mock_model.export.return_value = default_export_path
        mock_yolo.return_value = mock_model

        custom_output = "/tmp/custom_model.onnx"

        # Export with custom output path
        result = export_to_onnx(model_path, output_path=custom_output)

        # Verify shutil.move was called to move file to custom location
        mock_move.assert_called_once_with(default_export_path, custom_output)

        # Verify result is the custom path
        assert result == custom_output

    finally:
        # Cleanup
        Path(model_path).unlink(missing_ok=True)


@patch("bsort.export.YOLO")
def test_export_creates_parent_directories(mock_yolo):
    """Test that export creates parent directories for output path."""
    # Create a temporary model file
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        model_path = f.name

    try:
        # Mock the YOLO model
        mock_model = Mock()
        default_export_path = str(Path(model_path).with_suffix(".onnx"))
        mock_model.export.return_value = default_export_path
        mock_yolo.return_value = mock_model

        # Use a temporary directory for output
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_output = Path(tmpdir) / "nested" / "dir" / "model.onnx"

            # Export - should create nested directories
            with patch("shutil.move"):
                result = export_to_onnx(model_path, output_path=str(custom_output))

            # Verify parent directories were created
            assert custom_output.parent.exists()

    finally:
        # Cleanup
        Path(model_path).unlink(missing_ok=True)


@patch("bsort.export.YOLO")
def test_export_parameters_passed_correctly(mock_yolo):
    """Test that export parameters are passed correctly to model.export()."""
    # Create a temporary model file
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        model_path = f.name

    try:
        # Mock the YOLO model
        mock_model = Mock()
        mock_model.export.return_value = str(Path(model_path).with_suffix(".onnx"))
        mock_yolo.return_value = mock_model

        # Export with specific parameters
        export_to_onnx(model_path, imgsz=640, opset=14, simplify=False)

        # Verify export was called with all correct parameters
        mock_model.export.assert_called_once_with(
            format="onnx", imgsz=640, opset=14, simplify=False, dynamic=False
        )

    finally:
        # Cleanup
        Path(model_path).unlink(missing_ok=True)


@patch("bsort.export.YOLO")
def test_export_handles_export_failure(mock_yolo):
    """Test that export handles model.export() failures gracefully."""
    # Create a temporary model file
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        model_path = f.name

    try:
        # Mock the YOLO model to raise an exception
        mock_model = Mock()
        mock_model.export.side_effect = Exception("Export failed for some reason")
        mock_yolo.return_value = mock_model

        # Should raise RuntimeError with descriptive message
        with pytest.raises(RuntimeError, match="ONNX export failed"):
            export_to_onnx(model_path)

    finally:
        # Cleanup
        Path(model_path).unlink(missing_ok=True)


@patch("bsort.export.YOLO")
@patch("shutil.move")
def test_export_same_path_no_move(mock_move, mock_yolo):
    """Test that no file move occurs when output path matches export path."""
    # Create a temporary model file
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        model_path = f.name

    try:
        # Mock the YOLO model
        mock_model = Mock()
        export_path = str(Path(model_path).with_suffix(".onnx"))
        mock_model.export.return_value = export_path
        mock_yolo.return_value = mock_model

        # Export with output path same as default export path
        result = export_to_onnx(model_path, output_path=export_path)

        # Verify shutil.move was NOT called (paths are the same)
        mock_move.assert_not_called()

        # Verify result is the export path
        assert result == export_path

    finally:
        # Cleanup
        Path(model_path).unlink(missing_ok=True)


# ====================================================================
# Tests for validate_onnx_model()
# ====================================================================


def test_validate_onnx_valid_model():
    """Test validation of a valid ONNX model."""
    import sys

    # Create mock onnx module
    mock_onnx = Mock()
    mock_model = Mock()
    mock_onnx.load.return_value = mock_model
    mock_onnx.checker.check_model.return_value = None  # No exception = valid

    # Temporarily inject mock onnx into sys.modules
    with patch.dict(sys.modules, {"onnx": mock_onnx}):
        result = validate_onnx_model("/tmp/test_model.onnx")

        # Verify validation passed
        assert result is True

        # Verify onnx functions were called
        mock_onnx.load.assert_called_once_with("/tmp/test_model.onnx")
        mock_onnx.checker.check_model.assert_called_once_with(mock_model)


def test_validate_onnx_invalid_model():
    """Test validation of an invalid ONNX model."""
    import sys

    # Create mock onnx module that raises validation error
    mock_onnx = Mock()
    mock_onnx.load.return_value = Mock()
    mock_onnx.checker.check_model.side_effect = Exception("Invalid ONNX model")

    # Temporarily inject mock onnx into sys.modules
    with patch.dict(sys.modules, {"onnx": mock_onnx}):
        result = validate_onnx_model("/tmp/invalid_model.onnx")

        # Verify validation failed
        assert result is False


def test_validate_onnx_missing_library():
    """Test validation when onnx package is not installed."""
    import builtins
    import sys

    # Remove onnx from sys.modules if it exists
    original_onnx = sys.modules.pop("onnx", None)

    try:
        # Save original __import__
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "onnx":
                raise ImportError("No module named 'onnx'")
            return original_import(name, *args, **kwargs)

        # Patch builtins.__import__
        builtins.__import__ = mock_import

        try:
            result = validate_onnx_model("/tmp/test.onnx")

            # When onnx is missing, function returns True (skips validation gracefully)
            assert result is True
        finally:
            # Restore original __import__
            builtins.__import__ = original_import

    finally:
        # Restore original onnx module if it existed
        if original_onnx is not None:
            sys.modules["onnx"] = original_onnx
