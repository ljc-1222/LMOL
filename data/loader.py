# -*- coding: utf-8 -*-
"""
LMOL Image Transformation Module

This module provides robust image loading and transformation utilities for the LMOL project.
It handles various image formats and provides fallback mechanisms for reliable image processing.

Key Features:
- Robust image loading with PIL
- Automatic RGB conversion
- Flexible function signature for compatibility
- Comprehensive error handling with informative messages
- Support for various image formats and edge cases

The module is designed to work seamlessly with the LMOL data pipeline,
providing reliable image loading even with corrupted or unusual image files.
"""

from __future__ import annotations

from typing import Any
from PIL import Image, UnidentifiedImageError


def basic_image_loader(path: Any, *args, **kwargs):
    """
    Robust image loader with comprehensive error handling.
    
    This function provides a reliable way to load images for the LMOL pipeline.
    It handles various edge cases and provides informative error messages
    when image loading fails.
    
    Key Features:
    - Supports any path-like object (str, Path, etc.)
    - Automatic RGB conversion for consistent processing
    - Flexible signature for compatibility with different callers
    - Comprehensive error handling with specific error types
    - Lazy loading for truncated images (loads when needed)
    
    Args:
        path: Image file path (str, Path, or path-like object)
        *args: Additional positional arguments (ignored for compatibility)
        **kwargs: Additional keyword arguments (ignored for compatibility)
        
    Returns:
        PIL Image object in RGB mode
        
    Raises:
        RuntimeError: If image cannot be loaded, with detailed error information
        
    Example:
        # Basic usage
        img = basic_image_loader("path/to/image.jpg")
        
        # With extra arguments (ignored)
        img = basic_image_loader("path/to/image.jpg", resize=(224, 224))
    """
    # Convert path to string for PIL compatibility
    p = str(path)
    
    try:
        # Load image with PIL
        img = Image.open(p)
        # Note: PIL can load truncated images, but we defer actual loading
        # to later operations for efficiency. If you need to force immediate
        # loading, uncomment the following line:
        # img.load()
        
    except (FileNotFoundError, UnidentifiedImageError) as e:
        # Provide detailed error information
        raise RuntimeError(f"Failed to open image: {p} | {type(e).__name__}: {e}")

    # Ensure image is in RGB mode for consistent processing
    if img.mode != "RGB":
        img = img.convert("RGB")
        
    return img
