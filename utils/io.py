# -*- coding: utf-8 -*-
"""
LMOL I/O Utilities

This module provides I/O utilities for the LMOL project:
- Path resolution and validation
- File existence checking
- Path manipulation utilities

Key Features:
- Robust path resolution
- File existence validation
- Path object conversion
"""

from pathlib import Path
from typing import Union


def resolve_path(p: Union[str, Path]) -> Path:
    """
    Convert string path to Path object if needed.
    
    Args:
        p: Path string or Path object
        
    Returns:
        Path object
    """
    return p if isinstance(p, Path) else Path(p)


def ensure_exists(p: Path) -> Path:
    """
    Ensure path exists, raise error if not.
    
    Args:
        p: Path to check
        
    Returns:
        Path object if exists
        
    Raises:
        FileNotFoundError: If path does not exist
    """
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return p
