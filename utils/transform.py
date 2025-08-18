# -*- coding: utf-8 -*-

from typing import Callable
from PIL import Image

def basic_image_loader() -> Callable:
    """
    Return a callable that:
      1) Opens a path as PIL.Image.
      2) Converts it to RGB (LLaVA expects RGB).
    Note: LLaVA's image_processor will resize/normalize to 336x336 internally.
    """
    def _load(path: str) -> Image.Image:
        img = Image.open(path).convert("RGB")
        return img
    return _load
