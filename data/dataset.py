# -*- coding: utf-8 -*-
"""
LMOL Dataset Module

This module implements the SCUT_FBP5500_Pairs dataset for LMOL training.
It provides a PyTorch Dataset interface for loading image pairs with
their corresponding attractiveness comparison labels.

Key Features:
- PyTorch Dataset interface for seamless integration with DataLoader
- Flexible image loading with customizable loaders
- Robust error handling for missing or corrupted images
- Support for various image formats and configurations
- Efficient memory usage with lazy loading

Dataset Structure:
- Each sample contains two images and a comparison label
- Labels: "First.", "Second.", or "Similar."
- Images are loaded on-demand for memory efficiency
- Supports custom image loaders for different preprocessing needs

The dataset is designed to work seamlessly with the LMOL training
pipeline and data collator for efficient batch processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Any
from torch.utils.data import Dataset
from .processor import PairRecord, read_pairs_csv
from .loader import basic_image_loader


@dataclass
class PairSample:
    """
    Data structure representing a single training sample.
    
    This class encapsulates the data for a single training sample:
    two images and their comparison label. Images are loaded as PIL
    Image objects for compatibility with the data collator.
    
    Attributes:
        img1: First image (PIL Image object)
        img2: Second image (PIL Image object)
        label: Comparison label ("First.", "Second.", or "Similar.")
    """
    img1: Any  # PIL.Image.Image
    img2: Any  # PIL.Image.Image
    label: str


class SCUT_FBP5500_Pairs(Dataset):
    """
    PyTorch Dataset for SCUT-FBP5500 facial attractiveness comparison pairs.
    
    This dataset loads image pairs from CSV files and provides them
    as training samples for the LMOL model. It implements the standard
    PyTorch Dataset interface for seamless integration with DataLoader.
    
    Key Features:
    - Lazy loading: Images are loaded on-demand for memory efficiency
    - Flexible loader: Supports custom image loading functions
    - Robust error handling: Graceful handling of missing/corrupted images
    - CSV integration: Reads pair data from CSV files with scores and labels
    
    Usage:
        # Basic usage with default image loader
        dataset = SCUT_FBP5500_Pairs("path/to/pairs.csv")
        
        # With custom image loader
        dataset = SCUT_FBP5500_Pairs("path/to/pairs.csv", image_loader=custom_loader)
        
        # Use with DataLoader
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    """
    
    def __init__(self, csv_path: str, image_loader: Optional[Callable[..., Any]] = None):
        """
        Initialize the SCUT-FBP5500 pairs dataset.
        
        Args:
            csv_path: Path to CSV file containing image pair data
            image_loader: Optional custom image loading function.
                        If None, uses basic_image_loader with robust error handling.
        """
        super().__init__()
        
        # Load pair records from CSV
        self.records = read_pairs_csv(csv_path)
        
        # Set up image loader (default to robust loader)
        self.loader: Callable[..., Any] = image_loader or basic_image_loader

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        
        Returns:
            Number of image pairs in the dataset
        """
        return len(self.records)

    def __getitem__(self, idx: int) -> PairSample:
        """
        Get a single training sample by index.
        
        This method loads the images for the specified pair and returns
        a PairSample object containing both images and the label.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            PairSample object containing two images and comparison label
            
        Raises:
            IndexError: If idx is out of range
            RuntimeError: If image loading fails (from image_loader)
        """
        # Get the pair record
        r: PairRecord = self.records[idx]
        
        # Load images using the configured loader
        # Always pass a single string path argument for compatibility
        img1 = self.loader(str(r.img1))
        img2 = self.loader(str(r.img2))
        
        # Return PairSample with loaded images and label
        return PairSample(img1=img1, img2=img2, label=r.label)
