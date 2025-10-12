# -*- coding: utf-8 -*-
"""
LMOL Custom Projector Implementation

This module implements the LMOLProjector, a custom two-layer projector
for vision-text alignment in the LMOL architecture.

Key Features:
- Two-layer design: Linear(1024 → 4096) → GELU → Linear(4096 → 4096)
- Xavier initialization for stable training
- GELU activation with tanh approximation
- Designed for CLIP ViT-L/14-336px → LLaVA-1.5-7B alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LMOLProjector(nn.Module):
    """
    LMOL Custom Projector for Vision-Text Alignment.
    
    This projector implements the LMOL architecture for aligning vision features
    with text representations. It uses a two-layer design consistent with
    LLaVA-1.5 specifications:
    
    Architecture:
        Linear(1024 → 4096) → GELU → Linear(4096 → 4096)
    
    Key Features:
    - Xavier initialization for stable training
    - GELU activation with tanh approximation
    - Preserves sequence length (applied per patch token)
    - Designed for CLIP ViT-L/14-336px → 1024-dim vision features
    - Outputs 4096-dim features compatible with LLaVA-1.5-7B
    
    The projector is trained from scratch while the base LLaVA model
    remains frozen, enabling efficient adaptation to the facial
    attractiveness comparison task.
    """
    def __init__(self, vision_dim: int = 1024, text_hidden_dim: int = 4096):
        """
        Initialize the LMOL projector.
        
        Args:
            vision_dim: Input vision feature dimension (1024 for CLIP ViT-L/14)
            text_hidden_dim: Output text hidden dimension (4096 for LLaVA-1.5-7B)
        """
        super().__init__()
        self.linear_1 = nn.Linear(vision_dim, text_hidden_dim, bias=True)
        self.linear_2 = nn.Linear(text_hidden_dim, text_hidden_dim, bias=True)

        # Xavier initialization for stable training
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.zeros_(self.linear_1.bias)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.zeros_(self.linear_2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LMOL projector.
        
        Args:
            x: Input vision features of shape (B, N, vision_dim)
               where B=batch_size, N=sequence_length, vision_dim=1024
        
        Returns:
            Projected features of shape (B, N, text_hidden_dim)
            where text_hidden_dim=4096
        """
        # First linear transformation
        x = self.linear_1(x)
        # GELU activation with tanh approximation for efficiency
        x = F.gelu(x, approximate="tanh")
        # Second linear transformation
        x = self.linear_2(x)
        return x
