# -*- coding: utf-8 -*-
"""
Classification Data Collator for LMOL

This module implements a proper classification data collator that:
- Only provides the prompt to the model (no answer tokens)
- Creates proper labels for classification (0, 1, 2)
- Prevents answer leakage during training
- Enables proper accuracy computation

Key Features:
- No answer tokens in input sequence
- Model learns to classify rather than generate
- Proper label mapping for 3-class problem
- Swap doubling support for consistency training
"""

from typing import Any, Dict, List
import torch

from configs.config import config
from utils.constants import (
    DEFAULT_PATCH_SIZE,
    IGNORE_INDEX,
    PADDING_VALUE,
    DEFAULT_IMAGE_CHANNELS,
    DEFAULT_IMAGE_DTYPE,
)


class ClassificationCollator:
    """
    Data collator for LMOL classification training.
    
    This collator implements the proper classification approach:
    1. Takes image pairs and creates training samples
    2. Implements swap doubling: (A,B) → [(A,B), (B,A)]
    3. Tokenizes only the prompt (no answer tokens)
    4. Creates classification labels (0, 1, 2)
    5. Processes images and creates pixel values
    6. Handles padding and sequence truncation
    
    Batch Structure:
    - input_ids: (N, L) - Tokenized prompt sequences only
    - attention_mask: (N, L) - Attention masks
    - pixel_values: (2N, 3, H, W) - Two images per sample
    - label_ids: (N,) - Class labels (0=First, 1=Second, 2=Similar)
    - pair_ids: (N,) - Pair identifiers for consistency loss
    """
    
    def __init__(self, processor, tokenizer, max_length: int, is_training: bool = True):
        """
        Initialize the Classification Collator.
        
        Args:
            processor: HuggingFace processor for image and text processing
            tokenizer: HuggingFace tokenizer for text tokenization
            max_length: Maximum sequence length for truncation
            is_training: Whether this collator is used for training (enables swap doubling)
        """
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
        self.swap_double = bool(config.SWAP_DOUBLE and is_training)

        # Calculate tokens per image based on patch size and image dimensions
        ip = getattr(processor, "image_processor", None)
        patch_size = getattr(ip, "patch_size", DEFAULT_PATCH_SIZE) if ip is not None else DEFAULT_PATCH_SIZE
        if isinstance(patch_size, (list, tuple)):
            patch_size = patch_size[0]
        
        # Get image dimensions from processor
        H = getattr(ip, "size", {}).get("height", config.IMAGE_SIZE) if isinstance(getattr(ip, "size", None), dict) else config.IMAGE_SIZE
        W = getattr(ip, "size", {}).get("width", config.IMAGE_SIZE) if isinstance(getattr(ip, "size", None), dict) else config.IMAGE_SIZE
        
        # Calculate tokens per image (e.g., 24*24=576 for 336x336 images with patch_size=14)
        self.tokens_per_image = max(1, (H // patch_size) * (W // patch_size))

        # Create label mapping for answer classes
        self.label_to_id = {
            config.ANSWER_FIRST: 0,
            config.ANSWER_SECOND: 1,
            config.ANSWER_SIMILAR: 2,
        }
        
        # Create swap mapping for swap doubling
        self.swap_label_map = {
            config.ANSWER_FIRST: config.ANSWER_SECOND,
            config.ANSWER_SECOND: config.ANSWER_FIRST,
            config.ANSWER_SIMILAR: config.ANSWER_SIMILAR,  # Similar remains unchanged
        }

        # Ensure pad token exists
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Build prompt prefix with image tokens (NO ANSWER TOKENS)
        image_token = "<image>"
        self.prompt_prefix = image_token * (self.tokens_per_image * 2) + ", " + config.QUESTION_TEXT

    def __call__(self, features: List[Any]) -> Dict[str, torch.Tensor]:
        """
        Process a batch of features into training-ready tensors.
        
        Args:
            features: List of PairSample objects containing image pairs and labels
            
        Returns:
            Dictionary containing batched tensors ready for classification training
        """
        # Initialize lists for batch components
        input_ids_list: List[torch.Tensor] = []
        attention_mask_list: List[torch.Tensor] = []
        pixel_values_list: List[torch.Tensor] = []
        label_ids_list: List[int] = []
        pair_ids_list: List[int] = []

        # Process each feature (image pair)
        for idx, it in enumerate(features):
            # Process original sample (A, B)
            self._process_sample(
                it.img1, it.img2, it.label,
                input_ids_list, attention_mask_list,
                pixel_values_list, label_ids_list, pair_ids_list,
                self.prompt_prefix, idx
            )
            
            # Process swapped sample (B, A) if swap doubling is enabled
            if self.swap_double:
                swapped_label = self.swap_label_map[it.label]
                self._process_sample(
                    it.img2, it.img1, swapped_label,
                    input_ids_list, attention_mask_list,
                    pixel_values_list, label_ids_list, pair_ids_list,
                    self.prompt_prefix, idx
                )

        # Pad sequences to consistent lengths
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=PADDING_VALUE)

        # Truncate sequences if they exceed max_length
        if input_ids.size(1) > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            attention_mask = attention_mask[:, :self.max_length]

        # Stack images into pixel values tensor (2N, 3, H, W)
        pixel_values = torch.cat(pixel_values_list, dim=0).to(dtype=torch.float32)

        # Create final batch dictionary
        label_ids_tensor = torch.tensor(label_ids_list, dtype=torch.long)
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "label_ids": label_ids_tensor,
            "labels": label_ids_tensor,  # Add labels key for compatibility with sanity check
            "pair_ids": torch.tensor(pair_ids_list, dtype=torch.long),
        }
        return batch

    def _process_sample(
        self, img1, img2, answer_text: str,
        input_ids_list, attention_mask_list,
        pixel_values_list, label_ids_list, pair_ids_list,
        prompt_prefix: str, pair_idx: int
    ):
        """
        Process a single sample (image pair + answer) into training components.
        
        Args:
            img1: First image (PIL Image)
            img2: Second image (PIL Image)
            answer_text: Answer text ("First", "Second", or "Similar")
            input_ids_list: List to append tokenized input IDs
            attention_mask_list: List to append attention masks
            pixel_values_list: List to append pixel value tensors
            label_ids_list: List to append label IDs
            pair_ids_list: List to append pair IDs
            prompt_prefix: Pre-built prompt prefix with image tokens
            pair_idx: Index of the current pair
        """
        # CORRECT APPROACH: Only tokenize the prompt, no answer tokens!
        # The model learns to classify the answer, not generate it
        prompt_enc = self.tokenizer(prompt_prefix, add_special_tokens=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = prompt_enc.input_ids[0]
        
        # Create attention mask (all tokens are valid)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        # Append to batch lists
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)

        # Process images into pixel values
        try:
            # Process both images together for efficiency
            pv = self.processor.image_processor([img1, img2], return_tensors="pt").pixel_values  # (2,3,H,W)
            if pv.dtype != torch.float32:
                pv = pv.to(dtype=torch.float32)
            pixel_values_list.append(pv)
        except Exception:
            # Fallback: create empty pixel values if image processing fails
            H, W = config.IMAGE_SIZE, config.IMAGE_SIZE
            empty_pv = torch.zeros((2, DEFAULT_IMAGE_CHANNELS, H, W), dtype=torch.float32)
            pixel_values_list.append(empty_pv)

        # Append metadata
        label_ids_list.append(self.label_to_id[answer_text])
        pair_ids_list.append(pair_idx)
