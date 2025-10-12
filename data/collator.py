# -*- coding: utf-8 -*-
"""
LMOL Data Collator Module

This module implements the LlavaPairsCollator for batch processing in LMOL training.
The collator handles the complex task of creating training batches from image pairs
with proper tokenization, masking, and swap doubling.

Key Features:
- Swap doubling: Each pair generates (A,B) and (B,A) samples
- Token-level masking: Only answer tokens contribute to loss
- Robust image processing with fallback handling
- Proper sequence padding and truncation
- Comprehensive batch structure validation

Batch Structure:
- input_ids: (N, L) - Tokenized sequences
- attention_mask: (N, L) - Attention masks
- labels: (N, L) - Labels with IGNORE_INDEX for prompt tokens
- pixel_values: (2N, 3, H, W) - Two images per sample
- label_ids: (N,) - Class labels (0=First, 1=Second, 2=Similar)
- pair_ids: (N,) - Pair identifiers for consistency loss
"""

# Standard library imports
from typing import Any, Dict, List

# Third-party imports
import torch

from configs.config import config
from utils.constants import (
    DEFAULT_PATCH_SIZE,
    IGNORE_INDEX,
    PADDING_VALUE,
    DEFAULT_IMAGE_CHANNELS,
    DEFAULT_IMAGE_DTYPE,
)

class LlavaPairsCollator:
    """
    Data collator for LMOL training with LLaVA.
    
    This collator implements the complete batch processing pipeline for LMOL:
    1. Takes image pairs and creates training samples
    2. Implements swap doubling: (A,B) → [(A,B), (B,A)]
    3. Tokenizes text with proper prompt construction
    4. Creates labels with token-level masking
    5. Processes images and creates pixel values
    6. Handles padding and sequence truncation
    
    Prompt Format:
        <image>*K , QUESTION_TEXT + ' ' + ANSWER + <eos>
    
    Where K is the number of tokens per image (typically 576 for 336x336 images).
    
    Loss Masking:
        Only the answer span (including <eos>) contributes to loss.
        Prompt tokens are masked with IGNORE_INDEX.
    
    Swap Doubling:
        Each original pair (A,B) generates two samples:
        - Original: (A,B) with original label
        - Swapped: (B,A) with swapped label (First↔Second, Similar unchanged)
    
    Batch Shapes:
        - input_ids: (N, L) where N=2*samples, L=sequence_length
        - attention_mask: (N, L) with 1s for valid tokens
        - labels: (N, L) with IGNORE_INDEX for prompt tokens
        - pixel_values: (2N, 3, H, W) with 2 images per sample
        - label_ids: (N,) with class labels 0/1/2
        - pair_ids: (N,) with pair identifiers for consistency loss
    """
    def __init__(self, processor, tokenizer, max_length: int, is_training: bool = True):
        """
        Initialize the LlavaPairsCollator.
        
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

        # Build prompt prefix with image tokens
        image_token = "<image>"
        # Note: No trailing space to ensure consistent tokenization of canonical answers
        self.prompt_prefix = image_token * (self.tokens_per_image * 2) + ", " + config.QUESTION_TEXT

        # Calculate maximum answer length for validation
        prompt_ids = self.tokenizer(self.prompt_prefix, add_special_tokens=True, return_tensors="pt").input_ids[0]
        self.max_answer_len = max_length - len(prompt_ids)

    def __call__(self, features: List[Any]) -> Dict[str, torch.Tensor]:
        """
        Process a batch of features into training-ready tensors.
        
        This method implements the complete batch processing pipeline:
        1. Process each feature (image pair) into two samples (original + swapped)
        2. Tokenize text sequences with proper prompt construction
        3. Create labels with token-level masking
        4. Process images into pixel values
        5. Pad sequences to consistent lengths
        6. Truncate if necessary to respect max_length
        
        Args:
            features: List of PairSample objects containing image pairs and labels
            
        Returns:
            Dictionary containing batched tensors ready for training
        """
        # Initialize lists for batch components
        input_ids_list: List[torch.Tensor] = []
        attention_mask_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        pixel_values_list: List[torch.Tensor] = []
        label_ids_list: List[int] = []
        pair_ids_list: List[int] = []

        # Process each feature (image pair)
        for idx, it in enumerate(features):
            # Process original sample (A, B)
            self._process_sample(
                it.img1, it.img2, it.label,
                input_ids_list, attention_mask_list, labels_list,
                pixel_values_list, label_ids_list, pair_ids_list,
                self.prompt_prefix, idx
            )
            
            # Process swapped sample (B, A) if swap doubling is enabled
            if self.swap_double:
                swapped_label = self.swap_label_map[it.label]
                self._process_sample(
                    it.img2, it.img1, swapped_label,
                    input_ids_list, attention_mask_list, labels_list,
                    pixel_values_list, label_ids_list, pair_ids_list,
                    self.prompt_prefix, idx
                )

        # Pad sequences to consistent lengths
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=PADDING_VALUE)
        labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=IGNORE_INDEX)

        # Truncate sequences if they exceed max_length
        if input_ids.size(1) > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            attention_mask = attention_mask[:, :self.max_length]
            labels = labels[:, :self.max_length]

        # Stack images into pixel values tensor (2N, 3, H, W)
        pixel_values = torch.cat(pixel_values_list, dim=0).to(dtype=torch.float32)

        # Create final batch dictionary
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "label_ids": torch.tensor(label_ids_list, dtype=torch.long),
            "pair_ids": torch.tensor(pair_ids_list, dtype=torch.long),
        }
        return batch

    def _process_sample(
        self, img1, img2, answer_text: str,
        input_ids_list, attention_mask_list, labels_list,
        pixel_values_list, label_ids_list, pair_ids_list,
        prompt_prefix: str, pair_idx: int
    ):
        """
        Process a single sample (image pair + answer) into training components.
        
        This method handles:
        1. Text tokenization with prompt construction
        2. Label creation with token-level masking
        3. Image processing into pixel values
        4. Answer length validation and adjustment
        
        Args:
            img1: First image (PIL Image)
            img2: Second image (PIL Image)
            answer_text: Answer text ("First.", "Second.", or "Similar.")
            input_ids_list: List to append tokenized input IDs
            attention_mask_list: List to append attention masks
            labels_list: List to append label tensors
            pixel_values_list: List to append pixel value tensors
            label_ids_list: List to append label IDs
            pair_ids_list: List to append pair IDs
            prompt_prefix: Pre-built prompt prefix with image tokens
            pair_idx: Index of the current pair
        """
        # Build complete text sequence: prompt + answer
        full_text = prompt_prefix + answer_text
        enc = self.tokenizer(full_text, add_special_tokens=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = enc.input_ids[0]
        
        # Calculate answer length for masking
        ans_ids = self.tokenizer(answer_text + self.tokenizer.eos_token, add_special_tokens=False, return_tensors="pt").input_ids[0]
        ans_len = len(ans_ids)

        # Adjust answer length if necessary
        ans_len = min(ans_len, self.max_answer_len, input_ids.shape[0])

        # Create labels with token-level masking
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        labels[-ans_len:] = input_ids[-ans_len:]  # Only answer tokens contribute to loss

        # Create attention mask (all tokens are valid)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        # Append to batch lists
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

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
