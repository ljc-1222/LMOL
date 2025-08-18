# -*- coding: utf-8 -*-

from typing import Any, Dict, List
import torch

from configs.config import config

class LlavaPairsCollator:
    """
    Collator for LMOL with LLaVA-1.5:
    - Compose a prompt with two <image> placeholders followed by the exact question:
        "<image><image>, which face looks more attractive?"
      (paper uses "⟨image1⟩⟨image2⟩, which face looks more attractive?")
    - Concatenate the answer string so that labels only supervise the answer tokens.
    - Use processor to build pixel_values; use tokenizer to create input_ids.
    """

    def __init__(self, processor, tokenizer, max_length: int):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Build the exact prompt text with two image tokens + comma + space
        self.prompt_text = "<image><image>, " + config.QUESTION_TEXT

    def __call__(self, features: List[Any]) -> Dict[str, torch.Tensor]:
        batch_images: List[List[Any]] = []  # each item holds [img1, img2]
        input_ids_list: List[torch.Tensor] = []
        attention_mask_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []

        for it in features:
            img_pair = [it.img1, it.img2]
            batch_images.append(img_pair)

            # Prepare the full supervised text: prompt + answer (+ eos)
            answer_text = it.label  # already one of "First.", "Second.", "Similar."
            full_text = self.prompt_text + " " + answer_text

            # Tokenize the full text using the tokenizer (not the chat template).
            # We want to control masking: labels only on the trailing answer tokens.
            input_ids = self.tokenizer(
                full_text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).input_ids[0]

            # Compute the length of answer tokens (with eos appended)
            ans_ids = self.tokenizer(
                answer_text + self.tokenizer.eos_token,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length
            ).input_ids
            ans_len = len(ans_ids)

            # Build labels: -100 for prompt tokens, gold for answer tokens
            labels = torch.full_like(input_ids, fill_value=-100)
            labels[-ans_len:] = input_ids[-ans_len:]

            # Build attention mask
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        # Pad sequences to the same length
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)

        # Process images with processor.image_processor; we pass two images per sample.
        # LLaVA expects number of <image> tokens equals len(images) in each sample.
        all_pixel_values = []
        for img1, img2 in batch_images:
            enc = self.processor.image_processor([img1, img2], return_tensors="pt")
            # For LLaVA, stack into a single tensor per sample with shape (num_images, 3, H, W)
            pixel_values = enc["pixel_values"]  # shape (2, 3, H, W)
            all_pixel_values.append(pixel_values)

        # Pad image sequences into a batch; create a list to keep variable-length per sample
        # Many Trainer configs support passing a list of tensors for multi-image inputs.
        # If your local transformers requires a different key (e.g., "images"), adapt here.
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": all_pixel_values,  # list of (2,3,H,W) tensors
        }
        return batch
