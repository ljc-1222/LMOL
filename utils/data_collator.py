# -*- coding: utf-8 -*-

from typing import Any, Dict, List
import torch

from configs.config import config

class LlavaPairsCollator:
    """
    Collator for LMOL with LLaVA-1.5 (HF implementation expectations):
    - Must supply as many <image> placeholder tokens as produced visual patch tokens.
      Otherwise: ValueError: Image features and image tokens do not match.
    - We estimate patch token count = (H/patch_size)*(W/patch_size) where H=W=config.IMAGE_SIZE.
      Default CLIP ViT-L/14 -> patch_size=14 => 24*24=576 tokens per image (no CLS for LLaVA projector).
    - For 2 images => 1152 <image> tokens at sequence start, then comma + question.
    - Provide pixel_values as list[Tensor(num_images,3,H,W)].
    - Mask labels so only answer tokens (incl. eos) are supervised.
    """

    def __init__(self, processor, tokenizer, max_length: int):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Try to read patch size; fall back to 14
        patch_size = getattr(getattr(processor, 'image_processor', {}), 'patch_size', 14)
        if isinstance(patch_size, (list, tuple)):
            patch_size = patch_size[0]
        if isinstance(patch_size, dict):  # unlikely
            patch_size = patch_size.get('height', 14)
        self.patch_size = patch_size or 14
        self.image_size = getattr(getattr(processor, 'image_processor', {}), 'size', {})
        if isinstance(self.image_size, dict):
            h = self.image_size.get('height') or self.image_size.get('shortest_edge') or config.IMAGE_SIZE
            w = self.image_size.get('width') or self.image_size.get('shortest_edge') or config.IMAGE_SIZE
        else:
            h = w = config.IMAGE_SIZE
        self.h = h
        self.w = w
        # Precompute tokens per image
        self.tokens_per_image = (self.h // self.patch_size) * (self.w // self.patch_size)
        # Sanity lower bound
        if self.tokens_per_image <= 0:
            self.tokens_per_image = 576  # fallback typical value

    def build_prompt(self) -> str:
        # Repeat <image> tokens for both images
        total_img_tokens = self.tokens_per_image * 2
        placeholder = "<image>" * total_img_tokens
        return placeholder + ", " + config.QUESTION_TEXT

    def __call__(self, features: List[Any]) -> Dict[str, Any]:  # return values include lists
        input_ids_list: List[torch.Tensor] = []
        attention_mask_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []
        pixel_values_list: List[torch.Tensor] = []

        prompt_prefix = self.build_prompt()

        for it in features:
            answer_text = it.label
            full_text = prompt_prefix + " " + answer_text
            tok = self.tokenizer(
                full_text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            input_ids = tok.input_ids[0]
            # Determine answer token span length (answer + eos)
            ans_ids = self.tokenizer(
                answer_text + self.tokenizer.eos_token,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length
            ).input_ids
            ans_len = len(ans_ids)
            labels = torch.full_like(input_ids, -100)
            labels[-ans_len:] = input_ids[-ans_len:]
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

            pv = self.processor.image_processor([it.img1, it.img2], return_tensors="pt").pixel_values
            pixel_values_list.append(pv)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask_list, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )

        # Concatenate pixel_values into a single tensor
        pixel_values = torch.cat(pixel_values_list, dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }
