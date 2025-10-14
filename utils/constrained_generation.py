#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constrained Generation Utilities for LMOL

This module provides utilities for constraining model outputs to valid answer tokens.
It implements logits processors that can be used with HuggingFace's generate() method
to ensure the model only produces valid classification labels.

Key Classes:
- RestrictedVocabLogitsProcessor: Forces generation of specific tokens only
- AnswerConstraintProcessor: Specialized processor for LMOL answer tokens

Usage:
    from utils.constrained_generation import AnswerConstraintProcessor
    
    # Create processor for valid answer tokens
    processor = AnswerConstraintProcessor(tokenizer)
    
    # Generate with constraints
    output = model.generate(
        ...,
        logits_processor=[processor],
        max_new_tokens=1,  # Only need 1 token for classification
    )
"""

from typing import List, Optional
import torch
from transformers.generation.logits_process import LogitsProcessor

from configs.config import config


class RestrictedVocabLogitsProcessor(LogitsProcessor):
    """
    Logits processor that restricts generation to specific tokens.
    
    This processor modifies the logits tensor by setting all tokens outside
    the allowed set to -inf, effectively forcing the model to only generate
    from the specified vocabulary.
    
    This is the general-purpose implementation that works with any token set.
    For LMOL-specific answer constraint, use AnswerConstraintProcessor.
    
    Attributes:
        allowed_token_ids: Tensor of allowed token IDs
        
    Example:
        >>> # Allow only tokens 100, 200, 300
        >>> processor = RestrictedVocabLogitsProcessor([100, 200, 300])
        >>> 
        >>> # Use in generation
        >>> output = model.generate(
        ...     input_ids=inputs,
        ...     logits_processor=[processor],
        ...     max_new_tokens=10,
        ... )
        >>> # Output will only contain tokens 100, 200, 300
    """
    
    def __init__(self, allowed_token_ids: List[int]):
        """
        Initialize the processor with allowed token IDs.
        
        Args:
            allowed_token_ids: List of token IDs that are allowed to be generated.
                             All other tokens will be suppressed.
                             
        Raises:
            ValueError: If allowed_token_ids is empty
        """
        if not allowed_token_ids:
            raise ValueError("allowed_token_ids cannot be empty")
        
        self.allowed_token_ids = torch.tensor(allowed_token_ids, dtype=torch.long)
        
    def __call__(
        self, 
        input_ids: torch.LongTensor, 
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Process logits to restrict vocabulary.
        
        This method is called by the generate() function at each decoding step.
        It modifies the logits (scores) to suppress invalid tokens.
        
        Args:
            input_ids: Previously generated token IDs.
                      Shape: (batch_size, sequence_length)
                      Note: Not used in this implementation, but required by interface
            scores: Logits for next token prediction.
                   Shape: (batch_size, vocab_size)
                   
        Returns:
            Modified scores with invalid tokens set to -inf.
            Shape: (batch_size, vocab_size)
            
        Note:
            Setting logits to -inf ensures zero probability after softmax:
            softmax(-inf) = exp(-inf) / sum(exp(logits)) = 0 / Z = 0
        """
        # Move allowed tokens to same device as scores
        allowed_tokens = self.allowed_token_ids.to(scores.device)
        
        # Create mask: True for tokens to suppress, False for allowed tokens
        mask = torch.ones(scores.shape[-1], dtype=torch.bool, device=scores.device)
        mask[allowed_tokens] = False
        
        # Set invalid token logits to -inf
        # This ensures they have zero probability after softmax
        scores[:, mask] = float('-inf')
        
        return scores


class AnswerConstraintProcessor(LogitsProcessor):
    """
    LMOL-specific logits processor for answer token constraints.
    
    This processor is specialized for the LMOL task, constraining outputs
    to only the three valid answer tokens: "First", "Second", "Similar".
    
    It automatically extracts token IDs from the tokenizer based on the
    answer strings defined in config, making it easier to use than the
    general RestrictedVocabLogitsProcessor.
    
    Attributes:
        tokenizer: HuggingFace tokenizer
        answer_token_ids: List of valid answer token IDs [First, Second, Similar]
        
    Example:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>> 
        >>> # Create answer constraint processor
        >>> processor = AnswerConstraintProcessor(tokenizer)
        >>> 
        >>> # Use in generation
        >>> output = model.generate(
        ...     input_ids=inputs,
        ...     pixel_values=images,
        ...     logits_processor=[processor],
        ...     max_new_tokens=1,  # Only 1 token needed for classification
        ... )
    """
    
    def __init__(self, tokenizer, verbose: bool = False):
        """
        Initialize the processor with tokenizer.
        
        Args:
            tokenizer: HuggingFace tokenizer to extract answer token IDs
            verbose: If True, print token IDs during initialization
            
        Raises:
            AssertionError: If tokenization produces invalid token IDs
        """
        self.tokenizer = tokenizer
        
        # Extract answer token IDs
        # NOTE: Must tokenize with leading space to match collator behavior
        first_tokens = tokenizer(' ' + config.ANSWER_FIRST, add_special_tokens=False).input_ids
        second_tokens = tokenizer(' ' + config.ANSWER_SECOND, add_special_tokens=False).input_ids
        similar_tokens = tokenizer(' ' + config.ANSWER_SIMILAR, add_special_tokens=False).input_ids
        
        # Use last token (the actual word token after space token)
        # With LLaMA tokenizer:
        # " First"   -> [29871, 3824]  -> take 3824
        # " Second"  -> [29871, 6440]  -> take 6440
        # " Similar" -> [29871, 13999] -> take 13999
        first_id = first_tokens[-1] if first_tokens else -1
        second_id = second_tokens[-1] if second_tokens else -1
        similar_id = similar_tokens[-1] if similar_tokens else -1
        
        # Validate token IDs
        assert first_id > 0, f"Invalid First token ID: {first_id}"
        assert second_id > 0, f"Invalid Second token ID: {second_id}"
        assert similar_id > 0, f"Invalid Similar token ID: {similar_id}"
        
        # Ensure uniqueness
        assert len({first_id, second_id, similar_id}) == 3, \
            f"Answer tokens are not unique: First={first_id}, Second={second_id}, Similar={similar_id}"
        
        self.answer_token_ids = [first_id, second_id, similar_id]
        
        if verbose:
            print(f"[AnswerConstraintProcessor] Initialized with token IDs:")
            print(f"  First:   {first_id} = '{tokenizer.decode([first_id])}'")
            print(f"  Second:  {second_id} = '{tokenizer.decode([second_id])}'")
            print(f"  Similar: {similar_id} = '{tokenizer.decode([similar_id])}'")
        
        # Create underlying restricted vocab processor
        self._processor = RestrictedVocabLogitsProcessor(self.answer_token_ids)
    
    def __call__(
        self, 
        input_ids: torch.LongTensor, 
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Process logits to restrict to answer tokens only.
        
        Args:
            input_ids: Previously generated token IDs (unused)
            scores: Logits for next token prediction
            
        Returns:
            Modified scores with only answer tokens allowed
        """
        return self._processor(input_ids, scores)
    
    def get_answer_from_token_id(self, token_id: int) -> Optional[str]:
        """
        Convert token ID back to answer string.
        
        Args:
            token_id: Token ID to decode
            
        Returns:
            Answer string ("First", "Second", or "Similar"), or None if invalid
            
        Example:
            >>> processor = AnswerConstraintProcessor(tokenizer)
            >>> processor.get_answer_from_token_id(3824)
            'First'
            >>> processor.get_answer_from_token_id(99999)
            None
        """
        if token_id not in self.answer_token_ids:
            return None
        
        idx = self.answer_token_ids.index(token_id)
        return [config.ANSWER_FIRST, config.ANSWER_SECOND, config.ANSWER_SIMILAR][idx]


def generate_with_answer_constraint(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    max_new_tokens: int = 1,
    **kwargs
) -> str:
    """
    Convenience function for generating constrained answers.
    
    This function wraps model.generate() with automatic answer constraint
    processing, making it easy to generate valid answers.
    
    Args:
        model: LLaVA model
        tokenizer: HuggingFace tokenizer
        input_ids: Input token IDs, shape: (1, seq_len)
        pixel_values: Image tensor, shape: (1, num_images, C, H, W)
        max_new_tokens: Maximum new tokens to generate (default: 1 for classification)
        **kwargs: Additional arguments passed to model.generate()
        
    Returns:
        Generated answer string ("First", "Second", or "Similar")
        
    Example:
        >>> answer = generate_with_answer_constraint(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     input_ids=input_ids,
        ...     pixel_values=pixel_values,
        ... )
        >>> print(answer)  # Guaranteed to be "First", "Second", or "Similar"
        'First'
    """
    # Create answer constraint processor
    processor = AnswerConstraintProcessor(tokenizer, verbose=False)
    
    # Generate with constraint
    output_ids = model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        max_new_tokens=max_new_tokens,
        logits_processor=[processor],
        do_sample=False,  # Use greedy decoding for deterministic results
        **kwargs
    )
    
    # Extract generated token (last token)
    generated_token_id = output_ids[0, -1].item()
    
    # Convert to answer string
    answer = processor.get_answer_from_token_id(generated_token_id)
    
    # Defensive check (should never fail with constraint processor)
    if answer is None:
        raise RuntimeError(
            f"Generated invalid token ID: {generated_token_id}. "
            f"This should not happen with AnswerConstraintProcessor!"
        )
    
    return answer


def extract_answer_logits(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
) -> torch.Tensor:
    """
    Extract logits for the three answer tokens without generation.
    
    This function performs a single forward pass and extracts the logits
    for the three valid answer tokens, useful for confidence estimation
    and analysis.
    
    Args:
        model: LLaVA model
        tokenizer: HuggingFace tokenizer
        input_ids: Input token IDs, shape: (1, seq_len)
        pixel_values: Image tensor, shape: (1, num_images, C, H, W)
        
    Returns:
        Tensor of shape (3,) containing logits for [First, Second, Similar]
        
    Example:
        >>> logits = extract_answer_logits(model, tokenizer, input_ids, pixel_values)
        >>> probs = torch.softmax(logits, dim=0)
        >>> print(f"P(First)={probs[0]:.3f}, P(Second)={probs[1]:.3f}, P(Similar)={probs[2]:.3f}")
        P(First)=0.123, P(Second)=0.789, P(Similar)=0.088
    """
    # Get token IDs for answers
    first_tokens = tokenizer(' ' + config.ANSWER_FIRST, add_special_tokens=False).input_ids
    second_tokens = tokenizer(' ' + config.ANSWER_SECOND, add_special_tokens=False).input_ids
    similar_tokens = tokenizer(' ' + config.ANSWER_SIMILAR, add_special_tokens=False).input_ids
    
    first_id = first_tokens[-1]
    second_id = second_tokens[-1]
    similar_id = similar_tokens[-1]
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
        )
        
        # Get logits for last position (where answer is generated)
        last_logits = outputs.logits[0, -1, :]  # (vocab_size,)
        
        # Extract answer token logits
        answer_logits = torch.stack([
            last_logits[first_id],
            last_logits[second_id],
            last_logits[similar_id],
        ])
    
    return answer_logits


def get_answer_confidence(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
) -> tuple[str, float]:
    """
    Generate answer with confidence score.
    
    Combines constrained generation with confidence estimation to provide
    both the predicted answer and the model's confidence in that answer.
    
    Args:
        model: LLaVA model
        tokenizer: HuggingFace tokenizer
        input_ids: Input token IDs
        pixel_values: Image tensor
        
    Returns:
        Tuple of (answer, confidence) where:
        - answer: "First", "Second", or "Similar"
        - confidence: float in [0, 1] representing model confidence
        
    Example:
        >>> answer, confidence = get_answer_confidence(model, tokenizer, input_ids, pixel_values)
        >>> print(f"Answer: {answer} (confidence: {confidence:.2%})")
        Answer: First (confidence: 78.92%)
    """
    # Extract answer logits
    answer_logits = extract_answer_logits(model, tokenizer, input_ids, pixel_values)
    
    # Compute probabilities
    probs = torch.softmax(answer_logits, dim=0)
    
    # Get predicted class
    pred_idx = probs.argmax().item()
    confidence = probs[pred_idx].item()
    
    # Map to answer string
    answers = [config.ANSWER_FIRST, config.ANSWER_SECOND, config.ANSWER_SIMILAR]
    answer = answers[pred_idx]
    
    return answer, confidence

