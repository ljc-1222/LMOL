# LMOL Technical Reference

**⚠️ READ THIS BEFORE MAKING ANY CHANGES TO THE CODEBASE ⚠️**

This document contains critical information about the LMOL project architecture, a major tokenization bug that was fixed, and important implementation details. Always consult this file before modifying the codebase.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Critical Bug: Tokenization Mismatch](#critical-bug-tokenization-mismatch)
4. [Training Pipeline](#training-pipeline)
5. [Evaluation Pipeline](#evaluation-pipeline)
6. [Key Implementation Details](#key-implementation-details)
7. [Common Pitfalls](#common-pitfalls)
8. [Validation & Testing](#validation--testing)

---

## Project Overview

**LMOL (Large Multimodal model for facial attractiveness comparison)**

- **Task:** Compare facial attractiveness between two images
- **Base Model:** LLaVA-1.5-7B (llava-hf/llava-1.5-7b-hf)
- **Approach:** LoRA fine-tuning with custom projector
- **Dataset:** SCUT-FBP5500 facial beauty dataset
- **Classes:** 3-way classification
  - "First" - Left image more attractive
  - "Second" - Right image more attractive  
  - "Similar" - Both equally attractive

### Key Features

1. **Swap Consistency Loss:** Regularization ensuring (A,B) and (B,A) predictions are consistent
2. **Swap Doubling:** Each pair generates 2 training samples (original + swapped)
3. **Weighted Cross-Entropy:** Higher weight for "Similar" class
4. **Dynamic Consistency Weight:** Gradually increases during training
5. **Dual Learning Rates:** Different LRs for projector (from scratch) vs LoRA (fine-tuning)

---

## Architecture

### Directory Structure

```
LMOL/
├── configs/
│   └── config.py              # All configuration parameters
├── data/
│   ├── collator.py            # ⚠️ CRITICAL: Batch collation and tokenization
│   ├── dataset.py             # Dataset loading
│   └── pairs/                 # Train/eval CSV files
├── model/
│   ├── builder.py             # Model initialization
│   └── checkpoints/           # Saved model weights
├── training/
│   ├── trainer.py             # ⚠️ CRITICAL: Custom trainer with accuracy computation
│   ├── optimizer.py           # Dual LR optimizer setup
│   ├── callbacks.py           # Training callbacks
│   └── main.py                # Training orchestration
├── evaluation/
│   ├── evaluator.py           # ⚠️ CRITICAL: Inference and answer parsing
│   └── main.py                # Evaluation orchestration
├── scripts/
│   ├── train.py               # Training entry point
│   ├── evaluate.py            # Evaluation entry point
│   └── validate_tokenization.py  # ⚠️ RUN THIS BEFORE TRAINING
└── utils/
    └── constants.py           # IGNORE_INDEX = -100, etc.
```

### Model Architecture

```
Input: Two 336x336 images + Text prompt
  ↓
[CLIP ViT-L/14] → Image embeddings (576 tokens each, 1152 total)
  ↓
[Projector] → Project to LLaMA space (trainable, LR=2e-3)
  ↓
[LLaMA-2-7B + LoRA] → Language model (LoRA trainable, LR=1e-4)
  ↓
Output: Single token prediction ("First"/"Second"/"Similar")
```

---

## Critical Bug: Tokenization Mismatch

### ⚠️ THE BUG THAT CAUSED ZERO ACCURACY

**Fixed on:** 2025-10-13

**Symptom:**
- Training loss: ~3e-7 (very low)
- Training accuracy: 0.0000 (always zero)
- Evaluation: Model generates "ilarilarilarilar" repeatedly

### Root Cause

**Location:** `/root/LMOL/data/collator.py`, line 233

**Buggy Code (BEFORE FIX):**
```python
# Line 233 (OLD - BUGGY)
full_text = prompt_prefix + answer_text  # ❌ NO SPACE!
```

This created: `"...attractive)First"` instead of `"...attractive) First"`

**Impact on Tokenization:**

| Answer | Without Space (WRONG) | With Space (CORRECT) |
|--------|----------------------|---------------------|
| First | token 6730 (")First") | token 3824 ("First") |
| Second | token 11863 (")Second") | token 6440 ("Second") |
| Similar | tokens 8942+2327 (")Similar") | token 13999 ("Similar") |

**Why Accuracy Was Zero:**

1. **Training labels** contained wrong tokens: `[6730, 11863, 8942]`
2. **Model learned** to predict these wrong tokens
3. **Accuracy computation** checked for correct tokens: `[3824, 6440, 13999]`
4. **They never matched** → accuracy = 0!

**Why "ilarilar" Generation:**

- Model learned to generate token `2327` (decodes to "ilar")
- This was part of buggy token sequence `8942+2327` for ")Similar"
- Token 2327 kept repeating → "ilarilarilarilar"

### The Fix

**Fixed Code (AFTER FIX):**

**File:** `/root/LMOL/data/collator.py`

```python
# Line 233 (NEW - FIXED)
full_text = prompt_prefix + ' ' + answer_text  # ✅ SPACE ADDED!

# Line 241 (NEW - FIXED)  
# CRITICAL: Tokenize prompt WITHOUT trailing space
# The tokenizer merges " Answer" into single token in context
prompt_enc = self.tokenizer(prompt_prefix, add_special_tokens=True, return_tensors="pt")
```

**Key Insight:** LLaMA tokenizer has context-dependent tokenization:
- `" First"` alone → `[29871, 3824]` (space token + word token)
- `"prompt: First"` in context → `[..., 29901, 3824]` (colon + word token)
- The space is **absorbed** into the word token when tokenizing full sequence

**Other Files Fixed:**

1. **`/root/LMOL/training/trainer.py` (lines 586-597):**
   - Updated accuracy computation to tokenize with leading space
   - Extract last token (word token) from `tokenizer(' ' + answer)`

2. **`/root/LMOL/evaluation/evaluator.py` (lines 280-283):**
   - Updated answer parsing to expect word tokens
   - Tokenize with leading space for matching

3. **`/root/LMOL/scripts/validate_tokenization.py`:**
   - NEW: Validation script to check tokenization consistency

### ⚠️ CRITICAL: Old Models Cannot Be Fixed

**Models trained before 2025-10-13 have wrong token embeddings baked into their weights and MUST be deleted and retrained.**

Existing checkpoints:
```
model/checkpoints/llava-1.5-7b-hf_3class/
├── 20251012_145842/   ❌ BROKEN - trained with buggy tokenization
├── 20251012_181743/   ❌ BROKEN - trained with buggy tokenization
└── 20251012_190654/   ❌ BROKEN - trained with buggy tokenization
```

**To fix:** Delete these and retrain from scratch with fixed code.

---

## Training Pipeline

### Training Flow

```
1. Load dataset → SCUT_FBP5500_Pairs (45,000 pairs per fold)
   ↓
2. Data collation → LlavaPairsCollator
   - Tokenizes: prompt + ' ' + answer  ⚠️ SPACE IS CRITICAL!
   - Creates labels with IGNORE_INDEX for prompt tokens
   - Swap doubling: (A,B) → [(A,B), (B,A)]
   ↓
3. Model forward pass → LLaVA-1.5-7B + LoRA
   ↓
4. Loss computation → WeightedSwapConsistencyTrainer
   - Cross-entropy loss (weighted by class)
   - Consistency loss (symmetric KL between swaps)
   - Total loss = CE + λ(t) * consistency
   ↓
5. Accuracy computation (token-level)
   - Get prediction: logits.argmax(dim=-1)
   - Check if predicted token == label token
   - Label tokens are: 3824 (First), 6440 (Second), 13999 (Similar)
   ↓
6. Backward pass → Optimize with dual LR
   - Projector: LR = 2e-3
   - LoRA: LR = 1e-4
   ↓
7. Save checkpoints
   - Best model (lowest loss) → best/
   - Last model (end of training) → last/
```

### Critical Training Parameters

**From:** `/root/LMOL/configs/config.py`

```python
# Batch configuration
PER_DEVICE_TRAIN_BATCH_SIZE = 16  # 2 pairs → 4 samples with swap doubling
GRADIENT_ACCUMULATION_STEPS = 2   # Effective batch = 16 * 2 * 2 = 64

# Learning rates (DUAL LR STRATEGY)
LR_PROJECTION = 2e-3  # Higher LR for projector (from scratch)
LR_LORA = 1e-4        # Lower LR for LoRA (fine-tuning)

# Loss weights
WSIM = 1.2           # Weight for "Similar" class
CONS_WEIGHT = 20.0   # Consistency loss weight

# Dynamic consistency (NEW)
USE_DYNAMIC_CONSISTENCY = True
CONS_WEIGHT_START = 20.0   # Start value
CONS_WEIGHT_END = 50.0     # End value (stronger regularization)
CONS_WEIGHT_RAMP_RATIO = 0.5  # Reach end value at 50% of training

# Swap doubling
SWAP_DOUBLE = True   # ⚠️ CRITICAL: Must be True for consistency loss
```

### Training Validation Checks

**Before starting training, ALWAYS run:**

```bash
python scripts/validate_tokenization.py
```

**Expected output:**
```
✓ ALL CHECKS PASSED

Tokenization is consistent across all components:
  - Collator constructs labels correctly with leading space
  - Trainer accuracy computation uses matching tokenization
  - Evaluator parsing uses matching tokenization
```

**If validation fails, DO NOT TRAIN!**

### Expected Training Behavior

**CORRECT (After Fix):**
```
Batch    1 | Epoch 0.00023 | Loss: 2.1234e+00 | Acc: 0.3125  ✅
Batch   50 | Epoch 0.01172 | Loss: 1.2345e+00 | Acc: 0.5625  ✅
Batch  100 | Epoch 0.02344 | Loss: 0.8123e+00 | Acc: 0.7188  ✅
Batch 1000 | Epoch 0.23445 | Loss: 0.3456e+00 | Acc: 0.8594  ✅
```

**Key indicators:**
- ✅ Accuracy is **NON-ZERO** from the start
- ✅ Accuracy **INCREASES** over time
- ✅ Loss **DECREASES** smoothly
- ✅ Final accuracy typically 80-95%

**INCORRECT (Buggy Tokenization):**
```
Batch    1 | Epoch 0.00023 | Loss: 1.2345e+00 | Acc: 0.0000  ❌
Batch   50 | Epoch 0.01172 | Loss: 0.0012e+00 | Acc: 0.0000  ❌
Batch  100 | Epoch 0.02344 | Loss: 3.4567e-07 | Acc: 0.0000  ❌
```

**Warning signs:**
- ❌ Accuracy is **ALWAYS ZERO**
- ❌ Loss goes to **NEAR-ZERO** (model overfitting to wrong tokens)
- ❌ This indicates tokenization bug - STOP AND CHECK!

---

## Evaluation Pipeline

### Evaluation Flow

```
1. Load model checkpoint → PeftModel.from_pretrained()
   ↓
2. Validate tokenizer → Check vocab size, EOS token, etc.
   - Display answer token sequences with leading space
   ↓
3. Load evaluation pairs → read_pairs_csv()
   ↓
4. For each pair:
   - Build prompt: <image>*1152 + ', ' + QUESTION_TEXT
   - Tokenize prompt (for input_ids)
   - Process images (for pixel_values)
   - Generate answer: model.generate()
   ↓
5. Parse generated tokens → _parse_answer()
   - Look for token sequences: [3824], [6440], [13999]
   - Match with answer labels: First, Second, Similar
   ↓
6. Compute accuracy → correct / total
   ↓
7. Return y_true, y_pred for metrics
```

### Generation Parameters

**From:** `/root/LMOL/evaluation/evaluator.py`, lines 180-193

```python
gen_ids = model.generate(
    input_ids=prompt_ids,
    attention_mask=prompt_attn,
    pixel_values=pixel_pair,
    max_new_tokens=5,          # Only need 1-2 tokens for answer
    do_sample=True,            # Enable sampling for diversity
    temperature=0.1,           # Low temp for deterministic-like output
    top_p=0.9,                 # Nucleus sampling
    repetition_penalty=1.5,    # Prevent repetition (helps avoid "ilarilar")
    no_repeat_ngram_size=3,    # Block 3-gram repetition
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    stopping_criteria=stopping_criteria,  # Custom stopping for repetition
)
```

### Answer Parsing Logic

**Critical:** Answer parsing MUST match collator tokenization!

**From:** `/root/LMOL/evaluation/evaluator.py`, lines 264-294

```python
def _make_label_token_seqs(tokenizer) -> dict:
    """
    ⚠️ CRITICAL: Tokenize answers WITH leading space to match collator!
    
    The collator constructs: prompt + ' ' + answer
    So we must tokenize: ' ' + answer
    """
    seqs = {}
    for label in (config.ANSWER_FIRST, config.ANSWER_SECOND, config.ANSWER_SIMILAR):
        # Add leading space to match collator tokenization
        variants = [' ' + label, ' ' + label.lower()]  # [" First", " first"]
        token_variants = []
        for variant in variants:
            toks = tokenizer(variant, add_special_tokens=False).input_ids
            if toks not in token_variants:
                token_variants.append(toks)
        seqs[label] = token_variants
    return seqs
```

**Token Mappings (CORRECT):**
```python
' First'   → [29871, 3824]  → word_token = 3824
' Second'  → [29871, 6440]  → word_token = 6440
' Similar' → [29871, 13999] → word_token = 13999
```

**Parsing Strategy:**

1. **Exact prefix matching:** Check if generated tokens start with expected sequence
2. **Subsequence matching:** Check if expected tokens appear within first 3 positions
3. **Fallback:** If parsing fails, log [PARSE_FAIL] and return default (Similar)

### Expected Evaluation Behavior

**CORRECT (New model trained with fix):**
```
fold1 eval: 87%▮
[Result] fold1: accuracy=0.873456 (87345/100000)
```

**INCORRECT (Old model + new evaluator):**
```
fold1 eval: 0%
[PARSE_FAIL] Generated: 'ilarilarilarilar' | Tokens: [2327, 2327, 2327, 309, 2327]
```

---

## Key Implementation Details

### 1. Collator: Label Construction

**File:** `/root/LMOL/data/collator.py`

**⚠️ MOST CRITICAL FUNCTION:**

```python
def _process_sample(self, img1, img2, answer_text: str, ...):
    # STEP 1: Build full text with SPACE before answer
    # ⚠️ CRITICAL: Space is required for correct tokenization!
    full_text = prompt_prefix + ' ' + answer_text  # Line 233
    
    # STEP 2: Tokenize full sequence
    enc = self.tokenizer(full_text, add_special_tokens=True, ...)
    input_ids = enc.input_ids[0]
    
    # STEP 3: Find answer boundary
    # ⚠️ CRITICAL: Tokenize prompt WITHOUT trailing space
    # The tokenizer merges " Answer" into context, not separate tokens
    prompt_enc = self.tokenizer(prompt_prefix, add_special_tokens=True, ...)
    prompt_len = prompt_enc.input_ids.shape[1]
    
    # STEP 4: Create labels with masking
    ans_start = prompt_len
    ans_len = len(input_ids) - ans_start
    
    labels = torch.full_like(input_ids, IGNORE_INDEX)  # -100
    if ans_len > 0:
        # Only answer tokens contribute to loss
        labels[ans_start:ans_start + ans_len] = input_ids[ans_start:ans_start + ans_len]
    
    # STEP 5: Validate labels
    valid_label_count = (labels != IGNORE_INDEX).sum().item()
    if valid_label_count == 0:
        print(f"[LABEL_ERROR] Sample has ZERO valid labels! Loss will be zero.")
```

**Why tokenize prompt WITHOUT trailing space?**

The LLaMA tokenizer has context-dependent behavior:

```python
# Tokenizing " First" in isolation
tokenizer(' First', add_special_tokens=False).input_ids
# → [29871, 3824]  (space token + word token)

# Tokenizing "prompt: First" in context
prompt = "Answer with exactly one word:"
tokenizer(prompt + ' First', add_special_tokens=False).input_ids
# → [..., 29901, 3824]  (colon + word token, space is absorbed!)

# Finding the boundary
prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
# → [..., 29901]
full_ids = tokenizer(prompt + ' First', add_special_tokens=False).input_ids
# → [..., 29901, 3824]

# Answer tokens = full_ids[len(prompt_ids):] = [3824] ✓
```

### 2. Trainer: Accuracy Computation

**File:** `/root/LMOL/training/trainer.py`, lines 579-651

**⚠️ MUST MATCH COLLATOR TOKENIZATION:**

```python
# Get answer token IDs from tokenizer
# CRITICAL: Tokenize with leading space to match collator
first_tokens = self.tokenizer(' ' + config.ANSWER_FIRST, add_special_tokens=False).input_ids
# → [29871, 3824]

# Use LAST token (word token, not space token)
first_id = first_tokens[-1]  # → 3824

# For each sample, check if predicted token == label token
for i in range(batch_size):
    valid_positions = (labels[i] != IGNORE_INDEX).nonzero(as_tuple=True)[0]
    if len(valid_positions) > 0:
        ans_pos = valid_positions[0].item()
        pred_token = preds[i, ans_pos].item()
        true_token = labels[i, ans_pos].item()
        
        # Map tokens to classes
        if pred_token == first_id:  # 3824
            pred_class = 0
        elif pred_token == second_id:  # 6440
            pred_class = 1
        elif pred_token == similar_id:  # 13999
            pred_class = 2
```

### 3. Loss Function: Weighted CE + Consistency

**File:** `/root/LMOL/training/trainer.py`, lines 447-906

**Loss Formula:**
```
total_loss = weighted_ce_loss + λ(t) * consistency_loss
```

**Cross-Entropy Loss:**
```python
# Apply class weights
weights = torch.ones(batch_size)
weights[label_ids == 2] = config.WSIM  # Higher weight for "Similar"

# Compute token-level CE loss
ce_loss = F.cross_entropy(flat_logits, flat_labels, 
                          ignore_index=-100, reduction='none')

# Apply weights
weighted_ce = (ce_loss * token_weights).sum()
```

**Consistency Loss:**
```python
# Get class token logits for each sample
cls_logits = logits[batch_indices, cls_positions]  # (N, 3)

# Split into original and swapped pairs
p = probs[0::2]  # Original (A,B)
q = probs[1::2]  # Swapped (B,A)

# Apply permutation (First↔Second, Similar unchanged)
perm_indices = [1, 0, 2]
q_perm = q.index_select(-1, perm_indices)

# Symmetric KL divergence
kl_pq = (p * (log_p - log_q_perm)).sum(dim=-1)
kl_qp = (q_perm * (log_q_perm - log_p)).sum(dim=-1)
sym_kl = 0.5 * (kl_pq + kl_qp)

consistency_loss = sym_kl.mean()
```

**Dynamic Consistency Weight:**
```python
# Gradually increase consistency weight during training
progress = min(step / (total_steps * cons_weight_ramp_ratio), 1.0)
current_weight = cons_weight_start + (cons_weight_end - cons_weight_start) * progress

# Example: Start at 20.0, end at 50.0 over 50% of training
# Step 0:    λ = 20.0
# Step 50%:  λ = 50.0
# Step 100%: λ = 50.0
```

### 4. Prompt Construction

**Format:**
```
<image> × 1152 , QUESTION_TEXT ANSWER
```

**Breakdown:**
- `<image>` tokens: 576 per image × 2 images = 1152 tokens
- `, ` separator
- `QUESTION_TEXT`: "Compare the facial attractiveness of these two images. Which face looks more attractive? Answer with exactly one word: First (left image), Second (right image), or Similar (equally attractive)"
- Space + Answer: " First" / " Second" / " Similar"

**⚠️ CRITICAL:** The space before the answer is essential for correct tokenization!

---

## Common Pitfalls

### ❌ Pitfall 1: Missing Space Before Answer

**WRONG:**
```python
full_text = prompt + answer  # ❌ Creates ")First"
```

**CORRECT:**
```python
full_text = prompt + ' ' + answer  # ✅ Creates ") First"
```

**Why it matters:** Tokenization is completely different!
- ")First" → token 6730
- ") First" → token 3824

### ❌ Pitfall 2: Tokenizing Prompt with Trailing Space

**WRONG:**
```python
prompt_with_space = prompt + ' '
prompt_enc = tokenizer(prompt_with_space, ...)
prompt_len = len(prompt_enc.input_ids)
answer_start = prompt_len  # ❌ WRONG BOUNDARY!
```

**CORRECT:**
```python
prompt_enc = tokenizer(prompt, ...)  # WITHOUT trailing space
prompt_len = len(prompt_enc.input_ids)
answer_start = prompt_len  # ✅ CORRECT BOUNDARY
```

**Why it matters:** The tokenizer absorbs the space into the next token when tokenizing the full sequence.

### ❌ Pitfall 3: Using First Token Instead of Last Token

**WRONG:**
```python
toks = tokenizer(' First', add_special_tokens=False).input_ids  # [29871, 3824]
first_id = toks[0]  # ❌ 29871 = space token
```

**CORRECT:**
```python
toks = tokenizer(' First', add_special_tokens=False).input_ids  # [29871, 3824]
word_id = toks[-1]  # ✅ 3824 = word token
```

**Why it matters:** In context, only the word token appears in labels, not the space token.

### ❌ Pitfall 4: Evaluating Old Models with New Code

**WRONG:**
```bash
# Apply fix to code
# Immediately evaluate old model without retraining
python scripts/evaluate.py  # ❌ Will show "ilarilar" errors
```

**CORRECT:**
```bash
# Apply fix to code
# Delete old models
rm -rf model/checkpoints/llava-1.5-7b-hf_3class/202*
# Retrain from scratch
python scripts/train.py  # ✅ New model learns correct tokens
# Then evaluate
python scripts/evaluate.py  # ✅ Now works correctly
```

**Why it matters:** Old models have wrong token embeddings baked into weights.

### ❌ Pitfall 5: Not Validating Tokenization Before Training

**WRONG:**
```bash
# Make changes to collator
# Start training immediately
python scripts/train.py  # ❌ Might train with wrong tokenization
```

**CORRECT:**
```bash
# Make changes to collator
# Validate first
python scripts/validate_tokenization.py  # ✅ Check consistency
# If validation passes, then train
python scripts/train.py
```

**Why it matters:** Catches tokenization bugs before wasting hours of training.

---

## Validation & Testing

### Pre-Training Validation

**Always run before training:**

```bash
cd /root/LMOL
python scripts/validate_tokenization.py
```

**What it checks:**

1. **Answer tokenization consistency:**
   - Verifies word tokens match between isolated and in-context tokenization
   - Checks: 3824 (First), 6440 (Second), 13999 (Similar)

2. **Collator label construction:**
   - Simulates collator behavior
   - Verifies labels contain correct word tokens

3. **Cross-component consistency:**
   - Collator, trainer, and evaluator all use matching tokenization

**Expected output:**
```
✓ ALL CHECKS PASSED

Your model should now:
  ✓ Show non-zero training accuracy
  ✓ Generate correct answers during evaluation
  ✓ Avoid repetitive token generation
```

**If validation fails:**
- ❌ DO NOT TRAIN!
- Review recent changes to collator.py, trainer.py, evaluator.py
- Ensure space is added before answer text
- Check tokenization logic matches this document

### During Training Checks

**Monitor these metrics:**

1. **Training accuracy:**
   - Should be > 0.0 from first batch
   - Should increase over time (30% → 50% → 70% → 85%+)
   - Final accuracy typically 80-95%

2. **Loss:**
   - Should decrease smoothly
   - CE loss: ~2.0 → ~0.3
   - Consistency loss: ~0.5 → ~0.1

3. **Gradient norm:**
   - Should be stable (1e-2 to 1e-4 range)
   - Spikes indicate training instability

**Warning signs:**

- ⚠️ Accuracy stays at 0.0 → Tokenization bug
- ⚠️ Loss explodes (NaN/Inf) → Reduce learning rate
- ⚠️ Gradient vanishing (~0.0) → Check model initialization

### Post-Training Checks

**Run evaluation on small sample:**

```bash
python scripts/evaluate.py --samples 100
```

**Expected behavior:**

- ✅ No [PARSE_FAIL] messages
- ✅ Generates "First", "Second", or "Similar" (not "ilarilar")
- ✅ Accuracy > 70% (should match training accuracy roughly)

**If evaluation fails:**

- Check if you're loading the correct model (not an old one)
- Verify model was trained with fixed code
- Run tokenization validation again

---

## Quick Reference

### Token Mappings (CORRECT)

```python
# After fix (tokenizing with leading space)
' First'   → [29871, 3824]  → word_token: 3824
' Second'  → [29871, 6440]  → word_token: 6440  
' Similar' → [29871, 13999] → word_token: 13999
```

### Token Mappings (BUGGY - DO NOT USE)

```python
# Before fix (no leading space in context)
')First'   → [1723, 6730]       → wrong_token: 6730
')Second'  → [1723, 11863]      → wrong_token: 11863
')Similar' → [1723, 8942, 2327] → wrong_token: 8942, 2327
```

### Critical Files

1. **`data/collator.py`** - Line 233: Must have space before answer
2. **`training/trainer.py`** - Lines 586-597: Accuracy computation
3. **`evaluation/evaluator.py`** - Lines 280-283: Answer parsing
4. **`configs/config.py`** - All hyperparameters

### Commands

```bash
# Validate tokenization (run before every training)
python scripts/validate_tokenization.py

# Train (after validation passes)
python scripts/train.py

# Evaluate
python scripts/evaluate.py

# Cleanup and retrain (after bug fix)
./scripts/cleanup_and_retrain.sh
```

---

## Changelog

### 2025-10-13: Major Tokenization Bug Fix

**Issue:** Zero training accuracy, "ilarilar" generation during evaluation

**Root cause:** Missing space before answer text in collator

**Files changed:**
- `data/collator.py` - Added space before answer (line 233)
- `training/trainer.py` - Updated accuracy computation (lines 586-597)
- `evaluation/evaluator.py` - Updated answer parsing (lines 280-283)
- `scripts/validate_tokenization.py` - NEW: Validation script

**Impact:** All models trained before this date must be deleted and retrained

**Validation:** `python scripts/validate_tokenization.py` must pass

---

## Contact / Questions

If you encounter issues not covered in this document:

1. Check `FIX_SUMMARY.md` for detailed bug explanation
2. Check `IMPORTANT_READ_THIS.md` for retraining instructions
3. Run `python scripts/validate_tokenization.py` to check consistency
4. Review training logs for warning messages

**Remember:** Always read this document before making changes to the codebase!

---

**Last Updated:** 2025-10-13  
**Status:** Tokenization bug FIXED, validation passing, ready for training

