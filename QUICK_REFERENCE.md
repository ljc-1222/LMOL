# LMOL Quick Reference Card

**⚡ Read this before making ANY code changes! ⚡**

## 🚨 Critical Rules

### Rule #1: ALWAYS Check Before Changing Code
```bash
# Read the technical reference first
cat /root/LMOL/TECHNICAL_REFERENCE.md | less
```

### Rule #2: NEVER Remove Space Before Answer
```python
# ✅ CORRECT
full_text = prompt + ' ' + answer

# ❌ WRONG - Will cause zero accuracy
full_text = prompt + answer
```

### Rule #3: ALWAYS Validate Before Training
```bash
python scripts/validate_tokenization.py
# Must show: ✓ ALL CHECKS PASSED
```

### Rule #4: NEVER Use Old Models After Fix
Old models have wrong tokens baked in → Must delete and retrain!

---

## 📝 Token Mappings (Memorize These!)

**CORRECT (Current):**
```
' First'   → token 3824
' Second'  → token 6440
' Similar' → token 13999
```

**WRONG (Old Bug):**
```
')First'   → token 6730   ❌
')Second'  → token 11863  ❌
')Similar' → token 8942   ❌ (generated "ilar")
```

---

## 🔍 Quick Diagnostics

### Training Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Accuracy always 0.0 | Tokenization mismatch | Check collator space, validate |
| Loss → 0 but acc = 0 | Learning wrong tokens | Tokenization bug |
| NaN/Inf loss | LR too high | Reduce learning rate |
| "ilarilar" in eval | Using old model | Retrain from scratch |

### Validation Checks

```bash
# Before training
python scripts/validate_tokenization.py  # Must pass

# During training (monitor)
# - Accuracy > 0.0 and increasing
# - Loss decreasing smoothly

# After training
python scripts/evaluate.py --samples 100
# - No [PARSE_FAIL] errors
# - Generates First/Second/Similar correctly
```

---

## 📂 Critical Files (Don't Change Without Review!)

1. **`data/collator.py` line 233** - Space before answer
2. **`training/trainer.py` lines 586-597** - Accuracy computation  
3. **`evaluation/evaluator.py` lines 280-283** - Answer parsing
4. **`configs/config.py`** - All hyperparameters

---

## 🎯 Common Tasks

### Starting Fresh Training
```bash
cd /root/LMOL
python scripts/validate_tokenization.py  # Validate first!
python scripts/train.py                  # Then train
```

### After Bug Fix (Retrain Required)
```bash
cd /root/LMOL
rm -rf model/checkpoints/llava-1.5-7b-hf_3class/202*  # Delete old
python scripts/validate_tokenization.py               # Validate
python scripts/train.py                                # Retrain
```

### Quick Evaluation
```bash
python scripts/evaluate.py --samples 100  # Fast test
```

---

## 🧪 Expected Behavior

### Training (CORRECT)
```
Batch   1 | Loss: 2.1e+00 | Acc: 0.3125 ✅
Batch  50 | Loss: 1.2e+00 | Acc: 0.5625 ✅
Batch 100 | Loss: 0.8e+00 | Acc: 0.7188 ✅
```

### Training (BROKEN)
```
Batch   1 | Loss: 1.2e+00 | Acc: 0.0000 ❌
Batch  50 | Loss: 1.2e-05 | Acc: 0.0000 ❌
Batch 100 | Loss: 3.4e-07 | Acc: 0.0000 ❌
```

### Evaluation (CORRECT)
```
fold1 eval: 87%▮
[Result] accuracy=0.873456
```

### Evaluation (BROKEN)
```
[PARSE_FAIL] Generated: 'ilarilarilarilar' ❌
```

---

## 📚 Full Documentation

- **Technical deep dive:** `TECHNICAL_REFERENCE.md` (830 lines)
- **Bug explanation:** `FIX_SUMMARY.md`
- **Retraining guide:** `IMPORTANT_READ_THIS.md`
- **Project overview:** `README.md`

---

## ✅ Pre-Change Checklist

Before modifying code, check:

- [ ] Read relevant section in TECHNICAL_REFERENCE.md
- [ ] Understand why the change is needed
- [ ] Know which files will be affected
- [ ] Plan to run validation after changes
- [ ] Prepared to retrain if changing tokenization

---

**Last Updated:** 2025-10-13  
**Critical Fix Applied:** Tokenization space bug fixed  
**Status:** ✅ Ready for training with validation passing

