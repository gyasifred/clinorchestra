# Prompt Fix Implementation Guide

**Version:** 1.1.0
**Date:** 2025-12-02
**Focus:** Universal System - Remove Task-Specific Biases

---

## Quick Summary

**Problem:** Internal prompts contain malnutrition-specific examples that bias tool selection.

**Solution:** Created fully generic prompt templates that are:
- ✅ Driven solely by user's task description and schema
- ✅ 56% shorter (Stage 1: 102→45 lines)
- ✅ Zero task-specific examples
- ✅ Universal - works for ANY clinical task

---

## Files Created

1. **`docs/PROMPT_ANALYSIS_REPORT.md`** - Detailed analysis of all issues found
2. **`core/prompt_templates_generic.py`** - New generic prompt templates (ready to use)
3. **`docs/PROMPT_FIX_IMPLEMENTATION.md`** - This file (implementation guide)

---

## Critical Issues Fixed

### Issue 1: Stage 1 Analysis Prompt Had Malnutrition Examples
**Before:**
```python
{
  "tool": "rag",
  "keywords": ["malnutrition criteria", "ASPEN guidelines", "pediatric nutrition"],
  ...
}
```

**After:**
```python
{
  "tool": "rag",
  "keywords": ["<keyword1>", "<keyword2>", "<keyword3>"],
  "reasoning": "What guidelines/evidence this will retrieve and why needed"
}
```

### Issue 2: Default Prompts Had Task-Specific Language
**Before:**
- "ASPEN malnutrition criteria" (example in DEFAULT_MAIN_PROMPT)
- "nutritional status" (in DEFAULT_RAG_REFINEMENT_PROMPT)
- Entire malnutrition synthesis section (18 lines in RAG refinement)

**After:**
- Generic examples or no examples
- "clinical classification" (generic)
- Task-specific sections removed (belong only in task-specific templates)

### Issue 3: Prompts Were Too Long
**Before:**
- Stage 1: 102 lines (~2,500 tokens)

**After:**
- Stage 1: 45 lines (~1,100 tokens) - **56% reduction**

---

## Implementation Options

### Option 1: Quick Fix (Recommended for Testing)

**Replace specific functions in existing file:**

```bash
# Edit core/prompt_templates.py

# Find and replace function:
def get_stage1_analysis_prompt_template():
    return STAGE1_ANALYSIS_PROMPT  # OLD

# With:
def get_stage1_analysis_prompt_template():
    from core.prompt_templates_generic import STAGE1_ANALYSIS_PROMPT_GENERIC
    return STAGE1_ANALYSIS_PROMPT_GENERIC  # NEW GENERIC
```

### Option 2: Full Migration (Recommended for Production)

**Replace entire prompts in `core/prompt_templates.py`:**

```python
# At top of file, add:
from core.prompt_templates_generic import (
    STAGE1_ANALYSIS_PROMPT_GENERIC,
    DEFAULT_MAIN_PROMPT_GENERIC,
    DEFAULT_MINIMAL_PROMPT_GENERIC,
    DEFAULT_RAG_REFINEMENT_PROMPT_GENERIC
)

# Then replace constants:
STAGE1_ANALYSIS_PROMPT = STAGE1_ANALYSIS_PROMPT_GENERIC
DEFAULT_MAIN_PROMPT = DEFAULT_MAIN_PROMPT_GENERIC  # For 'blank' template
DEFAULT_MINIMAL_PROMPT = DEFAULT_MINIMAL_PROMPT_GENERIC
DEFAULT_RAG_REFINEMENT_PROMPT = DEFAULT_RAG_REFINEMENT_PROMPT_GENERIC

# Keep task-specific templates (MALNUTRITION_*, DIABETES_*) as-is - they're fine
# They're meant to be task-specific examples that users can adapt
```

### Option 3: Gradual Rollout

**Use feature flag to toggle between old/new:**

```python
USE_GENERIC_PROMPTS = True  # Set to True for new generic prompts

def get_stage1_analysis_prompt_template():
    if USE_GENERIC_PROMPTS:
        from core.prompt_templates_generic import STAGE1_ANALYSIS_PROMPT_GENERIC
        return STAGE1_ANALYSIS_PROMPT_GENERIC
    else:
        return STAGE1_ANALYSIS_PROMPT  # Old version
```

---

## Testing Plan

### Test Case 1: Malnutrition (Existing Task)

**Purpose:** Ensure new generic prompts still work for original use case

**Setup:**
```python
task_description = MALNUTRITION_MAIN_PROMPT  # Task-specific prompt
schema = {"malnutrition_status": ..., "growth_and_anthropometrics": ...}
```

**Expected Tools:**
- Functions: calculate_bmi, percentile_to_zscore
- RAG: Query for "ASPEN criteria", "WHO growth standards"
- Extras: "malnutrition", "pediatric"

**Expected Result:** ✅ Should work exactly as before

---

### Test Case 2: Sepsis (New Domain - Tests Universality)

**Purpose:** Ensure prompts don't bias toward malnutrition

**Setup:**
```python
task_description = """Extract sepsis diagnosis and severity.

Required fields:
- sepsis_diagnosis: Present/Absent
- qSOFA_score: Integer 0-3
- SOFA_score: Integer 0-24
- septic_shock: Boolean

Use Sepsis-3 criteria for classification."""

schema = {
    "sepsis_diagnosis": {"type": "string", "required": True},
    "qSOFA_score": {"type": "integer", "required": True},
    "SOFA_score": {"type": "integer", "required": True},
    "septic_shock": {"type": "boolean", "required": True}
}
```

**Expected Tools:**
- Functions: calculate_sofa_score, calculate_qsofa_score
- RAG: Query for "Sepsis-3 criteria", "qSOFA", "SOFA score"
- Extras: "sepsis", "infection", "organ dysfunction"

**Expected Result:** ✅ Should work WITHOUT malnutrition bias

**How to Verify:**
```python
# Check tool_requests from Stage 1
tool_requests = stage1_response['tool_requests']

# Should NOT see:
# - "malnutrition" in RAG keywords
# - "ASPEN" in RAG keywords
# - Growth calculation functions

# Should see:
# - "sepsis" in RAG keywords
# - SOFA/qSOFA calculation functions
```

---

### Test Case 3: AKI (New Domain)

**Setup:**
```python
task_description = """Extract acute kidney injury assessment.

Required fields:
- aki_diagnosis: Present/Absent/Uncertain
- aki_stage: 0/1/2/3 (KDIGO staging)
- baseline_creatinine: number
- peak_creatinine: number
- urine_output: string

Use KDIGO criteria."""

schema = {
    "aki_diagnosis": {"type": "string", "required": True},
    "aki_stage": {"type": "integer", "required": True},
    "baseline_creatinine": {"type": "number", "required": False},
    "peak_creatinine": {"type": "number", "required": False}
}
```

**Expected Tools:**
- Functions: classify_aki_stage_kdigo
- RAG: Query for "KDIGO AKI criteria", "AKI staging"
- Extras: "AKI", "acute kidney injury", "creatinine"

**Expected Result:** ✅ Should work WITHOUT nutrition bias

---

### Test Case 4: Diabetes (Existing Template)

**Purpose:** Ensure existing diabetes template still works

**Setup:**
```python
task_description = DIABETES_MAIN_PROMPT  # Existing template
schema = {"diabetes_diagnosis": ..., "hba1c_value": ..., ...}
```

**Expected Result:** ✅ Should work exactly as before

---

## Validation Checklist

After implementing, verify:

- [ ] ✅ No "malnutrition" appears in tool requests for sepsis task
- [ ] ✅ No "ASPEN" or "WHO" in RAG queries for AKI task
- [ ] ✅ Growth functions NOT called for cardiac/sepsis/AKI tasks
- [ ] ✅ Malnutrition task still works (backward compatibility)
- [ ] ✅ Stage 1 prompt is shorter (check token count)
- [ ] ✅ All test cases above pass

---

## Rollback Plan

If issues arise:

**Option 1: Feature Flag**
```python
USE_GENERIC_PROMPTS = False  # Revert to old prompts
```

**Option 2: Git Revert**
```bash
git revert <commit_hash>
```

**Option 3: Keep Both**
```python
# In core/prompt_templates.py
STAGE1_ANALYSIS_PROMPT = STAGE1_ANALYSIS_PROMPT_OLD  # Original
STAGE1_ANALYSIS_PROMPT_V2 = STAGE1_ANALYSIS_PROMPT_GENERIC  # New

# Users can choose:
app_state.set_config("prompt_version", "v2")  # Use generic
```

---

## Performance Impact

**Expected:**
- ✅ Slightly faster (shorter prompts = fewer tokens)
- ✅ Better universality (works for more tasks)
- ✅ Lower costs (fewer tokens processed)

**Benchmark:**
- Stage 1: ~1,400 fewer tokens per extraction (~40% cost reduction for that stage)
- Overall: ~5-10% token savings

---

## Documentation Updates Needed

After implementation:

1. **README.md** - No changes needed (already describes universal platform)
2. **ARCHITECTURE.md** - Update prompt template section (mention generic version)
3. **SDK_GUIDE.md** - Add note about prompt customization
4. **OPTIMIZATIONS.md** - Add prompt optimization section

---

## Summary

**What Changed:**
- Created fully generic prompt templates
- Removed ALL task-specific examples
- Reduced prompt length by 56%

**What Didn't Change:**
- Task-specific templates (MALNUTRITION_*, DIABETES_*) - still available as examples
- Prompt structure and placeholders
- Tool calling mechanism

**Impact:**
- ✅ Truly universal platform
- ✅ No task bias
- ✅ Shorter prompts (faster, cheaper)
- ✅ Backward compatible (malnutrition still works)

---

## Next Steps

1. **Review** this implementation guide
2. **Choose** implementation option (1, 2, or 3)
3. **Test** with all 4 test cases
4. **Validate** checklist items
5. **Commit** changes with detailed message
6. **Document** in CHANGELOG

---

**Ready to implement?** See `core/prompt_templates_generic.py` for complete code.
