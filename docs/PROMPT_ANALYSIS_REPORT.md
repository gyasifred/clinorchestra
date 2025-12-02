# ClinOrchestra Prompt Analysis Report

**Date:** 2025-12-02
**Analyst:** Claude (AI Assistant)
**Focus:** Universal System - Generic vs. Task-Specific Prompts

---

## Executive Summary

**CRITICAL FINDINGS:** The internal prompts that analyze tasks and select tools contain **task-specific examples** (malnutrition, ASPEN, WHO) that could bias the system toward those specific tasks. This contradicts the "universal platform" design goal.

**Impact:** Medium-High
- System may over-select malnutrition-related tools even for unrelated tasks
- RAG queries may be biased toward nutrition guidelines
- Examples in prompts act as implicit instructions

**Recommendation:** Remove ALL task-specific examples from internal prompts. Make them fully generic and concise.

---

## 1. Stage 1 Analysis Prompt (CRITICAL)

**File:** `core/prompt_templates.py`
**Lines:** 522-623
**Function:** `STAGE1_ANALYSIS_PROMPT`

### Issues Found:

#### ❌ Issue 1: Task-Specific Examples in "Intelligent Query Building" (Lines 554-576)

```python
INTELLIGENT QUERY BUILDING:
- RAG queries should target guidelines, criteria, and standards relevant to the extraction task
- Extract key medical concepts from clinical text: diagnoses, conditions, assessments
- Include classification/label terms in queries
- Include schema-related terms (e.g., if schema has "malnutrition_status", include "malnutrition" in queries)
                                        ^^^^^^^^^^^^^^^^^^^ TASK-SPECIFIC EXAMPLE!
```

**Problem:**
- Mentions "malnutrition_status" as an example schema field
- Could bias system to look for malnutrition-related schemas even when user is doing sepsis, AKI, cardiac, etc.

#### ❌ Issue 2: Malnutrition-Specific Tool Request Examples (Lines 593-617)

```python
[REQUIRED OUTPUT FORMAT]

Return your response in this EXACT JSON format:
{
  "analysis": "Brief analysis...",
  "tool_requests": [
    {
      "tool": "function",
      "name": "extract_age_from_dates",
      ...
    },
    {
      "tool": "function",
      "name": "calculate_bmi",
      ...
    },
    {
      "tool": "rag",
      "keywords": ["malnutrition criteria", "ASPEN guidelines", "pediatric nutrition"],
      ^^^^^^^^^^^^^^ ALL MALNUTRITION-SPECIFIC! ^^^^^^^^^^^^^^
      "reasoning": "Need evidence-based criteria for nutritional assessment and diagnosis"
    },
    {
      "tool": "extras",
      "keywords": ["ICD-10", "malnutrition", "coding"],
      ^^^^^^^^^^^ MALNUTRITION-SPECIFIC! ^^^^^^^^^^^
      ...
    }
  ]
}
```

**Problem:**
- ALL examples are malnutrition-specific: "malnutrition criteria", "ASPEN guidelines", "pediatric nutrition"
- Creates implicit bias toward nutrition tasks
- Should use generic medical examples or multiple diverse examples

**User's Concern:** ✅ Exactly right - these should be completely generic!

#### Severity: **HIGH**
This is the primary prompt that decides which tools to call. Task-specific examples here directly bias tool selection.

---

## 2. Default Main Prompt

**File:** `core/prompt_templates.py`
**Lines:** 36-70
**Function:** `DEFAULT_MAIN_PROMPT`

### Issues Found:

#### ❌ Issue 3: Malnutrition Example in RAG Query Guidance (Line 54)

```python
INTELLIGENT QUERIES:
Build RAG queries from clinical text and ICD classification, targeting relevant guidelines (e.g., "ASPEN malnutrition criteria").
                                                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                                                                              TASK-SPECIFIC EXAMPLE!
```

**Problem:**
- Suggests "ASPEN malnutrition criteria" as example
- Should be generic: "e.g., 'relevant clinical guidelines'"
- Or provide multiple diverse examples: "e.g., 'sepsis criteria', 'AKI staging', 'cardiac guidelines'"

#### Severity: **MEDIUM**
This is the default template users see, so the example might suggest the platform is nutrition-focused.

---

## 3. Default RAG Refinement Prompt

**File:** `core/prompt_templates.py`
**Lines:** 98-159
**Function:** `DEFAULT_RAG_REFINEMENT_PROMPT`

### Issues Found:

#### ❌ Issue 4: "Nutritional Status" in Task Description (Line 100)

```python
DEFAULT_RAG_REFINEMENT_PROMPT = """[RAG REFINEMENT TASK]

You are refining a preliminary clinical extraction using evidence from authoritative sources,
acting as a clinical expert to curate high-quality training data for a conversational AI.
Synthesize findings to support the ICD classification of nutritional status.
                                                              ^^^^^^^^^^^^^^^^^^
                                                              TASK-SPECIFIC!
```

**Problem:**
- "nutritional status" is task-specific
- Should be: "Synthesize findings to support the ICD classification."
- Or: "...to support the clinical classification/assessment."

#### ❌ Issue 5: ENTIRE Malnutrition Synthesis Section (Lines 129-146)

```python
SYNTHESIS GUIDELINES FOR MALNUTRITION ASSESSMENT:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ENTIRE SECTION IS TASK-SPECIFIC!

- For ICD classification "MALNUTRITION PRESENT":
  • Synthesize evidence supporting presence (e.g., low z-scores, wasting, inadequate intake)
  • Classify severity (mild, moderate, severe) per guidelines (e.g., ASPEN)
  • Identify etiology (e.g., illness-related, non-illness-related)
- For ICD classification "NO MALNUTRITION":
  • Synthesize evidence supporting adequate nutritional status
  ...
```

**Problem:**
- This ENTIRE section (18 lines) is malnutrition-specific
- Should NOT be in "DEFAULT" prompt
- Belongs only in malnutrition-specific template
- Severely biases the system toward nutrition tasks

#### Severity: **HIGH**
This is explicitly telling the LLM how to handle malnutrition cases, even when the user is working on completely different tasks.

---

## 4. Prompt Length Analysis

**User's Concern:** "Should not be too long. Remember the task prompt may be long itself"

### Current Lengths:

| Prompt | Lines | Tokens (est.) | Assessment |
|--------|-------|---------------|------------|
| STAGE1_ANALYSIS_PROMPT | 102 | ~2,500 | ⚠️ **TOO LONG** |
| DEFAULT_MAIN_PROMPT | 35 | ~900 | ✅ Reasonable |
| DEFAULT_MINIMAL_PROMPT | 27 | ~700 | ✅ Good |
| DEFAULT_RAG_REFINEMENT_PROMPT | 62 | ~1,600 | ⚠️ **Could be shorter** |

**Issue:**
- Stage 1 Analysis Prompt is 102 lines (~2,500 tokens)
- Combined with user's potentially long task prompt, could exceed context limits
- Contains verbose examples that could be removed

**Recommendation:**
- Reduce Stage 1 to ~40-50 lines (~1,200 tokens)
- Remove verbose examples
- Keep only essential instructions

---

## 5. Gap Analysis Summary

### What's Generic (✅ Good):

1. Prompt structure uses placeholders: `{task_description}`, `{json_schema}`, `{clinical_text}`
2. Instructions are generally task-agnostic
3. Tool descriptions are generic

### What's Task-Specific (❌ Needs Fixing):

1. **Stage 1 Analysis Prompt:**
   - Malnutrition schema example (line 576)
   - Malnutrition RAG query examples (lines 609-615)

2. **Default Main Prompt:**
   - "ASPEN malnutrition criteria" example (line 54)

3. **Default RAG Refinement Prompt:**
   - "nutritional status" (line 100)
   - Entire malnutrition synthesis section (lines 129-146)

### Prompt Length Issues:

4. **Stage 1 Analysis Prompt:**
   - Too long (102 lines, ~2,500 tokens)
   - Verbose examples

---

## 6. Recommended Fixes

### Fix 1: Make Stage 1 Analysis Prompt Fully Generic

**Current (Lines 554-576):**
```
- Include schema-related terms (e.g., if schema has "malnutrition_status", include "malnutrition" in queries)
```

**Fixed:**
```
- Include schema-related terms based on YOUR task's schema fields
```

**Current (Lines 593-617) - Examples:**
```json
{
  "tool": "rag",
  "keywords": ["malnutrition criteria", "ASPEN guidelines", "pediatric nutrition"],
  ...
}
```

**Fixed - Remove specific examples or use diverse generic ones:**
```json
{
  "tool": "rag",
  "keywords": ["<relevant clinical criteria>", "<relevant guidelines>", "<domain context>"],
  "reasoning": "Retrieve evidence-based criteria relevant to the extraction task"
}
```

### Fix 2: Shorten Stage 1 Prompt

**Reduce from 102 lines to ~50 lines by:**
- Removing verbose parameter extraction rules
- Simplifying examples
- Condensing instructions

### Fix 3: Remove Task-Specific Language from Defaults

**DEFAULT_MAIN_PROMPT - Remove:**
```
- Line 54: Change "ASPEN malnutrition criteria" → "relevant clinical guidelines"
```

**DEFAULT_RAG_REFINEMENT_PROMPT - Remove:**
```
- Line 100: Change "nutritional status" → generic "clinical classification"
- Lines 129-146: REMOVE entire malnutrition synthesis section (belongs only in malnutrition template)
```

---

## 7. Proposed Generic Stage 1 Prompt

```python
STAGE1_ANALYSIS_PROMPT = """[SYSTEM INSTRUCTION]
You are an intelligent task analyst planning clinical data extraction. Analyze the extraction requirements,
identify available data, and determine which tools would help complete the task.

[EXTRACTION TASK]
{task_description}

EXPECTED OUTPUT SCHEMA:
{json_schema}

[AVAILABLE TOOLS]
{available_tools_description}

[YOUR TASK]

1. **Understand Requirements**: Review the task description and output schema to understand what information is needed.

2. **Analyze Clinical Text**: Identify key information in the clinical text that relates to the extraction requirements.

3. **Gap Analysis**: Determine what's missing:
   - Which calculations are needed?
   - Which guidelines/criteria would help?
   - Which contextual information would assist?

4. **Select Tools**: Based on gaps, determine which tools to call:
   - **Functions**: When calculations or conversions are needed
   - **RAG**: When guidelines, criteria, or evidence-based standards would help
   - **Extras**: When supplementary context or domain knowledge would assist

[CLINICAL TEXT]
{clinical_text}

[LABEL CLASSIFICATION]
{label_context}

[OUTPUT FORMAT]
Return JSON with this structure:
{{
  "analysis": "Brief analysis of required information and tools that will help",
  "tool_requests": [
    {{
      "tool": "function",
      "name": "<function_name>",
      "arguments": {{"param1": "value1", ...}},
      "reasoning": "Why this function is needed for the task"
    }},
    {{
      "tool": "rag",
      "keywords": ["<keyword1>", "<keyword2>", ...],
      "reasoning": "What information this will retrieve and why it's needed"
    }},
    {{
      "tool": "extras",
      "keywords": ["<keyword1>", "<keyword2>", ...],
      "reasoning": "What contextual information this will provide"
    }}
  ]
}}

CRITICAL: Select tools that are REQUIRED to complete the extraction task as defined in the task description.
Tools should directly support completing the required output schema."""
```

**Length:** ~40 lines (~1,000 tokens) - 60% reduction!

---

## 8. Testing Plan

To validate these changes work for diverse tasks:

### Test Case 1: Malnutrition (Original)
- Schema: malnutrition_status, growth_and_anthropometrics
- Expected Tools: calculate_bmi, percentile_to_zscore, query_rag("ASPEN")
- ✅ Should still work with generic prompt

### Test Case 2: Sepsis (New Domain)
- Schema: sepsis_diagnosis, SOFA_score, qSOFA_criteria
- Expected Tools: calculate_sofa_score, query_rag("sepsis criteria"), query_extras("sepsis")
- ✅ Should NOT be biased toward malnutrition

### Test Case 3: AKI (New Domain)
- Schema: aki_stage, creatinine_trend, KDIGO_classification
- Expected Tools: classify_aki_stage_kdigo, query_rag("KDIGO AKI"), query_extras("AKI")
- ✅ Should work equally well

### Test Case 4: Diabetes (Existing Template)
- Schema: hba1c_value, diabetes_type, complications
- Expected Tools: None or minimal (mostly extraction from text)
- ✅ Should work without nutrition bias

---

## 9. Summary & Recommendations

### Critical Issues (Fix Immediately):

1. ❌ **Stage 1 Analysis Prompt** - Remove malnutrition examples (HIGH priority)
2. ❌ **Default RAG Refinement Prompt** - Remove malnutrition synthesis section (HIGH priority)
3. ⚠️ **Stage 1 Prompt Length** - Reduce from 102 to ~50 lines (MEDIUM priority)

### Minor Issues (Fix Soon):

4. ❌ **Default Main Prompt** - Change "ASPEN malnutrition criteria" to generic example (LOW priority)
5. ❌ **Default RAG Refinement** - Change "nutritional status" to generic "classification" (LOW priority)

### Validation:

6. ✅ **Test with diverse tasks** to ensure no bias

---

## 10. Implementation Plan

1. **Create new generic prompts** (this analysis)
2. **Update `core/prompt_templates.py`** with fixes
3. **Test with diverse use cases** (malnutrition, sepsis, AKI, diabetes)
4. **Validate performance** doesn't degrade
5. **Commit changes** with detailed explanation

**Estimated Impact:** Improves universality, reduces bias, shortens prompts, maintains quality

---

## Conclusion

**User's concern is 100% valid.** The current prompts contain task-specific examples that bias the system toward malnutrition tasks, contradicting the "universal platform" design.

**The fix is straightforward:** Remove task-specific examples, use generic language, and shorten prompts.

**After fix:** System will be truly universal, driven solely by user's task description and schema.

---

**Next Steps:** Implement proposed generic prompt templates?
