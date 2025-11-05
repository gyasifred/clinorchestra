# Prompt Flow Analysis: How Task-Specific Prompts are Used
## Complete Analysis of Malnutrition Prompt Integration in Agentic Mode

**Author:** Claude (responding to user question)
**Date:** 2025-11-05

---

## Your Question

> "Are you saying in the agentic system, the prompts, for example, of task-specific prompts like malnutrition, are not used?? If so, how will the agent understand the requirements of my task?"

## TL;DR Answer

**YES, task-specific prompts (like MALNUTRITION_MAIN_PROMPT) ARE USED in agentic mode!**

However, there is a **potential issue** with how they're integrated - they appear as "ADDITIONAL INSTRUCTIONS" at the bottom of the agentic prompt instead of being the PRIMARY task definition. This may reduce their effectiveness.

---

## Complete Prompt Flow Analysis

### 1. Template Registration (prompt_templates.py)

```python
# Line 648-658
PROMPT_TEMPLATE_REGISTRY_V1 = {
    "malnutrition": {
        "main": MALNUTRITION_MAIN_PROMPT,        # Your detailed malnutrition prompt
        "minimal": MALNUTRITION_MINIMAL_PROMPT,
        "rag_prompt": MALNUTRITION_RAG_REFINEMENT_PROMPT,
        "description": "Pediatric Malnutrition Assessment...",
        "version": "1.0.5",
        "schema": { ... }
    }
}
```

**MALNUTRITION_MAIN_PROMPT contains:**
- Task description for pediatric malnutrition assessment
- z-score interpretation guidelines (WHO/ASPEN thresholds)
- Function calling instructions (interpret_zscore_malnutrition, etc.)
- Temporal data capture requirements
- Synthesis structure (8 detailed sections)
- PHI anonymization rules
- Retrieved evidence integration instructions

### 2. User Loads Template (ui/prompt_tab.py)

```python
# Line 583-629: load_example()
def load_example(selected):
    """Load selected template and auto-populate schema"""
    template = get_template("malnutrition")  # Gets template from registry

    main_p = template.get("main", "")  # ← Gets MALNUTRITION_MAIN_PROMPT
    min_p = template.get("minimal", "")
    rag_p = template.get("rag_prompt", "")

    # Returns to UI
    return main_p, min_p, rag_p, ...
```

**Result:** User sees MALNUTRITION_MAIN_PROMPT in the UI text area.

### 3. User Saves Configuration (core/app_state.py)

```python
# Line 323-324
def set_prompt_config(..., main_prompt, ...):
    self.prompt_config.main_prompt = main_prompt  # ← MALNUTRITION_MAIN_PROMPT
    self.prompt_config.base_prompt = main_prompt  # ← SAME AS MAIN
```

**Result:** `app_state.prompt_config.base_prompt = MALNUTRITION_MAIN_PROMPT`

### 4. CLASSIC MODE: How Prompts are Used

```python
# core/agent_system.py Line 1005-1020
def _build_stage3_prompt(self):
    """Build Stage 3 extraction prompt (CLASSIC MODE)"""
    base_prompt = self.app_state.prompt_config.base_prompt or ""  # ← MALNUTRITION_MAIN_PROMPT

    # Format tool results
    tool_outputs = format_tool_outputs_for_prompt(self.context.tool_results)

    # MALNUTRITION_MAIN_PROMPT contains placeholders:
    # {clinical_text}, {label_context}, {rag_outputs}, {function_outputs}, {extras_outputs}

    # Fill in the placeholders
    prompt = base_prompt.format(
        clinical_text=self.context.clinical_text,
        label_context=self.context.label_context,
        rag_outputs=tool_outputs['rag_outputs'],
        function_outputs=tool_outputs['function_outputs'],
        extras_outputs=tool_outputs['extras_outputs']
    )

    return prompt
```

**In Classic Mode:**
- MALNUTRITION_MAIN_PROMPT is the COMPLETE prompt
- All task-specific requirements are PRIMARY
- Tool results are inserted via placeholders
- ✅ Works perfectly

### 5. AGENTIC MODE: How Prompts are Used

#### Step A: Agent Creates Initial Prompt

```python
# core/agentic_agent.py Line 603-619
def _build_agentic_initial_prompt(self):
    """Build initial agentic prompt"""
    from core.prompt_templates import get_agentic_extraction_prompt

    prompt = get_agentic_extraction_prompt(
        clinical_text=self.context.clinical_text,
        label_context=self.context.label_context or "No label provided",
        json_schema=json.dumps(self.app_state.prompt_config.json_schema, indent=2),
        schema_instructions=schema_instructions,
        base_prompt=self.app_state.prompt_config.base_prompt or ""  # ← MALNUTRITION_MAIN_PROMPT
    )

    return prompt
```

#### Step B: Agentic Prompt Template

```python
# core/prompt_templates.py Line 1059-1174
def get_agentic_extraction_prompt(..., base_prompt: str = ""):
    prompt = f"""You are a board-certified clinical expert performing structured information extraction from medical text.

**YOUR TASK:**
Extract structured clinical information to create comprehensive, expert-level annotations. Your extraction must support the ground truth diagnosis provided below.

**GROUND TRUTH DIAGNOSIS (YOU MUST SUPPORT THIS):**
{label_context}

**CLINICAL TEXT TO ANALYZE:**
{clinical_text}

**AVAILABLE TOOLS (call as many times as needed):**

1. **query_rag(query, purpose)**: Retrieve clinical guidelines...
2. **call_[function_name](parameters)**: Perform medical calculations...
3. **query_extras(keywords)**: Get supplementary hints...

**AGENTIC WORKFLOW (How to work):**
1. **ANALYZE**: Carefully read the clinical text and ground truth diagnosis
2. **DISCOVER**: Identify what information you need to complete the extraction
3. **REQUEST TOOLS**: Call tools to gather information...
... (more generic agentic instructions) ...

**EXPECTED OUTPUT SCHEMA:**
{json_schema}

**SCHEMA INSTRUCTIONS:**
{schema_instructions}

**ADDITIONAL INSTRUCTIONS FROM BASE PROMPT:**   ← ⚠️ ISSUE HERE
{base_prompt}                                   ← ⚠️ MALNUTRITION_MAIN_PROMPT GOES HERE

**CRITICAL RULES:**
- **ANONYMIZE**: NEVER use patient or family names...
- **SUPPORT GROUND TRUTH**: Your extraction MUST align with the ground truth diagnosis
... """

    return prompt
```

**⚠️ THE ISSUE:**

Your MALNUTRITION_MAIN_PROMPT (which contains ALL the task-specific requirements) is being placed under:

```
**ADDITIONAL INSTRUCTIONS FROM BASE PROMPT:**
```

This makes it seem SECONDARY/SUPPLEMENTARY rather than being the PRIMARY task definition!

---

## Example: What the LLM Actually Sees in Agentic Mode

When you run a malnutrition extraction in agentic mode, the LLM receives:

```
You are a board-certified clinical expert performing structured information extraction...

YOUR TASK:
Extract structured clinical information to create comprehensive, expert-level annotations.

GROUND TRUTH DIAGNOSIS:
MALNUTRITION PRESENT

CLINICAL TEXT:
3-year-old with poor appetite for 2 months. Weight 12.5 kg (8th percentile)...

AVAILABLE TOOLS:
1. query_rag(query, purpose): Retrieve clinical guidelines...
2. call_[function_name](parameters): Perform medical calculations...
3. query_extras(keywords): Get supplementary hints...

AGENTIC WORKFLOW:
1. ANALYZE: Carefully read the clinical text
2. DISCOVER: Identify what information you need
3. REQUEST TOOLS: Call tools to gather information
... (generic instructions) ...

EXPECTED OUTPUT SCHEMA:
{
  "malnutrition_status": "string [REQUIRED]",
  "symptoms_timeline": "string [REQUIRED]",
  ...
}

**ADDITIONAL INSTRUCTIONS FROM BASE PROMPT:**

[TASK DESCRIPTION - Pediatric Malnutrition Clinical Assessment]  ← YOUR MALNUTRITION_MAIN_PROMPT

You are a board-certified pediatric dietitian and clinical expert performing malnutrition assessment...

**Z-SCORE INTERPRETATION CONVENTION:**
Percentile < 50th = NEGATIVE z-score
- 3rd percentile = z-score -1.88
- 10th percentile = z-score -1.28
... (all your detailed malnutrition instructions) ...

**SYNTHESIS STRUCTURE:**
1. CASE PRESENTATION: Setting, chief concern, timeline...
2. CLINICAL SYMPTOMS - TEMPORAL: Document ALL symptoms with DATES...
... (all 8 sections) ...

**CRITICAL RULES:**
- ANONYMIZE: Use "the patient"...
```

---

## The Problem

1. **Generic agentic instructions come FIRST** (lines 1-100)
2. **Your specific malnutrition requirements come LAST** (lines 150-400)
3. Labeled as "ADDITIONAL" suggests they're supplementary
4. LLM attention may focus on the generic instructions at the top
5. Your detailed malnutrition requirements might be treated as "nice to have" instead of "must follow"

---

## The Solution: Restructure Agentic Prompt

### Option 1: Swap Order (Recommended)

Put task-specific requirements FIRST, generic agentic instructions SECOND:

```python
def get_agentic_extraction_prompt(..., base_prompt: str = ""):
    prompt = f"""
{base_prompt}  ← ✅ PUT MALNUTRITION_MAIN_PROMPT FIRST

**GROUND TRUTH DIAGNOSIS (YOU MUST SUPPORT THIS):**
{label_context}

**CLINICAL TEXT TO ANALYZE:**
{clinical_text}

**EXPECTED OUTPUT SCHEMA:**
{json_schema}

{schema_instructions}

═══════════════════════════════════════════════════════════════
AGENTIC EXECUTION FRAMEWORK
═══════════════════════════════════════════════════════════════

The task description above defines WHAT to extract and HOW to synthesize.
This section defines the AGENTIC WORKFLOW for tool calling:

**AVAILABLE TOOLS:**
1. query_rag(query, purpose): Retrieve clinical guidelines...
2. call_[function_name](parameters): Perform medical calculations...
3. query_extras(keywords): Get supplementary hints...

**AGENTIC WORKFLOW:**
1. ANALYZE: Read the task description above and clinical text
2. DISCOVER: Identify what tools you need based on the task requirements
3. REQUEST TOOLS: Call tools to gather information
4. LEARN: Analyze results
5. ITERATE: Call more tools if needed
6. EXTRACT: Complete the extraction per the task description above

**CRITICAL:**
- Follow the task-specific requirements defined at the top
- Use tools iteratively to gather needed information
- Output must match the synthesis structure defined in task description
"""
    return prompt
```

### Option 2: Integrate Base Prompt More Seamlessly

Instead of having a separate "ADDITIONAL INSTRUCTIONS" section, integrate the base_prompt's key elements:

```python
def get_agentic_extraction_prompt(..., base_prompt: str = ""):
    # Extract role from base_prompt if present
    role = extract_role_from_prompt(base_prompt) or "board-certified clinical expert"

    # Extract task requirements
    task_requirements = extract_task_section(base_prompt) or ""

    # Extract synthesis structure
    synthesis_structure = extract_synthesis_section(base_prompt) or ""

    prompt = f"""You are a {role} performing structured clinical information extraction.

**YOUR SPECIFIC TASK:**
{task_requirements}

**GROUND TRUTH DIAGNOSIS:**
{label_context}

**CLINICAL TEXT:**
{clinical_text}

**HOW TO COMPLETE THIS TASK (Agentic Workflow):**

You have access to tools that help gather information:
1. query_rag: Retrieve clinical guidelines (ASPEN, WHO, CDC)
2. call_[function]: Medical calculations (z-scores, BMI, percentiles)
3. query_extras: Supplementary hints

WORKFLOW:
1. Analyze the clinical text against task requirements
2. Call tools to gather needed information
3. Learn from results, call more tools if needed
4. Extract information following the synthesis structure below

**REQUIRED SYNTHESIS STRUCTURE:**
{synthesis_structure}

**OUTPUT SCHEMA:**
{json_schema}
{schema_instructions}
"""
    return prompt
```

### Option 3: Use Base Prompt AS-IS with Tool Instructions Appended

Treat base_prompt as the COMPLETE task definition, just append tool calling instructions:

```python
def get_agentic_extraction_prompt(..., base_prompt: str = ""):
    # Use base_prompt exactly as written
    # Just append agentic tool-calling framework

    prompt = f"""{base_prompt}

═══════════════════════════════════════════════════════════════
AGENTIC TOOL-CALLING FRAMEWORK (How to execute the task above)
═══════════════════════════════════════════════════════════════

To complete the task described above, you have access to these tools:

**AVAILABLE TOOLS:**
1. query_rag(query, purpose): Retrieve guidelines/standards from authoritative sources
2. call_[function_name](parameters): Perform medical calculations
3. query_extras(keywords): Get supplementary hints

**HOW TO USE TOOLS:**
- Call tools ITERATIVELY as you discover needs
- Don't plan all tools upfront - adapt based on results
- Call the same tool multiple times with refined queries if needed
- Tools execute in PARALLEL for performance

**EXECUTION FLOW:**
1. Start analyzing the clinical text per task requirements above
2. When you need information → Call appropriate tools
3. Receive results → Learn from them
4. Need more? → Call more tools
5. Have enough? → Complete extraction per task requirements

Follow the task requirements, synthesis structure, and schema defined above.
Use tools to gather the information you need to complete those requirements.
"""
    return prompt
```

---

## Recommendation

I recommend **Option 3** (Use base_prompt as-is with tool instructions appended) because:

1. ✅ Preserves your carefully crafted task-specific prompts EXACTLY as you wrote them
2. ✅ Makes task requirements PRIMARY, not "additional"
3. ✅ Clearly separates "WHAT to extract" (base_prompt) from "HOW to use tools" (agentic framework)
4. ✅ No complex parsing or extraction logic needed
5. ✅ User's malnutrition prompt requirements are front and center
6. ✅ Agentic tool-calling instructions are clearly positioned as the "execution mechanism"

---

## Files That Need Changes

### 1. core/prompt_templates.py

**Function:** `get_agentic_extraction_prompt()`
**Lines:** 1059-1174

**Change:** Restructure to put base_prompt FIRST as the primary task definition, append agentic tool-calling framework.

### 2. No other files need changes

The integration point is only in `get_agentic_extraction_prompt()`. The rest of the system correctly:
- Loads templates (prompt_tab.py) ✅
- Saves to base_prompt (app_state.py) ✅
- Passes to agentic prompt builder (agentic_agent.py) ✅

---

## Current vs Proposed Structure

### Current Structure (PROBLEMATIC)

```
[Generic agentic intro - 100 lines]
    ↓
[YOUR TASK: generic description]
    ↓
[AVAILABLE TOOLS: tool descriptions]
    ↓
[AGENTIC WORKFLOW: generic instructions]
    ↓
[SCHEMA: json schema]
    ↓
[**ADDITIONAL INSTRUCTIONS**: YOUR MALNUTRITION PROMPT]  ← Appears supplementary
```

### Proposed Structure (BETTER)

```
[YOUR MALNUTRITION PROMPT - Complete task definition]  ← PRIMARY
    ↓
[GROUND TRUTH DIAGNOSIS]
    ↓
[CLINICAL TEXT]
    ↓
[═══ AGENTIC TOOL-CALLING FRAMEWORK ═══]  ← Execution mechanism
[AVAILABLE TOOLS: how to gather information]
[EXECUTION FLOW: how to use tools iteratively]
```

---

## Summary

**Your Question:** Are task-specific prompts used in agentic mode?

**Answer:** YES, they are used (specifically `app_state.prompt_config.base_prompt`).

**BUT:** They're currently positioned as "ADDITIONAL INSTRUCTIONS" at the BOTTOM of a long prompt, which may reduce their effectiveness compared to classic mode where they ARE the complete prompt.

**Solution:** Restructure `get_agentic_extraction_prompt()` to put `base_prompt` FIRST as the primary task definition, with agentic tool-calling instructions APPENDED as the execution framework.

**Impact:** Ensures your carefully crafted malnutrition assessment requirements are treated as PRIMARY instructions, not supplementary additions.

---

**Ready to implement the fix?** Let me know if you want me to restructure the agentic prompt function!
