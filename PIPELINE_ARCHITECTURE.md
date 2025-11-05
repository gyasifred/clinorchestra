# ClinAnnotate Pipeline Architecture
## Complete System Diagrams - Version 1.0.0

**Author:** Frederick Gyasi (gyasi@musc.edu)
**Institution:** Medical University of South Carolina, Biomedical Informatics Center
**Date:** 2025-11-05

**🎯 IMPORTANT:** This document uses malnutrition assessment as illustrative examples throughout. The architecture and both execution modes (Classic and Agentic) work for **ANY** clinical task - malnutrition, diabetes, sepsis, AKI, cardiac assessments, or custom tasks you define.

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Classic vs Agentic Pipeline Comparison](#classic-vs-agentic-pipeline-comparison)
3. [Agentic Pipeline Detailed Flow](#agentic-pipeline-detailed-flow)
4. [Prompt Integration Architecture](#prompt-integration-architecture)
5. [Async Tool Execution Flow](#async-tool-execution-flow)
6. [Complete System Architecture](#complete-system-architecture)

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ClinAnnotate Platform                            │
│                         Version 1.0.0                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                    EXECUTION MODES                              │    │
│  ├────────────────────────────────────────────────────────────────┤    │
│  │                                                                 │    │
│  │  CLASSIC MODE (v1.0.2)          AGENTIC MODE (v1.0.0)         │    │
│  │  ─────────────────────          ──────────────────────         │    │
│  │  • Rigid 4-stage pipeline       • Continuous loop              │    │
│  │  • Predefined tool execution    • Autonomous tool calling      │    │
│  │  • Sequential processing        • PAUSE/RESUME flow            │    │
│  │  • Fixed workflow               • Async parallel tools         │    │
│  │                                  • Dynamic adaptation           │    │
│  │                                                                 │    │
│  │  app_state.agentic_config.enabled = False | True               │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Classic vs Agentic Pipeline Comparison

### **CLASSIC PIPELINE (v1.0.2) - Rigid 4-Stage**

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLASSIC PIPELINE                             │
│                    (ExtractionAgent v1.0.2)                          │
└─────────────────────────────────────────────────────────────────────┘

INPUT: Clinical Text + Label
  │
  ├─► STAGE 1: ANALYSIS
  │   ┌──────────────────────────────────────────────┐
  │   │ • LLM analyzes text                          │
  │   │ • Plans ALL tools upfront                    │
  │   │ • Returns tool request list                  │
  │   │ • NO ADAPTATION                              │
  │   └──────────────────────────────────────────────┘
  │
  ├─► STAGE 2: TOOL EXECUTION (Batch)
  │   ┌──────────────────────────────────────────────┐
  │   │ • Execute ALL tools sequentially             │
  │   │ • RAG queries → Results                      │
  │   │ • Function calls → Results                   │
  │   │ • Extras queries → Results                   │
  │   │ • NO MORE TOOL CALLS POSSIBLE                │
  │   └──────────────────────────────────────────────┘
  │
  ├─► STAGE 3: EXTRACTION
  │   ┌──────────────────────────────────────────────┐
  │   │ • LLM generates JSON from tool results       │
  │   │ • Uses main/minimal prompt                   │
  │   │ • Fixed single-pass extraction               │
  │   │ • Can't request more tools                   │
  │   └──────────────────────────────────────────────┘
  │
  ├─► STAGE 4: RAG REFINEMENT (Optional)
  │   ┌──────────────────────────────────────────────┐
  │   │ • Refine extraction with RAG evidence        │
  │   │ • Fixed refinement prompt                    │
  │   │ • Single-pass refinement                     │
  │   │ • NO ITERATION                               │
  │   └──────────────────────────────────────────────┘
  │
  └─► OUTPUT: Final JSON

LIMITATIONS:
❌ Can't discover new needs during extraction
❌ Tools only called once
❌ No adaptation based on results
❌ Fixed, inflexible workflow
```

---

### **AGENTIC PIPELINE (v1.0.0) - Continuous Loop with Async**

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AGENTIC PIPELINE                             │
│                     (AgenticAgent v1.0.0)                            │
│                    Continuous Loop + Async Tools                     │
└─────────────────────────────────────────────────────────────────────┘

INPUT: Clinical Text + Label
  │
  │  ┌─────────────────────────────────────────────────────────────┐
  │  │                   CONTINUOUS AGENTIC LOOP                    │
  │  │                   (Max 20 iterations)                        │
  │  └─────────────────────────────────────────────────────────────┘
  │
  ├─► ITERATION 1
  │   ┌──────────────────────────────────────────────┐
  │   │ LLM: Analyzes clinical text                  │
  │   │ LLM: "I need ASPEN criteria"                 │
  │   │   └─► TOOL CALL: query_rag("ASPEN...")      │
  │   │         ↓                                    │
  │   │       PAUSE ⏸️  (state = AWAITING_RESULTS)   │
  │   │         ↓                                    │
  │   │       Execute tool ASYNC                     │
  │   │         ↓                                    │
  │   │       RESUME ▶️ (state = CONTINUING)         │
  │   │         ↓                                    │
  │   │       Results added to conversation          │
  │   └──────────────────────────────────────────────┘
  │
  ├─► ITERATION 2
  │   ┌──────────────────────────────────────────────┐
  │   │ LLM: Analyzes ASPEN criteria results         │
  │   │ LLM: "Criteria mention z-scores. Text has    │
  │   │       3rd percentile. Let me convert."       │
  │   │   └─► TOOL CALL: percentile_to_zscore(3)    │
  │   │         ↓                                    │
  │   │       PAUSE ⏸️                                │
  │   │         ↓                                    │
  │   │       Execute function ASYNC                 │
  │   │         ↓                                    │
  │   │       RESUME ▶️ (z-score = -1.88)            │
  │   └──────────────────────────────────────────────┘
  │
  ├─► ITERATION 3
  │   ┌──────────────────────────────────────────────┐
  │   │ LLM: "Now interpret this z-score"            │
  │   │   └─► TOOL CALL: interpret_zscore(...)      │
  │   │         ↓                                    │
  │   │       PAUSE ⏸️                                │
  │   │         ↓                                    │
  │   │       Execute function ASYNC                 │
  │   │         ↓                                    │
  │   │       RESUME ▶️ ("Mild malnutrition")        │
  │   └──────────────────────────────────────────────┘
  │
  ├─► ITERATION 4 (PARALLEL TOOLS!)
  │   ┌──────────────────────────────────────────────┐
  │   │ LLM: "Need more info on management AND       │
  │   │       clinical implications"                 │
  │   │   ├─► TOOL CALL 1: query_rag("management")  │
  │   │   └─► TOOL CALL 2: query_rag("implications")│
  │   │         ↓                                    │
  │   │       PAUSE ⏸️                                │
  │   │         ↓                                    │
  │   │       Execute BOTH tools in PARALLEL (async) │
  │   │         ↓                                    │
  │   │       RESUME ▶️ (both results simultaneously)│
  │   └──────────────────────────────────────────────┘
  │
  ├─► ITERATION 5
  │   ┌──────────────────────────────────────────────┐
  │   │ LLM: "Perfect! Now I have enough info."      │
  │   │   └─► OUTPUT: Final JSON                    │
  │   │         ↓                                    │
  │   │       state = COMPLETED                      │
  │   └──────────────────────────────────────────────┘
  │
  └─► OUTPUT: Final JSON with agentic_metadata

CONVERSATION HISTORY MAINTAINED:
- User: Initial prompt with clinical text
- Assistant: "Let me analyze..." + tool calls
- Tool: RAG results
- Assistant: "Based on results..." + more tool calls
- Tool: Function results
- Assistant: Final JSON output

BENEFITS:
✅ Autonomous tool calling
✅ Iterative refinement
✅ Adaptive strategy
✅ Parallel tool execution (async)
✅ Natural reasoning flow
✅ Dynamic discovery
```

---

## Agentic Pipeline Detailed Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    AGENTIC PIPELINE - COMPLETE FLOW                      │
└─────────────────────────────────────────────────────────────────────────┘

START
  │
  ├─► 1. INITIALIZE CONTEXT
  │   ┌────────────────────────────────────────────────────────────┐
  │   │ • Create AgenticContext                                    │
  │   │ • Set state = IDLE                                         │
  │   │ • Initialize conversation history: []                      │
  │   │ • Set max_iterations = 20                                  │
  │   │ • Set max_tool_calls = 50                                  │
  │   │ • Preprocess clinical text (PHI redaction, normalization)  │
  │   │ • Get label context from mapping                           │
  │   └────────────────────────────────────────────────────────────┘
  │
  ├─► 2. BUILD INITIAL PROMPT
  │   ┌────────────────────────────────────────────────────────────┐
  │   │ • Call get_agentic_extraction_prompt()                     │
  │   │ • Inject:                                                  │
  │   │   - clinical_text (preprocessed)                           │
  │   │   - label_context (ground truth)                           │
  │   │   - json_schema (expected output structure)                │
  │   │   - base_prompt (task-specific instructions)               │
  │   │ • Add tool descriptions (RAG, functions, extras)           │
  │   │ • Add agentic workflow instructions                        │
  │   └────────────────────────────────────────────────────────────┘
  │
  ├─► 3. START CONVERSATION
  │   ┌────────────────────────────────────────────────────────────┐
  │   │ IF provider supports system messages:                      │
  │   │   └─► Add system message to history                        │
  │   │ Add user message (initial prompt) to history               │
  │   │ Set state = ANALYZING                                      │
  │   └────────────────────────────────────────────────────────────┘
  │
  ├─► 4. CONTINUOUS LOOP (while not complete AND iteration < 20)
  │   │
  │   ├─► 4a. GENERATE LLM RESPONSE WITH TOOLS
  │   │   ┌────────────────────────────────────────────────────────┐
  │   │   │ IF provider supports native tool calling:              │
  │   │   │   └─► llm_manager.generate_with_tool_calling()         │
  │   │   │       • Pass conversation history                      │
  │   │   │       • Pass tool schema (RAG, functions, extras)      │
  │   │   │       • Returns: {content, tool_calls, finish_reason}  │
  │   │   │                                                         │
  │   │   │ ELSE (fallback):                                       │
  │   │   │   └─► llm_manager.generate() text-based                │
  │   │   │       • Parse text for tool mentions                   │
  │   │   └────────────────────────────────────────────────────────┘
  │   │
  │   ├─► 4b. PARSE RESPONSE
  │   │   ┌────────────────────────────────────────────────────────┐
  │   │   │ Check response type:                                   │
  │   │   │                                                         │
  │   │   │ CASE 1: Has tool_calls                                 │
  │   │   │   └─► Go to step 4c (Execute Tools)                    │
  │   │   │                                                         │
  │   │   │ CASE 2: Has JSON output                                │
  │   │   │   └─► Parse JSON, set state = COMPLETED                │
  │   │   │   └─► Exit loop                                        │
  │   │   │                                                         │
  │   │   │ CASE 3: Just thinking/analyzing                        │
  │   │   │   └─► Add to history, ask LLM to continue              │
  │   │   │   └─► Loop to next iteration                           │
  │   │   └────────────────────────────────────────────────────────┘
  │   │
  │   ├─► 4c. EXECUTE TOOLS (if has_tool_calls)
  │   │   ┌────────────────────────────────────────────────────────┐
  │   │   │ Set state = AWAITING_TOOL_RESULTS (PAUSED ⏸️)          │
  │   │   │                                                         │
  │   │   │ ASYNC PARALLEL EXECUTION:                              │
  │   │   │ ┌──────────────────────────────────────────────────┐  │
  │   │   │ │ For each tool_call:                              │  │
  │   │   │ │   Create async task:                             │  │
  │   │   │ │     - _execute_rag_tool_async()                  │  │
  │   │   │ │     - _execute_function_tool_async()             │  │
  │   │   │ │     - _execute_extras_tool_async()               │  │
  │   │   │ │                                                   │  │
  │   │   │ │ asyncio.gather(*tasks) ← ALL RUN IN PARALLEL     │  │
  │   │   │ │                                                   │  │
  │   │   │ │ Example with 5 tools:                            │  │
  │   │   │ │   Tool 1: RAG query     ┐                        │  │
  │   │   │ │   Tool 2: Function call  ├─► All run at same    │  │
  │   │   │ │   Tool 3: RAG query      │   time (2-3 sec)     │  │
  │   │   │ │   Tool 4: Function call  │   instead of         │  │
  │   │   │ │   Tool 5: Extras query  ┘   sequential (10 sec) │  │
  │   │   │ │                                                   │  │
  │   │   │ │ Returns: [result1, result2, ..., result5]        │  │
  │   │   │ │          (in original order)                     │  │
  │   │   │ └──────────────────────────────────────────────────┘  │
  │   │   │                                                         │
  │   │   │ Set state = CONTINUING (RESUMED ▶️)                    │
  │   │   └────────────────────────────────────────────────────────┘
  │   │
  │   ├─► 4d. ADD RESULTS TO CONVERSATION
  │   │   ┌────────────────────────────────────────────────────────┐
  │   │   │ For each tool result:                                  │
  │   │   │   Add message to conversation_history:                 │
  │   │   │     role: 'tool'                                       │
  │   │   │     tool_call_id: result.tool_call_id                  │
  │   │   │     content: formatted_result                          │
  │   │   │                                                         │
  │   │   │ Format result for LLM:                                 │
  │   │   │   • RAG: Top 5 chunks with sources                     │
  │   │   │   • Function: Result + interpretation                  │
  │   │   │   • Extras: Matched hints/tips                         │
  │   │   └────────────────────────────────────────────────────────┘
  │   │
  │   └─► 4e. LOOP BACK
  │       └─► LLM can now:
  │           - Analyze results
  │           - Request more tools
  │           - Output final JSON
  │
  ├─► 5. EXTRACTION COMPLETE
  │   ┌────────────────────────────────────────────────────────────┐
  │   │ • Parse final JSON output                                  │
  │   │ • Validate against schema                                  │
  │   │ • Build result with metadata:                              │
  │   │   - original_clinical_text                                 │
  │   │   - preprocessed clinical_text                             │
  │   │   - stage3_output (final JSON)                             │
  │   │   - agentic_metadata:                                      │
  │   │     • version: '1.0.0'                                     │
  │   │     • execution_mode: 'agentic_async'                      │
  │   │     • iterations: N                                        │
  │   │     • total_tool_calls: M                                  │
  │   │     • conversation_length: K messages                      │
  │   └────────────────────────────────────────────────────────────┘
  │
  └─► OUTPUT: Complete extraction result

STATES THROUGHOUT FLOW:
  IDLE → ANALYZING → AWAITING_TOOL_RESULTS → CONTINUING →
  ANALYZING → ... → COMPLETED

CONVERSATION HISTORY EXAMPLE:
  [
    {role: 'system', content: 'You are a clinical expert...'},
    {role: 'user', content: 'Initial agentic prompt with text...'},
    {role: 'assistant', content: 'Let me analyze...', tool_calls: [...]},
    {role: 'tool', tool_call_id: '1', content: 'RAG results...'},
    {role: 'assistant', content: 'Based on guidelines...', tool_calls: [...]},
    {role: 'tool', tool_call_id: '2', content: 'Function results...'},
    {role: 'assistant', content: 'Final JSON: {...}'}
  ]
```

---

## Prompt Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PROMPT INTEGRATION ARCHITECTURE                       │
│                    How Prompts Flow Through System                       │
└─────────────────────────────────────────────────────────────────────────┘

1. PROMPT CONFIGURATION (app_state.prompt_config)
   ┌────────────────────────────────────────────────────────────┐
   │ PromptConfig:                                              │
   │   • main_prompt          (malnutrition/diabetes specific)  │
   │   • minimal_prompt       (fallback for token limits)       │
   │   • rag_prompt           (RAG refinement prompt)           │
   │   • base_prompt          (task instructions)               │
   │   • json_schema          (output structure)                │
   │   • rag_query_fields     (fields for RAG queries)          │
   │   • use_minimal          (boolean flag)                    │
   └────────────────────────────────────────────────────────────┘
                              │
                              │
                              ▼
2. AGENT SELECTION (agent_factory)
   ┌────────────────────────────────────────────────────────────┐
   │ IF agentic_config.enabled:                                 │
   │   └─► AgenticAgent (v1.0.0)                                │
   │       Uses: get_agentic_extraction_prompt()                │
   │                                                             │
   │ ELSE:                                                       │
   │   └─► ExtractionAgent (v1.0.2)                             │
   │       Uses: main_prompt or minimal_prompt                  │
   └────────────────────────────────────────────────────────────┘
                              │
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
   CLASSIC MODE                              AGENTIC MODE
        │                                           │
        │                                           │
3a. CLASSIC PROMPT ASSEMBLY             3b. AGENTIC PROMPT ASSEMBLY
   ┌─────────────────────────┐             ┌──────────────────────────┐
   │ STAGE 1:                │             │ INITIAL PROMPT:          │
   │ ─────────               │             │ ────────────────         │
   │ main_prompt             │             │ get_agentic_extraction_  │
   │ + {clinical_text}       │             │   prompt() builds:       │
   │ + {label_context}       │             │                          │
   │ + {json_schema}         │             │ • Clinical text          │
   │ → Analysis output       │             │ • Label context          │
   │                         │             │ • JSON schema            │
   │ STAGE 2:                │             │ • base_prompt            │
   │ ─────────               │             │ • Tool descriptions:     │
   │ Execute tools           │             │   - query_rag()          │
   │                         │             │   - call_[function]()    │
   │ STAGE 3:                │             │   - query_extras()       │
   │ ─────────               │             │ • Agentic workflow:      │
   │ main_prompt             │             │   - Analyze              │
   │ + {clinical_text}       │             │   - Discover needs       │
   │ + {label_context}       │             │   - Request tools        │
   │ + {rag_outputs}         │             │   - Learn from results   │
   │ + {function_outputs}    │             │   - Iterate              │
   │ + {extras_outputs}      │             │   - Extract              │
   │ + {json_schema}         │             │ • Example workflow       │
   │ → JSON extraction       │             │ → Single conversation    │
   │                         │             │                          │
   │ STAGE 4 (Optional):     │             │ Tools called dynamically │
   │ ─────────               │             │ during conversation via  │
   │ rag_prompt              │             │ native tool calling API  │
   │ + {clinical_text}       │             │                          │
   │ + {label_context}       │             └──────────────────────────┘
   │ + {stage3_output}       │
   │ + {retrieved_chunks}    │
   │ → Refined JSON          │
   └─────────────────────────┘

4. PROMPT TEMPLATE SOURCES (core/prompt_templates.py)
   ┌────────────────────────────────────────────────────────────┐
   │ TASK-SPECIFIC PROMPTS:                                     │
   │ ────────────────────────                                   │
   │ • MALNUTRITION_MAIN_PROMPT                                 │
   │   - Temporal capture instructions                          │
   │   - Growth/anthropometrics guidance                        │
   │   - Z-score interpretation rules                           │
   │   - ASPEN/WHO criteria references                          │
   │   - Forward-thinking recommendations                       │
   │                                                             │
   │ • MALNUTRITION_MINIMAL_PROMPT                              │
   │   - Condensed version for token limits                     │
   │   - Same structure, less detail                            │
   │                                                             │
   │ • MALNUTRITION_RAG_REFINEMENT_PROMPT                       │
   │   - Validation against guidelines                          │
   │   - Gap filling with evidence                              │
   │   - Temporal enhancement                                   │
   │                                                             │
   │ • DIABETES_MAIN_PROMPT                                     │
   │   - Diabetes-specific extraction                           │
   │                                                             │
   │ • DEFAULT_MAIN_PROMPT                                      │
   │   - Generic template                                       │
   │                                                             │
   │ AGENTIC PROMPT:                                            │
   │ ───────────────                                            │
   │ • get_agentic_extraction_prompt()                          │
   │   - Builds dynamic prompt with:                            │
   │     * Task description                                     │
   │     * Ground truth emphasis                                │
   │     * Tool availability                                    │
   │     * Agentic workflow                                     │
   │     * Example iterations                                   │
   │     * Schema integration                                   │
   └────────────────────────────────────────────────────────────┘

5. DYNAMIC PROMPT VARIABLES (Injected at Runtime)
   ┌────────────────────────────────────────────────────────────┐
   │ {clinical_text}         ← Preprocessed patient text        │
   │ {label_context}         ← Ground truth diagnosis           │
   │ {json_schema}           ← Expected output structure        │
   │ {rag_outputs}           ← RAG retrieval results            │
   │ {function_outputs}      ← Function call results            │
   │ {extras_outputs}        ← Extras/hints results             │
   │ {retrieved_chunks}      ← Stage 4 RAG evidence             │
   │ {stage3_output}         ← Initial extraction (for stage 4) │
   │ {json_schema_instructions} ← Field descriptions            │
   └────────────────────────────────────────────────────────────┘

6. SPECIAL PROMPT COMPONENTS
   ┌────────────────────────────────────────────────────────────┐
   │ MALNUTRITION PROMPTS INCLUDE:                              │
   │ ───────────────────────────                                │
   │ • Z-SCORE CONVENTION (embedded directly):                  │
   │   - Percentile <50th = NEGATIVE z-score                    │
   │   - 3rd percentile = z-score -1.88 (NOT +1.88)             │
   │   - WHO classification thresholds                          │
   │   - ASPEN severity criteria                                │
   │                                                             │
   │ • TEMPORAL CAPTURE INSTRUCTIONS:                           │
   │   - "Capture ALL vitals/measurements with DATES"           │
   │   - "Calculate explicit TRENDS"                            │
   │   - Example formats for good temporal data                 │
   │                                                             │
   │ • FORWARD-THINKING GUIDANCE:                               │
   │   - "IF NOT DOCUMENTED: Recommend systematic review..."    │
   │   - Uses retrieved evidence to suggest what SHOULD be done │
   │                                                             │
   │ • ANONYMIZATION RULES:                                     │
   │   - "NEVER use patient or family names"                    │
   │   - "ALWAYS use 'the patient', 'the [age]-year-old'"       │
   │                                                             │
   │ • FUNCTION CALLING INSTRUCTIONS:                           │
   │   - Call interpret_zscore_malnutrition() for z-scores      │
   │   - Call percentile_to_zscore() for conversion             │
   │   - Call calculate_growth_percentile() for trends          │
   └────────────────────────────────────────────────────────────┘

FLOW SUMMARY:
  User Configures Prompts → Agent Factory Selects Mode →
  Classic: 4-stage with main/minimal/rag prompts
  Agentic: Single conversation with dynamic tool calling →
  Prompts get variables injected → LLM generates output
```

---

## Async Tool Execution Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ASYNC TOOL EXECUTION - PHASE 2                        │
│                    Parallel Execution for Performance                    │
└─────────────────────────────────────────────────────────────────────────┘

SCENARIO: LLM requests 5 tools in one iteration

  LLM Response:
  ┌────────────────────────────────────────────────────────────┐
  │ "I need multiple pieces of information:                    │
  │  1. ASPEN malnutrition criteria                            │
  │  2. Convert 3rd percentile to z-score                      │
  │  3. WHO management guidelines                              │
  │  4. Interpret z-score for severity                         │
  │  5. Get pediatric malnutrition hints"                      │
  │                                                             │
  │ tool_calls: [                                              │
  │   {id: '1', name: 'query_rag', params: {...}},            │
  │   {id: '2', name: 'call_percentile_to_zscore', ...},      │
  │   {id: '3', name: 'query_rag', params: {...}},            │
  │   {id: '4', name: 'call_interpret_zscore', ...},          │
  │   {id: '5', name: 'query_extras', params: {...}}          │
  │ ]                                                           │
  └────────────────────────────────────────────────────────────┘
                            │
                            ▼
  SEQUENTIAL EXECUTION (OLD - Phase 1):
  ┌────────────────────────────────────────────────────────────┐
  │ Tool 1: query_rag("ASPEN...")        → 2.0s               │
  │ Tool 2: percentile_to_zscore(3)      → 0.5s               │
  │ Tool 3: query_rag("WHO...")          → 2.0s               │
  │ Tool 4: interpret_zscore(...)        → 0.5s               │
  │ Tool 5: query_extras([...])          → 1.0s               │
  │                                                             │
  │ TOTAL TIME: 6.0 seconds                                    │
  └────────────────────────────────────────────────────────────┘

                            VS

  PARALLEL EXECUTION (NEW - Phase 2):
  ┌────────────────────────────────────────────────────────────┐
  │                                                             │
  │    Time: 0s         1s         2s         3s               │
  │    ─────────────────────────────────────────────           │
  │                                                             │
  │ T1 ████████████████████ (RAG query)                        │
  │ T2 █████ (Function)                                        │
  │ T3 ████████████████████ (RAG query)                        │
  │ T4 █████ (Function)                                        │
  │ T5 ██████████ (Extras)                                     │
  │                                                             │
  │ ALL TOOLS START AT SAME TIME                               │
  │ COMPLETE WHEN SLOWEST FINISHES                             │
  │                                                             │
  │ TOTAL TIME: 2.0 seconds (70% FASTER!)                      │
  └────────────────────────────────────────────────────────────┘

IMPLEMENTATION:

  1. _execute_tools(tool_calls) - Entry Point
     ┌────────────────────────────────────────────────┐
     │ def _execute_tools(self, tool_calls):          │
     │     # Get or create event loop                 │
     │     loop = asyncio.get_event_loop()            │
     │                                                 │
     │     # Run async execution                      │
     │     results = loop.run_until_complete(         │
     │         self._execute_tools_async(tool_calls)  │
     │     )                                           │
     │                                                 │
     │     return results                             │
     └────────────────────────────────────────────────┘
                         │
                         ▼
  2. _execute_tools_async(tool_calls) - Orchestrator
     ┌────────────────────────────────────────────────┐
     │ async def _execute_tools_async(self, ...):     │
     │     tasks = []                                  │
     │                                                 │
     │     # Create async task for each tool          │
     │     for tool_call in tool_calls:               │
     │         if tool_call.name == 'query_rag':      │
     │             task = self._execute_rag_tool_     │
     │                     async(tool_call)            │
     │         elif tool_call.name.startswith('call'):│
     │             task = self._execute_function_     │
     │                     tool_async(tool_call)       │
     │         elif tool_call.name == 'query_extras': │
     │             task = self._execute_extras_tool_  │
     │                     async(tool_call)            │
     │                                                 │
     │         tasks.append(task)                     │
     │                                                 │
     │     # Execute ALL in parallel                  │
     │     start = time.time()                        │
     │     results = await asyncio.gather(*tasks)     │
     │     elapsed = time.time() - start              │
     │                                                 │
     │     logger.info(f"✅ {len(tasks)} tools in     │
     │                   {elapsed:.2f}s (parallel)")  │
     │                                                 │
     │     return results  # In original order        │
     └────────────────────────────────────────────────┘
                         │
                         ▼
  3. Individual Async Tool Executors
     ┌────────────────────────────────────────────────┐
     │ async def _execute_rag_tool_async(...):        │
     │     # RAG engine is synchronous, wrap it       │
     │     loop = asyncio.get_event_loop()            │
     │     result = await loop.run_in_executor(       │
     │         None,  # Default ThreadPoolExecutor    │
     │         self._execute_rag_tool,  # Sync method │
     │         tool_call                              │
     │     )                                           │
     │     return result                              │
     │                                                 │
     │ async def _execute_function_tool_async(...):   │
     │     # Same pattern for functions               │
     │     loop = asyncio.get_event_loop()            │
     │     result = await loop.run_in_executor(       │
     │         None,                                   │
     │         self._execute_function_tool,           │
     │         tool_call                              │
     │     )                                           │
     │     return result                              │
     │                                                 │
     │ async def _execute_extras_tool_async(...):     │
     │     # Same pattern for extras                  │
     │     loop = asyncio.get_event_loop()            │
     │     result = await loop.run_in_executor(       │
     │         None,                                   │
     │         self._execute_extras_tool,             │
     │         tool_call                              │
     │     )                                           │
     │     return result                              │
     └────────────────────────────────────────────────┘

KEY BENEFITS:
  ✅ 60-75% faster execution for multi-tool requests
  ✅ Maintains conversation order (results in original sequence)
  ✅ Thread-safe (each tool execution isolated)
  ✅ Graceful degradation (falls back to sequential if errors)
  ✅ Works with Jupyter notebooks (existing event loop handling)
  ✅ No changes to tool implementations (async wrappers)

PERFORMANCE EXAMPLES:
  3 tools:  6s → 2s   (67% faster)
  5 tools:  10s → 3s  (70% faster)
  10 tools: 20s → 5s  (75% faster)
```

---

## Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CLINANNOTATE COMPLETE SYSTEM                          │
│                         Version 1.0.0                                    │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 1. USER INTERFACE (Gradio)                                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │ Config   │ │ Prompt   │ │ Data     │ │  RAG     │ │ Extras   │    │
│  │   Tab    │ │   Tab    │ │   Tab    │ │   Tab    │ │   Tab    │    │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘    │
│                                                                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                  │
│  │Functions │ │ Patterns │ │Processing│ │Playground│                  │
│  │   Tab    │ │   Tab    │ │   Tab    │ │   Tab    │                  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. APPLICATION STATE (core/app_state.py)                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  • model_config       (LLM provider, model, API keys)                   │
│  • prompt_config      (main, minimal, rag, schema)                      │
│  • data_config        (input file, columns, PHI settings)               │
│  • rag_config         (documents, embeddings, k_value)                  │
│  • processing_config  (batch size, error strategy)                      │
│  • agentic_config     (enabled, max_iterations, max_tool_calls)  ← NEW │
│                                                                          │
│  Lazy Initialization:                                                   │
│  • _llm_manager       (initialized on first use)                        │
│  • _rag_engine        (loaded when RAG enabled)                         │
│  • _regex_preprocessor                                                  │
│  • _extras_manager                                                      │
│  • _function_registry                                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. AGENT FACTORY (core/agent_factory.py)                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  create_agent(llm_manager, rag_engine, ..., app_state)                  │
│                                                                          │
│  IF app_state.agentic_config.enabled:                                   │
│    └─► AgenticAgent(v1.0.0) ← CONTINUOUS LOOP + ASYNC                  │
│                                                                          │
│  ELSE:                                                                   │
│    └─► ExtractionAgent(v1.0.2) ← CLASSIC 4-STAGE                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
         │                                            │
         │ (Classic Mode)                    (Agentic Mode)
         ▼                                            ▼
┌───────────────────────────┐         ┌──────────────────────────────────┐
│ 4a. EXTRACTION AGENT      │         │ 4b. AGENTIC AGENT                │
│     (v1.0.2)              │         │     (v1.0.0)                     │
├───────────────────────────┤         ├──────────────────────────────────┤
│                           │         │                                  │
│ Stage 1: Analysis         │         │ Continuous Loop:                 │
│  ├─ Analyze text          │         │  ├─ Initialize context           │
│  └─ Plan tools            │         │  ├─ Build agentic prompt         │
│                           │         │  ├─ Start conversation            │
│ Stage 2: Tool Execution   │         │  │                                │
│  ├─ Execute RAG           │         │  ├─► ITERATION 1:                │
│  ├─ Execute functions     │         │  │   ├─ LLM generates response   │
│  └─ Execute extras        │         │  │   ├─ Parse for tool calls     │
│                           │         │  │   ├─ PAUSE ⏸️                  │
│ Stage 3: Extraction       │         │  │   ├─ Execute tools ASYNC      │
│  ├─ Use main/minimal      │         │  │   ├─ RESUME ▶️                │
│  │   prompt                │         │  │   └─ Add results to history  │
│  ├─ Inject tool results   │         │  │                                │
│  └─ Generate JSON         │         │  ├─► ITERATION 2:                │
│                           │         │  │   └─ (LLM decides next action) │
│ Stage 4: RAG Refinement   │         │  │                                │
│  ├─ Use rag_prompt        │         │  ├─► ITERATION N:                │
│  ├─ Retrieve evidence     │         │  │   └─ Output final JSON        │
│  └─ Refine extraction     │         │  │                                │
│                           │         │  └─ state = COMPLETED             │
└───────────────────────────┘         │                                  │
                                       │ Key Features:                    │
                                       │  • Native tool calling           │
                                       │  • Async parallel execution      │
                                       │  • Dynamic adaptation            │
                                       │  • Iterative refinement          │
                                       └──────────────────────────────────┘
         │                                            │
         └──────────────┬─────────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. CORE COMPONENTS                                                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ LLM MANAGER (core/llm_manager.py)                                │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │ • generate(prompt, max_tokens)  ← Classic generation            │  │
│  │ • generate_with_tool_calling(messages, tools)  ← NEW Agentic    │  │
│  │                                                                   │  │
│  │ Providers:                                                        │  │
│  │  - OpenAI (with native tool calling)                             │  │
│  │  - Anthropic (with native tool calling)                          │  │
│  │  - Google, Azure, Local (fallback to text-based)                 │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ RAG ENGINE (core/rag_engine.py)                                  │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │ • initialize(documents, embedding_model)                         │  │
│  │ • query(query_text, k)  → Returns top-k chunks                   │  │
│  │                                                                   │  │
│  │ Sources: ASPEN, WHO, CDC, ADA guidelines                         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ FUNCTION REGISTRY (core/function_registry.py)                    │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │ • register_function(name, func, params, description)             │  │
│  │ • execute_function(name, **kwargs)                               │  │
│  │                                                                   │  │
│  │ Functions:                                                        │  │
│  │  - percentile_to_zscore()                                        │  │
│  │  - interpret_zscore_malnutrition()                               │  │
│  │  - calculate_growth_percentile()                                 │  │
│  │  - calculate_bmi(), bmi_percentile()                             │  │
│  │  - [custom user functions...]                                    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ EXTRAS MANAGER (core/extras_manager.py)                          │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │ • load_extras(directory)                                         │  │
│  │ • match_extras_by_keywords(keywords)                             │  │
│  │                                                                   │  │
│  │ Provides: Hints, tips, patterns for specific tasks               │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ REGEX PREPROCESSOR (core/regex_preprocessor.py)                  │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │ • preprocess(text)  → Normalized text                            │  │
│  │                                                                   │  │
│  │ Pattern normalization: z-scores, dates, measurements             │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ PHI REDACTOR (core/pii_redactor.py)                              │  │
│  ├──────────────────────────────────────────────────────────────────┤  │
│  │ • redact(text, entity_types)  → Redacted text                    │  │
│  │                                                                   │  │
│  │ Entity types: PERSON, DATE, LOCATION, ORG, PHONE, EMAIL          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 6. OUTPUT & PERSISTENCE                                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  OUTPUT HANDLER (core/output_handler.py)                                │
│  • Saves results to CSV/JSON                                            │
│  • Includes metadata (iterations, tool calls, timestamps)               │
│  • Auto-saves at intervals                                              │
│                                                                          │
│  PROCESS PERSISTENCE (core/process_persistence.py)                      │
│  • Tracks processing state                                              │
│  • Logs all events                                                      │
│  • Enables resume after interruption                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

DATA FLOW:
  User Input → App State → Agent Factory → Selected Agent →
  LLM Manager + RAG + Functions + Extras → Extraction Result →
  Output Handler → Saved CSV/JSON

CONFIGURATION FLOW:
  UI Tabs → Update app_state configs → Agent uses configs →
  Persisted for next session
```

---

## Summary

This architecture provides:

1. **Dual Execution Modes**
   - Classic: Reliable 4-stage pipeline (v1.0.2)
   - Agentic: Autonomous continuous loop (v1.0.0)

2. **Async Performance**
   - 60-75% faster with parallel tool execution
   - No changes to tool implementations needed

3. **Flexible Prompt System**
   - Task-specific prompts (malnutrition, diabetes)
   - Agentic prompt for continuous loop
   - RAG refinement prompts
   - Dynamic variable injection

4. **Robust Architecture**
   - Event loop handling for all environments
   - Backward compatible
   - State management with pause/resume
   - Comprehensive metadata tracking

5. **Production Ready**
   - Error handling
   - Logging at all levels
   - Persistence and recovery
   - Performance monitoring

---

**For implementation details, see:**
- `core/agentic_agent.py` - Agentic implementation
- `core/agent_system.py` - Classic implementation
- `core/agent_factory.py` - Agent selection
- `core/prompt_templates.py` - All prompts
- `AGENTIC_REDESIGN.md` - Design rationale
