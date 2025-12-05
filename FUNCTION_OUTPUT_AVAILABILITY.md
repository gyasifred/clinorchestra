# Function Output Availability for Query Enhancement

## Question Answered

**Can function outputs be MADE AVAILABLE to the LLM so it can autonomously CHOOSE to use them for better RAG/extras queries?**

## Answer: ✅ YES - Already Supported in Both Pipelines

Function outputs ARE currently made available to the LLM, and the LLM can autonomously decide whether to use them for better queries. This is NOT enforced - the LLM makes the decision based on what it deems useful.

---

## ADAPTIVE Pipeline - Full Support ✅

### How Function Outputs Are Made Available

**Location:** `core/agentic_agent.py:384-394`

```python
# Add tool result messages
for result in tool_results:
    self.context.conversation_history.append(
        ConversationMessage(
            role='tool',
            tool_call_id=result.tool_call_id,
            name=result.type,
            content=self._format_tool_result_for_llm(result)
        )
    )
```

### Behavior

1. **Tool Results in Conversation**: All tool results (functions, RAG, extras) are added to conversation history as 'tool' role messages
2. **Visible Across Iterations**: LLM sees ALL previous tool results in every subsequent iteration
3. **Autonomous Decision Making**: LLM can choose to use function outputs to generate better RAG/extras queries in next iteration

### Example Workflow

```
Iteration 1:
  LLM calls: calculate_bmi(weight=20, height=1.1)
  Result: {"value": 16.5, "unit": "kg/m²"}
  ↓ Added to conversation history as tool message

Iteration 2:
  LLM sees: Previous tool results including BMI=16.5
  LLM autonomously decides: "This BMI indicates severe underweight"
  LLM chooses to call: query_rag("BMI 16.5 severe underweight malnutrition criteria")
  ↓ LLM made this decision - not enforced by system
```

### Current Implementation Status

- ✅ Function outputs available in conversation history
- ✅ LLM can see all previous tool results
- ✅ LLM can autonomously use outputs for better queries
- ✅ No enforcement - LLM decides what's useful

---

## STRUCTURED Pipeline - Full Support ✅

### Stage 3: Extraction - Function Outputs Available

**Location:** `core/agent_system.py:1907-1964`

```python
# Format tool results
tool_outputs = format_tool_outputs_for_prompt(self.context.tool_results)  # Line 1915

# Include in prompt
extraction_prompt = extraction_prompt.format(
    clinical_text=self.context.clinical_text,
    function_outputs=tool_outputs.get('function_outputs', ''),  # Line 1933
    # ...
)

# Default layout also includes function outputs
{tool_outputs.get('function_outputs', '')}  # Line 1955
```

**Behavior:**
- All function outputs from Stage 2 are formatted and included in Stage 3 extraction prompt
- LLM can see what functions were called and their results

### Stage 4: Refinement - Function Outputs Available + Tool Calling

**Location:** `core/agent_system.py:1310-1429`

#### Phase 1: Gap Analysis (Lines 1310-1429)

```python
# Format tool outputs from Stage 2
tool_outputs = format_tool_outputs_for_prompt(self.context.tool_results)  # Line 1313

# Build prompt with function outputs
prompt = f"""...
TOOLS ALREADY EXECUTED (Stage 2):
{tool_outputs}  # Lines 1383-1384

RETRIEVED EVIDENCE (Stage 2 RAG):
{rag_context}  # Lines 1386-1387

YOUR TASK:
1. ANALYZE CURRENT EXTRACTION:
   - Are there missing calculations or values?
   - Are there uncertainties that need more data?
   - Could additional guidelines help interpretation?

2. IDENTIFY GAPS:
   - Missing function calls (calculations not performed)
   - Missing RAG queries (guidelines not retrieved)
   - Missing context (extras hints needed)

3. REQUEST ADDITIONAL TOOLS IF NEEDED:
   You have access to:
   FUNCTIONS: [list of available functions]
   RAG: Retrieve additional clinical guidelines/standards
   EXTRAS: Get supplementary hints/patterns

RESPONSE FORMAT:
{{
  "additional_tools_needed": [
    {{
      "type": "rag",
      "query": "specific focused query with NEW keywords",  # Line 1428
      "reason": "Why this query helps (e.g., 'BMI result 16.5 needs malnutrition staging criteria')"
    }}
  ]
}}
"""
```

**Behavior:**
- LLM sees ALL function outputs from Stage 2
- LLM sees previous RAG queries to avoid repetition (Lines 1334-1373)
- LLM can autonomously request NEW RAG queries based on function results
- LLM provides reasoning for why the query is helpful

#### Phase 2: Final Refinement (Lines 1112-1150)

- Executes any additional tools requested in Phase 1
- Combines original RAG results + new tool results
- LLM refines extraction with complete information

### Example Workflow

```
Stage 1: Planning
  LLM plans: calculate_creatinine_clearance(creatinine=2.5, age=70, weight=80, sex="M")
  LLM plans: query_rag("chronic kidney disease staging")
  ↓ Generic query - doesn't know result yet

Stage 2: Execution (Parallel)
  Function executes: eGFR = 28 mL/min/1.73m²
  RAG executes: Returns general CKD staging info
  ↓ Results stored in tool_results

Stage 3: Extraction
  LLM sees: "eGFR = 28 mL/min/1.73m²"
  LLM extracts values using function results

Stage 4 Phase 1: Gap Analysis
  LLM sees: "TOOLS ALREADY EXECUTED: calculate_creatinine_clearance → 28"
  LLM autonomously decides: "eGFR of 28 indicates moderate-severe CKD"
  LLM chooses to request: query_rag("eGFR 28 CKD stage 3b management guidelines")
  ↓ LLM made this decision - not enforced

Stage 4 Phase 2: Refinement
  Executes new RAG query with better keywords
  Returns more specific CKD stage 3b guidelines
  LLM refines extraction with targeted information
```

### Current Implementation Status

- ✅ Function outputs available in Stage 3 extraction
- ✅ Function outputs available in Stage 4 gap analysis
- ✅ LLM can see previous RAG queries to avoid repetition
- ✅ LLM can autonomously request new RAG queries based on function results
- ✅ LLM provides reasoning for why query is helpful
- ✅ No enforcement - LLM decides what's useful

---

## Key Code Locations

### ADAPTIVE Pipeline
- **Tool results added to conversation**: `core/agentic_agent.py:384-394`
- **Tool result formatting**: `core/agentic_agent.py:392` (`_format_tool_result_for_llm`)
- **Previous tool tracking**: `core/agentic_agent.py:760-791`
- **Keyword variation guidance**: `core/agentic_agent.py:808-859`

### STRUCTURED Pipeline
- **Stage 3 extraction with function outputs**: `core/agent_system.py:1907-1964`
  - Tool output formatting: Line 1915
  - Function outputs in prompt: Lines 1933, 1955
- **Stage 4 gap analysis with function outputs**: `core/agent_system.py:1310-1429`
  - Tool output formatting: Line 1313
  - Function outputs in prompt: Lines 1383-1384
  - Previous tool tracking: Lines 1339-1373
  - RAG query requests: Lines 1428-1429
- **Stage 4 tool execution**: `core/agent_system.py:1097-1106`

---

## Summary

| Feature | ADAPTIVE | STRUCTURED |
|---------|----------|------------|
| Function outputs available to LLM | ✅ Yes (all iterations) | ✅ Yes (Stages 3 & 4) |
| LLM can see function results | ✅ Yes (conversation history) | ✅ Yes (tool_outputs section) |
| LLM can use results for better queries | ✅ Yes (autonomous decision) | ✅ Yes (Stage 4 gap analysis) |
| Enforcement of query enhancement | ❌ No (LLM decides) | ❌ No (LLM decides) |
| LLM provides reasoning | ⚠️ Implicit | ✅ Yes (required in response) |
| Previous query tracking | ✅ Yes | ✅ Yes |
| Keyword deduplication | ✅ Yes | ✅ Yes |

---

## Conclusion

**Both pipelines already support the requested capability:**

1. ✅ Function outputs ARE made available to the LLM
2. ✅ LLM can SEE the results autonomously
3. ✅ LLM can CHOOSE to use them for better RAG/extras queries
4. ✅ This is NOT enforced - LLM makes the decision based on what it deems useful
5. ✅ Previous queries are tracked to avoid repetition

**No code changes needed** - the system already works as requested. The LLM has full visibility into function outputs and can autonomously decide whether to use them for generating better queries.

---

Date: 2025-12-05
Author: Claude
