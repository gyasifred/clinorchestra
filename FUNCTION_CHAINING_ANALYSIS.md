# Function Chaining & Query Enhancement Analysis

## Investigation Results

### 1. Function Chaining (Function Output → Function Input)

#### ADAPTIVE Pipeline ✅ SUPPORTED
**Location:** `core/agentic_agent.py`

**Features:**
- **Dependency Detection** (Lines 895-945): `_detect_and_resolve_dependencies()`
  - Scans parameters for `$call_X` references
  - Automatically detects dependencies between function calls
  - Populates `depends_on` field for each tool call

- **Dependency Resolution** (Lines 947-993): `_resolve_dependencies_topological_sort()`
  - Sorts tool calls in dependency order
  - Ensures dependencies execute before dependent calls
  - Detects circular dependencies

- **Dependency Substitution** (Lines 995-1038): `_substitute_dependencies()`
  - Replaces `$call_X` with actual results from executed calls
  - Supports field access: `$call_X.field_name`
  - Handles nested dependencies

**Example from Documentation (Lines 1587-1618):**
```python
# Generic chained calculations
TOOL_CALL: {"id": "call_1", "tool": "call_function_a", "parameters": {"param": "value1"}}
TOOL_CALL: {"id": "call_2", "tool": "call_function_b", "parameters": {"param": "value2"}}
TOOL_CALL: {"id": "call_3", "tool": "call_function_c", "parameters": {"param_a": "$call_1", "param_b": "$call_2"}}
```

**How it works:**
1. LLM specifies dependencies using `$call_X` syntax
2. System detects dependencies automatically
3. Executes in correct order (call_1, call_2, then call_3)
4. Substitutes results before executing dependent calls

#### STRUCTURED Pipeline ❌ NOT SUPPORTED
**Location:** `core/agent_system.py`

**Current Behavior:**
- **Stage 1**: LLM plans all tool calls upfront
- **Stage 2**: ALL tools execute in PARALLEL (async) - Lines 679-759
- **No dependency tracking**: Functions cannot reference each other's outputs
- **No execution ordering**: All function calls run simultaneously

**Limitation:**
Cannot chain functions. For example:
```json
// CANNOT DO THIS:
{
  "functions_needed": [
    {"name": "convert_cm_to_m", "parameters": {"cm": 165}},
    {"name": "calculate_bmi", "parameters": {"height_m": "$call_1"}}  // ❌ NOT SUPPORTED
  ]
}
```

**Current Workaround:**
- LLM must extract all parameters from text
- Cannot use one function's output as another's input
- Must manually calculate intermediate values

---

### 2. Function Outputs → RAG/Extras Queries

#### ADAPTIVE Pipeline ⚠️ PARTIAL SUPPORT
**Current Behavior:**
- Tool results stored in `context.tool_results` (Line 94)
- LLM can SEE all previous tool results in conversation (Lines 760-826)
- LLM can manually use results to formulate new queries in next iteration

**Example:**
```
Iteration 1:
  - LLM calls: calculate_bmi(weight=20, height=1.1) → Result: 16.5

Iteration 2:
  - LLM sees BMI=16.5 in previous results
  - LLM can manually call: query_rag("BMI 16.5 malnutrition criteria")
```

**Limitation:**
- NOT AUTOMATIC - LLM must manually decide to use results
- Requires extra iteration (slower)
- LLM may not think to use results for better queries

#### STRUCTURED Pipeline ❌ NOT SUPPORTED
**Current Behavior:**
- **Stage 1**: LLM plans all tools BEFORE seeing any results
- **Stage 2**: Tools execute in parallel
- **Stage 3**: Extraction uses tool results
- **Stage 4**: Optional RAG refinement with fixed queries

**Critical Issue:**
- Stage 1 planning happens WITHOUT function results
- Cannot use function outputs to improve RAG/extras queries
- All queries must be planned upfront before calculations

**Example of What CANNOT Happen:**
```
Stage 1: Plan
  - Function: calculate_bmi(weight=20, height=1.1)
  - RAG: "BMI classification criteria"  // Generic, doesn't know result yet

Stage 2: Execute
  - BMI result: 16.5 (severe malnutrition range)

// ❌ CANNOT DO: Generate better query based on result
// ❌ CANNOT DO: query_rag("BMI 16.5 malnutrition severe classification")
```

---

## Recommendations

### Priority 1: Add Function Chaining to STRUCTURED Pipeline
**Impact:** HIGH
**Difficulty:** MEDIUM

**Implementation:**
1. Adapt dependency detection from ADAPTIVE pipeline
2. Add topological sort for execution ordering
3. Execute functions sequentially when dependencies exist
4. Keep parallel execution for independent functions

**Benefits:**
- Can chain calculations (cm→m→BMI)
- Can use one function's output as another's input
- More powerful and flexible

### Priority 2: Auto-Enhance Queries with Function Results (Both Pipelines)
**Impact:** HIGH
**Difficulty:** HIGH

**For ADAPTIVE Pipeline:**
- Add automatic query enhancement in iteration analysis
- When function returns a value, auto-generate refined RAG query
- Example: BMI=16.5 → auto-query "BMI 16.5 classification criteria"

**For STRUCTURED Pipeline:**
- Add Stage 2.5: Query Enhancement
  - After function execution
  - Before extraction
  - Generate enhanced RAG queries based on function results
  - Execute enhanced queries

**Benefits:**
- Better quality RAG results
- More relevant guideline retrieval
- Contextual search based on actual values

### Priority 3: Smart Query Generation from Results
**Impact:** MEDIUM
**Difficulty:** MEDIUM

**Implementation:**
- Monitor function results for key values
- Auto-generate relevant follow-up queries
- Use result magnitude/range to select query templates

**Example Logic:**
```python
if function_name == "calculate_bmi" and result < 18.5:
    auto_query = f"BMI {result} underweight malnutrition criteria"
elif result > 25:
    auto_query = f"BMI {result} overweight obesity guidelines"
```

---

## Current Capabilities Summary

| Feature | ADAPTIVE | STRUCTURED |
|---------|----------|------------|
| Function Chaining | ✅ Full Support | ❌ Not Supported |
| Dependency Detection | ✅ Automatic | ❌ None |
| Function Output → Function Input | ✅ Yes | ❌ No |
| Function Results Visible to LLM | ✅ Yes | ✅ Yes (Stage 3+) |
| Auto-Enhance RAG Queries | ❌ Manual Only | ❌ No |
| Function Results → Better Queries | ⚠️ Manual (extra iteration) | ❌ No |

---

## Examples of What's Missing

### Example 1: Height Conversion + BMI Calculation
**Current STRUCTURED Limitation:**
```json
// ❌ CANNOT chain:
{
  "functions_needed": [
    {"name": "cm_to_m", "parameters": {"cm": 165}},
    {"name": "calculate_bmi", "parameters": {"height_m": "$call_1"}}  // ❌ ERROR
  ]
}
```

**Workaround:**
- LLM must manually calculate: 165cm = 1.65m
- Or wait for ADAPTIVE mode's iterative approach

### Example 2: Calculate Then Query Guidelines
**Current STRUCTURED Limitation:**
```json
// Stage 1 planning (before calculation):
{
  "functions_needed": [{"name": "calculate_creatinine_clearance", ...}],
  "rag_queries": [{"query": "CKD staging criteria"}]  // Generic
}

// ❌ CANNOT DO after getting result (eGFR=35):
// Add better query: "eGFR 35 CKD stage 3 classification"
```

**What Should Happen:**
- Calculate eGFR = 35
- Auto-generate enhanced query: "eGFR 35 mL/min CKD stage classification"
- Retrieve more relevant guidelines

---

## Next Steps

1. **User Decision Needed:**
   - Implement function chaining in STRUCTURED? (High priority)
   - Implement auto query enhancement? (High priority)
   - Which pipeline to focus on first?

2. **If Approved:**
   - Implement dependency tracking for STRUCTURED
   - Add query enhancement logic
   - Test both features
   - Document new capabilities

3. **Testing Scenarios:**
   - Multi-step calculations (unit conversions)
   - Calculate → query → refine workflow
   - Verify performance impact

---

Date: 2025-12-05
Author: Claude
