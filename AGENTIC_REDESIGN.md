# Agentic Pipeline Redesign
## From Rigid Stages to Continuous Autonomous Loop

**Version:** 2.0.0 - Truly Agentic
**Author:** Frederick Gyasi (gyasi@musc.edu)
**Date:** 2025-11-05

---

## Current System (RIGID - v1.0)

```
┌─────────────┐
│  Stage 1:   │  LLM: "I need these 5 tools"
│  Analysis   │  ▼ Requests ALL tools upfront
└─────────────┘

┌─────────────┐
│  Stage 2:   │  System executes ALL 5 tools
│  Tools      │  ▼ Batch execution, no adaptation
└─────────────┘

┌─────────────┐
│  Stage 3:   │  LLM: "Here's extraction with those results"
│  Extract    │  ▼ NO MORE TOOL CALLS POSSIBLE
└─────────────┘

┌─────────────┐
│  Stage 4:   │  LLM: "Here's refined version"
│  Refine     │  ▼ Fixed, no iteration
└─────────────┘

DONE
```

### Problems:
1. **No Discovery**: LLM can't learn it needs more during extraction
2. **No Iteration**: Can't refine RAG queries based on initial results
3. **No Adaptation**: Can't say "that result tells me I need this other tool"
4. **Single-Shot**: Tools called once, never again
5. **Batch Execution**: All tools executed together, can't chain dynamically

---

## New System (AGENTIC - v2.0)

```
START
  │
  ▼
┌────────────────────────────────────┐
│ CONTINUOUS AGENTIC LOOP            │
│                                    │
│  LLM Analyzes Clinical Text        │
│  ↓                                 │
│  LLM Decides: Need tool?           │
│  ├─ YES → PAUSE                    │
│  │    ↓                            │
│  │    Execute Tool                 │
│  │    ↓                            │
│  │    RESUME with results          │
│  │    ↓                            │
│  │    Loop back (can request more) │
│  │                                 │
│  └─ NO → Continue extraction       │
│      ↓                             │
│      Extraction complete?          │
│      ├─ NO → Continue analyzing    │
│      └─ YES → Output JSON          │
└────────────────────────────────────┘
```

### Example Flow:

```
LLM: "Let me analyze this clinical text..."
LLM: "I see malnutrition mentioned. I need RAG on ASPEN criteria."
     → TOOL CALL: RAG("ASPEN malnutrition criteria")
     → PAUSE

[System executes RAG]

     → RESUME with RAG results

LLM: "Interesting. The ASPEN criteria mention z-scores. Let me check the text for z-scores..."
LLM: "I see '3rd percentile' but need to convert to z-score."
     → TOOL CALL: percentile_to_zscore(3)
     → PAUSE

[System executes function]

     → RESUME with z-score = -1.88

LLM: "Now I need to interpret this z-score using WHO/ASPEN standards."
     → TOOL CALL: interpret_zscore_malnutrition(-1.88, "weight-for-height")
     → PAUSE

[System executes function]

     → RESUME with interpretation: "Mild malnutrition risk per WHO"

LLM: "Wait, I should get more specific guidelines on mild malnutrition management."
     → TOOL CALL: RAG("mild malnutrition pediatric management guidelines")
     → PAUSE

[System executes RAG with refined query]

     → RESUME with management guidelines

LLM: "Perfect! Now I have enough information to complete the extraction."
LLM: Generates JSON with all fields properly filled

DONE
```

---

## Key Differences

| Aspect | Old (Rigid) | New (Agentic) |
|--------|-------------|---------------|
| **Tool Planning** | Upfront, all at once | Dynamic, as needed |
| **Tool Calls** | Single batch | Multiple rounds |
| **RAG Queries** | Fixed at Stage 1 | Can refine based on results |
| **Function Calls** | Pre-determined | Discovered during analysis |
| **Adaptation** | None | Full - learns and adapts |
| **Iteration** | Linear stages | Continuous loop |
| **Discovery** | No | Yes - can discover new needs |
| **LLM Autonomy** | Low | High - truly agentic |

---

## Implementation Architecture

### 1. Core Loop Structure

```python
def extract_agentic(clinical_text, label_context):
    """Truly agentic extraction with continuous loop"""

    conversation_history = []
    tools_available = {
        'rag': rag_engine,
        'function': function_registry,
        'extras': extras_manager
    }

    # Initial message to LLM
    initial_prompt = build_agentic_prompt(
        clinical_text=clinical_text,
        label_context=label_context,
        schema=json_schema,
        tools_available=tools_available
    )

    conversation_history.append({
        'role': 'user',
        'content': initial_prompt
    })

    max_iterations = 20  # Safety limit
    iteration = 0
    extraction_complete = False

    while not extraction_complete and iteration < max_iterations:
        iteration += 1

        # LLM generates response (may include tool calls)
        response = llm_manager.generate_with_tools(
            messages=conversation_history,
            tools=get_available_tools_schema(),
            max_tokens=max_tokens
        )

        # Check response type
        if response.has_tool_calls():
            # PAUSE - Execute tools
            tool_results = []
            for tool_call in response.tool_calls:
                result = execute_tool(tool_call)
                tool_results.append(result)

            # Add assistant response to history
            conversation_history.append({
                'role': 'assistant',
                'content': response.content,
                'tool_calls': response.tool_calls
            })

            # RESUME - Add tool results
            conversation_history.append({
                'role': 'tool',
                'tool_call_id': tool_call.id,
                'content': format_tool_result(result)
            })

            # Loop continues - LLM can request more tools or finish

        elif response.has_json_output():
            # Extraction complete
            extraction_complete = True
            json_output = parse_json(response.content)
            return json_output

        else:
            # LLM is thinking/analyzing - add to history and continue
            conversation_history.append({
                'role': 'assistant',
                'content': response.content
            })

            # Ask LLM to continue
            conversation_history.append({
                'role': 'user',
                'content': "Continue with extraction or request tools as needed."
            })

    # If max iterations reached
    raise Exception("Max iterations reached without completion")
```

### 2. Tool Schema for Native Tool Use

```python
def get_available_tools_schema():
    """Return OpenAI/Anthropic-compatible tool schema"""
    return [
        {
            'type': 'function',
            'function': {
                'name': 'query_rag',
                'description': 'Query RAG system for guidelines, standards, and reference information. Can be called MULTIPLE times with different queries to refine information.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {
                            'type': 'string',
                            'description': 'Focused query with 4-8 specific medical keywords targeting guidelines (ASPEN, WHO, CDC), criteria, or standards'
                        },
                        'purpose': {
                            'type': 'string',
                            'description': 'Why you need this information'
                        }
                    },
                    'required': ['query']
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'call_medical_function',
                'description': 'Call a medical calculation function. Available functions: ' + ', '.join(function_registry.get_all_function_names()),
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'function_name': {
                            'type': 'string',
                            'description': 'Name of function to call',
                            'enum': function_registry.get_all_function_names()
                        },
                        'parameters': {
                            'type': 'object',
                            'description': 'Function parameters as key-value pairs'
                        }
                    },
                    'required': ['function_name', 'parameters']
                }
            }
        },
        {
            'type': 'function',
            'function': {
                'name': 'query_extras',
                'description': 'Query supplementary hints/tips/patterns that help understand the task. Use keywords to match relevant hints.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'keywords': {
                            'type': 'array',
                            'items': {'type': 'string'},
                            'description': '3-5 specific medical keywords'
                        }
                    },
                    'required': ['keywords']
                }
            }
        }
    ]
```

### 3. Agentic Prompt Template

```python
AGENTIC_EXTRACTION_PROMPT = """You are a board-certified pediatric dietitian performing clinical information extraction from medical text.

**YOUR TASK:**
Extract structured information from the clinical text to complete the JSON schema below.

**GROUND TRUTH DIAGNOSIS:**
{label_context}

This is definitive. Your extraction must support this diagnosis.

**CLINICAL TEXT:**
{clinical_text}

**EXPECTED OUTPUT SCHEMA:**
{json_schema}

**AVAILABLE TOOLS (call as many times as needed):**

1. **query_rag(query, purpose)**: Retrieve guidelines, standards, criteria from authoritative sources
   - Call MULTIPLE times with different queries
   - Refine queries based on what you learn
   - Example: query_rag("ASPEN malnutrition criteria", "need severity classification")

2. **call_medical_function(function_name, parameters)**: Perform medical calculations
   - Call for z-scores, BMI, percentiles, etc.
   - Can call same function multiple times for serial data
   - Example: call_medical_function("percentile_to_zscore", {"percentile": 3})

3. **query_extras(keywords)**: Get supplementary hints/patterns
   - Helps understand task-specific patterns
   - Example: query_extras(["malnutrition", "pediatric", "z-score"])

**AGENTIC WORKFLOW:**

1. **ANALYZE**: Read the clinical text carefully
2. **DISCOVER**: Identify what information you need
3. **REQUEST TOOLS**: Call tools to get information (can call multiple times!)
4. **LEARN**: Analyze tool results
5. **ITERATE**: If you need more info, call more tools
6. **EXTRACT**: Once you have enough information, output the JSON

**CRITICAL PRINCIPLES:**

- **Iterative**: Don't request all tools at once. Analyze, call tools, learn, call more tools if needed.
- **Adaptive**: Let initial tool results guide what else you need.
- **Thorough**: Get enough information before finalizing extraction.
- **Efficient**: Don't call unnecessary tools, but don't hesitate to call multiple times.

**EXAMPLE WORKFLOW:**

"I see 'malnutrition' mentioned. Let me get ASPEN criteria."
→ query_rag("ASPEN pediatric malnutrition criteria")
[Receives results]
"The criteria mention z-scores. I see '3rd percentile' in text. Let me convert."
→ call_medical_function("percentile_to_zscore", {"percentile": 3})
[Receives z = -1.88]
"Now I need to interpret this z-score using WHO standards."
→ call_medical_function("interpret_zscore_malnutrition", {"zscore": -1.88, "measurement_type": "weight-for-height"})
[Receives interpretation]
"Perfect! Now I have enough to extract."
→ Output JSON

**Begin your analysis and call tools as needed. When ready, output the final JSON.**
"""
```

---

## Benefits of New Architecture

### 1. True Autonomy
- LLM decides what it needs when it needs it
- No forced upfront planning
- Natural discovery process

### 2. Iterative Refinement
- Initial RAG query → Learn → Refined RAG query
- Can correct course based on results
- Adaptive strategy

### 3. Context-Aware Tool Use
- "I got this result, so now I need that tool"
- Chained reasoning
- Dynamic dependencies

### 4. Multiple Tool Calls
- Same tool called multiple times
- Different queries based on learning
- Serial data properly handled

### 5. Natural Reasoning
- Mimics human clinical expert thought process
- "Let me check the criteria... now let me calculate... now let me verify..."
- Transparent decision-making

---

## Implementation Plan

### Phase 1: Core Agentic Loop (Week 1)
- [ ] Implement continuous loop with pause/resume
- [ ] Native tool calling integration
- [ ] Conversation history management
- [ ] Tool execution with resume

### Phase 2: Tool Schema (Week 1)
- [ ] Convert tools to native function calling schema
- [ ] RAG as function
- [ ] Functions as function calls
- [ ] Extras as function

### Phase 3: Prompt Engineering (Week 2)
- [ ] Design agentic prompt
- [ ] Add reasoning guidance
- [ ] Test iteration behavior
- [ ] Optimize for efficiency

### Phase 4: Safety & Limits (Week 2)
- [ ] Max iteration limits
- [ ] Tool call budgets
- [ ] Infinite loop detection
- [ ] Graceful degradation

### Phase 5: Testing & Validation (Week 3)
- [ ] Test with malnutrition cases
- [ ] Test with diabetes cases
- [ ] Compare v1.0 vs v2.0 outputs
- [ ] Benchmark performance

### Phase 6: Migration (Week 4)
- [ ] Backward compatibility mode
- [ ] Gradual rollout
- [ ] Documentation
- [ ] User training

---

## Next Steps

1. **Prototype Core Loop** - Build minimal viable agentic loop
2. **Test with One Case** - Malnutrition case end-to-end
3. **Iterate on Prompts** - Refine agentic behavior
4. **Full Implementation** - Complete system redesign

---

## Questions to Resolve

1. **Max Iterations**: What's reasonable limit? (Suggest: 20)
2. **Tool Call Limit**: Max tools per iteration? (Suggest: 5)
3. **Backward Compatibility**: Keep old system as fallback?
4. **API Compatibility**: OpenAI vs Anthropic tool calling?

---

## Success Metrics

- **Autonomy**: LLM makes intelligent tool choices
- **Iteration**: Successfully refines queries based on results
- **Efficiency**: Doesn't call unnecessary tools
- **Completeness**: Gets all needed information
- **Accuracy**: Extractions are more accurate than v1.0

---

*This represents a fundamental shift from staged pipeline to truly agentic AI system.*
