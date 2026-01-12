# ClinOrchestra: A Neuro-Symbolic AI Platform for Clinical NLP Annotation

## Project Name: ClinOrchestra

**Lead/Mentor:** Frederick Gyasi (gyasi@musc.edu)
**Institution:** Medical University of South Carolina, Biomedical Informatics Center
**Version:** 1.0.0

---

## Brief Summary

ClinOrchestra is a **Neuro-Symbolic AI Platform** designed for clinical natural language processing (NLP) annotation tasks. The platform uniquely combines the reasoning capabilities of large language models (LLMs) with deterministic symbolic computation and knowledge retrieval systems, implementing a novel **Neuro-Symbolic Feedback Loop** architecture.

The core innovation lies in augmenting LLM-based extraction with three complementary knowledge sources:
1. **Custom Functions** - Deterministic computations (clinical calculations, scoring systems)
2. **RAG (Retrieval Augmented Generation)** - Domain-specific document retrieval
3. **Extras** - Task-specific hints and domain knowledge patterns

---

## Important Research Questions

1. **Can neuro-symbolic integration improve clinical NLP annotation accuracy** compared to pure LLM approaches by grounding neural reasoning in symbolic domain knowledge?

2. **How does the Neuro-Symbolic Feedback Loop** (Functions + RAG + Extras) enhance LLM performance on complex clinical extraction tasks requiring domain expertise?

3. **What is the optimal balance** between neural (LLM reasoning) and symbolic (deterministic computation) components for different clinical annotation tasks?

---

## Core Architecture: The Neuro-Symbolic Feedback Loop

### Architectural Philosophy

ClinOrchestra implements a **hybrid neuro-symbolic architecture** where:
- **Neural Component**: LLMs (Claude, GPT-4, Llama) handle natural language understanding, context interpretation, and synthesis
- **Symbolic Component**: Functions, RAG, and Extras provide deterministic computation, factual grounding, and domain expertise

The feedback loop ensures that LLM outputs are informed by and validated against symbolic knowledge sources.

### Two Execution Modes

| Mode | Implementation | Characteristics | Best Use Case |
|------|---------------|-----------------|---------------|
| **STRUCTURED** | `core/agent_system.py` | 4-stage deterministic pipeline, predictable, reproducible | Production annotation |
| **ADAPTIVE** | `core/agentic_agent.py` | ReAct-style iterative agent, self-directed tool selection | Exploratory tasks |

---

## STRUCTURED Mode: 4-Stage Pipeline

The STRUCTURED mode (`core/agent_system.py:61-1200`) implements a deterministic 4-stage extraction pipeline:

### Stage 1: Task Understanding (Lines 373-513)
**Purpose:** LLM analyzes input text + output schema to plan tool usage

```
Input Text + Schema → LLM → Task Understanding Plan
                           ├── functions_needed[]
                           ├── rag_queries[]
                           └── extras_keywords[]
```

**Key Implementation:**
- `_execute_stage1_understanding()` (line 373)
- `_convert_task_understanding_to_tool_requests()` (line 514)
- LLM determines which functions to call, what RAG queries to execute, and which extras keywords to match

### Stage 2: Tool Execution (Lines 699-820)
**Purpose:** Execute all planned tool requests with parallel async processing

```
Tool Requests → Parallel Execution → Tool Results
               ├── Functions (deterministic)
               ├── RAG queries (vector search)
               └── Extras matching (keyword-based)
```

**Key Features:**
- **Parallel Async Execution** (`_execute_stage2_tools_async()`, line 784): 60-75% performance improvement
- **Tool Deduplication** (`_deduplicate_tool_requests()`, line 550): Prevents repeated identical calls
- **Parameter Validation** (`_validate_tool_parameters()`, line 597): Universal validation for any function
- **Function Chaining** (`_detect_tool_dependencies()`, line 821): Supports `$call_X` syntax for referencing outputs
- **Topological Sort** (`_topological_sort_tools()`, line 876): Correct execution order for dependencies

### Stage 3: Synthesis (Lines 1050-1150)
**Purpose:** LLM combines tool results with clinical understanding to produce structured output

```
Tool Results + Original Text + Schema → LLM → Structured JSON Output
```

**Key Implementation:**
- `_execute_stage3_synthesis()`
- Constructs enhanced prompt with all tool results
- LLM synthesizes final extraction following schema

### Stage 4: Validation (Lines 1150-1250)
**Purpose:** Validate output against schema with optional self-critique

```
JSON Output → Schema Validation → Optional Self-Critique → Final Output
```

**Key Features:**
- Pydantic-based schema validation
- Optional iterative self-critique for quality improvement
- Error recovery and re-extraction attempts

---

## The Three Knowledge Sources (Symbolic Components)

### 1. Custom Functions (`core/function_registry.py`)

**Purpose:** Register deterministic Python functions for clinical calculations

**Architecture (Lines 1-350):**

```python
class FunctionRegistry:
    def register_function(name, callable, parameters, description)
    def execute_function(name, parameters) -> result
    def get_function_info(name) -> function_definition
```

**Key Features:**
- **Parameter Validation** (line 180-260): Type checking, required/optional parameters
- **Parameter Transformation** (line 89-150): Maps schema fields to function parameters
- **Flexible Registration**: Any Python callable can be registered

**Example Function Registration:**
```python
registry.register_function(
    name="creatinine_clearance",
    callable=calculate_creatinine_clearance,
    parameters={
        "age": {"type": "number", "required": True},
        "weight": {"type": "number", "required": True},
        "creatinine": {"type": "number", "required": True},
        "sex": {"type": "string", "required": True}
    },
    description="Calculate creatinine clearance using Cockcroft-Gault"
)
```

**Use Cases:**
- Clinical scoring systems (SOFA, APACHE, Glasgow Coma Scale)
- Pharmacokinetic calculations (creatinine clearance, drug dosing)
- Risk calculators (cardiovascular risk, mortality prediction)
- Unit conversions and normalization

### 2. RAG Engine (`core/rag_engine.py`)

**Purpose:** Retrieval Augmented Generation for domain-specific knowledge retrieval

**Architecture (Lines 1-926):**

```
Documents → Chunking → Embedding → FAISS Index → Vector Search → Relevant Chunks
```

**Core Components:**

| Component | Class | Purpose |
|-----------|-------|---------|
| Document Loading | `DocumentLoader` (line 50-230) | Load from URLs, PDFs, HTML, TXT files |
| Chunking | `DocumentChunker` (line 322-358) | Sliding window chunking (default 512 chars, 50 overlap) |
| Embedding | `EmbeddingGenerator` (line 232-320) | SentenceTransformer embeddings with caching |
| Vector Store | `VectorStore` (line 360-646) | FAISS with cosine similarity, GPU acceleration |
| RAG Engine | `RAGEngine` (line 647-926) | Orchestrates retrieval pipeline |

**Key Features:**

1. **Batch Embedding Generation** (line 256-316): 25-40% faster with configurable batch size
2. **Query with Variations** (line 841-901): Improved recall through term expansion
   ```python
   results = engine.query_with_variations(
       primary_query="diagnostic criteria",
       variations=["diagnosis", "clinical criteria", "assessment"]
   )
   ```
3. **GPU Acceleration** (line 375-462): FAISS GPU mode for 10-90x faster searches
4. **Persistent Caching** (line 473-590): SQLite-based embedding and document cache
5. **Query Result Caching** (line 804-837): Avoid redundant vector searches

**Supported Document Formats:**
- PDF (with encryption handling)
- HTML (with script/style stripping)
- Plain text and Markdown
- Remote URLs with automatic caching

### 3. Extras Manager (`core/extras_manager.py`)

**Purpose:** Task-specific supplementary hints to help LLM understand domain patterns

**Architecture (Lines 1-643):**

```python
class ExtrasManager:
    def add_extra(type, content, metadata, name)
    def match_extras_by_keywords(keywords) -> matched_extras
    def match_extras_with_variations(core_keywords, variations_map)
```

**Critical Design Principle (Lines 20-33):**
> Extras are NOT for matching against input text - they are supplementary knowledge that helps the LLM better understand how to approach the extraction task.

**Matching Algorithm (Lines 105-170):**
```
Schema Fields → Keywords → Keyword Matching → Relevance Scoring → Top Extras
```

**Relevance Scoring (Lines 301-353):**
- Exact keyword match in content: +1.0
- Partial keyword match: +0.5
- Keyword in type field: +0.3
- Keyword in metadata: +0.2
- Multi-keyword bonus: ×1.2

**Key Features:**

1. **Keyword-Based Matching** (line 105): Matches against schema fields, NOT input text
2. **Variations Support** (line 172-220): Term expansion for better recall
3. **Enable/Disable** (line 463-483): Toggle individual extras
4. **Import/Export** (line 543-593): YAML/JSON support for extras management
5. **Result Caching** (line 127-138): Avoid redundant matching computations

**Example Extras:**
```yaml
- type: "diagnostic_criteria"
  content: "Assessment requires presence of at least 3 of 5 criteria..."

- type: "assessment_methodology"
  content: "When evaluating severity, consider temporal progression..."

- type: "domain_pattern"
  content: "In clinical notes, abbreviations commonly used include..."
```

---

## ADAPTIVE Mode: ReAct-Style Agent (`core/agentic_agent.py`)

**Purpose:** Self-directed iterative agent for exploratory annotation tasks

**Architecture (Lines 1-800):**

```
Observe → Think → Act → Observe → Think → Act → ... → Final Output
```

**Key Features:**
- **ReAct Pattern**: Iterative reasoning with tool selection
- **Self-Directed**: Agent decides which tools to call
- **Multiple Iterations**: Continues until extraction complete or max iterations
- **Dynamic Tool Selection**: Chooses between Functions, RAG, Extras based on need

**Configuration:**
```python
AgenticExtractionAgent(
    max_iterations=10,
    enable_self_critique=True,
    think_aloud=True
)
```

**Use Cases:**
- Complex extraction requiring exploration
- Tasks with unclear requirements
- Research and development

---

## Application State Management (`core/app_state.py`)

**Purpose:** Centralized state management for the annotation session

**Key Components (Lines 1-200):**
- `AppState`: Holds all session configuration
- `ExtractionContext`: Per-extraction state (tool requests, results)
- `OptimizationConfig`: GPU settings, caching, performance tuning

---

## Prompt Engineering (`core/prompt_templates.py`)

**Purpose:** Structured prompts for each pipeline stage

**Templates:**
- `STAGE1_TASK_UNDERSTANDING_PROMPT`: Plan tool usage
- `STAGE2_TOOL_EXECUTION_PROMPT`: Execute tools (internal)
- `STAGE3_SYNTHESIS_PROMPT`: Combine results into output
- `STAGE4_VALIDATION_PROMPT`: Self-critique and validation

---

## UI Components (`ui/`)

### Main Application (`ui/main.py`)
- Streamlit-based web interface
- Text input area for clinical notes
- Configuration panels for schema, functions, RAG, extras
- Results visualization with confidence scores

### Shared Components (`ui/shared_ui.py`)
- Reusable UI elements
- Status indicators
- RAG initialization monitoring

---

## Example Usage Patterns

### SDK Integration (from `examples/`)

**Basic Configuration:**
```python
from core.agent_system import StructuredExtractionAgent
from core.function_registry import FunctionRegistry
from core.rag_engine import RAGEngine
from core.extras_manager import ExtrasManager

# 1. Define output schema
schema = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string"},
        "severity": {"type": "string", "enum": ["mild", "moderate", "severe"]},
        "confidence": {"type": "number"}
    }
}

# 2. Register custom functions
registry = FunctionRegistry()
registry.register_function("calculate_score", score_function, params, desc)

# 3. Initialize RAG with domain documents
rag_config = {"sources": ["guidelines.pdf", "criteria.html"]}
rag_engine = RAGEngine(rag_config)

# 4. Load task-specific extras
extras_manager = ExtrasManager("./extras")
extras_manager.add_extra("hint", "Look for specific criteria...", {})

# 5. Create agent and extract
agent = StructuredExtractionAgent(
    function_registry=registry,
    rag_engine=rag_engine,
    extras_manager=extras_manager,
    schema=schema
)

result = agent.extract(clinical_text)
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ClinOrchestra Data Flow                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Input Text ─────────────────────────┐                                     │
│                                       │                                     │
│   Output Schema ──────────────────────┼───► STAGE 1: Task Understanding     │
│                                       │          (LLM Planning)             │
│   Task Labels ────────────────────────┘              │                      │
│                                                      │                      │
│                                                      ▼                      │
│                                        ┌─────────────────────────┐          │
│                                        │    Tool Request Plan    │          │
│                                        │  ├── functions_needed   │          │
│                                        │  ├── rag_queries        │          │
│                                        │  └── extras_keywords    │          │
│                                        └─────────────────────────┘          │
│                                                      │                      │
│                                                      ▼                      │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │                    STAGE 2: Tool Execution (Parallel)              │    │
│   │                                                                    │    │
│   │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐              │    │
│   │  │  Functions  │   │     RAG     │   │   Extras    │              │    │
│   │  │ (Symbolic)  │   │  (Hybrid)   │   │ (Symbolic)  │              │    │
│   │  │             │   │             │   │             │              │    │
│   │  │ Deterministic│   │Vector Search│   │  Keyword   │              │    │
│   │  │ Computation │   │ + Retrieval │   │  Matching   │              │    │
│   │  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘              │    │
│   │         │                 │                 │                      │    │
│   │         └─────────────────┴─────────────────┘                      │    │
│   │                           │                                        │    │
│   └───────────────────────────┼────────────────────────────────────────┘    │
│                               │                                              │
│                               ▼                                              │
│                    ┌─────────────────────────┐                              │
│                    │      Tool Results       │                              │
│                    │  ├── function_outputs   │                              │
│                    │  ├── rag_chunks         │                              │
│                    │  └── matched_extras     │                              │
│                    └─────────────────────────┘                              │
│                               │                                              │
│                               ▼                                              │
│                    STAGE 3: Synthesis (LLM)                                  │
│                    Tool Results + Text + Schema                              │
│                               │                                              │
│                               ▼                                              │
│                    ┌─────────────────────────┐                              │
│                    │   Structured Output     │                              │
│                    │      (JSON)             │                              │
│                    └─────────────────────────┘                              │
│                               │                                              │
│                               ▼                                              │
│                    STAGE 4: Validation                                       │
│                    Schema Check + Self-Critique                              │
│                               │                                              │
│                               ▼                                              │
│                    ┌─────────────────────────┐                              │
│                    │     Final Output        │                              │
│                    │   (Validated JSON)      │                              │
│                    └─────────────────────────┘                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Neuro-Symbolic Feedback Loop: Technical Details

### How the Loop Works

The neuro-symbolic feedback loop operates through structured interaction between neural (LLM) and symbolic (Functions/RAG/Extras) components:

```
┌──────────────────────────────────────────────────────────────────┐
│                  NEURO-SYMBOLIC FEEDBACK LOOP                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│     ┌─────────────┐                    ┌─────────────────┐       │
│     │             │    Tool Requests   │                 │       │
│     │    LLM      │ ─────────────────► │    Symbolic     │       │
│     │  (Neural)   │                    │   Components    │       │
│     │             │ ◄───────────────── │                 │       │
│     └─────────────┘    Tool Results    └─────────────────┘       │
│           │                                    │                  │
│           │ Stage 1: What tools do I need?     │                  │
│           │ Stage 3: How do I interpret        │                  │
│           │          these results?            │                  │
│           │                                    │                  │
│           │         ┌─────────────────────┐    │                  │
│           │         │   Functions:        │    │                  │
│           │         │   - Calculations    │    │                  │
│           │         │   - Scoring systems │    │                  │
│           └────────►│                     │◄───┘                  │
│                     │   RAG:              │                       │
│                     │   - Guidelines      │                       │
│                     │   - Criteria        │                       │
│                     │                     │                       │
│                     │   Extras:           │                       │
│                     │   - Domain hints    │                       │
│                     │   - Patterns        │                       │
│                     └─────────────────────┘                       │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Benefits of the Approach

1. **Grounded Reasoning**: LLM outputs are informed by factual, retrievable knowledge
2. **Deterministic Computation**: Clinical calculations are exact, not approximated
3. **Domain Expertise**: Task-specific hints guide LLM interpretation
4. **Transparency**: Tool calls and results are logged and traceable
5. **Reproducibility**: Same inputs produce same outputs (STRUCTURED mode)

---

## Performance Optimizations

### Implemented Optimizations (v1.0.0)

| Optimization | Location | Improvement |
|-------------|----------|-------------|
| Parallel Tool Execution | `agent_system.py:784` | 60-75% faster |
| Batch Embedding | `rag_engine.py:256` | 25-40% faster |
| Query Result Caching | `rag_engine.py:804` | Avoid redundant searches |
| Embedding Caching | `rag_engine.py:277` | In-memory cache |
| Document Caching | `rag_engine.py:59` | SQLite persistent cache |
| Extras Result Caching | `extras_manager.py:127` | Avoid redundant matching |
| GPU FAISS Acceleration | `rag_engine.py:375` | 10-90x faster vector search |
| Tool Deduplication | `agent_system.py:550` | Prevent repeated calls |

### Enabling GPU Acceleration

```bash
# Environment variable
export CLINORCHESTRA_ENABLE_GPU=1

# Or in config
rag_config = {
    "use_gpu_faiss": True,
    "gpu_device": 0  # Which GPU to use
}
```

---

## File Structure

```
clinorchestra/
├── core/                          # Core architecture
│   ├── agent_system.py            # STRUCTURED mode (4-stage pipeline)
│   ├── agentic_agent.py           # ADAPTIVE mode (ReAct agent)
│   ├── function_registry.py       # Custom function management
│   ├── rag_engine.py              # RAG with FAISS vector store
│   ├── extras_manager.py          # Task-specific hints
│   ├── app_state.py               # Session state management
│   ├── prompt_templates.py        # LLM prompt engineering
│   └── logging_config.py          # Logging configuration
│
├── ui/                            # Streamlit UI
│   ├── main.py                    # Main application
│   └── shared_ui.py               # Shared components
│
├── examples/                      # Example configurations
│   ├── adrd_classification_sdk.py # Example: Dementia classification
│   └── malnutrition_classification_sdk.py # Example: Malnutrition
│
├── annotate.py                    # CLI annotation entry point
├── requirements.txt               # Python dependencies
└── README.md                      # Quick start guide
```

---

## Resources

**Code Repository:** GitHub (private/institutional)

**Key Dependencies:**
- `anthropic` / `openai` / `ollama` - LLM backends
- `sentence-transformers` - Embedding generation
- `faiss-cpu` / `faiss-gpu` - Vector similarity search
- `streamlit` - Web UI framework
- `pydantic` - Schema validation

---

## Acknowledgements

**Author:** Frederick Gyasi (gyasi@musc.edu)
**Institution:** Medical University of South Carolina, Biomedical Informatics Center

---

## Summary

ClinOrchestra represents a novel approach to clinical NLP annotation by implementing a **Neuro-Symbolic Feedback Loop** that combines:

1. **Neural reasoning** (LLMs) for natural language understanding and synthesis
2. **Symbolic computation** (Functions) for deterministic clinical calculations
3. **Knowledge retrieval** (RAG) for domain-specific document grounding
4. **Task guidance** (Extras) for domain expertise hints

This hybrid architecture addresses key limitations of pure LLM approaches:
- **Hallucination mitigation** through factual grounding
- **Computational accuracy** through deterministic functions
- **Domain expertise** through retrievable knowledge
- **Reproducibility** through structured pipeline execution

The platform supports both **STRUCTURED** (production) and **ADAPTIVE** (exploratory) modes, making it suitable for research and clinical deployment.
