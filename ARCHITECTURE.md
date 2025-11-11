# ClinOrchestra Architecture

## System Overview

ClinOrchestra is a modular clinical data extraction platform built around large language models (LLMs) with tool orchestration capabilities.

```
┌─────────────────────────────────────────────────────────────┐
│                      Web Interface (Gradio)                  │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────────┐   │
│  │  Model  │ Prompt  │  Data   │  Tools  │  Processing │   │
│  │  Config │ Config  │  Upload │  Config │    Tab      │   │
│  └─────────┴─────────┴─────────┴─────────┴─────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Application State                         │
│  • Model Configuration    • Processing Configuration         │
│  • Prompt Configuration   • Optimization Settings            │
│  • Data Configuration     • Agentic Settings                 │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   Extraction Engines                         │
│  ┌──────────────────────┬────────────────────────┐          │
│  │  STRUCTURED Mode     │    ADAPTIVE Mode       │          │
│  │  (agent_system.py)   │  (agentic_agent.py)    │          │
│  │  4-stage pipeline    │  Iterative loop        │          │
│  └──────────────────────┴────────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
                           │
                  ┌────────┼────────┐
                  ▼        ▼        ▼
┌────────────┬────────────────┬──────────────┐
│ LLM Manager│  RAG Engine    │  Tools       │
│            │                │              │
│ • OpenAI   │ • Embeddings   │ • Functions  │
│ • Anthropic│ • Vector DB    │ • Patterns   │
│ • Google   │ • Retrieval    │ • Extras     │
│ • Azure    │                │              │
│ • Local    │                │              │
└────────────┴────────────────┴──────────────┘
```

## Core Components

### 1. Application State (`core/app_state.py`)
Central state manager for all configurations:
- Model settings (provider, API key, temperature, max tokens)
- Prompt configuration (main prompt, minimal prompt, JSON schema)
- Data settings (input file, columns, preprocessing options)
- Optimization settings (caching, batch processing, parallel execution)
- Agentic configuration (max iterations, max tool calls)

### 2. Extraction Engines

#### STRUCTURED Mode (`core/agent_system.py`)
Four-stage extraction pipeline:

**Stage 1: Task Analysis**
- LLM analyzes extraction task
- Determines required tools and parameters
- Generates intelligent RAG queries

**Stage 2: Tool Execution**
- Executes functions (medical calculations)
- Retrieves RAG documents
- Matches extras hints

**Stage 3: Extraction**
- LLM generates structured JSON output
- Uses preprocessed text and tool results

**Stage 4: RAG Refinement (Optional)**
- Refines selected fields with RAG evidence
- Adds source citations

#### ADAPTIVE Mode (`core/agentic_agent.py`)
Autonomous iterative loop:

**Continuous Iteration**
- LLM analyzes clinical text
- Autonomously decides tool calls
- Executes tools in parallel (async/await)
- Analyzes results and iterates
- Completes when JSON is valid

**Features:**
- Parallel tool execution (60-75% faster)
- Stall detection and recovery
- Automatic minimal prompt fallback

### 3. LLM Manager (`core/llm_manager.py`)
Unified interface for multiple LLM providers:
- Supports OpenAI, Anthropic, Google, Azure, local models
- Response caching for performance
- Cache invalidation on config changes
- Model-specific optimizations

### 4. RAG Engine (`core/rag_engine.py`)
Knowledge retrieval system:
- Document ingestion (PDF, URL, text)
- Text chunking and embedding
- Vector similarity search (FAISS)
- Relevant passage retrieval
- GPU acceleration (optional)

### 5. Tool Systems

#### Functions (`core/function_registry.py`)
Medical calculations:
- BMI, BSA, IBW calculations
- Growth percentiles and z-scores
- Lab value corrections
- Unit conversions
- Clinical scores

#### Patterns (`core/regex_preprocessor.py`)
Text normalization:
- Vital signs standardization
- Lab value formatting
- Medication parsing
- Diagnosis expansion

#### Extras (`core/extras_manager.py`)
Clinical hints and guidelines:
- Domain-specific knowledge
- Diagnostic criteria
- Reference ranges
- Assessment scales

### 6. Optimization Layer

#### LLM Cache (`core/llm_cache.py`)
- SQLite-based response caching
- 400x faster on cache hits
- Automatic cache invalidation

#### Batch Preprocessor (`core/batch_preprocessor.py`)
- Single-pass preprocessing for all texts
- 15-25% performance improvement

#### Parallel Processor (`core/parallel_processor.py`)
- Concurrent row processing
- 5-10x faster for cloud APIs
- Rate limiting and error handling

#### Performance Monitor (`core/performance_monitor.py`)
- Operation timing tracking
- Performance metrics collection

## Data Flow

### STRUCTURED Mode
```
1. Clinical Text Input
2. Preprocessing (PHI redaction, pattern normalization)
3. Stage 1: Task analysis → Tool planning
4. Stage 2: Execute tools (parallel async execution)
5. Stage 3: LLM extraction with tool results
6. Stage 4: Optional RAG refinement
7. Structured JSON Output
```

### ADAPTIVE Mode
```
1. Clinical Text Input
2. Preprocessing (PHI redaction, pattern normalization)
3. Initialize conversation with task prompt
4. Loop:
   a. LLM analyzes current state
   b. Decides tool calls OR outputs JSON
   c. If tools → Execute in parallel → Add results to history
   d. If JSON → Validate → Complete or retry
5. Output final JSON
```

## Configuration Persistence

### Automatic Saving (`core/config_persistence.py`)
All configurations automatically saved to `.clinannotate_config/`:
- Model configuration
- Prompt configuration
- Data configuration
- RAG configuration
- Processing configuration
- Optimization configuration
- Agentic configuration

## Web Interface (`ui/`)

### Tabs
- **Model Tab** (`config_tab.py`): LLM provider selection and configuration
- **Prompt Tab** (`prompt_tab.py`): Task definition and JSON schema
- **Data Tab** (`data_tab.py`): CSV upload and column selection
- **Functions Tab** (`functions_tab.py`): Medical calculation management
- **Patterns Tab** (`patterns_tab.py`): Text normalization rules
- **Extras Tab** (`extras_tab.py`): Clinical hints and guidelines
- **RAG Tab** (`rag_tab.py`): Document upload and retrieval configuration
- **Playground Tab** (`playground_tab.py`): Single extraction testing
- **Processing Tab** (`processing_tab.py`): Batch processing execution

## Performance Optimizations

### LLM Caching
- Cache responses based on prompt + config hash
- Automatic invalidation when prompts change
- 400x speedup on repeated queries

### Batch Preprocessing
- Single-pass PHI redaction and pattern normalization
- Agents skip redundant preprocessing
- 15-25% faster batch processing

### Parallel Processing
- ThreadPoolExecutor for concurrent rows
- Enabled automatically for cloud APIs
- Rate limiting to prevent throttling

### Logging Optimization
- 5MB max log file size (fast rotation)
- 3 backup files (disk space efficiency)
- INFO level default (reduced volume)

## Error Handling

### Retry Strategies
- Automatic retries on LLM failures
- Minimal prompt fallback after max retries
- JSON validation failure recovery

### Minimal Prompt Fallback
- STRUCTURED: Switches after retry_count >= max_retries
- ADAPTIVE: Switches after consecutive_json_failures >= max_retries

### Stall Detection
- Detects repeated tool calls
- Forces completion on infinite loops
- Extracts partial results when possible

## Extensibility

### Adding Custom Functions
1. Create Python file in `functions/` directory
2. Define function with proper signature
3. Add metadata (name, description, parameters)
4. Refresh in UI Functions tab

### Adding Custom Patterns
1. Define regex pattern with named groups
2. Add to Patterns tab
3. Specify replacement format

### Adding Custom Extras
1. Create JSON file with hints
2. Define keywords and content
3. Upload in Extras tab

## Security

### PHI Protection
- Automatic PHI detection (dates, names, IDs, locations)
- Configurable redaction methods
- Optional saving of redacted text

### API Key Security
- Keys stored in local config directory
- Never logged or exposed
- Environment variable fallback

## Version

**Current Version**: 1.0.0

**Key Features:**
- Dual execution modes (STRUCTURED + ADAPTIVE)
- Performance optimizations (caching, batching, parallelization)
- Automatic minimal prompt fallback
- JSON failure recovery
- Cache invalidation on config changes

---

**Author**: Frederick Gyasi
**Institution**: Medical University of South Carolina
