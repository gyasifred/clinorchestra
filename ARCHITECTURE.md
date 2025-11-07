# 🏗️ COMPLETE ARCHITECTURE ANALYSIS

## Overview

This platform is a **universal, autonomous clinical data extraction system** that uses LLM-powered agents to orchestrate multiple knowledge sources and computational tools. It adapts to ANY clinical task through prompts and JSON schemas, requiring no code changes for new use cases.

**Version:** 1.0.0
**Architecture Type:** Multi-Agent, Multi-Modal Knowledge System

---

## 📊 HIGH-LEVEL ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────┐
│                    USER INPUTS CLINICAL TASK                        │
│  (Via UI or API: Text + Task Definition + JSON Schema + Labels)    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      INTELLIGENT ORCHESTRATION                       │
│         (STRUCTURED Mode OR ADAPTIVE Mode - Both Autonomous)        │
│                                                                      │
│  The LLM analyzes the task and orchestrates these components:       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
    ┌──────────┐      ┌──────────┐     ┌──────────┐
    │   RAG    │      │ FUNCTION │     │  EXTRAS  │
    │  ENGINE  │      │ REGISTRY │     │ MANAGER  │
    └──────────┘      └──────────┘     └──────────┘
           │                 │                 │
           └─────────────────┼─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  JSON EXTRACTOR │
                    │   & VALIDATOR   │
                    └─────────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  STRUCTURED OUTPUT   │
                  │  (Task-specific JSON)│
                  └──────────────────────┘
```

---

## 🔧 CORE COMPONENTS

### 1. AGENT SYSTEM (The Orchestrators)

The agent system provides two execution modes, both autonomous and universal:

#### A. ExtractionAgent (STRUCTURED Mode)
**File:** `core/agent_system.py`

**Purpose:** Systematic, predictable 4-stage extraction for production workflows

**Execution Flow:**
```
Stage 1: TASK ANALYSIS
├─ LLM reads: Clinical text, task definition, JSON schema, labels (optional)
├─ LLM analyzes: What information is needed?
├─ LLM decides: Which tools to call? (RAG, Functions, Extras)
└─ Output: Tool request list

Stage 2: TOOL EXECUTION (ASYNC - Parallel!)
├─ Execute RAG queries (knowledge retrieval)
├─ Execute Functions (calculations, conversions)
├─ Execute Extras queries (supplementary hints)
└─ All run in PARALLEL for 60-75% speedup

Stage 3: EXTRACTION
├─ LLM receives: Original text + All tool results
├─ LLM extracts: Information matching JSON schema
├─ LLM uses: Tool results to inform extraction
└─ Output: Structured JSON

Stage 4: RAG REFINEMENT (Optional)
├─ If RAG was used, refine extraction
├─ Validate against retrieved knowledge
└─ Output: Final validated JSON
```

**Key Features:**
- ✅ Predictable, systematic workflow
- ✅ Autonomous tool selection (LLM decides what to call)
- ✅ ASYNC parallel tool execution
- ✅ Works for ANY clinical task
- ✅ Label context is OPTIONAL

---

#### B. AgenticAgent (ADAPTIVE Mode)
**File:** `core/agentic_agent.py`

**Purpose:** Continuous iteration for evolving, complex tasks requiring dynamic adaptation

**Execution Flow:**
```
Continuous Loop:
├─ Iteration 1: LLM analyzes → Calls tools → Gets results
├─ Iteration 2: LLM learns from results → Calls MORE tools → Gets results
├─ Iteration 3: LLM refines understanding → Calls tools → Gets results
├─ ...continues until extraction complete or max iterations
└─ PAUSE/RESUME states for dynamic control

Key Difference from STRUCTURED:
- Can call tools MULTIPLE times with different queries
- Learns and adapts strategy based on results
- "That BMI result tells me I need growth percentile next"
- More flexible but less predictable
```

**Key Features:**
- ✅ Continuous iterative refinement
- ✅ Dynamic tool calling (multiple iterations)
- ✅ Learns from tool results
- ✅ ASYNC parallel tool execution
- ✅ PAUSE/RESUME states
- ✅ Works for ANY clinical task
- ✅ Label context is OPTIONAL

**Native Tool Calling:**
- Uses OpenAI/Anthropic function calling API
- LLM natively requests tool calls
- Maintains conversation context across iterations

---

#### C. AgentFactory
**File:** `core/agent_factory.py`

**Purpose:** Creates the appropriate agent based on user's execution mode choice

**Logic:**
```python
if user_selects_adaptive_mode:
    return AdaptiveAgent(...)  # For evolving tasks
else:
    return ExtractionAgent(...)  # For predictable workflows
```

**Important:** Both agents are equally capable - they differ only in execution style, not capability!

---

### 2. LLM MANAGER (The Brain)

**File:** `core/llm_manager.py`

**Purpose:** Manages communication with ANY LLM provider through unified interface

**Supported Providers:**
```
├─ OpenAI (GPT-4, GPT-4 Turbo, GPT-3.5)
├─ Anthropic (Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku)
├─ Google (Gemini models)
├─ Azure OpenAI (all Azure-hosted OpenAI models)
├─ Local models (via Unsloth - quantized LLaMA, Mistral, etc.)
└─ Future: Any provider with OpenAI-compatible interface
```

**Key Features:**
- **Unified Interface:** Same code works across all providers
- **Native Tool Calling:** Supports function calling API
- **Streaming:** Real-time response streaming
- **Token Management:** Automatic token counting and limits
- **Retry Logic:** Exponential backoff on failures
- **Rate Limiting:** Respects provider rate limits
- **Error Handling:** Graceful degradation

**Why It Matters:**
Users can switch LLM providers without changing any code, enabling:
- Cost optimization (use cheaper models for simple tasks)
- Performance tuning (use powerful models for complex tasks)
- Provider redundancy (fallback if one provider is down)

---

### 3. RAG ENGINE (Knowledge Retrieval)

**File:** `core/rag_engine.py`

**Purpose:** Semantic search over medical documents to retrieve relevant knowledge

**Architecture:**
```
Indexing Phase:
├─ User uploads documents (PDFs, text files, markdown)
├─ System chunks documents into semantic segments
├─ Generates embeddings using transformer models:
│   ├─ sentence-transformers/all-MiniLM-L6-v2 (fast, lightweight)
│   └─ BAAI/bge-large-en-v1.5 (high quality, slower)
├─ Stores in vector database (FAISS)
└─ Persists to disk for reuse

Query Phase:
├─ Agent formulates query (e.g., "normal BMI ranges for children")
├─ RAG engine converts query to embedding
├─ Searches vector DB for most similar chunks
├─ Returns top-k most relevant passages with similarity scores
└─ Agent uses retrieved knowledge to inform extraction
```

**Use Cases:**
- Query growth charts for pediatric assessments
- Retrieve medication dosing guidelines
- Find disease classification criteria
- Look up normal lab value ranges
- Access clinical practice guidelines

**Key Features:**
- ✅ **Semantic Search:** Understands meaning, not just keywords
- ✅ **Configurable Models:** Choose speed vs quality
- ✅ **Persistent Storage:** Index once, query many times
- ✅ **Fast Retrieval:** Optimized vector search
- ✅ **Universal:** Works with ANY medical domain documents
- ✅ **Batch Processing:** Index multiple documents efficiently

**Technical Details:**
- Vector database: FAISS (Facebook AI Similarity Search - fast, local, persistent)
- Embedding dimensions: 384 (MiniLM) or 1024 (BGE)
- Chunking strategy: Semantic segmentation with overlap
- Similarity metric: Cosine similarity

---

### 4. FUNCTION REGISTRY (Computational Tools)

**File:** `core/function_registry.py`

**Purpose:** Provides calculators and data processing functions that agents can call

**Architecture:**
```
Function Definition (JSON format):
{
    "name": "calculate_bmi",
    "description": "Calculate Body Mass Index from weight and height",
    "parameters": {
        "weight_kg": {
            "type": "float",
            "description": "Weight in kilograms"
        },
        "height_m": {
            "type": "float",
            "description": "Height in meters"
        }
    },
    "code": "def calculate_bmi(weight_kg, height_m): ..."
}

Registration Process:
├─ Functions stored in /functions/*.json
├─ Registry loads all functions on startup
├─ Validates function signatures and parameters
├─ Makes them available to agents via tool calling
└─ Handles type conversion (string→number, etc.)

Execution Flow:
├─ Agent requests: calculate_bmi(weight_kg=70, height_m=1.75)
├─ Registry validates parameters
├─ Registry executes function in safe environment
├─ Returns result: {"bmi": 22.86, "category": "Normal"}
└─ Agent uses result in extraction process
```

**Built-in Functions:**
- `calculate_bmi()` - Body Mass Index calculation
- `calculate_weight_change_percent()` - Weight change over time
- `calculate_growth_percentile()` - CDC growth percentiles
- `calculate_z_score()` - Standard deviation scores
- `convert_units()` - Unit conversions (kg↔lbs, cm↔in, etc.)
- Custom functions can be added by dropping JSON files!

**Key Features:**
- ✅ **Dynamic Loading:** Add functions without code changes
- ✅ **Type Conversion:** Handles string→number, etc. automatically
- ✅ **Safe Execution:** Sandboxed environment
- ✅ **Parameter Validation:** Ensures correct inputs
- ✅ **Error Handling:** Graceful failure with informative messages
- ✅ **Extensible:** Easy to add domain-specific calculators

**Extension Example:**
To add a new function, simply create `/functions/my_function.json`:
```json
{
    "name": "calculate_egfr",
    "description": "Calculate estimated glomerular filtration rate",
    "parameters": {
        "creatinine": {"type": "float", "description": "Serum creatinine mg/dL"},
        "age": {"type": "int", "description": "Patient age in years"},
        "sex": {"type": "string", "description": "M or F"}
    },
    "code": "def calculate_egfr(creatinine, age, sex): ..."
}
```

---

### 5. EXTRAS MANAGER (Supplementary Hints)

**File:** `core/extras_manager.py`

**Purpose:** Provides task-specific hints, guidelines, and tips to help the LLM make accurate extractions

**Architecture:**
```
Extra Definition (JSON format):
{
    "id": "unique_identifier",
    "type": "guideline" | "reference" | "hint" | "conversion",
    "keywords": ["malnutrition", "z-score", "percentile"],
    "content": "CRITICAL: Lower percentiles correspond to NEGATIVE z-scores.
                Common conversions: 3rd percentile ≈ z-score -1.88...",
    "metadata": {
        "category": "nutrition",
        "priority": "high",
        "domain": "pediatrics"
    }
}

Matching Process:
├─ Agent provides keywords based on task (e.g., ["malnutrition", "pediatric"])
├─ Extras Manager performs fuzzy matching against all extras
├─ Ranks extras by relevance score
├─ Returns top matches with content
└─ Agent receives context-specific guidance

Example Extras:
├─ "Z-score and percentile relationship for growth assessment"
├─ "Common malnutrition indicators and diagnostic thresholds"
├─ "Unit conversion reference for medication dosing"
├─ "Clinical interpretation guidelines for laboratory values"
└─ "Disease classification criteria (ICD-10, DSM-5, etc.)"
```

**Why This Matters:**
- Reduces LLM hallucinations with domain-specific knowledge
- Provides critical context (e.g., "lower percentile = negative z-score")
- Helps avoid common clinical interpretation errors
- Works universally across domains (not limited to malnutrition!)

**Key Features:**
- ✅ **Keyword-based Matching:** Flexible query system
- ✅ **Fuzzy Search:** Typo-tolerant matching
- ✅ **Relevance Ranking:** Best matches returned first
- ✅ **Easy Extension:** Add extras by dropping JSON files
- ✅ **Universal:** Works across all clinical domains
- ✅ **Priority System:** High-priority extras surfaced first

**Storage:** `/extras/*.json`

---

### 6. PREPROCESSING SYSTEM

#### A. Regex Preprocessor
**File:** `core/regex_preprocessor.py`

**Purpose:** Normalizes clinical text before LLM processing to improve accuracy

**Transformations:**
```
Date Normalization:
├─ "12/25/2023" → "2023-12-25"
├─ "Dec 25, 2023" → "2023-12-25"
└─ "25-Dec-2023" → "2023-12-25"

Number Extraction:
├─ "weight: 25.5 kg" → captures "25.5"
├─ "BP 120/80" → captures "120", "80"
└─ "5'10\"" → converts to "177.8 cm"

Unit Standardization:
├─ "5 feet 10 inches" → "177.8 cm"
├─ "150 lbs" → "68.04 kg"
└─ "98.6°F" → "37°C"

Range Parsing:
├─ "BP 120-80" → "systolic: 120, diastolic: 80"
├─ "Weight 50-55 kg" → "weight_min: 50, weight_max: 55"
└─ "Age 5-10 years" → "age_min: 5, age_max: 10"
```

**Configuration:**
- Patterns stored in JSON format
- Loaded at startup
- Applied sequentially
- User-customizable per task

**Key Features:**
- ✅ **Customizable Patterns:** Define task-specific rules
- ✅ **Sequential Application:** Order matters for complex transforms
- ✅ **Original Preservation:** Keeps original text for reference
- ✅ **Improves Accuracy:** Standardized format helps LLM

---

#### B. PII Redactor
**File:** `core/pii_redactor.py`

**Purpose:** Removes or masks sensitive information for HIPAA compliance

**Entity Detection:**
```
Supported Entities:
├─ PERSON (names)
├─ DATE (dates of birth, visit dates)
├─ MRN (medical record numbers)
├─ ORGANIZATION (hospital names, clinics)
├─ GPE (cities, states)
├─ LOC (specific locations)
└─ Custom entity types (extensible)
```

**Redaction Modes:**
```
MASK Mode:
├─ "John Doe" → "[PERSON]"
├─ "12/25/2023" → "[DATE]"
└─ "MRN: 12345" → "MRN: [MRN]"

REMOVE Mode:
├─ "John Doe visited" → "visited"
├─ Completely removes entity
└─ Preserves sentence structure

HASH Mode:
├─ "John Doe" → "PERSON_abc123"
├─ Consistent hashing (same person = same hash)
└─ Enables re-identification if needed
```

**Key Features:**
- ✅ **NER-based Detection:** Uses spaCy for accuracy
- ✅ **Multiple Strategies:** Choose redaction approach
- ✅ **Configurable Entities:** Select which to redact
- ✅ **HIPAA Compliance:** Support for de-identification
- ✅ **Reversible Hashing:** Optional re-identification

---

### 7. JSON PROCESSING

#### A. JSON Parser
**File:** `core/json_parser.py`

**Purpose:** Extracts and validates JSON from LLM responses (handles LLM quirks)

**Parsing Strategies (Applied Sequentially):**
```
1. Clean JSON Extraction:
   ├─ Finds JSON blocks in markdown (```json ... ```)
   ├─ Handles nested structures
   ├─ Validates against schema
   └─ Returns if valid

2. Fuzzy JSON Extraction:
   ├─ Handles malformed JSON (missing quotes, trailing commas)
   ├─ Fixes common LLM errors
   ├─ Attempts repair and re-validation
   └─ Returns if repairable

3. Fallback Parsing:
   ├─ Extracts partial JSON
   ├─ Returns best-effort structure
   ├─ Logs which fields are missing
   └─ Allows graceful degradation
```

**Key Features:**
- ✅ **Robust:** Handles LLM inconsistencies
- ✅ **Multiple Strategies:** Fallback chain ensures success
- ✅ **Schema Validation:** Ensures output matches expected format
- ✅ **Detailed Logging:** Tracks which parsing method succeeded
- ✅ **Error Recovery:** Attempts repair before failing

---

#### B. Output Handler
**File:** `core/output_handler.py`

**Purpose:** Formats and exports extraction results in multiple formats

**Output Formats:**
```
CSV Export:
├─ Flattened JSON for spreadsheet compatibility
├─ Nested fields → dot notation (e.g., "patient.age")
├─ Arrays → comma-separated values
└─ Metadata columns (timestamp, model_used, etc.)

JSON Export:
├─ Full structured output preserved
├─ Pretty-printed for readability
├─ Includes extraction metadata
└─ Per-record or batch export

Excel Export:
├─ Multi-sheet workbooks
├─ Sheet 1: Extracted data
├─ Sheet 2: Metadata
├─ Sheet 3: Processing logs
└─ Formatted for analysis

Future:
└─ Database insertion (SQL, MongoDB, etc.)
```

**Features:**
- **Schema Flattening:** Nested → flat for CSV
- **Metadata Inclusion:** Timestamps, model info, tool usage
- **Batch Processing:** Efficient bulk export
- **Incremental Saving:** Save as processing completes

---

### 8. PROMPT MANAGEMENT

**File:** `core/prompt_templates.py`

**Purpose:** Manages all prompt templates and dynamic prompt generation

**Template Types:**
```
DEFAULT Templates (Generic):
├─ Task-agnostic base templates
├─ Work for any clinical domain
├─ Schema-driven prompt generation
└─ No hardcoded task assumptions

EXAMPLE Templates (Illustrative):
├─ MALNUTRITION_* (pre-configured for malnutrition)
├─ DIABETES_* (pre-configured for diabetes)
├─ NOT required - just examples!
└─ Users can define custom templates

Custom Templates:
├─ User-defined task-specific prompts
├─ Variable substitution ({task}, {schema}, etc.)
├─ Template inheritance
└─ Easy customization
```

**Dynamic Prompt Building:**
```
Prompt Generation Process:
├─ 1. Load base template (or use default)
├─ 2. Inject user's task definition
├─ 3. Format JSON schema as instructions
├─ 4. Include label context (if provided - OPTIONAL)
├─ 5. Format tool results (if Stage 2 complete)
├─ 6. Add examples (if configured)
├─ 7. Adapt for execution mode (STRUCTURED vs ADAPTIVE)
└─ 8. Return complete prompt for LLM
```

**Key Features:**
- ✅ **Template Inheritance:** Build on base templates
- ✅ **Dynamic Schema Integration:** Schema → instructions
- ✅ **Variable Substitution:** Flexible customization
- ✅ **Mode-specific Prompts:** Different for STRUCTURED vs ADAPTIVE
- ✅ **Optional Labels:** Works with or without label context
- ✅ **Easy Customization:** Users can edit/create templates

---

### 9. CONFIGURATION SYSTEM

#### A. Model Configuration
**File:** `core/model_config.py`

**Purpose:** Manages LLM provider settings and parameters

**Configuration Options:**
```python
ModelConfig:
├─ provider: "openai" | "anthropic" | "azure" | "local"
├─ model_name: "gpt-4" | "claude-3-5-sonnet-20241022" | ...
├─ api_key: str (user's API key)
├─ temperature: float (0.0 - 2.0, controls randomness)
├─ max_tokens: int (response length limit)
├─ top_p: float (nucleus sampling)
├─ frequency_penalty: float (reduce repetition)
├─ presence_penalty: float (encourage diversity)
└─ Provider-specific:
    ├─ endpoint: str (custom API endpoint)
    ├─ api_version: str (API version for Azure)
    ├─ deployment_name: str (Azure deployment)
    └─ base_url: str (for local models)
```

---

#### B. App State
**File:** `core/app_state.py`

**Purpose:** Central state management for the entire application

**Managed State:**
```
Configuration Objects:
├─ model_config: LLM settings
├─ prompt_config: Prompt templates and settings
├─ rag_config: RAG indexing and query settings
├─ processing_config: Batch processing parameters
├─ adaptive_config: ADAPTIVE mode settings (max_iterations, max_tool_calls)
└─ data_config: Input data column mappings, label mappings

Initialized Components:
├─ llm_manager: Initialized LLM client
├─ rag_engine: Initialized RAG engine with vector DB
├─ function_registry: Loaded function definitions
├─ extras_manager: Loaded extras
└─ Current processing status and logs
```

**Key Features:**
- ✅ **Centralized:** Single source of truth
- ✅ **Lazy Initialization:** Components created on demand
- ✅ **Configuration Validation:** Ensures valid settings
- ✅ **State Persistence:** Saves to disk
- ✅ **Easy Access:** All components accessible globally

---

#### C. Config Persistence
**File:** `core/config_persistence.py`

**Purpose:** Saves/loads configurations to/from disk

**Features:**
- Automatic saving to JSON files
- Load on application startup
- Version migration support
- Backup creation before overwrites
- Handles nested configuration objects

**Storage Location:** `~/.clinannotate/config.json`

---

### 10. UI SYSTEM (Gradio Interface)

**Directory:** `ui/`

The UI provides a user-friendly interface for all platform capabilities through **9 comprehensive tabs**:

**Complete Tab List:**
1. **Model Configuration** - LLM provider and execution mode setup
2. **Data Configuration** - Input data upload and column mapping
3. **Prompt Configuration** - Task definition and JSON schema
4. **RAG** - Document upload and vector database management
5. **Regex Patterns** - Text preprocessing rules (120+ built-in patterns)
6. **Extras (Hints)** - Task-specific knowledge (183+ built-in hints)
7. **Custom Functions** - Computational tools (40+ built-in calculators)
8. **Playground** - Single-text testing and debugging
9. **Processing** - Batch execution and monitoring

All tabs support **YAML and JSON** file loading for easy configuration sharing!

---

#### A. Config Tab (`ui/config_tab.py`)
**Purpose:** LLM and execution mode configuration

**Features:**
- LLM provider selection (OpenAI, Anthropic, Azure, Local)
- API key management (secure input)
- Model selection (GPT-4, Claude 3.5, etc.)
- Execution mode selection:
  - ✅ STRUCTURED Mode (for predictable workflows)
  - ✅ ADAPTIVE Mode (for evolving tasks)
- Model parameter tuning (temperature, max_tokens, etc.)
- ADAPTIVE mode settings (max iterations, max tool calls)
- Configuration save/load
- Connection testing

---

#### B. Data Tab (`ui/data_tab.py`)
**Purpose:** Data input and mapping configuration

**Features:**
- **File Upload:** CSV, Excel, JSON formats
- **Column Mapping:**
  - Select text column (clinical notes)
  - Select label column (OPTIONAL - for supervised tasks)
  - Select ID column (optional)
- **Label Mapping:**
  - Map label values to descriptive text
  - Example: 0 → "No malnutrition", 1 → "Malnutrition present"
  - COMPLETELY OPTIONAL - system works without labels
- **Data Preview:** View uploaded data
- **Validation:** Checks for required columns

**Important:** Label context is OPTIONAL. The system adapts to:
- Supervised tasks (with labels)
- Unsupervised tasks (without labels)
- Every task can have different requirements

---

#### C. Prompt Tab (`ui/prompt_tab.py`)
**Purpose:** Prompt template and task definition management

**Features:**
- **Prompt Template Selection:**
  - DEFAULT (generic, task-agnostic)
  - MALNUTRITION (example template)
  - DIABETES (example template)
  - CUSTOM (user-defined)
- **Custom Prompt Editing:**
  - Full text editor for prompt customization
  - Variable substitution support
  - Real-time preview
- **JSON Schema Definition:**
  - Define expected output structure
  - Field types, descriptions, requirements
  - Nested object support
- **Template Testing:**
  - Test prompts with sample text
  - Preview generated prompts
- **Task Description:**
  - Free-text task definition
  - Instructions for the LLM

---

#### D. RAG Tab (`ui/rag_tab.py`)
**Purpose:** Document upload and RAG management

**Features:**
- **Document Upload:**
  - PDF, TXT, Markdown support
  - Batch upload multiple documents
  - Preview uploaded documents
- **Vector Database Management:**
  - View indexed documents
  - Delete/update documents
  - Rebuild index
- **Embedding Model Selection:**
  - sentence-transformers/all-MiniLM-L6-v2 (fast)
  - BAAI/bge-large-en-v1.5 (high quality)
- **Index Building:**
  - Progress tracking
  - Chunking configuration
  - Status monitoring
- **Query Testing:**
  - Test RAG queries
  - View retrieved passages
  - Check relevance scores

---

#### E. Regex Patterns Tab (`ui/patterns_tab.py`)
**Purpose:** Text preprocessing pattern management

**Features:**
- **Pattern Registration:**
  - Define regex patterns for text normalization
  - Set replacement rules
  - Enable/disable patterns individually
- **File Upload:**
  - Upload pattern files (YAML/JSON)
  - Batch load multiple patterns
- **Pattern Testing:**
  - Test patterns against sample text
  - Preview before/after transformations
  - Validate regex syntax
- **Built-in Patterns:**
  - Standardize medical units (mg, kg, etc.)
  - Normalize blood pressure formats
  - Fix date formats
  - Remove extra whitespace
  - And 100+ more medical text patterns!
- **Preview/Edit:**
  - View all registered patterns in dataframe
  - Edit pattern details (name, regex, replacement)
  - Remove unwanted patterns
  - Toggle patterns on/off

**Use Case:** Standardize inconsistent clinical text BEFORE it goes to the LLM

---

#### F. Extras (Hints) Tab (`ui/extras_tab.py`)
**Purpose:** Task-specific hints and knowledge management

**Features:**
- **Extras Registration:**
  - Add task-specific hints, guidelines, criteria
  - Type classification (pattern, definition, guideline, reference, criteria, tip)
  - Metadata tagging (category, priority, domain)
- **File Upload:**
  - Upload extras files (YAML/JSON)
  - Batch load multiple extras
- **Built-in Extras:**
  - 183+ pre-loaded clinical hints including:
    - WHO growth standards
    - ASPEN malnutrition criteria
    - Diagnostic criteria (diabetes, AKI, etc.)
    - Lab value interpretation guides
    - Medication dosing guidelines
- **Preview/Edit:**
  - View all registered extras in dataframe
  - Edit extra content and metadata
  - Remove unwanted extras
  - Search and filter extras

**Use Case:** Provide domain-specific knowledge that agents can query when needed

---

#### G. Custom Functions Tab (`ui/functions_tab.py`)
**Purpose:** Computational tools and calculator management

**Features:**
- **Function Registration:**
  - Define Python functions with parameters
  - Set parameter types and descriptions
  - Specify return value format
- **File Upload:**
  - Upload function files (YAML/JSON)
  - Batch load multiple functions
- **Built-in Functions:**
  - 40+ medical calculators including:
    - BMI, BSA, ideal body weight
    - Growth percentiles and z-scores
    - Creatinine clearance, eGFR
    - Anion gap, corrected calcium
    - Unit conversions (kg↔lbs, cm↔in, etc.)
    - Weight change percentages
    - Mean arterial pressure
    - Pack-years smoking history
- **Function Testing:**
  - Test functions with custom arguments (JSON format)
  - View function results
  - Validate execution
- **Preview/Edit:**
  - View all registered functions in dataframe
  - Edit function code and parameters
  - Remove unwanted functions
  - Export/import function definitions

**Use Case:** Provide accurate calculations that LLMs struggle with (math, dates, complex formulas)

---

#### H. Processing Tab (`ui/processing_tab.py`)
**Purpose:** Batch processing execution and monitoring

**Features:**
- **Execution Controls:**
  - Start/stop processing
  - Pause/resume support
  - Cancel processing
- **Real-time Progress:**
  - Progress bar
  - Current record indicator
  - Records/second throughput
  - Estimated time remaining
- **Live Logging:**
  - Real-time log stream
  - Error highlighting
  - Success indicators
  - Tool usage tracking
- **Error Handling:**
  - Skip on error
  - Retry failed records
  - Max retry configuration
  - Error log export
- **Batch Configuration:**
  - Batch size
  - Parallel processing
  - Checkpoint frequency
- **Results Display:**
  - Extraction success indicators:
    - ✅ SUCCESS - Extraction completed with valid JSON
    - ⚠️ COMPLETED - Agent finished but no JSON
    - ❌ FAILED - Agent did not complete
  - Tool usage summary (RAG, Functions, Extras)
  - Export options

---

#### I. Playground Tab (`ui/playground_tab.py`)
**Purpose:** Single-text testing and debugging

**Features:**
- **Quick Testing:**
  - Paste single clinical note
  - Run extraction
  - View results immediately
- **Result Preview:**
  - Formatted JSON output
  - Extraction metadata
  - Tool calls log
- **Debugging:**
  - View LLM prompts
  - Check tool results
  - Trace execution flow
- **Iteration Testing:**
  - Test ADAPTIVE mode iterations
  - View each iteration's decisions

---

### 11. EVALUATION SYSTEM

**Directory:** `evaluation/`

**Purpose:** Measure extraction accuracy against ground truth

**Components:**

#### Metrics Calculator
**Features:**
- **Field-level Metrics:**
  - Precision, Recall, F1 score per field
  - Exact match accuracy
  - Partial match scoring
- **Aggregate Metrics:**
  - Overall F1 score
  - Average precision/recall
  - Error analysis
- **Per-record Scoring:**
  - Individual record performance
  - Error categorization

#### Evaluation Modes
```
Standard Evaluation:
├─ Exact string matching
├─ Numeric value comparison (with tolerance)
├─ Boolean comparison
└─ Null/missing handling

Relaxed Evaluation:
├─ Fuzzy string matching
├─ Case-insensitive comparison
├─ Whitespace normalization
└─ Synonym matching

Strict Evaluation:
├─ Exact match required
├─ Type-strict comparison
├─ No tolerance
└─ Perfect alignment required
```

**Output:**
- Detailed metrics report (CSV/JSON)
- Confusion matrices
- Error analysis by field
- Performance by execution mode

---

### 12. DATA PROCESSING UTILITIES

#### A. Growth Calculators
**Files:** `core/growth_calculators.py`, `core/cdc_growth_calculator.py`

**Purpose:** Clinical growth assessment calculations

**Features:**
- **CDC Growth Charts:**
  - Weight-for-age (0-36 months, 2-20 years)
  - Length/stature-for-age
  - BMI-for-age
  - Head circumference-for-age
  - Weight-for-stature
- **Calculations:**
  - Percentile computation (0-100)
  - Z-score calculation (-4 to +4)
  - Age-specific references
  - Sex-specific calculations
- **Data:**
  - CDC reference data included
  - LMS method (Lambda-Mu-Sigma)
  - Accurate interpolation

---

#### B. Data Processor
**File:** `core/data_processor.py`

**Purpose:** Input data parsing and validation

**Features:**
- **File Format Support:**
  - CSV (comma, tab, custom delimiters)
  - Excel (.xlsx, .xls)
  - JSON (records, objects)
- **Data Validation:**
  - Column existence checks
  - Data type validation
  - Required field verification
- **Column Mapping:**
  - Flexible column selection
  - Rename columns
  - Type conversion
- **Batch Preparation:**
  - Split into batches
  - Shuffle support
  - Sampling support

---

## 🔄 HOW EVERYTHING WORKS TOGETHER

### Example End-to-End Flow: Malnutrition Extraction

```
1. USER CONFIGURATION:
   ├─ Uploads clinical_notes.csv via Data Tab
   ├─ Maps columns: text_column="clinical_note", label_column="malnutrition_label"
   ├─ Selects STRUCTURED mode via Config Tab (predictable workflow)
   ├─ Chooses GPT-4 as LLM provider
   ├─ Uploads WHO growth chart PDFs via RAG Tab
   ├─ Defines JSON schema for malnutrition fields via Prompt Tab
   ├─ Adds malnutrition-specific extras (z-score guidelines)
   └─ Clicks "Start Processing" in Processing Tab

2. SYSTEM INITIALIZATION:
   ├─ App State loads all configurations
   ├─ LLM Manager establishes connection to OpenAI
   ├─ RAG Engine loads indexed growth chart documents
   ├─ Function Registry loads BMI, z-score, percentile calculators
   ├─ Extras Manager loads malnutrition interpretation hints
   ├─ Agent Factory creates ExtractionAgent (STRUCTURED mode)
   └─ Data Processor loads and validates CSV

3. FOR EACH CLINICAL NOTE:

   Stage 1: TASK ANALYSIS
   ├─ Regex Preprocessor normalizes text (dates, numbers, units)
   ├─ PII Redactor removes sensitive information (if enabled)
   ├─ LLM receives:
   │   ├─ Preprocessed clinical text
   │   ├─ Task definition: "Extract malnutrition indicators"
   │   ├─ JSON schema: {age, weight, height, bmi, z_score, percentile, ...}
   │   └─ Label context: "malnutrition_label=1" (OPTIONAL - only if provided)
   ├─ LLM analyzes clinical note
   ├─ LLM identifies needed information:
   │   ├─ "I need to calculate BMI"
   │   ├─ "I need growth percentile for age 5"
   │   ├─ "I should check normal ranges from documents"
   │   └─ "I need z-score interpretation guidelines"
   ├─ LLM decides tool calls:
   │   ├─ RAG query: "normal growth ranges for 5-year-old children"
   │   ├─ Function: calculate_bmi(weight_kg=18, height_m=1.1)
   │   ├─ Function: calculate_growth_percentile(weight=18, age_months=60, sex="M")
   │   └─ Extras: keywords=["malnutrition", "z-score", "pediatric"]
   └─ Returns: Structured tool request list

   Stage 2: TOOL EXECUTION (ASYNC - All in Parallel!)
   ├─ [PARALLEL TASK 1] RAG Engine:
   │   ├─ Converts query to embedding
   │   ├─ Searches vector database
   │   ├─ Retrieves: "For 5-year-old boys, 3rd percentile = -1.88 SD..."
   │   └─ Returns top-3 relevant passages (0.3s)
   │
   ├─ [PARALLEL TASK 2] Function Registry - BMI:
   │   ├─ Validates parameters: weight_kg=18.0, height_m=1.1
   │   ├─ Executes: calculate_bmi(18.0, 1.1)
   │   ├─ Returns: {"bmi": 14.88, "category": "Underweight"}
   │   └─ Completes in 0.1s
   │
   ├─ [PARALLEL TASK 3] Function Registry - Percentile:
   │   ├─ Loads CDC reference data
   │   ├─ Executes: calculate_growth_percentile(18, 60, "M")
   │   ├─ Returns: {"percentile": 3, "z_score": -1.88}
   │   └─ Completes in 0.2s
   │
   ├─ [PARALLEL TASK 4] Extras Manager:
   │   ├─ Matches keywords: ["malnutrition", "z-score", "pediatric"]
   │   ├─ Fuzzy search across extras database
   │   ├─ Returns: "CRITICAL: 3rd percentile = z-score -1.88 (moderate malnutrition)"
   │   └─ Completes in 0.1s
   │
   └─ Total Time: 0.3s (vs 0.7s if sequential) - 60% FASTER!

   Stage 3: EXTRACTION
   ├─ LLM receives comprehensive context:
   │   ├─ Original clinical note (preprocessed)
   │   ├─ RAG results: Growth chart references
   │   ├─ Function results: BMI=14.88, percentile=3, z-score=-1.88
   │   ├─ Extras: Interpretation guidelines
   │   └─ JSON schema: Expected output structure
   ├─ LLM extracts structured data informed by tools:
   │   {
   │     "patient_age_months": 60,
   │     "weight_kg": 18.0,
   │     "height_cm": 110.0,
   │     "bmi": 14.88,
   │     "growth_percentile": 3,
   │     "z_score": -1.88,
   │     "malnutrition_present": true,
   │     "malnutrition_severity": "moderate",
   │     "evidence": "Weight at 3rd percentile (z-score -1.88)"
   │   }
   ├─ JSON Parser extracts and validates JSON
   ├─ Schema validation ensures all required fields present
   └─ Validated JSON ready for Stage 4

   Stage 4: RAG REFINEMENT (Optional)
   ├─ Since RAG was used, perform refinement
   ├─ LLM validates extraction against RAG knowledge:
   │   ├─ Confirms: "3rd percentile indicates moderate malnutrition"
   │   ├─ Validates: z-score calculation correct
   │   └─ Verifies: interpretation aligns with guidelines
   ├─ Makes any necessary corrections
   └─ Returns: Final validated JSON

4. OUTPUT GENERATION:
   ├─ Output Handler collects all extraction results
   ├─ Adds metadata:
   │   ├─ timestamp: "2024-01-15T10:30:45Z"
   │   ├─ model_used: "gpt-4"
   │   ├─ execution_mode: "STRUCTURED"
   │   ├─ rag_used: true
   │   ├─ functions_called: ["calculate_bmi", "calculate_growth_percentile"]
   │   ├─ extras_used: true
   │   └─ processing_time_seconds: 2.1
   ├─ Exports to CSV:
   │   ├─ Flattened JSON columns
   │   ├─ Metadata columns
   │   └─ Original text preserved
   ├─ Exports to Excel:
   │   ├─ Sheet 1: Extraction results
   │   ├─ Sheet 2: Metadata
   │   └─ Sheet 3: Processing logs
   └─ User downloads structured dataset!

5. EVALUATION (Optional):
   ├─ If ground truth labels provided
   ├─ Evaluation system compares extraction vs truth
   ├─ Calculates:
   │   ├─ Field-level F1 scores
   │   ├─ Overall accuracy
   │   └─ Error analysis
   └─ Generates metrics report
```

---

## 🌟 KEY ARCHITECTURAL PRINCIPLES

### 1. TRUE UNIVERSALITY
**No Task Hardcoding:**
- ✅ System has NO hardcoded clinical tasks
- ✅ Works for malnutrition, diabetes, renal function, cardiac assessment, etc.
- ✅ User defines task via prompts and JSON schema
- ✅ Adapts to labeled OR unlabeled data
- ✅ No code changes needed for new clinical domains

**Adaptability:**
- Every task can have different requirements
- Label context is COMPLETELY OPTIONAL
- Schema defines output structure dynamically
- Prompt defines extraction instructions
- Works across all medical specialties

---

### 2. INTELLIGENT ORCHESTRATION
**Autonomous Decision-Making:**
- ✅ LLM analyzes task and decides which tools to call
- ✅ No manual configuration of tool selection
- ✅ Context-aware tool usage
- ✅ Dynamic adaptation based on available tools

**Multi-Modal Knowledge Integration:**
- RAG: Retrieval from documents (semantic search)
- Functions: Computational tools (calculations)
- Extras: Domain-specific hints (guidelines)
- LLM: Reasoning and extraction (intelligence)

All four knowledge sources work together seamlessly!

---

### 3. DUAL EXECUTION MODES
**STRUCTURED Mode:**
- For predictable, systematic workflows
- 4-stage pipeline (Analysis → Tools → Extraction → Refinement)
- Best for production environments
- Deterministic, repeatable

**ADAPTIVE Mode:**
- For evolving, complex tasks
- Continuous iteration with learning
- Dynamic tool calling across iterations
- Best for research and complex cases

**Important:** Both are EQUALLY autonomous and universal!

---

### 4. PERFORMANCE OPTIMIZATION
**ASYNC Tool Execution:**
- ✅ All Stage 2 tools run in PARALLEL
- ✅ 60-75% performance improvement
- ✅ Maintains execution order
- ✅ Handles failures gracefully

**Efficiency:**
- Lazy initialization (components created on demand)
- Vector database caching
- Persistent storage
- Batch processing support

---

### 5. EXTENSIBILITY
**Easy to Extend:**
- ✅ Add functions: Drop JSON file in `/functions/`
- ✅ Add extras: Drop JSON file in `/extras/`
- ✅ Add documents: Upload via RAG tab
- ✅ Add prompts: Create custom templates
- ✅ Add LLM providers: Implement provider interface

**No Code Changes Required:**
- Users extend via configuration and data files
- Developers extend via modular interfaces
- Plugin-style architecture

---

### 6. ROBUST ERROR HANDLING
**Multiple Fallbacks:**
- JSON parsing: 3 strategies (clean → fuzzy → fallback)
- LLM retries: Exponential backoff
- Tool failures: Graceful degradation
- Validation: Schema-based with informative errors

**Logging:**
- Detailed execution logs
- Error categorization
- Performance metrics
- Audit trail

---

## 💡 WHAT CAN USERS DO WITH THIS PLATFORM?

### 1. Clinical Data Extraction
- Extract structured data from unstructured clinical notes
- Convert narrative text → JSON/CSV
- Works for ANY clinical domain (universal)
- Supervised or unsupervised (labels optional)

### 2. Clinical Decision Support
- Query medical knowledge bases (RAG)
- Calculate clinical scores and metrics (Functions)
- Get guideline-based recommendations (Extras)
- Structured output for downstream applications

### 3. Research Data Collection
- Extract research variables from EHR notes
- Standardize clinical documentation
- Create datasets for clinical research
- Batch processing for large cohorts

### 4. Quality Improvement
- Monitor documentation completeness
- Track clinical metrics over time
- Identify documentation gaps
- Audit trail for compliance

### 5. Custom Clinical Applications
- Build task-specific extractors (no coding!)
- Create clinical calculators
- Develop knowledge bases for specialties
- Integrate with existing systems (API-ready)

### 6. Education and Training
- Demonstrate clinical reasoning
- Teach structured documentation
- Create annotated datasets
- Training data generation

---

## 🔐 SECURITY AND COMPLIANCE

### PII Protection
- Built-in PII redaction (HIPAA-aware)
- Multiple redaction strategies
- Configurable entity types
- De-identification support

### Data Privacy
- Local processing (no data sent except to chosen LLM)
- User controls LLM provider
- Optional local models
- Audit logging

### Access Control
- API key management
- Configuration persistence
- Secure credential storage

---

## 📈 PERFORMANCE CHARACTERISTICS

### Throughput
- **ASYNC Tool Execution:** 60-75% faster than sequential
- **Batch Processing:** Configurable batch sizes
- **Parallel Processing:** Multiple records simultaneously (future)

### Scalability
- Vector database: Handles 100K+ document chunks
- Function registry: Unlimited functions
- Extras: Unlimited supplementary hints
- Batch size: Limited only by memory

### Latency
- Single extraction: 2-10 seconds (depends on LLM)
- RAG query: 0.1-0.5 seconds
- Function call: 0.05-0.2 seconds
- Extras query: 0.05-0.1 seconds

---

## 🛠️ TECHNICAL STACK

### Core Dependencies
- **Python:** 3.8+
- **LLM Libraries:**
  - `openai` (OpenAI, Azure)
  - `anthropic` (Claude)
- **RAG:**
  - `faiss-cpu` (vector database - Facebook AI Similarity Search)
  - `sentence-transformers` (embeddings)
  - `torch` (transformer models)
- **NLP:**
  - `spacy` (NER for PII)
  - `transformers` (embeddings)
- **UI:**
  - `gradio` (web interface)
- **Data:**
  - `pandas` (data processing)
  - `openpyxl` (Excel support)
- **Utilities:**
  - `pydantic` (validation)
  - `asyncio` (async execution)

### Optional Dependencies
- **Local LLMs:**
  - `unsloth` (4-bit quantized models - LLaMA, Mistral, etc.)
  - `unsloth_zoo` (pre-trained model zoo)
  - `xformers` (memory-efficient transformers)
- Custom embedding models
- Additional NER models

---

## 📁 PROJECT STRUCTURE

```
clinannotate/
├── core/                          # Core system components
│   ├── agent_system.py           # STRUCTURED mode agent
│   ├── agentic_agent.py          # ADAPTIVE mode agent
│   ├── agent_factory.py          # Agent creation
│   ├── llm_manager.py            # LLM provider interface
│   ├── rag_engine.py             # RAG/vector search
│   ├── function_registry.py      # Computational tools
│   ├── extras_manager.py         # Supplementary hints
│   ├── json_parser.py            # JSON extraction
│   ├── regex_preprocessor.py     # Text normalization
│   ├── pii_redactor.py           # PII removal
│   ├── prompt_templates.py       # Prompt management
│   ├── output_handler.py         # Result export
│   ├── model_config.py           # LLM configuration
│   ├── app_state.py              # State management
│   ├── config_persistence.py     # Config save/load
│   ├── data_processor.py         # Data input handling
│   ├── growth_calculators.py     # Clinical calculators
│   └── logging_config.py         # Logging setup
│
├── ui/                            # Gradio interface
│   ├── config_tab.py             # LLM configuration
│   ├── data_tab.py               # Data upload/mapping
│   ├── prompt_tab.py             # Prompt templates
│   ├── rag_tab.py                # Document management
│   ├── processing_tab.py         # Batch processing
│   └── playground_tab.py         # Single-text testing
│
├── functions/                     # Function definitions (JSON)
│   ├── calculate_bmi.json
│   ├── calculate_weight_change.json
│   ├── calculate_growth_percentile.json
│   └── ... (user-extensible)
│
├── extras/                        # Supplementary hints (JSON)
│   ├── z_score_percentile.json
│   ├── malnutrition_guidelines.json
│   └── ... (user-extensible)
│
├── evaluation/                    # Evaluation tools
│   ├── metrics.py
│   ├── evaluator.py
│   └── datasets/
│
├── cdc_data/                      # Clinical reference data
│   └── growth_charts/
│
├── annotate.py                    # Main application entry
├── ARCHITECTURE.md                # This file!
├── README.md                      # User guide
└── requirements.txt               # Dependencies
```

---

## 🚀 FUTURE ENHANCEMENTS

### Planned Features
- Database output (SQL, MongoDB)
- Real-time API endpoints
- Parallel batch processing
- Additional LLM providers
- Custom evaluation metrics
- Multi-language support
- Advanced PII detection
- Workflow automation

---

## 📚 DOCUMENTATION

- **README.md:** Quick start guide
- **ARCHITECTURE.md:** This document - complete technical architecture
- **COMPLETE_USER_GUIDE.md:** Comprehensive user documentation
- **AGENTIC_USER_GUIDE.md:** ADAPTIVE mode detailed guide
- **PIPELINE_ARCHITECTURE.md:** Pipeline flow documentation
- **examples/:** Usage examples for different tasks

---

## 🎯 DESIGN PHILOSOPHY

1. **Universal First:** No task hardcoding, adapt to anything
2. **Autonomous Intelligence:** LLM orchestrates tools intelligently
3. **User Empowerment:** Users extend without coding
4. **Performance Matters:** ASYNC, caching, optimization
5. **Robust & Reliable:** Multiple fallbacks, error handling
6. **Clear & Intuitive:** STRUCTURED vs ADAPTIVE naming
7. **Production Ready:** Logging, validation, compliance
8. **Open & Extensible:** Plugin architecture

---

**Document Version:** 1.0.0
**Last Updated:** 2024-01-15
**Architecture Version:** 1.0.0
