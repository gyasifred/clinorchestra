# ClinOrchestra SDK Guide v1.0.0

**Programmatic Access to the Universal Clinical Data Extraction Platform**

This guide shows how to use ClinOrchestra as a Python SDK for programmatic clinical data extraction, instead of using the web UI. Perfect for integrating ClinOrchestra into your data pipelines, scripts, and applications.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Basic Usage](#basic-usage)
4. [Advanced Usage](#advanced-usage)
5. [Execution Modes](#execution-modes)
6. [Tool Configuration](#tool-configuration)
7. [Batch Processing](#batch-processing)
8. [Performance Optimization](#performance-optimization)
9. [Complete Examples](#complete-examples)
10. [API Reference](#api-reference)

---

## Quick Start

### Minimal Example (STRUCTURED Mode)

```python
from core.app_state import AppState
from core.agent_system import ExtractionAgent

# Initialize app state
app_state = AppState()

# Configure LLM
app_state.set_model_config({
    'provider': 'openai',
    'model_name': 'gpt-4o-mini',
    'api_key': 'your-api-key',
    'temperature': 0.01,
    'max_tokens': 4096
})

# Configure prompt and schema
app_state.set_prompt_config({
    'main_prompt': "Extract patient diagnosis and medications from the clinical text.",
    'json_schema': {
        'diagnosis': {'type': 'string', 'required': True},
        'medications': {'type': 'array', 'required': True}
    }
})

# Get LLM manager (auto-initialized from app_state)
llm_manager = app_state.get_llm_manager()

# Create agent
agent = ExtractionAgent(
    llm_manager=llm_manager,
    rag_engine=None,  # Optional
    extras_manager=app_state.get_extras_manager(),
    function_registry=app_state.get_function_registry(),
    regex_preprocessor=app_state.get_regex_preprocessor(),
    app_state=app_state
)

# Extract from clinical text
clinical_text = "Patient presents with Type 2 DM. Current meds: Metformin 1000mg BID, Lisinopril 10mg daily."

result = agent.extract(clinical_text)

print(result)
# Output: {'diagnosis': 'Type 2 Diabetes Mellitus', 'medications': ['Metformin 1000mg twice daily', 'Lisinopril 10mg daily']}
```

---

## Core Concepts

### 1. AppState - Central Configuration Manager

`AppState` manages all configurations and component initialization:

```python
from core.app_state import AppState

app_state = AppState()

# Components auto-initialized on first access:
llm_manager = app_state.get_llm_manager()           # LLM interface
rag_engine = app_state.get_rag_engine()             # RAG retrieval
extras_manager = app_state.get_extras_manager()     # Clinical hints
function_registry = app_state.get_function_registry()  # Medical calculations
regex_preprocessor = app_state.get_regex_preprocessor()  # Text normalization
```

### 2. Agent Types

**ExtractionAgent (STRUCTURED Mode)**
- 4-stage predictable pipeline
- Best for production, reliability
- Automatic tool orchestration

**AgenticAgent (ADAPTIVE Mode)**
- Iterative autonomous loop
- Best for complex, evolving tasks
- LLM decides tool usage

### 3. Configuration Objects

```python
# Model Configuration
model_config = {
    'provider': 'openai',  # 'anthropic', 'google', 'azure', 'local'
    'model_name': 'gpt-4o-mini',
    'api_key': 'your-key',
    'temperature': 0.01,
    'max_tokens': 4096
}

# Prompt Configuration
prompt_config = {
    'main_prompt': "Your extraction instructions...",
    'minimal_prompt': "Simplified fallback prompt...",  # Optional
    'json_schema': {...},  # Required
    'rag_prompt': "Optional RAG refinement prompt...",  # Optional
    'rag_query_fields': ['field1', 'field2']  # Optional
}

# Data Configuration
data_config = {
    'text_column': 'clinical_text',
    'has_labels': True,
    'label_column': 'diagnosis',
    'label_mapping': {1: "Diabetes", 2: "Hypertension"},  # Optional
    'prompt_input_columns': ['age', 'gender', 'admission_type'],  # Optional - NEW in v1.0.0
    'enable_phi_redaction': False,  # Optional
    'enable_pattern_normalization': True  # Optional
}

# Agentic Configuration (for ADAPTIVE mode)
agentic_config = {
    'enabled': True,
    'max_iterations': 10,
    'max_tool_calls': 50
}

# Optimization Configuration
optimization_config = {
    'llm_cache_enabled': True,  # 400x faster for repeated queries
    'use_parallel_processing': True,
    'use_batch_preprocessing': True,
    'max_parallel_workers': 5
}
```

---

## Basic Usage

### Example 1: Simple Extraction (No Tools)

```python
from core.app_state import AppState
from core.agent_system import ExtractionAgent

# Setup
app_state = AppState()

app_state.set_model_config({
    'provider': 'openai',
    'model_name': 'gpt-4o-mini',
    'api_key': 'your-key'
})

app_state.set_prompt_config({
    'main_prompt': "Extract vital signs from the clinical note.",
    'json_schema': {
        'blood_pressure': {'type': 'string'},
        'heart_rate': {'type': 'number'},
        'temperature': {'type': 'number'}
    }
})

# Create agent
agent = ExtractionAgent(
    llm_manager=app_state.get_llm_manager(),
    rag_engine=None,
    extras_manager=None,
    function_registry=None,
    regex_preprocessor=None,
    app_state=app_state
)

# Extract
text = "Vitals: BP 120/80, HR 72 bpm, Temp 98.6°F"
result = agent.extract(text)
print(result)
```

### Example 2: With Label Context

```python
# Configure label mapping
app_state.set_data_config({
    'has_labels': True,
    'label_column': 'diagnosis_id',
    'label_mapping': {
        1: "Type 2 Diabetes Mellitus - chronic metabolic disorder...",
        2: "Hypertension - elevated blood pressure ≥130/80..."
    }
})

# Extract with label
result = agent.extract(
    clinical_text="Patient presents with elevated HbA1c of 8.2%",
    label_value=1  # Will inject label context into prompt
)
```

### Example 3: With Multi-Column Prompt Variables (NEW v1.0.0)

```python
import pandas as pd

# Load your dataset
df = pd.read_csv('patients.csv')
# Columns: patient_id, age, gender, admission_type, diagnosis, clinical_text

# Configure which columns to pass as prompt variables
app_state.set_data_config({
    'text_column': 'clinical_text',
    'prompt_input_columns': ['patient_id', 'age', 'gender', 'admission_type']
})

# Update prompt to use variables
app_state.set_prompt_config({
    'main_prompt': """
    Extract clinical information for:
    Patient ID: {patient_id}
    Age: {age}
    Gender: {gender}
    Admission Type: {admission_type}

    Clinical Text: {clinical_text}

    Extract: {...}
    """,
    'json_schema': {...}
})

# Process row with prompt variables
row = df.iloc[0]
result = agent.extract(
    clinical_text=row['clinical_text'],
    prompt_variables={
        'patient_id': row['patient_id'],
        'age': row['age'],
        'gender': row['gender'],
        'admission_type': row['admission_type']
    }
)
```

---

## Advanced Usage

### Example 4: With RAG (Clinical Guidelines)

```python
from core.app_state import AppState
from core.agent_system import ExtractionAgent

app_state = AppState()

# Configure model
app_state.set_model_config({
    'provider': 'openai',
    'model_name': 'gpt-4o-mini',
    'api_key': 'your-key'
})

# Configure RAG
app_state.set_rag_config({
    'enabled': True,
    'documents': [
        '/path/to/diabetes_guidelines.pdf',
        '/path/to/hypertension_guidelines.pdf',
        'https://example.com/clinical-guideline.pdf'
    ],
    'embedding_model': 'sentence-transformers/all-mpnet-base-v2',
    'chunk_size': 512,
    'k_value': 3  # Return top 3 most relevant chunks
})

# Configure prompt with RAG
app_state.set_prompt_config({
    'main_prompt': "Extract diabetes management plan based on clinical text and guidelines.",
    'json_schema': {
        'hba1c_target': {'type': 'number'},
        'medication_plan': {'type': 'array'},
        'lifestyle_interventions': {'type': 'array'}
    },
    'rag_prompt': "Refine the HbA1c target based on retrieved guidelines",  # Optional refinement
    'rag_query_fields': ['hba1c_target']  # Which fields to refine with RAG
})

# Create agent with RAG
agent = ExtractionAgent(
    llm_manager=app_state.get_llm_manager(),
    rag_engine=app_state.get_rag_engine(),  # RAG enabled
    extras_manager=app_state.get_extras_manager(),
    function_registry=app_state.get_function_registry(),
    regex_preprocessor=app_state.get_regex_preprocessor(),
    app_state=app_state
)

# Extract with RAG retrieval
text = "65-year-old male with T2DM, HbA1c 8.5%, eGFR 45, no hypoglycemia history."
result = agent.extract(text)
# RAG will retrieve relevant guideline sections and use them during extraction
```

### Example 5: With Custom Functions

```python
from core.function_registry import FunctionRegistry

# Initialize function registry
registry = FunctionRegistry()

# Register custom medical calculation
registry.register_function(
    name="calculate_bmi",
    code="""
def calculate_bmi(weight_kg, height_m):
    '''Calculate BMI from weight and height'''
    if height_m <= 0:
        return None
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 2)
""",
    description="Calculate Body Mass Index",
    parameters={
        "weight_kg": {"type": "number", "description": "Weight in kg"},
        "height_m": {"type": "number", "description": "Height in meters"}
    },
    return_type="number"
)

# Use in agent
agent = ExtractionAgent(
    llm_manager=app_state.get_llm_manager(),
    rag_engine=None,
    extras_manager=None,
    function_registry=registry,  # Custom functions
    regex_preprocessor=None,
    app_state=app_state
)

# LLM can now call calculate_bmi during extraction
text = "Patient weight 75kg, height 1.75m"
result = agent.extract(text)
# Agent may call calculate_bmi(75, 1.75) and include BMI in output
```

### Example 6: With Clinical Hints (Extras)

```python
from core.extras_manager import ExtrasManager

# Initialize extras manager
extras = ExtrasManager()

# Add clinical knowledge
extras.add_extra(
    extra_type="guideline",
    content="WHO Growth Standards: Z-scores <-2 SD indicate wasted, >+2 SD indicate overweight",
    metadata={"category": "pediatrics", "subcategory": "growth"},
    name="WHO Growth Standards"
)

extras.add_extra(
    extra_type="criteria",
    content="ASPEN Malnutrition requires ≥2 of: insufficient intake, weight loss, loss of muscle/fat, edema",
    metadata={"category": "malnutrition"},
    name="ASPEN Criteria"
)

# Use in agent
agent = ExtractionAgent(
    llm_manager=app_state.get_llm_manager(),
    rag_engine=None,
    extras_manager=extras,  # Clinical hints
    function_registry=None,
    regex_preprocessor=None,
    app_state=app_state
)

# Relevant extras will be automatically matched and injected into prompts
```

### Example 7: With Text Normalization (Patterns)

```python
from core.regex_preprocessor import RegexPreprocessor

# Initialize preprocessor
preprocessor = RegexPreprocessor()

# Load patterns from directory (JSON files)
preprocessor.load_patterns_from_directory('./patterns')

# Or add patterns programmatically
preprocessor.add_pattern(
    name="normalize_blood_pressure",
    pattern=r"BP:?\s*(\d{2,3})/(\d{2,3})",
    replacement=r"Blood pressure \1/\2 mmHg",
    description="Normalize BP format",
    enabled=True
)

# Use in agent
agent = ExtractionAgent(
    llm_manager=app_state.get_llm_manager(),
    rag_engine=None,
    extras_manager=None,
    function_registry=None,
    regex_preprocessor=preprocessor,  # Text normalization
    app_state=app_state
)

# Text will be normalized before extraction
text = "BP: 140/90, HR: 85"
# Normalized to: "Blood pressure 140/90 mmHg, heart rate 85 bpm"
result = agent.extract(text)
```

---

## Execution Modes

### STRUCTURED Mode (Default - Production Ready)

4-stage predictable pipeline:

```python
from core.agent_system import ExtractionAgent

agent = ExtractionAgent(
    llm_manager=llm_manager,
    rag_engine=rag_engine,
    extras_manager=extras_manager,
    function_registry=function_registry,
    regex_preprocessor=regex_preprocessor,
    app_state=app_state
)

result = agent.extract(clinical_text)
```

**Stages:**
1. **Task Analysis**: LLM determines which tools are needed
2. **Tool Execution**: Functions, RAG, extras executed in parallel (async)
3. **Extraction**: Generate structured JSON output
4. **RAG Refinement** (optional): Enhance specific fields with guidelines

### ADAPTIVE Mode (Iterative Autonomous)

Continuous loop for complex tasks:

```python
from core.agentic_agent import AgenticAgent

# Enable agentic mode
app_state.set_agentic_config({
    'enabled': True,
    'max_iterations': 10,
    'max_tool_calls': 50
})

# Create agentic agent
agent = AgenticAgent(
    llm_manager=llm_manager,
    rag_engine=rag_engine,
    extras_manager=extras_manager,
    function_registry=function_registry,
    regex_preprocessor=regex_preprocessor,
    app_state=app_state
)

result = agent.extract(clinical_text)
```

**Flow:**
1. LLM analyzes text
2. Decides which tools to call (if any)
3. Executes tools in parallel (async)
4. Analyzes results
5. Iterates or completes extraction
6. Returns final JSON

---

## Tool Configuration

### Complete Tool Setup

```python
from core.app_state import AppState
from core.agent_system import ExtractionAgent

app_state = AppState()

# 1. Model Configuration
app_state.set_model_config({
    'provider': 'openai',
    'model_name': 'gpt-4o-mini',
    'api_key': 'your-key',
    'temperature': 0.01,
    'max_tokens': 4096
})

# 2. Prompt Configuration
app_state.set_prompt_config({
    'main_prompt': """
    Extract comprehensive clinical information from the text.

    Clinical Text: {clinical_text}
    Label Context: {label_context}

    Available Functions: {function_outputs}
    Retrieved Guidelines: {rag_outputs}
    Clinical Hints: {extras_outputs}

    Extract according to schema: {json_schema_instructions}
    """,
    'json_schema': {
        'diagnosis': {'type': 'string', 'required': True},
        'severity': {'type': 'string'},
        'vital_signs': {'type': 'object'},
        'lab_results': {'type': 'object'},
        'medications': {'type': 'array'}
    },
    'rag_prompt': "Enhance diagnosis field with guideline evidence",
    'rag_query_fields': ['diagnosis', 'medications']
})

# 3. RAG Configuration
app_state.set_rag_config({
    'enabled': True,
    'documents': ['/path/to/guidelines.pdf'],
    'k_value': 3
})

# 4. Data Configuration
app_state.set_data_config({
    'has_labels': True,
    'label_column': 'diagnosis_id',
    'label_mapping': {...},
    'prompt_input_columns': ['age', 'gender'],  # NEW v1.0.0
    'enable_pattern_normalization': True
})

# 5. Optimization Configuration
app_state.set_optimization_config({
    'llm_cache_enabled': True,
    'use_parallel_processing': True,
    'max_parallel_workers': 5
})

# Get all components (auto-initialized)
llm_manager = app_state.get_llm_manager()
rag_engine = app_state.get_rag_engine()
extras_manager = app_state.get_extras_manager()
function_registry = app_state.get_function_registry()
regex_preprocessor = app_state.get_regex_preprocessor()

# Create fully-equipped agent
agent = ExtractionAgent(
    llm_manager=llm_manager,
    rag_engine=rag_engine,
    extras_manager=extras_manager,
    function_registry=function_registry,
    regex_preprocessor=regex_preprocessor,
    app_state=app_state
)

# Extract with all tools available
result = agent.extract(
    clinical_text="Patient presents with...",
    label_value=1,
    prompt_variables={'age': 65, 'gender': 'M'}  # NEW v1.0.0
)
```

---

## Batch Processing

### Example: Process Multiple Patients

```python
import pandas as pd
from core.app_state import AppState
from core.agent_system import ExtractionAgent

# Load dataset
df = pd.read_csv('patients.csv')

# Setup
app_state = AppState()
app_state.set_model_config({...})
app_state.set_prompt_config({...})

# Enable optimizations
app_state.set_optimization_config({
    'llm_cache_enabled': True,  # Cache repeated extractions
    'use_parallel_processing': False,  # Set True for cloud APIs
    'use_batch_preprocessing': True  # Preprocess all texts at once
})

# Create agent
agent = ExtractionAgent(
    llm_manager=app_state.get_llm_manager(),
    rag_engine=None,
    extras_manager=app_state.get_extras_manager(),
    function_registry=app_state.get_function_registry(),
    regex_preprocessor=app_state.get_regex_preprocessor(),
    app_state=app_state
)

# Process all rows
results = []
for idx, row in df.iterrows():
    try:
        result = agent.extract(
            clinical_text=row['clinical_text'],
            label_value=row.get('diagnosis_id'),
            prompt_variables={
                'patient_id': row['patient_id'],
                'age': row['age'],
                'gender': row['gender']
            }
        )
        results.append(result)
        print(f"✅ Processed row {idx+1}/{len(df)}")
    except Exception as e:
        print(f"❌ Error on row {idx}: {e}")
        results.append({'error': str(e)})

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('extraction_results.csv', index=False)
```

### Example: Parallel Batch Processing

```python
from concurrent.futures import ThreadPoolExecutor
from functools import partial

def extract_row(row, agent):
    """Extract from a single row"""
    return agent.extract(
        clinical_text=row['clinical_text'],
        label_value=row.get('diagnosis_id'),
        prompt_variables={
            'age': row.get('age'),
            'gender': row.get('gender')
        }
    )

# Enable parallel processing
app_state.set_optimization_config({
    'use_parallel_processing': True,
    'max_parallel_workers': 5  # Adjust based on API rate limits
})

# Process in parallel
extract_func = partial(extract_row, agent=agent)

with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(
        lambda row: extract_func(row),
        [row for _, row in df.iterrows()]
    ))

print(f"Processed {len(results)} rows in parallel")
```

---

## Performance Optimization

### 1. LLM Response Caching (400x Faster)

```python
# Enable caching (default: enabled)
app_state.set_optimization_config({
    'llm_cache_enabled': True,
    'llm_cache_db_path': 'cache/llm_responses.db'
})

# Identical prompts will be served from cache
# Perfect for processing similar clinical notes
```

### 2. Batch Text Preprocessing

```python
# Preprocess all texts at once (faster than row-by-row)
app_state.set_optimization_config({
    'use_batch_preprocessing': True
})

# Normalization patterns applied to entire dataset upfront
```

### 3. Model Profiles (Optimized Settings)

```python
# Automatically uses optimized settings for known models
app_state.set_optimization_config({
    'use_model_profiles': True
})

# Profiles for: GPT-4o, GPT-4o-mini, Claude Sonnet, Gemini, etc.
# Sets optimal temperature, max_tokens, etc.
```

### 4. Async Tool Execution

Both STRUCTURED and ADAPTIVE modes execute tools in parallel using async/await:

```python
# Tools run simultaneously (not sequentially)
# 60-75% faster for multi-tool extractions
# Enabled by default - no configuration needed
```

### 5. Cache Bypass (for Development)

```python
# Force fresh LLM calls (ignore cache)
app_state.set_optimization_config({
    'llm_cache_bypass': True
})

# Useful when testing prompt changes
```

---

## Complete Examples

### Example A: MIMIC-IV Diagnosis Extraction

```python
import pandas as pd
from core.app_state import AppState
from core.agent_system import ExtractionAgent

# Load MIMIC dataset
df = pd.read_csv('mimic_patients.csv')
# Columns: subject_id, hadm_id, consolidated_diagnosis_name, clinical_text, age, gender

# Setup
app_state = AppState()

# Model
app_state.set_model_config({
    'provider': 'openai',
    'model_name': 'gpt-4o-mini',
    'api_key': 'your-key'
})

# Prompt with multi-column variables
app_state.set_prompt_config({
    'main_prompt': """
    Extract comprehensive clinical evidence for:

    Patient: {subject_id} (Admission: {hadm_id})
    Demographics: {age}yo {gender}
    Primary Diagnosis: {consolidated_diagnosis_name}

    Clinical Text: {clinical_text}

    Extract all evidence supporting the diagnosis.
    """,
    'json_schema': {
        'evidence_summary': {'type': 'object', 'required': True},
        'symptoms': {'type': 'array', 'required': True},
        'vital_signs': {'type': 'object'},
        'lab_results': {'type': 'object'},
        'imaging_findings': {'type': 'array'},
        'medications': {'type': 'array'},
        'clinical_reasoning': {'type': 'string', 'required': True}
    }
})

# Label mapping for diagnosis context
diagnosis_mapping = {
    "Sepsis": "Life-threatening organ dysfunction from infection. Requires immediate antibiotics and hemodynamic support.",
    "Chest pain": "Cardiovascular symptom requiring careful evaluation to rule out MI, PE, or other emergencies.",
    # ... other diagnoses
}

app_state.set_data_config({
    'has_labels': True,
    'label_column': 'consolidated_diagnosis_name',
    'label_mapping': diagnosis_mapping,
    'prompt_input_columns': ['subject_id', 'hadm_id', 'age', 'gender', 'consolidated_diagnosis_name']
})

# RAG with clinical guidelines
app_state.set_rag_config({
    'enabled': True,
    'documents': [
        './guidelines/sepsis_guidelines.pdf',
        './guidelines/chest_pain_guidelines.pdf'
    ],
    'k_value': 3
})

# Create agent with all tools
agent = ExtractionAgent(
    llm_manager=app_state.get_llm_manager(),
    rag_engine=app_state.get_rag_engine(),
    extras_manager=app_state.get_extras_manager(),
    function_registry=app_state.get_function_registry(),
    regex_preprocessor=app_state.get_regex_preprocessor(),
    app_state=app_state
)

# Process dataset
results = []
for idx, row in df.iterrows():
    result = agent.extract(
        clinical_text=row['clinical_text'],
        label_value=row['consolidated_diagnosis_name'],
        prompt_variables={
            'subject_id': row['subject_id'],
            'hadm_id': row['hadm_id'],
            'age': row['age'],
            'gender': row['gender'],
            'consolidated_diagnosis_name': row['consolidated_diagnosis_name']
        }
    )
    results.append(result)
    print(f"Processed {idx+1}/{len(df)}")

# Save
pd.DataFrame(results).to_json('mimic_extractions.json', orient='records', indent=2)
```

### Example B: Malnutrition Assessment

```python
from core.app_state import AppState
from core.agentic_agent import AgenticAgent  # Using ADAPTIVE mode

app_state = AppState()

# Model
app_state.set_model_config({
    'provider': 'anthropic',
    'model_name': 'claude-3-5-sonnet-20241022',
    'api_key': 'your-key'
})

# Prompt
app_state.set_prompt_config({
    'main_prompt': """
    Assess pediatric malnutrition from clinical notes.
    Use WHO growth standards, CDC charts, and ASPEN criteria.
    Call functions to calculate z-scores and percentiles.
    """,
    'json_schema': {
        'malnutrition_present': {'type': 'boolean', 'required': True},
        'severity': {'type': 'string'},  # Mild, Moderate, Severe
        'anthropometric_data': {'type': 'object'},
        'growth_indicators': {'type': 'object'},
        'aspen_criteria_met': {'type': 'array'},
        'etiology': {'type': 'string'}
    }
})

# Enable ADAPTIVE mode
app_state.set_agentic_config({
    'enabled': True,
    'max_iterations': 15,
    'max_tool_calls': 50
})

# RAG with nutrition guidelines
app_state.set_rag_config({
    'enabled': True,
    'documents': ['./guidelines/who_growth_standards.pdf', './guidelines/aspen_malnutrition.pdf']
})

# Create ADAPTIVE agent
agent = AgenticAgent(
    llm_manager=app_state.get_llm_manager(),
    rag_engine=app_state.get_rag_engine(),
    extras_manager=app_state.get_extras_manager(),
    function_registry=app_state.get_function_registry(),
    regex_preprocessor=app_state.get_regex_preprocessor(),
    app_state=app_state
)

# Extract
text = """
4-year-old male, weight 12.5kg (down from 14kg 6 months ago).
Height 95cm. Poor oral intake for 3 months. Visible muscle wasting.
"""

result = agent.extract(text)
print(result)
```

---

## API Reference

### AppState Methods

```python
# Configuration
app_state.set_model_config(config: Dict)
app_state.set_prompt_config(config: Dict)
app_state.set_data_config(**kwargs)
app_state.set_rag_config(config: Dict)
app_state.set_agentic_config(config: Dict)
app_state.set_optimization_config(config: Dict)

# Component Access (auto-initializes on first call)
app_state.get_llm_manager() -> LLMManager
app_state.get_rag_engine() -> RAGEngine
app_state.get_extras_manager() -> ExtrasManager
app_state.get_function_registry() -> FunctionRegistry
app_state.get_regex_preprocessor() -> RegexPreprocessor

# Cache Management
app_state.invalidate_llm_cache()
app_state.clear_all_caches()
```

### ExtractionAgent Methods (STRUCTURED Mode)

```python
from core.agent_system import ExtractionAgent

agent = ExtractionAgent(
    llm_manager: LLMManager,
    rag_engine: Optional[RAGEngine],
    extras_manager: Optional[ExtrasManager],
    function_registry: Optional[FunctionRegistry],
    regex_preprocessor: Optional[RegexPreprocessor],
    app_state: AppState
)

# Main extraction method
result = agent.extract(
    clinical_text: str,
    label_value: Optional[Any] = None,
    prompt_variables: Optional[Dict[str, Any]] = None  # NEW v1.0.0
) -> Dict[str, Any]
```

### AgenticAgent Methods (ADAPTIVE Mode)

```python
from core.agentic_agent import AgenticAgent

agent = AgenticAgent(
    llm_manager: LLMManager,
    rag_engine: Optional[RAGEngine],
    extras_manager: Optional[ExtrasManager],
    function_registry: Optional[FunctionRegistry],
    regex_preprocessor: Optional[RegexPreprocessor],
    app_state: AppState
)

# Main extraction method (same signature as STRUCTURED)
result = agent.extract(
    clinical_text: str,
    label_value: Optional[Any] = None,
    prompt_variables: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]
```

### LLMManager Methods

```python
from core.llm_manager import LLMManager

llm = LLMManager(config: Dict)

# Generate response
response = llm.generate(
    prompt: str,
    max_tokens: Optional[int] = None
) -> str

# Generate with conversation history (for agentic mode)
response = llm.generate_with_messages(
    messages: List[Dict],
    max_tokens: Optional[int] = None
) -> str
```

### FunctionRegistry Methods

```python
from core.function_registry import FunctionRegistry

registry = FunctionRegistry()

# Register function
registry.register_function(
    name: str,
    code: str,
    description: str,
    parameters: Dict,
    return_type: str
) -> bool

# Execute function
success, result, message = registry.execute_function(
    name: str,
    **kwargs
)

# List available functions
functions = registry.list_functions() -> List[Dict]

# Load from directory
registry.load_from_directory(path: str)
```

### ExtrasManager Methods

```python
from core.extras_manager import ExtrasManager

extras = ExtrasManager()

# Add extra
extras.add_extra(
    extra_type: str,  # 'guideline', 'criteria', 'tip', 'reference', 'pattern'
    content: str,
    metadata: Dict,
    name: str
) -> bool

# Search by keywords
relevant = extras.search(keywords: List[str]) -> List[Dict]

# Load from directory
extras.load_from_directory(path: str)
```

### RAGEngine Methods

```python
from core.rag_engine import RAGEngine

rag = RAGEngine(config: Dict)

# Add documents
rag.add_documents(documents: List[str])  # Paths or URLs

# Retrieve
results = rag.retrieve(
    query: str,
    k: int = 3
) -> List[Dict]  # Returns: [{'text': ..., 'source': ..., 'score': ...}, ...]
```

### RegexPreprocessor Methods

```python
from core.regex_preprocessor import RegexPreprocessor

preprocessor = RegexPreprocessor()

# Add pattern
preprocessor.add_pattern(
    name: str,
    pattern: str,
    replacement: str,
    description: str,
    enabled: bool = True
)

# Preprocess text
normalized = preprocessor.preprocess(text: str) -> str

# Load from directory
preprocessor.load_patterns_from_directory(path: str)
```

---

## Error Handling

```python
from core.agent_system import ExtractionAgent

agent = ExtractionAgent(...)

try:
    result = agent.extract(clinical_text)

    # Check for errors in result
    if 'error' in result:
        print(f"Extraction failed: {result['error']}")
    else:
        # Success - use result
        print(f"Diagnosis: {result.get('diagnosis')}")

except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Tips and Best Practices

### 1. Start Simple, Add Complexity

```python
# Start with minimal configuration
agent = ExtractionAgent(
    llm_manager=llm_manager,
    rag_engine=None,  # Add later
    extras_manager=None,  # Add later
    function_registry=None,  # Add later
    regex_preprocessor=None,  # Add later
    app_state=app_state
)

# Once working, add tools incrementally
```

### 2. Use LLM Cache for Development

```python
# Cache saves money and time during development
app_state.set_optimization_config({
    'llm_cache_enabled': True
})

# Identical prompts won't hit API again
# Clear cache when changing prompts:
app_state.invalidate_llm_cache()
```

### 3. Test Single Extractions First

```python
# Test on one example before batch processing
result = agent.extract("Test clinical text...")
print(result)

# Verify output structure matches schema
# Then scale to full dataset
```

### 4. Handle Missing Data

```python
# Safely access optional fields
result = agent.extract(text)

diagnosis = result.get('diagnosis', 'Unknown')
medications = result.get('medications', [])
severity = result.get('severity')  # None if missing
```

### 5. Monitor Performance

```python
import time

start = time.time()
result = agent.extract(clinical_text)
duration = time.time() - start

print(f"Extraction took {duration:.2f} seconds")

# Optimize if too slow:
# - Enable caching
# - Use faster model (gpt-4o-mini vs gpt-4o)
# - Reduce RAG k_value
# - Simplify prompt
```

---

## Migration from UI to SDK

If you've been using the web UI and want to migrate to SDK:

```python
# UI Configuration → SDK Equivalents

# UI: Model Config Tab
app_state.set_model_config({
    'provider': 'openai',  # From dropdown
    'model_name': 'gpt-4o-mini',  # From dropdown
    'api_key': 'your-key',  # From input field
    'temperature': 0.01,  # From slider
    'max_tokens': 4096  # From input
})

# UI: Prompt Config Tab
app_state.set_prompt_config({
    'main_prompt': "...",  # From main prompt textarea
    'minimal_prompt': "...",  # From minimal prompt textarea
    'json_schema': {...},  # From schema editor
    'rag_prompt': "...",  # From RAG refinement prompt
    'rag_query_fields': ['field1']  # From checkboxes
})

# UI: Data Tab
app_state.set_data_config({
    'text_column': 'clinical_text',  # From dropdown
    'has_labels': True,  # From checkbox
    'label_column': 'diagnosis',  # From dropdown
    'prompt_input_columns': ['age', 'gender'],  # From checkboxes (NEW)
    'enable_pattern_normalization': True  # From checkbox
})

# UI: Tools Tab
app_state.set_rag_config({
    'enabled': True,  # From RAG toggle
    'documents': [...],  # From file uploads
    'k_value': 3  # From slider
})

# UI: Config Tab (Advanced)
app_state.set_agentic_config({
    'enabled': False,  # From "Use Agentic Mode" toggle
    'max_iterations': 10,  # From slider
    'max_tool_calls': 50  # From slider
})

app_state.set_optimization_config({
    'llm_cache_enabled': True,  # From checkbox
    'use_parallel_processing': True,  # Auto-detected in UI
    'max_parallel_workers': 5  # From slider
})
```

---

## Version History

**v1.0.0** (2025-11-13)
- Initial SDK documentation
- Multi-column prompt variables support
- STRUCTURED and ADAPTIVE modes
- Comprehensive tool configuration
- Batch processing examples
- Performance optimization guide

---

**ClinOrchestra SDK v1.0.0** - Universal Clinical Data Extraction Platform

For UI usage, see main [README.md](README.md)

For architecture details, see [ARCHITECTURE.md](ARCHITECTURE.md)
