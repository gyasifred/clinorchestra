# ClinOrchestra Complete User Guide
## Comprehensive Configuration Guide for All Tabs

**Author:** Frederick Gyasi (gyasifred)
**Institution:** Medical University of South Carolina, Biomedical Informatics Center
**Version:** 1.0.0
**Date:** 2025-11-05

**🎯 Platform:** Universal clinical data extraction & orchestration system - works for ANY clinical task

---

## Table of Contents

1. [Introduction](#introduction)
2. [Tab 1: Model Configuration](#tab-1-model-configuration)
3. [Tab 2: Prompt Configuration](#tab-2-prompt-configuration)
4. [Tab 3: Data Configuration](#tab-3-data-configuration)
5. [Tab 4: Regex Patterns](#tab-4-regex-patterns)
6. [Tab 5: Extras (Hints)](#tab-5-extras-hints)
7. [Tab 6: Custom Functions](#tab-6-custom-functions)
8. [Tab 7: RAG (Retrieval)](#tab-7-rag-retrieval)
9. [Tab 8: Playground (Testing)](#tab-8-playground-testing)
10. [Tab 9: Processing (Batch Execution)](#tab-9-processing-batch-execution)
11. [Configuration Files Reference](#configuration-files-reference)
12. [Complete Workflow Examples](#complete-workflow-examples)

---

## Introduction

**🎯 IMPORTANT:** This guide uses malnutrition and diabetes as **illustrative examples**. ClinOrchestra is a **universal system** that works for **ANY clinical extraction task** - not just the examples shown! The examples help you understand the framework; apply the same principles to YOUR specific use case (sepsis, AKI, medications, oncology, etc.).

ClinOrchestra provides **9 configuration tabs** for complete control over clinical data extraction:

```
┌─────────────────────────────────────────────────────────────┐
│ TAB WORKFLOW                                                │
├─────────────────────────────────────────────────────────────┤
│ 1. Model Configuration    → Set LLM provider & execution mode│
│ 2. Prompt Configuration   → Define YOUR extraction task     │
│ 3. Data Configuration     → Load input data                 │
│ 4. Regex Patterns         → Text normalization rules        │
│ 5. Extras (Hints)         → Task-specific hints for YOUR task│
│ 6. Custom Functions       → Medical calculations YOU need   │
│ 7. RAG                    → Clinical guidelines for YOUR domain│
│ 8. Playground             → Test YOUR extractions           │
│ 9. Processing             → Batch process YOUR dataset      │
└─────────────────────────────────────────────────────────────┘
```

**Configuration Methods:**
- ✅ **UI**: Interactive Gradio interface (recommended for beginners)
- ✅ **YAML**: Human-readable configuration files
- ✅ **JSON**: Machine-readable configuration files
- ✅ **Python API**: Programmatic configuration

---

## Tab 1: Model Configuration

**Purpose:** Configure LLM provider, model settings, and execution mode (Classic vs Agentic)

### UI Configuration

**Location:** First tab in the interface

**Settings:**

#### Provider Settings
```
Provider: [Dropdown]
├─ openai
├─ anthropic
├─ google
├─ azure
└─ local

Model: [Dropdown - changes based on provider]
├─ OpenAI: gpt-4o, gpt-4-turbo, gpt-3.5-turbo
├─ Anthropic: claude-3-opus, claude-3-sonnet, claude-3-haiku
├─ Google: gemini-pro, gemini-1.5-pro
└─ Local: Any Unsloth-compatible model

API Key: [Password field]
[Load from Environment] button
```

#### Model Parameters
```
Temperature: [Slider 0.0-2.0]
├─ 0.0 = Deterministic (recommended for clinical tasks)
├─ 0.5 = Balanced
└─ 1.0+ = Creative (not recommended for extraction)

Max Output Tokens: [Number 100-16000]
├─ Simple tasks: 1000-2000
├─ Complex tasks: 4000-8000
└─ Very detailed: 8000-16000

Model Type: [Dropdown]
├─ chat (recommended)
└─ completion (legacy)
```

#### Execution Mode Settings ⭐ NEW!
```
[✓] Enable Agentic Mode (AgenticAgent v1.0.0)
    ├─ OFF: Uses Classic Mode (ExtractionAgent v1.0.2)
    │   └─ Fixed 4-stage pipeline
    └─ ON: Uses Agentic Mode
        └─ Continuous loop with async tool calling (60-75% faster)

Max Iterations: [Number 5-100]
├─ Default: 20
├─ Simple tasks: 10-15
└─ Complex tasks: 30-50

Max Tool Calls: [Number 10-200]
├─ Default: 50
├─ Simple tasks: 20-30
└─ Complex tasks: 100+
```

#### Provider-Specific Settings

**Azure OpenAI:**
```
Azure Endpoint: [Text]
├─ Example: https://your-resource.openai.azure.com/

Deployment Name: [Text]
├─ Example: gpt-4o-deployment
```

**Google AI:**
```
Project ID: [Text]
├─ Example: my-gcp-project-123
```

**Local Models:**
```
Model Path: [Text]
├─ HuggingFace: unsloth/llama-3-8b-Instruct
├─ Local path: /path/to/model

Max Sequence Length: [Number 512-32768]
├─ Default: 16384
├─ Model context window limit

Quantization: [Dropdown]
├─ 4bit (recommended for memory efficiency)
└─ none (full precision)

GPU Layers: [Slider -1 to 100]
├─ -1 = Use all GPU
├─ 0 = CPU only
└─ Custom = Partial GPU
```

### YAML Configuration

**File:** `config/model_config.yaml`

```yaml
# Model Configuration
model:
  # Provider settings
  provider: "openai"  # openai | anthropic | google | azure | local
  model_name: "gpt-4o"

  # API credentials (or set environment variables)
  api_key: "${OPENAI_API_KEY}"  # Can use env var

  # Model parameters
  temperature: 0.0  # 0.0-2.0, use 0.0 for clinical tasks
  max_tokens: 4000  # Maximum output tokens
  model_type: "chat"  # chat | completion

  # Execution mode settings
  agentic_mode:
    enabled: false  # true = Agentic mode, false = Classic mode
    max_iterations: 20  # Max conversation iterations (5-100)
    max_tool_calls: 50  # Max total tool calls (10-200)
    iteration_logging: true  # Log each iteration
    tool_call_logging: true  # Log each tool call

  # Provider-specific settings (optional)
  azure:
    endpoint: "https://your-resource.openai.azure.com/"
    deployment: "gpt-4o-deployment"

  google:
    project_id: "my-gcp-project"

  local:
    model_path: "unsloth/llama-3-8b-Instruct"
    max_seq_length: 16384
    quantization: "4bit"
    gpu_layers: -1
```

### JSON Configuration

**File:** `config/model_config.json`

```json
{
  "provider": "openai",
  "model_name": "gpt-4o",
  "api_key": "${OPENAI_API_KEY}",
  "temperature": 0.0,
  "max_tokens": 4000,
  "model_type": "chat",
  "agentic_mode": {
    "enabled": false,
    "max_iterations": 20,
    "max_tool_calls": 50,
    "iteration_logging": true,
    "tool_call_logging": true
  },
  "azure": {
    "endpoint": "https://your-resource.openai.azure.com/",
    "deployment": "gpt-4o-deployment"
  }
}
```

### Python API

```python
from core.app_state import AppState
from core.model_config import ModelConfig

app_state = AppState()

# Configure model
model_config = ModelConfig(
    provider="openai",
    model_name="gpt-4o",
    api_key="your-api-key",  # or set OPENAI_API_KEY env var
    temperature=0.0,
    max_tokens=4000,
    model_type="chat"
)

app_state.set_model_config(model_config)

# Configure execution mode
app_state.set_agentic_config(
    enabled=True,  # Enable agentic mode
    max_iterations=20,
    max_tool_calls=50
)
```

---

## Tab 2: Prompt Configuration

**Purpose:** Define your extraction task, JSON schema, and prompts (main, minimal, RAG refinement)

### UI Configuration

**Location:** Second tab "Prompt Configuration"

#### Load Pre-built Templates

```
Template Selector: [Dropdown]
├─ Malnutrition Assessment (Pediatric)
├─ Diabetes Extraction
├─ Blank Template (custom)
└─ More templates...

[Load Template] button
```

**What Loading Does:**
- Populates Main Prompt with task-specific instructions
- Fills JSON Schema with relevant fields
- Sets RAG Refinement Prompt if applicable
- Adds example fields to schema table

#### Main Prompt (Primary Extraction)

```
Main Prompt: [Large text area]
```

**Template Variables Available:**
- `{clinical_text}` - The clinical text to analyze
- `{label_context}` - Ground truth diagnosis/label
- `{rag_outputs}` - Retrieved guidelines (if RAG enabled)
- `{function_outputs}` - Medical calculation results
- `{extras_outputs}` - Matched hints

**Example Main Prompt:**
```
You are a board-certified pediatric dietitian performing malnutrition assessment.

**GROUND TRUTH DIAGNOSIS:**
{label_context}

**CLINICAL TEXT:**
{clinical_text}

**TASK:**
Extract comprehensive malnutrition data including:
1. Anthropometric measurements with dates
2. WHO/ASPEN classification
3. Temporal trends
4. Clinical symptoms
5. Management plan

**GUIDELINES REFERENCE:**
{rag_outputs}

**CALCULATED VALUES:**
{function_outputs}

**ASSESSMENT CRITERIA:**
{extras_outputs}

Output JSON matching the schema exactly.
```

#### Minimal Prompt (Optional)

```
[✓] Use Minimal Prompt
```

**Purpose:** Shorter prompt for token-limited scenarios

```
Minimal Prompt: [Text area]
```

**Example:**
```
Extract malnutrition data from clinical text.

Clinical Text: {clinical_text}
Label: {label_context}

Output JSON per schema.
```

#### JSON Schema Definition

**Two Methods:**

**Method 1: UI Table (Interactive)**
```
[Add Field] [Remove Selected] [Load from File]

Table:
┌─────┬────────────────┬────────┬──────────┬─────────────────┐
│ #   │ Field Name     │ Type   │ Required │ Description     │
├─────┼────────────────┼────────┼──────────┼─────────────────┤
│ 1   │ status         │ string │ Yes      │ Malnutrition... │
│ 2   │ measurements   │ array  │ Yes      │ Serial weights  │
│ 3   │ z_scores       │ object │ No       │ WHO z-scores    │
└─────┴────────────────┴────────┴──────────┴─────────────────┘
```

**Field Types Available:**
- `string` - Text field
- `number` - Numeric value
- `integer` - Whole number
- `boolean` - True/False
- `array` - List of items
- `object` - Nested structure
- `null` - Optional null value

**Method 2: Upload File**
```
[📁 Upload Schema File]
Accepts: .yaml, .yml, .json
```

#### RAG Refinement Prompt (Optional)

```
[✓] Enable RAG Refinement
```

**Purpose:** Refine specific fields with retrieved evidence

```
RAG Refinement Prompt: [Text area]
RAG Query Fields: [Multi-select]
```

**Example:**
```
Refine the following extraction fields using retrieved clinical guidelines:

Fields to refine: {fields_to_refine}
Current extraction: {current_extraction}
Retrieved evidence: {rag_outputs}

Provide refined values with evidence citations:
- Field: value (per [SOURCE])
```

**Select Fields for Refinement:**
```
[✓] status
[✓] severity
[ ] measurements
[✓] management_plan
```

### YAML Configuration

**File:** `config/prompts/malnutrition_prompt.yaml`

```yaml
# Prompt Configuration
prompt:
  name: "Pediatric Malnutrition Assessment"
  version: "1.0.5"
  description: "Comprehensive malnutrition extraction with WHO/ASPEN criteria"

  # Main extraction prompt
  main_prompt: |
    [TASK DESCRIPTION - Pediatric Malnutrition Clinical Assessment]

    You are a board-certified pediatric dietitian performing malnutrition assessment.

    **Z-SCORE INTERPRETATION CONVENTION:**
    - Percentile < 50th = NEGATIVE z-score
      * 3rd percentile = z-score -1.88
      * 10th percentile = z-score -1.28

    **WHO MALNUTRITION CLASSIFICATION:**
    - z < -3: SEVERE ACUTE MALNUTRITION (SAM)
    - -3 ≤ z < -2: MODERATE ACUTE MALNUTRITION (MAM)
    - -2 ≤ z < -1: MILD MALNUTRITION RISK

    **CLINICAL TEXT:**
    {clinical_text}

    **GROUND TRUTH:**
    {label_context}

    **RETRIEVED GUIDELINES:**
    {rag_outputs}

    **CALCULATED FUNCTIONS:**
    {function_outputs}

    **TASK-SPECIFIC HINTS:**
    {extras_outputs}

    Extract comprehensive malnutrition data following the schema.

  # Minimal prompt (optional)
  use_minimal: false
  minimal_prompt: |
    Extract malnutrition data from: {clinical_text}
    Label: {label_context}
    Output JSON per schema.

  # RAG refinement (optional)
  rag_refinement:
    enabled: true
    prompt: |
      Refine these malnutrition fields using retrieved evidence:

      Current extraction: {current_extraction}
      Retrieved evidence: {rag_outputs}

      Provide refined values with citations.

    query_fields:
      - "status"
      - "severity"
      - "management_plan"

  # JSON Schema
  schema:
    malnutrition_status:
      type: "string"
      required: true
      description: "Present/Absent with severity if present"
      enum: ["Absent", "Mild", "Moderate", "Severe", "Risk"]

    weight_measurements:
      type: "array"
      required: true
      description: "Serial weight measurements with dates"
      items:
        type: "object"
        properties:
          date:
            type: "string"
            format: "date"
          value_kg:
            type: "number"
          percentile:
            type: "number"
          zscore:
            type: "number"

    height_cm:
      type: "number"
      required: false
      description: "Height in centimeters"

    bmi:
      type: "object"
      required: false
      properties:
        value:
          type: "number"
        percentile:
          type: "number"
        zscore:
          type: "number"
        classification:
          type: "string"

    clinical_symptoms:
      type: "array"
      required: true
      description: "List of documented symptoms with dates"
      items:
        type: "object"
        properties:
          symptom:
            type: "string"
          onset_date:
            type: "string"
          severity:
            type: "string"

    aspen_criteria_met:
      type: "array"
      required: false
      description: "ASPEN diagnostic criteria met"
      items:
        type: "string"

    management_plan:
      type: "string"
      required: false
      description: "Nutritional management recommendations"
```

### JSON Configuration

**File:** `config/prompts/diabetes_prompt.json`

```json
{
  "prompt": {
    "name": "Diabetes Type Classification",
    "version": "1.0.0",
    "description": "Extract diabetes type, HbA1c, medications, complications",
    "main_prompt": "You are an endocrinologist extracting diabetes data.\n\nClinical Text: {clinical_text}\nLabel: {label_context}\n\nExtract: diabetes type, HbA1c values, medications, complications.\n\nGuidelines: {rag_outputs}\nCalculations: {function_outputs}\n\nOutput JSON per schema.",
    "use_minimal": false,
    "minimal_prompt": null,
    "rag_refinement": {
      "enabled": false,
      "prompt": null,
      "query_fields": []
    },
    "schema": {
      "diabetes_type": {
        "type": "string",
        "required": true,
        "description": "Type 1, Type 2, Gestational, or Other",
        "enum": ["Type 1", "Type 2", "Gestational", "MODY", "Secondary", "Unknown"]
      },
      "hba1c_values": {
        "type": "array",
        "required": true,
        "description": "Serial HbA1c measurements",
        "items": {
          "type": "object",
          "properties": {
            "date": {"type": "string"},
            "value": {"type": "number"},
            "unit": {"type": "string", "default": "%"}
          }
        }
      },
      "medications": {
        "type": "array",
        "required": true,
        "description": "List of diabetes medications",
        "items": {
          "type": "string"
        }
      },
      "complications": {
        "type": "array",
        "required": false,
        "description": "Documented diabetic complications",
        "items": {
          "type": "string"
        }
      }
    }
  }
}
```

### Python API

```python
from core.app_state import AppState

app_state = AppState()

# Define schema
schema = {
    "malnutrition_status": {
        "type": "string",
        "required": True,
        "description": "Presence and severity"
    },
    "weight_measurements": {
        "type": "array",
        "required": True,
        "description": "Serial weights with dates"
    }
}

# Set prompt configuration
app_state.set_prompt_config(
    main_prompt="""
    You are a pediatric dietitian performing malnutrition assessment.

    Clinical Text: {clinical_text}
    Label: {label_context}

    Guidelines: {rag_outputs}
    Calculations: {function_outputs}
    Hints: {extras_outputs}

    Extract malnutrition data per schema.
    """,
    minimal_prompt="Extract malnutrition: {clinical_text}",
    use_minimal=False,
    json_schema=schema,
    rag_prompt="Refine with evidence: {rag_outputs}"
)
```

---

## Tab 3: Data Configuration

**Purpose:** Load input dataset, specify columns, configure text processing

### UI Configuration

**Location:** Third tab "Data Configuration"

#### Input Data

```
Input File: [File picker]
├─ Accepts: .csv, .xlsx, .json, .parquet
└─ [Browse...] [Upload]

[Preview Data] button
```

**Data Preview:**
```
Showing 5 of 1,250 rows

┌─────┬──────────────────────────┬────────────┬─────────┐
│ id  │ clinical_text            │ diagnosis  │ age     │
├─────┼──────────────────────────┼────────────┼─────────┤
│ 001 │ 3-year-old with poor...  │ Malnutr... │ 3       │
│ 002 │ T2DM patient, HbA1c...   │ Diabetes   │ 52      │
└─────┴──────────────────────────┴────────────┴─────────┘
```

#### Column Mapping

```
Text Column: [Dropdown]
├─ clinical_text ← Select column containing clinical notes
└─ (Other columns shown)

[✓] Has Labels/Ground Truth
    Label Column: [Dropdown]
    ├─ diagnosis
    └─ (Other columns)

    Label Mapping (Optional):
    ├─ Map label values to descriptive text
    └─ Example: 1 → "Malnutrition Present"
```

#### De-identification Columns

```
De-ID Columns (Additional Context): [Multi-select]
├─ [✓] age
├─ [✓] gender
├─ [ ] mrn  (Don't include PHI!)
└─ [ ] admission_date
```

**Purpose:** Additional columns to append to clinical text as context

#### PHI Redaction (Optional)

```
[✓] Enable PHI Redaction

Entity Types to Redact: [Multi-select]
├─ [✓] PERSON (names)
├─ [✓] DATE_TIME (dates)
├─ [✓] LOCATION (addresses)
├─ [✓] PHONE_NUMBER
├─ [✓] EMAIL
├─ [✓] SSN
└─ [✓] MEDICAL_RECORD_NUMBER

Redaction Method: [Dropdown]
├─ Replace with tag (PERSON, DATE, etc.)
├─ Replace with asterisks (***)
└─ Remove completely

[✓] Save redacted text in output
```

#### Pattern Normalization

```
[✓] Enable Pattern Normalization
    ├─ Standardizes clinical text using regex patterns
    └─ See Regex Patterns tab for configuration

[✓] Save normalized text in output
```

### YAML Configuration

**File:** `config/data_config.yaml`

```yaml
# Data Configuration
data:
  # Input dataset
  input_file: "data/malnutrition_cases.csv"

  # Column mapping
  text_column: "clinical_text"
  has_labels: true
  label_column: "diagnosis"

  # Label mapping (optional)
  label_mapping:
    0: "Malnutrition Absent"
    1: "Malnutrition Present"
    2: "Malnutrition Risk"

  # De-identification columns (additional context)
  deid_columns:
    - "age"
    - "gender"
    - "weight_kg"

  additional_columns:
    - "encounter_id"
    - "facility"

  # PHI redaction
  phi_redaction:
    enabled: true
    entity_types:
      - "PERSON"
      - "DATE_TIME"
      - "LOCATION"
      - "PHONE_NUMBER"
      - "EMAIL"
      - "SSN"
      - "MEDICAL_RECORD_NUMBER"
    redaction_method: "Replace with tag"  # "Replace with tag" | "Replace with asterisks" | "Remove completely"
    save_redacted_text: true

  # Pattern normalization
  pattern_normalization:
    enabled: true
    save_normalized_text: false
```

### JSON Configuration

**File:** `config/data_config.json`

```json
{
  "data": {
    "input_file": "data/diabetes_cases.csv",
    "text_column": "note_text",
    "has_labels": true,
    "label_column": "dx_code",
    "label_mapping": {
      "E11": "Type 2 Diabetes",
      "E10": "Type 1 Diabetes",
      "O24": "Gestational Diabetes"
    },
    "deid_columns": ["age", "gender"],
    "additional_columns": [],
    "phi_redaction": {
      "enabled": true,
      "entity_types": ["PERSON", "DATE_TIME", "LOCATION"],
      "redaction_method": "Replace with tag",
      "save_redacted_text": true
    },
    "pattern_normalization": {
      "enabled": true,
      "save_normalized_text": false
    }
  }
}
```

### Python API

```python
from core.app_state import AppState
import pandas as pd

app_state = AppState()

# Load data
df = pd.read_csv("data/clinical_notes.csv")

# Configure data
app_state.set_data_config(
    input_file="data/clinical_notes.csv",
    text_column="clinical_text",
    has_labels=True,
    label_column="diagnosis",
    label_mapping={
        0: "Negative",
        1: "Positive"
    },
    deid_columns=["age", "gender"],
    additional_columns=["encounter_id"],
    enable_phi_redaction=True,
    phi_entity_types=["PERSON", "DATE_TIME", "LOCATION"],
    redaction_method="Replace with tag",
    save_redacted_text=True,
    enable_pattern_normalization=True,
    save_normalized_text=False
)
```

---

## Tab 4: Regex Patterns

**Purpose:** Define text normalization patterns for standardizing clinical text

### UI Configuration

**Location:** Fourth tab "Regex Patterns"

#### Pre-loaded Patterns

```
[Show Pre-loaded Patterns]

33 built-in patterns loaded:
├─ Measurement standardization
├─ Date formatting
├─ Unit normalization
├─ Abbreviation expansion
└─ Clinical term standardization
```

#### Pattern Management

```
[Add Pattern] [Edit Selected] [Delete Selected] [Import from File]

Pattern Table:
┌─────┬─────────────────┬──────────────────────┬─────────────────┐
│ #   │ Name            │ Pattern (Regex)      │ Replacement     │
├─────┼─────────────────┼──────────────────────┼─────────────────┤
│ 1   │ Weight kg       │ (\d+\.?\d*)\s*kg     │ \1 kilograms    │
│ 2   │ Height cm       │ (\d+\.?\d*)\s*cm     │ \1 centimeters  │
│ 3   │ BMI             │ BMI:?\s*(\d+\.?\d*)  │ BMI \1          │
│ 4   │ HbA1c percent   │ HbA1c:?\s*(\d+\.?\d*)│ HbA1c \1%       │
└─────┴─────────────────┴──────────────────────┴─────────────────┘
```

#### Add Custom Pattern

```
Pattern Name: [Text]
├─ Example: "Temperature Fahrenheit"

Regular Expression: [Text]
├─ Example: (\d+\.?\d*)\s*°?F
├─ Use capture groups: (...) for values to preserve

Replacement: [Text]
├─ Example: \1 degrees Fahrenheit
├─ Use \1, \2, etc. for captured groups

[✓] Case Insensitive
[✓] Enabled

[Save Pattern]
```

### YAML Configuration

**File:** `config/regex_patterns.yaml`

```yaml
# Regex Pattern Configuration
patterns:
  # Measurement standardization
  - name: "Weight in kilograms"
    pattern: '(\d+\.?\d*)\s*(?:kg|kilograms?)'
    replacement: '\1 kilograms'
    case_insensitive: true
    enabled: true
    category: "measurements"

  - name: "Weight in pounds"
    pattern: '(\d+\.?\d*)\s*(?:lb|lbs|pounds?)'
    replacement: '\1 pounds'
    case_insensitive: true
    enabled: true
    category: "measurements"

  - name: "Height in centimeters"
    pattern: '(\d+\.?\d*)\s*(?:cm|centimeters?)'
    replacement: '\1 centimeters'
    case_insensitive: true
    enabled: true
    category: "measurements"

  - name: "BMI value"
    pattern: 'BMI:?\s*(\d+\.?\d*)'
    replacement: 'BMI \1'
    case_insensitive: true
    enabled: true
    category: "measurements"

  # Laboratory values
  - name: "HbA1c percentage"
    pattern: 'HbA1c:?\s*(\d+\.?\d*)\s*%?'
    replacement: 'HbA1c \1%'
    case_insensitive: true
    enabled: true
    category: "labs"

  - name: "Glucose mg/dL"
    pattern: 'glucose:?\s*(\d+)\s*(?:mg/dl)?'
    replacement: 'glucose \1 mg/dL'
    case_insensitive: true
    enabled: true
    category: "labs"

  # Date standardization
  - name: "Date slash format"
    pattern: '(\d{1,2})/(\d{1,2})/(\d{2,4})'
    replacement: '\1-\2-\3'
    case_insensitive: false
    enabled: true
    category: "dates"

  # Abbreviation expansion
  - name: "y/o to years old"
    pattern: '(\d+)\s*y/?o'
    replacement: '\1 years old'
    case_insensitive: true
    enabled: true
    category: "abbreviations"

  - name: "pt to patient"
    pattern: '\bpt\b'
    replacement: 'patient'
    case_insensitive: true
    enabled: true
    category: "abbreviations"

  - name: "hx to history"
    pattern: '\bhx\b'
    replacement: 'history'
    case_insensitive: true
    enabled: true
    category: "abbreviations"

  # Clinical terms
  - name: "Standardize malnutrition"
    pattern: '\bmalnutr(?:ition)?\b'
    replacement: 'malnutrition'
    case_insensitive: true
    enabled: true
    category: "clinical"

  - name: "Standardize diabetes"
    pattern: '\bdiabet(?:es|ic)\b'
    replacement: 'diabetes'
    case_insensitive: true
    enabled: true
    category: "clinical"
```

### JSON Configuration

**File:** `config/regex_patterns.json`

```json
{
  "patterns": [
    {
      "name": "Weight in kilograms",
      "pattern": "(\\d+\\.?\\d*)\\s*(?:kg|kilograms?)",
      "replacement": "\\1 kilograms",
      "case_insensitive": true,
      "enabled": true,
      "category": "measurements"
    },
    {
      "name": "Height in centimeters",
      "pattern": "(\\d+\\.?\\d*)\\s*(?:cm|centimeters?)",
      "replacement": "\\1 centimeters",
      "case_insensitive": true,
      "enabled": true,
      "category": "measurements"
    },
    {
      "name": "Temperature Fahrenheit",
      "pattern": "(\\d+\\.?\\d*)\\s*°?F",
      "replacement": "\\1 degrees Fahrenheit",
      "case_insensitive": true,
      "enabled": true,
      "category": "measurements"
    },
    {
      "name": "HbA1c percentage",
      "pattern": "HbA1c:?\\s*(\\d+\\.?\\d*)\\s*%?",
      "replacement": "HbA1c \\1%",
      "case_insensitive": true,
      "enabled": true,
      "category": "labs"
    }
  ]
}
```

### Python API

```python
from core.app_state import AppState
from core.regex_preprocessor import RegexPreprocessor, RegexPattern

app_state = AppState()

# Create pattern
pattern = RegexPattern(
    name="Weight in kilograms",
    pattern=r'(\d+\.?\d*)\s*(?:kg|kilograms?)',
    replacement=r'\1 kilograms',
    case_insensitive=True,
    enabled=True
)

# Add to preprocessor
preprocessor = app_state.get_regex_preprocessor()
preprocessor.add_pattern(pattern)

# Or load multiple patterns
patterns = [
    RegexPattern(
        name="Height cm",
        pattern=r'(\d+\.?\d*)\s*cm',
        replacement=r'\1 centimeters',
        case_insensitive=True,
        enabled=True
    ),
    RegexPattern(
        name="BMI",
        pattern=r'BMI:?\s*(\d+\.?\d*)',
        replacement=r'BMI \1',
        case_insensitive=True,
        enabled=True
    )
]

for p in patterns:
    preprocessor.add_pattern(p)
```

---

## Tab 5: Extras (Hints)

**Purpose:** Provide task-specific hints, reference ranges, and diagnostic criteria

### UI Configuration

**Location:** Fifth tab "Extras (Hints)"

#### Pre-loaded Hints

```
[Show Pre-loaded Hints]

49 built-in hints loaded:
├─ WHO growth standards
├─ ASPEN malnutrition criteria
├─ CDC percentile references
├─ ADA diabetes guidelines
└─ Clinical assessment criteria
```

#### Hint Management

```
[Add Hint] [Edit Selected] [Delete Selected] [Import from File] [Export All]

Hint Table:
┌─────┬────────────────────┬────────────────────────┬─────────────┐
│ #   │ Title              │ Keywords               │ Category    │
├─────┼────────────────────┼────────────────────────┼─────────────┤
│ 1   │ WHO z-score...     │ malnutrition, z-score  │ standards   │
│ 2   │ ASPEN pediatric... │ malnutrition, ASPEN    │ guidelines  │
│ 3   │ HbA1c diagnostic...│ diabetes, HbA1c, dx    │ criteria    │
└─────┴────────────────────┴────────────────────────┴─────────────┘
```

#### Add Custom Hint

```
Hint Title: [Text]
├─ Example: "Hypoglycemia Severity Criteria"

Keywords (comma-separated): [Text]
├─ Example: hypoglycemia, glucose, low, blood sugar, severity
├─ Used for matching hints to extraction tasks

Category: [Dropdown]
├─ standards
├─ guidelines
├─ criteria
├─ reference_ranges
└─ custom

Hint Content: [Large text area]
├─ Provide detailed clinical information
├─ Reference ranges, criteria, guidelines
└─ Will be matched and provided to LLM during extraction

[Save Hint]
```

**Example Hint:**
```
Title: ASPEN Pediatric Malnutrition Severity Criteria

Keywords: malnutrition, ASPEN, pediatric, severity, classification, BMI, weight

Category: guidelines

Content:
ASPEN Pediatric Malnutrition Severity (requires 2+ indicators):

**Mild Malnutrition:**
- BMI-for-age z-score: -1 to -1.9 (3rd-15th percentile)
- Weight-for-length z-score: -1 to -1.9
- Mid-upper arm circumference z-score: -1 to -1.9
- Weight loss: 5% in 3 months OR
- Inadequate nutrient intake: 51-75% of estimated needs for >1 month

**Moderate Malnutrition:**
- BMI-for-age z-score: -2 to -2.9 (0.5th-3rd percentile)
- Weight-for-length z-score: -2 to -2.9
- MUAC z-score: -2 to -2.9
- Weight loss: 5-7.5% in 3 months OR 7.5-10% in 6 months
- Inadequate intake: 26-50% of needs for >1 month

**Severe Malnutrition:**
- BMI-for-age z-score: ≤ -3 (<0.5th percentile)
- Weight-for-length z-score: ≤ -3
- MUAC z-score: ≤ -3
- Weight loss: >7.5% in 3 months OR >10% in 6 months
- Inadequate intake: ≤25% of needs for >1 month

Reference: ASPEN Consensus Statement (2014)
```

### YAML Configuration

**File:** `config/extras/malnutrition_hints.yaml`

```yaml
# Extras (Hints) Configuration
extras:
  - title: "WHO Growth Standards Z-Score Classification"
    keywords:
      - "malnutrition"
      - "z-score"
      - "zscore"
      - "WHO"
      - "growth"
      - "anthropometric"
      - "percentile"
    category: "standards"
    content: |
      WHO Growth Standards - Z-Score Classification:

      **Weight-for-Height/BMI-for-Age:**
      - z < -3: SEVERE ACUTE MALNUTRITION (SAM) - <1st percentile
      - -3 ≤ z < -2: MODERATE ACUTE MALNUTRITION (MAM) - 2nd-3rd percentile
      - -2 ≤ z < -1: MILD MALNUTRITION RISK - 3rd-15th percentile
      - -1 ≤ z ≤ +1: NORMAL RANGE - 15th-85th percentile
      - +1 < z ≤ +2: OVERWEIGHT RISK - 85th-97th percentile
      - z > +2: OVERWEIGHT/OBESITY - >97th percentile

      **Height-for-Age (Stunting):**
      - z < -3: SEVERELY STUNTED (chronic malnutrition)
      - -3 ≤ z < -2: STUNTED (chronic undernutrition)
      - z ≥ -2: NORMAL HEIGHT

      **Z-Score Sign Convention:**
      - Percentile < 50th = NEGATIVE z-score (below average)
      - Percentile > 50th = POSITIVE z-score (above average)
      - 50th percentile = z-score of 0 (exactly average)

      Reference: WHO Child Growth Standards (2006)

  - title: "ASPEN Pediatric Malnutrition Severity Criteria"
    keywords:
      - "malnutrition"
      - "ASPEN"
      - "pediatric"
      - "severity"
      - "classification"
    category: "guidelines"
    content: |
      [Content from example above]

  - title: "ADA Diabetes Diagnostic Criteria"
    keywords:
      - "diabetes"
      - "diagnosis"
      - "HbA1c"
      - "glucose"
      - "ADA"
      - "diagnostic"
    category: "criteria"
    content: |
      ADA Diabetes Diagnostic Criteria (2024):

      **Diabetes Diagnosis (any one criterion):**
      1. HbA1c ≥ 6.5% (48 mmol/mol)
      2. Fasting plasma glucose ≥ 126 mg/dL (7.0 mmol/L)
         - Fasting = no caloric intake for ≥8 hours
      3. 2-hour plasma glucose ≥ 200 mg/dL (11.1 mmol/L) during OGTT
      4. Random plasma glucose ≥ 200 mg/dL (11.1 mmol/L) with classic symptoms

      **Prediabetes Criteria:**
      1. HbA1c: 5.7-6.4% (39-47 mmol/mol)
      2. Fasting glucose: 100-125 mg/dL (5.6-6.9 mmol/L) - Impaired Fasting Glucose
      3. 2-hour glucose: 140-199 mg/dL (7.8-11.0 mmol/L) - Impaired Glucose Tolerance

      **Normal:**
      - HbA1c < 5.7%
      - Fasting glucose < 100 mg/dL
      - 2-hour glucose < 140 mg/dL

      Reference: ADA Standards of Care (2024)
```

### JSON Configuration

**File:** `config/extras/cardiac_hints.json`

```json
{
  "extras": [
    {
      "title": "ACC/AHA Heart Failure Classification",
      "keywords": ["heart failure", "HF", "ACC", "AHA", "stage", "classification", "cardiac"],
      "category": "guidelines",
      "content": "ACC/AHA Heart Failure Stages:\n\nStage A: At Risk\n- Hypertension, diabetes, CAD, family history\n- No structural heart disease\n- No symptoms\n\nStage B: Pre-Heart Failure\n- Structural heart disease (LV hypertrophy, reduced EF)\n- No signs/symptoms of HF\n\nStage C: Symptomatic Heart Failure\n- Structural heart disease\n- Current or prior symptoms of HF\n- NYHA Class I-IV\n\nStage D: Advanced Heart Failure\n- Marked symptoms at rest despite max therapy\n- Requires specialized interventions\n\nReference: ACC/AHA Guidelines (2022)"
    },
    {
      "title": "Hypertension Blood Pressure Categories",
      "keywords": ["hypertension", "blood pressure", "BP", "HTN", "ACC", "AHA"],
      "category": "criteria",
      "content": "ACC/AHA Blood Pressure Categories:\n\n**Normal:**\n- Systolic: <120 mmHg AND\n- Diastolic: <80 mmHg\n\n**Elevated:**\n- Systolic: 120-129 mmHg AND\n- Diastolic: <80 mmHg\n\n**Stage 1 Hypertension:**\n- Systolic: 130-139 mmHg OR\n- Diastolic: 80-89 mmHg\n\n**Stage 2 Hypertension:**\n- Systolic: ≥140 mmHg OR\n- Diastolic: ≥90 mmHg\n\n**Hypertensive Crisis:**\n- Systolic: >180 mmHg AND/OR\n- Diastolic: >120 mmHg\n- Requires immediate medical attention\n\nReference: ACC/AHA Guideline (2017)"
    }
  ]
}
```

### Python API

```python
from core.app_state import AppState
from core.extras_manager import ExtrasManager, HintItem

app_state = AppState()

# Create hint
hint = HintItem(
    title="Hypoglycemia Severity Levels",
    keywords=["hypoglycemia", "glucose", "low", "blood sugar"],
    category="criteria",
    content="""
    Hypoglycemia Severity Classification:

    Level 1 (Alert): Glucose 54-70 mg/dL
    - Take action to raise glucose

    Level 2 (Serious): Glucose <54 mg/dL
    - Clinically significant hypoglycemia
    - Requires immediate treatment

    Level 3 (Severe): Altered mental status
    - Requires assistance
    - May have seizures or loss of consciousness

    Reference: ADA Classification
    """
)

# Add to extras manager
extras_manager = app_state.get_extras_manager()
extras_manager.add_hint(hint)

# Or load from file
extras_manager.load_hints_from_file("config/extras/custom_hints.yaml")
```

---

## Tab 6: Custom Functions

**Purpose:** Define medical calculations and clinical functions for automated computation

### UI Configuration

**Location:** Sixth tab "Custom Functions"

#### Pre-loaded Functions

```
[Show Pre-loaded Functions]

15 built-in functions loaded:
├─ calculate_bmi(weight_kg, height_cm)
├─ percentile_to_zscore(percentile)
├─ zscore_to_percentile(zscore)
├─ interpret_zscore_malnutrition(zscore, measurement_type)
├─ calculate_growth_percentile(value, age_months, gender, measurement_type)
└─ ... more functions
```

#### Function Management

```
[Add Function] [Edit Selected] [Delete Selected] [Import from File] [Test Function]

Function Table:
┌─────┬──────────────────────┬────────────────────┬──────────────┐
│ #   │ Function Name        │ Parameters         │ Category     │
├─────┼──────────────────────┼────────────────────┼──────────────┤
│ 1   │ calculate_bmi        │ weight_kg, height  │ anthropometric│
│ 2   │ percentile_to_zscore │ percentile         │ statistics   │
│ 3   │ egfr_ckd_epi         │ creatinine, age... │ renal        │
└─────┴──────────────────────┴────────────────────┴──────────────┘
```

#### Add Custom Function

```
Function Name: [Text]
├─ Example: calculate_framingham_risk

Description: [Text]
├─ Example: Calculate 10-year cardiovascular risk using Framingham equation

Category: [Dropdown]
├─ anthropometric
├─ laboratory
├─ cardiovascular
├─ renal
├─ statistics
└─ custom

Parameters: [Dynamic form]
├─ [Add Parameter]
└─ For each parameter:
    - Name: [Text]
    - Type: [number|string|boolean|array|object]
    - Required: [✓]
    - Description: [Text]
    - Default value: [Text] (optional)

Function Code: [Code editor with Python syntax]
├─ Write Python function implementation
└─ Use provided parameters

[Test Function] [Save Function]
```

**Example Function Definition:**
```python
Function Name: calculate_bmi

Description: Calculate Body Mass Index from weight and height

Category: anthropometric

Parameters:
- weight_kg (number, required): Weight in kilograms
- height_cm (number, required): Height in centimeters

Function Code:
def calculate_bmi(weight_kg, height_cm):
    """
    Calculate BMI from weight (kg) and height (cm)

    Args:
        weight_kg: Weight in kilograms
        height_cm: Height in centimeters

    Returns:
        dict with 'bmi' and 'classification'
    """
    if height_cm <= 0 or weight_kg <= 0:
        return {"error": "Invalid input values"}

    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)

    # WHO classification
    if bmi < 18.5:
        classification = "Underweight"
    elif 18.5 <= bmi < 25:
        classification = "Normal weight"
    elif 25 <= bmi < 30:
        classification = "Overweight"
    elif 30 <= bmi < 35:
        classification = "Obesity Class I"
    elif 35 <= bmi < 40:
        classification = "Obesity Class II"
    else:
        classification = "Obesity Class III"

    return {
        "bmi": round(bmi, 1),
        "classification": classification
    }
```

### YAML Configuration

**File:** `config/functions/medical_functions.yaml`

```yaml
# Custom Functions Configuration
functions:
  - name: "calculate_bmi"
    description: "Calculate Body Mass Index from weight and height"
    category: "anthropometric"
    parameters:
      weight_kg:
        type: "number"
        required: true
        description: "Weight in kilograms"
      height_cm:
        type: "number"
        required: true
        description: "Height in centimeters"
    code: |
      def calculate_bmi(weight_kg, height_cm):
          if height_cm <= 0 or weight_kg <= 0:
              return {"error": "Invalid input values"}

          height_m = height_cm / 100
          bmi = weight_kg / (height_m ** 2)

          if bmi < 18.5:
              classification = "Underweight"
          elif 18.5 <= bmi < 25:
              classification = "Normal weight"
          elif 25 <= bmi < 30:
              classification = "Overweight"
          elif 30 <= bmi < 35:
              classification = "Obesity Class I"
          elif 35 <= bmi < 40:
              classification = "Obesity Class II"
          else:
              classification = "Obesity Class III"

          return {
              "bmi": round(bmi, 1),
              "classification": classification
          }

  - name: "percentile_to_zscore"
    description: "Convert percentile to z-score"
    category: "statistics"
    parameters:
      percentile:
        type: "number"
        required: true
        description: "Percentile value (0-100)"
    code: |
      from scipy import stats

      def percentile_to_zscore(percentile):
          if percentile < 0 or percentile > 100:
              return {"error": "Percentile must be 0-100"}

          # Convert percentile to proportion
          proportion = percentile / 100.0

          # Calculate z-score using inverse normal CDF
          zscore = stats.norm.ppf(proportion)

          return {
              "percentile": percentile,
              "zscore": round(zscore, 2)
          }

  - name: "egfr_ckd_epi"
    description: "Calculate eGFR using CKD-EPI equation"
    category: "renal"
    parameters:
      creatinine:
        type: "number"
        required: true
        description: "Serum creatinine in mg/dL"
      age:
        type: "number"
        required: true
        description: "Age in years"
      gender:
        type: "string"
        required: true
        description: "Gender (M/F)"
      race:
        type: "string"
        required: false
        description: "Race (optional for 2021 equation)"
        default: "unknown"
    code: |
      import math

      def egfr_ckd_epi(creatinine, age, gender, race="unknown"):
          """Calculate eGFR using CKD-EPI equation (2021 version, race-free)"""

          if gender.upper() == "F":
              kappa = 0.7
              alpha = -0.241
              female_factor = 1.012
          else:
              kappa = 0.9
              alpha = -0.302
              female_factor = 1.0

          # CKD-EPI 2021 (race-free)
          egfr = 142 * min(creatinine/kappa, 1)**alpha * max(creatinine/kappa, 1)**-1.200 * 0.9938**age * female_factor

          if egfr >= 90:
              stage = "G1 (Normal or high)"
          elif 60 <= egfr < 90:
              stage = "G2 (Mild decreased)"
          elif 45 <= egfr < 60:
              stage = "G3a (Mild to moderate decreased)"
          elif 30 <= egfr < 45:
              stage = "G3b (Moderate to severe decreased)"
          elif 15 <= egfr < 30:
              stage = "G4 (Severe decreased)"
          else:
              stage = "G5 (Kidney failure)"

          return {
              "egfr": round(egfr, 1),
              "stage": stage,
              "equation": "CKD-EPI 2021"
          }
```

### JSON Configuration

**File:** `config/functions/cardiac_functions.json`

```json
{
  "functions": [
    {
      "name": "calculate_map",
      "description": "Calculate Mean Arterial Pressure",
      "category": "cardiovascular",
      "parameters": {
        "systolic": {
          "type": "number",
          "required": true,
          "description": "Systolic blood pressure (mmHg)"
        },
        "diastolic": {
          "type": "number",
          "required": true,
          "description": "Diastolic blood pressure (mmHg)"
        }
      },
      "code": "def calculate_map(systolic, diastolic):\n    map_value = (systolic + 2 * diastolic) / 3\n    return {\n        'map': round(map_value, 1),\n        'systolic': systolic,\n        'diastolic': diastolic\n    }"
    },
    {
      "name": "calculate_chads2_vasc",
      "description": "Calculate CHA2DS2-VASc score for stroke risk in atrial fibrillation",
      "category": "cardiovascular",
      "parameters": {
        "age": {
          "type": "number",
          "required": true,
          "description": "Age in years"
        },
        "gender": {
          "type": "string",
          "required": true,
          "description": "Gender (M/F)"
        },
        "chf": {
          "type": "boolean",
          "required": true,
          "description": "Congestive heart failure"
        },
        "hypertension": {
          "type": "boolean",
          "required": true,
          "description": "Hypertension"
        },
        "stroke_tia": {
          "type": "boolean",
          "required": true,
          "description": "Prior stroke or TIA"
        },
        "vascular_disease": {
          "type": "boolean",
          "required": true,
          "description": "Vascular disease"
        },
        "diabetes": {
          "type": "boolean",
          "required": true,
          "description": "Diabetes mellitus"
        }
      },
      "code": "def calculate_chads2_vasc(age, gender, chf, hypertension, stroke_tia, vascular_disease, diabetes):\n    score = 0\n    \n    # CHF\n    if chf:\n        score += 1\n    \n    # Hypertension\n    if hypertension:\n        score += 1\n    \n    # Age\n    if age >= 75:\n        score += 2\n    elif age >= 65:\n        score += 1\n    \n    # Diabetes\n    if diabetes:\n        score += 1\n    \n    # Stroke/TIA\n    if stroke_tia:\n        score += 2\n    \n    # Vascular disease\n    if vascular_disease:\n        score += 1\n    \n    # Female gender (if age >=65)\n    if gender.upper() == 'F' and age >= 65:\n        score += 1\n    \n    # Risk stratification\n    if score == 0:\n        risk = 'Low (0.2% annual stroke risk)'\n        recommendation = 'No anticoagulation or aspirin'\n    elif score == 1:\n        risk = 'Low-Moderate (0.6-2.0% annual)'\n        recommendation = 'Consider anticoagulation'\n    else:\n        risk = 'Moderate-High (>2.0% annual)'\n        recommendation = 'Anticoagulation recommended'\n    \n    return {\n        'chads2_vasc_score': score,\n        'stroke_risk': risk,\n        'recommendation': recommendation\n    }"
    }
  ]
}
```

### Python API

```python
from core.app_state import AppState
from core.function_registry import FunctionRegistry

app_state = AppState()

# Define function
function_def = {
    "name": "calculate_bsa",
    "description": "Calculate Body Surface Area using Mosteller formula",
    "category": "anthropometric",
    "parameters": {
        "weight_kg": {
            "type": "number",
            "required": True,
            "description": "Weight in kilograms"
        },
        "height_cm": {
            "type": "number",
            "required": True,
            "description": "Height in centimeters"
        }
    },
    "code": """
def calculate_bsa(weight_kg, height_cm):
    import math
    bsa = math.sqrt((weight_kg * height_cm) / 3600)
    return {
        "bsa_m2": round(bsa, 2),
        "formula": "Mosteller"
    }
"""
}

# Register function
registry = app_state.get_function_registry()
registry.register_function(function_def)

# Or load from file
registry.load_functions_from_file("config/functions/custom.yaml")
```

---

## Tab 7: RAG (Retrieval)

**Purpose:** Configure Retrieval-Augmented Generation for clinical guidelines and evidence retrieval

### UI Configuration

**Location:** Seventh tab "RAG"

#### Enable RAG

```
[✓] Enable RAG (Retrieval-Augmented Generation)
```

#### Document Management

```
[Add Documents] [Remove Selected] [Clear All]

Document List:
┌─────┬────────────────────────────────┬─────────┬─────────────┐
│ #   │ Document                       │ Type    │ Status      │
├─────┼────────────────────────────────┼─────────┼─────────────┤
│ 1   │ WHO_Growth_Standards_2006.pdf  │ PDF     │ ✓ Indexed   │
│ 2   │ ASPEN_Consensus_2014.pdf       │ PDF     │ ✓ Indexed   │
│ 3   │ ADA_Standards_2024.pdf         │ PDF     │ ✓ Indexed   │
│ 4   │ https://guideline-url.org      │ URL     │ ✓ Indexed   │
└─────┴────────────────────────────────┴─────────┴─────────────┘

Supported Formats:
├─ PDF documents (.pdf)
├─ Text files (.txt, .md)
├─ Web URLs (https://...)
└─ Word documents (.docx)
```

#### RAG Configuration

```
Embedding Model: [Dropdown]
├─ sentence-transformers/all-mpnet-base-v2 (default, recommended)
├─ sentence-transformers/all-MiniLM-L6-v2 (faster, smaller)
├─ BAAI/bge-large-en-v1.5 (higher quality)
└─ Custom model path

Chunk Size: [Slider 128-1024]
├─ Default: 512 tokens
├─ Smaller (256): More precise retrieval
└─ Larger (1024): More context per chunk

Chunk Overlap: [Slider 0-200]
├─ Default: 50 tokens
├─ Ensures context continuity across chunks

Top-K Results: [Number 1-10]
├─ Default: 3
├─ Number of most relevant chunks to retrieve
```

#### RAG Query Configuration

```
Query Fields (Auto RAG): [Multi-select from schema]
├─ [✓] status
├─ [✓] severity
├─ [ ] measurements
├─ [✓] management_plan
└─ Select fields that benefit from guideline evidence
```

#### Initialize RAG

```
[Initialize RAG Engine]
```

**Status Display:**
```
RAG Status: Ready ✓
Documents: 4 indexed
Total Chunks: 3,247
Index Size: 45.2 MB
```

### YAML Configuration

**File:** `config/rag_config.yaml`

```yaml
# RAG Configuration
rag:
  enabled: true

  # Documents to index
  documents:
    - path: "guidelines/WHO_Growth_Standards_2006.pdf"
      type: "pdf"
      category: "growth_standards"

    - path: "guidelines/ASPEN_Pediatric_Malnutrition_2014.pdf"
      type: "pdf"
      category: "malnutrition_guidelines"

    - path: "guidelines/ADA_Standards_of_Care_2024.pdf"
      type: "pdf"
      category: "diabetes_guidelines"

    - path: "https://www.cdc.gov/growthcharts/clinical_charts.htm"
      type: "url"
      category: "growth_charts"

    - path: "guidelines/ACC_AHA_Heart_Failure_2022.pdf"
      type: "pdf"
      category: "cardiac_guidelines"

  # Embedding configuration
  embedding_model: "sentence-transformers/all-mpnet-base-v2"
  # Alternatives:
  # - "sentence-transformers/all-MiniLM-L6-v2"  # Faster, smaller
  # - "BAAI/bge-large-en-v1.5"  # Higher quality

  # Chunking configuration
  chunk_size: 512  # Tokens per chunk
  chunk_overlap: 50  # Overlap between chunks

  # Retrieval configuration
  k_value: 3  # Top-K results to retrieve

  # Query fields (automatic RAG for these schema fields)
  rag_query_fields:
    - "malnutrition_status"
    - "severity"
    - "aspen_criteria_met"
    - "management_plan"

  # Cache configuration
  cache_dir: "./rag_cache"
  force_reindex: false  # Set to true to rebuild index
```

### JSON Configuration

**File:** `config/rag_config.json`

```json
{
  "rag": {
    "enabled": true,
    "documents": [
      {
        "path": "guidelines/WHO_Growth_Standards_2006.pdf",
        "type": "pdf",
        "category": "growth_standards"
      },
      {
        "path": "guidelines/ADA_Standards_2024.pdf",
        "type": "pdf",
        "category": "diabetes"
      },
      {
        "path": "https://www.acc.org/guidelines",
        "type": "url",
        "category": "cardiac"
      }
    ],
    "embedding_model": "sentence-transformers/all-mpnet-base-v2",
    "chunk_size": 512,
    "chunk_overlap": 50,
    "k_value": 3,
    "rag_query_fields": [
      "diagnosis",
      "severity",
      "management",
      "guidelines_met"
    ],
    "cache_dir": "./rag_cache",
    "force_reindex": false
  }
}
```

### Python API

```python
from core.app_state import AppState

app_state = AppState()

# Configure RAG
app_state.set_rag_config(
    enabled=True,
    documents=[
        "guidelines/WHO_Growth_Standards.pdf",
        "guidelines/ASPEN_Malnutrition_2014.pdf",
        "https://ada.org/standards-2024"
    ],
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    chunk_size=512,
    chunk_overlap=50,
    rag_query_fields=["status", "severity", "management_plan"],
    k_value=3,
    initialized=False
)

# Initialize RAG engine
rag_engine = app_state.get_rag_engine()
if rag_engine:
    rag_engine.initialize()
```

---

## Tab 8: Playground (Testing)

**Purpose:** Test single extractions with detailed logging before batch processing

### UI Configuration

**Location:** Eighth tab "Playground"

#### Full Pipeline Test

```
Clinical Text: [Large text area]
├─ Paste or type clinical note
└─ Example available

[✓] Test with Label
    Label/Diagnosis: [Text field]
    └─ Example: "Malnutrition Present"

[Run Full Pipeline Test]
```

**Output:**
```
=== EXTRACTION RESULTS ===

Agent Info:
- Version: 1.0.0
- Type: AgenticAgent
- Mode: continuous_agentic

Execution Metadata:
- Iterations: 5
- Tool Calls: 12 (RAG: 4, Functions: 6, Extras: 2)
- Execution Time: 8.3s
- Async Speedup: 68%

=== STAGE LOGS ===

[Stage 1: Task Analysis]
✓ Identified required tools
✓ Generated 3 RAG queries
✓ Identified 4 function calls

[Stage 2: Tool Execution - ASYNC]
✓ RAG Query 1: "ASPEN malnutrition criteria" (2.1s)
  Retrieved 3 chunks from ASPEN_2014.pdf
✓ Function 1: percentile_to_zscore(3) = -1.88
✓ Function 2: interpret_zscore_malnutrition(-1.88, "weight")
...

[Stage 3: Extraction]
✓ Generated JSON extraction

[Stage 4: RAG Refinement]
✓ Refined 3 fields with evidence

=== FINAL JSON ===
{
  "malnutrition_status": "Present - Mild",
  "weight_measurements": [
    {
      "date": "2024-01-15",
      "value_kg": 12.5,
      "percentile": 3,
      "zscore": -1.88
    }
  ],
  ...
}
```

#### Function Testing

```
Tab: Function Testing

Function: [Dropdown - select from registered functions]
├─ calculate_bmi
├─ percentile_to_zscore
├─ egfr_ckd_epi
└─ ...

Parameters (dynamic based on function):
├─ weight_kg: [50.5]
├─ height_cm: [165]

[Test Function]

Result:
{
  "bmi": 18.5,
  "classification": "Normal weight"
}
```

#### Extras Testing

```
Tab: Extras Testing

Keywords: [Text field]
├─ Example: malnutrition, pediatric, z-score, WHO

[Search Extras]

Matched Hints (3):
1. WHO Growth Standards Z-Score Classification
2. ASPEN Pediatric Malnutrition Criteria
3. Z-Score Interpretation Convention

[Show Full Content]
```

### No File Configuration
(Playground is interactive testing only)

### Python API

```python
from core.app_state import AppState
from core.agent_factory import create_agent

app_state = AppState()

# Configure all components (model, prompt, etc.)
# ...

# Create agent
agent = create_agent(
    llm_manager=app_state.get_llm_manager(),
    rag_engine=app_state.get_rag_engine(),
    extras_manager=app_state.get_extras_manager(),
    function_registry=app_state.get_function_registry(),
    regex_preprocessor=app_state.get_regex_preprocessor(),
    app_state=app_state
)

# Test single extraction
clinical_text = """
3-year-old male with poor appetite for 2 months.
Weight: 12.5 kg (3rd percentile)
Height: 92 cm
"""

result = agent.extract(
    clinical_text=clinical_text,
    label_value="Malnutrition Present"
)

# View results
print("Extraction:", result['extraction_output'])
print("Metadata:", result['metadata'])
```

---

## Tab 9: Processing (Batch Execution)

**Purpose:** Process entire datasets in batch with error handling and progress tracking

### UI Configuration

**Location:** Ninth tab "Processing"

#### Configuration Summary

```
=== CONFIGURATION SUMMARY ===

Model: gpt-4o (OpenAI) ✓
Execution Mode: Agentic Mode (AgenticAgent v1.0.0) ✓
Prompt: Pediatric Malnutrition Assessment ✓
Data: malnutrition_cases.csv (1,250 rows) ✓
RAG: Enabled (4 documents indexed) ✓
Functions: 15 functions registered ✓
Extras: 49 hints loaded ✓

[Refresh Configuration]
```

#### Processing Settings

```
Batch Size: [Number 1-100]
├─ Default: 10
├─ Number of rows to process before saving checkpoint
└─ Smaller batch = more frequent saves

Error Strategy: [Dropdown]
├─ skip: Skip failed rows, continue processing
├─ stop: Stop processing on first error
└─ retry: Retry failed rows (max 3 attempts)

Output Path: [File picker]
├─ Default: ./outputs/results_{timestamp}.csv
└─ [Browse...]

[✓] Dry Run (Test first 5 rows)
```

#### Start Processing

```
[Start Processing] (variant="primary", size="lg")
```

#### Progress Display

```
=== PROCESSING PROGRESS ===

Status: Processing... (Row 523/1,250)

Progress: [████████████░░░░░░░░] 42% (523/1,250)

Current Batch: 3/125
├─ Batch Start: Row 501
├─ Batch End: Row 510
└─ Time Remaining: ~15 minutes

Statistics:
├─ Successful: 518 (99.0%)
├─ Failed: 5 (1.0%)
├─ Avg Time/Row: 8.3s
└─ Total Time: 1h 12m

Recent Logs:
[12:34:56] Row 523: Extraction complete (8.1s, 12 tool calls)
[12:34:48] Row 522: Extraction complete (7.9s, 10 tool calls)
[12:34:40] Row 521: Warning - Hit max iterations (20)
[12:34:32] Row 520: Extraction complete (9.2s, 15 tool calls)

[Pause Processing] [Cancel]
```

#### Output Files

```
=== OUTPUT FILES ===

✓ results_20250105_123456.csv (main output)
├─ Size: 2.5 MB
├─ Rows: 523
└─ [Download] [View]

✓ processing_log_20250105_123456.txt
├─ Detailed execution logs
└─ [Download] [View]

✓ errors_20250105_123456.csv (5 failed rows)
├─ Failed rows with error messages
└─ [Download] [Retry Failed]

✓ checkpoint_batch_003.json
├─ Processing state checkpoint
└─ Can resume if interrupted
```

### YAML Configuration

**File:** `config/processing_config.yaml`

```yaml
# Processing Configuration
processing:
  # Batch settings
  batch_size: 10  # Rows per batch
  error_strategy: "skip"  # skip | stop | retry

  # Output settings
  output_path: "./outputs/results_{timestamp}.csv"
  save_logs: true
  log_path: "./outputs/logs/processing_{timestamp}.txt"
  save_errors: true
  error_path: "./outputs/errors/errors_{timestamp}.csv"

  # Checkpoint settings
  enable_checkpoints: true
  checkpoint_dir: "./outputs/checkpoints"
  checkpoint_frequency: 10  # Save every N rows

  # Retry settings (if error_strategy = "retry")
  max_retries: 3
  retry_delay_seconds: 5

  # Performance settings
  parallel_processing: false  # Set true for multi-threading (experimental)
  max_workers: 4  # If parallel_processing enabled

  # Dry run
  dry_run: false  # Test first 5 rows only
  dry_run_rows: 5
```

### JSON Configuration

**File:** `config/processing_config.json`

```json
{
  "processing": {
    "batch_size": 10,
    "error_strategy": "skip",
    "output_path": "./outputs/results_{timestamp}.csv",
    "save_logs": true,
    "log_path": "./outputs/logs/processing_{timestamp}.txt",
    "save_errors": true,
    "error_path": "./outputs/errors/errors_{timestamp}.csv",
    "enable_checkpoints": true,
    "checkpoint_dir": "./outputs/checkpoints",
    "checkpoint_frequency": 10,
    "max_retries": 3,
    "retry_delay_seconds": 5,
    "parallel_processing": false,
    "max_workers": 4,
    "dry_run": false,
    "dry_run_rows": 5
  }
}
```

### Python API

```python
from core.app_state import AppState
from core.agent_factory import create_agent
import pandas as pd

app_state = AppState()

# Configure processing
app_state.set_processing_config(
    batch_size=10,
    error_strategy="skip",
    output_path="./outputs/results.csv",
    dry_run=False
)

# Load data
df = pd.read_csv("data/clinical_notes.csv")

# Create agent
agent = create_agent(
    llm_manager=app_state.get_llm_manager(),
    rag_engine=app_state.get_rag_engine(),
    extras_manager=app_state.get_extras_manager(),
    function_registry=app_state.get_function_registry(),
    regex_preprocessor=app_state.get_regex_preprocessor(),
    app_state=app_state
)

# Process batch
results = []
for idx, row in df.iterrows():
    try:
        result = agent.extract(
            clinical_text=row['clinical_text'],
            label_value=row.get('diagnosis', '')
        )
        results.append(result['extraction_output'])
    except Exception as e:
        print(f"Row {idx} failed: {e}")
        continue

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("outputs/results.csv", index=False)
```

---

## Configuration Files Reference

### Directory Structure

```
clinorchestra/
├─ config/
│  ├─ model_config.yaml          # Model & execution mode
│  ├─ prompts/
│  │  ├─ malnutrition_prompt.yaml
│  │  ├─ diabetes_prompt.yaml
│  │  └─ custom_prompt.yaml
│  ├─ data_config.yaml           # Data loading & processing
│  ├─ regex_patterns.yaml        # Text normalization
│  ├─ extras/
│  │  ├─ malnutrition_hints.yaml
│  │  ├─ diabetes_hints.yaml
│  │  └─ cardiac_hints.yaml
│  ├─ functions/
│  │  ├─ anthropometric.yaml
│  │  ├─ laboratory.yaml
│  │  └─ cardiovascular.yaml
│  ├─ rag_config.yaml            # RAG configuration
│  └─ processing_config.yaml     # Batch processing
├─ data/
│  └─ clinical_notes.csv
├─ guidelines/
│  ├─ WHO_Growth_Standards.pdf
│  ├─ ASPEN_Malnutrition_2014.pdf
│  └─ ADA_Standards_2024.pdf
└─ outputs/
   ├─ results_*.csv
   ├─ logs/
   └─ checkpoints/
```

### Loading All Configurations

**Option 1: Load from UI**
1. Launch `clinorchestra`
2. Configure each tab
3. Click "Save Configuration" in each tab
4. Settings persist to `./config_persistence/` directory

**Option 2: Load from YAML/JSON**
```python
from core.app_state import AppState
from core.config_loader import ConfigLoader

app_state = AppState()
loader = ConfigLoader()

# Load all configurations
loader.load_all_configs(
    model_config_path="config/model_config.yaml",
    prompt_config_path="config/prompts/malnutrition_prompt.yaml",
    data_config_path="config/data_config.yaml",
    patterns_config_path="config/regex_patterns.yaml",
    extras_config_path="config/extras/malnutrition_hints.yaml",
    functions_config_path="config/functions/anthropometric.yaml",
    rag_config_path="config/rag_config.yaml",
    processing_config_path="config/processing_config.yaml",
    app_state=app_state
)
```

---

## Complete Workflow Examples

### Example 1: Malnutrition Assessment (Full Configuration)

**Step 1: Model Configuration**
```yaml
# config/model_config.yaml
model:
  provider: "openai"
  model_name: "gpt-4o"
  api_key: "${OPENAI_API_KEY}"
  temperature: 0.0
  max_tokens: 8000
  agentic_mode:
    enabled: true
    max_iterations: 30
    max_tool_calls: 100
```

**Step 2: Prompt Configuration**
```yaml
# config/prompts/malnutrition_prompt.yaml
# (See full example in Tab 2 section)
```

**Step 3: Data Configuration**
```yaml
# config/data_config.yaml
data:
  input_file: "data/malnutrition_cases.csv"
  text_column: "clinical_text"
  has_labels: true
  label_column: "diagnosis"
  deid_columns: ["age", "gender", "weight_kg"]
  phi_redaction:
    enabled: true
    entity_types: ["PERSON", "DATE_TIME"]
  pattern_normalization:
    enabled: true
```

**Step 4: Load Patterns**
```yaml
# config/regex_patterns.yaml
# (See full example in Tab 4 section)
```

**Step 5: Load Extras**
```yaml
# config/extras/malnutrition_hints.yaml
# (See full example in Tab 5 section)
```

**Step 6: Load Functions**
```yaml
# config/functions/anthropometric.yaml
# (See full example in Tab 6 section)
```

**Step 7: Configure RAG**
```yaml
# config/rag_config.yaml
rag:
  enabled: true
  documents:
    - "guidelines/WHO_Growth_Standards.pdf"
    - "guidelines/ASPEN_Malnutrition_2014.pdf"
  k_value: 3
```

**Step 8: Configure Processing**
```yaml
# config/processing_config.yaml
processing:
  batch_size: 10
  error_strategy: "skip"
  output_path: "./outputs/malnutrition_results.csv"
```

**Step 9: Run**
```bash
# Via UI
clinorchestra

# Via Python
python run_extraction.py --config config/malnutrition_full.yaml
```

---

### Example 2: Diabetes Extraction (Minimal Configuration)

**All-in-One Configuration File:**

```yaml
# config/diabetes_simple.yaml

# Model
model:
  provider: "anthropic"
  model_name: "claude-3-sonnet"
  temperature: 0.0
  agentic_mode:
    enabled: false  # Use classic mode

# Prompt
prompt:
  main_prompt: |
    Extract diabetes data from clinical text.

    Text: {clinical_text}
    Label: {label_context}

    Extract: type, HbA1c, medications, complications.

  schema:
    diabetes_type:
      type: "string"
      required: true
    hba1c_values:
      type: "array"
      required: true
    medications:
      type: "array"
      required: true

# Data
data:
  input_file: "data/diabetes_notes.csv"
  text_column: "note_text"
  has_labels: true
  label_column: "dx_code"

# Processing
processing:
  batch_size: 20
  output_path: "./outputs/diabetes_results.csv"
```

---

## Summary

This guide covered **ALL 9 tabs** with complete configuration methods:

✅ **Tab 1 - Model Configuration:** LLM setup + Agentic mode
✅ **Tab 2 - Prompt Configuration:** Task definition + schema + templates
✅ **Tab 3 - Data Configuration:** Input data + PHI redaction
✅ **Tab 4 - Regex Patterns:** Text normalization rules
✅ **Tab 5 - Extras (Hints):** Clinical guidelines & criteria
✅ **Tab 6 - Custom Functions:** Medical calculations
✅ **Tab 7 - RAG:** Clinical evidence retrieval
✅ **Tab 8 - Playground:** Interactive testing
✅ **Tab 9 - Processing:** Batch execution

**Configuration Methods:**
- ✅ UI (Gradio interface)
- ✅ YAML files
- ✅ JSON files
- ✅ Python API

**Author:** Frederick Gyasi (gyasifred)
**Institution:** MUSC Biomedical Informatics Center
**Platform:** Universal - works for ANY clinical task!

---

For more information:
- Architecture: `PIPELINE_ARCHITECTURE.md`
- Agentic Mode: `AGENTIC_USER_GUIDE.md`
- Quick Start: `README.md`
