# ClinOrchestra Multi-Column Prompt Support - Implementation Summary

## Problem Statement

The current system only supports passing two values to prompts:
1. `clinical_text` - The main text column from the dataset
2. `label_context` - The mapped label value

However, for complex prompts like MIMIC-IV tasks, we need to pass MULTIPLE columns from the dataset (subject_id, hadm_id, age, gender, race, admission_type, etc.) as separate placeholder variables in the prompt.

## Current State Analysis

### What Works Now
- ✅ Malnutrition prompts work because they only use `{clinical_text}` and `{label_context}`
- ✅ RAG, functions, and extras outputs are already supported via `{rag_outputs}`, `{function_outputs}`, `{extras_outputs}`
- ✅ JSON schema instructions are supported via `{json_schema_instructions}`

### What Doesn't Work
- ❌ MIMIC prompts need `{subject_id}`, `{hadm_id}`, `{consolidated_diagnosis_name}`, `{anchor_age}`, `{gender}`, `{race}`, `{admission_type}`, `{admittime}`, etc.
- ❌ Current agent only accepts `clinical_text` as a string, not a dict of column values
- ❌ Data_tab doesn't have UI for selecting "prompt input columns" (only text column, identifier columns, additional columns)
- ❌ Processing pipeline doesn't extract and pass these columns to the prompt formatter

## Changes Completed

### 1. MIMIC Prompts Updated ✅
- **File**: `mimic-iv/prompts/task1_annotation_prompt.txt`
  - Changed `{primary_diagnosis_name}` → `{consolidated_diagnosis_name}` (to match consolidated dataset column)
  - Added `{rag_outputs}`, `{function_outputs}`, `{extras_outputs}`, `{json_schema_instructions}` placeholders

- **File**: `mimic-iv/prompts/task2_classification_prompt_v2.txt`
  - Added `{rag_outputs}`, `{function_outputs}`, `{extras_outputs}`, `{json_schema_instructions}` placeholders

## Changes Needed

### 2. Core Architecture - Support Multiple Prompt Columns

#### A. ExtractationAgentContext (agent_system.py)
**Current**:
```python
@dataclass
class ExtractionAgentContext:
    clinical_text: str
    label_value: Optional[Any] = None
    label_context: str = ""
    ...
```

**Needed**:
```python
@dataclass
class ExtractionAgentContext:
    clinical_text: str
    label_value: Optional[Any] = None
    label_context: str = ""
    prompt_variables: Dict[str, Any] = field(default_factory=dict)  # NEW: Additional columns for prompt
    ...
```

#### B. ExtractionAgent.extract() Method
**Current**:
```python
def extract(self, clinical_text: str, label_value: Optional[Any] = None) -> Dict[str, Any]:
```

**Needed**:
```python
def extract(
    self,
    clinical_text: str,
    label_value: Optional[Any] = None,
    prompt_variables: Optional[Dict[str, Any]] = None  # NEW: Additional prompt placeholders
) -> Dict[str, Any]:
```

#### C. Prompt Formatting Logic
**Current**: Prompts are formatted with only `{clinical_text}`, `{label_context}`, and tool outputs

**Needed**: Format prompts with ALL variables from `prompt_variables` dict plus existing ones

### 3. Data Configuration Tab (data_tab.py)

Add new UI component for selecting "Prompt Input Columns":

```python
with gr.Row():
    prompt_input_columns = gr.CheckboxGroup(
        choices=[],
        label="Prompt Input Columns",
        info="Columns to pass as variables to the prompt template (e.g., patient_id, age, gender). These will be available as {column_name} in your prompt."
    )
    components['prompt_input_columns'] = prompt_input_columns
```

**Location**: After "Additional Columns", before "Label Configuration"

### 4. Processing Tab (processing_tab.py)

#### A. Extract Selected Columns
When processing each row, extract the values for all selected prompt input columns:

```python
# Get prompt variables from selected columns
prompt_variables = {}
if prompt_input_cols:
    for col in prompt_input_cols:
        if col in row:
            prompt_variables[col] = row[col]
```

#### B. Pass to Agent
```python
result = agent.extract(
    clinical_text=text,
    label_value=label_val,
    prompt_variables=prompt_variables  # NEW
)
```

### 5. Prompt Template Formatting

When formatting prompts (in agent_system.py or wherever prompts are filled), use:

```python
# Build complete variables dict
format_vars = {
    'clinical_text': self.context.clinical_text,
    'label_context': self.context.label_context,
    'rag_outputs': rag_output,
    'function_outputs': func_output,
    'extras_outputs': extras_output,
    'json_schema_instructions': schema_instructions,
    **self.context.prompt_variables  # Add all user-defined variables
}

# Format prompt
filled_prompt = user_prompt_template.format(**format_vars)
```

### 6. Playground Tab (playground_tab.py)

Add UI for manually entering prompt variable values for testing:

```python
# If prompt has custom variables like {subject_id}, {hadm_id}, show input fields
detected_variables = extract_prompt_variables(prompt_text)
variable_inputs = {}
for var in detected_variables:
    if var not in ['clinical_text', 'label_context', 'rag_outputs', 'function_outputs', 'extras_outputs', 'json_schema_instructions']:
        variable_inputs[var] = gr.Textbox(
            label=f"Prompt Variable: {var}",
            placeholder=f"Enter value for {{{var}}}"
        )
```

### 7. ClientDisconnect Error Fix

The error occurs during file upload when client disconnects. This is typically due to:
1. File too large (timeout)
2. Network interruption
3. Missing async handling

**File**: Likely in `ui/data_tab.py` or Gradio's upload handler

**Fix**: Add proper error handling and increase timeout:

```python
file_upload = gr.File(
    label="CSV File",
    file_types=[".csv"],
    type="filepath",
    # Add proper configuration
)

# In upload handler:
try:
    df = pd.read_csv(filepath, low_memory=False, chunksize=None)
except Exception as e:
    return f"Error loading file: {str(e)}", None, None
```

## Implementation Order

1. ✅ **Update MIMIC prompts** (COMPLETED)
2. ⏭️ **Update agent_system.py** - Add `prompt_variables` support to context and extract method
3. ⏭️ **Update data_tab.py** - Add "Prompt Input Columns" UI
4. ⏭️ **Update processing_tab.py** - Extract and pass prompt variables to agent
5. ⏭️ **Update playground_tab.py** - Add UI for testing with custom variables
6. ⏭️ **Fix ClientDisconnect error** - Add error handling to file upload
7. ⏭️ **End-to-end testing** - Test with MIMIC dataset and prompts
8. ⏭️ **Commit and push** - Professional commit messages

## Testing Plan

### Test Case 1: Malnutrition (Existing - Should Still Work)
- Dataset: Only has `clinical_text` and `label`
- Prompt: Uses `{clinical_text}` and `{label_context}`
- Expected: Works as before (no breaking changes)

### Test Case 2: MIMIC Annotation (New)
- Dataset: Has `subject_id`, `hadm_id`, `consolidated_diagnosis_name`, `clinical_text`, `anchor_age`, `gender`, `race`, `admission_type`
- Prompt: Uses all these as `{subject_id}`, `{hadm_id}`, etc.
- Expected: All variables are populated correctly in the prompt

### Test Case 3: MIMIC Classification (New)
- Same as Test Case 2
- Expected: All variables populated, LLM can access patient demographics

## File Locations

- ✅ `mimic-iv/prompts/task1_annotation_prompt.txt` - UPDATED
- ✅ `mimic-iv/prompts/task2_classification_prompt_v2.txt` - UPDATED
- ⏭️ `core/agent_system.py` - NEEDS UPDATE (context, extract method, formatting)
- ⏭️ `ui/data_tab.py` - NEEDS UPDATE (add prompt input columns UI)
- ⏭️ `ui/processing_tab.py` - NEEDS UPDATE (extract and pass variables)
- ⏭️ `ui/playground_tab.py` - NEEDS UPDATE (manual variable entry)

## Consolidated Dataset Column Names (for reference)

```python
# Available in MIMIC-IV consolidated dataset:
columns = [
    'subject_id',           # Patient ID
    'hadm_id',              # Hospital admission ID
    'consolidated_diagnosis_id',     # Consolidated diagnosis ID
    'consolidated_diagnosis_name',   # Consolidated diagnosis name (USE THIS, not primary_diagnosis_name)
    'consolidated_category',         # Category
    'icd_code',             # Original ICD code
    'icd_version',          # ICD-9 or ICD-10
    'original_icd_description',      # Original ICD description
    'clinical_text',        # Combined clinical text (discharge + radiology)
    'admission_type',       # Emergency, elective, etc.
    'gender',               # M/F
    'anchor_age',           # Patient age
    'race',                 # Patient race
    'insurance',            # Insurance type
    'admittime',            # Admission datetime
    'dischtime',            # Discharge datetime
    'hospital_expire_flag'  # Died in hospital (0/1)
]
```

## Notes

- This is version 1.0.0 - production-ready code required
- Maintain backward compatibility with existing malnutrition workflows
- All changes must be thoroughly tested
- Use professional git commit messages
- Document all changes clearly
