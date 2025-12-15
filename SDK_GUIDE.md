# ClinOrchestra SDK Guide

**Using ClinOrchestra Programmatically (Without UI)**

This guide shows how to use ClinOrchestra's core packages directly in your Python applications, bypassing the Gradio UI entirely.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Components](#core-components)
- [Loading YAML Configurations](#loading-yaml-configurations)
- [Running Extraction Pipelines](#running-extraction-pipelines)
- [Advanced Usage](#advanced-usage)
- [Complete Examples](#complete-examples)

---

## Installation

```bash
pip install clinorchestra

# For local model support
pip install clinorchestra[local]

# From source
git clone https://github.com/gyasifred/clinorchestra.git
cd clinorchestra
pip install -e .
```

---

## Quick Start

### Minimal Example

```python
from core.llm_interface import LLMInterface
from core.agent_system import AgentSystem

# 1. Initialize LLM
llm = LLMInterface(
    provider="openai",
    model_name="gpt-4",
    api_key="your-api-key"
)

# 2. Create extraction agent
agent = AgentSystem(llm_interface=llm)

# 3. Define task
task_prompt = "Extract patient demographics and vital signs"
schema = {
    "age": "Patient age in years",
    "gender": "Patient gender",
    "blood_pressure": "Blood pressure reading"
}

# 4. Extract data
clinical_note = """
Patient is a 45-year-old male presenting with hypertension.
Vital signs: BP 150/90, HR 82 bpm, Temperature 98.6°F.
"""

result = agent.extract(
    text=clinical_note,
    task_prompt=task_prompt,
    schema=schema
)

print(result)
# {
#     "age": 45,
#     "gender": "male",
#     "blood_pressure": "150/90"
# }
```

---

## Core Components

### 1. LLM Interface

The `LLMInterface` manages connections to language model providers.

```python
from core.llm_interface import LLMInterface

# OpenAI
llm = LLMInterface(
    provider="openai",
    model_name="gpt-4",
    api_key="sk-..."
)

# Anthropic Claude
llm = LLMInterface(
    provider="anthropic",
    model_name="claude-3-5-sonnet-20241022",
    api_key="sk-ant-..."
)

# Google Gemini
llm = LLMInterface(
    provider="google",
    model_name="gemini-1.5-pro",
    api_key="..."
)

# Local models (Unsloth)
llm = LLMInterface(
    provider="unsloth",
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct",
    device="cuda"
)
```

### 2. Function Registry

Load and execute Python functions for calculations.

```python
from core.function_registry import FunctionRegistry

# Initialize (loads from functions/*.yaml)
registry = FunctionRegistry(storage_path="./functions")

# Import from YAML
with open("yaml_configs/malnutrition_functions.yaml", "r") as f:
    yaml_content = f.read()

success, count, message = registry.import_functions(yaml_content)
print(f"Imported {count} functions")

# Execute a function
success, result, error = registry.execute_function(
    name="calculate_bmi",
    weight_kg=70,
    height_m=1.75
)

if success:
    print(f"BMI: {result}")  # BMI: 22.86
```

### 3. Regex Preprocessor

Apply text normalization patterns.

```python
from core.regex_preprocessor import RegexPreprocessor

# Initialize (loads from patterns/*.yaml)
preprocessor = RegexPreprocessor(storage_path="./patterns")

# Import patterns from YAML
with open("yaml_configs/shared_patterns.yaml", "r") as f:
    yaml_content = f.read()

# Process text
text = "BP is 150 / 90 and patient weighs 70 kg"
normalized = preprocessor.process(text)
print(normalized)  # "BP 150/90 and patient weighs 70 kg"
```

### 4. Extras Manager

Load clinical knowledge and guidelines.

```python
from core.extras_manager import ExtrasManager

# Initialize (loads from extras/*.yaml)
extras_mgr = ExtrasManager(storage_path="./extras")

# Search for relevant extras
keywords = ["malnutrition", "aspen", "pediatric"]
matched_extras = extras_mgr.search_by_keywords(keywords, top_k=5)

for extra in matched_extras:
    print(f"{extra['name']}: {extra['content'][:100]}...")
```

### 5. RAG Engine

Query clinical guidelines and publications.

```python
from core.rag_engine import RAGEngine

# Initialize
rag = RAGEngine(
    index_path="./rag_index",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# Add documents
rag.add_pdf("guidelines/aspen_2014.pdf")
rag.add_url("https://pubmed.ncbi.nlm.nih.gov/...")

# Query
results = rag.query(
    query_text="ASPEN pediatric malnutrition diagnostic criteria",
    k=3
)

for result in results:
    print(f"Source: {result['source']}")
    print(f"Content: {result['content'][:200]}...")
```

---

## Loading YAML Configurations

### Method 1: Import via Managers

```python
import yaml
from pathlib import Path
from core.function_registry import FunctionRegistry
from core.regex_preprocessor import RegexPreprocessor
from core.extras_manager import ExtrasManager

# Initialize managers
func_registry = FunctionRegistry("./functions")
preprocessor = RegexPreprocessor("./patterns")
extras_mgr = ExtrasManager("./extras")

# Import malnutrition task configurations
yaml_configs = Path("yaml_configs")

# Import functions
with open(yaml_configs / "malnutrition_functions.yaml") as f:
    func_registry.import_functions(f.read())

# Import patterns
with open(yaml_configs / "malnutrition_patterns.yaml") as f:
    patterns_data = yaml.safe_load(f)
    for pattern in patterns_data:
        preprocessor.add_pattern(
            name=pattern['name'],
            pattern=pattern['pattern'],
            replacement=pattern['replacement'],
            description=pattern.get('description', ''),
            enabled=pattern.get('enabled', True)
        )

# Import extras
with open(yaml_configs / "malnutrition_extras.yaml") as f:
    extras_data = yaml.safe_load(f)
    for extra in extras_data:
        extras_mgr.add_extra(
            extra_type=extra.get('type', 'tip'),
            content=extra['content'],
            metadata=extra.get('metadata', {}),
            name=extra.get('name', extra.get('id'))
        )

print(f"Loaded {len(func_registry.functions)} functions")
print(f"Loaded {len(preprocessor.patterns)} patterns")
print(f"Loaded {len(extras_mgr.extras)} extras")
```

### Method 2: Automatic Loading at Startup

Managers automatically load from their directories on initialization:

```python
from core.function_registry import FunctionRegistry
from core.regex_preprocessor import RegexPreprocessor
from core.extras_manager import ExtrasManager

# These will auto-load all .yaml files from respective directories
func_registry = FunctionRegistry("./functions")      # Loads functions/*.yaml
preprocessor = RegexPreprocessor("./patterns")       # Loads patterns/*.yaml
extras_mgr = ExtrasManager("./extras")               # Loads extras/*.yaml
```

---

## Running Extraction Pipelines

### STRUCTURED Pipeline (4-Stage)

Recommended for production tasks with consistent requirements.

```python
from core.llm_interface import LLMInterface
from core.agent_system import AgentSystem
from core.function_registry import FunctionRegistry
from core.regex_preprocessor import RegexPreprocessor
from core.extras_manager import ExtrasManager
from core.rag_engine import RAGEngine

# 1. Initialize components
llm = LLMInterface(provider="openai", model_name="gpt-4", api_key="...")
func_registry = FunctionRegistry("./functions")
preprocessor = RegexPreprocessor("./patterns")
extras_mgr = ExtrasManager("./extras")
rag_engine = RAGEngine("./rag_index")

# 2. Create STRUCTURED agent
agent = AgentSystem(
    llm_interface=llm,
    function_registry=func_registry,
    regex_preprocessor=preprocessor,
    extras_manager=extras_mgr,
    rag_engine=rag_engine
)

# 3. Define extraction task
task_prompt = """
Extract pediatric malnutrition assessment including:
- Anthropometric measurements
- Z-scores for weight, height, BMI
- Malnutrition classification per ASPEN criteria
"""

schema = {
    "weight_kg": "Weight in kilograms",
    "height_cm": "Height in centimeters",
    "age_months": "Age in months",
    "weight_zscore": "Weight-for-age z-score",
    "height_zscore": "Height-for-age z-score",
    "bmi_zscore": "BMI-for-age z-score",
    "malnutrition_severity": "Severity: none, mild, moderate, severe"
}

# 4. Process clinical note
clinical_note = """
Patient: 24-month-old male
Weight: 10.5 kg
Height: 82 cm
Growth assessment shows weight-for-age z-score of -2.3 and
height-for-age z-score of -1.8. Child meets criteria for
moderate acute malnutrition per WHO standards.
"""

# 5. Run extraction
result = agent.extract(
    text=clinical_note,
    task_prompt=task_prompt,
    schema=schema,
    use_functions=True,
    use_rag=True,
    use_extras=True
)

print(result)
```

### ADAPTIVE Pipeline (Dynamic)

Recommended for complex tasks requiring autonomous decision-making.

```python
from core.llm_interface import LLMInterface
from core.agentic_agent import AgenticAgent
from core.function_registry import FunctionRegistry

# 1. Initialize components
llm = LLMInterface(provider="anthropic", model_name="claude-3-5-sonnet-20241022", api_key="...")
func_registry = FunctionRegistry("./functions")

# 2. Create ADAPTIVE agent
agent = AgenticAgent(
    llm_interface=llm,
    function_registry=func_registry,
    max_iterations=10  # Max autonomous iterations
)

# 3. Define complex task
task_prompt = """
Perform comprehensive ADRD diagnostic assessment:
1. Evaluate cognitive test scores (MMSE, CDR)
2. Assess functional decline
3. Review biomarker results
4. Determine diagnostic classification per NIA-AA criteria
5. Provide confidence level and supporting evidence
"""

schema = {
    "mmse_score": "MMSE score (0-30)",
    "mmse_severity": "Cognitive impairment level",
    "cdr_global": "CDR global score",
    "functional_decline": "Description of functional changes",
    "biomarkers": "Amyloid/tau/neurodegeneration status",
    "diagnosis": "Diagnostic classification",
    "confidence": "Diagnostic confidence (low/medium/high)",
    "evidence": "Supporting evidence for diagnosis"
}

# 4. Run adaptive extraction
result = agent.extract(
    text=clinical_note,
    task_prompt=task_prompt,
    schema=schema
)

# 5. Access execution trace
print("Iterations:", len(result.get('trace', [])))
print("Functions called:", result.get('functions_used', []))
print("Result:", result.get('extracted_data'))
```

---

## Advanced Usage

### 1. Batch Processing

```python
from core.llm_interface import LLMInterface
from core.agent_system import AgentSystem
import pandas as pd

# Initialize
llm = LLMInterface(provider="openai", model_name="gpt-4", api_key="...")
agent = AgentSystem(llm_interface=llm)

# Load dataset
df = pd.read_csv("clinical_notes.csv")

# Define task
task_prompt = "Extract vital signs"
schema = {
    "blood_pressure": "BP reading",
    "heart_rate": "HR in bpm",
    "temperature": "Temperature in °F"
}

# Process batch
results = []
for idx, row in df.iterrows():
    try:
        result = agent.extract(
            text=row['clinical_note'],
            task_prompt=task_prompt,
            schema=schema
        )
        results.append(result)
    except Exception as e:
        print(f"Error on row {idx}: {e}")
        results.append(None)

# Save results
df['extracted'] = results
df.to_csv("results.csv", index=False)
```

### 2. Custom Function Registration

```python
from core.function_registry import FunctionRegistry

registry = FunctionRegistry("./functions")

# Register custom function
code = """
def calculate_ckd_stage(gfr):
    '''Calculate CKD stage from GFR'''
    if gfr >= 90:
        return {"stage": 1, "description": "Normal"}
    elif gfr >= 60:
        return {"stage": 2, "description": "Mild"}
    elif gfr >= 45:
        return {"stage": 3, "description": "Moderate"}
    elif gfr >= 30:
        return {"stage": 3, "description": "Moderate-Severe"}
    elif gfr >= 15:
        return {"stage": 4, "description": "Severe"}
    else:
        return {"stage": 5, "description": "Kidney Failure"}
"""

success, message = registry.register_function(
    name="calculate_ckd_stage",
    code=code,
    description="Calculate CKD stage from GFR",
    parameters={
        "gfr": {
            "type": "number",
            "description": "GFR in mL/min/1.73m²",
            "required": True,
            "min": 0,
            "max": 200
        }
    },
    returns={"type": "object", "description": "CKD stage and description"}
)

# Execute
success, result, error = registry.execute_function(
    name="calculate_ckd_stage",
    gfr=45
)
print(result)  # {"stage": 3, "description": "Moderate"}
```

### 3. Selective Tool Enablement

```python
from core.agent_system import AgentSystem

agent = AgentSystem(llm_interface=llm)

# Enable only specific functions
agent.function_registry.enable_function("calculate_bmi")
agent.function_registry.enable_function("calculate_zscore")
agent.function_registry.disable_function("calculate_sofa_score")

# Enable only specific patterns
agent.regex_preprocessor.enable_pattern("normalize_bp")
agent.regex_preprocessor.disable_pattern("normalize_temperature")

# Run extraction with selective tools
result = agent.extract(
    text=clinical_note,
    task_prompt=task_prompt,
    schema=schema,
    use_functions=True  # Only enabled functions will be available
)
```

### 4. Multi-Document RAG

```python
from core.rag_engine import RAGEngine

rag = RAGEngine("./rag_index")

# Add multiple document types
rag.add_pdf("guidelines/aspen_2014.pdf")
rag.add_pdf("guidelines/who_2006.pdf")
rag.add_url("https://pubmed.ncbi.nlm.nih.gov/12345678/")
rag.add_text(
    text="Custom clinical protocol: ...",
    source="internal_protocol_v2"
)

# Query with filtering
results = rag.query(
    query_text="malnutrition diagnostic criteria",
    k=5,
    filter_source="aspen"  # Only ASPEN guidelines
)
```

### 5. Error Handling and Retry

```python
from core.agent_system import AgentSystem

agent = AgentSystem(
    llm_interface=llm,
    max_retries=3,
    retry_delay=2.0
)

try:
    result = agent.extract(
        text=clinical_note,
        task_prompt=task_prompt,
        schema=schema
    )
except Exception as e:
    print(f"Extraction failed after retries: {e}")
    # Fallback logic
```

### 6. Result Validation

```python
from core.agent_system import AgentSystem
import jsonschema

agent = AgentSystem(llm_interface=llm)

# Define JSON schema for validation
json_schema = {
    "type": "object",
    "properties": {
        "age": {"type": "integer", "minimum": 0, "maximum": 120},
        "blood_pressure": {"type": "string", "pattern": r"^\d+/\d+$"},
        "diagnosis": {"type": "string", "minLength": 1}
    },
    "required": ["age", "blood_pressure", "diagnosis"]
}

result = agent.extract(
    text=clinical_note,
    task_prompt=task_prompt,
    schema=schema
)

# Validate result
try:
    jsonschema.validate(instance=result, schema=json_schema)
    print("Validation passed!")
except jsonschema.ValidationError as e:
    print(f"Validation failed: {e.message}")
```

---

## Complete Examples

### Example 1: Malnutrition Assessment Pipeline

```python
"""
Complete malnutrition assessment using YAML configs
"""
import yaml
from pathlib import Path
from core.llm_interface import LLMInterface
from core.agent_system import AgentSystem
from core.function_registry import FunctionRegistry
from core.regex_preprocessor import RegexPreprocessor
from core.extras_manager import ExtrasManager

# 1. Initialize components
llm = LLMInterface(provider="openai", model_name="gpt-4", api_key="sk-...")

func_registry = FunctionRegistry("./functions")
preprocessor = RegexPreprocessor("./patterns")
extras_mgr = ExtrasManager("./extras")

# 2. Import malnutrition configs
yaml_dir = Path("yaml_configs")

with open(yaml_dir / "malnutrition_functions.yaml") as f:
    func_registry.import_functions(f.read())

with open(yaml_dir / "malnutrition_patterns.yaml") as f:
    patterns = yaml.safe_load(f)
    for p in patterns:
        preprocessor.add_pattern(
            name=p['name'],
            pattern=p['pattern'],
            replacement=p['replacement'],
            description=p.get('description', ''),
            enabled=p.get('enabled', True)
        )

with open(yaml_dir / "malnutrition_extras.yaml") as f:
    extras = yaml.safe_load(f)
    for e in extras:
        extras_mgr.add_extra(
            extra_type=e.get('type', 'tip'),
            content=e['content'],
            metadata=e.get('metadata', {}),
            name=e.get('name', e.get('id'))
        )

# 3. Create agent
agent = AgentSystem(
    llm_interface=llm,
    function_registry=func_registry,
    regex_preprocessor=preprocessor,
    extras_manager=extras_mgr
)

# 4. Define task
task_prompt = """
Extract anthropometric data and classify malnutrition per ASPEN criteria:
- Calculate z-scores for weight, height, BMI
- Determine malnutrition severity
- Identify etiology (illness-related vs non-illness-related)
"""

schema = {
    "age_months": "Age in months",
    "weight_kg": "Weight in kg",
    "height_cm": "Height in cm",
    "sex": "Male or female",
    "weight_zscore": "Weight-for-age z-score",
    "height_zscore": "Height-for-age z-score",
    "bmi_zscore": "BMI-for-age z-score",
    "malnutrition_severity": "None, mild, moderate, or severe",
    "malnutrition_etiology": "Illness-related or non-illness-related",
    "aspen_criteria_met": "Number of ASPEN criteria met (0-4)"
}

# 5. Process note
note = """
24-month-old male with chronic diarrhea
Weight: 10.2 kg (weight-for-age z-score -2.5)
Height: 81 cm (height-for-age z-score -2.1)
BMI-for-age z-score: -2.3
Assessment: Moderate acute malnutrition secondary to
chronic GI illness. Meets 3/4 ASPEN criteria.
"""

result = agent.extract(
    text=note,
    task_prompt=task_prompt,
    schema=schema,
    use_functions=True,
    use_extras=True
)

print("Extraction Result:")
print(f"Age: {result['age_months']} months")
print(f"Weight z-score: {result['weight_zscore']}")
print(f"Severity: {result['malnutrition_severity']}")
print(f"Etiology: {result['malnutrition_etiology']}")
```

### Example 2: ADRD Diagnostic Assessment

```python
"""
ADRD cognitive assessment with adaptive workflow
"""
from core.llm_interface import LLMInterface
from core.agentic_agent import AgenticAgent
from core.function_registry import FunctionRegistry
import yaml

# 1. Initialize
llm = LLMInterface(
    provider="anthropic",
    model_name="claude-3-5-sonnet-20241022",
    api_key="sk-ant-..."
)

func_registry = FunctionRegistry("./functions")

# 2. Import ADRD functions
with open("yaml_configs/adrd_functions.yaml") as f:
    func_registry.import_functions(f.read())

# 3. Create adaptive agent
agent = AgenticAgent(
    llm_interface=llm,
    function_registry=func_registry,
    max_iterations=10
)

# 4. Define task
task_prompt = """
Comprehensive ADRD assessment:
1. Calculate cognitive test scores
2. Determine impairment severity
3. Assess functional independence
4. Apply NIA-AA diagnostic criteria
5. Provide diagnostic classification
"""

schema = {
    "mmse_total": "MMSE score (0-30)",
    "mmse_severity": "Normal, mild, moderate, or severe impairment",
    "cdr_global": "CDR global score (0, 0.5, 1, 2, 3)",
    "cdr_interpretation": "CDR interpretation",
    "functional_status": "Independent, partially dependent, or dependent",
    "diagnosis": "Normal, MCI, or dementia",
    "dementia_type": "If dementia: Alzheimer's, vascular, mixed, etc.",
    "confidence": "Diagnostic confidence (low, medium, high)"
}

# 5. Process note
note = """
78-year-old female with progressive memory decline
MMSE: 18/30 (orientation 7/10, recall 1/3, language 6/8)
CDR: Global score 1.0 (memory=1, orientation=1, judgment=1)
Functional assessment: Requires assistance with finances,
medications, and complex tasks. Still independent for ADLs.
MRI: Bilateral hippocampal atrophy, no vascular changes
Diagnosis: Probable Alzheimer's disease dementia, mild stage
"""

result = agent.extract(
    text=note,
    task_prompt=task_prompt,
    schema=schema
)

print("Diagnostic Assessment:")
print(f"MMSE: {result['mmse_total']} - {result['mmse_severity']}")
print(f"CDR: {result['cdr_global']} - {result['cdr_interpretation']}")
print(f"Diagnosis: {result['diagnosis']}")
print(f"Type: {result.get('dementia_type', 'N/A')}")
print(f"Confidence: {result['confidence']}")
```

### Example 3: Parallel Processing with Multiple Workers

```python
"""
High-throughput batch processing with parallel workers
"""
from core.llm_interface import LLMInterface
from core.agent_system import AgentSystem
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def process_note(note_data):
    """Process single clinical note"""
    llm = LLMInterface(provider="openai", model_name="gpt-4", api_key="...")
    agent = AgentSystem(llm_interface=llm)

    try:
        result = agent.extract(
            text=note_data['text'],
            task_prompt=task_prompt,
            schema=schema
        )
        return {"success": True, "data": result, "id": note_data['id']}
    except Exception as e:
        return {"success": False, "error": str(e), "id": note_data['id']}

# Load dataset
df = pd.read_csv("clinical_notes.csv")

# Define task
task_prompt = "Extract vital signs and chief complaint"
schema = {
    "chief_complaint": "Reason for visit",
    "blood_pressure": "BP reading",
    "heart_rate": "HR in bpm"
}

# Parallel processing
notes = df.to_dict('records')
results = []

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(process_note, note): note for note in notes}

    for future in tqdm(as_completed(futures), total=len(futures)):
        result = future.result()
        results.append(result)

# Process results
successful = [r for r in results if r['success']]
failed = [r for r in results if not r['success']]

print(f"Success: {len(successful)}/{len(results)}")
print(f"Failed: {len(failed)}")

# Save
df_results = pd.DataFrame(successful)
df_results.to_csv("extraction_results.csv", index=False)
```

---

## API Reference

### LLMInterface

```python
LLMInterface(
    provider: str,              # "openai", "anthropic", "google", "azure", "unsloth"
    model_name: str,            # Model identifier
    api_key: str = None,        # API key (not needed for local models)
    temperature: float = 0.0,   # Sampling temperature
    max_tokens: int = 4000,     # Max response tokens
    device: str = "cuda"        # For local models: "cuda" or "cpu"
)
```

### AgentSystem (STRUCTURED)

```python
AgentSystem(
    llm_interface: LLMInterface,
    function_registry: FunctionRegistry = None,
    regex_preprocessor: RegexPreprocessor = None,
    extras_manager: ExtrasManager = None,
    rag_engine: RAGEngine = None,
    max_retries: int = 3,
    retry_delay: float = 1.0
)

# Extract method
agent.extract(
    text: str,                   # Clinical note text
    task_prompt: str,            # Extraction task description
    schema: dict,                # Field definitions
    use_functions: bool = False, # Enable function calling
    use_rag: bool = False,       # Enable RAG queries
    use_extras: bool = False,    # Enable extras matching
    use_patterns: bool = False   # Enable pattern preprocessing
) -> dict                        # Extracted data
```

### AgenticAgent (ADAPTIVE)

```python
AgenticAgent(
    llm_interface: LLMInterface,
    function_registry: FunctionRegistry = None,
    max_iterations: int = 10     # Max autonomous iterations
)

# Extract method
agent.extract(
    text: str,
    task_prompt: str,
    schema: dict
) -> dict                        # Extracted data with trace
```

### FunctionRegistry

```python
FunctionRegistry(storage_path: str = "./functions")

# Register function
registry.register_function(
    name: str,
    code: str,
    description: str,
    parameters: dict,
    returns: dict
) -> Tuple[bool, str]

# Import from YAML
registry.import_functions(yaml_str: str) -> Tuple[bool, int, str]

# Execute function
registry.execute_function(name: str, **kwargs) -> Tuple[bool, Any, str]

# Export to YAML
registry.export_functions() -> str
```

---

## Best Practices

1. **Initialize Once**: Create manager instances once and reuse them
2. **Use Batch Processing**: For large datasets, use parallel processing
3. **Enable Selective Tools**: Only enable tools needed for your task
4. **Validate Results**: Always validate extracted data against schema
5. **Handle Errors**: Implement proper error handling and retry logic
6. **Cache LLM Responses**: Use caching for repeated queries
7. **Monitor Costs**: Track API usage for cloud-based models

---

## Troubleshooting

**Issue**: Functions not found
```python
# Solution: Check if functions are loaded
print(f"Loaded functions: {list(registry.functions.keys())}")
```

**Issue**: Extraction returns incomplete data
```python
# Solution: Increase max_tokens or use more capable model
llm = LLMInterface(provider="openai", model_name="gpt-4", max_tokens=8000)
```

**Issue**: Slow batch processing
```python
# Solution: Use parallel processing
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(process_note, notes))
```

---

## Support

- **Issues**: https://github.com/gyasifred/clinorchestra/issues
- **Documentation**: See main README.md
- **Email**: gyasi@musc.edu

---

**ClinOrchestra SDK** - Build powerful clinical AI applications programmatically
