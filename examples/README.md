# ClinOrchestra Examples

This directory contains sample data and configurations for testing ClinOrchestra.

## Sample Data

### `sample_clinical_notes.csv`
Contains 10 diverse clinical scenarios:
1. **Diabetes** - Type 2 DM with labs and medications
2. **Cardiac** - MI with ECG and troponin
3. **Malnutrition** - Pediatric case with growth parameters
4. **Sepsis** - Elderly patient with qSOFA criteria
5. **Prediabetes** - Routine visit with metabolic panel
6. **Growth Monitoring** - Normal infant assessment
7. **Preeclampsia** - Severe features in pregnancy
8. **Asthma** - Acute exacerbation
9. **Geriatric** - Polypharmacy and falls risk
10. **DKA** - Diabetic ketoacidosis

## Example Configurations

### `example_config_diabetes.json`
- **Purpose**: Extract diabetes assessment data
- **Fields**: Diagnosis, HbA1c, glucose, medications, BP, complications
- **Features**: Pattern normalization enabled
- **Use case**: Diabetes registry, quality monitoring

### `example_config_malnutrition.json`
- **Purpose**: Pediatric malnutrition assessment per ASPEN criteria
- **Fields**: Growth parameters, symptoms timeline, intake, physical exam, labs
- **Features**: RAG enabled for guideline retrieval, temporal tracking
- **Use case**: Nutrition clinic documentation, research data curation

## How to Use

### Method 1: Via UI

1. Launch ClinOrchestra:
   ```bash
   clinorchestra
   ```

2. Load sample data:
   - Go to **Data Configuration** tab
   - Upload `sample_clinical_notes.csv`
   - Select `clinical_text` as text column
   - Select `patient_id`, `encounter_id` as identifier columns

3. Apply example configuration:
   - **For Diabetes**: Copy settings from `example_config_diabetes.json`
   - **For Malnutrition**: Copy settings from `example_config_malnutrition.json`
   - Or manually configure similar schema in **Prompt Configuration**

4. Test in Playground:
   - Go to **Playground** tab
   - Paste a clinical note
   - Click "Test Extraction"
   - Review results

5. Batch process:
   - Go to **Processing** tab
   - Enable "Dry Run" for first 5 rows
   - Click "Start Processing"
   - Review output CSV

### Method 2: Programmatic

```python
from core.app_state import AppState
from core.agent_factory import create_agent
import pandas as pd

# Load sample data
df = pd.read_csv('examples/sample_clinical_notes.csv')

# Configure for diabetes extraction
app_state = AppState()
app_state.prompt_config.json_schema = {
    "diabetes_diagnosis": {"type": "string", "required": True},
    "hba1c_value": {"type": "number", "required": False},
    "medications": {"type": "array", "required": False}
}

# OPTIONAL: Enable Agentic Mode (v1.0.0) for autonomous tool calling
# app_state.set_agentic_config(enabled=True, max_iterations=20, max_tool_calls=50)

# Create agent (uses factory to select Classic or Agentic mode)
agent = create_agent(
    llm_manager=app_state.get_llm_manager(),
    rag_engine=app_state.get_rag_engine(),
    extras_manager=app_state.get_extras_manager(),
    function_registry=app_state.get_function_registry(),
    regex_preprocessor=app_state.get_regex_preprocessor(),
    app_state=app_state
)

# Extract from first row
result = agent.extract(
    clinical_text=df.iloc[0]['clinical_text'],
    label_value=df.iloc[0]['diagnosis_label']
)

print(result['extraction_output'])  # Final JSON extraction
print(result.get('metadata', {}))   # Metadata (iterations, tool calls, etc.)
```

**Note on Execution Modes:**
- **Classic Mode** (default): Reliable 4-stage pipeline (ExtractionAgent v1.0.2)
- **Agentic Mode**: Continuous loop with autonomous tool calling + async parallel execution (AgenticAgent v1.0.0) - 60-75% faster

See [AGENTIC_USER_GUIDE.md](../AGENTIC_USER_GUIDE.md) for detailed mode comparison.

## Expected Outputs

### Diabetes Case (P001)
```json
{
  "diabetes_diagnosis": "Type 2 Diabetes Mellitus",
  "hba1c_value": 8.2,
  "fasting_glucose": null,
  "current_medications": ["Metformin 1000mg twice daily", "Lisinopril 10mg once daily"],
  "blood_pressure": "145/92 mmHg",
  "complications": [],
  "control_status": "Uncontrolled (HbA1c 8.2%)"
}
```

### Malnutrition Case (P003)
```json
{
  "malnutrition_status": "Malnutrition Present - Moderate Severity",
  "age_years": 3,
  "weight_measurements": "12.5 kg (10th percentile, z-score -1.28)",
  "height_measurements": "92 cm (25th percentile, z-score -0.67)",
  "symptoms_timeline": "Poor appetite for 2 months, eating 40-50% of meals. Vomiting 2-3 times daily.",
  "intake_pattern": "40-50% of meals for past 2 months",
  "physical_exam_findings": "Mild muscle wasting, no edema",
  "laboratory_values": "Albumin 3.2 g/dL (low), Hemoglobin 10.5 g/dL",
  "diagnosis_reasoning": "Meets ASPEN criteria with 3+ indicators: insufficient intake, weight deceleration, muscle wasting, low albumin"
}
```

## Testing Functions

The sample data includes values that can test the registered medical functions:

```python
# Test BMI calculation
from core.function_registry import FunctionRegistry

registry = FunctionRegistry()

# P001: 185 lbs, 5'10" → BMI ~26.5
success, result, msg = registry.execute_function(
    "calculate_bmi",
    weight_kg=83.9,  # 185 lbs converted
    height_m=1.78    # 5'10" converted
)
print(f"BMI: {result}")  # Expected: ~26.5

# Test unit conversions
success, kg, _ = registry.execute_function("lbs_to_kg", lbs=185)
print(f"Weight: {kg} kg")  # Expected: ~83.9

success, cm, _ = registry.execute_function("inches_to_cm", inches=70)
print(f"Height: {cm} cm")  # Expected: 177.8
```

## Testing Patterns

Sample data includes various formats that patterns will normalize:

- `"BP 145/92"` → `"Blood pressure 145/92 mmHg"`
- `"HR 95 bpm"` → `"heart rate 95 bpm"`
- `"HbA1c 8.2%"` → `"HbA1c 8.2%"`
- `"Metformin 1000mg BID"` → `"Metformin 1000mg twice daily"`
- `"Lisinopril 10mg QD"` → `"Lisinopril 10mg once daily"`
- `"DM"` → `"diabetes mellitus"`
- `"HTN"` → `"hypertension"`

## Testing RAG

For RAG testing with malnutrition config:

1. Upload clinical guidelines (WHO, ASPEN) in **RAG Configuration** tab
2. System will retrieve relevant sections for:
   - WHO growth standards
   - ASPEN malnutrition criteria
   - Z-score interpretation
3. Stage 4 refinement will cite specific guideline sources

## Testing Extras

Sample keywords that should match loaded extras:

- "malnutrition" → ASPEN Pediatric Malnutrition Criteria
- "growth", "z-score" → WHO Growth Standards, Z-Score Interpretation
- "diabetes", "HbA1c" → Diabetes Diagnostic Criteria, HbA1c Target Goals
- "sepsis", "qSOFA" → Sepsis Recognition, Vital Sign Ranges
- "preeclampsia" → Preeclampsia Diagnosis
- "asthma" → Asthma Severity Classification

## Tips

1. **Start Small**: Test with single row in Playground before batch processing
2. **Enable Dry Run**: Use dry run (5 rows) to validate before full dataset
3. **Check Patterns**: Enable pattern normalization to see text standardization
4. **Use RAG Selectively**: Only enable RAG for fields that need guideline references
5. **Monitor Functions**: Check Playground logs to see which functions were called
6. **Iterate**: Refine prompts and schema based on initial results

## Troubleshooting

**Issue**: Functions not being called
- **Fix**: Check function descriptions match extraction needs
- **Fix**: Ensure numeric values are present in text
- **Fix**: Review Playground logs for function execution

**Issue**: RAG returning generic results
- **Fix**: Upload more specific clinical guidelines
- **Fix**: Specify `rag_query_fields` to focus refinement
- **Fix**: Check query keywords in Playground logs

**Issue**: Empty extractions
- **Fix**: Verify schema field descriptions are clear
- **Fix**: Test with simpler schema first
- **Fix**: Check LLM is configured correctly (Model Configuration tab)

---

**Questions?** Review the main README.md or check the documentation.
