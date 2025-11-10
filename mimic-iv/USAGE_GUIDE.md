# MIMIC-IV Project - Usage Guide

## Quick Start

### Step 1: Get Top 20 Diagnoses

Run the simple extraction script with your MIMIC-IV path:

```bash
cd mimic-iv/scripts
python get_top_diagnoses_simple.py

# Or provide path directly:
python get_top_diagnoses_simple.py "C:\Users\gyasi\Documents\mimic-iv-3.1"
```

This will create `mimic-iv/top_20_primary_diagnoses.csv`

### Step 2: Load Patterns, Functions, and Extras into ClinOrchestra

All files are now in proper JSON format matching ClinOrchestra's expectations.

#### Load Patterns (Regex Preprocessing)
In ClinOrchestra UI → **Patterns Tab**:
- Upload each JSON file from `mimic-iv/patterns/`
- These will standardize vital signs and lab values in clinical text

Available patterns:
- `vital_signs_bp.json` - Blood pressure
- `vital_signs_hr.json` - Heart rate
- `vital_signs_rr.json` - Respiratory rate
- `vital_signs_temp.json` - Temperature
- `vital_signs_spo2.json` - Oxygen saturation
- `lab_wbc.json` - White blood cell count
- `lab_creatinine.json` - Creatinine levels
- `lab_lactate.json` - Lactate levels

#### Load Functions (Clinical Calculations)
In ClinOrchestra UI → **Functions Tab**:
- Upload each JSON file from `mimic-iv/functions/`
- These enable calculations in your prompts

Available functions:
- `calculate_sofa_score.json` - SOFA score for sepsis severity
- `calculate_curb65.json` - CURB-65 for pneumonia severity
- `calculate_map.json` - Mean arterial pressure
- `calculate_creatinine_clearance.json` - Renal function

Usage in prompt:
```
Use calculate_sofa_score(platelets=95, bilirubin=2.5, gcs=12, creatinine=2.8, on_vasopressor=True) to assess sepsis severity.
```

#### Load Extras (Clinical Knowledge)
In ClinOrchestra UI → **Extras Tab**:
- Upload each JSON file from `mimic-iv/extras/`
- These provide diagnostic criteria and clinical knowledge

Available extras:
- `sepsis_diagnostic_criteria.json` - Sepsis-3 criteria
- `heart_failure_classification.json` - HF types and diagnosis
- `aki_staging.json` - KDIGO AKI staging
- `respiratory_failure_types.json` - Type 1 vs Type 2
- `clinical_annotation_approach.json` - Systematic annotation methodology

### Step 3: Use Prompts and Schemas

#### Task 1: Clinical Consultant Annotation
1. In ClinOrchestra → **Prompt Tab**:
   - Paste content from `prompts/task1_annotation_prompt.txt`
2. In ClinOrchestra → **Processing Tab**:
   - Upload `schemas/task1_annotation_schema.json`
3. In ClinOrchestra → **Data Tab**:
   - Upload your annotation dataset CSV

#### Task 2: Multiclass Classification
1. Generate prompt with diagnosis list:
   ```bash
   python scripts/generate_classification_prompt.py
   ```
2. In ClinOrchestra → **Prompt Tab**:
   - Paste content from generated prompt file
3. In ClinOrchestra → **Processing Tab**:
   - Upload `schemas/task2_classification_schema_v2.json`
4. In ClinOrchestra → **Data Tab**:
   - Upload your classification dataset CSV

### Step 4: Process Data

In ClinOrchestra:
1. Configure batch processing settings
2. Click **Start Processing**
3. Monitor progress
4. Download results when complete

### Step 5: Evaluate (Task 2 only)

```bash
python scripts/evaluate_classification.py
```

Provide:
- Predictions JSON (from ClinOrchestra output)
- Ground truth CSV
- Top 20 diagnoses mapping

## File Format Reference

### Pattern JSON Format
```json
{
  "name": "pattern_name",
  "pattern": "regex pattern",
  "replacement": "replacement string",
  "description": "What this pattern does",
  "enabled": true
}
```

### Function JSON Format
```json
{
  "name": "function_name",
  "code": "\ndef function_name(param1, param2):\n    return result\n",
  "description": "What this function calculates",
  "parameters": {
    "param1": {"type": "number", "description": "Parameter description"}
  },
  "returns": "Return value description",
  "signature": "(param1, param2)"
}
```

### Extra JSON Format
```json
{
  "id": "unique_id",
  "type": "diagnostic_criteria|pattern|methodology",
  "content": "The clinical knowledge text",
  "metadata": {
    "category": "category_name",
    "priority": "CRITICAL|HIGH|MEDIUM|LOW"
  },
  "created_at": "2025-11-10T00:00:00",
  "name": "Display name"
}
```

## Tips

1. **Start Small**: Test with 10-100 records first
2. **Use Extras**: They significantly improve annotation quality
3. **Enable Patterns**: Standardizes text before LLM processing
4. **Use Functions**: Enables calculations in outputs
5. **Monitor Costs**: Track API usage for large datasets

## Troubleshooting

**Patterns not working?**
- Check regex syntax is valid
- Ensure `enabled: true`
- Test pattern on sample text first

**Functions erroring?**
- Verify function code syntax
- Check parameter types match
- Test function with sample values

**Extras not appearing?**
- Ensure metadata keywords match your prompt/schema
- Check category and priority settings
- Verify JSON format is correct

## Next Steps

1. Extract your top 20 diagnoses
2. Load patterns, functions, extras into ClinOrchestra
3. Process your datasets
4. Evaluate results
5. Iterate and improve

See README.md for complete documentation.
