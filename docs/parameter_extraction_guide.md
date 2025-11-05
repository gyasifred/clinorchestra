# Parameter Extraction and Mapping in ClinAnnotate

## Overview

The ClinAnnotate agent system uses a sophisticated **Stage 1 Analysis** process to extract parameters from clinical text and map them to function arguments. This document explains how parameters are understood, extracted, and transformed.

---

## Stage 1: Task Analysis & Parameter Extraction

### What the Agent Does

The agent analyzes the clinical text and extraction schema, then:

1. **Identifies needed functions** based on available measurements and schema requirements
2. **Extracts parameter values** from the clinical text with high precision
3. **Maps extracted values** to function parameter names
4. **Handles temporal context** for serial measurements
5. **Plans RAG queries** and extras keywords

---

## Parameter Mapping Rules

### Basic Mapping Conventions

The Stage 1 prompt instructs the agent to map clinical text values to function parameters:

| Clinical Text | Parameter Name | Parameter Value | Notes |
|---------------|----------------|-----------------|-------|
| "45.5 kg" | `weight_kg` | 45.5 | Weight in kilograms |
| "165 cm" | `height_cm` | 165 | Height in centimeters |
| "5'5\"" | `height` | 65 (inches) | Convert to cm if needed |
| "65 year old" | `age` or `age_years` | 65 | Age in years |
| "30 months old" | `age_months` | 30 | Age in months |
| "BP 140/90" | `systolic=140, diastolic=90` | 140, 90 | Blood pressure |
| "male" or "boy" | `sex` | "male" | Sex (string) |
| "female" or "girl" | `sex` | "female" | Sex (string) |

---

## Sex Parameter: Special Handling

### Convention: 1 = Male, 2 = Female

Growth percentile functions (CDC/WHO) use numeric codes for sex:
- **1 = Male** (boys' growth charts)
- **2 = Female** (girls' growth charts)

This is the standard convention in pediatric growth assessment.

### Automatic String-to-Number Conversion

**The agent can pass sex as a human-readable string**, and the system automatically converts it:

#### Conversion Table:

| String Values (Case-Insensitive) | Converts To | Meaning |
|-----------------------------------|-------------|---------|
| "male", "m", "boy", "man" | 1 | Male |
| "female", "f", "girl", "woman" | 2 | Female |

#### Implementation

**Location:** `core/function_registry.py` → `_apply_parameter_transformations()`

```python
# Handle sex parameter - convert string to number
# CDC/WHO convention: 1 = male, 2 = female
if 'sex' in transformed:
    sex_value = transformed['sex']
    if isinstance(sex_value, str):
        sex_lower = sex_value.lower().strip()
        if sex_lower in ['male', 'm', 'boy', 'man']:
            transformed['sex'] = 1
            logger.info(f"Converted sex '{sex_value}' to 1 (male)")
        elif sex_lower in ['female', 'f', 'girl', 'woman']:
            transformed['sex'] = 2
            logger.info(f"Converted sex '{sex_value}' to 2 (female)")
```

**Applied to these functions:**
- `calculate_growth_percentile`
- `calculate_cdc_growth_percentile`
- `calculate_who_growth_percentile`

#### Example Flow:

```
Clinical Text: "3-year-old male, weight 12.5 kg"

Stage 1 (Agent extracts):
{
  "name": "calculate_growth_percentile",
  "parameters": {
    "measurement_type": "weight",
    "value": 12.5,
    "age_months": 36,
    "sex": "male"  // ← Agent passes string
  }
}

Stage 2 (System transforms):
{
  "measurement_type": "weight",
  "value": 12.5,
  "age_months": 36,
  "sex": 1  // ← Automatically converted to 1
}

Function executes with numeric code.
```

---

## Other Parameter Transformations

The system also performs automatic transformations for other parameters:

### 1. Age Transformations

```python
# If function needs age_months but agent provides 'age':
{'age': 36} → {'age_months': 36}
```

### 2. Weight/Height Normalization

```python
# Normalize parameter names:
{'weight': 12.5} → {'weight_kg': 12.5}
{'height': 92} → {'height_cm': 92}
```

### 3. Unit Conversions

For BMI calculations:
```python
# If height is in cm but function needs meters:
{'height_cm': 165} → {'height_m': 1.65}
```

---

## Stage 1 Prompt Instructions (Excerpt)

### Parameter Mapping Section:

```
C. MAP VALUES TO FUNCTION PARAMETERS:
   - Map values to function parameters based on context:
     * "45.5 kg" → weight_kg parameter
     * "165 cm" or "5'5\"" → height parameter (convert inches if needed)
     * "65 year old" or "age 65" → age parameter
     * "BP 140/90" → systolic=140, diastolic=90
     * "male" or "female" or "boy" or "girl" → sex parameter (pass as string 'male' or 'female')
     * IMPORTANT: For growth percentile functions, sex can be passed as string ('male'/'female')
       and will be automatically converted to numbers (1=male, 2=female) by the system
```

---

## Serial Measurements: Temporal Context

For serial measurements, the agent includes **date_context** for each function call:

```json
{
  "functions_needed": [
    {
      "name": "calculate_growth_percentile",
      "parameters": {
        "measurement_type": "weight",
        "value": 13.5,
        "age_months": 30,
        "sex": "male"
      },
      "date_context": "6 months ago (baseline)",
      "reason": "Calculate baseline weight percentile"
    },
    {
      "name": "calculate_growth_percentile",
      "parameters": {
        "measurement_type": "weight",
        "value": 13.0,
        "age_months": 33,
        "sex": "male"
      },
      "date_context": "3 months ago",
      "reason": "Track weight percentile at 3-month follow-up"
    },
    {
      "name": "calculate_growth_percentile",
      "parameters": {
        "measurement_type": "weight",
        "value": 12.5,
        "age_months": 36,
        "sex": "male"
      },
      "date_context": "current",
      "reason": "Assess current weight percentile for malnutrition"
    }
  ]
}
```

The date_context flows through:
- Stage 1: Planning
- Stage 2: Execution (logged)
- Stage 3: Available in formatted output for LLM

---

## Function Parameter Documentation

### Example: `calculate_growth_percentile`

**From `functions/calculate_growth_percentile.json`:**

```json
{
  "parameters": {
    "measurement_type": {
      "type": "string",
      "description": "Type of measurement: 'weight', 'height', 'stature', or 'bmi'"
    },
    "value": {
      "type": "number",
      "description": "Measurement value (kg for weight, cm for height, kg/m² for BMI)"
    },
    "age_months": {
      "type": "number",
      "description": "Age in months"
    },
    "sex": {
      "type": "number",
      "description": "Sex: 1 for male, 2 for female. Can also pass string 'male' or 'female' which will be automatically converted."
    },
    "height_cm": {
      "type": "number",
      "description": "Height in cm (only for weight-for-stature calculations)"
    }
  }
}
```

**Function Docstring:**

```python
def calculate_growth_percentile(measurement_type, value, age_months, sex, height_cm=None):
    '''
    Calculate growth percentile using CDC data

    Args:
        measurement_type: 'weight', 'height', 'stature', or 'bmi'
        value: measurement value (kg for weight, cm for height, kg/m² for BMI)
        age_months: age in months
        sex: 1 for male, 2 for female
             NOTE: The system automatically converts string values:
                   'male', 'm', 'boy', 'man' → 1
                   'female', 'f', 'girl', 'woman' → 2
             You can pass either format, the function registry will handle conversion.
        height_cm: height in cm (only needed for weight-for-stature)

    Returns:
        Dict with percentile, z_score, and metric information
    '''
```

---

## Parameter Extraction Best Practices

### For Function Developers:

1. **Document parameter types clearly** in JSON schema
2. **Use standard names** (weight_kg, height_cm, age_months, sex)
3. **Specify units** in descriptions
4. **Note any automatic conversions** in docstrings
5. **Provide examples** in parameter descriptions

### For Prompt Engineers:

1. **Instruct agent to extract with precision** ("45.5 kg" → 45.5, not "45 kg")
2. **Map to correct parameter names** (weight → weight_kg)
3. **Pass sex as human-readable string** ("male" not 1)
4. **Include temporal context** for serial measurements
5. **Document conventions** clearly in prompt

### For Users:

1. **Write clinical text naturally** ("3-year-old male")
2. **Include units** ("12.5 kg" not just "12.5")
3. **Specify dates** for serial measurements
4. **Use standard terminology** (male/female, not M/F codes)

---

## Debugging Parameter Issues

### Common Issues and Solutions:

**Issue 1: Function receives wrong type**
```
ERROR: could not convert string to float: 'male'
```
**Solution:** Check if transformation is defined in `_apply_parameter_transformations()`

**Issue 2: Missing required parameter**
```
ERROR: missing 1 required positional argument: 'sex'
```
**Solution:** Verify parameter name mapping in Stage 1 prompt

**Issue 3: Unit mismatch**
```
Result seems wrong (e.g., BMI = 16500 instead of 16.5)
```
**Solution:** Check if unit conversion is needed (cm → m)

---

## Testing Parameter Extraction

### Example Clinical Text:

```
Patient is a 3-year-old male child presenting with poor growth.

Anthropometrics:
- Weight: 12.5 kg (measured today)
- Previous weight: 13.0 kg (3 months ago)
- Height: 92 cm
- BMI: 14.8 kg/m² (calculated)

Assessment: Low weight percentile, possible malnutrition.
```

### Expected Function Calls:

```json
[
  {
    "name": "calculate_growth_percentile",
    "parameters": {
      "measurement_type": "weight",
      "value": 12.5,
      "age_months": 36,
      "sex": "male"  // Will be converted to 1
    },
    "date_context": "current"
  },
  {
    "name": "calculate_growth_percentile",
    "parameters": {
      "measurement_type": "weight",
      "value": 13.0,
      "age_months": 33,
      "sex": "male"  // Will be converted to 1
    },
    "date_context": "3 months ago"
  },
  {
    "name": "calculate_growth_percentile",
    "parameters": {
      "measurement_type": "height",
      "value": 92,
      "age_months": 36,
      "sex": "male"  // Will be converted to 1
    },
    "date_context": "current"
  }
]
```

---

## Summary

**Key Points:**

1. ✅ Agent extracts parameters as human-readable values from text
2. ✅ System automatically transforms parameters to function requirements
3. ✅ Sex parameter: strings ('male'/'female') → numbers (1/2)
4. ✅ Backward compatible with numeric codes
5. ✅ Temporal context preserved for serial measurements
6. ✅ Clear documentation throughout prompt and function definitions
7. ✅ Comprehensive logging for debugging

**The system handles the complexity of parameter transformation automatically, allowing the LLM agent to focus on intelligent extraction rather than format conversion.**
