# Serial Measurements Handling in ClinAnnotate

## Overview

The ClinAnnotate agent system is designed to handle **serial/temporal measurements** by calling the same function **multiple times** (once per time point). This is critical for clinical data where trends and progression are important.

---

## Example 1: Serial Creatinine Measurements (CKD Tracking)

### Clinical Note:
```
Patient is a 65-year-old male (70 kg) with worsening renal function.

Lab Results:
- 01/15/2024: Creatinine 1.2 mg/dL (baseline)
- 03/20/2024: Creatinine 1.5 mg/dL (3-month follow-up)
- 06/10/2024: Creatinine 1.8 mg/dL (6-month follow-up)

Assessment: Progressive decline in renal function, consistent with CKD progression.
```

### Agent Stage 1 Analysis Output:

```json
{
  "task_analysis": "Extract renal function data with serial creatinine measurements. Need to calculate eGFR for all three time points to assess CKD progression.",
  "functions_needed": [
    {
      "name": "calculate_creatinine_clearance",
      "parameters": {"creatinine": 1.2, "age": 65, "weight": 70, "sex": "male"},
      "date_context": "2024-01-15 (baseline)",
      "reason": "Calculate eGFR for baseline creatinine to establish baseline renal function"
    },
    {
      "name": "calculate_creatinine_clearance",
      "parameters": {"creatinine": 1.5, "age": 65, "weight": 70, "sex": "male"},
      "date_context": "2024-03-20 (3-month follow-up)",
      "reason": "Calculate eGFR for follow-up creatinine to track renal function decline"
    },
    {
      "name": "calculate_creatinine_clearance",
      "parameters": {"creatinine": 1.8, "age": 65, "weight": 70, "sex": "male"},
      "date_context": "2024-06-10 (6-month follow-up)",
      "reason": "Calculate eGFR for most recent creatinine to assess current renal function"
    }
  ]
}
```

### Agent Stage 2 Execution:

The system will execute `calculate_creatinine_clearance()` **THREE times** (not just once):

```
[CALCULATED VALUES FROM FUNCTIONS]

calculate_creatinine_clearance [2024-01-15 (baseline)]:
  Result: 68 mL/min (CKD Stage 2)

calculate_creatinine_clearance [2024-03-20 (3-month follow-up)]:
  Result: 54 mL/min (CKD Stage 3a)

calculate_creatinine_clearance [2024-06-10 (6-month follow-up)]:
  Result: 45 mL/min (CKD Stage 3b)
```

### Agent Stage 3 Extraction Output:

```json
{
  "creatinine_values": "1.2 mg/dL (01/15/2024), 1.5 mg/dL (03/20/2024), 1.8 mg/dL (06/10/2024)",
  "egfr_values": "68 mL/min (01/15/2024), 54 mL/min (03/20/2024), 45 mL/min (06/10/2024)",
  "ckd_stage_progression": "Stage 2 → Stage 3a → Stage 3b (worsening over 6 months)",
  "trend": "Declining renal function with 23 mL/min decrease in eGFR over 6 months"
}
```

---

## Example 2: Serial Weight Measurements (Malnutrition Assessment)

### Clinical Note:
```
3-year-old child with poor growth:
- Current weight: 12.5 kg (today)
- 3 months ago: 13.0 kg
- 6 months ago: 13.5 kg
Height: 92 cm (unchanged)

Assessment: Progressive weight loss concerning for malnutrition.
```

### Agent Stage 1 Analysis Output:

```json
{
  "functions_needed": [
    {
      "name": "calculate_bmi",
      "parameters": {"weight_kg": 13.5, "height_m": 0.92},
      "date_context": "6 months ago (baseline)",
      "reason": "Calculate baseline BMI to establish growth trajectory"
    },
    {
      "name": "calculate_bmi",
      "parameters": {"weight_kg": 13.0, "height_m": 0.92},
      "date_context": "3 months ago",
      "reason": "Calculate BMI at 3-month point to track weight decline"
    },
    {
      "name": "calculate_bmi",
      "parameters": {"weight_kg": 12.5, "height_m": 0.92},
      "date_context": "current",
      "reason": "Calculate current BMI to assess acute malnutrition status"
    },
    {
      "name": "percentile_to_zscore",
      "parameters": {"percentile": 10},
      "date_context": "current",
      "reason": "Convert current weight percentile to z-score for malnutrition assessment"
    }
  ]
}
```

### Agent Stage 2 Execution:

```
[CALCULATED VALUES FROM FUNCTIONS]

calculate_bmi [6 months ago (baseline)]:
  Result: 15.96 kg/m²

calculate_bmi [3 months ago]:
  Result: 15.37 kg/m²

calculate_bmi [current]:
  Result: 14.78 kg/m²

percentile_to_zscore [current]:
  Result: -1.28
```

---

## Example 3: Serial Blood Pressure Readings

### Clinical Note:
```
Patient with hypertension on new medication:
- Visit 1 (baseline): BP 140/90 mmHg
- Visit 2 (2 weeks): BP 135/85 mmHg
- Visit 3 (4 weeks): BP 130/80 mmHg

Assessment: Good response to antihypertensive therapy.
```

### Agent Stage 1 Analysis Output:

```json
{
  "functions_needed": [
    {
      "name": "calculate_mean_arterial_pressure",
      "parameters": {"systolic": 140, "diastolic": 90},
      "date_context": "baseline",
      "reason": "Calculate baseline MAP"
    },
    {
      "name": "calculate_mean_arterial_pressure",
      "parameters": {"systolic": 135, "diastolic": 85},
      "date_context": "2 weeks",
      "reason": "Calculate MAP at 2-week follow-up"
    },
    {
      "name": "calculate_mean_arterial_pressure",
      "parameters": {"systolic": 130, "diastolic": 80},
      "date_context": "4 weeks",
      "reason": "Calculate current MAP"
    }
  ]
}
```

---

## Key Design Principles

### 1. **Automatic Detection**
The agent automatically detects serial measurements by:
- Looking for temporal markers: "baseline", "3 months ago", dates, "visit 1/2/3"
- Identifying multiple instances of the same measurement type
- Recognizing progression patterns

### 2. **One Function Call Per Time Point**
- If there are 3 creatinine values → 3 function calls
- If there are 5 weight measurements → 5 function calls
- Each call gets its own `date_context` field

### 3. **Temporal Context Preservation**
Every function call includes:
```json
{
  "date_context": "2024-01-15 (baseline)"
}
```

This context is preserved through:
- Stage 1 analysis (planning)
- Stage 2 execution (calculation)
- Stage 3 extraction (final output)

### 4. **Trend Analysis**
With serial data, the agent can:
- Calculate changes over time
- Identify progression or regression
- Assess velocity of change
- Determine clinical significance

### 5. **General Design**
This pattern applies to **ANY** serial measurement:
- Lab values (Cr, HbA1c, albumin, CBC)
- Vital signs (BP, HR, RR, SpO2)
- Anthropometrics (weight, height, BMI)
- Scores/assessments over time
- Any numeric measurement with temporal dimension

---

## Implementation Details

### Stage 1 Prompt Instructions:
```
*** CRITICAL: If there are multiple measurements of the same type at different time points,
    call the SAME function MULTIPLE times (once for each time point) ***

Examples:
- Text: "Creatinine 1.2 mg/dL on 1/15, 1.5 mg/dL on 3/20, 1.8 mg/dL on 6/10"
  → Call calculate_creatinine_clearance() THREE times (once for each Cr value)

- Text: "Weight 12.5 kg today, was 13.0 kg 3 months ago, and 13.5 kg 6 months ago"
  → Call calculate_bmi() THREE times (once for each weight measurement)
```

### Code Implementation:
- `agent_system.py`: Enhanced Stage 1 prompt with serial measurement guidance
- `agent_system.py`: `_convert_task_understanding_to_tool_requests()` preserves `date_context`
- `agent_system.py`: `_execute_function_tool()` includes `date_context` in results
- `prompt_templates.py`: `format_tool_outputs_for_prompt()` displays date context

---

## Testing Serial Measurements

To test the serial measurement handling:

1. Create a clinical note with multiple measurements at different time points
2. Run extraction with appropriate schema
3. Verify that the agent:
   - Detects ALL measurements (not just first/last)
   - Calls the function MULTIPLE times
   - Preserves date/time context
   - Calculates trends and changes

Example test scenarios:
- ✅ Multiple creatinine values for CKD staging
- ✅ Serial weights for malnutrition assessment
- ✅ Serial HbA1c for diabetes control
- ✅ Serial BP readings for hypertension management
- ✅ Serial growth measurements for pediatric assessment

---

## Common Pitfalls (AVOIDED)

❌ **Wrong**: Only calling function once with latest value
```json
{
  "functions_needed": [
    {
      "name": "calculate_creatinine_clearance",
      "parameters": {"creatinine": 1.8, ...}
    }
  ]
}
```

✅ **Correct**: Calling function for ALL time points
```json
{
  "functions_needed": [
    {
      "name": "calculate_creatinine_clearance",
      "parameters": {"creatinine": 1.2, ...},
      "date_context": "baseline"
    },
    {
      "name": "calculate_creatinine_clearance",
      "parameters": {"creatinine": 1.5, ...},
      "date_context": "3 months"
    },
    {
      "name": "calculate_creatinine_clearance",
      "parameters": {"creatinine": 1.8, ...},
      "date_context": "6 months"
    }
  ]
}
```

---

## Conclusion

The ClinAnnotate agent system is designed from the ground up to handle serial/temporal measurements intelligently. By calling functions multiple times with proper date context, the system can:

1. Capture complete longitudinal data
2. Calculate trends and progression
3. Support clinical decision-making based on temporal patterns
4. Handle any type of serial measurement (labs, vitals, anthropometrics, etc.)

This general design pattern ensures that **ALL** data points are processed, not just the first or most recent value.
