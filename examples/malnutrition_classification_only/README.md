# Pediatric Malnutrition Classification - Criteria-Specific Documentation

This directory contains improved prompts for pediatric malnutrition classification that emphasize **specific criterion documentation** and **assessment-type-appropriate interpretation**.

## Key Improvements

### 1. **Assessment Type Determination**
Prompts now require determining assessment type FIRST:
- **Single-point**: One encounter/measurement set - can only assess current status
- **Serial**: Multiple measurements same encounter
- **Longitudinal**: Multiple measurements across encounters with dates

### 2. **Specific Criteria Documentation**
Instead of generic statements like "based on ASPEN criteria", prompts require:

❌ **Wrong**: "Patient has moderate malnutrition based on ASPEN criteria"

✅ **Correct**: "Moderate malnutrition per ASPEN anthropometric criterion z-score -2 to -2.9 (BMI-for-age z-score -2.3 measured on 3/15/25)"

### 3. **Assessment-Type-Appropriate Interpretation**

**For Single-Point Assessments:**
- Can classify current status using absolute z-scores
- CANNOT assess growth velocity or trends
- Must note limitation and recommend serial measurements

**For Serial/Longitudinal Assessments:**
- Must calculate trends, velocity, percentile trajectory
- Can assess ASPEN growth velocity criterion
- Document changes over time with specific timeframes

## Files

### `main_prompt.txt`
Comprehensive prompt for expert-level clinical annotations to train conversational AI.
- Natural expert language
- Detailed criteria documentation
- Temporal reasoning
- Use with: `schema_main.json`

### `minimal_prompt.txt`
Concise prompt for binary classification of clinical notes.
- Simple Malnourished/Not Malnourished output
- Focused on essential criteria
- Use with: `schema_minimal.json`

### `refinement_prompt.txt`
RAG-enhanced refinement to validate classifications against guidelines.
- Validates criteria specificity
- Corrects z-score sign conventions
- Ensures assessment type matches interpretation
- Verifies ASPEN indicator count
- Use with: `schema_main.json`

### `schema_main.json`
Structured output schema for main/refinement prompts.
- Comprehensive classification details
- Criteria-specific documentation
- Assessment type and confidence

### `schema_minimal.json`
Simple binary classification schema.
- Malnutrition status
- Concise reasoning

## Criterion Documentation Examples

### WHO Classification
```
"Severe acute malnutrition per WHO criterion z-score < -3 (weight-for-height z-score -3.2 on 3/15/25)"
"Moderate acute malnutrition per WHO criterion z-score -3 to -2 (BMI-for-age z-score -2.5 on 2/14/25)"
"Normal nutritional status per WHO criterion z-score -1 to +1 (weight-for-age z-score -0.3)"
```

### ASPEN Anthropometric Deficit
```
"Severe malnutrition per ASPEN anthropometric criterion z-score ≤ -3 (weight-for-height z-score -3.1)"
"Moderate malnutrition per ASPEN anthropometric criterion z-score -2 to -2.9 (BMI-for-age z-score -2.4)"
"Mild malnutrition per ASPEN anthropometric criterion z-score -1 to -1.9 (BMI-for-age z-score -1.5)"
```

### ASPEN Growth Velocity (Longitudinal Only)
```
"Moderate malnutrition per ASPEN velocity criterion decline of 2 z-scores (BMI-for-age declined from -0.5 on 1/15 to -2.5 on 3/15, decline of 2.0 over 59 days)"
"Severe malnutrition per ASPEN velocity criterion decline of 3+ z-scores (weight-for-age declined from +0.2 to -3.1 over 4 months, decline of 3.3)"
```

### ASPEN Inadequate Intake
```
"Inadequate intake per ASPEN criterion <50% estimated needs for ≥1 week (documented intake 30-40% over past 2 weeks per parent report)"
"Intake appears adequate at 80-90% estimated needs, does not meet ASPEN inadequate intake criterion"
```

### ASPEN Physical Findings
```
"Muscle wasting present per ASPEN criterion (temporal wasting, prominent ribs, decreased muscle tone noted on exam 3/15/25)"
"No muscle wasting or fat loss on examination, well-nourished appearance, does not meet ASPEN physical exam criteria"
```

## ASPEN Diagnostic Requirements

**Critical**: ASPEN requires ≥2 indicators for malnutrition diagnosis

Count indicators explicitly:
```
"ASPEN indicators met: 3/4
1. Anthropometric deficit (z-score -2.3)
2. Growth velocity decline (2 z-score drop)
3. Inadequate intake (<50% for 2 weeks)
Physical findings: Not documented

Meets ASPEN diagnostic threshold of ≥2 indicators for malnutrition diagnosis"
```

## Single-Point vs. Longitudinal Example

### Single-Point Assessment
```json
{
  "assessment_type": "Single-point",
  "assessment_justification": "Only one visit documented (3/15/25), no prior measurements available",
  "criteria_satisfied": [
    {
      "criterion": "WHO BMI-for-age z-score classification",
      "specific_documentation": "Moderate acute malnutrition per WHO criterion z-score -3 to -2 (BMI-for-age z-score -2.3 measured on 3/15/25)",
      "severity": "Moderate"
    },
    {
      "criterion": "ASPEN inadequate intake",
      "specific_documentation": "Inadequate intake per ASPEN criterion <50% estimated needs for ≥1 week (intake 30-40% over 2 weeks)",
      "severity": "N/A"
    }
  ],
  "criteria_not_met": ["ASPEN physical findings - no documentation of exam findings"],
  "missing_data": [
    "Prior growth measurements needed to assess ASPEN velocity criterion - recommend serial measurements every 2-4 weeks",
    "Physical exam documentation needed to assess muscle/fat stores"
  ],
  "malnutrition_status": "Present",
  "severity": "Moderate",
  "aspen_indicator_count": "2/4 indicators met (meets threshold of 2+ for diagnosis)",
  "confidence": "Moderate",
  "clinical_reasoning": "Single-point assessment limits ability to assess growth trends. Based on current WHO classification (z-score -2.3) and reported inadequate intake, meets ASPEN threshold with 2 indicators. Cannot assess velocity due to lack of serial measurements - this is a limitation of single-point assessment. Recommend establishing serial measurements to monitor trajectory."
}
```

### Longitudinal Assessment
```json
{
  "assessment_type": "Longitudinal",
  "assessment_justification": "Three visits documented with anthropometric measurements (1/15/25, 2/14/25, 3/15/25)",
  "criteria_satisfied": [
    {
      "criterion": "WHO BMI-for-age z-score classification",
      "specific_documentation": "Moderate acute malnutrition per WHO criterion z-score -3 to -2 (current BMI-for-age z-score -2.3 on 3/15/25, declined from -0.7 on 1/15/25)",
      "severity": "Moderate"
    },
    {
      "criterion": "ASPEN anthropometric deficit",
      "specific_documentation": "Moderate malnutrition per ASPEN anthropometric criterion z-score -2 to -2.9 (BMI-for-age z-score -2.3)",
      "severity": "Moderate"
    },
    {
      "criterion": "ASPEN growth velocity deceleration",
      "specific_documentation": "Moderate malnutrition per ASPEN velocity criterion decline of 2 z-scores (BMI-for-age declined from -0.7 on 1/15 to -2.3 on 3/15, decline of 1.6 z-scores over 59 days, approaching 2 z-score threshold)",
      "severity": "Moderate"
    },
    {
      "criterion": "ASPEN inadequate intake",
      "specific_documentation": "Inadequate intake per ASPEN criterion <50% estimated needs for ≥1 week (intake declined from 80% in January to 30-40% by March)",
      "severity": "N/A"
    }
  ],
  "criteria_not_met": [],
  "missing_data": ["Physical exam documentation would strengthen assessment"],
  "malnutrition_status": "Present",
  "severity": "Moderate",
  "aspen_indicator_count": "4/4 indicators met (exceeds threshold of 2+ for diagnosis)",
  "confidence": "High",
  "clinical_reasoning": "Longitudinal assessment demonstrates clear malnutrition with progressive deterioration. All 4 ASPEN indicators present: anthropometric deficit (z-score -2.3), velocity decline (1.6 z-score drop over 2 months), inadequate intake (30-40%), and documented trend from adequate nutrition to moderate malnutrition. WHO classification confirms moderate acute malnutrition. Serial measurements enable confident trend assessment showing active nutritional decline requiring urgent intervention."
}
```

## Usage

Use these prompts when you need malnutrition **classification** with:
- Specific documentation of which criteria are met
- Appropriate interpretation based on assessment type
- Clear distinction between single-point limitations and longitudinal capabilities
- Explicit ASPEN indicator counting for diagnostic validation

## See Also

- For comprehensive clinical annotation with care plans and monitoring: See `/examples/malnutrition_classification/`
- For z-score interpretation functions: See `/functions/interpret_zscore_malnutrition.py`
- For ASPEN/WHO criteria reference: See `/extras/extra_*_aspen_pediatric_malnutrition_criteria_*.json`
