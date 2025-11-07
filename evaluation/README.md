# ClinOrchestra System Evaluation

This directory contains tools and resources for evaluating your ClinOrchestra system performance.

## Quick Start

### 1. Prepare Your Data

**Gold Standard Format** (example: `gold_standard.csv`):
```csv
id,text,label,extracted_value,reasoning
001,"Patient has BMI of 16.2 kg/m2",1,16.2,"BMI < 18.5 indicates malnutrition"
002,"Patient has normal weight",0,22.1,"BMI within normal range"
003,"Creatinine 2.1 mg/dL on 3/15/2024",1,2.1,"Elevated creatinine suggests renal impairment"
```

**System Output Format** (from ClinOrchestra processing):
```csv
id,text,label,extracted_value,reasoning
001,"Patient has BMI of 16.2 kg/m2",1,16.2,"BMI is 16.2 which is below 18.5..."
002,"Patient has normal weight",0,22.1,"BMI is 22.1 which is normal..."
003,"Creatinine 2.1 mg/dL on 3/15/2024",1,2.1,"Creatinine 2.1 is elevated..."
```

### 2. Run Evaluation

```bash
# Classification evaluation
python evaluate_system.py \
    --gold gold_standard.csv \
    --system system_output.csv \
    --task classification \
    --output results.json

# Extraction evaluation
python evaluate_system.py \
    --gold gold_standard.csv \
    --system system_output.csv \
    --task extraction \
    --output results.json

# Both
python evaluate_system.py \
    --gold gold_standard.csv \
    --system system_output.csv \
    --task both \
    --output results.json
```

### 3. Review Results

The script generates:
- `results.json` - Full evaluation metrics
- `results_errors.csv` - Detailed error analysis

## Public Datasets for Benchmarking

### 1. i2b2/n2c2 Challenges (RECOMMENDED)

**Access:** https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

**Available Datasets:**
- **2008 Obesity Challenge**: Obesity + 15 comorbidities (asthma, diabetes, hypertension, etc.)
  - ~1000 discharge summaries
  - Binary labels for each condition
  - Perfect for testing your classification

- **2018 Cohort Selection**: Patient eligibility criteria
  - 13 selection criteria tasks
  - Good for complex clinical reasoning

- **2022 Contextualized Medication Events**
  - Medication extraction with context
  - Tests extraction + reasoning

**How to Access:**
1. Go to https://portal.dbmi.hms.harvard.edu/
2. Create account
3. Sign Data Use Agreement
4. Download datasets (usually approved within 1-2 days)

**Example i2b2 2008 Task:**
```
Text: "Patient is obese with BMI of 32. Has type 2 diabetes managed with metformin."

Gold Standard Labels:
- Obesity: Present
- Diabetes: Present
- Hypertension: Absent
- CAD: Absent
... (15 conditions total)

Your Task: Configure ClinOrchestra to predict all 15 labels
```

### 2. MIMIC-III/MIMIC-IV (For Serial Measurements)

**Access:** https://physionet.org/content/mimiciii/

**Requirements:**
1. Complete CITI "Data or Specimens Only Research" course
2. Create PhysioNet account
3. Request credentialed access (~1 week approval)

**Perfect For Testing:**
- Serial creatinine → AKI detection
- Serial weights → Malnutrition tracking
- Serial vitals → Sepsis screening
- Temporal reasoning capabilities

**Example MIMIC Task:**
```
Patient measurements:
- Day 1: Cr 1.0 mg/dL
- Day 3: Cr 1.8 mg/dL (80% increase)
- Day 5: Cr 2.2 mg/dL

Gold Standard: AKI Stage 2 on Day 3

Your Task: Test if ClinOrchestra detects AKI from serial Cr
```

### 3. Quick Start with Synthetic Data

If you don't have access to public datasets yet, create synthetic test cases:

```python
# Example: Generate synthetic malnutrition cases
import pandas as pd
import random

def generate_test_cases(n=100):
    cases = []
    for i in range(n):
        bmi = random.uniform(14, 30)
        label = 1 if bmi < 18.5 else 0
        text = f"Patient has BMI of {bmi:.1f} kg/m2"

        cases.append({
            'id': f'SYNTH_{i:03d}',
            'text': text,
            'label': label,
            'extracted_value': bmi,
            'reasoning': f"BMI {bmi:.1f} is {'below' if label else 'above'} 18.5"
        })

    return pd.DataFrame(cases)

# Generate gold standard
gold_df = generate_test_cases(100)
gold_df.to_csv('synthetic_gold_standard.csv', index=False)

# Process through ClinOrchestra to get system output
# Then evaluate!
```

## Evaluation Workflow

```
┌─────────────────────────────────────────┐
│ 1. Get Public Dataset (i2b2, MIMIC)   │
│    OR Create Gold Standard Annotations │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ 2. Format Data for ClinOrchestra         │
│    - Input CSV with clinical text       │
│    - Task configuration                 │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ 3. Process Through ClinOrchestra         │
│    - Run batch processing               │
│    - Save output CSV                    │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ 4. Run Evaluation Script                │
│    python evaluate_system.py ...        │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ 5. Analyze Results                      │
│    - Review metrics (accuracy, F1)      │
│    - Error analysis                     │
│    - Identify failure patterns          │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ 6. Improve System                       │
│    - Refine prompts                     │
│    - Add functions/patterns             │
│    - Update RAG documents               │
└────────────────┬────────────────────────┘
                 │
                 ▼
        (Repeat from step 3)
```

## Metrics Interpretation

### Classification Metrics

| Metric | Good | Excellent | Interpretation |
|--------|------|-----------|----------------|
| Accuracy | >0.80 | >0.90 | Overall correctness |
| Precision | >0.80 | >0.90 | How many predictions are correct |
| Recall | >0.80 | >0.90 | How many cases did we catch |
| F1 Score | >0.80 | >0.90 | Balance of precision/recall |
| Cohen's Kappa | >0.60 | >0.80 | Agreement beyond chance |

### Extraction Metrics

| Metric | Good | Excellent | Interpretation |
|--------|------|-----------|----------------|
| Exact Match | >0.75 | >0.90 | Extracted exact value |
| Partial Match | >0.85 | >0.95 | Extracted similar value |
| Extraction Precision | >0.80 | >0.95 | Few false extractions |
| Extraction Recall | >0.80 | >0.95 | Few missed extractions |

### Benchmark Comparisons

**i2b2 2008 Obesity Challenge:**
- Rule-based systems: F1 ~0.70-0.80
- ML systems: F1 ~0.85-0.92
- Top systems: F1 ~0.93-0.95

**Your Goal:** Beat rule-based baselines (F1 > 0.80)

## Common Error Patterns to Analyze

1. **Negation errors**: "No evidence of diabetes" → Wrongly labeled as diabetes present
2. **Temporal errors**: "History of diabetes" vs "Current diabetes"
3. **Severity confusion**: "Mild malnutrition" vs "Severe malnutrition"
4. **Extraction precision**: Extracted wrong number from text
5. **Missing context**: Failed to use surrounding clinical context

## Tips for Improvement

If your F1 < 0.80:
1. Review error cases in `*_errors.csv`
2. Add medical patterns for common errors
3. Enhance RAG documents with similar examples
4. Refine Stage 1 analysis prompt
5. Add relevant functions/extras

If your F1 > 0.85:
- Great job! Consider publishing your approach
- Test on more challenging datasets (MIMIC)
- Try multi-task scenarios

## Questions?

Check the main ClinOrchestra documentation or create an issue on GitHub.
