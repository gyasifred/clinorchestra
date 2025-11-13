# Evaluation Quick Start Guide

Get your first evaluation running in 10 minutes!

## Step 1: Generate Sample Data (2 minutes)

```bash
cd evaluation

# Generate 100 malnutrition test cases
python create_sample_dataset.py --task malnutrition --samples 100

# This creates: sample_datasets/malnutrition_gold_standard.csv
```

**What you'll see:**
```
Generating 100 malnutrition cases...
  Saved to sample_datasets/malnutrition_gold_standard.csv
  Label distribution: {0: 45, 1: 32, 2: 23}

0 = No malnutrition
1 = Malnutrition risk
2 = Moderate-severe malnutrition
```

## Step 2: Process Through ClinOrchestra (5 minutes)

### Option A: Using Gradio UI

1. Start ClinOrchestra:
```bash
python annotate.py
```

2. In the browser:
   - Go to **Processing Tab**
   - Upload: `evaluation/sample_datasets/malnutrition_gold_standard.csv`
   - **Important**: Map columns:
     - Text column: `text`
     - ID column: `id`
   - Configure task (select malnutrition template or create new)
   - Click "Start Processing"
   - Download results when done

### Option B: Using Python API

```python
import pandas as pd
from core.app_state import AppState
from core.agent_factory import create_agent

# Load test data (only text and id columns)
gold_df = pd.read_csv('evaluation/sample_datasets/malnutrition_gold_standard.csv')
input_df = gold_df[['id', 'text']].copy()

# Save as input file
input_df.to_csv('evaluation/malnutrition_input.csv', index=False)

# Set up ClinOrchestra (configure your task)
app_state = AppState()
# ... configure your malnutrition task ...

# OPTIONAL: Enable Agentic Mode for better accuracy (slower but more thorough)
# app_state.set_agentic_config(enabled=True, max_iterations=20, max_tool_calls=50)

# Create appropriate agent (Classic or Agentic based on config)
agent = create_agent(
    llm_manager=app_state.get_llm_manager(),
    rag_engine=app_state.get_rag_engine(),
    extras_manager=app_state.get_extras_manager(),
    function_registry=app_state.get_function_registry(),
    regex_preprocessor=app_state.get_regex_preprocessor(),
    app_state=app_state
)

# Process
results = []
for idx, row in input_df.iterrows():
    result = agent.extract(clinical_text=row['text'], label_value=row.get('label', ''))
    results.append(result)

# Save output
results_df = pd.DataFrame(results)
results_df.to_csv('evaluation/malnutrition_system_output.csv', index=False)
```

**Note**: ADAPTIVE mode provides 60-75% faster execution due to parallel tool execution and can achieve higher accuracy through iterative refinement. See [SDK_GUIDE.md](../SDK_GUIDE.md) for mode comparison.

## Step 3: Run Evaluation (1 minute)

```bash
python evaluate_system.py \
    --gold sample_datasets/malnutrition_gold_standard.csv \
    --system malnutrition_system_output.csv \
    --task classification \
    --output malnutrition_results.json
```

**You'll see:**

```
============================================================
CLASSIFICATION RESULTS
============================================================
Accuracy:  0.873
Precision: 0.869
Recall:    0.873
F1 Score:  0.868
Cohen's Kappa: 0.794
Samples:   100

Per-Class Performance:
  0: P=0.918, R=0.911, F1=0.914
  1: P=0.781, R=0.812, F1=0.796
  2: P=0.913, R=0.913, F1=0.913

Confusion Matrix:
[[41  3  1]
 [ 4 26  2]
 [ 1  1 21]]

============================================================
ERROR ANALYSIS
============================================================
Total Errors: 13
Error Types:
  misclassification: 8
  false_positive: 3
  false_negative: 2

Full report saved to: malnutrition_results.json
Error details saved to: malnutrition_results_errors.csv
============================================================
```

## Step 4: Analyze Results (2 minutes)

### Review Errors

```bash
# Look at the error cases
head -20 malnutrition_results_errors.csv
```

Example error:
```csv
id,text,label_gold,label_system,error_type,reasoning_system
MAL_042,"Patient has BMI 18.3 kg/m2",1,0,false_negative,"BMI 18.3 is within normal range"
```

**Why did this fail?**
- Gold label: 1 (malnutrition risk, BMI < 18.5 ✓)
- System predicted: 0 (no malnutrition)
- Issue: System may need to be more strict about the 18.5 threshold

### Check Full Metrics

```bash
# View detailed JSON report
cat malnutrition_results.json | python -m json.tool | less
```

## Step 5: Improve and Re-evaluate

Based on errors, improve your system:

### Common Improvements:

**1. Refine Stage 1 Prompt** (`core/agent_system.py`)
```python
# Add specific guidance for edge cases
"""
IMPORTANT: For malnutrition assessment:
- BMI < 18.5 = malnutrition risk (even if close, like 18.3)
- BMI < 16 = severe malnutrition
- Be conservative - borderline cases should be flagged
"""
```

**2. Add Medical Patterns** (`medical_knowledge/patterns/`)
```json
{
  "name": "borderline_malnutrition",
  "pattern": "BMI (17\\.|18\\.[0-4])",
  "label": 1,
  "context": "Borderline low BMI suggests malnutrition risk"
}
```

**3. Update RAG Documents** (`rag_documents/`)
```
Add examples of borderline cases:
- "BMI 18.3 kg/m2 is technically below 18.5 threshold and warrants malnutrition risk classification per WHO criteria."
```

**4. Add Helper Functions**
```python
# functions/classify_malnutrition_by_bmi.py
def classify_malnutrition_by_bmi(bmi: float) -> int:
    """
    Classify malnutrition status by BMI (WHO criteria)

    Returns:
        0: Normal (BMI ≥ 18.5)
        1: Malnutrition risk (16 ≤ BMI < 18.5)
        2: Severe malnutrition (BMI < 16)
    """
    if bmi < 16:
        return 2
    elif bmi < 18.5:
        return 1
    else:
        return 0
```

### Re-run Evaluation

```bash
# Process again with improvements
python annotate.py  # or use API

# Evaluate again
python evaluate_system.py \
    --gold sample_datasets/malnutrition_gold_standard.csv \
    --system malnutrition_system_output_v2.csv \
    --task classification \
    --output malnutrition_results_v2.json

# Compare results
echo "Version 1: F1 = 0.868"
echo "Version 2: F1 = ???"
```

## Iterative Improvement Cycle

```
┌─────────────────────────────────┐
│ 1. Evaluate (F1 = 0.87)        │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ 2. Analyze errors (13 errors)  │
│    - 8 borderline BMI cases     │
│    - 3 albumin interpretation   │
│    - 2 z-score confusion        │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ 3. Make improvements            │
│    - Refined prompt             │
│    - Added BMI function         │
│    - Enhanced RAG docs          │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ 4. Re-evaluate (F1 = 0.94!)    │
└────────────┬────────────────────┘
             │
             ▼
        (Success! Or iterate again)
```

## Target Metrics by Iteration

**First Evaluation (Baseline):**
- F1 = 0.70-0.85 (typical)
- Many edge case errors
- Goal: Identify error patterns

**After 1-2 Iterations:**
- F1 = 0.85-0.90
- Most obvious errors fixed
- Goal: Handle edge cases

**After 3-5 Iterations:**
- F1 = 0.90-0.95
- Production-ready performance
- Goal: Optimize for rare cases

**Research-Grade:**
- F1 > 0.95
- Beats most published systems
- Ready for publication!

## Next Steps

Once you're happy with synthetic data:

1. **Try i2b2 Dataset**: Real clinical notes, established benchmarks
   ```bash
   # After getting i2b2 access
   python evaluate_system.py \
       --gold i2b2_2008_obesity_test.csv \
       --system i2b2_system_output.csv \
       --task classification

   # Compare to published baselines
   # Your F1: ???
   # Rule-based: ~0.75
   # ML systems: ~0.88
   # Top systems: ~0.93
   ```

2. **Test Serial Measurements**: Use MIMIC-III for temporal tasks
   ```bash
   python create_sample_dataset.py --task serial --samples 50
   # Then process and evaluate
   ```

3. **Multi-Task Evaluation**: Test on multiple clinical tasks
   ```bash
   # Generate all task types
   python create_sample_dataset.py --task all --samples 100

   # Evaluate each
   for task in malnutrition aki diabetes serial; do
       python evaluate_system.py \
           --gold sample_datasets/${task}_gold_standard.csv \
           --system ${task}_output.csv \
           --task classification \
           --output ${task}_results.json
   done
   ```

4. **Cross-Task Performance**: Does malnutrition tuning hurt AKI performance?

5. **Publication**: If F1 > 0.90 on i2b2, consider writing it up!

## Troubleshooting

**Q: My F1 is < 0.60, what's wrong?**
- Check if system is even running correctly
- Review a few predictions manually
- Ensure task configuration matches test data

**Q: System predicts all 0s or all 1s**
- Label imbalance issue
- Add class-specific examples to RAG
- Adjust classification threshold

**Q: Evaluation script fails on merge**
- ID columns don't match
- Check: `gold_df['id']` vs `system_df['id']`
- Make sure IDs are preserved during processing

**Q: How long should processing take?**
- ~100 samples: 5-10 minutes
- ~1000 samples: 30-60 minutes
- Depends on: API speed, task complexity, batch size

## Summary

You now have:
- ✅ Sample test datasets
- ✅ Evaluation metrics pipeline
- ✅ Error analysis workflow
- ✅ Improvement iteration cycle

**Your goal**: Achieve F1 > 0.85 on synthetic data, then test on i2b2 real clinical data!

Good luck! 🚀
