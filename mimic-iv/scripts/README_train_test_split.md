# Train/Test Split Script Usage Guide

## Overview

The `create_balanced_train_test.py` script intelligently samples 5,000 cases from your large MIMIC-IV dataset and creates balanced train (4,000) and test (1,000) sets.

## Features

### Multi-Dimensional Balancing

The script balances across **4 dimensions simultaneously**:

1. **Diagnoses**: Proportional to original distribution across top 20 diagnoses
2. **Gender**: M/F/Unknown
3. **Race/Ethnicity**: WHITE, BLACK, ASIAN, HISPANIC, OTHER, UNKNOWN
4. **Text Complexity**: Quartiles (Q1=short, Q2=medium, Q3=long, Q4=very long)

### Smart Sampling Strategy

- **Stratified Sampling**: Within each diagnosis, samples are balanced by gender, race, and text length
- **Proportional Selection**: Maintains original diagnosis distribution
- **Memory Efficient**: Reads large CSV in chunks to avoid memory issues
- **Reproducible**: Fixed random seed (42) for consistent results

## Requirements

```bash
pip install pandas numpy scikit-learn
```

## Usage

### Step 1: Check Configuration

Edit the script to set your paths:

```python
INPUT_FILE = r"C:\Users\gyasi\Documents\mimic-iv-3.1\outputs\classification_dataset.csv"
OUTPUT_DIR = r"C:\Users\gyasi\Documents\mimic-iv-3.1\outputs"
```

### Step 2: Run the Script

```bash
python create_balanced_train_test.py
```

### Step 3: Review Output

The script will:
1. Read your large dataset (~582MB)
2. Calculate text complexity
3. Perform balanced sampling
4. Split into train/test
5. Display detailed distribution statistics
6. Save files

## Output Files

Three files will be created in `OUTPUT_DIR`:

1. **train_dataset_4000.csv** - Training set (4,000 cases)
2. **test_dataset_1000.csv** - Test set (1,000 cases)
3. **train_test_split_metadata.txt** - Metadata and statistics

## Distribution Statistics

The script prints comprehensive statistics for:
- Original full dataset
- Sampled 5,000 cases
- Train set (4,000)
- Test set (1,000)

For each dataset, you'll see:
- Diagnosis distribution
- Gender distribution
- Race distribution
- Text complexity distribution
- Text length statistics (mean, median, quartiles)

## Example Output

```
============================================================
TRAIN SET (4000 cases) - Distribution Statistics
============================================================
Total samples: 4000

------------------------------------------------------------
Diagnosis Distribution:
------------------------------------------------------------
  78650: 1168 (29.20%)
  78659:  834 (20.85%)
  R079:   465 (11.63%)
  ...

------------------------------------------------------------
Gender Distribution:
------------------------------------------------------------
  F: 1820 (45.50%)
  M: 2150 (53.75%)
  U:   30 ( 0.75%)

------------------------------------------------------------
Race Distribution:
------------------------------------------------------------
  WHITE:    2400 (60.00%)
  BLACK:     800 (20.00%)
  ASIAN:     320 ( 8.00%)
  HISPANIC:  280 ( 7.00%)
  OTHER:     200 ( 5.00%)

------------------------------------------------------------
Text Complexity Distribution:
------------------------------------------------------------
  Q1_short:      1000 (25.00%)
  Q2_medium:     1000 (25.00%)
  Q3_long:       1000 (25.00%)
  Q4_very_long:  1000 (25.00%)
```

## Text Complexity Calculation

Text complexity is determined by character length quartiles:

- **Q1 (Short)**: ≤ 25th percentile (e.g., < 2,000 characters)
- **Q2 (Medium)**: 25th-50th percentile (e.g., 2,000-5,000 characters)
- **Q3 (Long)**: 50th-75th percentile (e.g., 5,000-10,000 characters)
- **Q4 (Very Long)**: > 75th percentile (e.g., > 10,000 characters)

This ensures your training set includes diverse clinical notes from brief ED notes to comprehensive discharge summaries.

## Race Standardization

The script standardizes race/ethnicity into consistent categories:

| Original Values | Standardized |
|-----------------|--------------|
| WHITE, CAUCASIAN | WHITE |
| BLACK, AFRICAN AMERICAN | BLACK |
| ASIAN | ASIAN |
| HISPANIC, LATINO | HISPANIC |
| UNKNOWN, UNABLE TO OBTAIN, DECLINED | UNKNOWN |
| All others | OTHER |

## Stratification Strategy

### Level 1: Diagnosis
- Calculate proportion of each diagnosis in original dataset
- Sample proportionally (e.g., if 29% are chest pain, sample 1,450 chest pain cases)

### Level 2: Within Each Diagnosis
- Create strata by combining: Gender × Race × Text Complexity
- Sample proportionally from each stratum
- If a stratum is too small, take all available cases

### Level 3: Train/Test Split
- Use stratified split maintaining diagnosis distribution
- Fallback to diagnosis-only stratification if some strata are too small

## Troubleshooting

### Error: "Input file not found"
- Check the `INPUT_FILE` path
- Ensure you're using the correct directory separators (Windows: `\`, Linux: `/`)

### Error: "Could not find clinical text column"
- The script looks for: `clinical_text`, `text`, `note_text`, `discharge_text`
- Check your CSV column names
- Edit the `text_columns` list in the script if needed

### Error: "Could not find diagnosis column"
- The script looks for: `icd_code`, `diagnosis_code`, `primary_diagnosis`
- Check your CSV column names
- Edit the diagnosis detection logic if needed

### Warning: "Some strata too small for perfect stratification"
- This is normal if some diagnosis+gender+race+complexity combinations have <2 cases
- The script will use diagnosis-only stratification as fallback
- Balance will still be maintained for diagnoses

### Memory Issues
- The script reads in 10,000-row chunks to minimize memory usage
- If still encountering issues, reduce `chunk_size` in the script
- Ensure you have at least 2GB free RAM

## Customization

### Change Train/Test Ratio

Edit these constants:
```python
TOTAL_SAMPLES = 5000  # Total to select
TRAIN_SIZE = 4000     # Training set
TEST_SIZE = 1000      # Test set
```

### Change Random Seed

For different random sampling:
```python
RANDOM_SEED = 42  # Change to any integer
```

### Add More Balancing Dimensions

To add more dimensions (e.g., age groups):

1. Add preprocessing:
```python
df['age_group'] = pd.cut(df['age'], bins=[0, 40, 60, 80, 120],
                         labels=['young', 'middle', 'senior', 'elderly'])
```

2. Update stratification:
```python
stratify_columns = [diagnosis_col, 'gender', 'race_category',
                    'text_complexity', 'age_group']
```

## Best Practices

### Do's ✅

- **Always review distribution statistics** before using the datasets
- **Keep train and test separate** throughout entire pipeline
- **Use the same random seed** for reproducibility
- **Document any customizations** you make to the script
- **Validate** that your model never sees test data during training

### Don'ts ❌

- **Don't modify** train or test sets after creation
- **Don't combine** train and test sets
- **Don't peek** at test set during model development
- **Don't use test set** for hyperparameter tuning (use validation split from train)
- **Don't share** test set labels with your model before final evaluation

## Next Steps After Creating Datasets

### 1. Quality Check (5-10 minutes)
```python
import pandas as pd

train = pd.read_csv('train_dataset_4000.csv')
test = pd.read_csv('test_dataset_1000.csv')

# Check for data leakage
train_ids = set(train['subject_id'].astype(str) + '_' + train['hadm_id'].astype(str))
test_ids = set(test['subject_id'].astype(str) + '_' + test['hadm_id'].astype(str))
overlap = train_ids & test_ids
print(f"Overlap: {len(overlap)} cases")  # Should be 0

# Check distributions
print("Train diagnosis distribution:")
print(train['icd_code'].value_counts())
print("\nTest diagnosis distribution:")
print(test['icd_code'].value_counts())
```

### 2. Create Validation Split from Train (Recommended)

For hyperparameter tuning, create a validation set:

```python
from sklearn.model_selection import train_test_split

train = pd.read_csv('train_dataset_4000.csv')

# 80% train, 20% validation (3200/800)
train_final, val = train_test_split(
    train,
    test_size=0.2,
    stratify=train['icd_code'],
    random_state=42
)

train_final.to_csv('train_final_3200.csv', index=False)
val.to_csv('validation_800.csv', index=False)
```

### 3. Use with ClinOrchestra

**For Task 1 (Annotation)**:
- Upload `train_dataset_4000.csv` to ClinOrchestra Data Tab
- Process annotations
- Save as `train_annotations_4000.json`

**For Task 2 (Classification)**:
- Upload `train_dataset_4000.csv` to ClinOrchestra Data Tab
- Process predictions
- Save as `train_predictions_4000.json`

**For Evaluation**:
- After model training, process `test_dataset_1000.csv`
- Compare predictions vs ground truth
- Calculate metrics

### 4. Evaluation Metrics

Use the provided evaluation script:

```bash
python evaluate_classification.py \
    --predictions test_predictions_1000.json \
    --ground_truth test_dataset_1000.csv \
    --output evaluation_results.json
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the distribution statistics for anomalies
3. Verify your input CSV has expected columns
4. Check that paths are correct for your OS

## Technical Details

### Time Complexity
- Reading dataset: O(n) where n = total rows
- Text length calculation: O(n)
- Stratified sampling: O(n × log n)
- Train/test split: O(m × log m) where m = 5000
- **Total runtime**: ~2-5 minutes for 76,594 rows

### Space Complexity
- Chunk reading: O(chunk_size) = O(10,000) = ~50-100 MB
- Full dataset in memory: O(n) = ~600 MB
- Output datasets: O(m) = ~40 MB
- **Peak memory usage**: ~800 MB - 1 GB

### Reproducibility
- Fixed random seed ensures same results every run
- Deterministic stratification algorithm
- Documented all preprocessing steps
- Metadata file records all parameters

## Version History

- **v1.0** (2025-11-10): Initial release
  - Multi-dimensional stratified sampling
  - Automatic text complexity calculation
  - Race standardization
  - Comprehensive distribution statistics
  - Memory-efficient chunk reading

---

**Author**: Claude (Anthropic)
**Date**: 2025-11-10
**For**: MIMIC-IV Clinical Consultant AI Training Project
