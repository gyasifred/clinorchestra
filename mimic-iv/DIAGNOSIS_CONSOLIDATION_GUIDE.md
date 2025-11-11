# Diagnosis Consolidation Guide
# MIMIC-IV Top Diagnoses - Unified Clinical Conditions

**Date**: 2025-11-11
**Purpose**: Consolidate same clinical conditions with different ICD codes (ICD-9 vs ICD-10) into single diagnoses

---

## OVERVIEW

### Previous System: 20 Individual ICD Codes
The original system treated each ICD code as a separate diagnosis, even when they represented the same clinical condition:
- ❌ "Chest pain (78650)" - 7,297 cases
- ❌ "Other chest pain (78659)" - 5,212 cases
- ❌ "Chest pain (R079)" - 2,906 cases

### New System: 13 Consolidated Diagnoses
Same clinical conditions are now unified:
- ✅ **"Chest pain"** - 15,415 cases (combines 78650 + 78659 + R079)

**Result**: 20 ICD codes → 13 unique clinical diagnoses

---

## CONSOLIDATED DIAGNOSES

### Summary
| ID | Diagnosis Name | Category | Total Cases | ICD-9 Codes | ICD-10 Codes |
|----|----------------|----------|-------------|-------------|--------------|
| 1 | Chest pain | Cardiovascular | 15,415 | 78650, 78659 | R079 |
| 2 | Coronary atherosclerosis | Cardiovascular | 5,751 | 41401 | - |
| 3 | NSTEMI | Cardiovascular | 3,265 | - | I214 |
| 4 | Subendocardial infarction | Cardiovascular | 2,873 | 41071 | - |
| 5 | Atrial fibrillation | Cardiovascular | 3,090 | 42731 | - |
| 6 | Hypertensive heart disease with CKD | Cardiovascular | 3,313 | - | I130 |
| 7 | **Sepsis** | Infectious | **8,239** | **0389, 389** | **A419** |
| 8 | Pneumonia | Infectious | 3,726 | 486 | - |
| 9 | Urinary tract infection | Infectious | 2,967 | 5990 | - |
| 10 | **Acute kidney injury** | Renal | **6,185** | **5849** | **N179** |
| 11 | **Depression** | Psychiatric | **7,202** | **311** | **F329** |
| 12 | **Alcohol use disorder** | Psychiatric | **8,038** | **30500** | **F10129** |
| 13 | **Chemotherapy encounter** | Oncology | **6,530** | **V5811** | **Z5111** |

**Bold** = Multiple ICD codes consolidated

---

## FILES UPDATED

### 1. Core Mapping (`diagnosis_mapping.py`)
**NEW FILE** - Central source of truth for all diagnosis consolidation

```python
from diagnosis_mapping import (
    DIAGNOSIS_CONSOLIDATION,      # ICD code → diagnosis mapping
    CONSOLIDATED_DIAGNOSES,        # Diagnosis ID → full info
    get_consolidated_diagnosis,    # Helper function
    get_diagnosis_info            # Helper function
)
```

**Key Functions**:
```python
# Get consolidated diagnosis from ICD code
diagnosis_name, category, diagnosis_id = get_consolidated_diagnosis('78650')
# Returns: ('Chest pain', 'cardiovascular', 1)

# Get full diagnosis info
info = get_diagnosis_info(1)
# Returns: {..., 'all_codes': ['78650', '78659', 'R079'], 'total_cases': 15415, ...}
```

### 2. Data Extraction Scripts

#### Original: `scripts/get_top_diagnoses_simple.py` (20 ICD codes)
#### NEW: `scripts/get_top_diagnoses_consolidated.py` (13 diagnoses) ✅

**Usage**:
```bash
python scripts/get_top_diagnoses_consolidated.py /path/to/mimic-iv
```

**Output**:
- `mimic-iv/top_diagnoses_consolidated.csv` - 13 consolidated diagnoses
- `mimic-iv/top_diagnoses_detailed_breakdown.csv` - ICD code details

#### Original: `scripts/extract_dataset.py` (20 ICD codes)
#### NEW: `scripts/extract_dataset_consolidated.py` (13 diagnoses) ✅

**Usage**:
```bash
python scripts/extract_dataset_consolidated.py
```

**Key Changes**:
- Adds `consolidated_diagnosis_id` (1-13)
- Adds `consolidated_diagnosis_name` ("Chest pain", "Sepsis", etc.)
- Adds `consolidated_category` ("cardiovascular", "infectious", etc.)
- Keeps `icd_code` and `original_icd_description` for reference

**Output Columns**:
```
BEFORE (20 ICD codes):
  - icd_code: "78650"
  - icd_version: 9
  - primary_diagnosis_name: "Chest pain, unspecified"

AFTER (13 diagnoses):
  - consolidated_diagnosis_id: 1
  - consolidated_diagnosis_name: "Chest pain"
  - consolidated_category: "cardiovascular"
  - icd_code: "78650" (kept for reference)
  - icd_version: 9
  - original_icd_description: "Chest pain, unspecified"
```

### 3. Prompts

#### Original: `prompts/task2_classification_prompt_FINAL.txt` (20 diagnoses)
#### NEW: `prompts/task2_classification_prompt_CONSOLIDATED.txt` (13 diagnoses) ✅

**Key Changes**:
- Lists 13 consolidated diagnoses instead of 20
- Shows all ICD codes within each consolidated diagnosis
- Emphasizes that ICD-9 and ICD-10 versions are unified
- Probability distribution must sum to 1.0 across 13 (not 20) diagnoses

**Example Excerpt**:
```
7. Sepsis - 8,239 cases
   ICD-9: 0389 (Unspecified septicemia), 389 (Septicemia)
   ICD-10: A419 (Sepsis, unspecified organism)
   Note: Life-threatening organ dysfunction from infection (ICD-9 and ICD-10 consolidated)
```

### 4. Schemas

#### Original: `schemas/task2_classification_schema_v2.json` (20 diagnoses)
#### NEW: `schemas/task2_classification_schema_CONSOLIDATED.json` (13 diagnoses) ✅

**Key Changes**:
- `minItems: 13`, `maxItems: 13` (was 20)
- Adds `diagnosis_id` field (1-13)
- Adds `diagnosis_name` enum with 13 values
- Adds `category` field (cardiovascular, infectious, renal, psychiatric, oncology)
- Adds `icd_codes_included` array showing all ICD codes in consolidated diagnosis
- New field: `icd_code_consolidation_notes` in clinical_reasoning

**Example Structure**:
```json
{
  "diagnosis_id": 1,
  "diagnosis_name": "Chest pain",
  "category": "cardiovascular",
  "icd_codes_included": ["78650", "78659", "R079"],
  "probability": 0.65,
  ...
}
```

### 5. Train/Test Split & EDA Scripts

#### Files to Update:
- `scripts/create_balanced_train_test.py` → Use `consolidated_diagnosis_id` for balancing
- `scripts/eda_train_test_publication.py` → Update for 13 diagnoses

**Changes Needed**:
```python
# OLD: Balance across 20 ICD codes
df.groupby('icd_code')...

# NEW: Balance across 13 consolidated diagnoses
df.groupby('consolidated_diagnosis_id')...
# OR
df.groupby('consolidated_diagnosis_name')...
```

**Note**: These scripts should be run on datasets created with `extract_dataset_consolidated.py`

---

## HOW TO USE THE NEW SYSTEM

### Step 1: Extract Consolidated Top Diagnoses
```bash
python mimic-iv/scripts/get_top_diagnoses_consolidated.py /path/to/mimic-iv
```
**Output**: `mimic-iv/top_diagnoses_consolidated.csv`

### Step 2: Extract Datasets with Consolidation
```bash
python mimic-iv/scripts/extract_dataset_consolidated.py
```
**Input**: Prompts for MIMIC-IV path
**Output**:
- `mimic-iv/annotation_dataset_consolidated.csv`
- `mimic-iv/classification_dataset_consolidated.csv`

### Step 3: Create Train/Test Split
```bash
# UPDATE THIS SCRIPT to use consolidated_diagnosis_id
python mimic-iv/scripts/create_balanced_train_test.py
```

### Step 4: Run ClinOrchestra with Updated Files
```
1. Load extras (same 15 files - already support both ICD-9 and ICD-10)
2. Load patterns (same 12 files)
3. Load functions (same 7 files)
4. Use CONSOLIDATED prompt: prompts/task2_classification_prompt_CONSOLIDATED.txt
5. Use CONSOLIDATED schema: schemas/task2_classification_schema_CONSOLIDATED.json
6. Process datasets created in Step 2
```

### Step 5: Evaluate Results
- Accuracy metrics calculated on 13 diagnoses (not 20)
- Confusion matrix: 13×13 (not 20×20)
- Per-diagnosis metrics for 13 diagnoses

---

## MIGRATION NOTES

### For Existing Datasets
If you already have datasets with individual ICD codes, you can add consolidated diagnosis columns:

```python
import sys
sys.path.append('mimic-iv')
from diagnosis_mapping import get_consolidated_diagnosis

# Add consolidated columns
df['consolidated_diagnosis_id'] = df['icd_code'].apply(
    lambda x: get_consolidated_diagnosis(x)[2] if get_consolidated_diagnosis(x) else None
)
df['consolidated_diagnosis_name'] = df['icd_code'].apply(
    lambda x: get_consolidated_diagnosis(x)[0] if get_consolidated_diagnosis(x) else None
)
df['consolidated_category'] = df['icd_code'].apply(
    lambda x: get_consolidated_diagnosis(x)[1] if get_consolidated_diagnosis(x) else None
)
```

### Evaluation Metrics Update
```python
# OLD: 20-class classification
from sklearn.metrics import accuracy_score
y_true = df['icd_code']  # 20 possible values
y_pred = df['predicted_icd_code']  # 20 possible values
accuracy = accuracy_score(y_true, y_pred)

# NEW: 13-class classification
y_true = df['consolidated_diagnosis_id']  # 13 possible values (1-13)
y_pred = df['predicted_diagnosis_id']  # 13 possible values (1-13)
accuracy = accuracy_score(y_true, y_pred)
```

---

## BENEFITS OF CONSOLIDATION

### 1. Clinical Accuracy ✅
- Same disease is treated as same disease regardless of ICD code version
- More clinically meaningful than arbitrary ICD code distinctions

### 2. Increased Sample Sizes ✅
- Chest pain: 7,297 → **15,415** cases (+112%)
- Sepsis: 5,095 → **8,239** cases (+62%)
- AKI: 3,532 → **6,185** cases (+75%)
- Depression: 3,703 → **7,202** cases (+94%)
- Alcohol: 4,591 → **8,038** cases (+75%)
- Chemotherapy: 3,557 → **6,530** cases (+84%)

### 3. Simpler Model ✅
- 13-class problem instead of 20-class
- Easier to train and interpret
- Better performance on common diagnoses

### 4. ICD Version Agnostic ✅
- Works with ICD-9, ICD-10, or mixed data
- Future-proof for ICD-11 migration

---

## BACKWARDS COMPATIBILITY

### Original Files (Still Available)
- `scripts/get_top_diagnoses_simple.py` - Still works for 20 ICD codes
- `scripts/extract_dataset.py` - Still creates 20-code datasets
- `prompts/task2_classification_prompt_FINAL.txt` - Still lists 20 diagnoses
- `schemas/task2_classification_schema_v2.json` - Still supports 20 diagnoses

### When to Use Each
- **Use CONSOLIDATED** (recommended): For clinically meaningful diagnosis prediction
- **Use ORIGINAL**: If you specifically need ICD code-level granularity

---

## REFERENCE: CONSOLIDATION MAPPING

### Diagnoses with Multiple ICD Codes (6 total)

**1. Chest Pain** (3 codes → 1 diagnosis)
- ICD-9 78650: "Chest pain, unspecified" (7,297)
- ICD-9 78659: "Other chest pain" (5,212)
- ICD-10 R079: "Chest pain, unspecified" (2,906)
- **Total**: 15,415 cases

**2. Sepsis** (3 codes → 1 diagnosis)
- ICD-9 0389: "Unspecified septicemia" (3,144)
- ICD-9 389: "Septicemia" (counted in 0389)
- ICD-10 A419: "Sepsis, unspecified organism" (5,095)
- **Total**: 8,239 cases

**3. Acute Kidney Injury** (2 codes → 1 diagnosis)
- ICD-9 5849: "Acute kidney failure, unspecified" (3,532)
- ICD-10 N179: "Acute kidney failure, unspecified" (2,653)
- **Total**: 6,185 cases

**4. Depression** (2 codes → 1 diagnosis)
- ICD-9 311: "Depressive disorder, NEC" (3,499)
- ICD-10 F329: "Major depressive disorder, single episode, unspecified" (3,703)
- **Total**: 7,202 cases

**5. Alcohol Use Disorder** (2 codes → 1 diagnosis)
- ICD-9 30500: "Alcohol abuse, unspecified" (4,591)
- ICD-10 F10129: "Alcohol abuse with intoxication, unspecified" (3,447)
- **Total**: 8,038 cases

**6. Chemotherapy Encounter** (2 codes → 1 diagnosis)
- ICD-9 V5811: "Encounter for antineoplastic chemotherapy" (2,973)
- ICD-10 Z5111: "Encounter for antineoplastic chemotherapy" (3,557)
- **Total**: 6,530 cases

### Diagnoses with Single ICD Code (7 total)
- Coronary atherosclerosis (41401) - 5,751 cases
- NSTEMI (I214) - 3,265 cases
- Subendocardial infarction (41071) - 2,873 cases
- Atrial fibrillation (42731) - 3,090 cases
- Hypertensive heart disease + CKD (I130) - 3,313 cases
- Pneumonia (486) - 3,726 cases
- UTI (5990) - 2,967 cases

---

## TOTAL IMPACT

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Diagnoses** | 20 ICD codes | 13 clinical diagnoses | -35% |
| **Total Cases** | 76,594 | 76,594 | 0% |
| **Avg Cases/Diagnosis** | 3,830 | 5,892 | +54% |
| **Model Complexity** | 20-class | 13-class | -35% |
| **Clinical Accuracy** | ICD-dependent | ICD-agnostic | ✅ |

---

## CONTACT & SUPPORT

- **Diagnosis Mapping**: `mimic-iv/diagnosis_mapping.py`
- **Documentation**: This file
- **ICD Consistency Report**: `mimic-iv/ICD_CODE_CONSISTENCY_REPORT.md`
- **Fixes Applied**: `mimic-iv/ICD_CODE_FIXES_APPLIED.md`

---

**Last Updated**: 2025-11-11
**Version**: 1.0
**Status**: ✅ Ready for Production
