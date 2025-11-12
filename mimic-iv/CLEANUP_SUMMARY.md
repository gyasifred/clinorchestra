# MIMIC-IV Cleanup Summary

## Date: 2025-11-12

## What Was Done

### 1. Expanded to 20 Consolidated Diagnoses ✅

**Before**: 13 diagnoses
**After**: 20 diagnoses (no ICD-9/10 duplicates)

**New diagnoses added** (#14-20):
- Heart failure (ICD-9: 42833, 42823, 4280 | ICD-10: I5023, I5033, I509)
- Respiratory failure / COPD (ICD-9: 51881, 4941 | ICD-10: J9620, J449)
- Gastrointestinal bleeding (ICD-9: 5789 | ICD-10: K922)
- Hypertension (ICD-9: 4019 | ICD-10: I10)
- Diabetes (ICD-9: 25000 | ICD-10: E119, E1165)
- Chronic kidney disease (ICD-9: 5859 | ICD-10: N189, N183)
- Anemia (ICD-9: 2859 | ICD-10: D649)

**Category breakdown**:
- Cardiovascular: 8 diagnoses
- Infectious: 3 diagnoses
- Renal: 2 diagnoses
- Psychiatric: 2 diagnoses
- Respiratory: 1 diagnosis
- Gastrointestinal: 1 diagnosis
- Endocrine: 1 diagnosis
- Hematologic: 1 diagnosis
- Oncology: 1 diagnosis

**Total**: 20 unique clinical conditions consolidated from ~40 ICD codes

---

### 2. Removed Duplicate Scripts ✅

**Deleted files**:
- `scripts/get_top_diagnoses_simple.py` → Use `get_top_diagnoses_consolidated.py`
- `scripts/extract_dataset.py` → Use `extract_dataset_consolidated.py`
- `scripts/extract_top_diagnoses.py` → Redundant

**Single clear pathway now**:
1. Run `get_top_diagnoses_consolidated.py`
2. Run `extract_dataset_consolidated.py`
3. Continue with analysis scripts

No more confusion about which scripts to use!

---

### 3. Cleaned Up Prompts ✅

**Deleted files**:
- `prompts/task2_classification_prompt.txt`
- `prompts/task2_classification_prompt_CONSOLIDATED.txt`
- `prompts/task2_classification_prompt_FINAL.txt`

**Kept**:
- `prompts/task1_annotation_prompt.txt` - For Task 1
- `prompts/task2_classification_prompt_v2.txt` - For Task 2 (recommended)

Single recommended prompt for each task.

---

### 4. Consolidated Documentation ✅

**Deleted 10 redundant documentation files**:
- COMPLETE_SYSTEM_OVERVIEW.md
- DIAGNOSIS_CONSOLIDATION_GUIDE.md
- ICD_CODE_CONSISTENCY_REPORT.md
- ICD_CODE_FIXES_APPLIED.md
- SETUP_COMPLETE.md
- COST_ESTIMATION_GPT4o_MINI.md
- RAG_RESOURCES_COMPREHENSIVE.md
- CLINICAL_RESOURCES.md
- USAGE_GUIDE.md
- QUICKSTART.md

**Created 1 comprehensive guide**:
- **GUIDE.md** - Complete step-by-step instructions for entire workflow

**Updated README.md**:
- Concise overview
- Table of 20 diagnoses
- Workflow diagram
- Quick start instructions

---

## Project Structure (After Cleanup)

```
mimic-iv/
├── README.md                               # Concise overview (START HERE)
├── GUIDE.md                                # Complete step-by-step guide (DETAILED)
├── CLEANUP_SUMMARY.md                      # This file
├── diagnosis_mapping.py                    # 20 consolidated diagnoses
│
├── scripts/                                # Scripts in order
│   ├── get_top_diagnoses_consolidated.py  # Step 1
│   ├── extract_dataset_consolidated.py    # Step 2
│   ├── analyze_clinical_notes.py          # Step 3
│   ├── create_balanced_train_test.py      # Step 4
│   ├── eda_train_test_publication.py      # Step 5
│   └── evaluate_classification.py         # Step 8
│
├── prompts/                                # One prompt per task
│   ├── task1_annotation_prompt.txt
│   └── task2_classification_prompt_v2.txt
│
├── schemas/                                # Schemas for both tasks
│   ├── task1_annotation_schema.json
│   └── task2_classification_schema_v2.json
│
├── patterns/                               # Preprocessing patterns
├── functions/                              # Clinical calculations
└── extras/                                 # Clinical knowledge
```

---

## The 20 Consolidated Diagnoses

| # | Diagnosis | Category | ICD Codes |
|---|-----------|----------|-----------|
| 1 | Chest pain | Cardiovascular | 78650, 78659, R079 |
| 2 | Coronary atherosclerosis | Cardiovascular | 41401 |
| 3 | NSTEMI | Cardiovascular | I214 |
| 4 | Subendocardial infarction | Cardiovascular | 41071 |
| 5 | Atrial fibrillation | Cardiovascular | 42731 |
| 6 | Hypertensive heart disease with CKD | Cardiovascular | I130 |
| 7 | Sepsis | Infectious | 0389, 389, A419 |
| 8 | Pneumonia | Infectious | 486 |
| 9 | Urinary tract infection | Infectious | 5990 |
| 10 | Acute kidney injury | Renal | 5849, N179 |
| 11 | Depression | Psychiatric | 311, F329 |
| 12 | Alcohol use disorder | Psychiatric | 30500, F10129 |
| 13 | Chemotherapy encounter | Oncology | V5811, Z5111 |
| 14 | **Heart failure** | Cardiovascular | 42833, I5023, 42823, I5033, 4280, I509 |
| 15 | **Respiratory failure / COPD** | Respiratory | 51881, J9620, 4941, J449 |
| 16 | **Gastrointestinal bleeding** | Gastrointestinal | 5789, K922 |
| 17 | **Hypertension** | Cardiovascular | 4019, I10 |
| 18 | **Diabetes** | Endocrine | 25000, E119, E1165 |
| 19 | **Chronic kidney disease** | Renal | 5859, N189, N183 |
| 20 | **Anemia** | Hematologic | 2859, D649 |

**Bold** = newly added diagnoses

---

## Single Clear Workflow

```
Step 1: Extract Diagnoses
        python get_top_diagnoses_consolidated.py
              ↓
        Output: top_diagnoses_consolidated.csv (20 diagnoses)
              ↓
Step 2: Create Datasets
        python extract_dataset_consolidated.py
              ↓
        Output: annotation_dataset.csv + classification_dataset.csv
              ↓
Step 3: Analyze Text (Optional)
        python analyze_clinical_notes.py annotation_dataset.csv
              ↓
        Output: clinical_notes_analysis_*.csv/png
              ↓
Step 4: Train/Test Split (Optional)
        python create_balanced_train_test.py
              ↓
        Output: train_dataset.csv + test_dataset.csv
              ↓
Step 5: Generate EDA (Optional)
        python eda_train_test_publication.py
              ↓
        Output: eda_results/ (tables and figures)
              ↓
Step 6-7: Process with ClinOrchestra
              ↓
Step 8: Evaluate Results
        python evaluate_classification.py
              ↓
        Output: evaluation_results/ (metrics)
```

---

## What to Read

1. **Start here**: `README.md` - Quick overview and 5-minute setup
2. **Detailed guide**: `GUIDE.md` - Complete step-by-step instructions for entire workflow
3. **Diagnosis mapping**: `diagnosis_mapping.py` - See ICD code consolidation logic

---

## Key Improvements

✅ **20 consolidated diagnoses** (not 13)
✅ **No ICD-9/10 duplicates** for same conditions
✅ **Single clear pathway** (no duplicate scripts)
✅ **Clean documentation** (1 comprehensive guide vs 10+ files)
✅ **Ordered workflow** (steps numbered 1-8)
✅ **Ready to use** (no confusion about which files to use)

---

## Changes Committed

- **Commit**: `2f42679` - "MIMIC-IV: Major cleanup and expansion to 20 consolidated diagnoses"
- **Branch**: `claude/check-ui-011CV3DreSiCngqa9W5wtwNU`
- **Files changed**: 19 files (880 insertions, 6266 deletions)

---

## Next Steps

1. **Read GUIDE.md** for complete instructions
2. **Run Step 1**: Extract top 20 diagnoses from your MIMIC-IV data
3. **Run Step 2**: Create datasets
4. **Run Step 3**: Analyze text statistics (optional but recommended)
5. **Process with ClinOrchestra**

All scripts are ready to use with the 20 consolidated diagnoses!

---

## Questions?

- See **GUIDE.md** for detailed instructions
- See **README.md** for quick reference
- See **diagnosis_mapping.py** for ICD code mappings

---

**Status**: ✅ Complete - Ready for use
**Date**: 2025-11-12
**Version**: 1.0 (20 consolidated diagnoses)
