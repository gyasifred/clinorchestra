# MIMIC-IV Clinical Annotation Project - Complete Guide

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Step-by-Step Instructions](#step-by-step-instructions)
4. [Project Structure](#project-structure)
5. [Understanding the Outputs](#understanding-the-outputs)
6. [Troubleshooting](#troubleshooting)

---

## Overview

This project uses ClinOrchestra to create comprehensive clinical annotation datasets from MIMIC-IV for training medical AI systems.

### Two Main Tasks

**Task 1: Clinical Consultant AI Training (Comprehensive Annotation)**
- **Input**: Patient clinical record + Known primary diagnosis
- **Output**: Comprehensive extraction of all clinical evidence, contextualized findings, and expert clinical reasoning
- **Purpose**: Train LLMs to function as **CLINICAL CONSULTANTS** providing comprehensive clinical analysis

**Task 2: Multiclass Diagnosis Prediction (Classification)**
- **Input**: Patient clinical record only (diagnosis hidden)
- **Output**: Probability scores for ALL 20 possible diagnoses + detailed reasoning
- **Purpose**: Train and evaluate AI systems for **DIAGNOSTIC PREDICTION** using multiclass classification

### Dataset Scope
- **Focus**: Top 20 most common primary diagnoses in MIMIC-IV (consolidated from ICD-9 and ICD-10)
- **No ICD Duplicates**: Same clinical conditions with different ICD codes are treated as ONE diagnosis
- **Data Sources**: Discharge summaries, radiology reports, lab results, medications, vital signs

---

## Prerequisites

1. **MIMIC-IV Dataset**: You must have access to MIMIC-IV database (requires credentialing through PhysioNet)
2. **Python 3.8+** with pandas, numpy, matplotlib, seaborn
3. **ClinOrchestra** installed and configured

---

## Step-by-Step Instructions

Follow these steps in order to create your datasets and process them with ClinOrchestra.

### Step 1: Extract Top 20 Consolidated Diagnoses

**What this does**: Identifies the top 20 most common primary diagnoses in MIMIC-IV, consolidating ICD-9 and ICD-10 codes for the same clinical condition.

```bash
cd mimic-iv/scripts
python get_top_diagnoses_consolidated.py
```

**When prompted**, enter your MIMIC-IV path (example: `C:\Users\gyasi\Documents\mimic-iv-3.1`)

**Output Files**:
- `mimic-iv/top_diagnoses_consolidated.csv` - Summary of 20 consolidated diagnoses
- `mimic-iv/top_diagnoses_detailed_breakdown.csv` - Detailed breakdown showing which ICD codes map to each diagnosis

**Expected Output**:
```
CONSOLIDATED TOP DIAGNOSES (BY CLINICAL CONDITION)
================================================================================

1. Chest pain (CARDIOVASCULAR)
   Total Cases: 15,415
   ICD-9: 78650, 78659
   ICD-10: R079
   Code Breakdown:
     - 78650 (ICD-9): 7,297 cases - Chest pain, unspecified
     - 78659 (ICD-9): 5,212 cases - Other chest pain
     - R079 (ICD-10): 2,906 cases - Chest pain, unspecified

...and 19 more diagnoses
```

**Why this is important**: This consolidation ensures we don't have duplicates like "Sepsis (ICD-9)" and "Sepsis (ICD-10)" counting as separate diagnoses.

---

### Step 2: Create Annotation & Classification Datasets

**What this does**: Creates two CSV datasets from MIMIC-IV data - one for annotation training and one for classification training.

```bash
python extract_dataset_consolidated.py
```

**When prompted**:
- Enter MIMIC-IV path (same as Step 1)
- Enter sample size (optional - leave blank for full dataset, or enter 100 for testing)

**Output Files**:
- `mimic-iv/annotation_dataset.csv` - For Task 1 (Clinical Consultant Training)
- `mimic-iv/classification_dataset.csv` - For Task 2 (Diagnosis Prediction)

**Dataset Columns**:
| Column | Description |
|--------|-------------|
| `subject_id` | Unique patient identifier |
| `hadm_id` | Unique hospital admission identifier |
| `icd_code` | ICD code of primary diagnosis |
| `icd_version` | ICD version (9 or 10) |
| `primary_diagnosis_name` | Full name of diagnosis |
| `consolidated_diagnosis` | Consolidated diagnosis name (e.g., "Sepsis" for all sepsis ICD codes) |
| `clinical_text` | Combined discharge summary and radiology reports |
| `admission_type` | Type of admission (EMERGENCY, ELECTIVE, etc.) |
| `gender` | Patient gender |
| `anchor_age` | Patient age (de-identified anchor) |
| `race` | Patient race/ethnicity |
| `insurance` | Insurance type |
| `admittime` | Admission timestamp |
| `dischtime` | Discharge timestamp |
| `hospital_expire_flag` | Whether patient died in hospital (0/1) |

**Processing Time**: ~5-10 minutes for small sample (100 records), ~30-60 minutes for full dataset

---

### Step 3: Analyze Clinical Notes (Optional but Recommended)

**What this does**: Analyzes text length statistics across the 20 consolidated diagnoses to understand text complexity before LLM processing.

```bash
python analyze_clinical_notes.py annotation_dataset.csv
```

**Output Files**:
- `clinical_notes_by_diagnosis_top20.csv` - Per-diagnosis statistics
- `clinical_notes_analysis_summary.json` - Overall summary
- `clinical_notes_analysis_consolidated.png` - Visualizations
- `clinical_notes_top20_comparison.png` - Top 20 diagnosis comparison

**Example Output**:
```
Top 20 Diagnoses (Consolidated):
  1. Sepsis
     ICD Codes: 0389, A419
     Cases: 8,239
     Characters: avg=18,234, median=16,456, min=2,345, max=67,890
     Words: avg=2,945, median=2,678

  2. Chest pain
     ICD Codes: 78650, 78659, R079
     Cases: 15,415
     Characters: avg=14,567, median=13,234, min=1,890, max=52,345
     Words: avg=2,356, median=2,145

  ...and 18 more diagnoses
```

**Why this is important**:
- Helps you understand which diagnoses have complex vs simple documentation
- Identifies potential issues with very long or very short notes
- Informs token budget and processing time estimates

---

### Step 4: Create Balanced Train/Test Split (Optional)

**What this does**: Creates a balanced training and testing split stratified by diagnosis, demographics, and text complexity.

```bash
python create_balanced_train_test.py
```

**When prompted**:
- Path to dataset (e.g., `../annotation_dataset.csv`)
- Training set size (default: 4000)
- Test set size (default: 1000)

**Output Files**:
- `train_dataset_4000.csv`
- `test_dataset_1000.csv`
- `train_test_split_metadata.txt` - Split statistics

**Stratification Ensures**:
- Proportional diagnosis distribution in train and test
- Balanced gender and race distribution
- Similar text complexity distribution

---

### Step 5: Generate Exploratory Data Analysis (Optional)

**What this does**: Generates publication-ready statistics, tables, and visualizations for your dataset.

```bash
python eda_train_test_publication.py
```

**Output Files** (in `eda_results/` directory):
- `table1_baseline_characteristics.csv` (also .html, .tex)
- `table2_diagnosis_distribution.csv` (also .html, .tex)
- `figure1_diagnosis_distribution.png` (also .pdf)
- `figure2_demographics.png` (also .pdf)
- `figure3_text_complexity.png` (also .pdf)
- `methods_section.txt` - Ready-to-use methods text for manuscripts
- `supplementary_statistics.json` - Additional statistics

**Publication-Ready Tables**:
```
Table 1: Baseline Characteristics
==================================
Characteristic         | Train (n=4000) | Test (n=1000) | p-value
--------------------|----------------|---------------|--------
Age (years), mean±SD   | 65.3±18.2     | 64.9±17.8    | 0.542
Gender (%)
  Male                | 2,280 (57.0%)  | 565 (56.5%)  | 0.723
  Female              | 1,720 (43.0%)  | 435 (43.5%)  |
...
```

---

### Step 6: Configure ClinOrchestra for Task 1 (Annotation)

**What this does**: Set up ClinOrchestra to perform comprehensive clinical annotation.

1. **Open ClinOrchestra UI**

2. **Prompt Tab**:
   - Copy contents from `mimic-iv/prompts/task1_annotation_prompt.txt`
   - Paste into prompt field
   - This prompt instructs the AI to extract comprehensive clinical evidence

3. **Schema Tab**:
   - Upload `mimic-iv/schemas/task1_annotation_schema.json`
   - This defines the structured output format with 10 evidence categories

4. **Data Tab**:
   - Upload `mimic-iv/annotation_dataset.csv`
   - Verify column mappings (should auto-detect)

5. **RAG Tab** (Optional but Recommended):
   - Upload clinical practice guidelines PDFs (if collected)
   - This significantly improves annotation quality

6. **Patterns Tab** (Optional):
   - Upload pattern JSONs from `mimic-iv/patterns/`
   - Standardizes vital signs and lab values

7. **Functions Tab** (Optional):
   - Upload function JSONs from `mimic-iv/functions/`
   - Enables clinical calculations (SOFA, CURB-65, etc.)

8. **Processing Tab**:
   - Batch size: 10-50 records recommended
   - Configure API settings and retry logic

9. **Start Processing**

**Expected Processing Time**:
- 100 records: ~2-4 hours
- 1,000 records: ~20-40 hours
- 10,000 records: ~8-16 days

**Cost Estimates** (using GPT-4):
- ~$0.50-1.00 per patient record

---

### Step 7: Configure ClinOrchestra for Task 2 (Classification)

**What this does**: Set up ClinOrchestra to perform multiclass diagnosis prediction.

1. **Prompt Tab**:
   - Copy contents from `mimic-iv/prompts/task2_classification_prompt_v2.txt`
   - This prompt enforces probability assignments for all 20 diagnoses

2. **Schema Tab**:
   - Upload `mimic-iv/schemas/task2_classification_schema_v2.json`
   - Ensures multiclass output with probabilities summing to 1.0

3. **Data Tab**:
   - Upload `mimic-iv/classification_dataset.csv`

4. **Processing Tab**:
   - Configure as needed

5. **Start Processing**

**Expected Processing Time**:
- 100 records: ~3-5 hours
- 1,000 records: ~30-50 hours

**Cost Estimates**:
- ~$0.70-1.50 per patient record

---

### Step 8: Evaluate Classification Results

**What this does**: Evaluates the multiclass diagnosis predictions from Task 2.

```bash
python evaluate_classification.py
```

**When prompted**:
- Path to predictions JSON file (ClinOrchestra output)
- Path to ground truth CSV (`classification_dataset.csv`)
- Path to diagnosis mapping (`top_diagnoses_consolidated.csv`)

**Metrics Computed**:

**Accuracy Metrics**:
- Top-1 Accuracy: Correct diagnosis has highest probability
- Top-3 Accuracy: Correct diagnosis in top 3
- Top-5 Accuracy: Correct diagnosis in top 5

**Calibration Metrics**:
- Cross-Entropy Loss: Measures probability calibration (lower is better)
- Brier Score: Mean squared error of predictions (0-2, lower is better)

**Per-Class Metrics**:
- Precision: Of predictions for diagnosis X, how many were correct?
- Recall: Of all actual diagnosis X cases, how many were predicted?
- F1-Score: Harmonic mean of precision and recall

**Output Files**:
- `evaluation_results/per_class_metrics.csv`
- `evaluation_results/confusion_matrix.png`
- `evaluation_results/evaluation_summary.json`

**Interpreting Good Performance**:
- Top-1 Accuracy > 70%
- Top-5 Accuracy > 90%
- Cross-Entropy < 0.5
- Brier Score < 0.3

---

## Project Structure

```
mimic-iv/
├── README.md                                    # Overview and reference
├── GUIDE.md                                     # This file - complete step-by-step guide
├── diagnosis_mapping.py                         # Consolidation mapping (20 diagnoses)
│
├── scripts/                                     # All Python scripts
│   ├── get_top_diagnoses_consolidated.py       # Step 1: Extract top 20 diagnoses
│   ├── extract_dataset_consolidated.py         # Step 2: Create datasets
│   ├── analyze_clinical_notes.py               # Step 3: Analyze text statistics
│   ├── create_balanced_train_test.py           # Step 4: Create train/test split
│   ├── eda_train_test_publication.py           # Step 5: Generate EDA
│   ├── evaluate_classification.py              # Step 8: Evaluate classification
│   ├── gather_clinical_guidelines.py           # Helper: Generate guideline collection guide
│   ├── generate_classification_prompt.py       # Helper: Generate prompt with diagnosis list
│   ├── README_EDA.md                            # EDA documentation
│   └── README_train_test_split.md               # Train/test split documentation
│
├── prompts/                                     # Ready-to-use prompts
│   ├── task1_annotation_prompt.txt             # Clinical Consultant training prompt
│   └── task2_classification_prompt_v2.txt       # Multiclass classification prompt
│
├── schemas/                                     # JSON schemas for structured output
│   ├── task1_annotation_schema.json            # Comprehensive annotation schema
│   └── task2_classification_schema_v2.json     # Multiclass prediction schema
│
├── patterns/                                    # Regex patterns for preprocessing
│   ├── vital_signs_*.json                      # Vital signs standardization
│   └── lab_*.json                               # Lab values standardization
│
├── functions/                                   # Clinical calculation functions
│   ├── calculate_sofa_score.json               # SOFA score (sepsis severity)
│   ├── calculate_curb65.json                   # CURB-65 (pneumonia severity)
│   ├── calculate_map.json                      # Mean arterial pressure
│   └── calculate_creatinine_clearance.json     # Renal function
│
├── extras/                                      # Clinical knowledge for prompts
│   ├── sepsis_diagnostic_criteria.json         # Sepsis-3 criteria
│   ├── heart_failure_classification.json       # Heart failure types
│   ├── aki_staging.json                        # KDIGO AKI staging
│   └── clinical_annotation_approach.json       # Systematic methodology
│
└── [Generated Files]
    ├── top_diagnoses_consolidated.csv          # From Step 1
    ├── top_diagnoses_detailed_breakdown.csv    # From Step 1
    ├── annotation_dataset.csv                  # From Step 2
    ├── classification_dataset.csv              # From Step 2
    ├── clinical_notes_*.csv/png/json           # From Step 3
    ├── train_dataset_4000.csv                  # From Step 4
    ├── test_dataset_1000.csv                   # From Step 4
    └── eda_results/                            # From Step 5
```

---

## Understanding the Outputs

### Top 20 Consolidated Diagnoses

The 20 diagnoses cover major categories:

**Cardiovascular (8 diagnoses)**:
1. Chest pain
2. Coronary atherosclerosis
3. NSTEMI
4. Subendocardial infarction
5. Atrial fibrillation
6. Hypertensive heart disease with CKD
7. Heart failure
8. Hypertension

**Infectious (3 diagnoses)**:
9. Sepsis
10. Pneumonia
11. Urinary tract infection

**Renal (2 diagnoses)**:
12. Acute kidney injury
13. Chronic kidney disease

**Psychiatric (2 diagnoses)**:
14. Depression
15. Alcohol use disorder

**Oncology (1 diagnosis)**:
16. Chemotherapy encounter

**Respiratory (1 diagnosis)**:
17. Respiratory failure / COPD

**Gastrointestinal (1 diagnosis)**:
18. Gastrointestinal bleeding

**Endocrine (1 diagnosis)**:
19. Diabetes

**Hematologic (1 diagnosis)**:
20. Anemia

### Task 1 Output (Annotation)

For each patient, you get a comprehensive JSON with:
- **Evidence summary**: Overall quality and key findings
- **Symptoms and presentation**: All symptoms with onset, severity, progression (with direct quotes)
- **Physical examination**: Vital signs and exam findings
- **Laboratory results**: All abnormal lab values
- **Imaging and diagnostics**: Radiology findings
- **Medications and treatments**: Medications with responses
- **Medical history**: Past medical, surgical, family history
- **Risk factors**: Lifestyle, environmental, demographic
- **Clinical reasoning**: Diagnostic criteria, differential diagnoses
- **Temporal timeline**: Disease progression over time
- **Severity assessment**: Complications, staging, functional impact

**Quality Expectations**:
- 200-500 data points extracted per patient
- All evidence cited with direct quotes
- Each evidence item rated for strength (DEFINITIVE, STRONG, MODERATE, WEAK, CONTEXTUAL)

### Task 2 Output (Classification)

For each patient, you get:
- **Clinical data extraction**: Systematically extracted findings
- **Clinical pattern analysis**: Pattern recognition, organ systems involved
- **Multiclass prediction**: Probability scores for ALL 20 diagnoses (sum = 1.0)
  - Each diagnosis: probability, supporting evidence, contradicting evidence, reasoning
- **Top diagnosis**: Most likely diagnosis with detailed reasoning
- **Top-5 differential**: Top 5 ranked diagnoses
- **Clinical reasoning**: Step-by-step diagnostic thinking

**Example Output**:
```json
{
  "multiclass_prediction": {
    "Sepsis": {
      "probability": 0.75,
      "supporting_evidence": ["fever 39.2°C", "elevated lactate 4.5", "hypotension 85/50"],
      "contradicting_evidence": [],
      "reasoning": "Strong evidence for sepsis with SIRS criteria met..."
    },
    "Pneumonia": {
      "probability": 0.12,
      "supporting_evidence": ["infiltrate on CXR"],
      "contradicting_evidence": ["blood cultures positive for GNR"],
      "reasoning": "Pneumonia possible but sepsis more likely given systemic findings..."
    },
    ...all 20 diagnoses with probabilities summing to 1.0
  },
  "top_diagnosis": {
    "diagnosis": "Sepsis",
    "confidence": 0.75,
    "reasoning": "Patient meets Sepsis-3 criteria with SOFA score 8..."
  }
}
```

---

## Troubleshooting

### Common Issues

**Issue**: Script can't find MIMIC-IV files
- **Solution**: Check path has `hosp/`, `icu/`, `note/` folders
- **Example**: `C:\Users\gyasi\Documents\mimic-iv-3.1`

**Issue**: Out of memory when loading large files
- **Solution**: Scripts automatically sample large files (labevents, chartevents)
- Use sample size option for testing

**Issue**: "Missing diagnosis codes" warning
- **Solution**: Some diagnoses may not be present in your MIMIC-IV version
- Check `top_diagnoses_consolidated.csv` to see actual counts

**Issue**: Prompt too long for model
- **Solution**: Some discharge notes are very long
- Truncate notes or use model with larger context window (e.g., GPT-4 Turbo)

**Issue**: JSON schema validation errors in ClinOrchestra
- **Solution**: Ensure schema file is uploaded correctly
- Check that schema matches prompt structure

**Issue**: EDA script errors on missing data
- **Solution**: Ensure train/test CSV files exist and have required columns
- Re-run `create_balanced_train_test.py` if needed

### Performance Tips

1. **Start Small**: Test with 10-100 records before processing thousands
2. **Review Quality**: Manually review first outputs before scaling
3. **Use RAG**: Clinical guidelines significantly improve quality
4. **Batch Processing**: Process in batches of 50-100 for easier monitoring
5. **Save Progress**: ClinOrchestra saves progress automatically
6. **Monitor Costs**: Track API usage, especially for large datasets
7. **Optimize Prompts**: Iterate on prompts based on initial results

---

## Data Privacy & Usage Notes

### Data Privacy
- MIMIC-IV data is de-identified but still protected
- Follow all PhysioNet data use agreements
- Do not attempt to re-identify patients
- Store securely, do not share publicly

### Clinical Validity
- Annotations are based on documented evidence only
- Do not make clinical decisions based on this data
- For research and educational purposes only
- Not for actual patient care

### Limitations
- Depends on documentation quality in source records
- ICD codes may not perfectly reflect clinical complexity
- Some diagnoses may be under-documented
- De-identification may remove some clinical context

---

## Citation

If you use this dataset or methodology, please cite:
- **MIMIC-IV Database**: Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). MIMIC-IV (version 2.2). PhysioNet. https://doi.org/10.13026/6mm1-ek67
- **ClinOrchestra**: [Your citation]
- Your own research paper describing the dataset creation

---

## Support

For questions about:
- **MIMIC-IV data**: Contact PhysioNet support
- **ClinOrchestra**: See main project README
- **This workflow**: Open an issue in the repository

---

**Last Updated**: 2025-11-12

**Status**: Ready for use with 20 consolidated diagnoses and streamlined workflow

**Next Steps**: Start with Step 1 above! 🚀
