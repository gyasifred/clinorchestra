# MIMIC-IV Example: Clinical Annotation & Classification

**⚠️ THIS IS AN EXAMPLE USE CASE - ClinOrchestra is a Universal Platform**

This directory demonstrates ClinOrchestra's capabilities through a comprehensive MIMIC-IV clinical annotation project. **ClinOrchestra is NOT limited to MIMIC-IV or diagnosis tasks** - it adapts to ANY clinical extraction task through customizable prompts and schemas.

---

## What This Example Demonstrates

✅ **Multi-column prompt variables**: Passing patient demographics, diagnosis info as template variables
✅ **Complex structured extraction**: 200-500 data points per patient
✅ **Multiclass probabilistic classification**: 20-way diagnosis prediction with calibrated probabilities
✅ **RAG integration**: Retrieving clinical guidelines during extraction
✅ **Custom functions**: Medical calculations (HEART score, NIHSS, KDIGO staging)
✅ **Large-scale processing**: 113,000+ clinical cases

**But you can use ClinOrchestra for ANY clinical task** - just define your own prompts and schemas.

---

## Quick Start

**For detailed step-by-step instructions, see [GUIDE.md](GUIDE.md)**

### Five-Minute Setup

1. **Extract top 20 diagnoses** (consolidated, no ICD duplicates):
   ```bash
   cd mimic-iv/scripts
   python get_top_diagnoses_consolidated.py /path/to/mimic-iv-3.1
   ```

2. **Create datasets**:
   ```bash
   python extract_dataset_consolidated.py
   ```

3. **Analyze text statistics** (optional):
   ```bash
   python analyze_clinical_notes.py ../annotation_dataset.csv
   ```

4. **Use with ClinOrchestra**:
   - Upload prompts from `prompts/`
   - Upload schemas from `schemas/`
   - Upload datasets and process

---

## Two Example Tasks

### Task 1: Clinical Consultant AI Training
**Goal**: Extract comprehensive clinical evidence supporting a diagnosis
**Use Case**: Train LLMs for clinical reasoning and evidence-based assessment
**Output**: 200-500 structured data points per patient

### Task 2: Multiclass Diagnosis Prediction
**Goal**: Predict probability distribution across all 20 diagnoses
**Use Case**: Evaluate diagnostic accuracy with calibrated probabilities
**Output**: Probability scores for each diagnosis (sum = 1.0)

---

## Dataset Scope

- **20 consolidated diagnoses** (no ICD-9/ICD-10 duplicates)
- **Categories**: Cardiovascular (9), Infectious (3), Renal (1), Psychiatric (3), Gastrointestinal (1), Neurological (1), Respiratory (1), Oncology (1)
- **Total cases**: 113,421 patients
- **Data sources**: Discharge summaries, radiology reports, labs, medications, vitals

---

## Project Structure

```
mimic-iv/
├── GUIDE.md                             # Complete step-by-step guide (START HERE)
├── README.md                            # This file (emphasizing universal platform)
├── diagnosis_mapping.py                 # 20 consolidated diagnoses mapping
├── diagnosis_mapping.yaml               # Label context import file
├── diagnosis_mapping.json               # Label context import file (JSON format)
│
├── scripts/                             # All scripts in order
│   ├── get_top_diagnoses_consolidated.py   # Step 1: Extract diagnoses
│   ├── extract_dataset_consolidated.py     # Step 2: Create datasets
│   ├── analyze_clinical_notes.py           # Step 3: Analyze text
│   ├── create_balanced_train_test.py       # Step 4: Train/test split
│   ├── eda_train_test_publication.py       # Step 5: Generate EDA
│   └── evaluate_classification.py          # Step 8: Evaluate results
│
├── prompts/                             # Example prompts
│   ├── task1_annotation_prompt.txt
│   └── task2_classification_prompt_v2.txt
│
├── schemas/                             # Example JSON schemas
│   ├── task1_annotation_schema.json
│   └── task2_classification_schema_v2.json
│
├── patterns/                            # Regex preprocessing (optional)
├── functions/                           # Clinical calculations (optional)
└── extras/                              # Clinical knowledge (optional)
```

---

## The 20 Consolidated Diagnoses (Based on MIMIC-IV Top 70)

| # | Diagnosis | Category | Cases | ICD Codes |
|---|-----------|----------|-------|-----------|
| 1 | Chest pain | Cardiovascular | 17,535 | 78650, 78659, R079, R0789 |
| 2 | Coronary atherosclerosis | Cardiovascular | 8,903 | 41401, I25110, I2510 |
| 3 | Myocardial infarction | Cardiovascular | 6,138 | I214, 41071 |
| 4 | Heart failure | Cardiovascular | 6,648 | 42833, I110, 42823 |
| 5 | Atrial fibrillation | Cardiovascular | 4,145 | 42731, I480 |
| 6 | Hypertensive heart disease with CKD | Cardiovascular | 3,313 | I130 |
| 7 | Sepsis | Infectious | 8,239 | A419, 0389, 389 |
| 8 | Pneumonia | Infectious | 5,843 | 486, J189 |
| 9 | Urinary tract infection | Infectious | 5,138 | 5990, N390 |
| 10 | Acute kidney injury | Renal | 6,185 | 5849, N179 |
| 11 | Depression | Psychiatric | 7,202 | F329, 311 |
| 12 | Alcohol use disorder | Psychiatric | 8,038 | 30500, F10129 |
| 13 | Chemotherapy encounter | Oncology | 6,530 | Z5111, V5811 |
| 14 | Syncope | Cardiovascular | 4,327 | 7802, R55 |
| 15 | Aortic valve disorders | Cardiovascular | 2,973 | 4241, I350 |
| 16 | Acute pancreatitis | Gastrointestinal | 2,620 | 5770 |
| 17 | Psychosis | Psychiatric | 2,450 | 2989, F29 |
| 18 | Stroke | Neurological | 2,437 | 43491, 43411 |
| 19 | COPD | Respiratory | 2,403 | 49121, J441 |
| 20 | Pulmonary embolism | Cardiovascular | 2,354 | 41519, I2699 |

**Total**: 113,421 cases consolidated from 43 ICD codes

---

## Workflow Overview

```
Step 1: Extract Diagnoses → top_diagnoses_consolidated.csv (20 diagnoses)
              ↓
Step 2: Create Datasets → annotation_dataset.csv + classification_dataset.csv
              ↓
Step 3: Analyze Text → clinical_notes_analysis_*.csv/png (optional)
              ↓
Step 4: Train/Test Split → train_dataset.csv + test_dataset.csv (optional)
              ↓
Step 5: Generate EDA → eda_results/ with tables and figures (optional)
              ↓
Step 6-7: Process with ClinOrchestra
              ↓
Step 8: Evaluate Results → evaluation_results/ with metrics
```

---

## Prerequisites

1. **MIMIC-IV Dataset**: PhysioNet credentialed access required
2. **Python 3.8+**: With pandas, numpy, matplotlib, seaborn
3. **ClinOrchestra v1.0.0**: Installed and configured

---

## Documentation

- **[GUIDE.md](GUIDE.md)** - Complete step-by-step guide (START HERE)
- **[diagnosis_mapping.py](diagnosis_mapping.py)** - ICD code consolidation logic
- **[diagnosis_mapping.yaml](diagnosis_mapping.yaml)** - Label context for ClinOrchestra import
- **[scripts/README_EDA.md](scripts/README_EDA.md)** - EDA documentation
- **[scripts/README_train_test_split.md](scripts/README_train_test_split.md)** - Train/test split docs

---

## Expected Outputs

### Task 1: Clinical Consultant Annotation

Comprehensive JSON for each patient with:
- Evidence summary (overall quality, key findings)
- Symptoms and presentation (with quotes from text)
- Physical examination (vital signs, exam findings)
- Laboratory results (all abnormal values)
- Imaging and diagnostics (radiology findings)
- Medications and treatments (with responses)
- Medical history (past medical, surgical, family)
- Risk factors (lifestyle, environmental, demographic)
- Clinical reasoning (diagnostic criteria, differentials)
- Temporal timeline (disease progression)
- Severity assessment (complications, staging)

**Quality**: 200-500 structured data points per patient

### Task 2: Multiclass Classification

For each patient:
- **Multiclass prediction**: Probability for ALL 20 diagnoses (sum = 1.0)
- **Top diagnosis**: Most likely diagnosis with reasoning
- **Top-5 differential**: Top 5 ranked diagnoses
- **Clinical reasoning**: Step-by-step diagnostic thinking

**Example**:
```json
{
  "multiclass_prediction": {
    "Sepsis": {"probability": 0.75, "supporting_evidence": [...], ...},
    "Pneumonia": {"probability": 0.12, ...},
    ...all 20 diagnoses
  },
  "top_diagnosis": {"diagnosis": "Sepsis", "confidence": 0.75, ...}
}
```

---

## Performance Estimates

| Dataset Size | Processing Time (Task 1) | Cost (GPT-4) |
|--------------|-------------------------|--------------|
| 100 records  | 2-4 hours               | $50-100      |
| 1,000 records| 20-40 hours             | $500-1,000   |
| 10,000 records| 8-16 days              | $5,000-10,000|

---

## Evaluation Metrics (Task 2)

- **Top-1 Accuracy**: Correct diagnosis has highest probability
- **Top-3 Accuracy**: Correct diagnosis in top 3
- **Top-5 Accuracy**: Correct diagnosis in top 5
- **Cross-Entropy Loss**: Probability calibration quality
- **Brier Score**: Prediction accuracy (0-2, lower is better)
- **Per-Class Precision/Recall/F1**: Performance per diagnosis

---

## Key Benefits of This Example

1. **No ICD Duplicates**: Same condition with different codes = ONE diagnosis
2. **Single Clear Pathway**: No confusion about which scripts to run
3. **Consolidated from Day 1**: Mapping built into all scripts
4. **Publication-Ready**: EDA generates tables and figures for papers
5. **Comprehensive Documentation**: Everything explained in [GUIDE.md](GUIDE.md)

---

## Adapting to Your Own Task

**This is just an example!** To use ClinOrchestra for YOUR task:

1. **Define your prompt**: Describe what you want to extract
2. **Create your schema**: Define the JSON structure you want
3. **Prepare your data**: CSV with clinical text column
4. **Optional - Add tools**:
   - Functions for your domain calculations
   - Patterns for your text normalization
   - Extras for your domain knowledge
   - RAG documents for your guidelines
5. **Run ClinOrchestra**: Same interface, different task!

**See the main [ClinOrchestra README](../README.md) for universal platform documentation.**

---

## Data Privacy

- MIMIC-IV data is de-identified but still protected
- Follow all PhysioNet data use agreements
- Do not attempt to re-identify patients
- Store securely, do not share publicly
- For research and educational purposes only
- Not for actual patient care

---

## Citation

If you use this example methodology, please cite:

**MIMIC-IV Database**:
```
Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023).
MIMIC-IV (version 2.2). PhysioNet. https://doi.org/10.13026/6mm1-ek67
```

**ClinOrchestra v1.0.0**: [Citation for the universal platform]

---

## Support

- **MIMIC-IV data questions**: PhysioNet support
- **ClinOrchestra platform questions**: See main [README](../README.md)
- **This example workflow**: See [GUIDE.md](GUIDE.md) or open an issue

---

**Version**: 1.0.0 (Example Use Case)
**Platform**: ClinOrchestra Universal Clinical Data Extraction System
**Last Updated**: 2025-11-13
**Status**: Production-ready example demonstrating platform capabilities

**⚠️ Remember**: This is ONE example. ClinOrchestra adapts to ANY clinical task!
