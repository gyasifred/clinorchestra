# MIMIC-IV Clinical Annotation & Classification Project

This project uses ClinOrchestra to create comprehensive clinical annotation datasets from MIMIC-IV for training medical AI systems.

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

## Project Overview

### Two Main Tasks

**Task 1: Clinical Consultant AI Training**
- Extract comprehensive clinical evidence supporting a diagnosis
- Train LLMs for clinical reasoning and evidence-based assessment
- Output: 200-500 structured data points per patient

**Task 2: Multiclass Diagnosis Prediction**
- Predict probability distribution across all 20 diagnoses
- Evaluate diagnostic accuracy with calibrated probabilities
- Output: Probability scores for each diagnosis (sum = 1.0)

### Dataset Scope

- **20 consolidated diagnoses** (no ICD-9/ICD-10 duplicates)
- **Categories**: Cardiovascular (8), Infectious (3), Renal (2), Psychiatric (2), Respiratory (1), GI (1), Endocrine (1), Hematologic (1), Oncology (1)
- **Data sources**: Discharge summaries, radiology reports, labs, medications, vitals

### Key Features

✅ **No Duplicates**: ICD-9 and ICD-10 codes for same condition consolidated
✅ **One Clear Pathway**: Single workflow from extraction to train/test split
✅ **Publication-Ready**: EDA generates tables and figures for manuscripts
✅ **Comprehensive Documentation**: See [GUIDE.md](GUIDE.md) for everything

## Project Structure

```
mimic-iv/
├── GUIDE.md                             # Complete step-by-step guide (START HERE)
├── README.md                            # This file
├── diagnosis_mapping.py                 # 20 consolidated diagnoses mapping
│
├── scripts/                             # All scripts in order
│   ├── get_top_diagnoses_consolidated.py   # Step 1: Extract diagnoses
│   ├── extract_dataset_consolidated.py     # Step 2: Create datasets
│   ├── analyze_clinical_notes.py           # Step 3: Analyze text
│   ├── create_balanced_train_test.py       # Step 4: Train/test split
│   ├── eda_train_test_publication.py       # Step 5: Generate EDA
│   └── evaluate_classification.py          # Step 8: Evaluate results
│
├── prompts/                             # Ready-to-use prompts
│   ├── task1_annotation_prompt.txt
│   └── task2_classification_prompt_v2.txt
│
├── schemas/                             # JSON schemas
│   ├── task1_annotation_schema.json
│   └── task2_classification_schema_v2.json
│
├── patterns/                            # Regex preprocessing
├── functions/                           # Clinical calculations
└── extras/                              # Clinical knowledge
```

## The 20 Consolidated Diagnoses

| # | Diagnosis | Category | ICD Codes Consolidated |
|---|-----------|----------|----------------------|
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
| 14 | Heart failure | Cardiovascular | 42833, I5023, 42823, I5033, 4280, I509 |
| 15 | Respiratory failure / COPD | Respiratory | 51881, J9620, 4941, J449 |
| 16 | Gastrointestinal bleeding | Gastrointestinal | 5789, K922 |
| 17 | Hypertension | Cardiovascular | 4019, I10 |
| 18 | Diabetes | Endocrine | 25000, E119, E1165 |
| 19 | Chronic kidney disease | Renal | 5859, N189, N183 |
| 20 | Anemia | Hematologic | 2859, D649 |

**Total**: 20 unique clinical conditions consolidated from ~40 ICD codes

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

## Prerequisites

1. **MIMIC-IV Dataset**: PhysioNet credentialed access required
2. **Python 3.8+**: With pandas, numpy, matplotlib, seaborn
3. **ClinOrchestra**: Installed and configured

## Documentation

- **[GUIDE.md](GUIDE.md)** - Complete step-by-step guide (START HERE)
- **[diagnosis_mapping.py](diagnosis_mapping.py)** - ICD code consolidation logic
- **[scripts/README_EDA.md](scripts/README_EDA.md)** - EDA documentation
- **[scripts/README_train_test_split.md](scripts/README_train_test_split.md)** - Train/test split docs

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

## Performance Estimates

| Dataset Size | Processing Time (Task 1) | Cost (GPT-4) |
|--------------|-------------------------|--------------|
| 100 records  | 2-4 hours               | $50-100      |
| 1,000 records| 20-40 hours             | $500-1,000   |
| 10,000 records| 8-16 days              | $5,000-10,000|

## Evaluation Metrics (Task 2)

- **Top-1 Accuracy**: Correct diagnosis has highest probability
- **Top-3 Accuracy**: Correct diagnosis in top 3
- **Top-5 Accuracy**: Correct diagnosis in top 5
- **Cross-Entropy Loss**: Probability calibration quality
- **Brier Score**: Prediction accuracy (0-2, lower is better)
- **Per-Class Precision/Recall/F1**: Performance per diagnosis

**Good Performance**:
- Top-1 Accuracy > 70%
- Top-5 Accuracy > 90%
- Cross-Entropy < 0.5
- Brier Score < 0.3

## Key Benefits

1. **No ICD Duplicates**: Same condition with different codes = ONE diagnosis
2. **Single Clear Pathway**: No confusion about which scripts to run
3. **Consolidated from Day 1**: Mapping built into all scripts
4. **Publication-Ready**: EDA generates tables and figures for papers
5. **Comprehensive Documentation**: Everything explained in [GUIDE.md](GUIDE.md)

## Usage Tips

1. **Start Small**: Test with 100 records before processing thousands
2. **Review Quality**: Manually check first outputs
3. **Use RAG**: Upload clinical guidelines to significantly improve quality
4. **Monitor Costs**: Track API usage for large datasets
5. **Iterate Prompts**: Refine based on initial results

## Data Privacy

- MIMIC-IV data is de-identified but still protected
- Follow all PhysioNet data use agreements
- Do not attempt to re-identify patients
- Store securely, do not share publicly
- For research and educational purposes only
- Not for actual patient care

## Citation

If you use this dataset or methodology, please cite:

**MIMIC-IV Database**:
```
Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023).
MIMIC-IV (version 2.2). PhysioNet. https://doi.org/10.13026/6mm1-ek67
```

**ClinOrchestra**: [Your citation]

**Your research paper**: Describing the dataset creation methodology

## Support

- **MIMIC-IV data questions**: PhysioNet support
- **ClinOrchestra questions**: Main project README
- **This workflow questions**: Open an issue in the repository

---

**Last Updated**: 2025-11-12

**Status**: Production-ready with 20 consolidated diagnoses

**Next Steps**: See [GUIDE.md](GUIDE.md) for complete instructions
