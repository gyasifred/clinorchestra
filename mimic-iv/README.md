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
- **Categories**: Cardiovascular (7), Infectious (7), Renal (1), Psychiatric (2), Gastrointestinal (2), Neurological (1), Oncology (1)
- **Total cases**: 110,596 patients
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

## The 20 Consolidated Diagnoses (Based on Actual MIMIC-IV Data)

| # | Diagnosis | Category | Cases | ICD Codes Consolidated |
|---|-----------|----------|-------|----------------------|
| 1 | Chest pain | Cardiovascular | 17,535 | 78650, 78659, R079, R0789 |
| 2 | Coronary atherosclerosis | Cardiovascular | 8,903 | 41401, I25110, I2510 |
| 3 | Myocardial infarction | Cardiovascular | 6,138 | I214, 41071 |
| 4 | Heart failure | Cardiovascular | 6,648 | 42833, I110, 42823 |
| 5 | Atrial fibrillation | Cardiovascular | 3,090 | 42731 |
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
| 17 | Postoperative infection | Infectious | 2,198 | 99859 |
| 18 | Cellulitis | Infectious | 2,152 | 6826 |
| 19 | COVID-19 | Infectious | 1,937 | U071 |
| 20 | Altered mental status | Neurological | 1,587 | 78097 |

**Total**: 20 unique clinical conditions consolidated from 32 ICD codes
**Total Cases**: 110,596

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
