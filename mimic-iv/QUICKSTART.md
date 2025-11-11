# MIMIC-IV ClinOrchestra Project - Quick Start Guide

## Overview

This project creates two types of datasets from MIMIC-IV for training medical AI:
1. **Annotation Dataset**: Extract clinical evidence supporting a diagnosis
2. **Classification Dataset**: Predict diagnosis from clinical evidence

## Prerequisites

✓ MIMIC-IV database access (from PhysioNet)
✓ Python 3.8+
✓ ClinOrchestra installed

## Step-by-Step Instructions

### Step 1: Extract Top 20 Diagnoses (5 minutes)

```bash
cd mimic-iv/scripts
python extract_top_diagnoses.py
```

When prompted, enter your MIMIC-IV path (e.g., `C:\Users\gyasi\Documents\mimic-iv-3.1`)

**Output**: `top_20_primary_diagnoses.csv` with the most common diagnoses

### Step 2: Create Datasets (15-30 minutes)

```bash
python extract_dataset.py
```

Enter:
- MIMIC-IV path
- (Optional) Sample size for testing (e.g., 100)

**Outputs**:
- `annotation_dataset.csv` - For Task 1
- `classification_dataset.csv` - For Task 2

### Step 3: Generate Classification Prompt (1 minute)

```bash
python generate_classification_prompt.py
```

This creates a prompt with the top 20 diagnoses embedded.

### Step 4: Set Up ClinOrchestra

#### For Task 1: Annotation

1. Open ClinOrchestra UI
2. **Prompt Tab**:
   - Copy all contents from `prompts/task1_annotation_prompt.txt`
   - Paste into prompt field
3. **Data Tab**:
   - Upload `annotation_dataset.csv`
   - Map columns to placeholders (should auto-detect)
4. **Processing Tab**:
   - Upload `schemas/task1_annotation_schema.json` as JSON schema
   - Configure batch settings (recommend: 10-50 records per batch)
5. **RAG Tab** (optional but recommended):
   - Upload clinical guidelines PDFs (see Step 5)
6. Click **Start Processing**

#### For Task 2: Classification

1. **Prompt Tab**:
   - Copy from `prompts/task2_classification_prompt_generated.txt`
2. **Data Tab**:
   - Upload `classification_dataset.csv`
3. **Processing Tab**:
   - Upload `schemas/task2_classification_schema.json`
4. Click **Start Processing**

### Step 5: Gather Clinical Guidelines (Optional, 1-2 hours)

```bash
python gather_clinical_guidelines.py
```

This generates:
- PubMed search URLs for each diagnosis
- Recommended guideline sources
- Directory structure for organizing PDFs

Manually download guidelines and save to the suggested directories.

Upload PDFs to ClinOrchestra RAG system for enhanced annotations.

## File Structure Created

```
mimic-iv/
├── scripts/               # Python extraction scripts
├── prompts/              # Ready-to-use prompts
├── schemas/              # JSON schemas for output
├── content_mappings/     # Diagnosis-specific mappings
├── patterns/             # Regex patterns for extraction
├── functions/            # Clinical calculation functions
└── guidelines/           # Clinical practice guidelines
```

## Expected Outputs

### Task 1 (Annotation)
For each patient, you get a comprehensive JSON with:
- Evidence summary (overall strength, key findings)
- Symptoms and presentation (with quotes from text)
- Physical exam findings (vital signs, exam)
- Laboratory results (all abnormal values)
- Imaging findings
- Medications and treatments
- Clinical reasoning (differential diagnosis considered)
- Temporal timeline
- Severity assessment

### Task 2 (Classification)
For each patient, you get:
- Extracted clinical data (organized systematically)
- Clinical pattern analysis
- Differential diagnosis (ranked list of 3-10 possibilities)
- Primary diagnosis prediction (with confidence score)
- Alternative diagnoses (with reasoning why less likely)
- Detailed clinical reasoning process

## Processing Time Estimates

| Dataset Size | Processing Time (Task 1) | Processing Time (Task 2) |
|--------------|-------------------------|-------------------------|
| 100 records  | 2-4 hours               | 3-5 hours               |
| 1,000 records| 20-40 hours             | 30-50 hours             |
| 10,000 records| 8-16 days              | 12-20 days              |

*Estimates assume using GPT-4 or similar model with batch processing*

## Cost Estimates (Using GPT-4)

- **Task 1 (Annotation)**: ~$0.50-1.00 per patient record
- **Task 2 (Classification)**: ~$0.70-1.50 per patient record

*Costs vary based on clinical text length and model used*

## Tips for Success

1. **Start Small**: Begin with 10-100 records to test prompts and schemas
2. **Review Quality**: Manually review first few outputs before processing thousands
3. **Use RAG**: Clinical guidelines significantly improve annotation quality
4. **Batch Processing**: Process in batches of 50-100 for easier monitoring
5. **Save Progress**: ClinOrchestra saves progress, so you can pause/resume
6. **Monitor Costs**: Keep track of API usage, especially for large datasets

## Common Issues

**Issue**: Script can't find MIMIC-IV files
**Solution**: Check path, ensure you have hosp/, icu/, note/ folders

**Issue**: Out of memory when loading labevents.csv
**Solution**: The extraction script samples large files automatically

**Issue**: Prompt too long for model
**Solution**: Some discharge notes are very long. You may need to truncate or use a model with larger context window

**Issue**: JSON schema validation errors
**Solution**: Ensure the schema file is uploaded correctly in ClinOrchestra

## Next Steps After Dataset Creation

1. **Quality Review**: Sample and validate annotations
2. **Train LLM**: Use for fine-tuning medical language models
3. **Evaluate**: Compare predictions against ground truth
4. **Analyze**: Study diagnostic patterns and model performance
5. **Iterate**: Refine prompts based on results

## Support & Documentation

- Full documentation: See `README.md`
- Content mapping template: `content_mappings/template_content_mapping.json`
- Example functions: `functions/function_calculate_*.py`
- Extraction patterns: `patterns/*.txt`

## Citation

If you use this dataset, please cite:
- MIMIC-IV Database (PhysioNet)
- ClinOrchestra
- Your research describing the dataset creation

---

**Questions?** Check the main README.md or open an issue.

**Ready to start?** Run Step 1 above! 🚀
