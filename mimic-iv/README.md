# MIMIC-IV Clinical Annotation & Classification Project

This project uses ClinOrchestra to create comprehensive clinical annotation datasets from MIMIC-IV for training medical AI systems.

## Project Overview

### Two Main Tasks

#### **Task 1: Clinical Consultant AI Training (Comprehensive Annotation)**
- **Input**: Patient clinical record + Known primary diagnosis
- **Output**: Comprehensive extraction of all clinical evidence, contextualized findings, and expert clinical reasoning
- **Purpose**: Train LLMs to function as **CLINICAL CONSULTANTS** - not just diagnostic tools, but comprehensive clinical reasoning systems that can:
  - Integrate complex multi-system clinical data
  - Provide evidence-based clinical assessments
  - Explain clinical reasoning and decision-making
  - Identify critical findings and their clinical significance
  - Support differential diagnosis and clinical management
- **Example**: Given "Patient has Sepsis (ICD: A41.9)" → Extract ALL clinical evidence with expert-level analysis: symptoms (onset, severity, progression), vital signs (trends, clinical significance), labs (abnormal values, pathophysiology), imaging findings, treatments (indications, responses), medical history, risk factors, clinical reasoning, temporal timeline, severity assessment
- **Goal**: Create training data for developing AI systems that think like experienced clinicians providing consultative-level clinical analysis

#### **Task 2: Multiclass Diagnosis Prediction (Classification)**
- **Input**: Patient clinical record only (diagnosis hidden)
- **Output**: Probability scores for ALL 20 possible diagnoses (multiclass classification) + detailed reasoning
- **Purpose**: Train and evaluate AI systems for **DIAGNOSTIC PREDICTION** using multiclass classification
  - Predict probability distribution across all 20 diagnoses
  - Evaluate using classification metrics (accuracy, precision, recall, F1, cross-entropy, Brier score)
  - Assess probability calibration and diagnostic performance
- **Example**: Given clinical presentation → Output:
  ```
  Sepsis: 0.75 (supporting: fever, elevated lactate, hypotension...)
  Pneumonia: 0.12 (supporting: infiltrate on CXR, but...)
  Heart Failure: 0.08 (some overlap but...)
  [... all 20 diagnoses with probabilities summing to 1.0]
  Top prediction: Sepsis (75% confidence)
  ```
- **Evaluation**: Compare predictions against ground truth using:
  - Top-1, Top-3, Top-5 Accuracy
  - Per-diagnosis Precision/Recall/F1
  - Cross-Entropy Loss (probability calibration)
  - Brier Score (prediction accuracy)
  - Confusion Matrix
- **Goal**: Develop and evaluate diagnostic prediction models with well-calibrated probability estimates

### Dataset Scope
- **Focus**: Top 20 most common primary diagnoses in MIMIC-IV
- **Data Sources**: Discharge summaries, radiology reports, lab results, medications, vital signs
- **Scale**: Designed to create a massive, high-quality dataset for LLM training

## Project Structure

```
mimic-iv/
├── README.md                                    # This file
├── QUICKSTART.md                                # Quick start guide
├── scripts/
│   ├── get_top_diagnoses_simple.py             # Step 1A: Extract top 20 individual ICD codes
│   ├── get_top_diagnoses_consolidated.py       # Step 1B: Extract consolidated diagnoses
│   ├── analyze_clinical_notes.py               # Step 2: Analyze note lengths & statistics
│   ├── extract_dataset.py                      # Step 3A: Create datasets (individual codes)
│   ├── extract_dataset_consolidated.py         # Step 3B: Create datasets (consolidated)
│   ├── create_balanced_train_test.py           # Step 4: Create balanced train/test split
│   ├── eda_train_test_publication.py           # Step 5: Generate EDA and publication tables
│   ├── gather_clinical_guidelines.py           # Step 6: Helper for collecting guidelines
│   ├── evaluate_classification.py              # Step 8: Evaluate classification predictions
│   ├── generate_classification_prompt.py       # Utility: Generate prompt with diagnosis list
│   └── README_*.md                              # Additional documentation
├── prompts/
│   ├── task1_annotation_prompt.txt             # Consultant AI training prompt
│   ├── task2_classification_prompt.txt         # Original classification prompt
│   └── task2_classification_prompt_v2.txt      # Multiclass classification prompt (USE THIS)
├── schemas/
│   ├── task1_annotation_schema.json            # Comprehensive annotation schema
│   ├── task2_classification_schema.json        # Original classification schema
│   └── task2_classification_schema_v2.json     # Multiclass prediction schema (USE THIS)
├── content_mappings/
│   └── template_content_mapping.json           # Template for diagnosis mappings
├── patterns/
│   ├── vital_signs_patterns.txt                # Vital signs extraction patterns
│   └── lab_values_patterns.txt                 # Lab values extraction patterns
├── functions/
│   ├── function_calculate_sofa_score.py        # SOFA score (sepsis severity)
│   └── function_calculate_curb65.py            # CURB-65 (pneumonia severity)
├── guidelines/
│   └── [Clinical practice guidelines - to be populated]
└── extras/
    └── [Additional clinical resources]
```

## Getting Started

### Prerequisites
1. **MIMIC-IV Dataset**: You must have access to MIMIC-IV database (requires credentialing through PhysioNet)
2. **Python 3.8+** with pandas, numpy
3. **ClinOrchestra** installed and configured

### Step-by-Step Workflow

#### **Step 1: Extract Top 20 Primary Diagnoses**

Choose ONE of the following approaches:

**Option A: Simple extraction (individual ICD codes)**
```bash
cd mimic-iv/scripts
python get_top_diagnoses_simple.py /path/to/mimic-iv-3.1
```

**Option B: Consolidated extraction (groups related ICD-9/ICD-10 codes)**
```bash
python get_top_diagnoses_consolidated.py /path/to/mimic-iv-3.1
```

**Outputs**:
- Option A: `top_20_primary_diagnoses.csv` (20 individual ICD codes)
- Option B: `top_diagnoses_consolidated.csv` (13 consolidated diagnoses)

#### **Step 2: Analyze Clinical Notes (Optional but Recommended)**

Before creating datasets, analyze clinical note characteristics:

```bash
python analyze_clinical_notes.py /path/to/mimic-iv-3.1
```

**What it does**:
- Calculates average, median, min, max text lengths
- Analyzes word counts and line counts
- Generates statistics by note type (discharge, radiology)
- Creates visualization of note length distributions

**Output**: `clinical_notes_analysis.csv` and visualizations

#### **Step 3: Create Annotation & Classification Datasets**

Choose based on Step 1 option:

**Option A: Standard datasets (20 individual diagnoses)**
```bash
python extract_dataset.py
```

**Option B: Consolidated datasets (13 consolidated diagnoses)**
```bash
python extract_dataset_consolidated.py
```

When prompted:
- Enter MIMIC-IV path
- Enter sample size (optional - for testing with subset)

**Outputs**:
- `annotation_dataset.csv` - For Task 1 (Evidence Extraction)
- `classification_dataset.csv` - For Task 2 (Diagnosis Prediction)

#### **Step 4: Create Balanced Train/Test Split (Optional)**

If you want to create a balanced training and testing split:

```bash
python create_balanced_train_test.py
```

**What it does**:
- Creates balanced 5000-sample subset (4000 train, 1000 test)
- Stratifies by diagnosis, gender, race, text complexity
- Ensures reproducibility with fixed random seed

**Outputs**:
- `train_dataset_4000.csv`
- `test_dataset_1000.csv`
- `train_test_split_metadata.txt`

#### **Step 5: Exploratory Data Analysis (Optional)**

Generate publication-ready statistics and visualizations:

```bash
python eda_train_test_publication.py
```

**What it generates**:
- Table 1: Baseline characteristics (CSV, HTML, LaTeX)
- Table 2: Diagnosis distribution (CSV, HTML, LaTeX)
- Figure 1: Diagnosis distribution comparison (PNG, PDF)
- Figure 2: Demographics (PNG, PDF)
- Figure 3: Text complexity analysis (PNG, PDF)
- Methods section text for manuscripts
- Supplementary statistics

**Output directory**: `eda_results/`

#### **Step 6: Gather Clinical Guidelines (Optional)**

Generate search queries and organize guideline collection:

```bash
python gather_clinical_guidelines.py
```

**What it does**:
- Creates directory structure for each diagnosis
- Generates PubMed search URLs
- Provides recommended clinical societies and sources
- Creates collection checklist

**Output**: `mimic-iv/guidelines/` with subdirectories and collection guides

#### **Step 7: Configure ClinOrchestra**

1. Open ClinOrchestra UI
2. Navigate to the **Prompt** tab
3. Copy the contents of `mimic-iv/prompts/task1_annotation_prompt.txt`
4. Paste into the prompt field
5. Navigate to the **RAG** tab
6. Upload relevant clinical guidelines from `mimic-iv/guidelines/` (if collected)
7. Navigate to the **Processing** tab
8. Upload the JSON schema from `mimic-iv/schemas/task1_annotation_schema.json`
9. Load your dataset `mimic-iv/annotation_dataset.csv`
10. Configure batch processing settings
11. Start annotation process

Repeat for Task 2 using the classification prompt, schema, and dataset.

#### **Step 8: Evaluate Classification Results (Optional)**

After running Task 2 classification, evaluate model performance:

```bash
python evaluate_classification.py
```

**What it computes**:
- Top-1, Top-3, Top-5 accuracy
- Cross-entropy loss and Brier score
- Per-class precision, recall, F1-score
- Confusion matrix visualization

**Output**: `evaluation_results/` with metrics and visualizations

## Dataset Column Descriptions

### Annotation Dataset (Task 1)
| Column | Description |
|--------|-------------|
| `subject_id` | Unique patient identifier |
| `hadm_id` | Unique hospital admission identifier |
| `icd_code` | ICD code of primary diagnosis |
| `icd_version` | ICD version (9 or 10) |
| `primary_diagnosis_name` | Full name of diagnosis |
| `clinical_text` | Combined discharge summary and radiology reports |
| `admission_type` | Type of admission (EMERGENCY, ELECTIVE, etc.) |
| `gender` | Patient gender |
| `anchor_age` | Patient age (de-identified anchor) |
| `race` | Patient race/ethnicity |
| `insurance` | Insurance type |
| `admittime` | Admission timestamp |
| `dischtime` | Discharge timestamp |
| `hospital_expire_flag` | Whether patient died in hospital (0/1) |

### Classification Dataset (Task 2)
Same as above, but the diagnosis columns are included only as ground truth for evaluation - they should NOT be included in the prompt to the model.

## Prompts

### Task 1: Annotation Prompt
Located at: `mimic-iv/prompts/task1_annotation_prompt.txt`

**Key Features**:
- Systematic extraction across 10 evidence categories
- Quality levels for each piece of evidence (DEFINITIVE, STRONG, MODERATE, WEAK, CONTEXTUAL)
- Requires direct quotes from clinical text
- Links evidence to diagnostic criteria
- Comprehensive coverage of symptoms, labs, imaging, medications, history, risk factors

**Usage**:
- Copy entire contents into ClinOrchestra prompt field
- Variables `{subject_id}`, `{hadm_id}`, `{primary_diagnosis_name}`, `{icd_code}`, `{clinical_text}`, etc. will be auto-filled from dataset columns

### Task 2: Classification Prompt
Located at: `mimic-iv/prompts/task2_classification_prompt.txt`

**Key Features**:
- Structured 5-phase diagnostic approach
- Systematic data extraction → Pattern recognition → Differential diagnosis → Prediction → Alternatives
- Explicitly addresses cognitive biases
- Requires detailed clinical reasoning
- Confidence scoring and uncertainty acknowledgment

**Usage**:
- Copy into ClinOrchestra prompt field
- Requires dynamic insertion of top 20 diagnoses list (will be generated from your top_20_primary_diagnoses.csv)

## JSON Schemas

### Task 1: Annotation Schema
Located at: `mimic-iv/schemas/task1_annotation_schema.json`

**Key Sections**:
- `evidence_summary`: Overall quality and key findings
- `symptoms_and_presentation`: All symptoms with onset, severity, progression
- `physical_examination`: Vital signs and exam findings
- `laboratory_results`: All lab values with abnormal flags
- `imaging_and_diagnostics`: Radiology and other studies
- `medications_and_treatments`: Medications with responses
- `medical_history`: Past medical, surgical, family history
- `risk_factors`: Lifestyle, environmental, demographic
- `clinical_reasoning`: Diagnostic criteria, differential diagnoses
- `temporal_timeline`: Disease progression over time
- `severity_assessment`: Complications, staging, functional impact

### Task 2: Multiclass Classification Schema (V2 - RECOMMENDED)
Located at: `mimic-iv/schemas/task2_classification_schema_v2.json`

**Key Sections**:
- `clinical_data_extraction`: Systematically extracted clinical findings
- `clinical_pattern_analysis`: Pattern recognition, organ systems, severity
- `multiclass_prediction`: **PROBABILITY SCORES FOR ALL 20 DIAGNOSES** (must sum to 1.0)
  - Each diagnosis gets: probability, supporting evidence, contradicting evidence, reasoning
  - Enforces probability calibration
- `top_diagnosis`: Single most likely diagnosis with detailed reasoning
- `top_5_differential`: Top 5 ranked diagnoses for evaluation
- `clinical_reasoning`: Detailed diagnostic thought process

**Critical Feature**:
This schema enforces **true multiclass classification** - the model must assign probabilities to ALL 20 diagnoses that sum to 1.0, enabling proper evaluation of probability calibration and diagnostic confidence.

## Content Mappings

Content mappings will be created after identifying the specific top 20 diagnoses. These will map:
- Diagnosis-specific symptoms
- Relevant lab tests for each diagnosis
- Typical imaging findings
- Standard treatments
- Diagnostic criteria

Location: `mimic-iv/content_mappings/[diagnosis_name]_mapping.json`

## Clinical Guidelines

For each of the top 20 diagnoses, we will gather:
- Clinical practice guidelines (AHA, ATS, IDSA, etc.)
- Diagnostic criteria (established scoring systems)
- Evidence-based treatment protocols
- Recent literature from PubMed, Nature, NEJM, etc.

These will be used as RAG knowledge base in ClinOrchestra to improve annotation quality.

Location: `mimic-iv/guidelines/[diagnosis_name]/`

## Extraction Patterns

Regular expression patterns for extracting clinical entities:
- Vital signs patterns
- Lab value patterns
- Medication patterns
- Temporal expressions
- Severity indicators

Location: `mimic-iv/patterns/`

## Clinical Functions

Custom functions for clinical calculations:
- Severity scores (APACHE, SOFA, CURB-65, etc.)
- Risk calculators
- Lab value interpretations
- Clinical criteria evaluators

Location: `mimic-iv/functions/`

## Expected Output Quality

### Annotation Task (Task 1)
For each patient, you should get:
- **Comprehensive**: 200-500 data points extracted per patient
- **Structured**: All evidence organized into defined categories
- **Cited**: Direct quotes from clinical text for verification
- **Explained**: Clinical reasoning for each piece of evidence
- **Quality-rated**: Each evidence item rated for strength

### Classification Task (Task 2 - Multiclass)
For each patient, you should get:
- **Probability distribution**: Probability scores for ALL 20 diagnoses (sum = 1.0)
- **Top diagnosis**: Highest probability diagnosis with detailed reasoning
- **Top-5 differential**: Top 5 ranked diagnoses with probabilities
- **Supporting/Contradicting evidence**: For each of the 20 diagnoses
- **Clinical reasoning**: Step-by-step diagnostic thinking
- **Probability justification**: Explanation for each probability assignment
- **Well-calibrated**: Probabilities reflect true diagnostic confidence

## Evaluation of Classification Results

After running Task 2 (multiclass classification), evaluate your model's performance:

### Running Evaluation

```bash
cd mimic-iv/scripts
python evaluate_classification.py
```

When prompted, provide:
- Path to predictions JSON file (output from ClinOrchestra)
- Path to ground truth CSV (classification_dataset.csv with actual diagnoses)
- Path to top 20 diagnoses mapping

### Metrics Computed

**Accuracy Metrics**:
- **Top-1 Accuracy**: Percentage where correct diagnosis has highest probability
- **Top-3 Accuracy**: Percentage where correct diagnosis is in top 3
- **Top-5 Accuracy**: Percentage where correct diagnosis is in top 5

**Calibration Metrics**:
- **Cross-Entropy Loss**: Measures how well probabilities match reality (lower is better)
- **Brier Score**: Mean squared error of probability predictions (0-2, lower is better)

**Per-Class Metrics**:
- **Precision**: Of predictions for diagnosis X, how many were correct?
- **Recall**: Of all actual diagnosis X cases, how many were predicted?
- **F1-Score**: Harmonic mean of precision and recall
- **Support**: Number of actual cases for each diagnosis

**Aggregate Metrics**:
- **Macro-averaged**: Simple average across all diagnoses
- **Weighted-averaged**: Weighted by number of cases per diagnosis

**Visualizations**:
- **Confusion Matrix**: Heatmap showing predicted vs actual diagnoses

### Evaluation Output

The script generates:
- `per_class_metrics.csv`: Precision/Recall/F1 for each diagnosis
- `confusion_matrix.png`: Visual confusion matrix
- `evaluation_summary.json`: All metrics in JSON format

### Interpreting Results

**Good Performance**:
- Top-1 Accuracy > 70%
- Top-5 Accuracy > 90%
- Cross-Entropy < 0.5
- Brier Score < 0.3

**Probability Calibration**:
- Well-calibrated: When model says 80% confident, it's correct ~80% of the time
- Cross-entropy and Brier score measure calibration quality
- Lower values = better calibration

**Per-Class Analysis**:
- Identify which diagnoses are easy/hard to predict
- Examine confusion matrix for common misclassifications
- Low recall = model misses this diagnosis
- Low precision = model over-predicts this diagnosis

## Next Steps After Dataset Creation

1. **Quality Review**: Sample and review annotations for accuracy
2. **Inter-rater Reliability**: If multiple annotators, calculate agreement
3. **Dataset Splitting**: Train/Validation/Test splits
4. **LLM Training**: Use for fine-tuning medical LLMs
5. **Evaluation**: Compare predictions against ground truth
6. **Analysis**: Identify common diagnostic patterns, model strengths/weaknesses

## Research Applications

This dataset can be used for:
- **Training clinical LLMs** for diagnostic reasoning
- **Evaluation benchmarks** for medical AI systems
- **Clinical decision support** system development
- **Medical education** case studies
- **Research** on diagnostic patterns in critical care
- **Natural language processing** for clinical text

## Important Notes

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

## Citation

If you use this dataset or methodology, please cite:
- MIMIC-IV Database (PhysioNet)
- ClinOrchestra (this project)
- Your own research paper describing the dataset creation

## Scripts Reference

### Core Data Extraction Scripts

#### `get_top_diagnoses_simple.py`
- **Purpose**: Extract top 20 most common primary diagnoses (individual ICD codes)
- **Input**: MIMIC-IV directory path
- **Output**: `top_20_primary_diagnoses.csv`
- **Use when**: You want to work with individual ICD-9 and ICD-10 codes separately

#### `get_top_diagnoses_consolidated.py`
- **Purpose**: Extract and consolidate diagnoses (groups ICD-9/ICD-10 codes for same condition)
- **Input**: MIMIC-IV directory path, uses diagnosis_mapping.py
- **Output**: `top_diagnoses_consolidated.csv`, `top_diagnoses_detailed_breakdown.csv`
- **Use when**: You want to treat "Chest pain ICD-9" and "Chest pain ICD-10" as one diagnosis
- **Reduces**: 20+ codes → 13 consolidated diagnoses

#### `analyze_clinical_notes.py`
- **Purpose**: Analyze clinical note characteristics and length statistics
- **Input**: MIMIC-IV directory path
- **Output**: `clinical_notes_analysis.csv`, visualization charts
- **Features**:
  - Average, median, min, max text length (characters)
  - Word count and line count statistics
  - Analysis by note type (discharge, radiology)
  - Text length distribution visualizations
  - Per-diagnosis text length statistics
- **Use when**: You want to understand note complexity before processing

#### `extract_dataset.py`
- **Purpose**: Create annotation and classification datasets (individual codes)
- **Input**: MIMIC-IV directory, top_20_primary_diagnoses.csv
- **Output**: `annotation_dataset.csv`, `classification_dataset.csv`
- **Features**:
  - Combines discharge summaries + radiology reports
  - Patient demographics and admission info
  - Optional: labs, medications, vital signs
  - Sample size option for testing

#### `extract_dataset_consolidated.py`
- **Purpose**: Create datasets with consolidated diagnoses
- **Input**: MIMIC-IV directory (uses diagnosis_mapping.py)
- **Output**: `annotation_dataset_consolidated.csv`, `classification_dataset_consolidated.csv`
- **Use when**: Working with consolidated diagnoses from Step 1B

### Train/Test Split and Analysis Scripts

#### `create_balanced_train_test.py`
- **Purpose**: Create balanced stratified train/test split
- **Input**: classification_dataset.csv (or any large dataset)
- **Output**: `train_dataset_4000.csv`, `test_dataset_1000.csv`, metadata
- **Features**:
  - 5000 balanced samples (80/20 split)
  - Stratifies by: diagnosis, gender, race, text complexity
  - Maintains proportional diagnosis distribution
  - Fixed random seed for reproducibility
- **Use when**: You need a balanced subset for training/evaluation

#### `eda_train_test_publication.py`
- **Purpose**: Generate publication-ready EDA with statistics and visualizations
- **Input**: train_dataset_4000.csv, test_dataset_1000.csv
- **Output**: Multiple files in `eda_results/` directory
  - Table 1: Baseline characteristics (CSV, HTML, LaTeX)
  - Table 2: Diagnosis distribution (CSV, HTML, LaTeX)
  - Figure 1-3: High-resolution visualizations (PNG, PDF)
  - Methods section text for manuscripts
  - Supplementary statistics
- **Features**:
  - Statistical comparisons (t-test, chi-square)
  - Publication-quality tables and figures
  - Ready-to-use methods section text
- **Use when**: Preparing data for publication or presentation

### Evaluation and Utility Scripts

#### `evaluate_classification.py`
- **Purpose**: Evaluate multiclass diagnosis predictions
- **Input**: predictions JSON, ground truth CSV, diagnosis mapping
- **Output**: Per-class metrics, confusion matrix, evaluation summary
- **Metrics**:
  - Top-1, Top-3, Top-5 accuracy
  - Cross-entropy loss (probability calibration)
  - Brier score
  - Per-class precision, recall, F1-score
  - Confusion matrix visualization
- **Use when**: Evaluating classification model performance

#### `gather_clinical_guidelines.py`
- **Purpose**: Generate guideline collection guide with search queries
- **Input**: top_20_primary_diagnoses.csv
- **Output**: Directory structure + JSON guides for each diagnosis
- **Features**:
  - PubMed search URLs (pre-configured)
  - Recommended clinical societies and sources
  - File naming conventions
  - Collection checklist
- **Use when**: Setting up RAG knowledge base for ClinOrchestra

#### `generate_classification_prompt.py`
- **Purpose**: Insert actual diagnosis list into classification prompt template
- **Input**: top_20_primary_diagnoses.csv, prompt template
- **Output**: Generated prompt with diagnosis list
- **Use when**: Creating final classification prompt for ClinOrchestra

## Quick Reference: Which Scripts to Use

**For Standard Workflow (Individual ICD codes)**:
1. `get_top_diagnoses_simple.py`
2. `analyze_clinical_notes.py` (optional)
3. `extract_dataset.py`
4. Continue with ClinOrchestra processing

**For Consolidated Workflow (Grouped diagnoses)**:
1. `get_top_diagnoses_consolidated.py`
2. `analyze_clinical_notes.py` (optional)
3. `extract_dataset_consolidated.py`
4. Continue with ClinOrchestra processing

**For Publication/Research**:
1-3. (Same as above)
4. `create_balanced_train_test.py`
5. `eda_train_test_publication.py`
6. Process with ClinOrchestra
7. `evaluate_classification.py`

## Support

For questions about:
- **MIMIC-IV data**: Contact PhysioNet support
- **ClinOrchestra**: See main project README
- **This specific workflow**: Open an issue in this repository

## License

- MIMIC-IV data: PhysioNet Credentialed Health Data License
- ClinOrchestra code: [Your license]
- Generated annotations: Should follow MIMIC-IV license restrictions

---

**Last Updated**: 2025-11-10

**Status**: Initial setup complete. Ready for Step 1 (extract top diagnoses).

**Contributors**: [Your name/team]
