# MIMIC-IV Clinical Annotation & Classification Project

This project uses ClinOrchestra to create comprehensive clinical annotation datasets from MIMIC-IV for training medical AI systems.

## Project Overview

### Two Main Tasks

#### **Task 1: Clinical Evidence Annotation**
- **Input**: Patient clinical record + Known primary diagnosis
- **Output**: Comprehensive extraction of all clinical evidence supporting the diagnosis
- **Purpose**: Create training data for diagnostic reasoning and clinical evidence extraction
- **Example**: Given "Patient has Sepsis (ICD: A41.9)" → Extract all symptoms, labs, vital signs, treatments that support this diagnosis

#### **Task 2: Diagnosis Classification**
- **Input**: Patient clinical record only (diagnosis hidden)
- **Output**: Predicted primary diagnosis from top 20 most common diagnoses
- **Purpose**: Train AI systems for diagnostic prediction and differential diagnosis
- **Example**: Given clinical presentation → Predict: "Sepsis" (with confidence and reasoning)

### Dataset Scope
- **Focus**: Top 20 most common primary diagnoses in MIMIC-IV
- **Data Sources**: Discharge summaries, radiology reports, lab results, medications, vital signs
- **Scale**: Designed to create a massive, high-quality dataset for LLM training

## Project Structure

```
mimic-iv/
├── README.md                          # This file
├── scripts/
│   ├── extract_top_diagnoses.py      # Step 1: Identify top 20 diagnoses
│   └── extract_dataset.py            # Step 2: Create annotation & classification datasets
├── prompts/
│   ├── task1_annotation_prompt.txt   # Prompt for annotation task
│   └── task2_classification_prompt.txt # Prompt for classification task
├── schemas/
│   ├── task1_annotation_schema.json  # JSON schema for annotation output
│   └── task2_classification_schema.json # JSON schema for classification output
├── content_mappings/
│   └── [To be filled after identifying conditions]
├── extras/
│   └── [Clinical guidelines, diagnostic criteria, etc.]
├── patterns/
│   └── [Regex patterns for clinical entity extraction]
├── functions/
│   └── [Clinical calculation functions]
└── guidelines/
    └── [Clinical practice guidelines for top 20 diagnoses]
```

## Getting Started

### Prerequisites
1. **MIMIC-IV Dataset**: You must have access to MIMIC-IV database (requires credentialing through PhysioNet)
2. **Python 3.8+** with pandas, numpy
3. **ClinOrchestra** installed and configured

### Step-by-Step Workflow

#### **Step 1: Extract Top 20 Primary Diagnoses**

```bash
cd mimic-iv/scripts
python extract_top_diagnoses.py
```

When prompted, enter the path to your MIMIC-IV directory (e.g., `C:\Users\gyasi\Documents\mimic-iv-3.1`)

**Output**: `mimic-iv/top_20_primary_diagnoses.csv`

This will display:
```
TOP 20 PRIMARY DIAGNOSES IN MIMIC-IV
================================================================================
 1. I50.23      (ICD-10) -  5234 cases - Acute on chronic systolic heart failure
 2. J96.01      (ICD-10) -  4821 cases - Acute respiratory failure with hypoxia
 3. A41.9       (ICD-10) -  4156 cases - Sepsis, unspecified
...
```

#### **Step 2: Create Annotation & Classification Datasets**

```bash
python extract_dataset.py
```

When prompted:
- Enter MIMIC-IV path
- Enter path to top_20_primary_diagnoses.csv (or press Enter for default)
- Enter sample size (optional - for testing with subset)

**Outputs**:
- `mimic-iv/annotation_dataset.csv` - For Task 1 (Evidence Extraction)
- `mimic-iv/classification_dataset.csv` - For Task 2 (Diagnosis Prediction)

#### **Step 3: Configure ClinOrchestra**

1. Open ClinOrchestra UI
2. Navigate to the **Prompt** tab
3. Copy the contents of `mimic-iv/prompts/task1_annotation_prompt.txt`
4. Paste into the prompt field
5. Navigate to the **RAG** tab
6. Upload relevant clinical guidelines from `mimic-iv/guidelines/` (to be populated)
7. Navigate to the **Processing** tab
8. Upload the JSON schema from `mimic-iv/schemas/task1_annotation_schema.json`
9. Load your dataset `mimic-iv/annotation_dataset.csv`
10. Configure batch processing settings
11. Start annotation process

Repeat for Task 2 using the classification prompt, schema, and dataset.

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

### Task 2: Classification Schema
Located at: `mimic-iv/schemas/task2_classification_schema.json`

**Key Sections**:
- `clinical_data_extraction`: Organized extraction of all clinical data
- `clinical_pattern_analysis`: Pattern recognition and pathophysiology
- `differential_diagnosis`: Ranked list of possible diagnoses (3-10)
- `primary_diagnosis_prediction`: Final prediction with reasoning
- `alternative_diagnoses`: Why alternatives are less likely
- `clinical_reasoning`: Decision points, biases avoided, confidence factors

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

### Classification Task (Task 2)
For each patient, you should get:
- **Differential diagnosis**: 5-10 ranked possibilities
- **Primary prediction**: Single diagnosis with confidence score
- **Detailed reasoning**: Step-by-step diagnostic thinking
- **Alternatives explained**: Why other diagnoses were less likely
- **Uncertainty acknowledged**: Clear about limitations and missing data

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
