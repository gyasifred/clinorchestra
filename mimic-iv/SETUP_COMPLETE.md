# MIMIC-IV Project Setup Complete! 🎉

## Your Top 20 Diagnoses from MIMIC-IV

Based on your actual MIMIC-IV data analysis (76,594 total cases):

### Cardiovascular (10 diagnoses, 38.4% of cases)
1. Chest pain, unspecified (78650) - 7,297 cases
2. Coronary atherosclerosis (41401) - 5,751 cases
3. Other chest pain (78659) - 5,212 cases
4. Chest pain, unspecified (R079) - 2,906 cases
5. NSTEMI (I214) - 3,265 cases
6. Subendocardial infarction (41071) - 2,873 cases
7. Atrial fibrillation (42731) - 3,090 cases
8. Hypertensive heart + CKD (I130) - 3,313 cases

### Infectious (4 diagnoses, 19.2% of cases)
9. Sepsis, unspecified (A419) - 5,095 cases
10. Unspecified septicemia (0389) - 3,144 cases
11. Pneumonia, organism unspecified (486) - 3,726 cases
12. UTI, site not specified (5990) - 2,967 cases

### Renal (2 diagnoses, 8.1% of cases)
13. Acute kidney failure (5849) - 3,532 cases
14. Acute kidney failure (N179) - 2,653 cases

### Psychiatric/Substance (4 diagnoses, 19.9% of cases)
15. Major depressive disorder (F329) - 3,703 cases
16. Depressive disorder (311) - 3,499 cases
17. Alcohol abuse (30500) - 4,591 cases
18. Alcohol abuse with intoxication (F10129) - 3,447 cases

### Oncology (2 diagnoses, 8.5% of cases)
19. Encounter for chemotherapy (Z5111) - 3,557 cases
20. Encounter for chemotherapy (V5811) - 2,973 cases

---

## What's Ready for ClinOrchestra

### ✅ EXTRAS (16 JSON files) - Clinical Knowledge
Upload to **ClinOrchestra → Extras Tab**

**Diagnosis-Specific Extras:**
1. `chest_pain_evaluation.json` - Chest pain diagnostic approach, RED FLAGS
2. `coronary_artery_disease.json` - CAD risk factors, diagnosis, treatment
3. `nstemi_criteria.json` - NSTEMI diagnosis, TIMI/GRACE scores
4. `atrial_fibrillation.json` - AFib types, CHA2DS2-VASc, HAS-BLED
5. `pneumonia_criteria.json` - CAP vs HAP, CURB-65, PSI
6. `sepsis_diagnostic_criteria.json` - Sepsis-3, qSOFA, SOFA
7. `uti_criteria.json` - UTI types, complications, treatment
8. `aki_detailed.json` - KDIGO staging, prerenal/intrinsic/postrenal
9. `aki_staging.json` - Additional AKI criteria
10. `depression_criteria.json` - DSM-5, PHQ-9, suicide assessment
11. `alcohol_use_disorder.json` - DSM-5 criteria, withdrawal, CIWA
12. `chemotherapy_encounter.json` - Not a disease, encounter code context
13. `hypertensive_heart_ckd.json` - Combined HHD + CKD pathology
14. `heart_failure_classification.json` - Acute vs chronic, HFrEF vs HFpEF
15. `respiratory_failure_types.json` - Type 1 vs 2, ABG interpretation
16. `clinical_annotation_approach.json` - Systematic methodology

### ✅ PATTERNS (12 JSON files) - Text Extraction
Upload to **ClinOrchestra → Patterns Tab**

**Vital Signs:**
- `vital_signs_bp.json` - Blood pressure
- `vital_signs_hr.json` - Heart rate
- `vital_signs_rr.json` - Respiratory rate
- `vital_signs_temp.json` - Temperature
- `vital_signs_spo2.json` - Oxygen saturation

**Labs:**
- `lab_wbc.json` - White blood cell count
- `lab_creatinine.json` - Creatinine (renal)
- `lab_lactate.json` - Lactate (sepsis)
- `lab_bun.json` - BUN (renal)
- `cardiac_troponin.json` - Troponin (cardiac)
- `cardiac_bnp.json` - BNP (heart failure)

**Clinical Scores:**
- `gcs_score.json` - Glasgow Coma Scale

### ✅ FUNCTIONS (7 JSON files) - Clinical Calculations
Upload to **ClinOrchestra → Functions Tab**

**Severity Scores:**
- `calculate_sofa_score.json` - Sepsis/organ dysfunction (0-24)
- `calculate_curb65.json` - Pneumonia severity (0-5)
- `calculate_ciwa.json` - Alcohol withdrawal (0-67)
- `calculate_phq9.json` - Depression severity (0-27)
- `calculate_chadsvasc.json` - AFib stroke risk (0-9)

**Clinical Calculations:**
- `calculate_map.json` - Mean arterial pressure
- `calculate_creatinine_clearance.json` - Renal function (Cockcroft-Gault)

### ✅ PROMPTS - Ready to Use

**Task 1: Clinical Consultant Annotation**
- File: `prompts/task1_annotation_prompt.txt`
- Updated with consultant perspective
- Comprehensive 10-category evidence extraction

**Task 2: Multiclass Classification**
- File: `prompts/task2_classification_prompt_FINAL.txt`
- **Contains YOUR actual top 20 diagnoses with case counts**
- Diagnosis-specific guidance for each condition
- Probability assignment framework

### ✅ SCHEMAS - JSON Output Format

**Task 1:**
- `schemas/task1_annotation_schema.json`
- Comprehensive evidence annotation structure

**Task 2:**
- `schemas/task2_classification_schema_v2.json`
- Multiclass probability prediction (all 20 diagnoses)

### ✅ RESOURCES

**Clinical Guidelines:**
- File: `CLINICAL_RESOURCES.md`
- Direct links to official guidelines for each diagnosis
- AHA/ACC, IDSA, KDIGO, APA, ASAM guidelines
- Clinical calculators (MDCalc)
- Evidence-based medicine resources

**Usage Guide:**
- File: `USAGE_GUIDE.md`
- Step-by-step instructions
- File format reference
- Troubleshooting

**Extraction Script:**
- `scripts/get_top_diagnoses_simple.py` - Already ran successfully

---

## Quick Start Steps

### 1. Open ClinOrchestra UI

### 2. Load Extras
- Go to **Extras Tab**
- Upload all 16 JSON files from `mimic-iv/extras/`
- These provide diagnostic criteria and clinical knowledge to help the LLM

### 3. Load Patterns
- Go to **Patterns Tab**
- Upload all 12 JSON files from `mimic-iv/patterns/`
- These standardize clinical text before LLM processing

### 4. Load Functions
- Go to **Functions Tab**
- Upload all 7 JSON files from `mimic-iv/functions/`
- These enable clinical calculations

### 5. Set Up Task 1 (Annotation)
1. **Prompt Tab**: Copy `prompts/task1_annotation_prompt.txt`
2. **Data Tab**: Upload your annotation dataset CSV
3. **Processing Tab**: Upload `schemas/task1_annotation_schema.json`
4. **Configure batch settings**: 10-50 records per batch
5. **Start Processing**

### 6. Set Up Task 2 (Classification)
1. **Prompt Tab**: Copy `prompts/task2_classification_prompt_FINAL.txt`
   - This prompt already has your top 20 diagnoses embedded!
2. **Data Tab**: Upload classification dataset CSV
3. **Processing Tab**: Upload `schemas/task2_classification_schema_v2.json`
4. **Start Processing**

### 7. Evaluate Results (Task 2)
```bash
python scripts/evaluate_classification.py
```
Get comprehensive metrics: accuracy, calibration, per-diagnosis performance

---

## File Summary

```
mimic-iv/
├── SETUP_COMPLETE.md          ← YOU ARE HERE
├── CLINICAL_RESOURCES.md       ← Clinical guidelines with links
├── USAGE_GUIDE.md              ← How to use everything
├── README.md                   ← Full documentation
│
├── extras/ (16 files)          ← Upload to ClinOrchestra Extras Tab
│   ├── chest_pain_evaluation.json
│   ├── coronary_artery_disease.json
│   ├── nstemi_criteria.json
│   ├── atrial_fibrillation.json
│   ├── pneumonia_criteria.json
│   ├── sepsis_diagnostic_criteria.json
│   ├── uti_criteria.json
│   ├── aki_detailed.json
│   ├── aki_staging.json
│   ├── depression_criteria.json
│   ├── alcohol_use_disorder.json
│   ├── chemotherapy_encounter.json
│   ├── hypertensive_heart_ckd.json
│   ├── heart_failure_classification.json
│   ├── respiratory_failure_types.json
│   └── clinical_annotation_approach.json
│
├── patterns/ (12 files)        ← Upload to ClinOrchestra Patterns Tab
│   ├── vital_signs_*.json (5 files)
│   ├── lab_*.json (4 files)
│   ├── cardiac_*.json (2 files)
│   └── gcs_score.json
│
├── functions/ (7 files)        ← Upload to ClinOrchestra Functions Tab
│   ├── calculate_sofa_score.json
│   ├── calculate_curb65.json
│   ├── calculate_ciwa.json
│   ├── calculate_phq9.json
│   ├── calculate_chadsvasc.json
│   ├── calculate_map.json
│   └── calculate_creatinine_clearance.json
│
├── prompts/
│   ├── task1_annotation_prompt.txt              ← Use this
│   └── task2_classification_prompt_FINAL.txt   ← Use this (has your top 20!)
│
├── schemas/
│   ├── task1_annotation_schema.json
│   └── task2_classification_schema_v2.json
│
└── scripts/
    ├── get_top_diagnoses_simple.py    ← Already ran ✓
    ├── extract_dataset.py             ← Run next to create datasets
    └── evaluate_classification.py     ← Run after Task 2 processing
```

---

## What Extras, Patterns, and Functions Actually Do

### EXTRAS = Clinical Knowledge for the LLM

**Purpose**: Provide the LLM with expert clinical knowledge to improve annotation quality

**How ClinOrchestra uses them**:
1. Reads your prompt and schema to identify keywords (diagnosis names, clinical terms)
2. Matches extras based on metadata categories and keywords
3. Injects relevant extras into the LLM context automatically
4. LLM uses this knowledge to:
   - Apply correct diagnostic criteria
   - Recognize severity indicators
   - Use proper clinical terminology
   - Link findings to pathophysiology

**Example**: When annotating a sepsis case:
- Extra provides Sepsis-3 criteria
- LLM recognizes SOFA score requirements
- LLM correctly identifies organ dysfunction
- LLM applies appropriate severity staging

### PATTERNS = Text Standardization

**Purpose**: Clean and standardize clinical text BEFORE the LLM sees it

**How ClinOrchestra uses them**:
1. Runs regex patterns on input text sequentially
2. Replaces messy variations with standard format
3. Cleaned text goes to LLM
4. Makes extraction more reliable

**Example**:
- Input: "BP 120 / 80", "blood pressure: 120/80", "BP:120-80"
- Pattern standardizes all to: "BP: 120/80"
- LLM sees consistent format, extracts more accurately

### FUNCTIONS = Executable Calculations

**Purpose**: Enable the LLM to perform clinical calculations

**How ClinOrchestra uses them**:
1. LLM can call functions in its output
2. Functions execute with provided parameters
3. Results incorporated into final output

**Example**:
- LLM extracts: platelets=95, Cr=2.8, on vasopressors
- LLM calls: `calculate_sofa_score(platelets=95, creatinine=2.8, on_vasopressor=True)`
- Function returns: SOFA score = 7 (moderate-severe organ dysfunction)
- LLM includes this in annotation

---

## Expected Performance

### Task 1 (Annotation)
- **Processing time**: 2-5 minutes per case (depends on model)
- **Output size**: 200-500 clinical data points per case
- **Quality**: Comprehensive, structured, cited evidence

### Task 2 (Classification)
- **Processing time**: 3-7 minutes per case
- **Target metrics**:
  - Top-1 Accuracy: >60-70%
  - Top-5 Accuracy: >85-90%
  - Cross-Entropy: <0.7
  - Brier Score: <0.4
- **Output**: Probabilities for all 20 diagnoses + reasoning

---

## Cost Estimates

Using GPT-4 (adjust for your model):
- **Task 1**: $0.50-1.00 per case
- **Task 2**: $0.70-1.50 per case

For 1,000 cases: ~$1,500-2,500

**Tip**: Start with 100-case sample to test quality before processing thousands.

---

## Next Steps

1. ✅ **You've already extracted top 20 diagnoses** - Done!
2. ⏭️ **Create your datasets** - Run `extract_dataset.py`
3. ⏭️ **Load extras/patterns/functions into ClinOrchestra** - Use UI
4. ⏭️ **Process Task 1 (Annotation)** - Build consultant training data
5. ⏭️ **Process Task 2 (Classification)** - Build classification dataset
6. ⏭️ **Evaluate Task 2** - Run `evaluate_classification.py`
7. ⏭️ **Iterate** - Refine prompts based on results

---

## Support

- **Full docs**: README.md
- **Quick reference**: USAGE_GUIDE.md
- **Clinical resources**: CLINICAL_RESOURCES.md
- **All guidelines**: Links in CLINICAL_RESOURCES.md

---

**Everything is ready to go! 🚀**

Upload extras/patterns/functions to ClinOrchestra and start processing!
