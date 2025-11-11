# Complete MIMIC-IV ClinOrchestra System Overview
# Clinical Consultant AI Training Framework

**Project Goal**: Create comprehensive dataset for training Clinical Consultant AI using MIMIC-IV data

**Dataset Size**: 76,594 cases across top 20 primary diagnoses

**Last Updated**: 2025-11-10

---

## TABLE OF CONTENTS

1. [Task Overview](#task-overview)
2. [Annotation Schema (Task 1)](#annotation-schema-task-1)
3. [Classification Schema (Task 2)](#classification-schema-task-2)
4. [Extras - Clinical Knowledge (16 files)](#extras---clinical-knowledge)
5. [Patterns - Text Standardization (12 files)](#patterns---text-standardization)
6. [Functions - Clinical Calculations (7 files)](#functions---clinical-calculations)
7. [RAG Resources - Guidelines for Embedding](#rag-resources)
8. [Top 20 Diagnoses Distribution](#top-20-diagnoses-distribution)
9. [File Inventory](#file-inventory)
10. [Next Steps](#next-steps)

---

## TASK OVERVIEW

### Task 1: Clinical Consultant Annotation
**Purpose**: Generate comprehensive clinical evidence annotations to train Clinical Consultant AI

**Input**: Patient clinical records + known primary diagnosis
**Output**: Structured evidence extraction with 13 comprehensive categories
**Training Goal**: Teach AI to think like an expert clinical consultant

**Key Features**:
- Evidence-based reasoning
- Diagnostic certainty assessment
- Temporal timeline tracking
- Severity scoring
- Differential diagnosis consideration
- Quality ratings for each piece of evidence

### Task 2: Multiclass Diagnosis Classification
**Purpose**: Predict which of 20 primary diagnoses best fits a patient's clinical presentation

**Input**: Patient clinical records (diagnosis unknown to model)
**Output**: Probability distribution across all 20 diagnoses + detailed reasoning
**Training Goal**: Build diagnostic prediction model with uncertainty quantification

**Key Features**:
- Probabilities for ALL 20 diagnoses (must sum to 1.0)
- Top-5 differential diagnosis
- Evidence for AND against each diagnosis
- Clinical reasoning documentation
- Confidence assessment

---

## ANNOTATION SCHEMA (Task 1)

**File**: `schemas/task1_annotation_schema.json`
**Total Fields**: 13 required top-level objects
**Estimated Output Size**: 200-500 clinical data points per case

### Schema Structure:

```json
{
  "required": [
    "patient_info",           // Demographics, admission details
    "diagnosis_info",         // ICD code, diagnosis name
    "evidence_summary",       // Overall evidence strength, key findings
    "symptoms_and_presentation",  // Chief complaint, symptom list with details
    "physical_examination",   // Vital signs, physical exam findings
    "laboratory_results",     // Lab values with interpretation
    "imaging_and_diagnostics", // Radiology, ECG, echo, etc.
    "medications_and_treatments", // Meds as diagnostic clues
    "medical_history",        // PMH, previous hospitalizations, family hx
    "risk_factors",           // Lifestyle, environmental, demographic
    "clinical_reasoning",     // Diagnostic certainty, criteria met
    "temporal_timeline",      // Chronology of events
    "severity_assessment"     // Severity scores, complications, comorbidities
  ]
}
```

### Key Annotation Components:

#### 1. Evidence Quality Levels
Every piece of evidence is rated:
- **DEFINITIVE**: Pathognomonic finding or gold standard test
- **STRONG**: Highly specific finding with high positive predictive value
- **MODERATE**: Supportive finding, not specific
- **WEAK**: Non-specific finding, requires context
- **CONTEXTUAL**: Only meaningful in combination with other findings

#### 2. Symptoms Documentation
For each symptom, capture:
- Symptom name
- Onset (when started)
- Duration (how long)
- Severity (mild/moderate/severe)
- Character (specific quality)
- Progression (improving/worsening/stable)
- **Evidence quote** (direct text from record)
- Evidence quality rating
- Relevance to diagnosis (how it supports diagnosis)

**Example**:
```json
{
  "symptom": "Chest pain",
  "onset": "3 hours prior to admission",
  "duration": "Persistent for 3 hours",
  "severity": "Severe (8/10)",
  "character": "Crushing, substernal, radiating to left arm",
  "progression": "Worsening",
  "evidence_quote": "Patient reports severe crushing chest pain...",
  "evidence_quality": "STRONG",
  "relevance_to_diagnosis": "Classic anginal symptoms suggest acute coronary syndrome"
}
```

#### 3. Laboratory Results
For each lab, capture:
- Test name
- Value with units
- Reference range
- Abnormal flag (HIGH/LOW/CRITICAL_HIGH/CRITICAL_LOW/NORMAL)
- Timing (when tested)
- Trend (improving/worsening/stable)
- Evidence quote
- Evidence quality
- **Clinical significance** (how this supports diagnosis)

**Example**:
```json
{
  "test_name": "Troponin I",
  "value": "2.5 ng/mL",
  "reference_range": "<0.04 ng/mL",
  "abnormal_flag": "CRITICAL_HIGH",
  "timing": "Admission",
  "trend": "Elevated and rising",
  "evidence_quote": "Troponin I 2.5 ng/mL (elevated)",
  "evidence_quality": "DEFINITIVE",
  "clinical_significance": "Troponin elevation >50x ULN confirms acute myocardial injury, diagnostic for NSTEMI in context of symptoms"
}
```

#### 4. Imaging & Diagnostics
- Study type (CXR, CT, ECG, echo, etc.)
- Body region
- **Findings** (detailed description)
- Evidence quote (from radiology report)
- Evidence quality
- **Clinical significance** (interpretation)

**Example**:
```json
{
  "study_type": "ECG",
  "findings": "ST segment depression 2mm in leads V2-V5, T wave inversions in lateral leads",
  "evidence_quote": "ECG shows diffuse ST depressions and T wave inversions",
  "evidence_quality": "STRONG",
  "clinical_significance": "ECG changes consistent with acute ischemia, supports NSTEMI diagnosis"
}
```

#### 5. Medications as Diagnostic Clues
- Medication name
- **Indication** (why given)
- Dosage, route
- **Response** (did it work?)
- Evidence quote
- **Relevance to diagnosis** (what does this medication tell us?)

**Example**:
```json
{
  "medication_name": "Aspirin 325mg + Ticagrelor 180mg",
  "indication": "Acute coronary syndrome",
  "response": "Chest pain improved after antiplatelet therapy",
  "relevance_to_diagnosis": "DAPT initiated confirms clinical team's diagnosis of ACS. Response to therapy supports ischemic etiology."
}
```

#### 6. Clinical Reasoning Section
Most important for Consultant AI training:
- **Documented clinical reasoning** (what providers wrote)
- **Differential diagnoses considered** (alternatives ruled out)
- **Diagnostic criteria met** (specific criteria checklist)
- **Diagnostic certainty**: CONFIRMED | HIGHLY_PROBABLE | PROBABLE | POSSIBLE | UNCERTAIN
- **Supporting reasoning** (expert explanation of evidence synthesis)

**Example**:
```json
{
  "diagnostic_criteria_met": [
    {
      "criterion_name": "Fourth Universal Definition of MI - Type 1 NSTEMI",
      "criterion_met": true,
      "evidence": "Troponin elevation >99th percentile ULN + ischemic symptoms + ECG changes"
    },
    {
      "criterion_name": "TIMI Risk Score for NSTEMI",
      "criterion_met": true,
      "evidence": "TIMI score = 5/7 (age >65, known CAD, ST changes, elevated troponin, aspirin use)"
    }
  ],
  "diagnostic_certainty": "CONFIRMED",
  "supporting_reasoning": "This case definitively meets criteria for NSTEMI based on the Fourth Universal Definition: (1) myocardial injury evidenced by troponin elevation >50x ULN, (2) ischemic symptoms of crushing chest pain with arm radiation, (3) acute ischemic ECG changes with ST depressions. The diagnosis is further supported by response to antiplatelet therapy and known CAD history."
}
```

#### 7. Temporal Timeline
Critical for understanding disease progression:
- Symptom onset
- Presentation timing
- Diagnosis timing
- **Key events** (chronological)

**Example**:
```json
{
  "key_events": [
    {
      "event": "Onset of crushing chest pain",
      "timing": "08:00 AM, 3 hours before ED presentation",
      "significance": "Symptom onset defines window for reperfusion therapy"
    },
    {
      "event": "ED arrival and first troponin",
      "timing": "11:00 AM",
      "significance": "Troponin elevated at presentation, confirming acute MI"
    },
    {
      "event": "Cardiac catheterization",
      "timing": "14:30 PM",
      "significance": "Revascularization within guideline-recommended timeframe"
    }
  ]
}
```

#### 8. Severity Assessment
- **Severity score** (SOFA, CURB-65, CIWA, PHQ-9, etc.)
- **Staging** (AKI stage, CKD stage, HF class, etc.)
- **Complications** (what went wrong)
- **Comorbidities** (coexisting conditions)
- **Functional status** (impact on patient's life)

---

## CLASSIFICATION SCHEMA (Task 2)

**File**: `schemas/task2_classification_schema_v2.json`
**Required Fields**: 6 top-level objects
**Output**: Probability for ALL 20 diagnoses + detailed reasoning

### Schema Structure:

```json
{
  "required": [
    "patient_info",              // Demographics
    "clinical_data_extraction",  // Systematic data pull
    "clinical_pattern_analysis", // Pattern recognition
    "multiclass_prediction",     // PROBABILITIES FOR ALL 20 DIAGNOSES
    "top_diagnosis",             // Single best prediction
    "clinical_reasoning"         // Detailed reasoning documentation
  ]
}
```

### Key Classification Components:

#### 1. Clinical Data Extraction
Systematic extraction of:
- Chief complaint
- Key symptoms (array)
- Vital signs summary
- **Critical lab findings** (array of abnormal labs)
- Imaging findings (array)
- Medications (array - used as diagnostic clues)
- Relevant history (array)

#### 2. Clinical Pattern Analysis
- **Primary organ systems involved** (cardiovascular, respiratory, renal, etc.)
- **Disease time course**: ACUTE | SUBACUTE | CHRONIC | ACUTE_ON_CHRONIC
- **Severity assessment**: MILD | MODERATE | SEVERE | CRITICAL
- **Key pathophysiologic features**

#### 3. Multiclass Prediction (MOST IMPORTANT)
**Required**: Array of exactly 20 predictions, one for EACH diagnosis

For each of the 20 diagnoses:
```json
{
  "diagnosis_name": "Non-ST elevation myocardial infarction",
  "icd_code": "I214",
  "icd_version": 10,
  "probability": 0.65,  // MUST be 0.0-1.0
  "confidence": "HIGH", // VERY_HIGH | HIGH | MODERATE | LOW | VERY_LOW
  "supporting_evidence": [
    "Troponin elevation 2.5 ng/mL (>50x ULN)",
    "ST depressions on ECG",
    "Classic anginal chest pain"
  ],
  "contradicting_evidence": [
    "No Q waves on ECG (not STEMI)"
  ],
  "clinical_reasoning": "High probability for NSTEMI given troponin elevation, ischemic symptoms, and ECG changes meeting Fourth Universal Definition criteria."
}
```

**CRITICAL REQUIREMENT**:
```json
{
  "probability_sum_check": 1.0  // Sum of all 20 probabilities MUST equal 1.0
}
```

#### 4. Top Diagnosis
Single most likely diagnosis with:
- Predicted diagnosis name
- ICD code
- **Probability** (highest probability from multiclass)
- **Detailed reasoning** (comprehensive explanation)
- **Key discriminating features** (what makes this #1 vs #2)
- **Diagnostic certainty**: DEFINITIVE | HIGHLY_PROBABLE | PROBABLE | POSSIBLE | UNCERTAIN

#### 5. Top 5 Differential
Ranked list of top 5 diagnoses:
```json
{
  "rank": 1,
  "diagnosis": "NSTEMI",
  "icd_code": "I214",
  "probability": 0.65,
  "key_supporting_evidence": [...],
  "why_not_higher": null  // For rank 1, this is null
}
```

For ranks 2-5:
```json
{
  "rank": 2,
  "diagnosis": "Unstable angina",
  "icd_code": "I200",
  "probability": 0.20,
  "key_supporting_evidence": [
    "Ischemic chest pain",
    "ECG changes"
  ],
  "why_not_higher": "Troponin elevation definitively rules out unstable angina per guidelines; this is MI not UA"
}
```

#### 6. Clinical Reasoning
Documentation of thought process:
- **Reasoning approach**: PATTERN_RECOGNITION | BAYESIAN_REASONING | CRITERIA_BASED | EXCLUSION_APPROACH | PROBABILISTIC
- **Key decision points** (questions asked, evidence reviewed, conclusions)
- **Diagnostic challenges** (difficulties encountered and how addressed)
- **Data quality assessment**: EXCELLENT | GOOD | FAIR | POOR
- **Uncertainty factors** (what creates uncertainty in prediction)

---

## EXTRAS - CLINICAL KNOWLEDGE

**Location**: `mimic-iv/extras/` (16 JSON files)
**Purpose**: Provide LLM with expert clinical knowledge during annotation/classification
**Format**: JSON with id, type, content, metadata

### How ClinOrchestra Uses Extras:
1. Reads prompt and identifies keywords (diagnosis names, clinical terms)
2. Matches extras by metadata categories and keywords
3. **Automatically injects** relevant extras into LLM context
4. LLM uses knowledge to apply correct criteria, recognize severity, use proper terminology

### Complete Extras List (16 files):

#### Cardiovascular Extras (6 files):

1. **chest_pain_evaluation.json**
   - Type: diagnostic_criteria
   - Content: Chest pain diagnostic approach, HEART score, RED FLAGS (aortic dissection, PE, pneumothorax)
   - Keywords: chest pain, angina, ischemia

2. **coronary_artery_disease.json**
   - Type: diagnostic_criteria + treatment
   - Content: CAD risk factors, ASCVD risk calculator, stable vs unstable, medical therapy
   - Keywords: CAD, atherosclerosis, ischemic heart disease

3. **nstemi_criteria.json**
   - Type: diagnostic_criteria
   - Content: Fourth Universal Definition of MI, NSTEMI vs STEMI, troponin criteria, TIMI/GRACE scores
   - Keywords: NSTEMI, myocardial infarction, troponin, ACS

4. **atrial_fibrillation.json**
   - Type: diagnostic_criteria + risk_stratification
   - Content: AFib types (paroxysmal/persistent/permanent), CHA2DS2-VASc, HAS-BLED, rate vs rhythm control
   - Keywords: atrial fibrillation, AFib, anticoagulation

5. **hypertensive_heart_ckd.json**
   - Type: diagnostic_criteria + pathophysiology
   - Content: Combined HHD + CKD pathology, LVH, proteinuria, BP targets
   - Keywords: hypertensive heart disease, CKD, LVH

6. **heart_failure_classification.json**
   - Type: diagnostic_criteria
   - Content: Acute vs chronic, HFrEF vs HFpEF, NYHA class, ACC/AHA stages
   - Keywords: heart failure, HFrEF, HFpEF, cardiomyopathy

#### Infectious Disease Extras (4 files):

7. **sepsis_diagnostic_criteria.json**
   - Type: diagnostic_criteria + severity_scoring
   - Content: Sepsis-3 definition, qSOFA, SOFA score, septic shock, lactate, Hour-1 bundle
   - Keywords: sepsis, septic shock, SOFA, infection

8. **pneumonia_criteria.json**
   - Type: diagnostic_criteria + treatment
   - Content: CAP vs HAP, CURB-65, PSI, infiltrate on CXR, empiric antibiotics
   - Keywords: pneumonia, CAP, HAP, infiltrate

9. **uti_criteria.json**
   - Type: diagnostic_criteria
   - Content: UTI types (cystitis, pyelonephritis, complicated), UA findings, empiric antibiotics
   - Keywords: UTI, urinary tract infection, dysuria, pyelonephritis

10. **chemotherapy_encounter.json**
    - Type: context_explanation
    - Content: NOT A DISEASE - encounter code, underlying malignancy, chemotherapy complications
    - Keywords: chemotherapy, Z5111, V5811, cancer

#### Renal Extras (2 files):

11. **aki_detailed.json**
    - Type: diagnostic_criteria + staging
    - Content: KDIGO staging (Cr and UOP criteria), prerenal/intrinsic/postrenal, FENa, nephrotoxins
    - Keywords: acute kidney injury, AKI, creatinine, oliguria

12. **aki_staging.json**
    - Type: staging_criteria
    - Content: KDIGO Stage 1/2/3 definitions with exact Cr and UOP cutoffs
    - Keywords: AKI staging, KDIGO

#### Psychiatric/Substance Extras (2 files):

13. **depression_criteria.json**
    - Type: diagnostic_criteria + screening
    - Content: DSM-5 criteria (5+ symptoms for 2+ weeks), PHQ-9, suicide assessment, treatment
    - Keywords: depression, MDD, depressive disorder, PHQ-9

14. **alcohol_use_disorder.json**
    - Type: diagnostic_criteria + management
    - Content: DSM-5 AUD criteria, withdrawal symptoms (tremor, tachycardia, hallucinations), CIWA, DTs, benzodiazepines
    - Keywords: alcohol, alcohol abuse, withdrawal, CIWA, delirium tremens

#### General Clinical Extras (2 files):

15. **respiratory_failure_types.json**
    - Type: diagnostic_criteria
    - Content: Type 1 (hypoxemic) vs Type 2 (hypercapnic), ABG interpretation, mechanical ventilation
    - Keywords: respiratory failure, hypoxia, hypercapnia, ABG

16. **clinical_annotation_approach.json**
    - Type: methodology
    - Content: Systematic approach to clinical annotation, evidence hierarchy, consultant mindset
    - Keywords: annotation, clinical reasoning, evidence

### Example Extra Structure:
```json
{
  "id": "extra_sepsis_001",
  "type": "diagnostic_criteria",
  "content": "Sepsis-3 Definition (2016): Sepsis is life-threatening organ dysfunction caused by a dysregulated host response to infection. Clinically defined as suspected or documented infection AND acute increase in SOFA score ≥2 points. Septic shock is sepsis with persistent hypotension requiring vasopressors to maintain MAP ≥65 AND lactate >2 mmol/L despite adequate fluid resuscitation. qSOFA (quick SOFA) for bedside screening: 1 point each for RR≥22, altered mentation, SBP≤100. qSOFA ≥2 suggests higher mortality risk. SOFA score components: PaO2/FiO2, platelets, bilirubin, MAP/vasopressors, GCS, creatinine/urine output. Full SOFA 0-24 points.",
  "metadata": {
    "category": "infectious_disease",
    "diagnosis": "sepsis",
    "priority": "CRITICAL",
    "icd10_codes": ["A41.9", "A41.89"],
    "icd9_codes": ["038.9"],
    "keywords": ["sepsis", "septic shock", "SOFA", "qSOFA", "infection", "organ dysfunction"]
  }
}
```

---

## PATTERNS - TEXT STANDARDIZATION

**Location**: `mimic-iv/patterns/` (12 JSON files)
**Purpose**: Clean and standardize clinical text BEFORE LLM sees it
**Format**: JSON with name, pattern (regex), replacement, description

### How ClinOrchestra Uses Patterns:
1. Runs regex patterns on input text **sequentially**
2. Replaces messy variations with standard format
3. Cleaned text sent to LLM
4. Makes extraction more reliable and consistent

### Complete Patterns List (12 files):

#### Vital Signs Patterns (5 files):

1. **vital_signs_bp.json**
   - Pattern: `(?i)(?:bp|blood\s+pressure)[:\s]*([0-9]{2,3})\s*/\s*([0-9]{2,3})`
   - Replacement: `BP: \1/\2`
   - Example: "BP 120 / 80" → "BP: 120/80"

2. **vital_signs_hr.json**
   - Pattern: `(?i)(?:hr|heart\s+rate)[:\s]*([0-9]{2,3})`
   - Replacement: `HR: \1 bpm`
   - Example: "heart rate 85" → "HR: 85 bpm"

3. **vital_signs_rr.json**
   - Pattern: `(?i)(?:rr|respiratory\s+rate)[:\s]*([0-9]{1,2})`
   - Replacement: `RR: \1 /min`
   - Example: "RR: 22" → "RR: 22 /min"

4. **vital_signs_temp.json**
   - Pattern: `(?i)(?:temp|temperature)[:\s]*([0-9]{2,3}\.?[0-9]*)`
   - Replacement: `Temp: \1°F`
   - Example: "temperature 101.5" → "Temp: 101.5°F"

5. **vital_signs_spo2.json**
   - Pattern: `(?i)(?:spo2|o2\s+sat|oxygen\s+sat)[:\s]*([0-9]{2,3})%?`
   - Replacement: `SpO2: \1%`
   - Example: "O2 sat 95" → "SpO2: 95%"

#### Lab Patterns (4 files):

6. **lab_wbc.json**
   - Pattern: `(?i)(?:wbc|white\s+blood\s+cell)[:\s]*([0-9]+\.?[0-9]*)`
   - Replacement: `WBC: \1 K/uL`
   - Example: "WBC 15.2" → "WBC: 15.2 K/uL"

7. **lab_creatinine.json**
   - Pattern: `(?i)(?:creatinine|cr)[:\s]*([0-9]+\.?[0-9]*)`
   - Replacement: `Creatinine: \1 mg/dL`
   - Example: "Cr 2.8" → "Creatinine: 2.8 mg/dL"

8. **lab_lactate.json**
   - Pattern: `(?i)lactate[:\s]*([0-9]+\.?[0-9]*)`
   - Replacement: `Lactate: \1 mmol/L`
   - Example: "lactate 4.2" → "Lactate: 4.2 mmol/L"

9. **lab_bun.json**
   - Pattern: `(?i)bun[:\s]*([0-9]+\.?[0-9]*)`
   - Replacement: `BUN: \1 mg/dL`
   - Example: "BUN 45" → "BUN: 45 mg/dL"

#### Cardiac Markers (2 files):

10. **cardiac_troponin.json**
    - Pattern: `(?i)troponin[:\s]*([0-9]+\.?[0-9]*)`
    - Replacement: `Troponin: \1 ng/mL`
    - Example: "troponin 2.5" → "Troponin: 2.5 ng/mL"

11. **cardiac_bnp.json**
    - Pattern: `(?i)(?:bnp|b-type\s+natriuretic)[:\s]*([0-9]+)`
    - Replacement: `BNP: \1 pg/mL`
    - Example: "BNP 850" → "BNP: 850 pg/mL"

#### Clinical Scores (1 file):

12. **gcs_score.json**
    - Pattern: `(?i)(?:gcs|glasgow\s+coma)[:\s]*([0-9]{1,2})`
    - Replacement: `GCS: \1/15`
    - Example: "GCS 12" → "GCS: 12/15"

### Example Pattern Structure:
```json
{
  "name": "extract_blood_pressure",
  "pattern": "(?i)(?:bp|blood\\s+pressure)[:\\s]*([0-9]{2,3})\\s*/\\s*([0-9]{2,3})",
  "replacement": "BP: \\1/\\2",
  "description": "Extract and standardize blood pressure readings. Handles variations like 'BP 120/80', 'blood pressure: 120 / 80', etc.",
  "enabled": true
}
```

---

## FUNCTIONS - CLINICAL CALCULATIONS

**Location**: `mimic-iv/functions/` (7 JSON files)
**Purpose**: Enable LLM to perform clinical calculations
**Format**: JSON with name, code (Python function), parameters, returns, signature

### How ClinOrchestra Uses Functions:
1. LLM can **call functions** in its output
2. Functions execute with provided parameters
3. Results incorporated into final annotation/prediction
4. Enables objective severity scoring

### Complete Functions List (7 files):

#### Severity Scores (5 functions):

1. **calculate_sofa_score.json**
   - Input: pao2_fio2_ratio, platelets, bilirubin, map_value, on_vasopressor, gcs, creatinine, urine_output
   - Output: SOFA score (0-24)
   - Interpretation: 0-6 low mortality, 7-9 moderate, 10-12 high, >13 very high
   - **Use**: Sepsis severity, organ dysfunction

2. **calculate_curb65.json**
   - Input: confusion, bun, respiratory_rate, sbp, dbp, age
   - Output: CURB-65 score (0-5)
   - Interpretation: 0-1 outpatient, 2 inpatient, 3-5 ICU
   - **Use**: Pneumonia severity, site-of-care decisions

3. **calculate_ciwa.json**
   - Input: nausea, tremor, sweating, anxiety, agitation, tactile_disturbances, auditory_disturbances, visual_disturbances, headache, orientation
   - Output: CIWA-Ar score (0-67)
   - Interpretation: <8 minimal, 8-15 mild-moderate, >15 severe (needs benzos)
   - **Use**: Alcohol withdrawal severity

4. **calculate_phq9.json**
   - Input: little_interest, feeling_down, sleep_problems, tired, appetite, feeling_bad, concentration, moving_slow, thoughts_harm
   - Output: PHQ-9 score (0-27)
   - Interpretation: 0-4 minimal, 5-9 mild, 10-14 moderate, 15-19 moderately severe, 20-27 severe
   - **Use**: Depression severity

5. **calculate_chadsvasc.json**
   - Input: chf, hypertension, age, diabetes, stroke_tia_thromboembolism, vascular_disease, female
   - Output: CHA2DS2-VASc score (0-9)
   - Interpretation: 0 no anticoagulation, 1 consider, ≥2 anticoagulate
   - **Use**: AFib stroke risk

#### Clinical Calculations (2 functions):

6. **calculate_map.json**
   - Input: sbp, dbp
   - Output: Mean Arterial Pressure (mmHg)
   - Formula: MAP = DBP + (SBP - DBP)/3
   - **Use**: Septic shock definition (MAP <65), organ perfusion

7. **calculate_creatinine_clearance.json**
   - Input: age, weight_kg, creatinine, gender
   - Output: Creatinine clearance (mL/min) via Cockcroft-Gault
   - **Use**: Renal function assessment, drug dosing

### Example Function Structure:
```json
{
  "name": "calculate_sofa_score",
  "code": "\ndef calculate_sofa_score(pao2_fio2_ratio=None, platelets=None, bilirubin=None, map_value=None, on_vasopressor=False, gcs=None, creatinine=None, urine_output=None):\n    score = 0\n    # Respiration\n    if pao2_fio2_ratio is not None:\n        if pao2_fio2_ratio < 100: score += 4\n        elif pao2_fio2_ratio < 200: score += 3\n        elif pao2_fio2_ratio < 300: score += 2\n        elif pao2_fio2_ratio < 400: score += 1\n    # ... [additional organ systems]\n    return score\n",
  "description": "Calculate SOFA score for sepsis/organ dysfunction severity (0-24). Higher scores indicate more severe organ dysfunction.",
  "parameters": {
    "pao2_fio2_ratio": {"type": "number", "description": "PaO2/FiO2 ratio"},
    "platelets": {"type": "number", "description": "Platelet count (x10^3/uL)"},
    ...
  },
  "returns": "SOFA score (0-24)",
  "signature": "(pao2_fio2_ratio, platelets, bilirubin, map_value, on_vasopressor, gcs, creatinine, urine_output)"
}
```

---

## RAG RESOURCES

**File**: `mimic-iv/RAG_RESOURCES_COMPREHENSIVE.md`
**Total Resources**: 50+ downloadable PDFs
**Coverage**: All 20 diagnoses + clinical scores + validation studies

### Major Guidelines Included:

#### Cardiovascular (8 guidelines):
- AHA/ACC Chest Pain 2021
- ACC/AHA Chronic Coronary Disease 2023
- ESC Chronic Coronary Syndromes 2019
- ACC/AHA Acute Coronary Syndromes 2023
- Fourth Universal Definition of MI 2018
- ACC/AHA Atrial Fibrillation 2023
- AHA/ACC Heart Failure 2022
- KDIGO CKD 2024

#### Infectious Disease (5 guidelines):
- Surviving Sepsis Campaign 2021
- Sepsis-3 Definitions JAMA 2016
- IDSA/ATS Pneumonia 2019
- IDSA UTI 2011
- HAP/VAP Guidelines

#### Renal (2 guidelines):
- KDIGO AKI 2012
- NICE AKI Guidelines

#### Psychiatric (3 guidelines):
- APA Major Depressive Disorder 2010
- DSM-5-TR Criteria
- ASAM Alcohol Withdrawal 2020

#### Oncology (3 guidelines):
- NCCN Antiemesis 2025
- ASCO Antiemetic 2020
- MASCC/ESMO 2023

#### Critical Care (3 guidelines):
- ATS ARDS Guidelines
- Noninvasive Ventilation Guidelines
- Respiratory Failure Reviews

#### Validation Studies (10+ papers):
- PHQ-9 validation (Kroenke 2001)
- TIMI score validation
- GRACE score validation
- SOFA/qSOFA validation
- CURB-65 validation
- Clinical score comparisons

### Access Summary:
- **Free PDFs**: ~90% of resources
- **Free registration required**: NCCN guidelines
- **Institutional access may help**: Some journal articles
- **Alternative sources**: ResearchGate, PubMed Central

---

## TOP 20 DIAGNOSES DISTRIBUTION

**Total Cases: 76,594**

### By Category:

#### CARDIOVASCULAR (10 diagnoses) - 29,417 cases (38.4%)
1. Chest pain, unspecified (ICD-9: 78650) - 7,297 cases
2. Other chest pain (ICD-9: 78659) - 5,212 cases
3. Chest pain, unspecified (ICD-10: R079) - 2,906 cases
4. Coronary atherosclerosis (ICD-9: 41401) - 5,751 cases
5. NSTEMI (ICD-10: I214) - 3,265 cases
6. Subendocardial infarction (ICD-9: 41071) - 2,873 cases
7. Atrial fibrillation (ICD-9: 42731) - 3,090 cases
8. Hypertensive heart + CKD (ICD-10: I130) - 3,313 cases

#### INFECTIOUS (4 diagnoses) - 14,932 cases (19.5%)
9. Sepsis, unspecified (ICD-10: A419) - 5,095 cases
10. Unspecified septicemia (ICD-9: 0389) - 3,144 cases
11. Pneumonia, organism unspecified (ICD-9: 486) - 3,726 cases
12. UTI, site not specified (ICD-9: 5990) - 2,967 cases

#### RENAL (2 diagnoses) - 6,185 cases (8.1%)
13. Acute kidney failure (ICD-9: 5849) - 3,532 cases
14. Acute kidney failure (ICD-10: N179) - 2,653 cases

#### PSYCHIATRIC/SUBSTANCE (4 diagnoses) - 15,240 cases (19.9%)
15. Major depressive disorder (ICD-10: F329) - 3,703 cases
16. Depressive disorder (ICD-9: 311) - 3,499 cases
17. Alcohol abuse (ICD-9: 30500) - 4,591 cases
18. Alcohol abuse with intoxication (ICD-10: F10129) - 3,447 cases

#### ONCOLOGY (2 diagnoses) - 6,530 cases (8.5%)
19. Encounter for chemotherapy (ICD-10: Z5111) - 3,557 cases
20. Encounter for chemotherapy (ICD-9: V5811) - 2,973 cases

---

## FILE INVENTORY

### Complete File Listing:

```
mimic-iv/
│
├── COMPLETE_SYSTEM_OVERVIEW.md          ← YOU ARE HERE
├── SETUP_COMPLETE.md                    ← Quick start guide
├── CLINICAL_RESOURCES.md                ← Original guideline links
├── RAG_RESOURCES_COMPREHENSIVE.md       ← NEW! Complete RAG resource list with PDFs
├── USAGE_GUIDE.md                       ← How to use everything
├── README.md                            ← Full documentation
│
├── extras/                              ← 16 JSON files for ClinOrchestra Extras Tab
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
├── patterns/                            ← 12 JSON files for ClinOrchestra Patterns Tab
│   ├── vital_signs_bp.json
│   ├── vital_signs_hr.json
│   ├── vital_signs_rr.json
│   ├── vital_signs_temp.json
│   ├── vital_signs_spo2.json
│   ├── lab_wbc.json
│   ├── lab_creatinine.json
│   ├── lab_lactate.json
│   ├── lab_bun.json
│   ├── cardiac_troponin.json
│   ├── cardiac_bnp.json
│   └── gcs_score.json
│
├── functions/                           ← 7 JSON files for ClinOrchestra Functions Tab
│   ├── calculate_sofa_score.json
│   ├── calculate_curb65.json
│   ├── calculate_ciwa.json
│   ├── calculate_phq9.json
│   ├── calculate_chadsvasc.json
│   ├── calculate_map.json
│   └── calculate_creatinine_clearance.json
│
├── prompts/                             ← Paste into ClinOrchestra Prompt Tab
│   ├── task1_annotation_prompt.txt      ← Clinical Consultant Annotation
│   └── task2_classification_prompt_FINAL.txt  ← Multiclass Classification (has your top 20!)
│
├── schemas/                             ← Upload to ClinOrchestra Processing Tab
│   ├── task1_annotation_schema.json     ← Annotation output structure
│   └── task2_classification_schema_v2.json  ← Classification output structure
│
└── scripts/                             ← Python utilities
    ├── get_top_diagnoses_simple.py      ← Already ran ✓ (got your top 20)
    ├── extract_dataset.py               ← Run next to create datasets
    └── evaluate_classification.py       ← Run after Task 2 to get metrics
```

**Total Files Created**: 45+ files
- 16 Extras
- 12 Patterns
- 7 Functions
- 2 Prompts
- 2 Schemas
- 6 Documentation files
- 3 Python scripts

---

## NEXT STEPS

### Immediate Actions (In Order):

#### 1. ✅ DONE: Extract Top 20 Diagnoses
You've already run `get_top_diagnoses_simple.py` and confirmed 76,594 cases.

#### 2. ⏭️ Create Your Datasets
Run the extraction script to create annotation and classification datasets:

```bash
python mimic-iv/scripts/extract_dataset.py
```

This will create:
- `annotation_dataset.csv` (for Task 1)
- `classification_dataset.csv` (for Task 2)

Each row contains: patient_id, admission_id, demographics, clinical_text, diagnosis

#### 3. ⏭️ Download RAG Resources
Use `RAG_RESOURCES_COMPREHENSIVE.md` to download guidelines for embedding:

**Priority Downloads** (do these first):
- Surviving Sepsis 2021
- ACC/AHA ACS 2023
- IDSA/ATS Pneumonia 2019
- KDIGO AKI 2012
- KDIGO CKD 2024
- ACC/AHA AFib 2023
- APA Depression 2010
- ASAM Alcohol 2020

**Total time estimate**: 30-60 minutes to download all 50+ PDFs

#### 4. ⏭️ Set Up ClinOrchestra

**A. Load Extras (16 files)**
- Open ClinOrchestra UI → Extras Tab
- Upload all 16 JSON files from `mimic-iv/extras/`
- These provide clinical knowledge to the LLM

**B. Load Patterns (12 files)**
- Go to Patterns Tab
- Upload all 12 JSON files from `mimic-iv/patterns/`
- These standardize clinical text

**C. Load Functions (7 files)**
- Go to Functions Tab
- Upload all 7 JSON files from `mimic-iv/functions/`
- These enable clinical calculations

**D. Embed RAG Resources**
- Use ClinOrchestra RAG system
- Upload downloaded guideline PDFs
- Tag with diagnosis ICD codes and categories
- Enable semantic search over guidelines

#### 5. ⏭️ Run Task 1 (Annotation)

**A. Configure Task**
1. **Prompt Tab**: Copy `prompts/task1_annotation_prompt.txt`
2. **Data Tab**: Upload `annotation_dataset.csv`
3. **Processing Tab**: Upload `schemas/task1_annotation_schema.json`
4. **Settings**:
   - Batch size: 10-50 records
   - Model: GPT-4 or equivalent
   - Temperature: 0.1 (low for consistency)

**B. Start Processing**
- Start with 100-case pilot to test quality
- Review outputs for completeness
- Adjust prompt if needed
- Run full dataset (expect 2-5 min per case)

**C. Expected Output**
- Comprehensive evidence annotation for each case
- 200-500 clinical data points per case
- Ready for Clinical Consultant AI training

#### 6. ⏭️ Run Task 2 (Classification)

**A. Configure Task**
1. **Prompt Tab**: Copy `prompts/task2_classification_prompt_FINAL.txt`
   - This prompt ALREADY has your actual top 20 diagnoses with case counts!
2. **Data Tab**: Upload `classification_dataset.csv`
3. **Processing Tab**: Upload `schemas/task2_classification_schema_v2.json`
4. **Settings**:
   - Batch size: 10-50 records
   - Model: GPT-4 or equivalent
   - Temperature: 0.2 (slightly higher for probability calibration)

**B. Start Processing**
- Start with 100-case pilot
- Check that probabilities sum to 1.0
- Review probability calibration
- Run full dataset (expect 3-7 min per case)

**C. Expected Output**
- Probability for ALL 20 diagnoses per case
- Top diagnosis with reasoning
- Top 5 differential
- Clinical reasoning documentation

#### 7. ⏭️ Evaluate Results

**A. Run Evaluation Script**
```bash
python mimic-iv/scripts/evaluate_classification.py
```

**B. Metrics Produced**
- **Top-k Accuracy**: Top-1, Top-3, Top-5
- **Cross-Entropy**: Measure of probability calibration
- **Brier Score**: Squared error of probabilities
- **Per-Diagnosis Metrics**: Precision, recall, F1 for each of 20 diagnoses
- **Confusion Matrix**: See common misclassifications

**C. Target Performance**
- Top-1 Accuracy: >60-70%
- Top-5 Accuracy: >85-90%
- Cross-Entropy: <0.7
- Brier Score: <0.4

#### 8. ⏭️ Iterate and Improve

Based on evaluation results:

**A. If Accuracy Is Low**
- Review misclassified cases
- Add more diagnosis-specific extras
- Enhance prompt with specific guidance
- Add more RAG documents for weak diagnoses

**B. If Probabilities Are Poorly Calibrated**
- Adjust temperature
- Add probability calibration examples to prompt
- Review cases where probabilities don't sum to 1.0

**C. If Specific Diagnoses Perform Poorly**
- Add more extras for those diagnoses
- Enhance RAG with more guidelines
- Add diagnosis-specific patterns
- Review prompt guidance for those conditions

#### 9. ⏭️ Train Your Clinical Consultant AI

Once you have high-quality annotated data:

**A. Prepare Training Data**
- Format annotations as instruction-response pairs
- Create prompts from clinical text
- Use annotations as target outputs

**B. Fine-Tuning Approaches**
- **Option 1**: Fine-tune large model (GPT-3.5, GPT-4, Claude, Llama)
- **Option 2**: Train smaller specialized model
- **Option 3**: Use annotations as few-shot examples

**C. Evaluation**
- Hold out 10-20% of cases for testing
- Measure: completeness, accuracy, clinical validity
- Have physicians review sample outputs

---

## COST ESTIMATES

### Using GPT-4 (adjust for your model):

**Task 1 (Annotation)**:
- Input: ~2,000 tokens (clinical text + prompt + extras)
- Output: ~3,000 tokens (comprehensive annotation)
- Cost: $0.50-1.00 per case
- **For 76,594 cases**: $38,000-76,000

**Task 2 (Classification)**:
- Input: ~2,500 tokens (clinical text + prompt + extras)
- Output: ~4,000 tokens (all 20 probabilities + reasoning)
- Cost: $0.70-1.50 per case
- **For 76,594 cases**: $54,000-115,000

**Total Estimated Cost**: $92,000-191,000 for full dataset

### Cost Optimization Strategies:

1. **Start Small**: Process 1,000-5,000 cases initially
2. **Use Cheaper Models**: Try GPT-3.5-turbo, Claude Sonnet, or Llama
3. **Batch Processing**: Use batch API for 50% discount
4. **Subset by Diagnosis**: Process only certain high-value diagnoses
5. **Active Learning**: Start with diverse sample, add more as needed

**Recommended Initial Budget**:
- 1,000 cases × $1.50 average = $1,500
- This gives you substantial dataset to evaluate approach

---

## QUALITY ASSURANCE

### For Task 1 (Annotation):

**Check for**:
- All 13 required sections completed
- Evidence quotes actually from the record (not fabricated)
- Evidence quality ratings appropriate
- Clinical reasoning makes sense
- Diagnostic certainty justified
- No hallucinated findings

**Validation Method**:
- Physician review of 50-100 random cases
- Check inter-rater reliability
- Verify evidence quotes are accurate
- Confirm clinical reasoning is sound

### For Task 2 (Classification):

**Check for**:
- All 20 probabilities provided
- Probabilities sum to 1.0 (within 0.01)
- Top diagnosis matches highest probability
- Top 5 ranked correctly
- Clinical reasoning supports probabilities
- Contradicting evidence acknowledged

**Validation Method**:
- Calculate metrics on held-out test set
- Review high-confidence errors
- Check probability calibration curves
- Analyze per-diagnosis performance

---

## TROUBLESHOOTING

### Common Issues:

**Issue**: LLM fabricates evidence not in record
- **Solution**: Emphasize evidence quotes, add "ONLY use information from the provided clinical record" to prompt

**Issue**: Probabilities don't sum to 1.0
- **Solution**: Add explicit validation step in prompt, use schema constraint

**Issue**: Annotations are inconsistent across cases
- **Solution**: Lower temperature, add more examples, use stronger extras

**Issue**: Specific diagnosis always gets low probability
- **Solution**: Add more extras for that diagnosis, review RAG content, check prompt guidance

**Issue**: Processing is too slow
- **Solution**: Reduce batch size, use faster model, optimize prompt length

**Issue**: Costs are too high
- **Solution**: Start with subset, use cheaper model, try batch API

---

## SUPPORT RESOURCES

**Documentation**:
- SETUP_COMPLETE.md - Quick start
- USAGE_GUIDE.md - Detailed how-to
- CLINICAL_RESOURCES.md - Original guideline links
- RAG_RESOURCES_COMPREHENSIVE.md - Complete RAG resource list

**Clinical Guidelines**:
- All linked in RAG_RESOURCES_COMPREHENSIVE.md
- 50+ downloadable PDFs
- Official professional society guidelines

**Technical Support**:
- ClinOrchestra documentation
- JSON schema validation tools
- Python script debugging

---

## SUMMARY

You now have a **complete, production-ready system** for:

1. ✅ Annotating 76,594 MIMIC-IV cases with comprehensive clinical evidence
2. ✅ Classifying cases into 20 primary diagnoses with probability scores
3. ✅ Training a Clinical Consultant AI with expert-level annotations
4. ✅ Evaluating classification performance with standard metrics

**What's Ready**:
- 16 Extras (clinical knowledge)
- 12 Patterns (text standardization)
- 7 Functions (clinical calculations)
- 2 Prompts (annotation + classification)
- 2 Schemas (output structures)
- 50+ RAG resources (guidelines for embedding)
- 3 Python scripts (extraction, evaluation)
- 6 Documentation files

**Next Step**: Download RAG resources and start processing your first 100 cases!

---

**Everything is ready! 🚀**

Upload extras/patterns/functions to ClinOrchestra and start building your Clinical Consultant AI training dataset!
