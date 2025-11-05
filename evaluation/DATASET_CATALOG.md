# clinAnnotate Dataset Catalog & Configuration Guide

Complete reference for public clinical datasets with task configuration examples, functions, extras, and patterns.

---

## Table of Contents

1. [i2b2/n2c2 Clinical NLP Challenges](#i2b2n2c2-clinical-nlp-challenges)
2. [MIMIC-III/MIMIC-IV ICU Database](#mimic-iiimimic-iv-icu-database)
3. [PubMed Central Open Access](#pubmed-central-open-access)
4. [CASI - Clinical Abbreviation Sense Inventory](#casi---clinical-abbreviation-sense-inventory)
5. [MedNLI - Medical Natural Language Inference](#mednli---medical-natural-language-inference)
6. [MTSamples - Medical Transcription Samples](#mtsamples---medical-transcription-samples)
7. [Quick Reference: Functions, Extras, Patterns by Task](#quick-reference-functions-extras-patterns-by-task)

---

## i2b2/n2c2 Clinical NLP Challenges

### Access Information
- **Website**: https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
- **Cost**: FREE
- **Requirements**: Data Use Agreement (DUA), 1-2 days approval
- **Best for**: Classification tasks, concept extraction, temporal relations
- **Data type**: De-identified discharge summaries, clinical notes

### Available Datasets

#### 1. i2b2 2008 Obesity Challenge ⭐ **RECOMMENDED FOR STARTING**

**Task**: Classify obesity and 15 comorbidities (binary classification)

**Dataset Details**:
- ~1,000 discharge summaries
- 16 binary classification tasks:
  - Obesity (Present/Absent/Questionable)
  - Asthma, CAD, CHF, Depression, Diabetes, GERD, Gallstones
  - Gout, Hypercholesterolemia, Hypertension, Hypertriglyceridemia
  - OA (Osteoarthritis), OSA, PVD, Venous Insufficiency

**Benchmark Performance**:
- Rule-based systems: F1 ~0.70-0.75
- ML systems: F1 ~0.85-0.92
- Top systems: F1 ~0.93-0.95

**clinAnnotate Configuration**:

```json
{
  "task_name": "i2b2_2008_Obesity_Classification",
  "description": "Binary classification of obesity status from discharge summaries",
  "prompt": "You are a clinical expert extracting obesity and metabolic comorbidity information from discharge summaries.\n\nYour task is to classify whether OBESITY is:\n- Present: Explicitly documented or BMI ≥30\n- Absent: Explicitly stated as absent or BMI <30\n- Questionable: Uncertain or conflicting information\n\nLook for:\n1. Explicit mentions: 'obese', 'obesity'\n2. BMI values: ≥30 = Present, <30 = Absent\n3. Qualifying terms: 'morbidly obese', 'overweight but not obese'\n4. Historical vs current status\n\nUse available functions for BMI calculation if height/weight provided.",
  "json_schema": {
    "obesity_status": {
      "type": "string",
      "description": "Classification: Present, Absent, or Questionable",
      "required": true
    },
    "evidence": {
      "type": "string",
      "description": "Supporting evidence from text (quote relevant sentences)",
      "required": true
    },
    "bmi_value": {
      "type": "number",
      "description": "BMI if calculable or documented",
      "required": false
    },
    "reasoning": {
      "type": "string",
      "description": "Clinical reasoning for classification",
      "required": true
    }
  },
  "enable_rag": true,
  "rag_query_fields": ["obesity_status", "reasoning"],
  "enable_pattern_normalization": true
}
```

**Recommended Functions**:
```bash
# BMI calculation
functions/calculate_bmi.py
functions/interpret_bmi_category.py

# For comorbidities
functions/calculate_creatinine_clearance.py  # For CKD assessment
functions/calculate_cvd_risk.py              # For CAD/CHF risk
```

**Recommended Extras**:
```bash
# ICD-10 coding
extras/icd10_obesity_codes.json
extras/icd10_diabetes_codes.json
extras/icd10_cardiac_codes.json

# Clinical criteria
extras/metabolic_syndrome_criteria.json
extras/obesity_classification_who.json
```

**Recommended Patterns**:
```bash
patterns/extract_bmi_values.json
patterns/obesity_synonyms.json
patterns/negation_detection.json  # "no evidence of obesity"
patterns/temporal_qualifiers.json  # "history of obesity"
```

**RAG Documents to Upload**:
1. WHO obesity classification guidelines
2. ICD-10 diagnostic criteria for obesity
3. Clinical documentation patterns for obesity

---

#### 2. i2b2 2018 Cohort Selection

**Task**: Determine patient eligibility for clinical trials (13 selection criteria)

**Dataset Details**:
- Cohort selection for clinical trials
- 13 criteria: Abdominal pain, Advanced CAD, Alcohol abuse, ASP for MI, Creatinine, Dietary adherence, Drug abuse, HbA1c, Keto acidosis, Major diabetes, Makes decisions, MI 6 months

**clinAnnotate Configuration**:

```json
{
  "task_name": "i2b2_2018_Cohort_Selection",
  "description": "Extract clinical trial eligibility criteria from patient records",
  "prompt": "You are determining patient eligibility for clinical trials based on 13 specific criteria.\n\nFor each criterion, classify as:\n- Met: Patient meets the inclusion/exclusion criterion\n- Not Met: Patient does not meet the criterion\n- Not Mentioned: Insufficient information\n\nCriteria focus on:\n1. Laboratory values (HbA1c, Creatinine)\n2. Medical history (MI within 6 months, Advanced CAD)\n3. Behavioral factors (Drug/alcohol abuse, dietary adherence)\n4. Decision-making capacity\n\nUse functions to calculate relevant values and RAG to retrieve specific trial criteria thresholds.",
  "json_schema": {
    "abdominal_pain": {"type": "string", "description": "Met/Not Met/Not Mentioned", "required": true},
    "advanced_cad": {"type": "string", "description": "Met/Not Met/Not Mentioned", "required": true},
    "hba1c_value": {"type": "number", "description": "Most recent HbA1c if documented", "required": false},
    "creatinine_value": {"type": "number", "description": "Most recent creatinine if documented", "required": false},
    "eligibility_summary": {"type": "string", "description": "Overall eligibility assessment with reasoning", "required": true}
  },
  "enable_rag": true
}
```

**Recommended Functions**:
```bash
functions/calculate_creatinine_clearance.py
functions/calculate_egfr.py
functions/interpret_hba1c.py
functions/calculate_time_since_event.py  # For "MI within 6 months"
```

**Recommended Patterns**:
```bash
patterns/advanced_cad_indicators.json
patterns/substance_abuse_terms.json
patterns/decision_making_capacity_phrases.json
```

---

#### 3. i2b2 2022 Contextualized Medication Events

**Task**: Extract medications with context (dosage, route, frequency, indication)

**Dataset Details**:
- Medication extraction with rich context
- Challenges: Disambiguation, temporal resolution, indication extraction

**clinAnnotate Configuration**:

```json
{
  "task_name": "i2b2_2022_Medication_Extraction",
  "description": "Extract medications with full context including dosage, route, frequency, duration, and indication",
  "json_schema": {
    "medications": {
      "type": "array",
      "description": "List of all medications mentioned",
      "required": true
    },
    "medication_contexts": {
      "type": "string",
      "description": "For each medication: drug name, dosage, route, frequency, duration, indication, status (current/discontinued/historical)",
      "required": true
    }
  }
}
```

**Recommended Patterns**:
```bash
patterns/medication_dosage_extraction.json
patterns/medication_routes.json  # PO, IV, IM, SubQ, etc.
patterns/medication_frequencies.json  # BID, TID, QD, PRN, etc.
```

---

## MIMIC-III/MIMIC-IV ICU Database

### Access Information
- **Website**: https://physionet.org/content/mimiciii/ (MIMIC-III)
- **Website**: https://physionet.org/content/mimiciv/ (MIMIC-IV)
- **Cost**: FREE
- **Requirements**:
  1. Complete CITI "Data or Specimens Only Research" course (~2 hours)
  2. Create PhysioNet account
  3. Request credentialed access (~1 week approval)
  4. Sign DUA
- **Best for**: Serial/temporal measurements, ICU outcomes, risk prediction
- **Data type**: ICU patient data (vitals, labs, medications, notes, diagnoses)

### Dataset Details

**Available Data**:
- ~60,000 ICU admissions (MIMIC-III)
- Clinical notes (discharge summaries, nursing notes, radiology reports)
- Hourly vital signs
- Lab results with timestamps
- Medication administrations
- ICD-9/ICD-10 diagnoses
- Procedures

**Perfect for Testing**:
1. Serial creatinine → AKI detection
2. Serial weights/albumin → Malnutrition tracking
3. Serial vitals → Sepsis screening
4. Temporal medication dosing adjustments

### Example Task 1: AKI Detection from Serial Creatinine

**clinAnnotate Configuration**:

```json
{
  "task_name": "MIMIC_AKI_Detection",
  "description": "Detect and stage acute kidney injury using KDIGO criteria from serial creatinine measurements",
  "prompt": "You are a nephrologist detecting acute kidney injury (AKI) using KDIGO criteria.\n\nKDIGO AKI Staging:\n- Stage 1: Cr increase ≥0.3 mg/dL within 48h OR 1.5-1.9x baseline\n- Stage 2: Cr increase 2.0-2.9x baseline\n- Stage 3: Cr increase ≥3.0x baseline OR Cr ≥4.0 mg/dL OR initiation of RRT\n\nYour task:\n1. Extract ALL creatinine measurements with dates/times\n2. Identify baseline creatinine (lowest in past 7 days or admission)\n3. Calculate ratios and absolute changes\n4. Determine AKI stage if present\n5. Track progression across time points\n\nThis is SERIAL MEASUREMENT task - call functions for EACH time point.",
  "json_schema": {
    "aki_status": {
      "type": "string",
      "description": "No AKI, AKI Stage 1, AKI Stage 2, AKI Stage 3",
      "required": true
    },
    "creatinine_timeline": {
      "type": "string",
      "description": "ALL creatinine values with dates: 'Baseline: 1.0 on 1/15, Peak: 2.1 on 1/18 (2.1x baseline), Current: 1.8 on 1/20'",
      "required": true
    },
    "kdigo_criteria_met": {
      "type": "string",
      "description": "Specific KDIGO criteria met with calculations",
      "required": true
    },
    "aki_trajectory": {
      "type": "string",
      "description": "Progression: improving, worsening, stable, resolved",
      "required": true
    }
  },
  "enable_rag": true,
  "rag_documents": ["Upload KDIGO AKI guidelines PDF"]
}
```

**Recommended Functions**:
```bash
functions/calculate_creatinine_clearance.py
functions/calculate_egfr.py
functions/classify_aki_stage_kdigo.py  # Create this
functions/extract_date_from_text.py
```

**Example Function to Create - classify_aki_stage_kdigo.py**:

```python
def classify_aki_stage_kdigo(baseline_cr: float, current_cr: float,
                             cr_change_48h: float = None,
                             on_rrt: bool = False) -> dict:
    """
    Classify AKI stage per KDIGO criteria

    Args:
        baseline_cr: Baseline creatinine (mg/dL)
        current_cr: Current creatinine (mg/dL)
        cr_change_48h: Change in Cr over 48h (mg/dL)
        on_rrt: Patient on renal replacement therapy

    Returns:
        {
            'aki_stage': 0-3,
            'criteria_met': str,
            'cr_ratio': float,
            'absolute_change': float
        }
    """
    cr_ratio = current_cr / baseline_cr if baseline_cr > 0 else 0
    absolute_change = current_cr - baseline_cr

    # Stage 3
    if on_rrt or current_cr >= 4.0 or cr_ratio >= 3.0:
        return {
            'aki_stage': 3,
            'criteria_met': f"Stage 3: RRT={on_rrt}, Cr={current_cr:.1f}, Ratio={cr_ratio:.2f}x",
            'cr_ratio': cr_ratio,
            'absolute_change': absolute_change
        }

    # Stage 2
    if cr_ratio >= 2.0:
        return {
            'aki_stage': 2,
            'criteria_met': f"Stage 2: Ratio {cr_ratio:.2f}x baseline",
            'cr_ratio': cr_ratio,
            'absolute_change': absolute_change
        }

    # Stage 1
    if cr_ratio >= 1.5 or (cr_change_48h is not None and cr_change_48h >= 0.3):
        criteria = []
        if cr_ratio >= 1.5:
            criteria.append(f"{cr_ratio:.2f}x baseline")
        if cr_change_48h and cr_change_48h >= 0.3:
            criteria.append(f"+{cr_change_48h:.1f} mg/dL in 48h")

        return {
            'aki_stage': 1,
            'criteria_met': f"Stage 1: {', '.join(criteria)}",
            'cr_ratio': cr_ratio,
            'absolute_change': absolute_change
        }

    # No AKI
    return {
        'aki_stage': 0,
        'criteria_met': "No AKI criteria met",
        'cr_ratio': cr_ratio,
        'absolute_change': absolute_change
    }
```

**Recommended Extras**:
```bash
extras/kdigo_aki_stages.json
extras/aki_risk_factors.json
extras/renal_replacement_therapy_indications.json
```

**Recommended Patterns**:
```bash
patterns/extract_creatinine_values.json
patterns/extract_dates_and_times.json
patterns/dialysis_mentions.json
```

---

### Example Task 2: Malnutrition Detection from Serial Measurements

```json
{
  "task_name": "MIMIC_Malnutrition_Detection",
  "description": "Detect malnutrition from serial weight, albumin, and prealbumin measurements in ICU patients",
  "prompt": "Use the malnutrition template",
  "json_schema": {
    "malnutrition_status": {"type": "string", "description": "Present/Absent/At-Risk", "required": true},
    "weight_timeline": {"type": "string", "description": "Serial weights with dates and % change", "required": true},
    "albumin_timeline": {"type": "string", "description": "Serial albumin with dates and trends", "required": false},
    "prealbumin_timeline": {"type": "string", "description": "Serial prealbumin with dates and trends", "required": false},
    "malnutrition_indicators": {"type": "string", "description": "Which indicators are met (weight loss, low albumin, etc.)", "required": true}
  },
  "enable_rag": true
}
```

**Recommended Functions**:
```bash
functions/calculate_weight_change_percentage.py
functions/calculate_bmi.py
functions/interpret_albumin_malnutrition.py
functions/calculate_days_between_measurements.py
```

---

## PubMed Central Open Access

### Access Information
- **Website**: https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/
- **Cost**: FREE
- **Requirements**: None
- **Best for**: Medical concept extraction, entity recognition, relationship extraction
- **Data type**: Full-text biomedical literature

### Dataset Details
- 3+ million open access articles
- Well-structured XML format
- MeSH terms as ground truth
- Abstracts and full text

### Example Task: Disease-Symptom Extraction

```json
{
  "task_name": "PubMed_Disease_Symptom_Extraction",
  "description": "Extract disease-symptom relationships from biomedical literature",
  "json_schema": {
    "disease": {"type": "string", "required": true},
    "symptoms": {"type": "array", "description": "List of associated symptoms", "required": true},
    "relationships": {"type": "string", "description": "How symptoms relate to disease", "required": true}
  }
}
```

**Recommended Extras**:
```bash
extras/mesh_terms_diseases.json
extras/mesh_terms_symptoms.json
extras/symptom_synonyms.json
```

---

## CASI - Clinical Abbreviation Sense Inventory

### Access Information
- **Website**: https://conservancy.umn.edu/handle/11299/137703
- **Cost**: FREE
- **Requirements**: None
- **Best for**: Abbreviation disambiguation
- **Data type**: Clinical abbreviations with context

### Dataset Details
- 440 instances of 75 ambiguous abbreviations
- Example: "RA" → Rheumatoid Arthritis vs Right Atrium vs Room Air
- Clinical notes with annotated sense

### Example Task: Abbreviation Disambiguation

```json
{
  "task_name": "CASI_Abbreviation_Disambiguation",
  "description": "Disambiguate ambiguous clinical abbreviations based on context",
  "json_schema": {
    "abbreviation": {"type": "string", "required": true},
    "disambiguated_meaning": {"type": "string", "description": "Full expansion in this context", "required": true},
    "evidence": {"type": "string", "description": "Contextual clues supporting disambiguation", "required": true},
    "alternative_meanings_considered": {"type": "array", "description": "Other possible meanings ruled out", "required": false}
  },
  "enable_rag": true
}
```

**Recommended Extras**:
```bash
extras/clinical_abbreviations_dictionary.json
extras/specialty_specific_abbreviations.json  # Cardiology, Neurology, etc.
```

**Recommended Patterns**:
```bash
patterns/abbreviation_context_clues.json
```

---

## MedNLI - Medical Natural Language Inference

### Access Information
- **Website**: https://physionet.org/content/mednli/
- **Cost**: FREE
- **Requirements**: PhysioNet account
- **Best for**: Clinical reasoning, logical inference
- **Data type**: Premise-hypothesis pairs from MIMIC-III

### Dataset Details
- 14,049 sentence pairs
- Labels: Entailment, Contradiction, Neutral
- Example:
  - Premise: "Patient has elevated creatinine of 2.1"
  - Hypothesis: "Patient has renal impairment"
  - Label: Entailment

### Example Task

```json
{
  "task_name": "MedNLI_Clinical_Inference",
  "description": "Determine logical relationship between clinical statements",
  "json_schema": {
    "relationship": {"type": "string", "description": "Entailment/Contradiction/Neutral", "required": true},
    "reasoning": {"type": "string", "description": "Clinical reasoning for the relationship", "required": true}
  }
}
```

---

## MTSamples - Medical Transcription Samples

### Access Information
- **Website**: https://www.mtsamples.com/ or https://github.com/thoppe/medical_transcriptions
- **Cost**: FREE
- **Requirements**: None
- **Best for**: Multi-specialty clinical text, variety of note types
- **Data type**: ~5,000 medical transcriptions across 40+ specialties

### Dataset Details
- Transcribed medical reports
- Specialties: Cardiology, Dermatology, ENT, Gastroenterology, Neurology, Orthopedic, Psychiatry, Radiology, Surgery, etc.
- Note types: H&P, Consultation, Operative notes, Discharge summaries

### Example Task: Procedure Extraction

```json
{
  "task_name": "MTSamples_Procedure_Extraction",
  "description": "Extract surgical procedures and findings from operative notes",
  "json_schema": {
    "procedures_performed": {"type": "array", "required": true},
    "indication": {"type": "string", "required": true},
    "findings": {"type": "string", "required": true},
    "complications": {"type": "string", "required": false}
  }
}
```

**Recommended Extras**:
```bash
extras/cpt_procedure_codes.json
extras/surgical_terminology.json
```

---

## Quick Reference: Functions, Extras, Patterns by Task

### Malnutrition/Nutrition Tasks

**Functions**:
```bash
functions/calculate_bmi.py
functions/interpret_bmi_category.py
functions/calculate_growth_percentile.py
functions/percentile_to_zscore.py
functions/zscore_to_percentile.py
functions/interpret_zscore_malnutrition.py
functions/calculate_ideal_body_weight.py
functions/calculate_weight_change_percentage.py
```

**Extras**:
```bash
extras/who_malnutrition_z_score_criteria.json
extras/aspen_pediatric_malnutrition_criteria_with_z_scores.json
extras/cdc_growth_chart_percentiles.json
extras/nutrition_focused_physical_exam.json
extras/refeeding_syndrome_risk_factors.json
extras/malnutrition_icd10_codes.json
```

**Patterns**:
```bash
patterns/extract_bmi_values.json
patterns/extract_weight_measurements.json
patterns/extract_albumin_values.json
patterns/malnutrition_physical_exam_findings.json
patterns/standardize_malnutrition_severity.json
```

**RAG Documents**:
- ASPEN Pediatric Malnutrition Guidelines PDF
- WHO Child Growth Standards
- Academy of Nutrition and Dietetics Malnutrition Documentation
- Refeeding Syndrome Guidelines

---

### AKI/Renal Function Tasks

**Functions**:
```bash
functions/calculate_creatinine_clearance.py
functions/calculate_egfr.py
functions/classify_aki_stage_kdigo.py
functions/calculate_ckd_stage.py
```

**Extras**:
```bash
extras/kdigo_aki_criteria.json
extras/kdigo_ckd_stages.json
extras/aki_risk_factors.json
extras/renal_replacement_therapy_indications.json
extras/nephrotoxic_medications.json
```

**Patterns**:
```bash
patterns/extract_creatinine_values.json
patterns/extract_urine_output.json
patterns/dialysis_mentions.json
```

**RAG Documents**:
- KDIGO AKI Guidelines
- KDIGO CKD Guidelines
- Nephrotoxic Medication Lists

---

### Diabetes Tasks

**Functions**:
```bash
functions/interpret_hba1c.py
functions/calculate_insulin_dose.py
functions/calculate_carbohydrate_ratio.py
```

**Extras**:
```bash
extras/ada_diabetes_diagnostic_criteria.json
extras/diabetes_complications_screening.json
extras/insulin_types_and_timing.json
extras/diabetes_medication_classes.json
```

**Patterns**:
```bash
patterns/extract_hba1c_values.json
patterns/extract_glucose_values.json
patterns/diabetes_medications.json
```

**RAG Documents**:
- ADA Standards of Care
- Insulin Dosing Guidelines
- Diabetes Complications Screening Recommendations

---

### Cardiac/Cardiovascular Tasks

**Functions**:
```bash
functions/calculate_cvd_risk.py
functions/calculate_framingham_risk.py
functions/calculate_ascvd_risk.py
functions/interpret_troponin.py
functions/interpret_bnp.py
```

**Extras**:
```bash
extras/aha_heart_failure_stages.json
extras/nyha_classification.json
extras/stemi_diagnostic_criteria.json
extras/anticoagulation_indications.json
```

**Patterns**:
```bash
patterns/extract_blood_pressure.json
patterns/extract_troponin_values.json
patterns/extract_bnp_values.json
patterns/cardiac_exam_findings.json
```

**RAG Documents**:
- AHA/ACC Heart Failure Guidelines
- STEMI/NSTEMI Guidelines
- Lipid Management Guidelines

---

### Medication-Related Tasks

**Functions**:
```bash
functions/calculate_creatinine_clearance.py  # For renal dosing
functions/calculate_medication_dose_adjustment.py
```

**Extras**:
```bash
extras/medication_routes.json  # PO, IV, IM, SubQ, etc.
extras/medication_frequencies.json  # BID, TID, QD, PRN
extras/high_alert_medications.json
extras/renal_dose_adjustments.json
```

**Patterns**:
```bash
patterns/medication_dosage_extraction.json
patterns/medication_routes.json
patterns/medication_frequencies.json
patterns/medication_discontinuation.json
```

---

### General Clinical Tasks

**Functions**:
```bash
functions/extract_age_from_dates.py
functions/calculate_days_between_dates.py
functions/convert_units.py
```

**Extras**:
```bash
extras/icd10_codes_common.json
extras/clinical_abbreviations_dictionary.json
extras/vital_sign_normal_ranges.json
extras/lab_value_normal_ranges.json
```

**Patterns**:
```bash
patterns/extract_dates_and_times.json
patterns/negation_detection.json
patterns/temporal_qualifiers.json  # "history of", "acute", "chronic"
patterns/severity_modifiers.json  # "mild", "moderate", "severe"
patterns/extract_vital_signs.json
```

---

## How to Use This Catalog

### Step 1: Choose Your Dataset

Match your research question to the appropriate dataset:
- **Classification tasks** → i2b2 2008 Obesity, Cohort Selection
- **Temporal/Serial data** → MIMIC-III/IV
- **Medication extraction** → i2b2 2022, MIMIC
- **Abbreviation disambiguation** → CASI
- **Clinical reasoning** → MedNLI
- **Multi-specialty variety** → MTSamples

### Step 2: Download and Prepare

1. Follow access instructions for chosen dataset
2. Download data
3. Format as CSV for clinAnnotate:
   - Required columns: `id`, `text`
   - Optional: `label` (for evaluation)

### Step 3: Configure clinAnnotate

Use the JSON configuration examples above, or start with a template:

```bash
# Use malnutrition template for nutrition tasks
python annotate.py --template malnutrition

# Use diabetes template for diabetes tasks
python annotate.py --template diabetes

# Start from scratch
python annotate.py --template blank
```

### Step 4: Add Recommended Resources

Based on the Quick Reference section:

1. **Add Functions**: Copy relevant functions to `functions/` directory
2. **Add Extras**: Copy relevant extras to `extras/` directory
3. **Add Patterns**: Copy relevant patterns to `patterns/` directory
4. **Upload RAG Documents**: Add PDFs of clinical guidelines to RAG system

### Step 5: Process and Evaluate

```bash
# Process through clinAnnotate
python annotate.py

# Evaluate results
python evaluation/evaluate_system.py \
    --gold dataset_gold_standard.csv \
    --system clinannotate_output.csv \
    --task classification
```

### Step 6: Iterate and Improve

Based on evaluation results:
1. Review errors
2. Refine prompts
3. Add missing functions/extras/patterns
4. Enhance RAG documents
5. Re-evaluate

---

## Creating Your Own Functions, Extras, and Patterns

### Creating a Function

Template: `functions/your_function_name.json`

```json
{
  "name": "your_function_name",
  "description": "What this function does",
  "parameters": {
    "param1": {
      "type": "number",
      "description": "First parameter",
      "required": true
    },
    "param2": {
      "type": "string",
      "description": "Second parameter",
      "required": false
    }
  }
}
```

Python implementation: `functions/your_function_name.py`

```python
def your_function_name(param1: float, param2: str = None) -> dict:
    """
    Function description

    Args:
        param1: Description
        param2: Description

    Returns:
        Dictionary with results
    """
    # Your implementation
    result = param1 * 2  # Example

    return {
        "calculated_value": result,
        "interpretation": f"Result is {result}",
        "category": "normal" if result < 100 else "elevated"
    }
```

### Creating an Extra

Template: `extras/your_extra_name.json`

```json
{
  "name": "your_extra_name",
  "type": "guideline",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "content": "Detailed guideline or hint content that will be retrieved when keywords match. This should be comprehensive and cite sources.",
  "source": "Citation or source reference",
  "relevance_score": 1.0
}
```

### Creating a Pattern

Template: `patterns/your_pattern_name.json`

```json
{
  "name": "your_pattern_name",
  "pattern": "regex pattern here",
  "replacement": "standardized output",
  "description": "What this pattern extracts or normalizes",
  "examples": [
    {
      "input": "Example input text",
      "output": "Expected output"
    }
  ]
}
```

---

## Support and Troubleshooting

### Common Issues

**Q: Dataset access denied?**
- Verify DUA is signed
- Check CITI training is complete (for MIMIC)
- Allow 1-2 days for approval

**Q: Poor performance on dataset?**
1. Check if using appropriate functions for the task
2. Add domain-specific RAG documents
3. Review error cases - add patterns for common failures
4. Consider task-specific prompt refinement

**Q: Which dataset for my use case?**
- **Testing serial measurements**: MIMIC
- **Benchmarking classification**: i2b2 2008 Obesity
- **Complex reasoning**: MedNLI or i2b2 Cohort Selection
- **Medication extraction**: i2b2 2022
- **Quick prototyping**: MTSamples (no registration needed)

### Getting Help

- Check main documentation: `README.md`
- Evaluation guide: `evaluation/README.md`
- Quick start: `evaluation/QUICKSTART.md`
- Create an issue: [GitHub repository]

---

## Summary Table: Dataset Quick Reference

| Dataset | Best For | Access Time | Difficulty | Recommended First Task |
|---------|----------|-------------|------------|------------------------|
| i2b2 2008 Obesity | Classification | 1-2 days | ⭐ Easy | Obesity binary classification |
| i2b2 2022 Medication | Entity extraction | 1-2 days | ⭐⭐ Medium | Medication list extraction |
| MIMIC-III | Temporal/Serial | 1 week | ⭐⭐⭐ Hard | AKI detection |
| CASI | Disambiguation | Immediate | ⭐ Easy | Abbreviation sense inventory |
| MedNLI | Clinical reasoning | Immediate | ⭐⭐ Medium | Inference tasks |
| MTSamples | General variety | Immediate | ⭐ Easy | Procedure extraction |

---

**Last Updated**: 2025-01-05
**Version**: 1.0.0
