# MIMIC-IV 20 Diagnoses - Complete Resources Creation Guide

**Purpose**: This document provides comprehensive specifications for creating ClinOrchestra resources (patterns, functions, extras) for all 20 consolidated diagnoses.

**Last Updated**: 2025-11-12

---

## Table of Contents

1. [Patterns to Create](#patterns-to-create)
2. [Functions to Create](#functions-to-create)
3. [Extras to Create](#extras-to-create)
4. [Implementation Examples](#implementation-examples)
5. [RAG Resources](#rag-resources)

---

## PATTERNS TO CREATE

Patterns are regex-based text normalizers that standardize clinical terminology before LLM processing.

### Cardiovascular Patterns (9 diagnoses)

#### 1. Chest Pain Patterns
```json
{
  "name": "normalize_chest_pain_location",
  "pattern": "(chest pain|CP)\\s+(in|at|located in)\\s+(left|right|center|substernal)",
  "replacement": "chest pain \\3",
  "description": "Standardize chest pain location descriptions",
  "enabled": true
}
```

```json
{
  "name": "extract_chest_pain_character",
  "pattern": "(sharp|dull|crushing|pressure|squeezing|burning)\\s+(chest pain|CP)",
  "replacement": "chest pain (character: \\1)",
  "description": "Extract and standardize chest pain character",
  "enabled": true
}
```

#### 2. Troponin Value Patterns
```json
{
  "name": "normalize_troponin_values",
  "pattern": "troponin[-\\s]?([IT])\\s*:?\\s*(\\d+\\.?\\d*)\\s*(ng/mL|ng/dL)?",
  "replacement": "troponin-\\1: \\2 ng/mL",
  "description": "Standardize troponin reporting",
  "enabled": true
}
```

#### 3. ECG Findings Patterns
```json
{
  "name": "standardize_st_changes",
  "pattern": "ST\\s+(elevation|depression)\\s+(in\\s+)?(leads?\\s+)?([IVaVLF,\\s]+)",
  "replacement": "ST \\1 in leads \\4",
  "description": "Standardize ST segment change reporting",
  "enabled": true
}
```

#### 4. Blood Pressure Patterns
```json
{
  "name": "normalize_blood_pressure",
  "pattern": "BP:?\\s*(\\d{2,3})\\s*/\\s*(\\d{2,3})\\s*(mmHg)?",
  "replacement": "BP: \\1/\\2 mmHg",
  "description": "Standardize blood pressure format",
  "enabled": true
}
```

#### 5. Heart Rate Patterns
```json
{
  "name": "normalize_heart_rate",
  "pattern": "(HR|heart rate):?\\s*(\\d{2,3})\\s*(bpm|beats?/min)?",
  "replacement": "heart rate \\2 bpm",
  "description": "Standardize heart rate format",
  "enabled": true
}
```

#### 6. Ejection Fraction Patterns
```json
{
  "name": "extract_ejection_fraction",
  "pattern": "(EF|ejection fraction)\\s*:?\\s*(\\d{1,2})[-\\s]?(\\d{1,2})?%?",
  "replacement": "EF: \\2% (\\3% if range)",
  "description": "Extract and standardize ejection fraction",
  "enabled": true
}
```

#### 7. Syncope Duration Patterns
```json
{
  "name": "normalize_syncope_duration",
  "pattern": "(syncope|LOC|loss of consciousness)\\s+(for|lasting|duration)\\s+(\\d+)\\s*(seconds?|minutes?|sec|min)",
  "replacement": "syncope duration: \\3 \\4",
  "description": "Standardize syncope duration",
  "enabled": true
}
```

### Infectious Disease Patterns (3 diagnoses)

#### 8. Temperature Patterns
```json
{
  "name": "normalize_temperature",
  "pattern": "(temp|temperature):?\\s*(\\d{2,3}\\.?\\d?)\\s*°?(F|C|fahrenheit|celsius)?",
  "replacement": "temperature \\2°\\3",
  "description": "Standardize temperature recording",
  "enabled": true
}
```

#### 9. WBC Count Patterns
```json
{
  "name": "extract_wbc_count",
  "pattern": "(WBC|white blood cell count):?\\s*(\\d+\\.?\\d*)\\s*(k|K|thousand|x10\\^3)?/?μL",
  "replacement": "WBC: \\2 x10^3/μL",
  "description": "Standardize WBC count format",
  "enabled": true
}
```

#### 10. Lactate Level Patterns
```json
{
  "name": "normalize_lactate",
  "pattern": "lactate:?\\s*(\\d+\\.?\\d*)\\s*(mmol/L|mg/dL)?",
  "replacement": "lactate \\1 mmol/L",
  "description": "Standardize lactate level reporting",
  "enabled": true
}
```

#### 11. Blood Culture Patterns
```json
{
  "name": "standardize_blood_culture",
  "pattern": "blood culture[s]?\\s+(positive|negative|pending)\\s+(for\\s+)?([A-Za-z\\s]+)?",
  "replacement": "blood culture: \\1 (organism: \\3)",
  "description": "Standardize blood culture results",
  "enabled": true
}
```

### Renal Patterns (1 diagnosis)

#### 12. Creatinine Patterns
```json
{
  "name": "normalize_creatinine",
  "pattern": "(Cr|creatinine):?\\s*(\\d+\\.?\\d*)\\s*(mg/dL)?",
  "replacement": "creatinine \\2 mg/dL",
  "description": "Standardize creatinine reporting",
  "enabled": true
}
```

#### 13. Urine Output Patterns
```json
{
  "name": "extract_urine_output",
  "pattern": "(UOP|urine output):?\\s*(\\d+\\.?\\d*)\\s*(mL|cc)\\s*(in|over|per)\\s*(\\d+)\\s*(hour|hr|h)",
  "replacement": "urine output: \\2 mL over \\5 hours",
  "description": "Standardize urine output recording",
  "enabled": true
}
```

### Psychiatric Patterns (3 diagnoses)

#### 14. CAGE Score Patterns
```json
{
  "name": "extract_cage_score",
  "pattern": "CAGE\\s+score:?\\s*(\\d)\\s*/\\s*4",
  "replacement": "CAGE score: \\1/4",
  "description": "Standardize CAGE score for alcohol use",
  "enabled": true
}
```

#### 15. PHQ-9 Score Patterns
```json
{
  "name": "extract_phq9_score",
  "pattern": "PHQ[-\\s]?9\\s+score:?\\s*(\\d{1,2})",
  "replacement": "PHQ-9 score: \\1",
  "description": "Extract PHQ-9 depression screening score",
  "enabled": true
}
```

### Respiratory Patterns (1 diagnosis)

#### 16. Oxygen Saturation Patterns
```json
{
  "name": "normalize_spo2",
  "pattern": "(SpO2|O2 sat|oxygen saturation):?\\s*(\\d{1,3})%?\\s*(on|@)?\\s*(\\d+)?\\s*L?\\s*(room air|RA)?",
  "replacement": "SpO2: \\2% (\\3 \\4L or \\5)",
  "description": "Standardize oxygen saturation reporting",
  "enabled": true
}
```

#### 17. FEV1 Patterns
```json
{
  "name": "extract_fev1",
  "pattern": "FEV1:?\\s*(\\d+\\.?\\d*)\\s*L?\\s*\\((\\d{1,3})%\\s*predicted\\)?",
  "replacement": "FEV1: \\1L (\\2% predicted)",
  "description": "Standardize FEV1 reporting for COPD",
  "enabled": true
}
```

### Gastrointestinal Patterns (1 diagnosis)

#### 18. Lipase/Amylase Patterns
```json
{
  "name": "normalize_pancreatic_enzymes",
  "pattern": "(lipase|amylase):?\\s*(\\d+\\.?\\d*)\\s*(U/L|IU/L)?",
  "replacement": "\\1: \\2 U/L",
  "description": "Standardize lipase/amylase for pancreatitis",
  "enabled": true
}
```

### Neurological Patterns (1 diagnosis)

#### 19. NIHSS Score Patterns
```json
{
  "name": "extract_nihss_score",
  "pattern": "NIHSS\\s+score:?\\s*(\\d{1,2})",
  "replacement": "NIHSS score: \\1",
  "description": "Extract NIH Stroke Scale score",
  "enabled": true
}
```

#### 20. Glasgow Coma Scale Patterns
```json
{
  "name": "normalize_gcs",
  "pattern": "GCS:?\\s*(\\d{1,2})\\s*\\((E\\d+|eye\\s*\\d+)[,\\s]*(V\\d+|verbal\\s*\\d+)[,\\s]*(M\\d+|motor\\s*\\d+)\\)?",
  "replacement": "GCS: \\1 (\\2, \\3, \\4)",
  "description": "Standardize Glasgow Coma Scale reporting",
  "enabled": true
}
```

---

## FUNCTIONS TO CREATE

Functions enable clinical calculations and severity scoring.

### Cardiovascular Functions (9 diagnoses)

#### 1. TIMI Risk Score (for MI/ACS)
```python
def calculate_timi_risk_score(age, diabetes=False, hypertension=False,
                               angina=False, st_elevation=False,
                               elevated_biomarkers=False,
                               aspirin_use=False):
    """
    Calculate TIMI Risk Score for UA/NSTEMI
    Score range: 0-7

    Parameters:
        age (int): Patient age in years
        diabetes (bool): History of diabetes
        hypertension (bool): History of hypertension
        angina (bool): ≥2 anginal events in past 24h
        st_elevation (bool): ST deviation ≥0.5mm
        elevated_biomarkers (bool): Elevated cardiac markers
        aspirin_use (bool): Aspirin use in past 7 days

    Returns:
        int: TIMI score (0-7)
    """
    score = 0
    if age >= 65:
        score += 1
    if diabetes:
        score += 1
    if hypertension:
        score += 1
    if angina:
        score += 1
    if st_elevation:
        score += 1
    if elevated_biomarkers:
        score += 1
    if aspirin_use:
        score += 1
    return score
```

#### 2. Heart Score (for chest pain)
```python
def calculate_heart_score(history_score, ecg_score, age,
                          risk_factors, troponin_elevated):
    """
    Calculate HEART Score for chest pain evaluation
    Score range: 0-10
    Low risk: 0-3, Moderate: 4-6, High: 7-10

    Parameters:
        history_score (int): 0=slightly suspicious, 1=moderately, 2=highly
        ecg_score (int): 0=normal, 1=non-specific, 2=significant
        age (int): Patient age in years
        risk_factors (int): Number of risk factors (0-3+)
        troponin_elevated (int): 0=normal, 1=1-3x limit, 2=>3x limit

    Returns:
        dict: {'score': int, 'risk_category': str}
    """
    score = history_score + ecg_score

    # Age component
    if age < 45:
        score += 0
    elif age < 65:
        score += 1
    else:
        score += 2

    # Risk factors (≥3 factors = 2 points)
    score += min(risk_factors, 2)

    # Troponin
    score += troponin_elevated

    if score <= 3:
        category = "Low risk"
    elif score <= 6:
        category = "Moderate risk"
    else:
        category = "High risk"

    return {'score': score, 'risk_category': category}
```

#### 3. Mean Arterial Pressure
```python
def calculate_map(systolic, diastolic):
    """
    Calculate Mean Arterial Pressure

    Parameters:
        systolic (float): Systolic BP in mmHg
        diastolic (float): Diastolic BP in mmHg

    Returns:
        float: MAP in mmHg
    """
    if systolic <= 0 or diastolic <= 0:
        return None
    map_value = diastolic + (systolic - diastolic) / 3
    return round(map_value, 1)
```

#### 4. CHA2DS2-VASc Score (for AFib)
```python
def calculate_chads2_vasc_score(age, female, chf=False, hypertension=False,
                                 diabetes=False, stroke_tia=False,
                                 vascular_disease=False):
    """
    Calculate CHA2DS2-VASc score for stroke risk in atrial fibrillation
    Score range: 0-9

    Parameters:
        age (int): Patient age
        female (bool): Female gender
        chf (bool): Congestive heart failure
        hypertension (bool): Hypertension
        diabetes (bool): Diabetes mellitus
        stroke_tia (bool): Prior stroke/TIA/thromboembolism
        vascular_disease (bool): Vascular disease (MI, PAD, aortic plaque)

    Returns:
        dict: {'score': int, 'risk_category': str}
    """
    score = 0
    if chf:
        score += 1
    if hypertension:
        score += 1
    if age >= 75:
        score += 2
    elif age >= 65:
        score += 1
    if diabetes:
        score += 1
    if stroke_tia:
        score += 2
    if vascular_disease:
        score += 1
    if female:
        score += 1

    if score == 0:
        risk = "Low risk"
    elif score == 1:
        risk = "Low-moderate risk"
    else:
        risk = "Moderate-high risk (consider anticoagulation)"

    return {'score': score, 'risk_category': risk}
```

### Infectious Disease Functions (3 diagnoses)

#### 5. SOFA Score (for Sepsis)
```python
def calculate_sofa_score(pao2_fio2=None, platelets=None, bilirubin=None,
                         map_value=None, gcs=None, creatinine=None,
                         on_vasopressor=False, urine_output=None):
    """
    Calculate Sequential Organ Failure Assessment (SOFA) score
    Score range: 0-24 (higher = more severe)

    Parameters:
        pao2_fio2 (float): PaO2/FiO2 ratio (if <400)
        platelets (float): Platelet count (x10^3/μL)
        bilirubin (float): Bilirubin (mg/dL)
        map_value (float): Mean arterial pressure (mmHg)
        gcs (int): Glasgow Coma Scale score (3-15)
        creatinine (float): Creatinine (mg/dL)
        on_vasopressor (bool): On vasopressor support
        urine_output (float): 24h urine output (mL)

    Returns:
        dict: {'total_score': int, 'subscores': dict}
    """
    subscores = {}

    # Respiration
    resp_score = 0
    if pao2_fio2:
        if pao2_fio2 < 100:
            resp_score = 4
        elif pao2_fio2 < 200:
            resp_score = 3
        elif pao2_fio2 < 300:
            resp_score = 2
        elif pao2_fio2 < 400:
            resp_score = 1
    subscores['respiration'] = resp_score

    # Coagulation
    coag_score = 0
    if platelets:
        if platelets < 20:
            coag_score = 4
        elif platelets < 50:
            coag_score = 3
        elif platelets < 100:
            coag_score = 2
        elif platelets < 150:
            coag_score = 1
    subscores['coagulation'] = coag_score

    # Liver
    liver_score = 0
    if bilirubin:
        if bilirubin >= 12.0:
            liver_score = 4
        elif bilirubin >= 6.0:
            liver_score = 3
        elif bilirubin >= 2.0:
            liver_score = 2
        elif bilirubin >= 1.2:
            liver_score = 1
    subscores['liver'] = liver_score

    # Cardiovascular
    cv_score = 0
    if on_vasopressor:
        cv_score = 3  # Simplified - varies by dose
    elif map_value and map_value < 70:
        cv_score = 1
    subscores['cardiovascular'] = cv_score

    # CNS
    cns_score = 0
    if gcs:
        if gcs < 6:
            cns_score = 4
        elif gcs < 10:
            cns_score = 3
        elif gcs < 13:
            cns_score = 2
        elif gcs < 15:
            cns_score = 1
    subscores['cns'] = cns_score

    # Renal
    renal_score = 0
    if creatinine:
        if creatinine >= 5.0:
            renal_score = 4
        elif creatinine >= 3.5:
            renal_score = 3
        elif creatinine >= 2.0:
            renal_score = 2
        elif creatinine >= 1.2:
            renal_score = 1
    if urine_output and urine_output < 200:
        renal_score = max(renal_score, 4)
    elif urine_output and urine_output < 500:
        renal_score = max(renal_score, 3)
    subscores['renal'] = renal_score

    total = sum(subscores.values())

    return {
        'total_score': total,
        'subscores': subscores
    }
```

#### 6. CURB-65 Score (for Pneumonia)
```python
def calculate_curb65_score(confusion=False, bun=None, resp_rate=None,
                           sbp=None, dbp=None, age=None):
    """
    Calculate CURB-65 score for pneumonia severity
    Score range: 0-5 (higher = more severe)

    Parameters:
        confusion (bool): New onset confusion
        bun (float): Blood urea nitrogen (mg/dL)
        resp_rate (int): Respiratory rate (breaths/min)
        sbp (int): Systolic BP (mmHg)
        dbp (int): Diastolic BP (mmHg)
        age (int): Patient age in years

    Returns:
        dict: {'score': int, 'severity': str, 'recommendation': str}
    """
    score = 0

    if confusion:
        score += 1
    if bun and bun > 19:  # >7 mmol/L
        score += 1
    if resp_rate and resp_rate >= 30:
        score += 1
    if (sbp and sbp < 90) or (dbp and dbp <= 60):
        score += 1
    if age and age >= 65:
        score += 1

    if score <= 1:
        severity = "Low severity"
        recommendation = "Outpatient treatment suitable"
    elif score == 2:
        severity = "Moderate severity"
        recommendation = "Consider short inpatient stay or close outpatient monitoring"
    else:
        severity = "High severity"
        recommendation = "Inpatient treatment recommended"

    return {
        'score': score,
        'severity': severity,
        'recommendation': recommendation
    }
```

### Renal Functions (1 diagnosis)

#### 7. KDIGO AKI Stage
```python
def calculate_aki_stage(creatinine_current, creatinine_baseline,
                        urine_output_6h=None, urine_output_12h=None,
                        urine_output_24h=None, weight_kg=None):
    """
    Calculate KDIGO Acute Kidney Injury Stage

    Parameters:
        creatinine_current (float): Current creatinine (mg/dL)
        creatinine_baseline (float): Baseline creatinine (mg/dL)
        urine_output_6h (float): 6-hour urine output (mL)
        urine_output_12h (float): 12-hour urine output (mL)
        urine_output_24h (float): 24-hour urine output (mL)
        weight_kg (float): Patient weight (kg)

    Returns:
        dict: {'stage': int, 'criteria_met': list}
    """
    criteria = []
    stage = 0

    # Creatinine criteria
    cr_increase = creatinine_current - creatinine_baseline
    cr_ratio = creatinine_current / creatinine_baseline if creatinine_baseline > 0 else 0

    if cr_increase >= 0.3:
        stage = max(stage, 1)
        criteria.append("Cr increase ≥0.3 mg/dL")

    if cr_ratio >= 1.5 and cr_ratio < 2.0:
        stage = max(stage, 1)
        criteria.append("Cr 1.5-1.9x baseline")
    elif cr_ratio >= 2.0 and cr_ratio < 3.0:
        stage = max(stage, 2)
        criteria.append("Cr 2.0-2.9x baseline")
    elif cr_ratio >= 3.0:
        stage = max(stage, 3)
        criteria.append("Cr ≥3.0x baseline")

    if creatinine_current >= 4.0:
        stage = max(stage, 3)
        criteria.append("Cr ≥4.0 mg/dL")

    # Urine output criteria (if weight provided)
    if weight_kg:
        if urine_output_6h:
            uo_6h_rate = urine_output_6h / (weight_kg * 6)
            if uo_6h_rate < 0.5:
                stage = max(stage, 1)
                criteria.append(f"UO <0.5 mL/kg/h x 6h (actual: {uo_6h_rate:.2f})")

        if urine_output_12h:
            uo_12h_rate = urine_output_12h / (weight_kg * 12)
            if uo_12h_rate < 0.5:
                stage = max(stage, 2)
                criteria.append(f"UO <0.5 mL/kg/h x 12h (actual: {uo_12h_rate:.2f})")

        if urine_output_24h:
            uo_24h_rate = urine_output_24h / (weight_kg * 24)
            if uo_24h_rate < 0.3:
                stage = max(stage, 3)
                criteria.append(f"UO <0.3 mL/kg/h x 24h (actual: {uo_24h_rate:.2f})")

    return {
        'stage': stage if stage > 0 else 0,
        'criteria_met': criteria
    }
```

### Psychiatric Functions (3 diagnoses)

#### 8. CAGE Score
```python
def calculate_cage_score(cut_down=False, annoyed=False,
                         guilty=False, eye_opener=False):
    """
    Calculate CAGE score for alcohol use disorder screening
    Score range: 0-4

    Parameters:
        cut_down (bool): Felt need to Cut down on drinking
        annoyed (bool): People Annoyed you by criticizing drinking
        guilty (bool): Felt Guilty about drinking
        eye_opener (bool): Had Eye-opener (drink first thing in morning)

    Returns:
        dict: {'score': int, 'interpretation': str}
    """
    score = sum([cut_down, annoyed, guilty, eye_opener])

    if score == 0:
        interpretation = "Low likelihood of alcohol use disorder"
    elif score <= 1:
        interpretation = "Consider further assessment"
    else:
        interpretation = "High likelihood of alcohol use disorder - further evaluation recommended"

    return {'score': score, 'interpretation': interpretation}
```

### Respiratory Functions (1 diagnosis)

#### 9. COPD GOLD Stage
```python
def calculate_copd_gold_stage(fev1_percent_predicted, symptoms_score=None):
    """
    Calculate COPD GOLD Spirometric Grade

    Parameters:
        fev1_percent_predicted (float): FEV1 as % of predicted
        symptoms_score (int): Optional - mMRC or CAT score for combined assessment

    Returns:
        dict: {'grade': int, 'severity': str, 'description': str}
    """
    if fev1_percent_predicted >= 80:
        grade = 1
        severity = "Mild"
        description = "FEV1 ≥80% predicted"
    elif fev1_percent_predicted >= 50:
        grade = 2
        severity = "Moderate"
        description = "50% ≤ FEV1 < 80% predicted"
    elif fev1_percent_predicted >= 30:
        grade = 3
        severity = "Severe"
        description = "30% ≤ FEV1 < 50% predicted"
    else:
        grade = 4
        severity = "Very Severe"
        description = "FEV1 < 30% predicted"

    return {
        'grade': grade,
        'severity': severity,
        'description': description
    }
```

### Neurological Functions (1 diagnosis)

#### 10. NIHSS Score
```python
def calculate_nihss_total(loc=0, loc_questions=0, loc_commands=0,
                          gaze=0, visual=0, facial_palsy=0,
                          motor_left_arm=0, motor_right_arm=0,
                          motor_left_leg=0, motor_right_leg=0,
                          limb_ataxia=0, sensory=0,
                          language=0, dysarthria=0, extinction=0):
    """
    Calculate total NIH Stroke Scale (NIHSS) score
    Score range: 0-42 (higher = more severe)

    All parameters are subscores from NIHSS assessment

    Returns:
        dict: {'total_score': int, 'severity': str}
    """
    total = (loc + loc_questions + loc_commands + gaze + visual +
             facial_palsy + motor_left_arm + motor_right_arm +
             motor_left_leg + motor_right_leg + limb_ataxia +
             sensory + language + dysarthria + extinction)

    if total == 0:
        severity = "No stroke symptoms"
    elif total <= 4:
        severity = "Minor stroke"
    elif total <= 15:
        severity = "Moderate stroke"
    elif total <= 20:
        severity = "Moderate to severe stroke"
    else:
        severity = "Severe stroke"

    return {
        'total_score': total,
        'severity': severity
    }
```

---

## EXTRAS TO CREATE

Extras provide clinical knowledge snippets that help LLM understand diagnostic criteria and clinical context.

### Cardiovascular Extras (9 diagnoses)

#### 1. Chest Pain - Cardiac vs Non-Cardiac Features
```json
{
  "type": "diagnostic_criteria",
  "name": "Chest Pain - Cardiac vs Non-Cardiac Features",
  "content": "CARDIAC CHEST PAIN Features: Substernal/left chest pressure or squeezing, radiates to arm/jaw/back, associated with exertion, relieved by rest/nitroglycerin, accompanied by diaphoresis/dyspnea/nausea. Duration >20min concerning for MI. NON-CARDIAC Features: Sharp/stabbing pain, localized to small area, reproduced by palpation, pleuritic (worse with breathing), positional, brief duration (<5min). Risk factors: Age >55, diabetes, hypertension, hyperlipidemia, smoking, family history.",
  "metadata": {
    "category": "cardiology",
    "diagnosis": "chest_pain",
    "priority": "CRITICAL"
  }
}
```

#### 2. MI - Universal Definition & Types
```json
{
  "type": "diagnostic_criteria",
  "name": "Myocardial Infarction - Universal Definition",
  "content": "MI DIAGNOSIS requires: (1) Elevated cardiac troponin (>99th percentile) with rise/fall pattern, PLUS (2) ≥1 of: symptoms of myocardial ischemia, new ST-T changes or LBBB, pathological Q waves, new wall motion abnormality, intracoronary thrombus. MI Types: Type 1 (plaque rupture/thrombosis), Type 2 (supply-demand mismatch), Type 3 (sudden cardiac death), Type 4a (PCI-related), Type 4b (stent thrombosis), Type 5 (CABG-related). STEMI: ST elevation ≥1mm in 2 contiguous leads or new LBBB. NSTEMI: Elevated troponin without ST elevation.",
  "metadata": {
    "category": "cardiology",
    "diagnosis": "myocardial_infarction",
    "priority": "CRITICAL"
  }
}
```

#### 3. Heart Failure - Classification & Ejection Fraction
```json
{
  "type": "diagnostic_criteria",
  "name": "Heart Failure - Classification by Ejection Fraction",
  "content": "HF Classification by EF: HFrEF (reduced EF <40%), HFmrEF (mildly reduced 40-49%), HFpEF (preserved ≥50%). Acute Decompensation Signs: Orthopnea, PND, lower extremity edema, JVD, S3 gallop, pulmonary rales/crackles, hepatomegaly. Chest X-ray: Cardiomegaly, pulmonary vascular congestion, Kerley B lines, pleural effusions. BNP >400 pg/mL or NT-proBNP >900 pg/mL supportive. NYHA Class: I (no limitation), II (slight limitation), III (marked limitation), IV (symptoms at rest).",
  "metadata": {
    "category": "cardiology",
    "diagnosis": "heart_failure",
    "priority": "CRITICAL"
  }
}
```

#### 4. Atrial Fibrillation - Types & Management
```json
{
  "type": "diagnostic_criteria",
  "name": "Atrial Fibrillation - Classification & Anticoagulation",
  "content": "AFib Types: Paroxysmal (self-terminating <7 days), Persistent (>7 days or requiring cardioversion), Long-standing persistent (>12 months), Permanent (accepted). ECG: Irregularly irregular rhythm, absent P waves, variable RR intervals. Stroke Risk (CHA2DS2-VASc): CHF(1), Hypertension(1), Age≥75(2), Diabetes(1), Stroke/TIA(2), Vascular disease(1), Age 65-74(1), Sex=female(1). Score ≥2: Anticoagulation recommended. Bleeding Risk (HAS-BLED): Hypertension, Abnormal renal/liver function, Stroke, Bleeding history, Labile INR, Elderly, Drugs/alcohol.",
  "metadata": {
    "category": "cardiology",
    "diagnosis": "atrial_fibrillation",
    "priority": "HIGH"
  }
}
```

### Infectious Disease Extras (3 diagnoses)

#### 5. Sepsis - Sepsis-3 Criteria
```json
{
  "type": "diagnostic_criteria",
  "name": "Sepsis - Sepsis-3 Definition & qSOFA",
  "content": "SEPSIS-3: Life-threatening organ dysfunction caused by dysregulated host response to infection. Requires: (1) Suspected/confirmed infection, PLUS (2) SOFA score increase ≥2 points. SEPTIC SHOCK: Sepsis + vasopressor requirement to maintain MAP≥65 mmHg + lactate >2 mmol/L despite adequate fluid resuscitation. qSOFA (quick screening): RR≥22, Altered mentation, SBP≤100 (≥2 = high risk). SOFA Score: Respiration (PaO2/FiO2), Coagulation (platelets), Liver (bilirubin), Cardiovascular (MAP/vasopressors), CNS (GCS), Renal (creatinine/UOP). SIRS Criteria (legacy): Temp >38°C or <36°C, HR >90, RR >20, WBC >12k or <4k.",
  "metadata": {
    "category": "infectious_disease",
    "diagnosis": "sepsis",
    "priority": "CRITICAL"
  }
}
```

#### 6. Pneumonia - Community vs Healthcare Associated
```json
{
  "type": "diagnostic_criteria",
  "name": "Pneumonia - CAP vs HAP/VAP Classification",
  "content": "PNEUMONIA DIAGNOSIS: Clinical (cough, fever, dyspnea, sputum) + Radiographic (infiltrate on CXR/CT). CAP (Community-Acquired): Onset in community or <48h hospitalization. HAP (Healthcare-Associated): Onset ≥48h after admission. VAP (Ventilator-Associated): Onset ≥48h after intubation. Severity Assessment (CURB-65): Confusion, BUN>19, RR≥30, BP(SBP<90 or DBP≤60), Age≥65. Score 0-1: Outpatient, 2: Inpatient, ≥3: ICU consideration. PSI/PORT Score: Classes I-V for mortality prediction. Typical organisms: S.pneumoniae, H.influenzae, M.catarrhalis. Atypical: Mycoplasma, Legionella, Chlamydia.",
  "metadata": {
    "category": "infectious_disease",
    "diagnosis": "pneumonia",
    "priority": "HIGH"
  }
}
```

#### 7. UTI - Cystitis vs Pyelonephritis
```json
{
  "type": "diagnostic_criteria",
  "name": "UTI - Uncomplicated vs Complicated Classification",
  "content": "UNCOMPLICATED CYSTITIS: Dysuria, frequency, urgency, suprapubic pain. No fever/systemic symptoms. UA: Pyuria (WBC >10/hpf), bacteriuria, +/- hematuria, +/- nitrites. PYELONEPHRITIS: Fever, chills, flank pain, CVA tenderness, nausea/vomiting. COMPLICATED UTI: Males, pregnancy, catheter, immunosuppression, anatomic abnormality, recent instrumentation, resistant organisms. Urine Culture: ≥10^5 CFU/mL significant (≥10^3 in catheterized). Common organisms: E.coli (80%), Klebsiella, Proteus, Enterococcus, Staphylococcus saprophyticus (young females).",
  "metadata": {
    "category": "infectious_disease",
    "diagnosis": "uti",
    "priority": "MODERATE"
  }
}
```

### Renal Extras (1 diagnosis)

#### 8. AKI - KDIGO Staging & Etiology
```json
{
  "type": "diagnostic_criteria",
  "name": "AKI - KDIGO Staging & Etiology",
  "content": "ACUTE KIDNEY INJURY (AKI): Abrupt decline in renal function. KDIGO Criteria - Stage 1: Cr 1.5-1.9x baseline OR ≥0.3 mg/dL increase OR UOP <0.5 mL/kg/h x 6-12h. Stage 2: Cr 2.0-2.9x baseline OR UOP <0.5 mL/kg/h x ≥12h. Stage 3: Cr ≥3.0x baseline OR Cr ≥4.0 mg/dL OR UOP <0.3 mL/kg/h x ≥24h OR anuria x ≥12h OR initiation of RRT. ETIOLOGY: Pre-renal (hypovolemia, hypotension, heart failure, NSAIDs), Intrinsic (ATN, interstitial nephritis, glomerulonephritis), Post-renal (obstruction: BPH, stones, tumor). Workup: FENa <1% = pre-renal, >2% = intrinsic ATN. UA: Muddy brown casts = ATN, WBC casts = pyelonephritis, RBC casts = glomerulonephritis.",
  "metadata": {
    "category": "nephrology",
    "diagnosis": "acute_kidney_injury",
    "priority": "HIGH"
  }
}
```

### Psychiatric Extras (3 diagnoses)

#### 9. Depression - DSM-5 Criteria
```json
{
  "type": "diagnostic_criteria",
  "name": "Major Depressive Disorder - DSM-5 Criteria",
  "content": "MAJOR DEPRESSIVE DISORDER requires ≥5 symptoms during 2-week period, including (1) Depressed mood OR (2) Anhedonia (loss of interest/pleasure), PLUS: Significant weight/appetite change, insomnia/hypersomnia, psychomotor agitation/retardation, fatigue, worthlessness/guilt, decreased concentration, recurrent thoughts of death/suicidal ideation. Symptoms cause clinically significant distress/impairment. PHQ-9 Screening: Score 0-4=minimal, 5-9=mild, 10-14=moderate, 15-19=moderately severe, 20-27=severe depression. Severity Specifiers: Mild (few symptoms beyond minimum), Moderate, Severe (marked impairment). Features: With anxious distress, melancholic features, atypical features, psychotic features, peripartum onset, seasonal pattern.",
  "metadata": {
    "category": "psychiatry",
    "diagnosis": "depression",
    "priority": "HIGH"
  }
}
```

#### 10. Alcohol Use Disorder - DSM-5 & CAGE
```json
{
  "type": "diagnostic_criteria",
  "name": "Alcohol Use Disorder - DSM-5 & Screening Tools",
  "content": "ALCOHOL USE DISORDER (DSM-5): ≥2 of 11 criteria in 12-month period: Larger amounts/longer than intended, unsuccessful efforts to cut down, excessive time obtaining/using/recovering, craving, failure to fulfill obligations, continued use despite social/interpersonal problems, activities given up, use in hazardous situations, continued despite physical/psychological problems, tolerance, withdrawal. Severity: Mild (2-3 criteria), Moderate (4-5), Severe (≥6). CAGE Screening: Cut down, Annoyed by criticism, Guilty feelings, Eye-opener morning drink. Score ≥2 = high likelihood. AUDIT-C: 3 questions on frequency/quantity, score ≥4 (men) or ≥3 (women) = positive. Withdrawal: Tremor, anxiety, agitation, tachycardia, hypertension, diaphoresis, seizures (12-48h), delirium tremens (48-96h).",
  "metadata": {
    "category": "psychiatry",
    "diagnosis": "alcohol_use_disorder",
    "priority": "HIGH"
  }
}
```

### Respiratory Extras (1 diagnosis)

#### 11. COPD - GOLD Classification
```json
{
  "type": "diagnostic_criteria",
  "name": "COPD - GOLD Spirometric Grades & ABE Assessment",
  "content": "COPD DIAGNOSIS: Post-bronchodilator FEV1/FVC <0.70. GOLD Spirometric Grades: GOLD 1 (Mild): FEV1 ≥80% predicted. GOLD 2 (Moderate): 50% ≤ FEV1 < 80%. GOLD 3 (Severe): 30% ≤ FEV1 < 50%. GOLD 4 (Very Severe): FEV1 < 30%. Combined Assessment (ABE Groups): Group A (low symptoms, low risk), Group B (high symptoms, low risk), Group E (exacerbation history ≥2 or ≥1 leading to hospitalization). Symptoms: mMRC 0-4 (dyspnea scale) or CAT score 0-40 (COPD Assessment Test). Exacerbation: Acute worsening requiring antibiotics/steroids/hospitalization. ABG in severe: Hypoxemia (PaO2 <60 mmHg), hypercapnia (PaCO2 >50 mmHg), respiratory acidosis.",
  "metadata": {
    "category": "pulmonology",
    "diagnosis": "copd",
    "priority": "MODERATE"
  }
}
```

### Gastrointestinal Extras (1 diagnosis)

#### 12. Acute Pancreatitis - Revised Atlanta Criteria
```json
{
  "type": "diagnostic_criteria",
  "name": "Acute Pancreatitis - Revised Atlanta Classification",
  "content": "ACUTE PANCREATITIS DIAGNOSIS (≥2 of 3): (1) Abdominal pain consistent with pancreatitis (epigastric radiating to back), (2) Serum lipase or amylase ≥3x upper limit of normal, (3) Characteristic findings on CT/MRI/US. Lipase more specific than amylase. Severity: MILD (no organ failure, no complications), MODERATELY SEVERE (transient organ failure <48h and/or local/systemic complications), SEVERE (persistent organ failure >48h). Etiology: Gallstones (40-70%), Alcohol (25-35%), Hypertriglyceridemia (1-4%, usually >1000 mg/dL), Medications, Post-ERCP, Trauma. Ranson's Criteria: At admission - Age>55, WBC>16k, Glucose>200, LDH>350, AST>250. At 48h - Hct drop>10%, BUN rise>5, Ca<8, PaO2<60, Base deficit>4, Fluid sequestration>6L. Score ≥3 = severe.",
  "metadata": {
    "category": "gastroenterology",
    "diagnosis": "acute_pancreatitis",
    "priority": "HIGH"
  }
}
```

### Neurological Extras (1 diagnosis)

#### 13. Stroke - Ischemic vs Hemorrhagic & Time Windows
```json
{
  "type": "diagnostic_criteria",
  "name": "Stroke - Acute Evaluation & Treatment Time Windows",
  "content": "ACUTE STROKE: Sudden neurological deficit from vascular cause. ISCHEMIC (87%): Thrombotic, embolic, lacunar. HEMORRHAGIC (13%): Intracerebral, subarachnoid. CT HEAD: Rule out hemorrhage (hyperdense = acute blood). MRI: DWI shows acute infarct. NIHSS Score: 0-42 (level of consciousness, gaze, visual fields, facial palsy, motor arm/leg, ataxia, sensory, language, dysarthria, extinction). Treatment Windows: IV tPA within 4.5h of onset (exclude hemorrhage, recent surgery, platelets <100k, INR >1.7, glucose <50 or >400). Endovascular thrombectomy: Up to 24h in select patients with large vessel occlusion. Stroke Subtypes (TOAST): Large artery atherosclerosis, cardioembolism, small vessel (lacunar), other determined cause, undetermined.",
  "metadata": {
    "category": "neurology",
    "diagnosis": "stroke",
    "priority": "CRITICAL"
  }
}
```

### Oncology Extras (1 diagnosis)

#### 14. Chemotherapy - Common Regimens & Toxicities
```json
{
  "type": "reference",
  "name": "Chemotherapy - Common Regimens by Cancer Type",
  "content": "COMMON CHEMO REGIMENS: Breast (AC-T: doxorubicin/cyclophosphamide then paclitaxel; TCH: docetaxel/carboplatin/trastuzumab), Lung NSCLC (carboplatin/pem etrezed; cisplatin/etoposide), Colon (FOLFOX: 5-FU/leucovorin/oxaliplatin; FOLFIRI: 5-FU/leucovorin/irinotecan), Lymphoma (CHOP: cyclophosphamide/doxorubicin/vincristine/prednisone; R-CHOP adds rituximab). COMMON TOXICITIES: Myelosuppression (nadir day 7-14), N/V (give antiemetics), Mucositis, Diarrhea, Neuropathy (platinum, taxanes, vincristine), Cardiotoxicity (anthracyclines, trastuzumab), Nephrotoxicity (cisplatin), Hepatotoxicity. Monitor: CBC, CMP, Mg/Ca/Phos with platinum agents.",
  "metadata": {
    "category": "oncology",
    "diagnosis": "chemotherapy_encounter",
    "priority": "MODERATE"
  }
}
```

### Cardiovascular Extras (Continued)

#### 15. Syncope - Evaluation Algorithm
```json
{
  "type": "diagnostic_criteria",
  "name": "Syncope - Evaluation & Risk Stratification",
  "content": "SYNCOPE: Transient loss of consciousness due to cerebral hypoperfusion, rapid onset, short duration, spontaneous complete recovery. Initial Evaluation: History (prodrome, witnesses, post-event), vital signs (orthostatic: drop SBP ≥20 or DBP ≥10 within 3min standing), ECG. High-Risk Features: Age >60, cardiac disease, exertional syncope, syncope while supine, palpitations, sudden onset without prodrome, family history sudden death, abnormal ECG. ECG Red Flags: Bradycardia <40, Mobitz II/3rd degree block, prolonged QT (>500ms), Brugada pattern, epsilon wave (ARVC), pre-excitation (WPW), LVH with strain. Etiologies: Reflex (vasovagal, situational, carotid sinus), Orthostatic (volume depletion, autonomic dysfunction, medications), Cardiac (arrhythmia, structural heart disease, PE).",
  "metadata": {
    "category": "cardiology",
    "diagnosis": "syncope",
    "priority": "HIGH"
  }
}
```

#### 16. Aortic Stenosis - Severity Grading
```json
{
  "type": "diagnostic_criteria",
  "name": "Aortic Stenosis - Echo Criteria & Symptom Triad",
  "content": "AORTIC STENOSIS SEVERITY (Echo): MILD: Valve area >1.5 cm², mean gradient <25 mmHg, peak velocity <3 m/s. MODERATE: Area 1.0-1.5 cm², gradient 25-40 mmHg, velocity 3-4 m/s. SEVERE: Area <1.0 cm², gradient >40 mmHg, velocity >4 m/s. CRITICAL: Area <0.75 cm². Classic Triad: Angina, Syncope, Dyspnea (heart failure). Natural History: Asymptomatic with severe AS = 2% sudden death/year. Once symptoms develop: Angina (5yr survival), Syncope (3yr), CHF (2yr). Physical Exam: Crescendo-decrescendo systolic murmur loudest at RUSB, radiates to carotids, diminished/delayed carotid upstroke (pulsus parvus et tardus), paradoxically split S2 if severe. CXR: LVH, calcified aortic valve, pulmonary congestion if decompensated. Management: Asymptomatic severe + normal EF = follow q6-12mo. Symptomatic or EF<50% = valve replacement (SAVR or TAVR).",
  "metadata": {
    "category": "cardiology",
    "diagnosis": "aortic_valve_disorders",
    "priority": "MODERATE"
  }
}
```

#### 17. Pulmonary Embolism - Wells Score & PERC
```json
{
  "type": "diagnostic_criteria",
  "name": "Pulmonary Embolism - Risk Stratification & Diagnosis",
  "content": "PE DIAGNOSIS: Wells Score for pre-test probability: Clinical DVT signs (3pts), PE #1 diagnosis or equally likely (3pts), HR>100 (1.5pts), Immobilization/surgery in past 4wks (1.5pts), Prior PE/DVT (1.5pts), Hemoptysis (1pt), Malignancy (1pt). Score >6 = high probability. PERC Rule (exclude PE if all negative in low-risk): Age<50, HR<100, SaO2≥95%, no hemoptysis, no estrogen use, no prior PE/DVT, no unilateral leg swelling, no surgery/trauma in 4wks. D-Dimer: Age-adjusted cutoff (age x 10 if >50yo). Negative D-dimer + low Wells = PE excluded. CTA Chest: Gold standard. V/Q scan if contrast contraindicated. Massive PE: Hypotension/shock. Submassive: RV strain on echo/CT, elevated troponin/BNP. Low-risk: PESI class I-II, sPESI=0. Anticoagulation: DOACs first-line (apixaban, rivaroxaban), LMWH bridge to warfarin, or LMWH alone.",
  "metadata": {
    "category": "cardiology",
    "diagnosis": "pulmonary_embolism",
    "priority": "CRITICAL"
  }
}
```

### Psychiatric Extras (Continued)

#### 18. Psychosis - First-Episode Evaluation
```json
{
  "type": "diagnostic_criteria",
  "name": "Psychosis - First-Episode Workup & Differential",
  "content": "PSYCHOSIS: Loss of contact with reality (hallucinations, delusions, disorganized thinking/behavior). FIRST-EPISODE WORKUP: Rule out medical/substance causes: CBC, CMP, LFTs, TSH, RPR, HIV, urine drug screen, B12/folate, brain MRI if focal findings. Substance-Induced: Cannabis, amphetamines, cocaine, PCP, hallucinogens, alcohol withdrawal, anticholinergics, steroids. Medical Causes: Delirium, CNS infection (encephalitis, meningitis), seizures, brain tumor, stroke, thyroid disorder, autoimmune encephalitis. PRIMARY PSYCHOTIC DISORDERS: Schizophrenia (≥6mo symptoms), Schizophreniform (<6mo), Schizoaffective (mood + psychosis), Brief psychotic disorder (<1mo), Delusional disorder. Positive Symptoms: Hallucinations, delusions, disorganized speech/behavior. Negative Symptoms: Flat affect, avolition, alogia, anhedonia, social withdrawal.",
  "metadata": {
    "category": "psychiatry",
    "diagnosis": "psychosis",
    "priority": "HIGH"
  }
}
```

---

## IMPLEMENTATION PRIORITY

### Phase 1: High-Impact Patterns (Create First)
1. Vital signs normalization (BP, HR, RR, Temp, SpO2)
2. Lab value extraction (Troponin, Cr, WBC, Lactate)
3. Clinical scoring mentions (SOFA, TIMI, CURB-65, NIHSS)

### Phase 2: Critical Functions (Create First)
1. SOFA Score (sepsis severity)
2. TIMI Risk Score (ACS risk)
3. CURB-65 (pneumonia severity)
4. KDIGO AKI Staging
5. Mean Arterial Pressure

### Phase 3: Essential Extras (Create First)
1. Sepsis-3 Criteria
2. MI Universal Definition
3. Heart Failure Classification
4. AKI KDIGO Staging
5. Stroke Time Windows

### Phase 4: Complete Remaining Resources
- All other patterns for specialized findings
- Remaining clinical calculators
- Complete extras library for all 20 diagnoses

---

## FILE NAMING CONVENTIONS

### Patterns
```
patterns/
├── cardiovascular_chest_pain_location.json
├── cardiovascular_troponin_values.json
├── cardiovascular_blood_pressure.json
├── infectious_temperature.json
├── infectious_wbc_count.json
├── renal_creatinine.json
└── ...
```

### Functions
```
functions/
├── calculate_timi_risk_score.json
├── calculate_heart_score.json
├── calculate_sofa_score.json
├── calculate_curb65_score.json
├── calculate_kdigo_aki_stage.json
└── ...
```

### Extras
```
extras/
├── cardiovascular_chest_pain_features.json
├── cardiovascular_mi_universal_definition.json
├── infectious_sepsis3_criteria.json
├── renal_aki_kdigo_staging.json
├── psychiatric_depression_dsm5.json
└── ...
```

---

## TESTING CHECKLIST

For each resource type:

### Patterns
- [ ] Regex compiles without errors
- [ ] Test with 3-5 example clinical text snippets
- [ ] Verify replacement preserves clinical meaning
- [ ] Check for false positives

### Functions
- [ ] All parameters have type validation
- [ ] Edge cases handled (zero, negative, None)
- [ ] Returns correct data structure
- [ ] Test with boundary values (min/max scores)
- [ ] Validate against published calculators

### Extras
- [ ] Content is clinically accurate
- [ ] References current guidelines
- [ ] Metadata tags are appropriate
- [ ] Priority level is correct (CRITICAL/HIGH/MODERATE/LOW)

---

## NEXT STEPS

1. **Create JSON files**: Use specifications above to create actual JSON files
2. **Upload to ClinOrchestra**: Use UI tabs to import resources
3. **Test integration**: Process sample MIMIC-IV notes with new resources
4. **Iterate**: Refine patterns/functions based on extraction quality
5. **Document**: Update this guide with lessons learned

---

**Status**: Specification Complete - Ready for Implementation
**Version**: 1.0
**Last Updated**: 2025-11-12
