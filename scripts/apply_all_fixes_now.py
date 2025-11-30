#!/usr/bin/env python3
"""
Apply ALL function fixes immediately - complete implementation
"""
import json
from pathlib import Path

FUNCTIONS_DIR = Path("/home/user/clinorchestra/functions")

# Fix MoCA
moca_fix = {
    "name": "calculate_moca_severity",
    "code": """def calculate_moca_severity(moca_score, education_years=None):
    \"\"\"
    Interpret MoCA score with education correction

    IMPORTANT: Returns cognitive impairment assessment. Does NOT diagnose dementia.
    Dementia requires: cognitive impairment + functional decline + chronicity.

    MoCA scoring (Nasreddine et al., 2005):
    - ≥26: Normal (add 1 point if education ≤12 years)
    - 18-25: Mild cognitive impairment range
    - <18: Significant cognitive impairment range

    Args:
        moca_score (int): MoCA total score (0-30)
        education_years (int, optional): Years of formal education

    Returns:
        dict: Interpretation including education-corrected score
    \"\"\"
    if not isinstance(moca_score, (int, float)):
        return {
            "category": "Unable to determine",
            "interpretation": "Invalid MoCA score provided",
            "cognitive_impairment_level": None
        }

    score = float(moca_score)

    if score < 0 or score > 30:
        return {
            "category": "Invalid score",
            "interpretation": "MoCA score must be between 0-30",
            "cognitive_impairment_level": None
        }

    # Education correction
    corrected_score = score
    education_adjustment = False
    if education_years is not None and education_years <= 12 and score < 30:
        corrected_score = min(score + 1, 30)
        education_adjustment = True

    # Interpretation
    if corrected_score >= 26:
        category = "Normal Cognition"
        interpretation = "MoCA ≥26 suggests normal cognitive function."
        impairment_level = "no_impairment_detected"
        functional_note = "If cognitive symptoms present despite normal score, consider functional assessment and alternative testing."
    elif corrected_score >= 18:
        category = "Mild Cognitive Impairment Range"
        interpretation = "MoCA 18-25 indicates mild cognitive impairment. May represent MCI (if functionally independent) or mild dementia (if functional decline present)."
        impairment_level = "mild_impairment_detected"
        functional_note = "Functional assessment required to distinguish MCI from mild dementia."
    else:
        category = "Significant Cognitive Impairment Range"
        interpretation = "MoCA <18 indicates significant cognitive impairment across multiple domains. IF functional decline present, consistent with dementia. IF functionally independent, may represent severe MCI."
        impairment_level = "significant_impairment_detected"
        functional_note = "Assess functional status. Dementia requires functional decline interfering with independence."

    return {
        "raw_score": score,
        "corrected_score": corrected_score if education_adjustment else score,
        "education_adjustment_applied": education_adjustment,
        "category": category,
        "interpretation": interpretation,
        "cognitive_impairment_level": impairment_level,
        "functional_assessment_note": functional_note,
        "diagnostic_context": "MoCA score indicates cognitive status. Dementia diagnosis requires: cognitive impairment + functional decline + chronicity (≥6 months) + exclusion of delirium.",
        "note": "MoCA is more sensitive than MMSE for detecting MCI and early dementia, especially executive dysfunction."
    }
""",
    "description": "Interpret MoCA score - v1.0.0 NO boolean flags, requires functional assessment",
    "parameters": {
        "moca_score": {"type": "number", "description": "MoCA total score (0-30)", "required": True},
        "education_years": {"type": "number", "description": "Years of formal education (optional, adds 1 point if ≤12 years)", "required": False}
    },
    "returns": "dict with cognitive_impairment_level, interpretation, and functional assessment note",
    "enabled": True,
    "signature": "(moca_score, education_years=None)"
}

# Fix Vascular Risk
vascular_fix = {
    "name": "calculate_vascular_risk_score",
    "code": """def calculate_vascular_risk_score(hypertension=False, diabetes=False, hyperlipidemia=False,
                                  smoking=False, atrial_fib=False, stroke_history=False,
                                  cad=False):
    \"\"\"
    Calculate vascular risk factor burden

    IMPORTANT: Risk factors do NOT diagnose vascular dementia.
    Vascular dementia requires: (1) temporal relationship with stroke/TIA,
    (2) significant cerebrovascular disease on imaging, (3) stepwise cognitive decline.

    Args:
        hypertension (bool): History of hypertension
        diabetes (bool): History of diabetes mellitus
        hyperlipidemia (bool): History of high cholesterol
        smoking (bool): Current or former smoker
        atrial_fib (bool): Atrial fibrillation
        stroke_history (bool): Previous stroke or TIA
        cad (bool): Coronary artery disease

    Returns:
        dict: Risk factor assessment
    \"\"\"
    risk_factors = {
        "Hypertension": hypertension,
        "Diabetes": diabetes,
        "Hyperlipidemia": hyperlipidemia,
        "Smoking": smoking,
        "Atrial Fibrillation": atrial_fib,
        "Stroke/TIA History": stroke_history,
        "Coronary Artery Disease": cad
    }

    present_factors = [factor for factor, present in risk_factors.items() if present]
    risk_count = len(present_factors)

    if risk_count == 0:
        risk_level = "Low"
        interpretation = "No major vascular risk factors identified. Vascular contribution to cognitive impairment less likely."
    elif risk_count <= 2:
        risk_level = "Moderate"
        interpretation = f"{risk_count} vascular risk factor(s) present. Moderate vascular risk burden."
    else:
        risk_level = "High"
        interpretation = f"{risk_count} vascular risk factors present. High vascular risk burden."

    return {
        "vascular_risk_count": risk_count,
        "risk_level": risk_level,
        "present_risk_factors": present_factors,
        "interpretation": interpretation,
        "vascular_dementia_criteria": "For vascular dementia diagnosis, MUST have: (1) temporal relationship with stroke/TIA, (2) significant cerebrovascular disease on imaging (infarcts, white matter disease), (3) stepwise or fluctuating cognitive decline. Risk factors alone do NOT establish diagnosis.",
        "note": "Vascular risk factors increase probability of both vascular dementia AND Alzheimer's disease. Many patients have mixed pathology. Imaging and clinical course required for diagnosis."
    }
""",
    "description": "Calculate vascular risk factor burden - v1.0.0 emphasizes risk ≠ diagnosis",
    "parameters": {
        "hypertension": {"type": "boolean", "description": "History of hypertension", "required": False},
        "diabetes": {"type": "boolean", "description": "History of diabetes", "required": False},
        "hyperlipidemia": {"type": "boolean", "description": "History of high cholesterol", "required": False},
        "smoking": {"type": "boolean", "description": "Current/former smoker", "required": False},
        "atrial_fib": {"type": "boolean", "description": "Atrial fibrillation", "required": False},
        "stroke_history": {"type": "boolean", "description": "Previous stroke/TIA", "required": False},
        "cad": {"type": "boolean", "description": "Coronary artery disease", "required": False}
    },
    "returns": "dict with risk assessment and vascular dementia diagnostic criteria",
    "enabled": True,
    "signature": "(hypertension=False, diabetes=False, hyperlipidemia=False, smoking=False, atrial_fib=False, stroke_history=False, cad=False)"
}

# Fix Pediatric Nutrition Status
pediatric_fix = {
    "name": "calculate_pediatric_nutrition_status",
    "code": """def calculate_pediatric_nutrition_status(weight_kg, height_cm, age_months, sex):
    '''
    Calculate comprehensive pediatric growth assessment using CDC/WHO z-scores

    IMPORTANT: Returns growth parameters for assessment. Does NOT diagnose malnutrition.
    ASPEN malnutrition diagnosis requires ≥2 indicators.

    Args:
        weight_kg: Weight in kilograms
        height_cm: Height in centimeters
        age_months: Age in months
        sex: 'male' or 'female'

    Returns:
        Dictionary with z-scores, BMI, and WHO growth assessment categories
    '''
    # Calculate BMI first
    height_m = height_cm / 100.0
    bmi = call_function('calculate_bmi', weight_kg=weight_kg, height_m=height_m)

    # Calculate z-scores
    weight_zscore = call_function('calculate_zscore',
                                   measurement='weight',
                                   value=weight_kg,
                                   age_months=age_months,
                                   sex=sex)

    height_zscore = call_function('calculate_zscore',
                                   measurement='height',
                                   value=height_cm,
                                   age_months=age_months,
                                   sex=sex)

    bmi_zscore = call_function('calculate_zscore',
                                measurement='bmi',
                                value=bmi,
                                age_months=age_months,
                                sex=sex)

    # Growth assessment categories (descriptive, not diagnostic)
    def assess_wasting(bmi_z):
        if bmi_z < -3:
            return 'Severe growth deficit (wasting pattern)'
        elif bmi_z < -2:
            return 'Moderate growth deficit (wasting pattern)'
        elif bmi_z < -1:
            return 'Mild growth deficit (at-risk)'
        else:
            return 'Normal BMI-for-age'

    def assess_stunting(height_z):
        if height_z < -3:
            return 'Severe height deficit (stunting pattern)'
        elif height_z < -2:
            return 'Moderate height deficit (stunting pattern)'
        elif height_z < -1:
            return 'Mild height deficit (at-risk)'
        else:
            return 'Normal height-for-age'

    def assess_underweight(weight_z):
        if weight_z < -3:
            return 'Severe weight deficit'
        elif weight_z < -2:
            return 'Moderate weight deficit'
        elif weight_z < -1:
            return 'Mild weight deficit (at-risk)'
        else:
            return 'Normal weight-for-age'

    return {
        'bmi': bmi,
        'weight_zscore': weight_zscore,
        'height_zscore': height_zscore,
        'bmi_zscore': bmi_zscore,
        'wasting_assessment': assess_wasting(bmi_zscore),
        'stunting_assessment': assess_stunting(height_zscore),
        'underweight_assessment': assess_underweight(weight_zscore),
        'age_months': age_months,
        'sex': sex,
        'aspen_note': 'ASPEN malnutrition diagnosis requires ≥2 indicators: (1) low z-score, (2) inadequate intake, (3) weight loss/deceleration, (4) muscle/fat loss, or (5) edema. Z-scores alone do NOT diagnose malnutrition.'
    }
""",
    "description": "Calculate pediatric growth assessment - v1.0.0 descriptive categories, NOT diagnostic labels",
    "parameters": {
        "weight_kg": {"type": "number", "description": "Weight in kilograms", "required": True},
        "height_cm": {"type": "number", "description": "Height in centimeters", "required": True},
        "age_months": {"type": "number", "description": "Age in months", "required": True},
        "sex": {"type": "string", "description": "Sex: 'male' or 'female'", "required": True}
    },
    "returns": "Dictionary with z-scores, BMI, and growth assessment categories (NOT diagnoses) + ASPEN note",
    "enabled": True,
    "signature": "(weight_kg, height_cm, age_months, sex)"
}

# Apply fixes
def apply_fixes():
    for fix in [moca_fix, vascular_fix, pediatric_fix]:
        file_path = FUNCTIONS_DIR / f"{fix['name']}.json"
        with open(file_path, 'w') as f:
            json.dump(fix, f, indent=2)
        print(f"✓ Fixed: {fix['name']}.json")

if __name__ == "__main__":
    print("Applying ALL remaining function fixes...")
    apply_fixes()
    print("All critical function fixes applied!")
