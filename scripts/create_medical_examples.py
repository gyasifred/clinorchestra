#!/usr/bin/env python3
"""
Script to create comprehensive medical/clinical examples:
- Functions for medical calculations
- Patterns for text normalization
- Extras for task-specific hints

Run this to populate the platform with medical domain knowledge.
"""

import json
import sys
import os
from pathlib import Path

# Add parent to path
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

# Change to parent directory for proper paths
os.chdir(parent_dir)

from core.function_registry import FunctionRegistry
from core.extras_manager import ExtrasManager


# ============================================================================
# MEDICAL FUNCTIONS (20+ examples)
# ============================================================================

MEDICAL_FUNCTIONS = [
    {
        "name": "calculate_bmi",
        "code": """def calculate_bmi(weight_kg, height_m):
    '''Calculate BMI from weight and height'''
    if height_m <= 0:
        return None
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 2)""",
        "description": "Calculate Body Mass Index from weight (kg) and height (m)",
        "parameters": {
            "weight_kg": {"type": "number", "description": "Weight in kilograms"},
            "height_m": {"type": "number", "description": "Height in meters"}
        },
        "return_type": "number"
    },
    {
        "name": "kg_to_lbs",
        "code": """def kg_to_lbs(kg):
    '''Convert kilograms to pounds'''
    return round(kg * 2.20462, 2)""",
        "description": "Convert weight from kilograms to pounds",
        "parameters": {
            "kg": {"type": "number", "description": "Weight in kilograms"}
        },
        "return_type": "number"
    },
    {
        "name": "lbs_to_kg",
        "code": """def lbs_to_kg(lbs):
    '''Convert pounds to kilograms'''
    return round(lbs / 2.20462, 2)""",
        "description": "Convert weight from pounds to kilograms",
        "parameters": {
            "lbs": {"type": "number", "description": "Weight in pounds"}
        },
        "return_type": "number"
    },
    {
        "name": "cm_to_m",
        "code": """def cm_to_m(cm):
    '''Convert centimeters to meters'''
    return round(cm / 100, 3)""",
        "description": "Convert height from centimeters to meters",
        "parameters": {
            "cm": {"type": "number", "description": "Height in centimeters"}
        },
        "return_type": "number"
    },
    {
        "name": "inches_to_cm",
        "code": """def inches_to_cm(inches):
    '''Convert inches to centimeters'''
    return round(inches * 2.54, 2)""",
        "description": "Convert height from inches to centimeters",
        "parameters": {
            "inches": {"type": "number", "description": "Height in inches"}
        },
        "return_type": "number"
    },
    {
        "name": "calculate_weight_change_percent",
        "code": """def calculate_weight_change_percent(initial_weight, final_weight):
    '''Calculate percentage weight change'''
    if initial_weight == 0:
        return None
    change = ((final_weight - initial_weight) / initial_weight) * 100
    return round(change, 2)""",
        "description": "Calculate percentage weight change between two timepoints",
        "parameters": {
            "initial_weight": {"type": "number", "description": "Initial weight in any unit"},
            "final_weight": {"type": "number", "description": "Final weight in same unit"}
        },
        "return_type": "number"
    },
    {
        "name": "calculate_weight_change_absolute",
        "code": """def calculate_weight_change_absolute(initial_weight, final_weight):
    '''Calculate absolute weight change'''
    change = final_weight - initial_weight
    return round(change, 2)""",
        "description": "Calculate absolute weight change between two timepoints",
        "parameters": {
            "initial_weight": {"type": "number", "description": "Initial weight in any unit"},
            "final_weight": {"type": "number", "description": "Final weight in same unit"}
        },
        "return_type": "number"
    },
    {
        "name": "calculate_mean_arterial_pressure",
        "code": """def calculate_mean_arterial_pressure(systolic, diastolic):
    '''Calculate MAP from BP readings'''
    map_value = (systolic + 2 * diastolic) / 3
    return round(map_value, 1)""",
        "description": "Calculate Mean Arterial Pressure from systolic and diastolic BP",
        "parameters": {
            "systolic": {"type": "number", "description": "Systolic blood pressure (mmHg)"},
            "diastolic": {"type": "number", "description": "Diastolic blood pressure (mmHg)"}
        },
        "return_type": "number"
    },
    {
        "name": "calculate_body_surface_area",
        "code": """def calculate_body_surface_area(weight_kg, height_cm):
    '''Calculate BSA using Mosteller formula'''
    import math
    bsa = math.sqrt((weight_kg * height_cm) / 3600)
    return round(bsa, 3)""",
        "description": "Calculate Body Surface Area using Mosteller formula",
        "parameters": {
            "weight_kg": {"type": "number", "description": "Weight in kilograms"},
            "height_cm": {"type": "number", "description": "Height in centimeters"}
        },
        "return_type": "number"
    },
    {
        "name": "calculate_ideal_body_weight",
        "code": """def calculate_ideal_body_weight(height_cm, sex):
    '''Calculate IBW using Devine formula'''
    height_inches = height_cm / 2.54
    if sex.lower() in ['male', 'm']:
        ibw_kg = 50 + 2.3 * (height_inches - 60)
    elif sex.lower() in ['female', 'f']:
        ibw_kg = 45.5 + 2.3 * (height_inches - 60)
    else:
        return None
    return round(ibw_kg, 1)""",
        "description": "Calculate Ideal Body Weight using Devine formula",
        "parameters": {
            "height_cm": {"type": "number", "description": "Height in centimeters"},
            "sex": {"type": "string", "description": "Sex: 'male' or 'female'"}
        },
        "return_type": "number"
    },
    {
        "name": "calculate_anion_gap",
        "code": """def calculate_anion_gap(sodium, chloride, bicarbonate):
    '''Calculate serum anion gap'''
    ag = sodium - (chloride + bicarbonate)
    return round(ag, 1)""",
        "description": "Calculate serum anion gap",
        "parameters": {
            "sodium": {"type": "number", "description": "Serum sodium (mEq/L)"},
            "chloride": {"type": "number", "description": "Serum chloride (mEq/L)"},
            "bicarbonate": {"type": "number", "description": "Serum bicarbonate (mEq/L)"}
        },
        "return_type": "number"
    },
    {
        "name": "calculate_corrected_calcium",
        "code": """def calculate_corrected_calcium(calcium, albumin):
    '''Calculate corrected calcium for hypoalbuminemia'''
    corrected_ca = calcium + 0.8 * (4.0 - albumin)
    return round(corrected_ca, 2)""",
        "description": "Calculate corrected calcium for low albumin",
        "parameters": {
            "calcium": {"type": "number", "description": "Serum calcium (mg/dL)"},
            "albumin": {"type": "number", "description": "Serum albumin (g/dL)"}
        },
        "return_type": "number"
    },
    {
        "name": "calculate_creatinine_clearance",
        "code": """def calculate_creatinine_clearance(age, weight_kg, creatinine, sex):
    '''Calculate CrCl using Cockcroft-Gault'''
    if sex.lower() in ['male', 'm']:
        crcl = ((140 - age) * weight_kg) / (72 * creatinine)
    elif sex.lower() in ['female', 'f']:
        crcl = ((140 - age) * weight_kg) / (72 * creatinine) * 0.85
    else:
        return None
    return round(crcl, 1)""",
        "description": "Calculate creatinine clearance using Cockcroft-Gault equation",
        "parameters": {
            "age": {"type": "number", "description": "Age in years"},
            "weight_kg": {"type": "number", "description": "Weight in kilograms"},
            "creatinine": {"type": "number", "description": "Serum creatinine (mg/dL)"},
            "sex": {"type": "string", "description": "Sex: 'male' or 'female'"}
        },
        "return_type": "number"
    },
    {
        "name": "calculate_fluid_requirement",
        "code": """def calculate_fluid_requirement(weight_kg):
    '''Calculate daily maintenance fluid requirement (Holliday-Segar)'''
    if weight_kg <= 10:
        fluid = weight_kg * 100
    elif weight_kg <= 20:
        fluid = 1000 + (weight_kg - 10) * 50
    else:
        fluid = 1500 + (weight_kg - 20) * 20
    return round(fluid, 0)""",
        "description": "Calculate daily maintenance fluid requirement using Holliday-Segar method",
        "parameters": {
            "weight_kg": {"type": "number", "description": "Weight in kilograms"}
        },
        "return_type": "number"
    },
    {
        "name": "calculate_calorie_requirement",
        "code": """def calculate_calorie_requirement(weight_kg, activity_level='moderate'):
    '''Calculate daily calorie requirement'''
    base = weight_kg * 25
    multipliers = {'sedentary': 1.2, 'light': 1.375, 'moderate': 1.55, 'active': 1.725, 'very_active': 1.9}
    multiplier = multipliers.get(activity_level, 1.55)
    calories = base * multiplier
    return round(calories, 0)""",
        "description": "Calculate daily calorie requirement based on weight and activity",
        "parameters": {
            "weight_kg": {"type": "number", "description": "Weight in kilograms"},
            "activity_level": {"type": "string", "description": "Activity level: sedentary/light/moderate/active/very_active"}
        },
        "return_type": "number"
    },
    {
        "name": "calculate_protein_requirement",
        "code": """def calculate_protein_requirement(weight_kg, condition='normal'):
    '''Calculate daily protein requirement'''
    requirements = {'normal': 0.8, 'elderly': 1.0, 'athlete': 1.6, 'illness': 1.5, 'critical': 2.0}
    g_per_kg = requirements.get(condition, 0.8)
    protein_g = weight_kg * g_per_kg
    return round(protein_g, 1)""",
        "description": "Calculate daily protein requirement based on weight and clinical condition",
        "parameters": {
            "weight_kg": {"type": "number", "description": "Weight in kilograms"},
            "condition": {"type": "string", "description": "Condition: normal/elderly/athlete/illness/critical"}
        },
        "return_type": "number"
    },
    {
        "name": "calculate_glomerular_filtration_rate",
        "code": """def calculate_glomerular_filtration_rate(creatinine, age, sex, race='non_african_american'):
    '''Calculate eGFR using CKD-EPI equation (simplified)'''
    if sex.lower() in ['female', 'f']:
        if creatinine <= 0.7:
            gfr = 144 * (creatinine / 0.7) ** -0.329 * 0.993 ** age
        else:
            gfr = 144 * (creatinine / 0.7) ** -1.209 * 0.993 ** age
    else:
        if creatinine <= 0.9:
            gfr = 141 * (creatinine / 0.9) ** -0.411 * 0.993 ** age
        else:
            gfr = 141 * (creatinine / 0.9) ** -1.209 * 0.993 ** age
    if race.lower() in ['african_american', 'black']:
        gfr = gfr * 1.159
    return round(gfr, 1)""",
        "description": "Calculate estimated GFR using CKD-EPI equation",
        "parameters": {
            "creatinine": {"type": "number", "description": "Serum creatinine (mg/dL)"},
            "age": {"type": "number", "description": "Age in years"},
            "sex": {"type": "string", "description": "Sex: 'male' or 'female'"},
            "race": {"type": "string", "description": "Race: 'african_american' or 'non_african_american'"}
        },
        "return_type": "number"
    },
    {
        "name": "calculate_osmolality",
        "code": """def calculate_osmolality(sodium, glucose, bun):
    '''Calculate serum osmolality'''
    osm = 2 * sodium + (glucose / 18) + (bun / 2.8)
    return round(osm, 1)""",
        "description": "Calculate serum osmolality from electrolytes",
        "parameters": {
            "sodium": {"type": "number", "description": "Serum sodium (mEq/L)"},
            "glucose": {"type": "number", "description": "Serum glucose (mg/dL)"},
            "bun": {"type": "number", "description": "Blood urea nitrogen (mg/dL)"}
        },
        "return_type": "number"
    },
    {
        "name": "calculate_qtc_interval",
        "code": """def calculate_qtc_interval(qt_ms, rr_sec):
    '''Calculate corrected QT interval using Bazett formula'''
    import math
    qtc = qt_ms / math.sqrt(rr_sec)
    return round(qtc, 0)""",
        "description": "Calculate corrected QT interval (Bazett formula)",
        "parameters": {
            "qt_ms": {"type": "number", "description": "QT interval in milliseconds"},
            "rr_sec": {"type": "number", "description": "RR interval in seconds"}
        },
        "return_type": "number"
    },
    {
        "name": "calculate_pack_years",
        "code": """def calculate_pack_years(packs_per_day, years_smoked):
    '''Calculate smoking pack-years'''
    pack_years = packs_per_day * years_smoked
    return round(pack_years, 1)""",
        "description": "Calculate smoking history in pack-years",
        "parameters": {
            "packs_per_day": {"type": "number", "description": "Average packs smoked per day"},
            "years_smoked": {"type": "number", "description": "Number of years smoking"}
        },
        "return_type": "number"
    }
]


# ============================================================================
# MEDICAL PATTERNS (30+ examples)
# ============================================================================

MEDICAL_PATTERNS = [
    {
        "name": "normalize_blood_pressure",
        "pattern": r"BP:?\s*(\d{2,3})/(\d{2,3})",
        "replacement": r"Blood pressure \1/\2 mmHg",
        "description": "Normalize blood pressure format",
        "enabled": True
    },
    {
        "name": "normalize_temperature_f",
        "pattern": r"(\d{2,3}\.?\d*)\s*°?F",
        "replacement": r"\1°F",
        "description": "Normalize temperature in Fahrenheit",
        "enabled": True
    },
    {
        "name": "normalize_temperature_c",
        "pattern": r"(\d{2}\.?\d*)\s*°?C",
        "replacement": r"\1°C",
        "description": "Normalize temperature in Celsius",
        "enabled": True
    },
    {
        "name": "normalize_heart_rate",
        "pattern": r"HR:?\s*(\d{2,3})\s*b?pm?",
        "replacement": r"heart rate \1 bpm",
        "description": "Normalize heart rate format",
        "enabled": True
    },
    {
        "name": "normalize_respiratory_rate",
        "pattern": r"RR:?\s*(\d{1,2})",
        "replacement": r"respiratory rate \1 per minute",
        "description": "Normalize respiratory rate",
        "enabled": True
    },
    {
        "name": "normalize_oxygen_saturation",
        "pattern": r"O2\s*sat:?\s*(\d{2,3})%?",
        "replacement": r"oxygen saturation \1%",
        "description": "Normalize oxygen saturation",
        "enabled": True
    },
    {
        "name": "normalize_weight_kg",
        "pattern": r"wt:?\s*(\d{1,3}\.?\d*)\s*kg",
        "replacement": r"weight \1kg",
        "description": "Normalize weight in kilograms",
        "enabled": True
    },
    {
        "name": "normalize_weight_lbs",
        "pattern": r"wt:?\s*(\d{1,3}\.?\d*)\s*(?:lbs?|pounds?)",
        "replacement": r"weight \1 lbs",
        "description": "Normalize weight in pounds",
        "enabled": True
    },
    {
        "name": "normalize_height_cm",
        "pattern": r"ht:?\s*(\d{2,3})\s*cm",
        "replacement": r"height \1cm",
        "description": "Normalize height in centimeters",
        "enabled": True
    },
    {
        "name": "normalize_glucose",
        "pattern": r"(?:glucose|BG):?\s*(\d{2,4})\s*mg/dL",
        "replacement": r"glucose \1 mg/dL",
        "description": "Normalize blood glucose",
        "enabled": True
    },
    {
        "name": "normalize_hba1c",
        "pattern": r"HbA1[Cc]:?\s*(\d{1,2}\.?\d*)%?",
        "replacement": r"HbA1c \1%",
        "description": "Normalize HbA1c format",
        "enabled": True
    },
    {
        "name": "normalize_creatinine",
        "pattern": r"(?:creat|Cr):?\s*(\d{1}\.?\d*)\s*mg/dL",
        "replacement": r"creatinine \1 mg/dL",
        "description": "Normalize creatinine",
        "enabled": True
    },
    {
        "name": "normalize_sodium",
        "pattern": r"Na:?\s*(\d{2,3})\s*(?:mEq/L|mmol/L)?",
        "replacement": r"sodium \1 mEq/L",
        "description": "Normalize sodium level",
        "enabled": True
    },
    {
        "name": "normalize_potassium",
        "pattern": r"K:?\s*(\d{1}\.?\d*)\s*(?:mEq/L|mmol/L)?",
        "replacement": r"potassium \1 mEq/L",
        "description": "Normalize potassium level",
        "enabled": True
    },
    {
        "name": "normalize_hemoglobin",
        "pattern": r"(?:Hgb|Hb):?\s*(\d{1,2}\.?\d*)\s*g/dL",
        "replacement": r"hemoglobin \1 g/dL",
        "description": "Normalize hemoglobin",
        "enabled": True
    },
    {
        "name": "normalize_wbc",
        "pattern": r"WBC:?\s*(\d{1,2}\.?\d*)\s*(?:k/uL|×10³/µL)?",
        "replacement": r"white blood cell count \1 k/uL",
        "description": "Normalize WBC count",
        "enabled": True
    },
    {
        "name": "normalize_platelets",
        "pattern": r"(?:PLT|platelets):?\s*(\d{2,3})\s*(?:k/uL|×10³/µL)?",
        "replacement": r"platelet count \1 k/uL",
        "description": "Normalize platelet count",
        "enabled": True
    },
    {
        "name": "normalize_inr",
        "pattern": r"INR:?\s*(\d{1}\.?\d*)",
        "replacement": r"INR \1",
        "description": "Normalize INR format",
        "enabled": True
    },
    {
        "name": "normalize_medication_dose_mg",
        "pattern": r"(\w+)\s+(\d+)\s*mg\b",
        "replacement": r"\1 \2mg",
        "description": "Normalize medication dosing in mg",
        "enabled": True
    },
    {
        "name": "normalize_medication_dose_mcg",
        "pattern": r"(\w+)\s+(\d+)\s*mcg\b",
        "replacement": r"\1 \2mcg",
        "description": "Normalize medication dosing in mcg",
        "enabled": True
    },
    {
        "name": "normalize_frequency_bid",
        "pattern": r"\b(?:BID|bid|b\.i\.d\.)\b",
        "replacement": r"twice daily",
        "description": "Normalize BID to twice daily",
        "enabled": True
    },
    {
        "name": "normalize_frequency_tid",
        "pattern": r"\b(?:TID|tid|t\.i\.d\.)\b",
        "replacement": r"three times daily",
        "description": "Normalize TID to three times daily",
        "enabled": True
    },
    {
        "name": "normalize_frequency_qid",
        "pattern": r"\b(?:QID|qid|q\.i\.d\.)\b",
        "replacement": r"four times daily",
        "description": "Normalize QID to four times daily",
        "enabled": True
    },
    {
        "name": "normalize_frequency_qd",
        "pattern": r"\b(?:QD|qd|q\.d\.)\b",
        "replacement": r"once daily",
        "description": "Normalize QD to once daily",
        "enabled": True
    },
    {
        "name": "normalize_route_po",
        "pattern": r"\b(?:PO|p\.o\.)\b",
        "replacement": r"by mouth",
        "description": "Normalize PO to by mouth",
        "enabled": True
    },
    {
        "name": "normalize_route_iv",
        "pattern": r"\b(?:IV|i\.v\.)\b",
        "replacement": r"intravenously",
        "description": "Normalize IV to intravenously",
        "enabled": True
    },
    {
        "name": "normalize_route_im",
        "pattern": r"\b(?:IM|i\.m\.)\b",
        "replacement": r"intramuscularly",
        "description": "Normalize IM to intramuscularly",
        "enabled": True
    },
    {
        "name": "normalize_route_sc",
        "pattern": r"\b(?:SC|SQ|s\.c\.|subcutaneous)\b",
        "replacement": r"subcutaneously",
        "description": "Normalize SC/SQ to subcutaneously",
        "enabled": True
    },
    {
        "name": "normalize_diagnosis_dm",
        "pattern": r"\bDM\b",
        "replacement": r"diabetes mellitus",
        "description": "Expand DM to diabetes mellitus",
        "enabled": True
    },
    {
        "name": "normalize_diagnosis_htn",
        "pattern": r"\bHTN\b",
        "replacement": r"hypertension",
        "description": "Expand HTN to hypertension",
        "enabled": True
    },
    {
        "name": "normalize_diagnosis_chf",
        "pattern": r"\bCHF\b",
        "replacement": r"congestive heart failure",
        "description": "Expand CHF to congestive heart failure",
        "enabled": True
    },
    {
        "name": "normalize_diagnosis_copd",
        "pattern": r"\bCOPD\b",
        "replacement": r"chronic obstructive pulmonary disease",
        "description": "Expand COPD abbreviation",
        "enabled": True
    },
    {
        "name": "normalize_diagnosis_ckd",
        "pattern": r"\bCKD\b",
        "replacement": r"chronic kidney disease",
        "description": "Expand CKD abbreviation",
        "enabled": True
    }
]


# ============================================================================
# CLINICAL EXTRAS/HINTS (50+ examples)
# ============================================================================

CLINICAL_EXTRAS = [
    {
        "name": "WHO Growth Standards",
        "type": "guideline",
        "content": "WHO Child Growth Standards (0-5 years): Weight-for-age, length/height-for-age, weight-for-length/height, and BMI-for-age. Z-scores classify nutritional status: <-3 SD severely wasted, <-2 SD wasted, -2 to +1 SD normal, >+1 SD possible risk of overweight, >+2 SD overweight, >+3 SD obese.",
        "metadata": {"category": "pediatrics", "subcategory": "growth", "priority": "high"}
    },
    {
        "name": "CDC Growth Charts",
        "type": "guideline",
        "content": "CDC Growth Charts (2-20 years): Weight-for-age, stature-for-age, BMI-for-age. Percentiles classify: <5th underweight, 5-84th healthy weight, 85-94th overweight, ≥95th obese. Z-scores: -2 SD corresponds to ~2nd percentile, +2 SD to ~98th percentile.",
        "metadata": {"category": "pediatrics", "subcategory": "growth", "priority": "high"}
    },
    {
        "name": "ASPEN Pediatric Malnutrition Criteria",
        "type": "criteria",
        "content": "ASPEN Pediatric Malnutrition: Requires ≥2 of: 1) Insufficient energy intake, 2) Weight loss or deceleration, 3) Loss of muscle/subcutaneous fat, 4) Edema. Severity: Mild (1-2 SD), Moderate (2-3 SD), Severe (>3 SD below mean for anthropometrics). Context-dependent assessment for children with chronic conditions.",
        "metadata": {"category": "pediatrics", "subcategory": "malnutrition", "priority": "high"}
    },
    {
        "name": "Z-Score Interpretation",
        "type": "tip",
        "content": "Z-scores represent standard deviations from population mean. Negative z-scores are below average (e.g., -2.0 = 2 SD below mean, ~2nd percentile). Positive z-scores are above average (e.g., +1.5 = 1.5 SD above mean, ~93rd percentile). For percentiles <50th, expect negative z-scores; for >50th, expect positive z-scores.",
        "metadata": {"category": "statistics", "subcategory": "interpretation", "priority": "high"}
    },
    {
        "name": "Percentile to Z-Score Conversion",
        "type": "reference",
        "content": "Common conversions: 2nd%ile ≈ -2.0 SD, 5th%ile ≈ -1.64 SD, 10th%ile ≈ -1.28 SD, 25th%ile ≈ -0.67 SD, 50th%ile = 0 SD, 75th%ile ≈ +0.67 SD, 90th%ile ≈ +1.28 SD, 95th%ile ≈ +1.64 SD, 98th%ile ≈ +2.0 SD.",
        "metadata": {"category": "statistics", "subcategory": "conversion", "priority": "high"}
    },
    {
        "name": "Growth Velocity Assessment",
        "type": "tip",
        "content": "Growth velocity = change in measurement over time. Calculate: (final - initial) / time period. Express as g/day for weight, cm/month for height. Compare to expected velocity for age. Deceleration (crossing percentiles downward) is concerning even if absolute values are normal.",
        "metadata": {"category": "pediatrics", "subcategory": "growth", "priority": "medium"}
    },
    {
        "name": "Malnutrition Etiology Classification",
        "type": "criteria",
        "content": "Illness-related malnutrition: Associated with acute/chronic disease, inflammation. Non-illness-related: Caused by environmental factors, food insecurity, neglect. Socio-environmental: Social determinants of health. Mixed etiology: Combination of factors. Document primary driver.",
        "metadata": {"category": "malnutrition", "subcategory": "etiology", "priority": "medium"}
    },
    {
        "name": "Temporal Data Capture",
        "type": "pattern",
        "content": "Always capture dates with measurements: 'Weight 12.5kg on 1/15/25 (25th%ile, z-score -0.7)'. Calculate trends: absolute change, percentage change, rate of change, velocity, percentile trajectory. Identify assessment type: single-point, serial same-encounter, longitudinal multi-encounter.",
        "metadata": {"category": "documentation", "subcategory": "temporal", "priority": "high"}
    },
    {
        "name": "Normal Vital Sign Ranges - Pediatric",
        "type": "reference",
        "content": "Heart rate (bpm): Newborn 120-160, Infant 100-160, Toddler 90-150, Preschool 80-140, School-age 70-120, Adolescent 60-100. Respiratory rate: Newborn 30-60, Infant 24-40, Toddler 20-30, Preschool 20-25, School-age 18-22, Adolescent 12-20. BP increases with age; use age-specific percentiles.",
        "metadata": {"category": "pediatrics", "subcategory": "vitals", "priority": "medium"}
    },
    {
        "name": "Normal Vital Sign Ranges - Adult",
        "type": "reference",
        "content": "Heart rate: 60-100 bpm. Respiratory rate: 12-20/min. Blood pressure: <120/<80 normal, 120-129/<80 elevated, 130-139/80-89 Stage 1 HTN, ≥140/≥90 Stage 2 HTN. Temperature: 97-99°F (36.1-37.2°C) oral. SpO2: ≥95% normal.",
        "metadata": {"category": "vitals", "subcategory": "adults", "priority": "medium"}
    },
    {
        "name": "BMI Classification - Adults",
        "type": "criteria",
        "content": "BMI (kg/m²): <18.5 Underweight, 18.5-24.9 Normal weight, 25-29.9 Overweight, 30-34.9 Obesity Class I, 35-39.9 Obesity Class II, ≥40 Obesity Class III (severe). Limitations: Doesn't distinguish muscle vs fat, varies by ethnicity.",
        "metadata": {"category": "nutrition", "subcategory": "bmi", "priority": "high"}
    },
    {
        "name": "Laboratory Value Normal Ranges",
        "type": "reference",
        "content": "Sodium 136-145 mEq/L, Potassium 3.5-5.0 mEq/L, Chloride 98-106 mEq/L, Bicarbonate 22-29 mEq/L, BUN 7-20 mg/dL, Creatinine 0.6-1.2 mg/dL, Glucose 70-100 mg/dL (fasting), Calcium 8.5-10.5 mg/dL, Albumin 3.5-5.5 g/dL, Hemoglobin 13.5-17.5 g/dL (M), 12-16 g/dL (F).",
        "metadata": {"category": "laboratory", "subcategory": "chemistry", "priority": "high"}
    },
    {
        "name": "Anemia Classification",
        "type": "criteria",
        "content": "WHO criteria: Hemoglobin <13 g/dL (adult M), <12 g/dL (adult F), <11 g/dL (pregnant F), <11 g/dL (children 6mo-5yr), <11.5 g/dL (children 5-11yr), <12 g/dL (children 12-14yr). Severity: Mild 10-normal, Moderate 8-10, Severe <8 g/dL.",
        "metadata": {"category": "hematology", "subcategory": "anemia", "priority": "medium"}
    },
    {
        "name": "Diabetes Diagnostic Criteria",
        "type": "criteria",
        "content": "ADA Criteria: FPG ≥126 mg/dL, or 2-hr OGTT ≥200 mg/dL, or HbA1c ≥6.5%, or random glucose ≥200 mg/dL with symptoms. Prediabetes: FPG 100-125 mg/dL, 2-hr OGTT 140-199 mg/dL, or HbA1c 5.7-6.4%. Confirm with repeat testing unless symptomatic hyperglycemia.",
        "metadata": {"category": "endocrinology", "subcategory": "diabetes", "priority": "high"}
    },
    {
        "name": "HbA1c Target Goals",
        "type": "guideline",
        "content": "ADA targets: <7% for most adults, <6.5% if achieved safely without hypoglycemia, <8% for elderly/limited life expectancy/hypoglycemia risk/advanced complications. Pediatric: <7% (ADA), <7.5% (ISPAD). Pregnancy: <6% if safe, <7% otherwise. Individualize based on patient factors.",
        "metadata": {"category": "endocrinology", "subcategory": "diabetes_management", "priority": "medium"}
    },
    {
        "name": "Hypertension Diagnosis and Classification",
        "type": "criteria",
        "content": "ACC/AHA: Normal <120/<80, Elevated 120-129/<80, Stage 1 HTN 130-139/80-89, Stage 2 HTN ≥140/≥90. Confirm with multiple readings. Pediatric HTN: >95th percentile for age/sex/height. Target BP: <130/<80 for most adults, individualize for elderly and comorbidities.",
        "metadata": {"category": "cardiology", "subcategory": "hypertension", "priority": "medium"}
    },
    {
        "name": "CKD Staging by GFR",
        "type": "criteria",
        "content": "Stage 1: GFR ≥90 with kidney damage. Stage 2: GFR 60-89 with damage. Stage 3a: GFR 45-59. Stage 3b: GFR 30-44. Stage 4: GFR 15-29. Stage 5: GFR <15 or dialysis. Also classify by albuminuria: A1 <30 mg/g, A2 30-300, A3 >300 mg/g.",
        "metadata": {"category": "nephrology", "subcategory": "ckd", "priority": "medium"}
    },
    {
        "name": "Fluid Balance Calculation",
        "type": "pattern",
        "content": "Fluid balance = Intake - Output. Intake: oral fluids, IV fluids, tube feeds, medications. Output: urine, drains, emesis, diarrhea, insensible losses (~500-1000 mL/day). Positive balance suggests fluid retention, negative balance suggests dehydration. Monitor daily weights (1 kg ≈ 1 L fluid).",
        "metadata": {"category": "fluid_management", "subcategory": "balance", "priority": "medium"}
    },
    {
        "name": "Dehydration Assessment",
        "type": "criteria",
        "content": "Mild (3-5% weight loss): Slightly dry mucous membranes, normal vital signs. Moderate (6-9%): Dry mucous membranes, decreased skin turgor, tachycardia, decreased urine output. Severe (≥10%): Sunken eyes/fontanelle, poor perfusion, hypotension, lethargy. Infants and elderly at higher risk.",
        "metadata": {"category": "fluid_management", "subcategory": "dehydration", "priority": "medium"}
    },
    {
        "name": "Electrolyte Abnormalities - Hyponatremia",
        "type": "criteria",
        "content": "Mild 130-135 mEq/L, Moderate 125-129, Severe <125. Symptoms: headache, confusion, seizures (if acute/severe). Causes: SIADH, heart failure, cirrhosis, diuretics, polydipsia. Correction rate: ≤10-12 mEq/L per 24h to avoid osmotic demyelination syndrome.",
        "metadata": {"category": "laboratory", "subcategory": "electrolytes", "priority": "medium"}
    },
    {
        "name": "Electrolyte Abnormalities - Hyperkalemia",
        "type": "criteria",
        "content": "Mild 5.5-6.0 mEq/L, Moderate 6.1-7.0, Severe >7.0. ECG changes: peaked T waves, widened QRS, sine wave (severe). Causes: renal failure, K-sparing diuretics, ACE inhibitors, tissue breakdown. Treatment: Calcium gluconate (cardiac protection), insulin + glucose, albuterol, dialysis if severe.",
        "metadata": {"category": "laboratory", "subcategory": "electrolytes", "priority": "high"}
    },
    {
        "name": "Anticoagulation Management - Warfarin",
        "type": "guideline",
        "content": "Target INR: 2-3 for most indications (DVT, PE, AFib), 2.5-3.5 for mechanical valves. Monitor INR regularly. Bleeding risk increases with INR >4. Reversal: Vitamin K (oral/IV), PCC, FFP for severe bleeding. Drug-drug and drug-food interactions common - educate patients.",
        "metadata": {"category": "pharmacotherapy", "subcategory": "anticoagulation", "priority": "medium"}
    },
    {
        "name": "Pain Assessment Scales",
        "type": "reference",
        "content": "Numeric (0-10): 0=no pain, 1-3 mild, 4-6 moderate, 7-10 severe. Visual Analog Scale (VAS): Mark on line. FLACC (pediatric): Face, Legs, Activity, Cry, Consolability (0-10). Wong-Baker FACES (pediatric): Facial expressions (0-10). Reassess after interventions.",
        "metadata": {"category": "pain_management", "subcategory": "assessment", "priority": "medium"}
    },
    {
        "name": "Opioid Equianalgesic Dosing",
        "type": "reference",
        "content": "Approximate oral equivalents to morphine 30mg: Hydrocodone 30mg, Hydromorphone 6mg, Oxycodone 20mg, Tramadol 200mg. IV to PO conversion: Morphine 1:3 ratio. Fentanyl transdermal 12mcg/hr ≈ morphine 30mg/day oral. Reduce dose 25-50% when rotating opioids.",
        "metadata": {"category": "pain_management", "subcategory": "opioids", "priority": "medium"}
    },
    {
        "name": "Respiratory Distress Signs",
        "type": "criteria",
        "content": "Signs: Tachypnea, nasal flaring, retractions (intercostal, subcostal, suprasternal), grunting, head bobbing, cyanosis, altered mental status, inability to speak in full sentences. Assess oxygenation (SpO2), work of breathing, and air movement. Consider oxygen therapy, positioning, bronchodilators.",
        "metadata": {"category": "respiratory", "subcategory": "assessment", "priority": "high"}
    },
    {
        "name": "Asthma Severity Classification",
        "type": "criteria",
        "content": "Intermittent: Symptoms ≤2 days/week, nighttime ≤2x/month, no interference, FEV1 ≥80%. Mild persistent: >2 days/week but not daily. Moderate: Daily symptoms, nighttime >1x/week. Severe: Throughout day, often at night, extremely limited, FEV1 <60%. Step up therapy based on severity and control.",
        "metadata": {"category": "respiratory", "subcategory": "asthma", "priority": "medium"}
    },
    {
        "name": "COPD GOLD Staging",
        "type": "criteria",
        "content": "Based on post-bronchodilator FEV1: GOLD 1 (Mild) ≥80% predicted, GOLD 2 (Moderate) 50-79%, GOLD 3 (Severe) 30-49%, GOLD 4 (Very Severe) <30%. Also assess symptoms (mMRC/CAT) and exacerbation history. Combined ABCD assessment guides management.",
        "metadata": {"category": "respiratory", "subcategory": "copd", "priority": "medium"}
    },
    {
        "name": "Heart Failure NYHA Classification",
        "type": "criteria",
        "content": "Class I: No limitation, asymptomatic. Class II: Slight limitation, symptoms with ordinary activity. Class III: Marked limitation, symptoms with less than ordinary activity. Class IV: Unable to perform any activity without symptoms, symptoms at rest. Guides prognosis and treatment intensity.",
        "metadata": {"category": "cardiology", "subcategory": "heart_failure", "priority": "medium"}
    },
    {
        "name": "Shock Classification and Management",
        "type": "criteria",
        "content": "Hypovolemic: Low CVP, treat with fluids. Cardiogenic: High CVP, pulmonary edema, inotropes. Distributive (septic/anaphylactic): Warm extremities, wide pulse pressure, fluids + vasopressors. Obstructive: PE/tamponade, treat underlying cause. Assess perfusion, lactate, urine output.",
        "metadata": {"category": "critical_care", "subcategory": "shock", "priority": "high"}
    },
    {
        "name": "Sepsis Recognition - qSOFA",
        "type": "criteria",
        "content": "Quick SOFA (≥2 indicates sepsis risk): Respiratory rate ≥22/min, Altered mentation (GCS <15), Systolic BP ≤100 mmHg. SIRS criteria (≥2): Temp >38°C or <36°C, HR >90, RR >20 or PaCO2 <32, WBC >12k or <4k or >10% bands. Early recognition critical for outcomes.",
        "metadata": {"category": "infectious_disease", "subcategory": "sepsis", "priority": "high"}
    },
    {
        "name": "Glasgow Coma Scale",
        "type": "reference",
        "content": "Eye opening: 4 spontaneous, 3 to voice, 2 to pain, 1 none. Verbal: 5 oriented, 4 confused, 3 inappropriate words, 2 incomprehensible, 1 none. Motor: 6 obeys, 5 localizes, 4 withdraws, 3 flexion, 2 extension, 1 none. Total 3-15. ≤8 suggests severe injury, consider intubation.",
        "metadata": {"category": "neurology", "subcategory": "consciousness", "priority": "medium"}
    },
    {
        "name": "Stroke Assessment - NIHSS",
        "type": "reference",
        "content": "NIH Stroke Scale assesses: Level of consciousness, gaze, visual fields, facial palsy, motor arm/leg, limb ataxia, sensory, language, dysarthria, extinction/inattention. Score 0-42. >25 severe, 15-24 moderate-severe, 5-14 moderate, <5 minor. Helps triage for thrombolytics/thrombectomy.",
        "metadata": {"category": "neurology", "subcategory": "stroke", "priority": "medium"}
    },
    {
        "name": "Liver Function Interpretation",
        "type": "pattern",
        "content": "Hepatocellular injury: Elevated AST/ALT (ALT>AST usually, except alcoholic). Cholestatic: Elevated Alk Phos, GGT, bilirubin. Synthetic function: Albumin, INR/PT. AST/ALT >1000 suggests acute injury (toxin, ischemia, viral). Chronic: elevated Alk Phos suggests biliary obstruction.",
        "metadata": {"category": "hepatology", "subcategory": "lab_interpretation", "priority": "medium"}
    },
    {
        "name": "Child-Pugh Score for Cirrhosis",
        "type": "criteria",
        "content": "Assess: Bilirubin, Albumin, INR, Ascites, Encephalopathy. Class A (5-6 points): Compensated, 1-2 yr survival 100%. Class B (7-9): Intermediate, 1-2 yr survival 60-80%. Class C (10-15): Decompensated, 1-2 yr survival 35-45%. Guides prognosis and transplant timing.",
        "metadata": {"category": "hepatology", "subcategory": "cirrhosis", "priority": "medium"}
    },
    {
        "name": "AKI KDIGO Criteria",
        "type": "criteria",
        "content": "Stage 1: SCr 1.5-1.9x baseline or ≥0.3 mg/dL increase, or UO <0.5 mL/kg/h for 6-12h. Stage 2: SCr 2-2.9x baseline, or UO <0.5 mL/kg/h for ≥12h. Stage 3: SCr ≥3x baseline or ≥4.0 mg/dL increase, or UO <0.3 mL/kg/h for ≥24h, or anuria ≥12h. Requires baseline SCr.",
        "metadata": {"category": "nephrology", "subcategory": "aki", "priority": "medium"}
    },
    {
        "name": "Thyroid Function Interpretation",
        "type": "pattern",
        "content": "Primary hypothyroidism: High TSH, low T4. Primary hyperthyroidism: Low TSH, high T4/T3. Central hypothyroidism: Low/normal TSH, low T4. Subclinical hypothyroid: High TSH, normal T4. Subclinical hyperthyroid: Low TSH, normal T4/T3. Treat based on symptoms and severity.",
        "metadata": {"category": "endocrinology", "subcategory": "thyroid", "priority": "medium"}
    },
    {
        "name": "Pregnancy Terminology and Dating",
        "type": "reference",
        "content": "Gravida (G): Total pregnancies. Para (P): Births after 20 weeks. TPAL: Term, Preterm, Abortions, Living. Dating: LMP, ultrasound (most accurate 8-13 weeks). Trimesters: 1st 0-13wk, 2nd 14-27wk, 3rd 28-40wk. Term: 37-40+6wk. Preterm <37wk, post-term ≥42wk.",
        "metadata": {"category": "obstetrics", "subcategory": "pregnancy", "priority": "medium"}
    },
    {
        "name": "Fetal Heart Rate Interpretation",
        "type": "criteria",
        "content": "Baseline 110-160 bpm normal. Tachycardia >160, bradycardia <110. Variability: Absent <5 bpm, minimal 5-10, moderate 10-25 (reassuring), marked >25. Accelerations: 15 bpm above baseline for 15 sec (reactive NST). Decelerations: Early (head compression), variable (cord compression), late (uteroplacental insufficiency - concerning).",
        "metadata": {"category": "obstetrics", "subcategory": "fetal_monitoring", "priority": "medium"}
    },
    {
        "name": "Preeclampsia Diagnosis",
        "type": "criteria",
        "content": "After 20 weeks: BP ≥140/90 on 2 occasions 4h apart + proteinuria (≥300mg/24h or P/C ratio ≥0.3) or end-organ dysfunction. Severe features: BP ≥160/110, platelets <100k, Cr >1.1, pulmonary edema, visual symptoms, RUQ pain. HELLP: Hemolysis, Elevated Liver enzymes, Low Platelets.",
        "metadata": {"category": "obstetrics", "subcategory": "preeclampsia", "priority": "high"}
    },
    {
        "name": "Newborn APGAR Scoring",
        "type": "criteria",
        "content": "At 1 and 5 minutes. Each 0-2 points: Appearance (color), Pulse (HR), Grimace (reflex), Activity (tone), Respiration. 7-10 reassuring, 4-6 moderately abnormal (requires intervention), 0-3 critically low (immediate resuscitation). 5-min score <7 associated with increased risk.",
        "metadata": {"category": "neonatology", "subcategory": "assessment", "priority": "medium"}
    },
    {
        "name": "Developmental Milestones - Pediatric",
        "type": "reference",
        "content": "2mo: Social smile, tracks past midline. 4mo: Rolls, reaches. 6mo: Sits unsupported, babbles. 9mo: Crawls, pincer grasp, stranger anxiety. 12mo: Walks, 1-2 words, waves bye. 18mo: Runs, 10-25 words, uses spoon. 24mo: 2-word phrases, kicks ball. 3yr: 3-word sentences, pedals tricycle. Screen with validated tools (ASQ, MCHAT).",
        "metadata": {"category": "pediatrics", "subcategory": "development", "priority": "medium"}
    },
    {
        "name": "Immunization Schedule Highlights",
        "type": "reference",
        "content": "Birth: HepB. 2mo: DTaP, IPV, Hib, PCV13, RV. 6mo: Influenza (annually). 12-15mo: MMR, Varicella, HepA. 4-6yr: DTaP, IPV, MMR, Varicella boosters. 11-12yr: Tdap, MCV4, HPV. Adults: Tdap once, then Td q10yr; yearly flu; pneumococcal (≥65yr); shingles (≥50yr). Adjust for high-risk.",
        "metadata": {"category": "pediatrics", "subcategory": "immunizations", "priority": "medium"}
    },
    {
        "name": "Medication Dosing by Weight - Pediatric",
        "type": "pattern",
        "content": "Many pediatric medications dosed by mg/kg. Always verify: 1) Weight in kg, 2) Dose calculation, 3) Maximum dose limit, 4) Frequency. Example: Amoxicillin 40-50 mg/kg/day divided BID-TID, max 1500 mg/day. Acetaminophen 10-15 mg/kg q4-6h, max 75 mg/kg/day. Double-check all calculations.",
        "metadata": {"category": "pediatrics", "subcategory": "dosing", "priority": "high"}
    },
    {
        "name": "Geriatric Considerations",
        "type": "tip",
        "content": "Polypharmacy common - review medications regularly (deprescribing). Altered pharmacokinetics/dynamics: Lower GFR affects drug clearance, increased sensitivity to CNS medications. Fall risk assessment. Frailty screening. Functional status (ADLs/IADLs). Cognitive screening (MMSE, MoCA). Social support. Goals of care discussions.",
        "metadata": {"category": "geriatrics", "subcategory": "comprehensive_assessment", "priority": "medium"}
    },
    {
        "name": "Beers Criteria for Older Adults",
        "type": "guideline",
        "content": "Potentially inappropriate medications in elderly: Anticholinergics (confusion, falls), Benzodiazepines (falls, cognitive impairment), NSAIDs (GI bleed, renal injury), PPIs long-term (fractures, C.diff), First-gen antihistamines, Sliding scale insulin. Consider alternatives or lowest effective dose.",
        "metadata": {"category": "geriatrics", "subcategory": "medication_safety", "priority": "medium"}
    },
    {
        "name": "Fall Risk Assessment",
        "type": "criteria",
        "content": "Risk factors: Age >65, prior falls, mobility impairment, visual impairment, polypharmacy (especially sedatives, antihypertensives), environmental hazards, orthostatic hypotension, cognitive impairment. Screen with Timed Up and Go (>12 sec abnormal). Morse Fall Scale in hospitals. Multifactorial interventions.",
        "metadata": {"category": "geriatrics", "subcategory": "fall_prevention", "priority": "medium"}
    },
    {
        "name": "Pressure Injury Staging",
        "type": "criteria",
        "content": "Stage 1: Non-blanchable erythema, intact skin. Stage 2: Partial-thickness skin loss, shallow ulcer. Stage 3: Full-thickness skin loss, subcutaneous fat visible. Stage 4: Full-thickness tissue loss, bone/tendon/muscle exposed. Unstageable: Depth unknown due to slough/eschar. DTI: Purple/maroon intact skin or blood blister.",
        "metadata": {"category": "wound_care", "subcategory": "pressure_injuries", "priority": "medium"}
    },
    {
        "name": "Wound Assessment and Documentation",
        "type": "pattern",
        "content": "Document: Location, size (length x width x depth in cm), wound bed (percentage granulation/slough/eschar/necrotic), exudate (amount, color, odor), edges (attached, rolled, undermining, tunneling), periwound skin (intact, macerated, erythema), pain level. Photograph if possible. Measure weekly or with changes.",
        "metadata": {"category": "wound_care", "subcategory": "assessment", "priority": "medium"}
    },
    {
        "name": "Nutritional Risk Screening",
        "type": "criteria",
        "content": "Screen with validated tools: MST (Malnutrition Screening Tool), NRS-2002, MUST (Malnutrition Universal Screening Tool). Red flags: Unintentional weight loss ≥5% in 1 month or ≥10% in 6 months, BMI <18.5, decreased oral intake, GI symptoms, chronic disease. Refer to dietitian if positive screen.",
        "metadata": {"category": "nutrition", "subcategory": "screening", "priority": "medium"}
    }
]


def create_functions(registry):
    """Register medical functions"""
    print(f"\n{'='*60}")
    print("CREATING MEDICAL FUNCTIONS")
    print(f"{'='*60}\n")

    success_count = 0
    for func in MEDICAL_FUNCTIONS:
        try:
            success = registry.register_function(
                name=func["name"],
                code=func["code"],
                description=func["description"],
                parameters=func["parameters"],
                return_type=func["return_type"]
            )
            if success:
                success_count += 1
                print(f"✅ Created function: {func['name']}")
            else:
                print(f"❌ Failed to create: {func['name']}")
        except Exception as e:
            print(f"❌ Error creating {func['name']}: {e}")

    print(f"\n✅ Successfully created {success_count}/{len(MEDICAL_FUNCTIONS)} functions\n")


def create_patterns(preprocessor_path="./patterns"):
    """Create medical normalization patterns"""
    print(f"\n{'='*60}")
    print("CREATING MEDICAL PATTERNS")
    print(f"{'='*60}\n")

    patterns_dir = Path(preprocessor_path)
    patterns_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for pattern in MEDICAL_PATTERNS:
        try:
            pattern_file = patterns_dir / f"{pattern['name']}.json"
            with open(pattern_file, 'w') as f:
                json.dump(pattern, f, indent=2)
            success_count += 1
            print(f"✅ Created pattern: {pattern['name']}")
        except Exception as e:
            print(f"❌ Error creating {pattern['name']}: {e}")

    print(f"\n✅ Successfully created {success_count}/{len(MEDICAL_PATTERNS)} patterns\n")


def create_extras(manager):
    """Create clinical extras/hints"""
    print(f"\n{'='*60}")
    print("CREATING CLINICAL EXTRAS/HINTS")
    print(f"{'='*60}\n")

    success_count = 0
    for extra in CLINICAL_EXTRAS:
        try:
            success = manager.add_extra(
                extra_type=extra["type"],
                content=extra["content"],
                metadata=extra["metadata"],
                name=extra["name"]
            )
            if success:
                success_count += 1
                print(f"✅ Created extra: {extra['name']}")
            else:
                print(f"❌ Failed to create: {extra['name']}")
        except Exception as e:
            print(f"❌ Error creating {extra['name']}: {e}")

    print(f"\n✅ Successfully created {success_count}/{len(CLINICAL_EXTRAS)} extras\n")


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("CLINANNOTATE: Creating Medical Domain Examples")
    print("="*60)
    print("\nThis script creates:")
    print(f"  - {len(MEDICAL_FUNCTIONS)} medical calculation functions")
    print(f"  - {len(MEDICAL_PATTERNS)} text normalization patterns")
    print(f"  - {len(CLINICAL_EXTRAS)} clinical extras/hints")
    print("\n" + "="*60 + "\n")

    # Create functions
    try:
        registry = FunctionRegistry()
        create_functions(registry)
    except Exception as e:
        print(f"\n❌ Function creation failed: {e}")

    # Create patterns
    try:
        create_patterns()
    except Exception as e:
        print(f"\n❌ Pattern creation failed: {e}")

    # Create extras
    try:
        manager = ExtrasManager()
        create_extras(manager)
    except Exception as e:
        print(f"\n❌ Extras creation failed: {e}")

    print("\n" + "="*60)
    print("✅ COMPLETED: Medical Domain Examples Created!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
