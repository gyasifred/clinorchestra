"""
MIMIC-IV Diagnosis Consolidation Mapping

Maps individual ICD-9 and ICD-10 codes to consolidated clinical diagnoses.
Same clinical condition with different ICD codes (ICD-9 vs ICD-10) are treated as single diagnosis.

Total: 20 unique clinical diagnoses (consolidated from multiple ICD code entries)
"""

# Consolidated diagnosis mapping: ICD code -> (diagnosis_name, category, diagnosis_id)
DIAGNOSIS_CONSOLIDATION = {
    # 1. CHEST PAIN - 15,415 cases (3 ICD codes)
    '78650': ('Chest pain', 'cardiovascular', 1),  # ICD-9: 7,297 cases
    '78659': ('Chest pain', 'cardiovascular', 1),  # ICD-9: 5,212 cases
    'R079': ('Chest pain', 'cardiovascular', 1),   # ICD-10: 2,906 cases

    # 2. CORONARY ATHEROSCLEROSIS - 5,751 cases (1 ICD code)
    '41401': ('Coronary atherosclerosis', 'cardiovascular', 2),  # ICD-9: 5,751 cases

    # 3. NSTEMI - 3,265 cases (1 ICD code)
    'I214': ('NSTEMI', 'cardiovascular', 3),  # ICD-10: 3,265 cases

    # 4. SUBENDOCARDIAL INFARCTION - 2,873 cases (1 ICD code)
    '41071': ('Subendocardial infarction', 'cardiovascular', 4),  # ICD-9: 2,873 cases

    # 5. ATRIAL FIBRILLATION - 3,090 cases (1 ICD code)
    '42731': ('Atrial fibrillation', 'cardiovascular', 5),  # ICD-9: 3,090 cases

    # 6. HYPERTENSIVE HEART DISEASE + CKD - 3,313 cases (1 ICD code)
    'I130': ('Hypertensive heart disease with CKD', 'cardiovascular', 6),  # ICD-10: 3,313 cases

    # 7. SEPSIS - 8,239 cases (3 ICD codes)
    'A419': ('Sepsis', 'infectious', 7),    # ICD-10: 5,095 cases
    '0389': ('Sepsis', 'infectious', 7),    # ICD-9: 3,144 cases
    '389': ('Sepsis', 'infectious', 7),     # ICD-9 alternate: same as 0389

    # 8. PNEUMONIA - 3,726 cases (1 ICD code)
    '486': ('Pneumonia', 'infectious', 8),  # ICD-9: 3,726 cases

    # 9. UTI - 2,967 cases (1 ICD code)
    '5990': ('Urinary tract infection', 'infectious', 9),  # ICD-9: 2,967 cases

    # 10. ACUTE KIDNEY INJURY - 6,185 cases (2 ICD codes)
    '5849': ('Acute kidney injury', 'renal', 10),  # ICD-9: 3,532 cases
    'N179': ('Acute kidney injury', 'renal', 10),  # ICD-10: 2,653 cases

    # 11. DEPRESSION - 7,202 cases (2 ICD codes)
    'F329': ('Depression', 'psychiatric', 11),  # ICD-10: 3,703 cases
    '311': ('Depression', 'psychiatric', 11),   # ICD-9: 3,499 cases

    # 12. ALCOHOL USE DISORDER - 8,038 cases (2 ICD codes)
    '30500': ('Alcohol use disorder', 'psychiatric', 12),   # ICD-9: 4,591 cases
    'F10129': ('Alcohol use disorder', 'psychiatric', 12),  # ICD-10: 3,447 cases

    # 13. CHEMOTHERAPY ENCOUNTER - 6,530 cases (2 ICD codes)
    'Z5111': ('Chemotherapy encounter', 'oncology', 13),  # ICD-10: 3,557 cases
    'V5811': ('Chemotherapy encounter', 'oncology', 13),  # ICD-9: 2,973 cases

    # 14. HEART FAILURE - Common
    '42833': ('Heart failure', 'cardiovascular', 14),  # ICD-9: Acute on chronic systolic heart failure
    'I5023': ('Heart failure', 'cardiovascular', 14),  # ICD-10: Acute on chronic systolic heart failure
    '42823': ('Heart failure', 'cardiovascular', 14),  # ICD-9: Acute on chronic diastolic heart failure
    'I5033': ('Heart failure', 'cardiovascular', 14),  # ICD-10: Acute on chronic diastolic heart failure
    '4280': ('Heart failure', 'cardiovascular', 14),   # ICD-9: Congestive heart failure, unspecified
    'I509': ('Heart failure', 'cardiovascular', 14),   # ICD-10: Heart failure, unspecified

    # 15. RESPIRATORY FAILURE / COPD - Common
    '51881': ('Respiratory failure', 'respiratory', 15),  # ICD-9: Acute respiratory failure
    'J9620': ('Respiratory failure', 'respiratory', 15),  # ICD-10: Acute respiratory failure, unspecified
    '4941': ('COPD', 'respiratory', 15),   # ICD-9: Chronic obstructive pulmonary disease
    'J449': ('COPD', 'respiratory', 15),   # ICD-10: COPD, unspecified

    # 16. GASTROINTESTINAL BLEEDING - Common
    '5789': ('Gastrointestinal bleeding', 'gastrointestinal', 16),  # ICD-9: GI hemorrhage
    'K922': ('Gastrointestinal bleeding', 'gastrointestinal', 16),  # ICD-10: GI hemorrhage, unspecified

    # 17. HYPERTENSION - Common
    '4019': ('Hypertension', 'cardiovascular', 17),  # ICD-9: Unspecified essential hypertension
    'I10': ('Hypertension', 'cardiovascular', 17),   # ICD-10: Essential (primary) hypertension

    # 18. DIABETES - Common
    '25000': ('Diabetes', 'endocrine', 18),  # ICD-9: Diabetes mellitus without complication
    'E119': ('Diabetes', 'endocrine', 18),   # ICD-10: Type 2 diabetes without complications
    'E1165': ('Diabetes', 'endocrine', 18),  # ICD-10: Type 2 diabetes with hyperglycemia

    # 19. CHRONIC KIDNEY DISEASE - Common
    '5859': ('Chronic kidney disease', 'renal', 19),  # ICD-9: Chronic kidney disease, unspecified
    'N189': ('Chronic kidney disease', 'renal', 19),  # ICD-10: Chronic kidney disease, unspecified
    'N183': ('Chronic kidney disease', 'renal', 19),  # ICD-10: CKD Stage 3

    # 20. ANEMIA - Common
    '2859': ('Anemia', 'hematologic', 20),  # ICD-9: Anemia, unspecified
    'D649': ('Anemia', 'hematologic', 20),  # ICD-10: Anemia, unspecified
}

# Reverse mapping: diagnosis_id -> diagnosis info
CONSOLIDATED_DIAGNOSES = {
    1: {
        'name': 'Chest pain',
        'category': 'cardiovascular',
        'icd9_codes': ['78650', '78659'],
        'icd10_codes': ['R079'],
        'all_codes': ['78650', '78659', 'R079'],
        'case_counts': {'78650': 7297, '78659': 5212, 'R079': 2906},
        'total_cases': 15415,
        'description': 'Chest pain, unspecified or other'
    },
    2: {
        'name': 'Coronary atherosclerosis',
        'category': 'cardiovascular',
        'icd9_codes': ['41401'],
        'icd10_codes': [],
        'all_codes': ['41401'],
        'case_counts': {'41401': 5751},
        'total_cases': 5751,
        'description': 'Coronary atherosclerosis of native coronary artery'
    },
    3: {
        'name': 'NSTEMI',
        'category': 'cardiovascular',
        'icd9_codes': [],
        'icd10_codes': ['I214'],
        'all_codes': ['I214'],
        'case_counts': {'I214': 3265},
        'total_cases': 3265,
        'description': 'Non-ST elevation myocardial infarction'
    },
    4: {
        'name': 'Subendocardial infarction',
        'category': 'cardiovascular',
        'icd9_codes': ['41071'],
        'icd10_codes': [],
        'all_codes': ['41071'],
        'case_counts': {'41071': 2873},
        'total_cases': 2873,
        'description': 'Subendocardial infarction, initial episode'
    },
    5: {
        'name': 'Atrial fibrillation',
        'category': 'cardiovascular',
        'icd9_codes': ['42731'],
        'icd10_codes': [],
        'all_codes': ['42731'],
        'case_counts': {'42731': 3090},
        'total_cases': 3090,
        'description': 'Atrial fibrillation'
    },
    6: {
        'name': 'Hypertensive heart disease with CKD',
        'category': 'cardiovascular',
        'icd9_codes': [],
        'icd10_codes': ['I130'],
        'all_codes': ['I130'],
        'case_counts': {'I130': 3313},
        'total_cases': 3313,
        'description': 'Hypertensive heart and chronic kidney disease'
    },
    7: {
        'name': 'Sepsis',
        'category': 'infectious',
        'icd9_codes': ['0389', '389'],
        'icd10_codes': ['A419'],
        'all_codes': ['0389', '389', 'A419'],
        'case_counts': {'A419': 5095, '0389': 3144},
        'total_cases': 8239,
        'description': 'Sepsis, unspecified organism'
    },
    8: {
        'name': 'Pneumonia',
        'category': 'infectious',
        'icd9_codes': ['486'],
        'icd10_codes': [],
        'all_codes': ['486'],
        'case_counts': {'486': 3726},
        'total_cases': 3726,
        'description': 'Pneumonia, organism unspecified'
    },
    9: {
        'name': 'Urinary tract infection',
        'category': 'infectious',
        'icd9_codes': ['5990'],
        'icd10_codes': [],
        'all_codes': ['5990'],
        'case_counts': {'5990': 2967},
        'total_cases': 2967,
        'description': 'Urinary tract infection, site not specified'
    },
    10: {
        'name': 'Acute kidney injury',
        'category': 'renal',
        'icd9_codes': ['5849'],
        'icd10_codes': ['N179'],
        'all_codes': ['5849', 'N179'],
        'case_counts': {'5849': 3532, 'N179': 2653},
        'total_cases': 6185,
        'description': 'Acute kidney failure, unspecified'
    },
    11: {
        'name': 'Depression',
        'category': 'psychiatric',
        'icd9_codes': ['311'],
        'icd10_codes': ['F329'],
        'all_codes': ['311', 'F329'],
        'case_counts': {'F329': 3703, '311': 3499},
        'total_cases': 7202,
        'description': 'Major depressive disorder'
    },
    12: {
        'name': 'Alcohol use disorder',
        'category': 'psychiatric',
        'icd9_codes': ['30500'],
        'icd10_codes': ['F10129'],
        'all_codes': ['30500', 'F10129'],
        'case_counts': {'30500': 4591, 'F10129': 3447},
        'total_cases': 8038,
        'description': 'Alcohol abuse'
    },
    13: {
        'name': 'Chemotherapy encounter',
        'category': 'oncology',
        'icd9_codes': ['V5811'],
        'icd10_codes': ['Z5111'],
        'all_codes': ['V5811', 'Z5111'],
        'case_counts': {'Z5111': 3557, 'V5811': 2973},
        'total_cases': 6530,
        'description': 'Encounter for antineoplastic chemotherapy'
    },
    14: {
        'name': 'Heart failure',
        'category': 'cardiovascular',
        'icd9_codes': ['42833', '42823', '4280'],
        'icd10_codes': ['I5023', 'I5033', 'I509'],
        'all_codes': ['42833', 'I5023', '42823', 'I5033', '4280', 'I509'],
        'case_counts': {},
        'total_cases': 0,
        'description': 'Congestive heart failure (acute on chronic or unspecified)'
    },
    15: {
        'name': 'Respiratory failure',
        'category': 'respiratory',
        'icd9_codes': ['51881', '4941'],
        'icd10_codes': ['J9620', 'J449'],
        'all_codes': ['51881', 'J9620', '4941', 'J449'],
        'case_counts': {},
        'total_cases': 0,
        'description': 'Acute respiratory failure and COPD'
    },
    16: {
        'name': 'Gastrointestinal bleeding',
        'category': 'gastrointestinal',
        'icd9_codes': ['5789'],
        'icd10_codes': ['K922'],
        'all_codes': ['5789', 'K922'],
        'case_counts': {},
        'total_cases': 0,
        'description': 'Gastrointestinal hemorrhage, unspecified'
    },
    17: {
        'name': 'Hypertension',
        'category': 'cardiovascular',
        'icd9_codes': ['4019'],
        'icd10_codes': ['I10'],
        'all_codes': ['4019', 'I10'],
        'case_counts': {},
        'total_cases': 0,
        'description': 'Essential (primary) hypertension'
    },
    18: {
        'name': 'Diabetes',
        'category': 'endocrine',
        'icd9_codes': ['25000'],
        'icd10_codes': ['E119', 'E1165'],
        'all_codes': ['25000', 'E119', 'E1165'],
        'case_counts': {},
        'total_cases': 0,
        'description': 'Diabetes mellitus without complications'
    },
    19: {
        'name': 'Chronic kidney disease',
        'category': 'renal',
        'icd9_codes': ['5859'],
        'icd10_codes': ['N189', 'N183'],
        'all_codes': ['5859', 'N189', 'N183'],
        'case_counts': {},
        'total_cases': 0,
        'description': 'Chronic kidney disease (stages 3-5 or unspecified)'
    },
    20: {
        'name': 'Anemia',
        'category': 'hematologic',
        'icd9_codes': ['2859'],
        'icd10_codes': ['D649'],
        'all_codes': ['2859', 'D649'],
        'case_counts': {},
        'total_cases': 0,
        'description': 'Anemia, unspecified'
    },
}

# Helper function to get consolidated diagnosis
def get_consolidated_diagnosis(icd_code):
    """
    Get consolidated diagnosis info for an ICD code.

    Args:
        icd_code: ICD-9 or ICD-10 code (string)

    Returns:
        tuple: (diagnosis_name, category, diagnosis_id) or None if not found
    """
    return DIAGNOSIS_CONSOLIDATION.get(icd_code)


def get_diagnosis_info(diagnosis_id):
    """
    Get full diagnosis information by ID.

    Args:
        diagnosis_id: Diagnosis ID (1-13)

    Returns:
        dict: Full diagnosis information or None if not found
    """
    return CONSOLIDATED_DIAGNOSES.get(diagnosis_id)


def get_all_icd_codes_for_diagnosis(diagnosis_id):
    """
    Get all ICD codes (ICD-9 and ICD-10) for a consolidated diagnosis.

    Args:
        diagnosis_id: Diagnosis ID (1-13)

    Returns:
        list: All ICD codes for this diagnosis
    """
    info = get_diagnosis_info(diagnosis_id)
    return info['all_codes'] if info else []


# Summary statistics
TOTAL_CONSOLIDATED_DIAGNOSES = 20
ORIGINAL_ICD_CODES = 40  # Approximate - before consolidation

# Category breakdown
CATEGORY_COUNTS = {
    'cardiovascular': 8,  # diagnoses 1-6, 14, 17
    'infectious': 3,      # diagnoses 7-9
    'renal': 2,           # diagnoses 10, 19
    'psychiatric': 2,     # diagnoses 11-12
    'oncology': 1,        # diagnosis 13
    'respiratory': 1,     # diagnosis 15
    'gastrointestinal': 1,  # diagnosis 16
    'endocrine': 1,       # diagnosis 18
    'hematologic': 1,     # diagnosis 20
}
