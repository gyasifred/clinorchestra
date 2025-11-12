"""
MIMIC-IV Diagnosis Consolidation Mapping

Maps individual ICD-9 and ICD-10 codes to consolidated clinical diagnoses.
Same clinical condition with different ICD codes (ICD-9 vs ICD-10) are treated as single diagnosis.

Total: 20 unique clinical diagnoses (consolidated from multiple ICD code entries)
"""

# Consolidated diagnosis mapping: ICD code -> (diagnosis_name, category, diagnosis_id)
DIAGNOSIS_CONSOLIDATION = {
    # 1. CHEST PAIN - 17,535 cases (4 ICD codes)
    '78650': ('Chest pain', 'cardiovascular', 1),  # ICD-9: 7,297 cases
    '78659': ('Chest pain', 'cardiovascular', 1),  # ICD-9: 5,212 cases
    'R079': ('Chest pain', 'cardiovascular', 1),   # ICD-10: 2,906 cases
    'R0789': ('Chest pain', 'cardiovascular', 1),  # ICD-10: 2,120 cases

    # 2. CORONARY ATHEROSCLEROSIS - 8,903 cases (3 ICD codes)
    '41401': ('Coronary atherosclerosis', 'cardiovascular', 2),  # ICD-9: 5,751 cases
    'I25110': ('Coronary atherosclerosis', 'cardiovascular', 2),  # ICD-10: 1,670 cases - with unstable angina
    'I2510': ('Coronary atherosclerosis', 'cardiovascular', 2),  # ICD-10: 1,482 cases - without angina

    # 3. MYOCARDIAL INFARCTION (NSTEMI/Subendocardial) - 6,138 cases (2 ICD codes)
    'I214': ('Myocardial infarction', 'cardiovascular', 3),  # ICD-10: 3,265 cases - NSTEMI
    '41071': ('Myocardial infarction', 'cardiovascular', 3),  # ICD-9: 2,873 cases - Subendocardial infarction

    # 4. HEART FAILURE - 6,648 cases (3 ICD codes)
    '42833': ('Heart failure', 'cardiovascular', 4),  # ICD-9: 2,270 cases - Acute on chronic diastolic
    'I110': ('Heart failure', 'cardiovascular', 4),   # ICD-10: 2,459 cases - Hypertensive heart disease with HF
    '42823': ('Heart failure', 'cardiovascular', 4),  # ICD-9: 1,919 cases - Acute on chronic systolic

    # 5. ATRIAL FIBRILLATION - 3,090 cases (1 ICD code)
    '42731': ('Atrial fibrillation', 'cardiovascular', 5),  # ICD-9: 3,090 cases

    # 6. HYPERTENSIVE HEART DISEASE with CKD - 3,313 cases (1 ICD code)
    'I130': ('Hypertensive heart disease with CKD', 'cardiovascular', 6),  # ICD-10: 3,313 cases

    # 7. SEPSIS - 8,239 cases (3 ICD codes)
    'A419': ('Sepsis', 'infectious', 7),    # ICD-10: 5,095 cases
    '0389': ('Sepsis', 'infectious', 7),    # ICD-9: 3,144 cases
    '389': ('Sepsis', 'infectious', 7),     # ICD-9 alternate: same as 0389

    # 8. PNEUMONIA - 5,843 cases (2 ICD codes)
    '486': ('Pneumonia', 'infectious', 8),   # ICD-9: 3,726 cases
    'J189': ('Pneumonia', 'infectious', 8),  # ICD-10: 2,117 cases

    # 9. UTI - 5,138 cases (2 ICD codes)
    '5990': ('Urinary tract infection', 'infectious', 9),  # ICD-9: 2,967 cases
    'N390': ('Urinary tract infection', 'infectious', 9),  # ICD-10: 2,171 cases

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

    # 14. SYNCOPE - 4,327 cases (2 ICD codes)
    '7802': ('Syncope', 'cardiovascular', 14),  # ICD-9: 2,644 cases
    'R55': ('Syncope', 'cardiovascular', 14),   # ICD-10: 1,683 cases

    # 15. AORTIC VALVE DISORDERS - 2,973 cases (2 ICD codes)
    '4241': ('Aortic valve disorders', 'cardiovascular', 15),  # ICD-9: 1,631 cases
    'I350': ('Aortic valve disorders', 'cardiovascular', 15),  # ICD-10: 1,342 cases

    # 16. ACUTE PANCREATITIS - 2,620 cases (1 ICD code)
    '5770': ('Acute pancreatitis', 'gastrointestinal', 16),  # ICD-9: 2,620 cases

    # 17. POSTOPERATIVE INFECTION - 2,198 cases (1 ICD code)
    '99859': ('Postoperative infection', 'infectious', 17),  # ICD-9: 2,198 cases

    # 18. CELLULITIS - 2,152 cases (1 ICD code)
    '6826': ('Cellulitis', 'infectious', 18),  # ICD-9: 2,152 cases - leg, except foot

    # 19. COVID-19 - 1,937 cases (1 ICD code)
    'U071': ('COVID-19', 'infectious', 19),  # ICD-10: 1,937 cases

    # 20. ALTERED MENTAL STATUS - 1,587 cases (1 ICD code)
    '78097': ('Altered mental status', 'neurological', 20),  # ICD-9: 1,587 cases
}

# Reverse mapping: diagnosis_id -> diagnosis info
CONSOLIDATED_DIAGNOSES = {
    1: {
        'name': 'Chest pain',
        'category': 'cardiovascular',
        'icd9_codes': ['78650', '78659'],
        'icd10_codes': ['R079', 'R0789'],
        'all_codes': ['78650', '78659', 'R079', 'R0789'],
        'case_counts': {'78650': 7297, '78659': 5212, 'R079': 2906, 'R0789': 2120},
        'total_cases': 17535,
        'description': 'Chest pain, unspecified or other'
    },
    2: {
        'name': 'Coronary atherosclerosis',
        'category': 'cardiovascular',
        'icd9_codes': ['41401'],
        'icd10_codes': ['I25110', 'I2510'],
        'all_codes': ['41401', 'I25110', 'I2510'],
        'case_counts': {'41401': 5751, 'I25110': 1670, 'I2510': 1482},
        'total_cases': 8903,
        'description': 'Coronary atherosclerosis of native coronary artery'
    },
    3: {
        'name': 'Myocardial infarction',
        'category': 'cardiovascular',
        'icd9_codes': ['41071'],
        'icd10_codes': ['I214'],
        'all_codes': ['I214', '41071'],
        'case_counts': {'I214': 3265, '41071': 2873},
        'total_cases': 6138,
        'description': 'Myocardial infarction (NSTEMI and subendocardial)'
    },
    4: {
        'name': 'Heart failure',
        'category': 'cardiovascular',
        'icd9_codes': ['42833', '42823'],
        'icd10_codes': ['I110'],
        'all_codes': ['42833', 'I110', '42823'],
        'case_counts': {'42833': 2270, 'I110': 2459, '42823': 1919},
        'total_cases': 6648,
        'description': 'Heart failure (acute on chronic, systolic, diastolic)'
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
        'icd10_codes': ['J189'],
        'all_codes': ['486', 'J189'],
        'case_counts': {'486': 3726, 'J189': 2117},
        'total_cases': 5843,
        'description': 'Pneumonia, organism unspecified'
    },
    9: {
        'name': 'Urinary tract infection',
        'category': 'infectious',
        'icd9_codes': ['5990'],
        'icd10_codes': ['N390'],
        'all_codes': ['5990', 'N390'],
        'case_counts': {'5990': 2967, 'N390': 2171},
        'total_cases': 5138,
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
        'name': 'Syncope',
        'category': 'cardiovascular',
        'icd9_codes': ['7802'],
        'icd10_codes': ['R55'],
        'all_codes': ['7802', 'R55'],
        'case_counts': {'7802': 2644, 'R55': 1683},
        'total_cases': 4327,
        'description': 'Syncope and collapse'
    },
    15: {
        'name': 'Aortic valve disorders',
        'category': 'cardiovascular',
        'icd9_codes': ['4241'],
        'icd10_codes': ['I350'],
        'all_codes': ['4241', 'I350'],
        'case_counts': {'4241': 1631, 'I350': 1342},
        'total_cases': 2973,
        'description': 'Aortic valve stenosis and disorders'
    },
    16: {
        'name': 'Acute pancreatitis',
        'category': 'gastrointestinal',
        'icd9_codes': ['5770'],
        'icd10_codes': [],
        'all_codes': ['5770'],
        'case_counts': {'5770': 2620},
        'total_cases': 2620,
        'description': 'Acute pancreatitis'
    },
    17: {
        'name': 'Postoperative infection',
        'category': 'infectious',
        'icd9_codes': ['99859'],
        'icd10_codes': [],
        'all_codes': ['99859'],
        'case_counts': {'99859': 2198},
        'total_cases': 2198,
        'description': 'Other postoperative infection'
    },
    18: {
        'name': 'Cellulitis',
        'category': 'infectious',
        'icd9_codes': ['6826'],
        'icd10_codes': [],
        'all_codes': ['6826'],
        'case_counts': {'6826': 2152},
        'total_cases': 2152,
        'description': 'Cellulitis and abscess of leg, except foot'
    },
    19: {
        'name': 'COVID-19',
        'category': 'infectious',
        'icd9_codes': [],
        'icd10_codes': ['U071'],
        'all_codes': ['U071'],
        'case_counts': {'U071': 1937},
        'total_cases': 1937,
        'description': 'COVID-19'
    },
    20: {
        'name': 'Altered mental status',
        'category': 'neurological',
        'icd9_codes': ['78097'],
        'icd10_codes': [],
        'all_codes': ['78097'],
        'case_counts': {'78097': 1587},
        'total_cases': 1587,
        'description': 'Altered mental status'
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
ORIGINAL_ICD_CODES = 32  # Before consolidation (actual count from top 50)
TOTAL_CASES = sum(d['total_cases'] for d in CONSOLIDATED_DIAGNOSES.values())

# Category breakdown
CATEGORY_COUNTS = {
    'cardiovascular': 7,    # diagnoses 1-6, 14, 15
    'infectious': 7,        # diagnoses 7-9, 17-19
    'renal': 1,             # diagnosis 10
    'psychiatric': 2,       # diagnoses 11-12
    'oncology': 1,          # diagnosis 13
    'gastrointestinal': 2,  # diagnoses 16
    'neurological': 1,      # diagnosis 20
}
