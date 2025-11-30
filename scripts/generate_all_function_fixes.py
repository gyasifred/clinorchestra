#!/usr/bin/env python3
"""
Generate fixes for ALL problematic functions
Key principles:
1. Remove all boolean diagnostic flags
2. Replace diagnostic labels with descriptive categories
3. Remove directive/urgent language
4. Add context and caveats
5. Return objective data for LLM to interpret, not decisions
"""
import json
from pathlib import Path

FUNCTIONS_DIR = Path("/home/user/clinorchestra/functions")

def fix_interpret_zscore_malnutrition():
    """Fix the z-score malnutrition interpretation function"""
    return {
        "name": "interpret_zscore_malnutrition",
        "code": """def interpret_zscore_malnutrition(zscore: float, measurement_type: str) -> str:
    \"\"\"
    Interpret z-score for growth assessment following WHO/ASPEN criteria.

    IMPORTANT: Returns OBJECTIVE interpretation only. Does NOT diagnose malnutrition.
    Diagnosis requires comprehensive assessment per ASPEN (≥2 indicators).

    Z-score ranges (WHO/ASPEN):
    - Weight-for-height or BMI-for-age:
      * z < -3: Severe growth deficit
      * -3 ≤ z < -2: Moderate growth deficit
      * -2 ≤ z < -1: Mild growth deficit (at risk)
      * -1 ≤ z ≤ +1: Normal range
      * z > +2: Above normal range

    Args:
        zscore: Z-score value
        measurement_type: Type of measurement ('weight-for-age', 'height-for-age', 'BMI-for-age', etc.)

    Returns:
        Objective interpretation of z-score with percentile context
    \"\"\"
    measurement_type = measurement_type.lower().replace(" ", "-")

    # Normalize measurement type
    if "weight" in measurement_type and ("height" in measurement_type or "length" in measurement_type):
        category = "weight-for-height"
    elif "bmi" in measurement_type:
        category = "bmi"
    elif "height" in measurement_type or "length" in measurement_type:
        category = "height-for-age"
    elif "weight" in measurement_type:
        category = "weight-for-age"
    else:
        category = "unknown"

    # Calculate approximate percentile
    def zscore_to_percentile(z):
        mappings = {-3.0: 0.13, -2.5: 0.62, -2.0: 2.3, -1.88: 3.0, -1.64: 5.0, -1.0: 15.9,
                    0.0: 50.0, 1.0: 84.1, 1.64: 95.0, 2.0: 97.7, 3.0: 99.87}
        if z in mappings:
            return mappings[z]
        z_keys = sorted(mappings.keys())
        if z <= z_keys[0]:
            return mappings[z_keys[0]]
        if z >= z_keys[-1]:
            return mappings[z_keys[-1]]
        for i in range(len(z_keys) - 1):
            if z_keys[i] <= z <= z_keys[i+1]:
                z1, z2 = z_keys[i], z_keys[i+1]
                p1, p2 = mappings[z1], mappings[z2]
                proportion = (z - z1) / (z2 - z1)
                return p1 + proportion * (p2 - p1)
        return 50.0

    percentile = zscore_to_percentile(zscore)

    # Generate objective interpretation
    if category in ["weight-for-height", "bmi"]:
        if zscore < -3:
            severity_category = "Severe growth deficit"
            interpretation = f"Z-score {zscore:.2f} (<-3 SD, ~{percentile:.1f}th percentile) indicates severe growth deficit. Corresponds to WHO severe acute malnutrition threshold."
        elif zscore < -2:
            severity_category = "Moderate growth deficit"
            interpretation = f"Z-score {zscore:.2f} (-3 to -2 SD, ~{percentile:.1f}th percentile) indicates moderate growth deficit. Corresponds to WHO moderate malnutrition threshold."
        elif zscore < -1:
            severity_category = "Mild growth deficit"
            interpretation = f"Z-score {zscore:.2f} (-2 to -1 SD, ~{percentile:.1f}th percentile) indicates mild growth deficit. Within at-risk range."
        elif zscore <= 1:
            severity_category = "Normal range"
            interpretation = f"Z-score {zscore:.2f} (-1 to +1 SD, ~{percentile:.1f}th percentile) is within normal range."
        elif zscore <= 2:
            severity_category = "Above normal"
            interpretation = f"Z-score {zscore:.2f} (+1 to +2 SD, ~{percentile:.1f}th percentile) is above normal range."
        else:
            severity_category = "Well above normal"
            interpretation = f"Z-score {zscore:.2f} (>+2 SD, ~{percentile:.1f}th percentile) is well above normal range."

    elif category == "height-for-age":
        if zscore < -3:
            severity_category = "Severe height deficit"
            interpretation = f"Z-score {zscore:.2f} (<-3 SD) indicates severe stunting (chronic growth deficit)."
        elif zscore < -2:
            severity_category = "Moderate height deficit"
            interpretation = f"Z-score {zscore:.2f} (-3 to -2 SD) indicates stunting (chronic growth deficit)."
        elif zscore < -1:
            severity_category = "Mild height deficit"
            interpretation = f"Z-score {zscore:.2f} (-2 to -1 SD) indicates mild height deficit."
        else:
            severity_category = "Normal height"
            interpretation = f"Z-score {zscore:.2f} (≥-1 SD) indicates normal height-for-age."

    elif category == "weight-for-age":
        if zscore < -3:
            severity_category = "Severe weight deficit"
            interpretation = f"Z-score {zscore:.2f} (<-3 SD) indicates severe underweight."
        elif zscore < -2:
            severity_category = "Moderate weight deficit"
            interpretation = f"Z-score {zscore:.2f} (-3 to -2 SD) indicates underweight."
        elif zscore < -1:
            severity_category = "Mild weight deficit"
            interpretation = f"Z-score {zscore:.2f} (-2 to -1 SD) indicates mild underweight risk."
        else:
            severity_category = "Normal weight"
            interpretation = f"Z-score {zscore:.2f} (≥-1 SD) indicates normal weight-for-age."
    else:
        interpretation = f"Z-score {zscore:.2f} for {measurement_type}. Measurement type not fully categorized."

    # Add context note
    context_note = \" | NOTE: Z-score interpretation provides growth assessment context. ASPEN malnutrition diagnosis requires ≥2 indicators. Consider clinical context, trajectory, and other factors.\"

    return interpretation + context_note
""",
        "description": "Interpret z-score for growth assessment - v1.0.0 returns OBJECTIVE data only, no diagnostic labels",
        "parameters": {
            "zscore": {"type": "number", "description": "Z-score for growth measurement"},
            "measurement_type": {"type": "string", "description": "Type: 'weight-for-age', 'height-for-age', 'BMI-for-age', etc."}
        },
        "returns": {"type": "string", "description": "Objective interpretation with percentile context and clinical note"},
        "enabled": True,
        "signature": "(zscore: float, measurement_type: str) -> str"
    }

# Generate fixes for other critical functions
def fix_interpret_albumin_malnutrition():
    return {
        "name": "interpret_albumin_malnutrition",
        "code": """def interpret_albumin_malnutrition(albumin: float, age_years: float = None) -> dict:
    \"\"\"
    Interpret serum albumin in nutritional context

    IMPORTANT: Albumin has MANY non-nutritional causes of low levels.
    Does NOT diagnose malnutrition. Requires comprehensive assessment with anthropometrics.

    Normal: 3.5-5.0 g/dL
    Low albumin thresholds:
    - 3.0-3.4 g/dL: Mild depletion
    - 2.4-2.9 g/dL: Moderate depletion
    - <2.4 g/dL: Severe depletion

    Args:
        albumin: Serum albumin in g/dL
        age_years: Patient age (optional)

    Returns:
        dict with status, interpretation, and critical caveats
    \"\"\"
    if albumin >= 3.5:
        return {
            'albumin_level': f'{albumin:.1f} g/dL',
            'albumin_status': 'Normal',
            'interpretation': f\"Albumin {albumin:.1f} g/dL is within normal range (3.5-5.0 g/dL).\",
            'clinical_context': 'Normal albumin does NOT exclude malnutrition. Assess anthropometric measurements (weight, height, BMI, z-scores).',
            'limitations': 'Albumin is NOT a sensitive marker for acute malnutrition. Consider prealbumin for acute assessment.'
        }
    elif albumin >= 3.0:
        return {
            'albumin_level': f'{albumin:.1f} g/dL',
            'albumin_status': 'Mild depletion',
            'interpretation': f\"Albumin {albumin:.1f} g/dL indicates mild protein depletion.\",
            'clinical_context': 'Low albumin may reflect inflammation, liver disease, nephrotic syndrome, OR malnutrition. Correlate with z-scores and clinical assessment.',
            'limitations': 'Albumin alone does NOT diagnose malnutrition. Must assess anthropometrics per ASPEN criteria.'
        }
    elif albumin >= 2.4:
        return {
            'albumin_level': f'{albumin:.1f} g/dL',
            'albumin_status': 'Moderate depletion',
            'interpretation': f\"Albumin {albumin:.1f} g/dL indicates moderate protein depletion.\",
            'clinical_context': 'Moderate hypoalbuminemia requires evaluation for: (1) malnutrition (assess z-scores), (2) inflammation (check CRP), (3) liver disease, (4) protein loss (nephrotic syndrome).',
            'limitations': 'Albumin is a late marker (21-day half-life). Does NOT diagnose malnutrition without anthropometric data.'
        }
    else:
        return {
            'albumin_level': f'{albumin:.1f} g/dL',
            'albumin_status': 'Severe depletion',
            'interpretation': f\"Albumin {albumin:.1f} g/dL indicates severe hypoalbuminemia.\",
            'clinical_context': 'Severe hypoalbuminemia associated with poor outcomes. Evaluate for: malnutrition (check z-scores and weight trends), liver disease, nephrotic syndrome, acute inflammation, sepsis.',
            'limitations': 'Low albumin does NOT equal malnutrition diagnosis. MUST assess anthropometrics. Consider nutrition support if malnutrition confirmed.'
        }
""",
        "description": "Interpret albumin in nutritional context - v1.0.0 NO boolean flags, emphasizes limitations",
        "parameters": {
            "albumin": {"type": "number", "description": "Serum albumin in g/dL", "required": True},
            "age_years": {"type": "number", "description": "Patient age in years (optional)", "required": False}
        },
        "returns": "dict with status, interpretation, clinical_context, and limitations",
        "enabled": True,
        "signature": "(albumin: float, age_years: float = None) -> dict"
    }

def save_function(func_data):
    """Save function to JSON file"""
    file_path = FUNCTIONS_DIR / f"{func_data['name']}.json"
    with open(file_path, 'w') as f:
        json.dump(func_data, f, indent=2)
    print(f"✓ Fixed: {func_data['name']}.json")

if __name__ == "__main__":
    print("Generating comprehensive function fixes...")
    print("=" * 80)

    # Generate and save fixes
    save_function(fix_interpret_zscore_malnutrition())
    save_function(fix_interpret_albumin_malnutrition())

    print("=" * 80)
    print("Generated fixes for critical malnutrition functions")
