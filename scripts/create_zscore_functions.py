#!/usr/bin/env python3
"""
Create z-score and percentile conversion functions for malnutrition assessment
Author: Frederick Gyasi (gyasi@musc.edu)
"""

import json
from pathlib import Path

# Create functions directory if it doesn't exist
functions_dir = Path("./functions")
functions_dir.mkdir(parents=True, exist_ok=True)

# Function 1: Percentile to Z-score
percentile_to_zscore_json = {
    "name": "percentile_to_zscore",
    "description": "Convert growth percentile to z-score for malnutrition assessment. Lower percentiles = negative z-scores. Used for growth chart interpretation.",
    "parameters": {
        "percentile": {
            "type": "number",
            "description": "Growth percentile (0-100). Common values: 3rd, 5th, 10th, 25th, 50th, 75th, 90th, 95th, 97th percentile"
        }
    },
    "returns": {
        "type": "number",
        "description": "Z-score (standard deviations from mean). Negative z-scores indicate below average."
    }
}

percentile_to_zscore_code = """def percentile_to_zscore(percentile: float) -> float:
    \"\"\"
    Convert percentile to z-score for growth assessment.

    Clinical significance:
    - Below 5th percentile (z < -1.64): Malnutrition risk
    - Below 3rd percentile (z < -1.88): Moderate-severe malnutrition
    - 50th percentile (z = 0): Average
    - Above 95th percentile (z > 1.64): Overweight/obesity risk

    Args:
        percentile: Growth percentile (0-100)

    Returns:
        Z-score (standard deviations from mean)

    Examples:
        50th percentile -> z = 0 (average)
        3rd percentile -> z = -1.88 (malnutrition concern)
        97th percentile -> z = 1.88 (overweight concern)
    \"\"\"
    # Validate input
    if percentile < 0 or percentile > 100:
        raise ValueError("Percentile must be between 0 and 100")

    # Handle edge cases
    if percentile == 0:
        return -3.0  # Extremely low
    if percentile == 100:
        return 3.0  # Extremely high
    if percentile == 50:
        return 0.0  # Mean

    # Convert percentile to proportion
    p = percentile / 100.0

    # Common percentile to z-score mappings (for accuracy)
    common_mappings = {
        0.01: -2.33,  # 1st percentile
        0.03: -1.88,  # 3rd percentile
        0.05: -1.64,  # 5th percentile
        0.10: -1.28,  # 10th percentile
        0.25: -0.67,  # 25th percentile
        0.50: 0.00,   # 50th percentile (median)
        0.75: 0.67,   # 75th percentile
        0.90: 1.28,   # 90th percentile
        0.95: 1.64,   # 95th percentile
        0.97: 1.88,   # 97th percentile
        0.99: 2.33,   # 99th percentile
    }

    # Check for exact match
    if p in common_mappings:
        return round(common_mappings[p], 2)

    # Approximate using inverse normal CDF (rational approximation)
    # This is Beasley-Springer-Moro algorithm (simplified)
    if p < 0.5:
        # Lower tail
        sign = -1.0
        p_calc = p
    else:
        # Upper tail
        sign = 1.0
        p_calc = 1.0 - p

    # Rational approximation coefficients
    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
    b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
    c = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
         0.0276438810333863, 0.0038405729373609, 0.0003951896511919,
         0.0000321767881768, 0.0000002888167364, 0.0000003960315187]

    # Use simplified approximation
    if p_calc > 0.0 and p_calc < 1.0:
        import math
        t = math.sqrt(-2.0 * math.log(p_calc))
        z = t - (c[0] + c[1]*t + c[2]*t**2) / (1.0 + c[3]*t + c[4]*t**2 + c[5]*t**3)
        return round(sign * z, 2)

    return 0.0
"""

# Function 2: Z-score to Percentile
zscore_to_percentile_json = {
    "name": "zscore_to_percentile",
    "description": "Convert z-score to growth percentile for malnutrition assessment. Negative z-scores = lower percentiles. Used to interpret standard deviation scores.",
    "parameters": {
        "zscore": {
            "type": "number",
            "description": "Z-score (standard deviations from mean). Common values: -3, -2, -1.88, -1, 0, +1, +2"
        }
    },
    "returns": {
        "type": "number",
        "description": "Growth percentile (0-100). Lower percentiles indicate below average growth."
    }
}

zscore_to_percentile_code = """def zscore_to_percentile(zscore: float) -> float:
    \"\"\"
    Convert z-score to percentile for growth assessment.

    Clinical significance:
    - z < -2: Below 3rd percentile (severe malnutrition)
    - z < -1.64: Below 5th percentile (malnutrition risk)
    - z = 0: 50th percentile (average)
    - z > 1.64: Above 95th percentile (overweight/obesity)

    Args:
        zscore: Z-score (standard deviations from mean)

    Returns:
        Percentile (0-100)

    Examples:
        z = -2 -> 2.3rd percentile (severe malnutrition)
        z = -1.88 -> 3rd percentile (malnutrition)
        z = 0 -> 50th percentile (average)
        z = 1.88 -> 97th percentile (overweight)
    \"\"\"
    # Common z-score to percentile mappings
    common_mappings = {
        -3.0: 0.13,   # Extremely low
        -2.5: 0.62,
        -2.33: 1.0,   # 1st percentile
        -2.0: 2.3,    # Severe malnutrition threshold
        -1.88: 3.0,   # 3rd percentile
        -1.64: 5.0,   # 5th percentile (malnutrition risk)
        -1.28: 10.0,  # 10th percentile
        -1.0: 15.9,
        -0.67: 25.0,  # 25th percentile
        -0.5: 30.9,
        0.0: 50.0,    # 50th percentile (median)
        0.5: 69.1,
        0.67: 75.0,   # 75th percentile
        1.0: 84.1,
        1.28: 90.0,   # 90th percentile
        1.64: 95.0,   # 95th percentile (overweight risk)
        1.88: 97.0,   # 97th percentile
        2.0: 97.7,
        2.33: 99.0,   # 99th percentile
        3.0: 99.87,   # Extremely high
    }

    # Check for exact match
    if zscore in common_mappings:
        return round(common_mappings[zscore], 2)

    # Find closest matches for interpolation
    z_keys = sorted(common_mappings.keys())

    if zscore <= z_keys[0]:
        return common_mappings[z_keys[0]]
    if zscore >= z_keys[-1]:
        return common_mappings[z_keys[-1]]

    # Linear interpolation between closest points
    for i in range(len(z_keys) - 1):
        if z_keys[i] <= zscore <= z_keys[i+1]:
            z1, z2 = z_keys[i], z_keys[i+1]
            p1, p2 = common_mappings[z1], common_mappings[z2]

            # Linear interpolation
            proportion = (zscore - z1) / (z2 - z1)
            percentile = p1 + proportion * (p2 - p1)
            return round(percentile, 2)

    return 50.0  # Default to median
"""

# Function 3: Interpret Z-score for Malnutrition
interpret_zscore_json = {
    "name": "interpret_zscore_malnutrition",
    "description": "Interpret z-score for malnutrition severity classification (WHO/ASPEN criteria). Returns clinical interpretation and severity level.",
    "parameters": {
        "zscore": {
            "type": "number",
            "description": "Z-score for weight-for-age, height-for-age, or weight-for-height/BMI"
        },
        "measurement_type": {
            "type": "string",
            "description": "Type of measurement: 'weight-for-age', 'height-for-age', 'weight-for-height', or 'BMI-for-age'"
        }
    },
    "returns": {
        "type": "string",
        "description": "Clinical interpretation with severity classification"
    }
}

interpret_zscore_code = """def interpret_zscore_malnutrition(zscore: float, measurement_type: str) -> str:
    \"\"\"
    Interpret z-score for malnutrition assessment following WHO/ASPEN criteria.

    WHO Malnutrition Classification:
    - Weight-for-height or BMI-for-age:
      * z < -3: Severe acute malnutrition
      * -3 ≤ z < -2: Moderate acute malnutrition
      * -2 ≤ z < -1: Mild malnutrition risk
      * -1 ≤ z ≤ +1: Normal
      * z > +2: Overweight/obesity

    - Height-for-age (stunting):
      * z < -2: Stunted (chronic malnutrition)
      * z < -3: Severely stunted

    Args:
        zscore: Z-score value
        measurement_type: Type of growth measurement

    Returns:
        Clinical interpretation with severity and recommendations
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

    # Interpret based on category and z-score
    if category in ["weight-for-height", "bmi"]:
        if zscore < -3:
            severity = "Severe Acute Malnutrition"
            interpretation = f"Z-score {zscore:.2f} indicates SEVERE ACUTE MALNUTRITION (<-3 SD). Immediate intervention required. Risk of mortality increased."
            percentile = zscore_to_percentile(zscore)
            interpretation += f" (Below {percentile:.1f}th percentile)"
        elif zscore < -2:
            severity = "Moderate Acute Malnutrition"
            interpretation = f"Z-score {zscore:.2f} indicates MODERATE ACUTE MALNUTRITION (-3 to -2 SD). Nutritional intervention needed."
            percentile = zscore_to_percentile(zscore)
            interpretation += f" (Approximately {percentile:.1f}th percentile)"
        elif zscore < -1:
            severity = "Mild Malnutrition Risk"
            interpretation = f"Z-score {zscore:.2f} indicates MILD MALNUTRITION RISK (-2 to -1 SD). Monitor closely and consider intervention."
            percentile = zscore_to_percentile(zscore)
            interpretation += f" (Approximately {percentile:.1f}th percentile)"
        elif zscore <= 1:
            severity = "Normal"
            interpretation = f"Z-score {zscore:.2f} is NORMAL (-1 to +1 SD)."
            percentile = zscore_to_percentile(zscore)
            interpretation += f" (Approximately {percentile:.1f}th percentile)"
        elif zscore <= 2:
            severity = "Possible Overweight"
            interpretation = f"Z-score {zscore:.2f} indicates POSSIBLE OVERWEIGHT (+1 to +2 SD)."
            percentile = zscore_to_percentile(zscore)
            interpretation += f" (Approximately {percentile:.1f}th percentile)"
        else:
            severity = "Obesity Risk"
            interpretation = f"Z-score {zscore:.2f} indicates OBESITY RISK (>+2 SD). Assess for overnutrition."
            percentile = zscore_to_percentile(zscore)
            interpretation += f" (Above {percentile:.1f}th percentile)"

    elif category == "height-for-age":
        if zscore < -3:
            severity = "Severe Stunting"
            interpretation = f"Z-score {zscore:.2f} indicates SEVERE STUNTING (<-3 SD). Chronic malnutrition."
        elif zscore < -2:
            severity = "Stunting"
            interpretation = f"Z-score {zscore:.2f} indicates STUNTING (-3 to -2 SD). Chronic undernutrition."
        elif zscore < -1:
            severity = "Risk of Stunting"
            interpretation = f"Z-score {zscore:.2f} indicates RISK OF STUNTING (-2 to -1 SD). Monitor growth."
        else:
            severity = "Normal Height"
            interpretation = f"Z-score {zscore:.2f} indicates NORMAL HEIGHT-FOR-AGE (≥-1 SD)."

    elif category == "weight-for-age":
        if zscore < -3:
            severity = "Severe Underweight"
            interpretation = f"Z-score {zscore:.2f} indicates SEVERE UNDERWEIGHT (<-3 SD). Immediate assessment needed."
        elif zscore < -2:
            severity = "Underweight"
            interpretation = f"Z-score {zscore:.2f} indicates UNDERWEIGHT (-3 to -2 SD). Nutritional support needed."
        elif zscore < -1:
            severity = "Mild Underweight Risk"
            interpretation = f"Z-score {zscore:.2f} indicates MILD UNDERWEIGHT RISK (-2 to -1 SD)."
        else:
            severity = "Normal Weight"
            interpretation = f"Z-score {zscore:.2f} indicates NORMAL WEIGHT-FOR-AGE (≥-1 SD)."

    else:
        interpretation = f"Z-score {zscore:.2f} for {measurement_type}. Unable to classify - measurement type not recognized."

    return interpretation

def zscore_to_percentile(zscore: float) -> float:
    \"\"\"Helper function for percentile conversion\"\"\"
    common_mappings = {
        -3.0: 0.13, -2.5: 0.62, -2.33: 1.0, -2.0: 2.3, -1.88: 3.0,
        -1.64: 5.0, -1.28: 10.0, -1.0: 15.9, -0.67: 25.0, 0.0: 50.0,
        0.67: 75.0, 1.0: 84.1, 1.28: 90.0, 1.64: 95.0, 1.88: 97.0,
        2.0: 97.7, 2.33: 99.0, 3.0: 99.87
    }

    if zscore in common_mappings:
        return common_mappings[zscore]

    z_keys = sorted(common_mappings.keys())
    if zscore <= z_keys[0]:
        return common_mappings[z_keys[0]]
    if zscore >= z_keys[-1]:
        return common_mappings[z_keys[-1]]

    for i in range(len(z_keys) - 1):
        if z_keys[i] <= zscore <= z_keys[i+1]:
            z1, z2 = z_keys[i], z_keys[i+1]
            p1, p2 = common_mappings[z1], common_mappings[z2]
            proportion = (zscore - z1) / (z2 - z1)
            return p1 + proportion * (p2 - p1)

    return 50.0
"""

# Write all functions
functions = [
    ("percentile_to_zscore", percentile_to_zscore_json, percentile_to_zscore_code),
    ("zscore_to_percentile", zscore_to_percentile_json, zscore_to_percentile_code),
    ("interpret_zscore_malnutrition", interpret_zscore_json, interpret_zscore_code),
]

for func_name, json_data, code_data in functions:
    # Write JSON definition
    json_path = functions_dir / f"{func_name}.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"✓ Created {json_path}")

    # Write Python implementation
    py_path = functions_dir / f"{func_name}.py"
    with open(py_path, 'w') as f:
        f.write(code_data)
    print(f"✓ Created {py_path}")

print(f"\n✅ Created {len(functions)} z-score/percentile functions for malnutrition assessment!")
print("\nFunctions created:")
print("  1. percentile_to_zscore - Convert percentile to z-score (lower percentile = negative z)")
print("  2. zscore_to_percentile - Convert z-score to percentile")
print("  3. interpret_zscore_malnutrition - Clinical interpretation with severity classification")
