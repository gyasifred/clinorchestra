def interpret_zscore_malnutrition(zscore: float, measurement_type: str) -> str:
    """
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
    """
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
    """Helper function for percentile conversion"""
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
