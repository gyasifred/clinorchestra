def interpret_albumin_malnutrition(albumin: float, age_years: float = None) -> dict:
    """
    Interpret serum albumin in context of malnutrition

    Normal albumin: 3.5-5.0 g/dL
    Malnutrition thresholds:
    - Mild: 3.0-3.4 g/dL
    - Moderate: 2.4-2.9 g/dL
    - Severe: <2.4 g/dL

    Args:
        albumin: Serum albumin in g/dL
        age_years: Patient age (optional)

    Returns:
        {
            'albumin_status': str,
            'malnutrition_indicator': bool,
            'interpretation': str,
            'caveats': str
        }
    """
    if albumin >= 3.5:
        return {
            'albumin_status': 'Normal',
            'malnutrition_indicator': False,
            'interpretation': f"Albumin {albumin:.1f} g/dL is within normal range (3.5-5.0 g/dL)",
            'caveats': "Normal albumin does not exclude malnutrition. Consider other markers like prealbumin, weight loss, and physical exam findings."
        }

    elif albumin >= 3.0:
        return {
            'albumin_status': 'Mild depletion',
            'malnutrition_indicator': True,
            'interpretation': f"Albumin {albumin:.1f} g/dL suggests mild protein depletion",
            'caveats': "Low albumin may reflect inflammation, liver disease, or nephrotic syndrome rather than malnutrition alone. Correlate with clinical context."
        }

    elif albumin >= 2.4:
        return {
            'albumin_status': 'Moderate depletion',
            'malnutrition_indicator': True,
            'interpretation': f"Albumin {albumin:.1f} g/dL suggests moderate protein-energy malnutrition",
            'caveats': "Albumin is a late marker of malnutrition due to long half-life (21 days). Consider prealbumin for more acute assessment. Rule out non-nutritional causes."
        }

    else:  # <2.4
        return {
            'albumin_status': 'Severe depletion',
            'malnutrition_indicator': True,
            'interpretation': f"Albumin {albumin:.1f} g/dL indicates severe protein depletion and high risk for complications",
            'caveats': "Severe hypoalbuminemia (<2.4 g/dL) is associated with increased mortality and complications. Rule out liver disease, nephrotic syndrome, and acute inflammation. Consider nutrition support."
        }
