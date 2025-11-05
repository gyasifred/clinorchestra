def zscore_to_percentile(zscore: float) -> float:
    """
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
    """
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
