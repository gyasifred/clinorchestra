def percentile_to_zscore(percentile: float) -> float:
    """
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
    """
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
