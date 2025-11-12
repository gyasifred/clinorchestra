def calculate_percentile(measurement, value, age_months, sex, height_cm=None):
    '''
    Calculate growth percentile using CDC data

    Args:
        measurement: 'weight', 'height', 'stature', or 'bmi'
        value: measurement value (kg for weight, cm for height, kg/m² for BMI)
        age_months: age in months
        sex: 'male', 'female', 'm', 'f', 1 (male), or 2 (female)
        height_cm: height in cm (only needed for weight-for-stature)

    Returns:
        Percentile value (0-100) or error dict
    '''
    from core.growth_calculators import CDCGrowthCalculator, GrowthMetric, Sex

    # Initialize calculator
    calculator = CDCGrowthCalculator(data_directory="cdc_data")

    # Map measurement type to metric
    metric_map = {
        'weight': GrowthMetric.WEIGHT_FOR_AGE,
        'height': GrowthMetric.STATURE_FOR_AGE,
        'stature': GrowthMetric.STATURE_FOR_AGE,
        'bmi': GrowthMetric.BMI_FOR_AGE
    }

    metric = metric_map.get(measurement.lower() if isinstance(measurement, str) else '')
    if not metric:
        return {'error': f'Unknown measurement type: {measurement}'}

    # Convert sex to enum (handle both string and numeric inputs)
    if isinstance(sex, str):
        sex_lower = sex.lower().strip()
        if sex_lower in ['male', 'm', 'boy', 'man']:
            sex_enum = Sex.MALE
        elif sex_lower in ['female', 'f', 'girl', 'woman']:
            sex_enum = Sex.FEMALE
        else:
            return {'error': f'Unknown sex value: {sex}'}
    elif sex == 1:
        sex_enum = Sex.MALE
    elif sex == 2:
        sex_enum = Sex.FEMALE
    else:
        return {'error': f'Invalid sex value: {sex}'}

    # Calculate percentile and z-score
    result = calculator.calculate_growth_percentile(
        metric=metric,
        sex=sex_enum,
        value=value,
        age_months=age_months,
        height_cm=height_cm
    )

    if result is None:
        return {'error': 'Could not calculate percentile'}

    # Return just the percentile value
    return result.percentile
