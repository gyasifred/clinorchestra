def calculate_pediatric_nutrition_status(weight_kg, height_cm, age_months, sex):
    '''
    Calculate comprehensive pediatric nutrition status using WHO z-scores

    This function demonstrates advanced composition by calling multiple functions:
    1. calculate_zscore for weight-for-age
    2. calculate_zscore for height-for-age
    3. calculate_bmi to get BMI value
    4. calculate_zscore for BMI-for-age

    Args:
        weight_kg: Weight in kilograms
        height_cm: Height in centimeters
        age_months: Age in months
        sex: 'male' or 'female'

    Returns:
        Dictionary with z-scores, BMI, and WHO malnutrition classification
    '''
    # Calculate BMI first
    height_m = height_cm / 100.0
    bmi = call_function('calculate_bmi', weight_kg=weight_kg, height_m=height_m)

    # Calculate z-scores for different measurements
    weight_zscore = call_function('calculate_zscore',
                                   measurement='weight',
                                   value=weight_kg,
                                   age_months=age_months,
                                   sex=sex)

    height_zscore = call_function('calculate_zscore',
                                   measurement='height',
                                   value=height_cm,
                                   age_months=age_months,
                                   sex=sex)

    bmi_zscore = call_function('calculate_zscore',
                                measurement='bmi',
                                value=bmi,
                                age_months=age_months,
                                sex=sex)

    # WHO malnutrition classification based on z-scores
    def classify_wasting(bmi_z):
        if bmi_z < -3:
            return 'Severe wasting'
        elif bmi_z < -2:
            return 'Wasting'
        elif bmi_z < -1:
            return 'Risk of wasting'
        else:
            return 'Normal'

    def classify_stunting(height_z):
        if height_z < -3:
            return 'Severe stunting'
        elif height_z < -2:
            return 'Stunting'
        elif height_z < -1:
            return 'Risk of stunting'
        else:
            return 'Normal'

    def classify_underweight(weight_z):
        if weight_z < -3:
            return 'Severely underweight'
        elif weight_z < -2:
            return 'Underweight'
        elif weight_z < -1:
            return 'Risk of underweight'
        else:
            return 'Normal'

    return {
        'bmi': bmi,
        'weight_zscore': weight_zscore,
        'height_zscore': height_zscore,
        'bmi_zscore': bmi_zscore,
        'wasting_status': classify_wasting(bmi_zscore),
        'stunting_status': classify_stunting(height_zscore),
        'underweight_status': classify_underweight(weight_zscore),
        'age_months': age_months,
        'sex': sex
    }
