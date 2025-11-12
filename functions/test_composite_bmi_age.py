def test_composite_bmi_age(weight_kg, height_m, birth_date):
    '''
    TEST: Composite function demonstrating function composition

    This function calls other registered functions:
    1. calculate_age_months(birth_date) - to get age
    2. calculate_bmi(weight_kg, height_m) - to get BMI

    Args:
        weight_kg: Weight in kilograms
        height_m: Height in meters
        birth_date: Birth date in YYYY-MM-DD format

    Returns:
        Dictionary with age_months, bmi, and categorization
    '''
    # Call calculate_age_months function
    age_months = call_function('calculate_age_months', birth_date=birth_date)

    # Call calculate_bmi function
    bmi = call_function('calculate_bmi', weight_kg=weight_kg, height_m=height_m)

    # Categorize BMI
    if bmi < 18.5:
        category = 'Underweight'
    elif bmi < 25:
        category = 'Normal'
    elif bmi < 30:
        category = 'Overweight'
    else:
        category = 'Obese'

    return {
        'age_months': age_months,
        'bmi': bmi,
        'category': category,
        'weight_kg': weight_kg,
        'height_m': height_m
    }
