def calculate_bmi(weight_kg, height_m):
    '''Calculate BMI from weight and height'''
    if height_m <= 0:
        return None
    bmi = weight_kg / (height_m ** 2)
    return round(bmi, 2)