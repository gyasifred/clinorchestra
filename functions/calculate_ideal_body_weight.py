def calculate_ideal_body_weight(height_cm, sex):
    '''Calculate IBW using Devine formula'''
    height_inches = height_cm / 2.54
    if sex.lower() in ['male', 'm']:
        ibw_kg = 50 + 2.3 * (height_inches - 60)
    elif sex.lower() in ['female', 'f']:
        ibw_kg = 45.5 + 2.3 * (height_inches - 60)
    else:
        return None
    return round(ibw_kg, 1)