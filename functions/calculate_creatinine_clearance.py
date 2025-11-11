def calculate_creatinine_clearance(age, weight_kg, creatinine, sex):
    '''Calculate CrCl using Cockcroft-Gault'''
    if sex.lower() in ['male', 'm']:
        crcl = ((140 - age) * weight_kg) / (72 * creatinine)
    elif sex.lower() in ['female', 'f']:
        crcl = ((140 - age) * weight_kg) / (72 * creatinine) * 0.85
    else:
        return None
    return round(crcl, 1)