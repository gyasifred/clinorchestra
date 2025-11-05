def calculate_corrected_calcium(calcium, albumin):
    '''Calculate corrected calcium for hypoalbuminemia'''
    corrected_ca = calcium + 0.8 * (4.0 - albumin)
    return round(corrected_ca, 2)