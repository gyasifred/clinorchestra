def calculate_body_surface_area(weight_kg, height_cm):
    '''Calculate BSA using Mosteller formula'''
    import math
    bsa = math.sqrt((weight_kg * height_cm) / 3600)
    return round(bsa, 3)