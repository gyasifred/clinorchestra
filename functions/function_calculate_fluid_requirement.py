def calculate_fluid_requirement(weight_kg):
    '''Calculate daily maintenance fluid requirement (Holliday-Segar)'''
    if weight_kg <= 10:
        fluid = weight_kg * 100
    elif weight_kg <= 20:
        fluid = 1000 + (weight_kg - 10) * 50
    else:
        fluid = 1500 + (weight_kg - 20) * 20
    return round(fluid, 0)