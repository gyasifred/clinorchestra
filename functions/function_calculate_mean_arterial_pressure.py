def calculate_mean_arterial_pressure(systolic, diastolic):
    '''Calculate MAP from BP readings'''
    map_value = (systolic + 2 * diastolic) / 3
    return round(map_value, 1)