def calculate_weight_change_percent(initial_weight, final_weight):
    '''Calculate percentage weight change'''
    if initial_weight == 0:
        return None
    change = ((final_weight - initial_weight) / initial_weight) * 100
    return round(change, 2)