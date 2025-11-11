def calculate_pack_years(packs_per_day, years_smoked):
    '''Calculate smoking pack-years'''
    pack_years = packs_per_day * years_smoked
    return round(pack_years, 1)