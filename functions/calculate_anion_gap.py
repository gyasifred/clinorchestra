def calculate_anion_gap(sodium, chloride, bicarbonate):
    '''Calculate serum anion gap'''
    ag = sodium - (chloride + bicarbonate)
    return round(ag, 1)