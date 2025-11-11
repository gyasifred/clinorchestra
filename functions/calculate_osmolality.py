def calculate_osmolality(sodium, glucose, bun):
    '''Calculate serum osmolality'''
    osm = 2 * sodium + (glucose / 18) + (bun / 2.8)
    return round(osm, 1)