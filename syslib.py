import planet as pln

# Astronomical units used (i.e. mass in terms of solar masses, time in terms of years, distance in terms of AU)
D = 1/365       # Length of a day in years
M = 3e-6        # Mass of Earth in solar masses
KM = 6.6846e-9  # 1 km, in AU

def solar() -> list[pln.Planet]:
    '''
    The solar system, represented as a list of Planet objects.
    '''
    sun = pln.Star('Sun', 1.0, '#FFF6AA')
    merc = pln.Planet('Mercury', 0.055*M, 87.9691*D, 7.005, 0.387098, '#AFAFAF')
    venus = pln.Planet('Venus', 0.815*M, 224.701*D, 3.39458, 0.723332, '#FFCE79')
    earth = pln.Planet('Earth', M, 365*D, 0.0, 1.0, '#00A544')
    mars = pln.Planet('Mars', 0.107*M, 686.98*D, 1.85, 1.523681, '#FF5516')
    jup = pln.Planet('Jupiter', 317.8*M, 11.862, 1.303, 5.2038, '#DD8665')
    sat = pln.Planet('Saturn', 95.159*M, 29.4475, 2.485, 9.5826, '#FFC175')
    uranus = pln.Planet('Uranus', 14.536*M, 84.0205, 0.773, 19.19126, '#ACEAFF')
    nept = pln.Planet('Neptune', 17.147*M, 164.8, 1.77, 30.07, '#0072D1')

    return [sun, merc, venus, earth, mars, jup, sat, uranus, nept]

def inns() -> list[pln.Planet]:
    '''
    The inner solar system (up to Mars), represented as a list of Planet objects.
    '''
    return solar()[:5]

def mids() -> list[pln.Planet]:
    '''
    The mid solar system (Venus-Jupiter), represented as a list of Planet objects.
    '''
    s = solar()
    rv = [s[0]]
    [rv.append(p) for p in s[2:6]]
    return rv

def outs() -> list[pln.Planet]:
    '''
    The outer solar system (Jupiter-Neptune), represented as a list of Planet objects.
    '''
    s = solar()
    rv = [s[0]]
    [rv.append(p) for p in s[5:]]
    return  rv

def jup_moons() -> list[pln.Planet]:
    '''
    Jupiter's moons (Galilean, specifically), represented as a list of Planet objects.
    '''
    jup = pln.Star('Jupiter', 317.8*M, '#DD8665')
    io = pln.Planet('Io', 0.015*M, 1.769*D, 2.213, 0.0028189, '#EEC600')
    eur = pln.Planet('Europa', 0.008*M, 3.5512*D, 1.791, 670900*KM, '#80A0B1')
    gany = pln.Planet('Ganymede', 0.025*M, 7.1546*D, 2.214, 1.07e6*KM, '#AFAFAF')
    call = pln.Planet('Callisto', 0.018*M, 16.689*D, 2.017, 1.883e6*KM, '#A0B9A1')

    return [jup, io, eur, gany, call]