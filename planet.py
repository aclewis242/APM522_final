import numpy as np

G = 4*np.pi**2              # Gravitational constant in astronomical units
interactionsAllowed = True  # Allow/disallow interplanetary interactions

class Planet:
    '''
    OOP representation of a planet.
    '''
    name = ""
    mass = 0
    period = 0
    incl = 0
    pos = np.zeros(3, dtype=float)
    vel = np.zeros(3)
    is_star = False
    r = 0
    clr = None

    # All non-initial position vectors are in Cartesian coordinates

    def __init__(self, n: str, m: float, T: float, i: float, r: float, c: str=None): # Constructor
        '''
        Builds a Planet object.

        ### Parameters
        n: Name
        m: Mass (solar masses)
        T: Orbital period (years)
        i: Inclination (degrees, relative to ecliptic)
        r: Orbital radius (AU)
        c: Hex color code (leave blank for no particular color)
        '''
        self.name = n
        self.mass = m
        self.period = T
        self.incl = np.deg2rad(i)
        self.r = r
        self.pos = np.array([r, 0.0, 0.0], dtype=float)
        self.vel = self.initVel
        self.clr = c
    
    def dist(self, m2: 'Planet'=None) -> float:
        '''
        Returns distance between two Planet objects, or current distance from the Sun.

        ### Parameters
        m2: The other Planet object (optional)
        '''
        if m2 is not None:
            return np.linalg.norm(self.pos - m2.pos)
        else:
            return np.linalg.norm(self.pos)

    @property
    def initVel(self) -> np.ndarray[float]:
        '''
        Finds the velocity of the planet at the start of the simulation. Orbits are treated as circular for this purpose.
        '''
        v = 2*np.pi*self.dist()/self.period
        return np.array([0, -v*np.cos(self.incl), v*np.sin(self.incl)])

    @property
    def spd(self) -> float:
        '''
        Returns current (scalar) speed of the planet.
        '''
        return np.linalg.norm(self.vel)
    
    @property
    def KE(self) -> float:
        '''
        Returns current kinetic energy of the planet.
        '''
        return 0.5*self.mass*self.spd**2
    
    def PE(self, parent: 'Planet') -> float:
        '''
        Returns current potential energy of the planet with respect to another planet.
        For all intents and purposes, that will always be the star at the center, but it could be repurposed for moons without much
        difficulty.

        ### Parameters
        parent: The other planet in question
        '''
        if parent==self: return 0
        return G*self.mass*parent.mass/self.dist(parent)
    
    def E_net(self, parent: 'Planet') -> float:
        '''
        Returns current net energy of the planet with respect to another planet.
        For all intents and purposes, that will always be the star at the center, but it could be repurposed for moons without much
        difficulty.

        ### Parameters
        parent: The other planet in question
        '''
        return self.KE + self.PE(parent)
    
    def L(self, parent: 'Planet') -> float:
        '''
        Returns current angular momentum of the planet with respect to another planet.
        For all intents and purposes, that will always be the star at the center, but it could be repurposed for moons without much
        difficulty.
        Originally written as a check on accuracy (conservation of momentum); however, as that role is filled by conservation of energy,
        it presently sees no use.

        ### Parameters
        parent: The other planet in question
        '''
        return np.linalg.norm(np.cross(self.pos-parent.pos, self.mass*self.vel))
    
    def Fg(self, m2: 'Planet') -> np.ndarray[float]:
        '''
        Returns gravitational force vector between two planets.

        ### Parameters
        m2: The other planet in question
        '''
        dvec = m2.pos - self.pos
        rv = np.zeros_like(self.pos)
        if self == m2: return rv
        if interactionsAllowed or (not self.is_star and m2.is_star):
            rv = G*self.mass*m2.mass/(self.dist(m2)**2)*(dvec/np.linalg.norm(dvec))
        return rv
    
    def Fg_net(self, plns: list['Planet']) -> np.ndarray[float]:
        '''
        Returns net gravitational force vector between a planet and a list of others (i.e., the system as a whole).

        ### Parameters
        plns: The list of other planets
        '''
        rv = np.zeros_like(self.pos, dtype=float)
        for p in plns: rv += self.Fg(p)
        return rv
    
    def acc(self, plns: list['Planet']) -> np.ndarray[float]:
        '''
        Returns acceleration vector of the planet using Newton's second law.

        ### Parameters
        plns: The list of other planets
        '''
        return self.Fg_net(plns)/self.mass
    
    def rebuild(self) -> 'Planet':
        '''
        Returns an identical copy of itself. Used to avoid memory address issues.
        '''
        p = Planet(self.name, self.mass, self.period, self.incl, self.r, self.clr)
        p.pos = self.pos.copy()
        p.vel = self.vel.copy()
        return p
    
    def __eq__(self, m2: 'Planet') -> bool:
        '''
        Checks if two Planet objects are the same. (Assumes that names are unique!)

        ### Parameters
        m2: The other Planet object
        '''
        return self.name == m2.name
    
    def __repr__(self) -> str:
        '''
        String representation of the planet (as its name). Allows for much easier debugging.
        '''
        return self.name

class Star(Planet):
    '''
    A subclass of Planet that represents the object at the center of the system (even if it isn't technically a star).
    '''
    def __init__(self, n: str, m: float, c: str=None):
        '''
        Builds a new Star object.

        ### Parameters
        n: Name
        m: Mass (in solar masses)
        c: Hex color code
        '''
        super().__init__(n=n, m=m, T=1, i=0, r=0)
        self.is_star = True
        self.clr = c
    
    def rebuild(self) -> 'Star':
        '''
        Returns an identical copy of itself. Used to avoid memory address issues.
        '''
        s = Star(self.name, self.mass, self.clr)
        s.pos = self.pos.copy()
        s.vel = self.vel.copy()
        return s