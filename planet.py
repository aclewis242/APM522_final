import numpy as np
from typing import Type

G = 4*np.pi**2 # Gravitational constant in astronomical units
interactionsAllowed = True

class Planet:
    name = ""
    mass = 0
    period = 0
    incl = 0
    pos = np.zeros(3, dtype=float)
    vel = np.zeros(3)
    is_star = False
    r = 0

    # All non-initial position vectors are in Cartesian coordinates

    def __init__(self, n: str, m: float, T: float, i: float, r: float): # Constructor
        '''
        Builds a Planet object.

        ### Parameters
        n: Name
        m: Mass
        T: Orbital period
        i: Inclination
        r: Orbital radius
        '''
        self.name = n
        self.mass = m
        self.period = T
        self.incl = np.deg2rad(i)
        self.r = r
        self.pos = np.array([r, 0.0, 0.0], dtype=float)
        self.vel = self.initVel()
    
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
    
    def initVel(self): # Orbits treated as circular for the purposes of finding initial velocity vector
        v = 2*np.pi*self.dist()/self.period
        return np.array([0, -v*np.cos(self.incl), v*np.sin(self.incl)])

    def spd(self): # Returns scalar speed of the planet
        return np.linalg.norm(self.vel)
    
    def KE(self): # Returns kinetic energy of the planet
        return 0.5*self.mass*self.spd()**2
    
    def PE(self, parent: 'Planet'): # Returns potential energy of the planet with respect to its parent
        return G*self.mass*parent.mass/self.dist(parent)
    
    def L(self, parent: 'Planet'): # Returns angular momentum of the planet with respect to its parent
        return np.linalg.norm(np.cross(self.pos-parent.pos, self.mass*self.vel))
    
    def Fg(self, m2: 'Planet'): # Returns gravitational force vector between the two planets
        dvec = m2.pos - self.pos
        rv = np.zeros_like(self.pos)
        if self == m2: return rv
        if interactionsAllowed or (not self.is_star and m2.is_star):
            rv = G*self.mass*m2.mass/(self.dist(m2)**2)*(dvec/np.linalg.norm(dvec))
        return rv
    
    def Fg_net(self, plns: list['Planet']) -> np.ndarray[float]:
        rv = np.zeros_like(self.pos, dtype=float)
        for p in plns: rv += self.Fg(p)
        # if self.name == 'Earth': print(rv/self.mass)
        return rv
    
    def dvel(self, plns: list['Planet'], dt: float=2e-4) -> np.ndarray[float]:
        return self.Fg_net(plns)/self.mass*dt
    
    def rebuild(self): # Returns an identical copy of the object. Used to avoid memory address issues
        p = Planet(self.name, self.mass, self.period, self.incl, self.r)
        p.pos = self.pos.copy()
        p.vel = self.vel.copy()
        return p
    
    def __eq__(self, m2: 'Planet') -> bool: # Overloads == and != operators
        '''
        Checks if two Planet objects are the same. (Assumes that names are unique!)

        ### Parameters
        m2: The other Planet object
        '''
        return self.name == m2.name
    
    def __repr__(self) -> str:
        return self.name

class Star(Planet): # A variation of the generic "Planet" class that initialises a stationary star at the center of the solar system
    def __init__(self, n: str, m: float):
        super().__init__(n=n, m=m, T=1, i=0, r=0)
        self.is_star = True
    
    def rebuild(self):
        s = Star(self.name, self.mass)
        s.pos = self.pos.copy()
        s.vel = self.vel.copy()
        return s