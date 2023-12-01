import numpy as np
import planet as pln
from planet import DT
import matplotlib.pyplot as plt

### TODO
# Euler's
#   - base: r_n+1 = r_n + dt*v_n
#   - improved: refer to sept 27 pdf
# Heun's: refer to sept 27 pdf
# Runge-Kutta
#   - 3rd: refer to https://acadpubl.eu/jsi/2015-101-5-6-7-8/2015-101-8/21/21.pdf
#   - 4th: refer to sept 27 pdf
# Newton's: probably not?
# Adams-Bashforth: refer to sept 27 pdf

def euler(plns: list[pln.Planet], dt: float=DT):
    ps = np.array([])
    for p in plns:
        p.vel += dt*p.acc(plns)
        p.pos += dt*p.vel
        ps = np.append(ps, p.rebuild())
    return ps

def eulerImp(plns: list[pln.Planet], dt: float=DT):
    ps = np.array([])
    for p in plns:
        k1v = p.acc(plns)*dt
        p1 = p.rebuild()
        p1.vel += k1v
        k1r = p.vel*dt
        p1.pos += k1r
        k2v = p1.acc(plns)*dt
        p.vel += 0.5*(k1v + k2v)
        k2r = p1.vel*dt # maybe change to just 'p.vel*dt'? Gives better results, but it might not be the same algorithm
        p.pos += 0.5*(k1r + k2r)
        ps = np.append(ps, p.rebuild())
    return ps

def simulate(plns: list[pln.Planet], dt: float=DT, tmax: float=5.0, method=euler):
    ts = np.linspace(0, tmax, int(tmax/dt))
    all_ps = np.array([[p.rebuild() for p in plns]])
    for t in ts[1:]: all_ps = np.append(all_ps, [method(plns,dt)], axis=0)
    return all_ps, ts