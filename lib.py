import numpy as np
import planet as pln
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

def euler(plns: list[pln.Planet], dt: float=2e-4):
    ps = np.array([])
    for p in plns:
        p.vel += p.dvel(plns, dt)
        p.pos += p.vel*dt
        ps = np.append(ps, p.rebuild())
    # exit()
    return ps

def eulerImp(plns: list[pln.Planet], dt: float=2e-4):
    ps = np.array([])
    for p in plns:
        p1 = p.rebuild()
        p1.vel += p1.dvel(plns, dt)
        v1 = p1.vel
        p1.pos += v1*dt
        p2 = p1.rebuild()
        p2.vel += p2.dvel(plns, dt)
        v2 = p2.vel
        p2.pos += v2*dt
        p.vel += 0.5*(v1 + v2)
        p.pos += p.vel*dt
        ps = np.append(ps, p.rebuild())
    return ps

def simulate(plns: list[pln.Planet], dt: float=2e-4, tmax: float=2, method=euler):
    ts = np.linspace(0, tmax, int(tmax/dt))
    all_ps = np.array([plns])
    for t in ts: all_ps = np.append(all_ps, [method(plns,dt)], axis=0)
    return all_ps, ts