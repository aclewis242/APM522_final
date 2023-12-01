import numpy as np
import planet as pln
from planet import DT
import matplotlib.pyplot as plt

### TODO
# Euler's
#   - base: r_n+1 = r_n + dt*v_n DONE
#   - improved: refer to sept 27 pdf DONE
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

def heun(plns: list[pln.Planet], dt: float=DT):
    ps = np.array([])
    for p in plns:
        k1v = p.acc(plns)*dt
        p1 = p.rebuild()
        p1.vel += 2*k1v/3
        k1r = p.vel*dt
        p1.pos += 2*k1r/3
        k2v = p1.acc(plns)*dt
        p.vel += 0.25*k1v + 0.75*k2v
        k2r = p1.vel*dt
        p.pos += 0.25*k1r + 0.75*k2r
        ps = np.append(ps, p.rebuild())
    return ps

def rk3(plns: list[pln.Planet], dt: float=DT):
    ps = np.array([])
    for p in plns:
        p1 = p.rebuild()
        k1v = p.acc(plns)*dt
        p1.vel += 0.5*k1v
        k1r = p.vel*dt
        p1.pos += 0.5*k1r
        p2 = p.rebuild()
        k2v = p1.acc(plns)*dt
        p2.vel += -k1v + 2*k2v
        k2r = p1.vel*dt
        p2.pos += -k1r + 2*k2r
        k3v = p2.acc(plns)*dt
        p.vel += (k1v + 4*k2v + k3v)/6
        k3r = p2.vel*dt
        p.pos += (k1r + 4*k2r + k3r)/6
        ps = np.append(ps, p.rebuild())
    return ps

def rk4(plns: list[pln.Planet], dt: float=DT):
    ps = np.array([])
    for p in plns:
        p1 = p.rebuild()
        k1v = p.acc(plns)*dt
        p1.vel += 0.5*k1v
        k1r = p.vel*dt
        p1.pos += 0.5*k1r
        p2 = p.rebuild()
        k2v = p1.acc(plns)*dt
        p2.vel += 0.5*k2v
        k2r = p1.vel*dt
        p2.pos += 0.5*k2r
        p3 = p.rebuild()
        k3v = p2.acc(plns)*dt
        p3.vel += k3v
        k3r = p2.vel*dt
        p3.pos += k3r
        k4v = p3.acc(plns)*dt
        p.vel += (k1v + 2*k2v + 2*k3v + k4v)/6
        k4r = p3.vel*dt
        p.pos += (k1r + 2*k2r + 2*k3r + k4r)/6
        ps = np.append(ps, p.rebuild())
    return ps

def simulate(plns: list[pln.Planet], dt: float=DT, tmax: float=5.0, method=euler):
    ts = np.linspace(0, tmax, int(tmax/dt))
    all_ps = np.array([[p.rebuild() for p in plns]])
    for t in ts[1:]: all_ps = np.append(all_ps, [method(plns,dt)], axis=0)
    return all_ps, ts