import numpy as np
import planet as pln

def euler(plns: list[pln.Planet], dt: float, past=None) -> np.ndarray[pln.Planet]:
    '''
    Euler's method, governed by the expression y_n+1 = y_n + dt*y_n'.

    ### Parameters
    plns: The list of planets in the system
    dt: Timestep
    (past: Not used in this method. Included for shell compatibility with AB.)
    '''
    ps = np.array([])
    for p in plns:
        p.pos += dt*p.vel
        p.vel += dt*p.acc(plns)
        ps = np.append(ps, p.rebuild())
    return ps

def eulerImp(plns: list[pln.Planet], dt: float, past=None) -> np.ndarray[pln.Planet]:
    '''
    Improved Euler's method, governed by the expression y_n+1 = y_n + 0.5(k1 + k2).

    ### Parameters
    plns: The list of planets in the system
    dt: Timestep
    (past: Not used in this method. Included for shell compatibility with AB.)
    '''
    ps = np.array([])
    for p in plns:
        k1v = p.acc(plns)*dt
        p1 = p.rebuild()
        p1.vel += k1v
        k1r = p.vel*dt
        p1.pos += k1r
        k2v = p1.acc(plns)*dt
        p.vel += 0.5*(k1v + k2v)
        k2r = p1.vel*dt
        p.pos += 0.5*(k1r + k2r)
        ps = np.append(ps, p.rebuild())
    return ps

def heun(plns: list[pln.Planet], dt: float, past=None) -> np.ndarray[pln.Planet]:
    '''
    Heun's method, a variation on improved Euler governed by the formula y_n+1 = y_n + 0.25(k1 + 3k2).

    ### Parameters
    plns: The list of planets in the system
    dt: Timestep
    (past: Not used in this method. Included for shell compatibility with AB.)
    '''
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

def rk3(plns: list[pln.Planet], dt: float, past=None) -> np.ndarray[pln.Planet]:
    '''
    3rd-order Runge-Kutta method, governed by the formula [blah].

    ### Parameters
    plns: The list of planets in the system
    dt: Timestep
    (past: Not used in this method. Included for shell compatibility with AB.)
    '''
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

def rk4(plns: list[pln.Planet], dt: float, past=None) -> np.ndarray[pln.Planet]:
    '''
    4th-order Runge-Kutta method, governed by the expression y_n+1 = y_n + (k1 + 2k2 + 2k3 + k4)/6.

    ### Parameters
    plns: The list of planets in the system
    dt: Timestep
    (past: Not used in this method. Included for shell compatibility with AB.)
    '''
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

def ab2(plns: list[pln.Planet], dt: float, past: list[np.ndarray[pln.Planet]]) -> np.ndarray[pln.Planet]:
    '''
    2nd-order Adams-Bashforth method, governed by the expression y_n+1 = y_n + 0.5dt(3y_n' - y_n-1').

    ### Parameters
    plns: The list of planets in the system
    dt: Timestep
    past: The last iteration's results. Initialised using RK4.
    '''
    ps = np.array([])
    p1s = past[-1]
    for i in range(len(plns)):
        p, p1 = plns[i], p1s[i]
        k0v = p.acc(plns)*dt
        k0r = p.vel*dt
        k1v = p1.acc(p1s)*dt
        k1r = p1.vel*dt
        p.vel += 0.5*(3*k0v - k1v)
        p.pos += 0.5*(3*k0r - k1r)
        ps = np.append(ps, p.rebuild())
    return ps

def ab3(plns: list[pln.Planet], dt: float, past: list[np.ndarray[pln.Planet]]) -> np.ndarray[pln.Planet]:
    '''
    3rd-order Adams-Bashforth method, governed by the expression y_n+1 = y_n + dt(23y_n' - 16y_n-1' + 5y_n-2')/12.

    ### Parameters
    plns: The list of planets in the system
    dt: Timestep
    past: The last two iterations' results. Initialised using RK4.
    '''
    ps = np.array([])
    p1s = past[-1]
    p2s = past[-2]
    for i in range(len(plns)):
        p, p1, p2 = plns[i], p1s[i], p2s[i]
        k0v = p.acc(plns)*dt
        k0r = p.vel*dt
        k1v = p1.acc(p1s)*dt
        k1r = p1.vel*dt
        k2v = p2.acc(p2s)*dt
        k2r = p2.vel*dt
        p.vel += (23*k0v - 16*k1v + 5*k2v)/12
        p.pos += (23*k0r - 16*k1r + 5*k2r)/12
        ps = np.append(ps, p.rebuild())
    return ps

def simulate(plns: list[pln.Planet], dt: float, tmax: float=5.0, method=euler) -> tuple:
    '''
    Simulation shell. Runs the specified method over the specified time for the specified system.

    ### Parameters
    plns: The list of planets in the system
    dt: Timestep
    tmax: Max. time
    method: Iterative method to use

    ### Returns
    all_ps: An array of system lists for each t
    ts: A list of t values
    '''

    ts = np.linspace(0, tmax, int(tmax/dt))
    all_ps = np.array([[p.rebuild() for p in plns]])
    init_num = 1
    if method.__name__[:2] == 'ab': init_num = int(method.__name__[-1])
    if init_num != 1:
        for t in ts[1:init_num]: all_ps = np.append(all_ps, [rk4(plns,dt)], axis=0)
    for t in ts[init_num:]: all_ps = np.append(all_ps, [method(plns,dt,all_ps[-init_num:-1])], axis=0)
    return all_ps, ts