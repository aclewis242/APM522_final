import numpy as np
import matplotlib.pyplot as plt
import planet as pln
import axes
from lib import *
from syslib import *

def mainShell(m_c=euler, sys=solar, dt=2e-2, tmax=2.0, doInts=True):
    '''
    The main method shell. Contains the bulk of the code, and is separate to allow for multiple methods to be run at once.

    ### Parameters
    m_c: The method to use
    sys: The planetary system to use
    dt: Timestep
    tmax: Max. time
    doInts: Allow/disallow interplanetary interactions
    '''
    method_choice = m_c

    plns = sys()
    to_plot = plns
    if sys.__name__ == 'solar':
        inns = plns[:5]
        mids = plns[:6]
        outs = plns[5:]
        outs.append(plns[0])
        to_plot = plns # Which planets to plot (inner, mid, outer, or plns for all)

    pln.interactionsAllowed = doInts
    all_ps, ts = simulate(plns, dt=dt, tmax=tmax, method=method_choice)
    all_ps = all_ps.T

    ### GRAPH POSITION
    ax = plt.axes(projection='3d')
    for all_p in all_ps:
        all_pos = np.array([p.pos for p in all_p]).T
        if all_p[0] in to_plot: ax.plot3D(all_pos[0], all_pos[1], all_pos[2], label=all_p[0].name, color=all_p[0].clr)

    ax.set_title(rf'Orbital trajectories: {methods[method_choice]}')
    ax.set_xlabel(r'$x$ (AU)')
    ax.set_ylabel(r'$y$ (AU)')
    ax.set_zlabel(r'$z$ (AU)')
    ax.legend()
    axes.set_axes_equal(ax)
    plt.savefig(f'{method_choice.__name__}/{sys.__name__}_trajectories.png')
    plt.show()

    ### GRAPH ENERGY
    total_E = np.zeros_like(ts)
    for all_p in all_ps:
        all_Es = np.array([p.E_net(all_p[0]) for p in all_p])
        total_E += all_Es
        if all_p[0] in to_plot: plt.plot(ts, all_Es, label=all_p[0].name, color=all_p[0].clr)
    plt.plot(ts, total_E, label='Total energy')
    plt.title(rf'Energies: {methods[method_choice]}')
    plt.xlabel(r'$t$ (years)')
    plt.ylabel(r'$E$ ($M_{sun}(\frac{AU}{yr})^2$)')
    plt.legend()
    plt.savefig(f'{method_choice.__name__}/{sys.__name__}_energies.png')
    plt.show()

    ### RELATIVE ERROR
    rel_err = np.log(np.abs(total_E/total_E[0] - 1))
    print(f'Ultimate relative error ({method_choice.__name__}): {round(np.mean(rel_err[-10:])*1000)/1000}')
    plt.plot(ts, rel_err)
    plt.title(rf'Relative error: {methods[method_choice]}')
    plt.xlabel(r'$t$ (years)')
    plt.ylabel(r'$\ln\epsilon$')
    plt.savefig(f'{method_choice.__name__}/{sys.__name__}_rel_err.png')
    plt.show()

    return [ts, rel_err]

methods = {euler: 'Euler\'s method',
           eulerImp: 'Improved Euler\'s method',
           heun: 'Heun\'s method',
           rk3: 'Runge-Kutta (3rd order)',
           rk4: 'Runge-Kutta (4th order)',
           ab2: 'Adams-Bashforth (2nd order)',
           ab3: 'Adams-Bashforth (3rd order)'}

                                            # Recommended tmaxs/dts:
systems = {solar: 'Solar system',           # 2.0/2e-2
           inns: 'Inner solar system',      # 2.0/2e-2
           mids: 'Mid solar system',        # 12.0/2e-2
           outs: 'Outer solar system',      # 200.0/2.0
           jup_moons: 'Jupiter\'s moons'}   # 1.0/2e-4

if __name__ == '__main__':
    do_all = True # Whether or not to run all simulations at once
    # Choose all of the following from the above
    method_choice = rk4
    sys = jup_moons
    tmax = 1.0
    dt = 2e-4
    doInts = True # Allow/disallow interplanetary interactions
    names = [method_choice.__name__]

    if do_all:
        all_res = np.array([mainShell(m_c, sys, dt, tmax, doInts) for m_c in methods])
        names = [m_c.__name__ for m_c in methods]
        ts = all_res[0,0]
        rel_errs = all_res[:,1]
        for i in range(len(rel_errs)): plt.plot(ts, rel_errs[i], label=names[i])
        plt.title(r'Relative errors -- multiple methods')
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\ln\epsilon$')
        plt.legend()
        plt.savefig(f'{sys.__name__}_all_rel_errs.png')
        plt.show()
    else: mainShell(method_choice, sys, dt, tmax, doInts)