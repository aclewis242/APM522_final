import numpy as np
import matplotlib.pyplot as plt
import planet as pln
import axes
from lib import *

# Astronomical units used (i.e. mass in terms of solar masses, time in terms of years, distance in terms of AU)
D = 1/365 # Length of a day in years
M = 3e-6  # Mass of Earth in solar masses

if __name__ == '__main__':
    # Initialise solar system
    sun = pln.Star('Sun', 1.0)
    merc = pln.Planet('Mercury', 0.055*M, 87.9691*D, 7.005, 0.387098)
    venus = pln.Planet('Venus', 0.815*M, 224.701*D, 3.39458, 0.723332)
    earth = pln.Planet('Earth', M, 365*D, 0.0, 1.0)
    mars = pln.Planet('Mars', 0.107*M, 686.98*D, 1.85, 1.523681)
    jup = pln.Planet('Jupiter', 317.8*M, 11.862, 1.303, 5.2038)
    sat = pln.Planet('Saturn', 95.159*M, 29.4475, 2.485, 9.5826)
    uranus = pln.Planet('Uranus', 14.536*M, 84.0205, 0.773, 19.19126)
    nept = pln.Planet('Neptune', 17.147*M, 164.8, 1.77, 30.07)

    plns = [sun, merc, venus, earth, mars, jup, sat, uranus, nept]
    inns = plns[:5]
    mids = plns[:6]
    outs = plns[5:]
    outs.append(sun)

    ### USER MODIFICATIONS BEGIN ###
    methods = {euler: 'Euler\'s method',
               eulerImp: 'Improved Euler\'s method',
               heun: 'Heun\'s method',
               rk3: 'Runge-Kutta (3rd order)'}
    method_choice = rk3        # Choose from one of the above
    to_plot = inns                  # Which planets to plot (inner, mid, outer, or plns for all)
    pln.interactionsAllowed = True  # Allow/disallow interplanetary interactions
    dt = 2e-3                   # Set time step
    tmax = 2.0                      # Set cutoff time
    ### USER MODIFICATIONS END ###

    all_ps, ts = simulate(plns, dt=dt, tmax=tmax, method=method_choice)
    all_ps = all_ps.T
    # [print(ap.pos) for ap in all_ps[1]]

    ### GRAPH POSITION
    ax = plt.axes(projection='3d')
    for all_p in all_ps:
        all_pos = np.array([p.pos for p in all_p]).T
        if all_p[0] in to_plot: ax.plot3D(all_pos[0], all_pos[1], all_pos[2], label=all_p[0].name)

    ax.set_title(rf'Orbital trajectories: {methods[method_choice]}')
    ax.set_xlabel(r'$x$ (AU)')
    ax.set_ylabel(r'$y$ (AU)')
    ax.set_zlabel(r'$z$ (AU)')
    ax.legend()
    axes.set_axes_equal(ax)
    plt.savefig(f'{method_choice.__name__}/trajectories_{pln.interactionsAllowed}.png')
    plt.show()

    ### GRAPH ENERGY
    total_E = np.zeros_like(ts)
    for all_p in all_ps:
        all_Es = np.array([p.E_net(all_p[0]) for p in all_p])
        total_E += all_Es
        if all_p[0] in to_plot: plt.plot(ts, all_Es, label=all_p[0].name)
    plt.plot(ts, total_E, label='Total energy')
    plt.title(rf'Energies: {methods[method_choice]}')
    plt.xlabel(r'$t$ (years)')
    plt.ylabel(r'$E$ ($M_{sun}(\frac{AU}{yr})^2$)')
    plt.legend()
    plt.savefig(f'{method_choice.__name__}/energies_{pln.interactionsAllowed}.png')
    plt.show()

    ### RELATIVE ERROR
    plt.plot(ts, np.log(np.abs(total_E/total_E[0] - 1)))
    # [print(E/total_E[0]) for E in total_E]
    plt.title(rf'Relative error: {methods[method_choice]}')
    plt.xlabel(r'$t$ (years)')
    plt.ylabel(r'$\epsilon$')
    plt.savefig(f'{method_choice.__name__}/rel_err_{pln.interactionsAllowed}.png')
    plt.show()