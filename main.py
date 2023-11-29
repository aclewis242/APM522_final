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
    # [insert mercury here]
    # [insert venus here]
    earth = pln.Planet('Earth', M, 365*D, 0.0, 1.0)
    # [insert mars here]
    # [insert jupiter here]
    # [insert saturn here]
    # [insert uranus here]
    # [insert neptune here]

    plns = [sun, earth]

    methods = {euler: 'Euler\'s method',
               eulerImp: 'Improved Euler\'s method'}
    method_choice = euler # Choose from one of the above

    all_ps, ts = simulate(plns, method=method_choice)
    all_ps = all_ps.T

    ### GRAPH POSITION
    ax = plt.axes(projection='3d')
    for all_p in all_ps:
        all_pos = np.array([p.pos for p in all_p]).T
        # [print(f'{p.name}: {hex(id(p))}') for p in all_p]
        # print(f'Coords for {all_p[0].name}: {all_pos}')
        # if not all_p[0].is_star:
        ax.plot3D(all_pos[0], all_pos[1], all_pos[2], label=all_p[0].name)
        # else: ax.scatter3D(all_p[0].pos[0], all_p[0].pos[1], all_p[0].pos[2], label=all_p[0].name, color='k')

    ax.set_title(rf'Orbital trajectories: {methods[method_choice]}')
    ax.set_xlabel(r'$x$ (AU)')
    ax.set_ylabel(r'$y$ (AU)')
    ax.set_zlabel(r'$z$ (AU)')
    ax.legend()
    axes.set_axes_equal(ax)
    plt.savefig('fig.png')
    plt.show()