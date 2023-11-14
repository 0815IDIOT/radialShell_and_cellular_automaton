from __future__ import print_function

import numpy as np
# from lmfit import minimize, Parameters, Parameter, report_fit
from numba import jit, typed, types, objmode
from scipy.optimize import fsolve

from constants import CMODE_P0_GRID, CMODE_P0_INFTY, CMODE_P0_R0


# ODE = Ordinary differential equation
# analytical calculation of oxygen diffusion
# see GriKelBloPar2013, Equation 2.1

# r0 = radius of tumor
# rn = radius of anoxic region
# rc = radius of viable region
# rl = maximum tumor radius, so that rn = 0 (r0 == rc)

# a = consumption rate
# p0 = oxygen concentration outside tumor
# pcrit = oxygen concentration below which cells are hypoxic

# extra for p0 reached at R
#@jit(nopython=True)
def cr_rn_(p, D, a, pcrit, r0, p0, R):
    rn = 2 * D / a * (pcrit - p) + r0**2
    if rn > 0:
        rn = np.sqrt(rn)
    else:
        rn = 0
    return p0 + a / 3 / D * (r0**3 - rn**3) / R - p


# Nanal = spatial resolution
# cannot use @jit (numba) because it has no scipy support
def c_r_solutions(r0, parameters, N_grid, mode=0, Nanal=1000, pcrit=None, consuming_fraction=1.0, R=None):
    if pcrit is None:
        pcrit = parameters["pcrit"]
        # pcrit = parameters["phypoxia"] #why is this here? p_hypoxia =/= p_crit !!

    # lower bound because function not defined for 0
    # interval from 0 to grid_max with Nanal values
    #r0 += parameters["ca_dx"] * 1e6
    rb = np.linspace(N_grid * parameters["ca_dx"], N_grid * parameters["ca_dx"] * 1e6, Nanal)

    # index of tumor boundary
    r0ind = int(r0 / rb[-1] * Nanal)

    p0 = parameters["p0"]
    D = parameters["D_p"] * (1e6)**2 / parameters["m"]**2
    a = parameters["ca_a"] * consuming_fraction

    # analytic solution for rotational symmetric colony
    if R == None:
        R = parameters["ca_dx"] * N_grid * 1e6
    #with objmode():
    #    print(R)
    #    print(parameters["ca_dx"] * N_grid * 1e6)
    cr = np.zeros((Nanal, 2), dtype=float)
    cr[:, 0] = rb

    if mode == CMODE_P0_GRID:
        p0 = fsolve(cr_rn_, [p0], args=(D, a, pcrit, r0, p0, R))[0]
        rn = r0**2 - 2 * D / a * (p0 - pcrit)
        if rn > 0:
            rn = np.sqrt(rn)
        else:
            rn = 0
        necrotic = cr[:, 0] <= rn
        living = (cr[:, 0] > rn) * (cr[:, 0] < r0)
        outside = cr[:, 0] >= r0
        cr[necrotic, 1] = pcrit
        cr[outside, 1] = p0 - a / 3 / D * (r0**3 - rn**3) / cr[outside, 0]
        cr[living, 1] = a / 3 / D * (cr[living, 0]**2 / 2 + rn**3 / cr[living, 0]) + p0 - a / 2 / D * r0**2

    elif mode == CMODE_P0_R0:
        rl = np.sqrt(6 * D * (p0 - pcrit) / a)  # diffusion limit (GriKelBloPar2014, equation 2.6)
        if rl <= r0:
            # GriKelBloPar2014, equation 2.7 and 2.9
            xi = (np.arccos(1 - 2 * rl**2 / r0**2) - 2 * np.pi) / 3
            rn = r0 * (0.5 - np.cos(xi))
        else:
            rn = 0
        necrotic = cr[:, 0] <= rn
        living = (cr[:, 0] > rn) * (cr[:, 0] < r0)
        outside = cr[:, 0] >= r0
        cr[necrotic, 1] = pcrit
        cr[living, 1] = p0 + a / 3 / D * (cr[living, 0]**2 / 2 - r0**2 / 2 + rn**3 / cr[living, 0] - rn**3 / r0)
        cr[outside, 1] = p0 + a / 3 / D * (r0**2 - r0**3 / cr[outside, 0] - rn**3 / r0 + rn**3 / cr[outside, 0])

    elif mode == CMODE_P0_INFTY:
        rn = r0**2 - 2 * D / a * (p0 - pcrit)
        if rn > 0:
            rn = np.sqrt(rn)
        else:
            rn = 0
        necrotic = cr[:, 0] <= rn
        living = (cr[:, 0] > rn) * (cr[:, 0] < r0)
        outside = cr[:, 0] >= r0
        cr[necrotic, 1] = pcrit
        cr[outside, 1] = p0 - a / 3 / D * (r0**3 - rn**3) / cr[outside, 0]
        cr[living, 1] = a / 3 / D * (cr[living, 0]**2 / 2 + rn**3 / cr[living, 0]) + p0 - a / 2 / D * r0**2
        # p0 - a/2/D * r0**2 is in theory equal to pcrit - a/2/D * rn**2
        # however this only works if rn is either positive, or allowed negative values
        # thus we use the p0 variant. following code will show you:
        # (hint: don't run the 10Gy experiment, tumor does not grow large enough to develop rn > 0 !!)
        # print(f"r0: {r0}, p0: {p0}")
        # print(f"rn: {rn}, pcrit: {pcrit}")
        # print(f"r0 variant: {p0 - a/2/D * r0**2}")
        # print(f"rn variant: {pcrit - a/2/D * rn**2}")

    canal = cr[:, 1]  # analytical solution for concentration
    return rb, canal


if __name__ == "__main__":
    print('None')
