import argparse

import numpy.random as npr
from scipy import interpolate
from numba import jit, typed, types, objmode
import matplotlib.pyplot as plt
import time
import matplotlib.animation as animation

import sys
import os

from sphericalODE import c_r_solutions
import BruZieRivOelHaa2019
from plot_stuff import *
from constants import *

sys.path.insert(0,'../../.')
from tools import *
from o2_models import o2_models
from ode_setVolume import *
import models
from ode_plots import *
from CONSTANTS import *
import random

@jit(nopython=True)
def move_cells_toward_center_one_axis(chosen_cell, grid, cells_pos, cells, parameters, axis):
    dist0 = np.sqrt((cells_pos[chosen_cell, 0] - parameters["ca_centerx"])**2 +
                    (cells_pos[chosen_cell, 1] - parameters["ca_centery"])**2 +
                    (cells_pos[chosen_cell, 2] - parameters["ca_centerz"])**2)

    size = cells[chosen_cell, CELL_ATTR_SIZE]
    free_neighbor = False
    for neigh in NEIGHBORHOOD_NEUMANN1:
        if np.sum(neigh * axis) != 0:
            x_, y_, z_ = cells_pos[chosen_cell] + neigh
            dist = np.sqrt((x_ - parameters["ca_centerx"])**2 +
                            (y_ - parameters["ca_centery"])**2 +
                            (z_ - parameters["ca_centerz"])**2)
            if grid[x_, y_, z_, GRID_ATTR_FRAC_SPACE] + size <= 1 and dist < dist0:
                free_neighbor = True
                x, y, z = x_, y_, z_
                dist0 = dist  # update distance

    if free_neighbor:
        x_, y_, z_ = cells_pos[chosen_cell]
        # closer to center of tumor
        # move to free position
        grid[x, y, z, GRID_ATTR_FRAC_SPACE] = grid[x, y, z, GRID_ATTR_FRAC_SPACE] + size
        grid[x, y, z, GRID_ATTR_FRAC_CONSUME] = grid[x_, y_, z_, GRID_ATTR_FRAC_CONSUME]
        grid[x, y, z, GRID_ATTR_INDEX] = chosen_cell
        # remove from old position
        grid[x_, y_, z_, GRID_ATTR_FRAC_SPACE] = grid[x_, y_, z_, GRID_ATTR_FRAC_SPACE] - cells[chosen_cell, CELL_ATTR_SIZE]
        grid[x_, y_, z_, GRID_ATTR_FRAC_CONSUME] = 0
        grid[x_, y_, z_, GRID_ATTR_INDEX] = 0
        cells_pos[chosen_cell, :] = x, y, z


# shuffle cell inwards in order to maintain a dense spherical structure
@jit(nopython=True)
def move_cells_toward_center(chosen_cells, grid, cells_pos, cells, parameters, neighborhood):
    # possible meaning of inward shuffle from BruZieRivOelHaa2019
    for chosen_cell in chosen_cells:
        dist0 = np.sqrt((cells_pos[chosen_cell, 0] - parameters["ca_centerx"])**2 +
                        (cells_pos[chosen_cell, 1] - parameters["ca_centery"])**2 +
                        (cells_pos[chosen_cell, 2] - parameters["ca_centerz"])**2)
        size = cells[chosen_cell, CELL_ATTR_SIZE]
        free_neighbor = False
        for neigh in neighborhood:
            x_, y_, z_ = cells_pos[chosen_cell] + neigh
            dist = np.sqrt((x_ - parameters["ca_centerx"])**2 +
                           (y_ - parameters["ca_centery"])**2 +
                           (z_ - parameters["ca_centerz"])**2)
            if grid[x_, y_, z_, GRID_ATTR_FRAC_SPACE] + size <= 1 and dist < dist0:
                free_neighbor = True
                x, y, z = x_, y_, z_
                dist0 = dist  # update distance
        if free_neighbor:
            x_, y_, z_ = cells_pos[chosen_cell]
            # closer to center of tumor
            # move to free position
            grid[x, y, z, GRID_ATTR_FRAC_SPACE] = grid[x, y, z, GRID_ATTR_FRAC_SPACE] + size
            grid[x, y, z, GRID_ATTR_FRAC_CONSUME] = grid[x_, y_, z_, GRID_ATTR_FRAC_CONSUME]
            grid[x, y, z, GRID_ATTR_INDEX] = chosen_cell
            # remove from old position
            grid[x_, y_, z_, GRID_ATTR_FRAC_SPACE] = grid[x_, y_, z_, GRID_ATTR_FRAC_SPACE] - cells[chosen_cell, CELL_ATTR_SIZE]
            grid[x_, y_, z_, GRID_ATTR_FRAC_CONSUME] = 0
            grid[x_, y_, z_, GRID_ATTR_INDEX] = 0
            cells_pos[chosen_cell, :] = x, y, z


@jit(nopython=True)
def inward_shuffle(grid, cells, cells_pos, cell_indices, parameters, neighborhood):
    center = np.array([parameters['ca_centerx'], parameters['ca_centery'], parameters['ca_centerz']])

    # calculate distance of each grid point to center
    # because we use cell_indices to index cells_pos, the result has the same order as cell_indices
    # we use cell_indices so we are limited to only non-free cells
    with objmode(sorted_indices='intp[:]'):
        cell_distances = np.linalg.norm(cells_pos[cell_indices] - center[np.newaxis, :], axis=-1)
        # get indices that *would* sort the array
        # they sort ascending (closest distance FIRST)
        sorting = np.argsort(cell_distances)
        # then sort not the distances, but the INDICES by that sorting
        sorted_indices = np.array(cell_indices)[sorting]

    # move cells inward, starting at the center, moving outward
    move_cells_toward_center(sorted_indices, grid, cells_pos, cells, parameters, neighborhood)


# calculates analytical solution for oxygen concentration across grid
# writes values calculated by `c_r_solutions` into our data structures
def analytical_stationary_solution(concentration, ncells, parameters, only_boundary=False, consuming_fraction=1.0, R=None):
    # initialize concentration field by analytic solution
    # assume proliferating rim
    Radius0 = (3.0 / 4 / np.pi * ncells)**(1.0 / 3)
    center = np.array([parameters["ca_centerx"], parameters["ca_centery"], parameters["ca_centerz"]])
    # factor>1 --> analytical solution for more remote boundary,
    # consequently values on boundary are not constant but resemble
    # more realistic distribution
    # NOTE: factor should be >=1 in order to ensure that analytical
    # solution is at least provided on the whole grid
    factor = 200.0 / concentration.shape[0]

    # rb = steps along the radius until grid end
    # cr = concentration at that radius
    rb, cr = c_r_solutions(Radius0 * parameters["ca_dx"] * 1e6,
        parameters,
        concentration.shape[0] * 0.5 * factor,
       mode=parameters["ca_cmode"],
       Nanal=1000,
       pcrit=parameters["pcrit"],
       consuming_fraction=consuming_fraction,
       R=R)

    # magic, basically creates a distance vector from origin for each cell
    # ind.shape = (3,100,100,100) => values appear 4 arrays in => [[[[0,0,0]]]]
    ind = np.indices(concentration[:, :, :, 0].shape)
    ind = np.moveaxis(ind, 0, -1)  # magic
    # dx * distance to center
    # np.newaxis nests the values: [x,y,z] => [[[[x,y,z]]]] so it's compatible with the shape of ind
    Radius = parameters["ca_dx"] * np.linalg.norm(ind - center[np.newaxis, np.newaxis, np.newaxis, :], axis=-1) * 1e6

    # interpolate between cr values depending on radius
    fc = interpolate.interp1d(rb, cr)

    if only_boundary:
        tmp_ind = (np.amin(ind, axis=-1) == 0) + (np.amax(ind, axis=-1) == concentration.shape[0] - 1)
        concentration[:, :, :, 0][tmp_ind] = fc(Radius[tmp_ind])  # ?? Radius is not an array (or is it?)
    else:
        # set concentrations between min and max known radius
        concentration[:, :, :, 0][(Radius < rb[-1]) * (Radius >= rb[0])] = fc(Radius[(Radius < rb[-1]) * (Radius >= rb[0])])
        concentration[:, :, :, 0][Radius < rb[0]] = cr[0]  # default for cells below min known radius
        concentration[:, :, :, 0][Radius >= rb[-1]] = cr[-1]  # default for cells outside max known radius


# updates oxygen concentration in spheroid (solve numerically by iteration)
# this function == GriKelBloPar2013, equation 2.1
@jit(nopython=True)
def c_stationary(c_full, Lc_full, grid_full, ncells, cells_pos, cell_indices, consumption_rate, Dcoeff, parameters, R):

    consuming_fraction = 1.0

    with objmode():  # use Python Interpretor (because numba)
        # start_time = time.time()
        analytical_stationary_solution(c_full, ncells, parameters, consuming_fraction=consuming_fraction, only_boundary=False, R=R)
        # end_time = time.time()
        # print("Time elapsed for analytical solution:", end_time - start_time)

    # if using VICINITY mode, don't use full grid
    if abs(parameters["ca_cperformance"] - CPERFORMANCE_VICINITY) < 0.5:
        # Performance: only compute within vicinity of spheroid
        fac = 1.2  # calculate a bit outside of the spheroid
        R0 = fac * (3.0 / 4 / np.pi * ncells)**(1.0 / 3)  # extended radius of the spheroid

        if fac * parameters["ca_Rmax"] > R0:
            R0 = fac * parameters["ca_Rmax"]

        # calculate subgrid extent
        rx = np.array([parameters["ca_centerx"] - R0, parameters["ca_centerx"] + R0], dtype=np.int64)
        ry = np.array([parameters["ca_centery"] - R0, parameters["ca_centery"] + R0], dtype=np.int64)
        rz = np.array([parameters["ca_centerz"] - R0, parameters["ca_centerz"] + R0], dtype=np.int64)

        # select subgrid
        c = c_full[rx[0]:rx[1], ry[0]:ry[1], rz[0]:rz[1]]
        Lc = Lc_full[rx[0] + 1:rx[1] - 1, ry[0] + 1:ry[1] - 1, rz[0] + 1:rz[1] - 1]
        grid = grid_full[rx[0]:rx[1], ry[0]:ry[1], rz[0]:rz[1]]

        # transform positions to inside subgrid (undone later)
        cells_pos[:, 0] = cells_pos[:, 0] - rx[0]
        cells_pos[:, 1] = cells_pos[:, 1] - ry[0]
        cells_pos[:, 2] = cells_pos[:, 2] - rz[0]
    else:
        # operate on full grid
        c = c_full
        Lc = Lc_full
        grid = grid_full

    dx = parameters["ca_dx"] * 1e6
    cdiff = parameters["ca_cdiff"]

    # propagate concentrations until steady state
    max_diff = 100.0  # maximum difference, %
    counter = 0  # count iterations

    # von Neumann stability condition
    factor = 0.1  # factor < 1/2

    # with objmode(start_time="float64"):
    #    start_time = time.time()

    while max_diff > (factor * cdiff * 10):
        counter += 1
        # c = D Laplace c, constant concentration at boundary
        # stability of discretization: ftp://grey.colorado.edu/pub/oreilly/misc/disc_lapl.3.pdf

        # 7-point stencil: dtmax = 1.0 / 6 * dx**2 / Dcoeff
        factor = 0.1
        Lc = (c[2:, 1:-1, 1:-1] + c[:-2, 1:-1, 1:-1] +
              c[1:-1, 2:, 1:-1] + c[1:-1, :-2, 1:-1] +
              c[1:-1, 1:-1, 2:] + c[1:-1, 1:-1, :-2] - 6 * c[1:-1, 1:-1, 1:-1])

        # # 19-point stencil: dtmax = 3./8*dx**2/Dcoeff
        # factor = 3./8
        # Lc = 2./6*(c[2:,1:-1,1:-1]+c[:-2,1:-1,1:-1]+
        #            c[1:-1,2:,1:-1]+c[1:-1,:-2,1:-1]+
        #            c[1:-1,1:-1,2:]+c[1:-1,1:-1,:-2]-6*c[1:-1,1:-1,1:-1])
        # Lc = Lc + 1./6*(c[2:,2:,1:-1]+c[2:,:-2,1:-1]+c[:-2,2:,1:-1]+c[:-2,:-2,1:-1]+
        #                 c[1:-1,2:,2:]+c[1:-1,2:,:-2]+c[1:-1,:-2,2:]+c[1:-1,:-2,:-2]+
        #                 c[2:,1:-1,2:]+c[2:,1:-1,:-2]+c[:-2,1:-1,2:]+c[:-2,1:-1,:-2]-12*c[1:-1,1:-1,1:-1])
        # # 27-point stencil: dtmax = 0.5*dx**2/Dcoeff
        # factor = 0.5
        # Lc = 3./13*(c[2:,1:-1,1:-1]+c[:-2,1:-1,1:-1]+
        #             c[1:-1,2:,1:-1]+c[1:-1,:-2,1:-1]+
        #             c[1:-1,1:-1,2:]+c[1:-1,1:-1,:-2]-6*c[1:-1,1:-1,1:-1])
        # Lc = Lc + 3./26*(c[2:,2:,1:-1]+c[2:,:-2,1:-1]+c[:-2,2:,1:-1]+c[:-2,:-2,1:-1]+
        #                  c[1:-1,2:,2:]+c[1:-1,2:,:-2]+c[1:-1,:-2,2:]+c[1:-1,:-2,:-2]+
        #                  c[2:,1:-1,2:]+c[2:,1:-1,:-2]+c[:-2,1:-1,2:]+c[:-2,1:-1,:-2]-12*c[1:-1,1:-1,1:-1])
        # Lc = Lc + 1./13*(c[2:,2:,2:]+c[2:,2:,:-2]+c[2:,:-2,2:]+c[:-2,2:,2:]+
        #                  c[:-2,:-2,2:]+c[:-2,2:,:-2]+c[2:,:-2,:-2]+c[:-2,:-2,:-2]-8*c[1:-1,1:-1,1:-1])

        dt = factor * dx**2 / Dcoeff
        Lc = Lc * Dcoeff * dt / dx**2

        # consumption
        for ind in cell_indices:
            x, y, z = cells_pos[ind]
            Lc[x - 1, y - 1, z - 1] = Lc[x - 1, y - 1, z - 1] - consumption_rate * grid[x, y, z, GRID_ATTR_FRAC_CONSUME] * dt

            # avoid consumption to negative concentrations
            if Lc[x - 1, y - 1, z - 1, 0] + c[x, y, z, 0] < 0:
                Lc[x - 1, y - 1, z - 1, 0] = -c[x, y, z, 0]
        # compute maximum change in concentration
        max_diff = 100.0 * np.amax(np.abs(Lc) / parameters["p0"])
        c[1:-1, 1:-1, 1:-1] = c[1:-1, 1:-1, 1:-1] + Lc

    # with objmode():
    #    end_time = time.time()
    #    print("Time for iterations:", end_time - start_time)
    print("Iterations until Concentration Equilibrium:", counter)

    if parameters["ca_cperformance"] > 0.5:
        # transform positions back
        cells_pos[:, 0] = cells_pos[:, 0] + rx[0]
        cells_pos[:, 1] = cells_pos[:, 1] + ry[0]
        cells_pos[:, 2] = cells_pos[:, 2] + rz[0]


@jit(nopython=True)
def update_spheriod_center(ncells, cells_pos, cell_indices, parameters):
    cx, cy, cz = 0.0, 0.0, 0.0
    for i in range(ncells):
        x, y, z = cells_pos[cell_indices[i]]
        cx = cx + x
        cy = cy + y
        cz = cz + z
    parameters["ca_centerx"] = cx / ncells
    parameters["ca_centery"] = cy / ncells
    parameters["ca_centerz"] = cz / ncells

    # update maximal radius
    for i in range(ncells):
        x, y, z = cells_pos[cell_indices[i]]
        dist = np.sqrt((x - parameters["ca_centerx"])**2 +
                       (y - parameters["ca_centery"])**2 +
                       (z - parameters["ca_centerz"])**2)
        if dist > parameters["ca_Rmax"]:
            parameters["ca_Rmax"] = dist


@jit(nopython=True)
def tumor_simulation_concentration(c, parameters, ncells, Lc, grid, cells_pos, cell_indices, consumption_rate, Dcoeff, R):
    # if using analytical performance mode, only run analytical solution
    if parameters["ca_cperformance"] == CPERFORMANCE_ANALYTICAL:
        with objmode():
                analytical_stationary_solution(c, ncells, parameters, only_boundary=False, consuming_fraction=1.0,R=R)
    else:
        c_stationary(c, Lc, grid, ncells, cells_pos, cell_indices, consumption_rate, Dcoeff, parameters, R)


# Tumor Simulation
# grid ... 3D Grid
# cells_pos ... cell index -> cell position in grid
# cells ... cell index -> cell attributes
# cell_indices ... used indices
# free_indices ... unused indices
# c ... oxygen concentration
# neighborhood ... which neighboring cells to use in calculation
# event_list ... which events can possibly occur (proliferation, shuffle, apoptosis, etc)
# parameters ... model parameters
# returns t, ncells, ncell_list
@jit(nopython=True)
def tumor_simulation(grid, cells_pos, cells, cell_indices, free_indices, c, parameters, therapy_schedule, neighborh, filename, logPath):
    ncells = len(cell_indices)
    number_of_events = 1
    ncell_list = []  # keep track of number of cells over time
    necrotic_list = []
    hypoxic_list = []
    conc_list = []
    t_list = []

    # diffusion coefficients and consumption rates
    Dcoeff = np.zeros((1, 1, 1, c.shape[-1]), dtype=np.float64)  # shape = (1,1,1,1)
    consumption_rate = np.zeros((1, 1, 1, c.shape[-1]), dtype=np.float64)  # shape = (1,1,1,1)

    dummy_str = ['','0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # why?
    for i in range(c.shape[-1]):
        Dcoeff[0, 0, 0, i] = parameters["D_p" + dummy_str[i]] * (1e6)**2 / parameters["m"]**2
        consumption_rate[0, 0, 0, i] = parameters["ca_a" + dummy_str[i]]

    Lc = np.zeros(c[1:-1, 1:-1, 1:-1, :].shape)  # allocate space for Laplacian

    dt = parameters["ca_dt"]
    tmax = parameters["tmax"]
    t = 0
    therapy_counter = 0

    update_spheriod_center(ncells, cells_pos, cell_indices, parameters)

    # update concentration
    conc, conc_r0 = get_radial_cell_concentration(cells, parameters, grid)
    tumor_simulation_concentration(c, parameters, ncells, Lc, grid, cells_pos, cell_indices, consumption_rate, Dcoeff, R=conc_r0)

    # Do this AFTER oxygen calculation!!
    grid, cells = update_hypoxity(grid, cells, cell_indices, cells_pos, c, parameters)
    n_hypoxic = count_hypoxic_cells(cell_indices, cells_pos, c, parameters)
    n_anoxic = count_necrotic_cells(cells, cell_indices)
    hypoxic_list.append(n_hypoxic)
    necrotic_list.append(n_anoxic)
    conc_list.append(conc)
    t_list.append(t)
    ncell_list.append(ncells)
    radial_mean_jump = np.zeros((int(parameters["ode_gridPoints"]),2))

    with objmode():
        print2("Sum Anoxic/Hypoxic/prolif: " + str(n_anoxic) + "/" + str(n_hypoxic) + "/" + str(ncells - n_hypoxic - n_anoxic) + "\n",filename,logPath=logPath)
    while t < tmax and parameters["ca_Rmax"] < parameters["ca_stop_radius"]:
        t += dt
        with objmode():
            print2("Time (days): " + str(t / 3600.0 / 24),filename,logPath=logPath)
            print2("NumCells: " +str(ncells),filename,logPath=logPath)

        #################################################### MCS step

        MCS_step = ncells*number_of_events
        event_counter = 0
        nec_death_cnt = 0
        pro_new_cnt = 0
        # time_sum = 0
        #print(str(len(cell_indices) + len(free_indices)) + " <-> " + str(len(grid)) + " <-> " + str(len(cells_pos)))
        #print("1")
        if therapy_counter < len(therapy_schedule) and therapy_schedule[therapy_counter][0] < t:
            with objmode():
                print2("    --> TREATMENT",filename,logPath=logPath)

            BruZieRivOelHaa2019.therapy(therapy_schedule[therapy_counter], grid, cells_pos, cells, cell_indices,c, parameters, t, filename, logPath)
            therapy_counter += 1
        #print("2")
        while event_counter < MCS_step:

            event_counter += 1
            chosen_cell = npr.randint(ncells)
            chosen_event = npr.randint(number_of_events)
            #print("3")
            # alternating neighborhood
            neigh_ind = npr.randint(2)
            if neighborh == "neum1" or (neighborh == "neum1moor1" and neigh_ind == 0):
                neighborhood = NEIGHBORHOOD_NEUMANN1

            elif neighborh == "neum2" or (neighborh == "neum2moor2" and neigh_ind == 0):
                neighborhood = NEIGHBORHOOD_NEUMANN2

            elif neighborh == "moor1" or (neighborh == "neum1moor1" and neigh_ind == 1):
                neighborhood = NEIGHBORHOOD_MOORE1

            elif neighborh == "moor2" or (neighborh == "neum2moor2" and neigh_ind == 1) or (neighborh == "neum3moor2" and neigh_ind == 0):
                neighborhood = NEIGHBORHOOD_MOORE2

            elif neighborh == "neum3" or (neighborh == "neum3moor2" and neigh_ind == 1) or (neighborh == "neum3moor3" and neigh_ind == 0):
                neighborhood = NEIGHBORHOOD_NEUMANN3

            elif neighborh == "moor3" or (neighborh == "neum3moor3" and neigh_ind == 1) or (neighborh == "moor3moor4" and neigh_ind == 0):
                neighborhood = NEIGHBORHOOD_MOORE3

            elif neighborh == "moor4" or (neighborh == "moor3moor4" and neigh_ind == 1):
                neighborhood = NEIGHBORHOOD_MOORE4

            else:
                with objmode():
                    print2("[!] neighborhood '" + neighborh + "' unknown!",filename,logPath=logPath)

            #      np.random.choice(range(len(a)),3,replace=False)
            #test = np.random.choice(np.arange(0,len(neighborhood),1),2,replace=False)
            #neighborhood = NEIGHBORHOOD_MOORE1[test]
            #print("4")
            if chosen_event == 0:
                # prolif, cell death, nec clear
                ncells, diff = BruZieRivOelHaa2019.Execute(chosen_cell, grid, cells_pos, cells,
                                                     cell_indices, free_indices, ncells, c,
                                                     neighborhood, parameters, t, radial_mean_jump)
                #print("5")
                if diff == -1:
                    nec_death_cnt += 1
                elif diff == 1:
                    pro_new_cnt += 1
                diff = 0

            elif chosen_event == 1:
                #elif chosen_event in [1,2,3]:
                # Migration
                if npr.uniform(0, 1) <= parameters["ca_llambda"] * dt:
                    #axis = np.zeros(3)
                    #axis[npr.randint(3)] = 1
                    #axis[chosen_event-1] = 1
                    #move_cells_toward_center_one_axis(chosen_cell, grid, cells_pos, cells, parameters,axis)
                    move_cells_toward_center([chosen_cell], grid, cells_pos, cells, parameters, neighborhood)
            #print("16")
        #with objmode():
        #    print("Celldeath/NewCells: " + str(nec_death_cnt) + "/" + str(pro_new_cnt))

        inward_shuffle(grid, cells, cells_pos, cell_indices, parameters, neighborhood)

        #with objmode():
        #    print("inwards_shuffle")

        update_spheriod_center(ncells, cells_pos, cell_indices, parameters)

        #with objmode():
        #    print("cencter update")

        conc, conc_r0 = get_radial_cell_concentration(cells, parameters, grid)
        tumor_simulation_concentration(c, parameters, ncells, Lc, grid, cells_pos, cell_indices, consumption_rate, Dcoeff, R=conc_r0)

        #with objmode():
        #    print("tumor conc")

        # Do this AFTER oxygen calculation!!
        grid, cells = update_hypoxity(grid, cells, cell_indices, cells_pos, c, parameters)
        n_hypoxic = count_hypoxic_cells(cell_indices, cells_pos, c, parameters)
        n_anoxic = count_necrotic_cells(cells, cell_indices)
        hypoxic_list.append(n_hypoxic)
        necrotic_list.append(n_anoxic)
        conc_list.append(conc)
        t_list.append(t)
        ncell_list.append(ncells)

        with objmode():
            print2("Sum Anoxic/Hypoxic/prolif: " + str(n_anoxic) + "/" + str(n_hypoxic) + "/" + str(ncells - n_hypoxic - n_anoxic) + "\n",filename,logPath=logPath)

        if ncells == 0:
            with objmode():
                print2('Extinction',filename,logPath=logPath)
            break

    return t_list, ncells, ncell_list, necrotic_list, hypoxic_list, conc_list, radial_mean_jump


@jit(nopython=True)
def get_radial_cell_concentration(cells, parameters, grid):
    m = parameters["m"]
    rmax = parameters["ode_rmax"]
    dr = parameters["ode_dr"]
    gridPoints = int(parameters["ode_gridPoints"])

    N_grid = parameters["ca_N_grid"]

    conc = np.zeros(gridPoints * (CELL_STATE__LEN - 2))
    c_free = np.zeros(gridPoints)
    c_sum = np.zeros(gridPoints)

    for x in range(N_grid):
        for y in range(N_grid):
            for z in range(N_grid):
                dist = np.sqrt(((x - parameters["ca_centerx"]) ** 2 + (y - parameters["ca_centery"]) ** 2 + (z - parameters["ca_centerz"]) ** 2)) * parameters["ca_dx"] * m
                if rmax >= dist:
                    r_idx = np.floor(dist/dr)
                    if grid[x,y,z,2] != 0:
                        type = cells[int(grid[x,y,z,2])][CELL_ATTR_STATE]
                        if type == CELL_STATE_ANOXIC:
                            type = 1
                        elif type == CELL_STATE_MITCATA:
                            type = 2
                        else:
                            type = 0
                        conc[int(type * gridPoints + r_idx)] += 1
                    else:
                        c_free[int(r_idx)] += 1
                    c_sum[int(r_idx)] += 1

    #print(np.sum(conc[:gridPoints] + conc[-gridPoints:] + c_free - c_sum))
    #print(np.sum(c_sum==0))

    #c_sum[c_sum==0] = 1 # dirty bug fix when 2dr < dx -> c_sum[1] = 0 -> conc/c_sum = nan
    #conc[:gridPoints] = conc[:gridPoints] / c_sum
    #conc[-gridPoints:] = conc[-gridPoints:] / c_sum
    for i in range(3):
        conc[i*gridPoints:(1+i)*gridPoints] = conc[i*gridPoints:(1+i)*gridPoints] / c_sum

    #fix nan values in conc cause when 2dr < dx -> c_sum[1] = 0 -> conc/c_sum = nan
    out = conc.copy()
    for idx in range(1,out.shape[0]):
        if np.isnan(out[idx]):
            out[idx] = out[idx - 1]
    conc = out

    c0 = np.split(conc, 3)[0]
    og_dr = parameters["ode_og_dr"]
    n = len(c0)
    rmaxi = dr * n
    m = int(round(rmaxi/og_dr,2))
    radius = np.arange(0.5, n+0.5, 1) * dr
    og_radius = np.arange(0.5, m+0.5, 1) * og_dr
    og_c0 = np.interp(og_radius, radius, c0)

    n = len(og_c0)-1
    while og_c0[n] <= parameters["ode_cthreshold"] and n > 1:
        n -= 1

    return conc, n * og_dr * 1e1


@jit(nopython=True)
def update_hypoxity(grid, cells, cell_indices, cells_pos, concentrations, parameters):
    for idx in cell_indices:
        x, y, z = cells_pos[idx]
        if concentrations[x, y, z] <= parameters["ph"] and cells[idx][CELL_ATTR_STATE] == CELL_STATE_PROLIFERATING:
            cells[idx][CELL_ATTR_STATE] = CELL_STATE_HYPOXIC
        elif concentrations[x, y, z] > parameters["ph"] and cells[idx][CELL_ATTR_STATE] == CELL_STATE_HYPOXIC:
            cells[idx][CELL_ATTR_STATE] = CELL_STATE_PROLIFERATING

    return grid, cells


@jit(nopython=True)
def cells_to_radius(ncells, parameters):
    return (3.0 / 4 / np.pi * ncells)**(1.0 / 3.0) * parameters["dx"]


@jit(nopython=True)
def count_necrotic_cells(cells, cell_indices):
    ncells_necrotic = 0
    for idx in cell_indices:
        if cells[idx][CELL_ATTR_STATE] == CELL_STATE_ANOXIC:
            ncells_necrotic += 1
    return ncells_necrotic


@jit(nopython=True)
def count_hypoxic_cells(cell_indices, cells_pos, concentrations, parameters):
    ncells_hypoxic = 0
    for idx in cell_indices:

        x, y, z = cells_pos[idx]
        #x,y,z = 1,1,1
        if concentrations[x, y, z] <= parameters["ph"]:
            ncells_hypoxic += 1
    return ncells_hypoxic


# Main Function: init CA, run CA, plot results, open mayavi
def main(parameters, plot_para, string_para):

    N_grid, cell_indices, cells, cells_pos, center, concentrations, free_indices, grid, parameters, therapy_schedule, neighborhood, filename, logPath = init_ca(parameters, string_para)

    print2(f"Date: " + str(string_para["date"]),filename,logPath=logPath)
    print2(f"Stopping at Radius: " + str(parameters["ca_stop_radius"]),filename,logPath=logPath)
    
    t_list, ncell_list, necrotic_list, hypoxic_list, conc_list, comp_time, radial_mean_jump = run_ca(cell_indices, cells, cells_pos, concentrations, free_indices, grid, parameters, therapy_schedule, neighborhood, filename, logPath)

    ncell_list = np.array(ncell_list)
    
    if parameters["saveCa"]:
        save_results(filename, ncell_list, necrotic_list, hypoxic_list, parameters, N_grid, cell_indices, cells, cells_pos, concentrations, free_indices, filename, therapy_schedule)

    if parameters["saveFig"] or parameters["plotFig"]:
        print2("comp_time: " + str(comp_time),filename,logPath=logPath)

        if parameters["run_1d"]:
            v0 = ncell_list[0] * parameters["ca_dx"]**3 * parameters["m"]**3
            metrics, t, sol, jump_prop_1d = simulate_1D_model(parameters, therapy_schedule, filename, logPath, v0 = v0)
            plot_comparison_ca_1d(conc_list, sol, parameters, ncell_list, necrotic_list, metrics, t, filename, plot_para)
        else:
            metrics = None
            sol = None
            t = None
            jump_prop_1d = None

        #plot_volume_growth(parameters, conc_list)
        
        save_movie(conc_list, t_list, parameters, metrics, sol,filename)
        #show_results_in_mayavi(N_grid, cells, cells_pos, concentrations, parameters)
        #plot_3d(N_grid, cells, cells_pos, parameters)
        plot_results(N_grid, cell_indices, cells, cells_pos, concentrations, ncell_list, necrotic_list, hypoxic_list, parameters, None, conc_list, sol, metrics, radial_mean_jump, jump_prop_1d, t, filename, logPath, grid)
        #save_state(cells_pos, cells, parameters, concentrations, N_grid,filename)
    
def save_state(cells_pos, cells, parameters, concentrations, N_grid,filename):

    buff = (cells_pos[:, 0] > 1) * (cells_pos[:, 1] > 1) * (cells_pos[:, 2] > 1)
    cells_pos = cells_pos[buff]

    center = np.array([int(parameters["ca_centerx"]), int(parameters["ca_centery"]), int(parameters["ca_centerz"])])

    #plot_ind_cells = np.where((cells_pos[:, 0] < center[0]) + (cells_pos[:, 1] > center[1]) + (cells_pos[:, 2] < center[2]))[0] # only one quatrant missing
    plot_ind_cells = np.where((cells_pos[:, 0] <= center[0]) + (cells_pos[:, 1] <= center[1]) + (cells_pos[:, 2] <= center[2]))[0]

    state_file = open("./binarys/" + filename + "_state.bin", "wb")
    for indx in plot_ind_cells:
        state_file.write(cells_pos[indx, 0])  # x position der Zelle (int64)
        state_file.write(cells_pos[indx, 1])  # y position der Zelle (int64)
        state_file.write(cells_pos[indx, 2])  # z position der zelle (int64)
        state_file.write(cells[indx, 1])  # Status der zelle (lebend, tot, usw) (double / float64)
    state_file.close()



    ind = np.indices(concentrations[:, :, :, 0].shape)
    ind = np.moveaxis(ind, 0, -1)
    ind = ind.reshape((N_grid**3, 3))
    plot_ind_concentration = np.where((ind[:, 0] == center[0]) + (ind[:, 1] == center[1]) + (ind[:, 2] == center[2]))[0]

    con = concentrations[:, :, :, 0].flatten()
    oxygen_file = open("./binarys/" + filename + "_oxygen.bin", "wb")
    for indx in plot_ind_concentration:
        oxygen_file.write(ind[indx, 0])  # x position im Grid (int64)
        oxygen_file.write(ind[indx, 1])  # y position im Grid (int64)
        oxygen_file.write(ind[indx, 2])  # z position im Grid (int64)
        oxygen_file.write(con[indx])  # Sauerstoffkonzentration (double / float64)
    oxygen_file.close()


def save_movie(conc_list, t_list, parameters, metrics1, sol, filename):
    if parameters["saveFig"]:
        t = np.linspace(0, parameters["tmax"], int(parameters["ode_max_iter_time"]))

        def animate(k, conc_list, t_list, parameters, metrics1, t):
            plt.clf()
            res = np.split(conc_list[k],3)
            t = list(t)
            metric_ind = t.index(min(t, key=lambda x:abs(x-t_list[k])))

            dr = parameters["ode_dr"]
            radius = np.arange(0,len(res[0]),1) * dr * 1e6
            #radius = np.arange(0,len(res[0]),1)

            plt.hlines(0.5,0, radius[-1], color="gray", linestyle="dashed")
            plt.hlines(1,0, radius[-1], color="gray", linestyle="dashed")
            plt.plot(radius, res[0], color=COLOR_PROLIFERATION, label="c0")
            plt.plot(radius, res[1], color=COLOR_ANOXIC, label="c1")
            plt.plot(radius, res[2], color=COLOR_MITOTIC_CATASTROPHY, label="c2")
            plt.plot(radius, res[0]+res[1]+res[1], color=COLOR_SUM, linestyle="dashed", label="sum")
            plt.ylim(0,1.1)
            plt.title("t = " + str(round(t_list[k] / 3600.0 / 24,3)) + "d")
            plt.xlabel("radius")
            plt.ylabel("concentration")
            plt.legend()

        anim = animation.FuncAnimation(plt.figure(), animate, fargs=[conc_list, t_list, parameters, metrics1, t], interval=1, frames=len(conc_list), repeat=True)
        FFwriter = animation.FFMpegWriter(fps=10)
        var = anim.save(getPath()["ca3d"] + "log/" + filename + ".mp4", writer = FFwriter)
        plt.close()

def save_results(name, ncell_list, necrotic_list, hypoxic_list, parameters, N_grid, cell_indices, cells, cells_pos, concentrations, free_indices, filename, therapy_schedule):
    tsim = parameters["ca_dt"] * np.arange(len(ncell_list)) / 3600.0 / 24
    data = np.array([tsim,  ncell_list, necrotic_list, hypoxic_list])
    path = getPath()["ca3d"]+"experiments/"
    #path = "/home/florian/hdd/simus/"
    np.savez(path + filename + ".npz",data=data,
                                             parameters = dict(parameters),
                                             N_grid = N_grid,
                                             cell_indices = cell_indices,
                                             cells = cells,
                                             cells_pos = cells_pos,
                                             concentrations = concentrations,
                                             free_indices = free_indices,
                                             therapy_schedule = therapy_schedule)

def plot_3d(N_grid, cells, cells_pos, parameters):

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import LightSource
    from matplotlib.colors import LinearSegmentedColormap

    buff = (cells_pos[:, 0] > 1) * (cells_pos[:, 1] > 1) * (cells_pos[:, 2] > 1)
    cells_pos = cells_pos[buff]

    center = np.array([int(parameters["ca_centerx"]), int(parameters["ca_centery"]), int(parameters["ca_centerz"])])

    #plot_ind_cells_dark = np.where((cells_pos[:, 1] == center[1]) + (cells_pos[:, 2] == center[2]))[0]  # 3 slices
    plot_ind_cells = np.where((cells_pos[:, 0] < center[0]) + (cells_pos[:, 1] > center[1]) + (cells_pos[:, 2] < center[2]))[0] # only one quatrant missing
    maxi = np.max(cells_pos[plot_ind_cells],axis=0)
    mini = np.min(cells_pos[plot_ind_cells],axis=0)

    c = np.ones((len(cells[plot_ind_cells, 1]),3))
    c[cells[plot_ind_cells, 1]==CELL_STATE_PROLIFERATING] = COLOR_PROLIFERATION
    c[cells[plot_ind_cells, 1]==CELL_STATE_HYPOXIC] = COLOR_HYPOXIC
    c[cells[plot_ind_cells, 1]==CELL_STATE_ANOXIC] = COLOR_ANOXIC

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    cmap = LinearSegmentedColormap.from_list("idk", [COLOR_PROLIFERATION,COLOR_HYPOXIC,COLOR_ANOXIC], N=3)
    #ls = LightSource(azdeg=-30, altdeg=-30)
    #shade = ls.shade(cells_pos[plot_ind_cells, 2],cmap=cmap)

    #zorder=-32
    #ax.scatter(xs=cells_pos[plot_ind_cells, 0], ys=cells_pos[plot_ind_cells, 1], zs=cells_pos[plot_ind_cells, 2],c=c,s=1,alpha=1)#,facecolors=shade)

    surf = ax.plot_surface(X, Y, Z, cmap=cmap,linewidth=0, antialiased=False)

    #ax.scatter(xs=cells_pos[plot_ind_cells_dark, 0], ys=cells_pos[plot_ind_cells_dark, 1], zs=cells_pos[plot_ind_cells_dark, 2],c="black")
    #ax.plot([center[0], maxi[0]],[center[1]-5,center[1]-5],[center[2]-5,center[2]-5],color="black",linewidth=10)
    #ax.plot([center[0]-5, center[0]-5],[center[1],maxi[1]],[center[2]-5,center[2]-5],color="black",linewidth=10)
    ax.plot([mini[0], maxi[0]],[center[1],center[1]],[center[2],center[2]],color="black",linewidth=10)
    ax.plot([center[0], center[0]],[mini[1],maxi[1]],[center[2],center[2]],color="black",linewidth=10)
    ax.plot([center[0], center[0]],[center[1],center[1]],[mini[2],maxi[2]],color="black",linewidth=10)

    if parameters["plotFig"]:
        plt.show()

# Show 3D plots in Mayavi
def show_results_in_mayavi(N_grid, cells, cells_pos, concentrations, parameters):
    from mayavi import mlab

    buff = (cells_pos[:, 0] > 1) * (cells_pos[:, 1] > 1) * (cells_pos[:, 2] > 1)
    cells_pos = cells_pos[buff]

    center = np.array([int(parameters["ca_centerx"]), int(parameters["ca_centery"]), int(parameters["ca_centerz"])])

    # # 3D plotting - requires mayavi

    #plot_ind_cells = np.where((cells_pos[:, 0] == center[0]) + (cells_pos[:, 1] == center[1]) + (cells_pos[:, 2] == center[2]))[0]  # 3 slices
    #plot_ind_cells = np.where((cells_pos[:, 0] > center[0]) + (cells_pos[:, 1] > center[1]) + (cells_pos[:, 2] > center[2]))[0] # only one quatrant missing
    plot_ind_cells = np.where((cells_pos[:, 0] < center[0]) + (cells_pos[:, 1] > center[1]) + (cells_pos[:, 2] < center[2]))[0]

    #cmap = [COLOR_PROLIFERATION, COLOR_HYPOXIC, COLOR_ANOXIC]

    obj = mlab.points3d(cells_pos[plot_ind_cells, 0], cells_pos[plot_ind_cells, 1], cells_pos[plot_ind_cells, 2], cells[plot_ind_cells, 1], mode='point')
    obj.actor.property.point_size = 10  # QUICKFIX
    obj.actor.property.opacity = 1.
    #mlab.colorbar(object=obj, title="Celltypes")
    """
    ind = np.indices(concentrations[:, :, :, 0].shape)
    ind = np.moveaxis(ind, 0, -1)
    ind = ind.reshape((N_grid**3, 3))
    plot_ind_concentration = np.where((ind[:, 0] == center[0]) + (ind[:, 1] == center[1]) + (ind[:, 2] == center[2]))[0]  # 3 slices
    obj = mlab.points3d(ind[plot_ind_concentration, 0], ind[plot_ind_concentration, 1], ind[plot_ind_concentration, 2], concentrations[:, :, :, 0].flatten()[plot_ind_concentration], mode='point')
    obj.actor.property.point_size = 10  # QUICKFIX
    obj.actor.property.opacity = 1.0
    """
    mlab.show()

def simulate_1D_model(parameters, therapy_schedule, filename, logPath, c0 = None, v0 = None):

    print2("[*] Simulating 1D shell model",filename,logPath=logPath)

    gridPoints = int(parameters["ode_gridPoints"])
    rmax = parameters["ode_rmax"]
    max_iter_time = int(parameters["ode_max_iter_time"])
    parameters["rd"] = parameters["ode_rd"]
    parameters["dose"] = parameters["ode_dose"]
    lmfit_parameters = dictToLmfitParameters(parameters)

    v0 = parameters["v0"] if v0 == None else v0
    #v0 = parameters["v0_start"]
    dr = parameters["ode_dr"]
    fak = parameters["ode_fak"]
    #fak = 1
    m = parameters["m"]
    tmax = parameters["tmax"]

    lmfit_parameters["gamma"].value /= fak
    #lmfit_parameters["llambda"].value /= fak
    #lmfit_parameters["a"].value *= fak

    MyClass1 = getattr(models,"slice_model_RT")
    #MyClass1 = getattr(models,"slice_model_FG_prolifRange")
    #lmfit_parameters.add("prolifRange",value=2,vary=False)
    instance1 = MyClass1()
    """
    lmfit_parameters.add("tmax",value=1814400.0)
    lmfit_parameters.add("rmax",value=110.0)
    lmfit_parameters["alphaR"].value = 0
    lmfit_parameters["betaR"].value = 0
    lmfit_parameters["gammaR"].value = 0
    lmfit_parameters["p_mit_cata"].value = 0
    lmfit_parameters["dose"].value = 0
    """
    if c0 is None:
        t = np.linspace(0, tmax, max_iter_time)
        c = setVolume_3(lmfit_parameters, t, "slice_model_RT")
    else:
        c = c0

    t0 = 0
    metrics_save = []
    sol_save = []
    t_save = []

    if therapy_schedule[-1][0] > tmax:
        therapy_schedule[-1][0] = tmax
    if therapy_schedule[-1][0] != tmax:
        therapy_schedule = np.append(therapy_schedule,[[tmax,0.,0.]],axis=0)

    for i in range(len(therapy_schedule)):
        t = np.linspace(0, therapy_schedule[i][0]-t0, max_iter_time)

        t_therapy = np.array([therapy_schedule[i][0]-t0]) # d
        d_therapy = np.array([0]) # Gy
        arg = instance1.getValue(lmfit_parameters, t, specialParameters={"c":c,"t_therapy":t_therapy, "d_therapy":d_therapy})
        sol = arg[0]
        metrics = instance1.getMetrics(lmfit_parameters, sol)

        metrics_save.append(metrics)
        sol_save.append(sol)
        t_save.append(t)
        c = sol[-1]

        t0 = therapy_schedule[i][0]
        lmfit_parameters["p_mit_cata"].value = therapy_schedule[i][2]
        #lmfit_parameters["dose"].value = therapy_schedule[i][1]
        print("FixMe RT model does the RT itself")
        c = therapy_RT(lmfit_parameters,c,therapy_schedule[i][1])

    for i in range(len(t_save)):
        if i == 0:
            t = t_save[i]
            sol = sol_save[i]
            metrics1 = metrics_save[i]
        else:
            t = np.append(t,t_save[i]+t[-1])
            sol = np.append(sol,sol_save[i],axis=0)
            for j in range(6):
                metrics1[j] = np.append(metrics1[j], metrics_save[i][j])
    #t = np.linspace(0, tmax, max_iter_time)

    #sol, dist, num = instance1.getValue(lmfit_parameters, t, specialParameters={"c":c})
    #sol = instance1.getValue(lmfit_parameters, t, specialParameters={"c":c,"t_therapy":t_therapy, "d_therapy":d_therapy})[0]
    dist = num = None
    #np.save("neum1",sol)
    """
    sol = instance1.getValue(lmfit_parameters, t, specialParameters={"c":c})
    sol = sol[0]
    dist = num = None
    """
    #metrics1 = instance1.getMetrics(lmfit_parameters, sol)

    p = []
    i = 0

    for c in sol:
        print("Progess after analysis: " + str(i+1) + "/" + str(len(sol)) + "\r",end="")
        i += 1
        c = np.split(c,3)
        c0 = c[0]
        c1 = c[1]
        c2 = c[2]

        p0, rhypox, rcrit = o2_models.sauerstoffdruck_analytical_1(lmfit_parameters,c0+c2,mode=2)
        p.append(p0)
    print("")

    #anim = animation.FuncAnimation(plt.figure(), animate_4, fargs=[sol, t[-1]/(24*60*60), len(t), dr/m, p, None], interval=1, frames=len(t), repeat=True)
    anim = animation.FuncAnimation(plt.figure(), animate_6, fargs=[sol, t/(24*60*60), len(t), dr/m, p, None], interval=1, frames=len(t), repeat=True)
    FFwriter = animation.FFMpegWriter(fps=10)
    var = anim.save("model_1d.mp4", writer = FFwriter)
    plt.close()

    #return metrics1, t, sol, dist / num
    return metrics1, t, sol, None

def get_o2_from_cell_concentrtation(lmfit_parameters, conc_list):

    c = np.split(conc_list[-1],3)
    c0 = c[0] + c[2]
    args_o2 = o2_models.sauerstoffdruck_analytical(params=lmfit_parameters, c0=c0, mode=2)
    retOut = args_o2[0]
    rh = args_o2[1]
    rcrit = args_o2[2]
    retOut1, rh1, rcrit1 = o2_models.sauerstoffdruck_analytical_1(lmfit_parameters, c0, mode=2)
    #print("ana1: " + str(rcrit1))
    #print("ana3: " + str(rcrit))
    return retOut, retOut1

# Show Plots
def plot_results(N_grid, cell_indices, cells, cells_pos, concentrations, ncell_list, necrotic_list, hypoxic_list, parameters, expData, conc_list, sol, metrics, radial_mean_jump, jump_prop_1d, t, filename, logPath, grid):
    """
    ### Plot für die Prolif. Distance
    plt.figure()

    #radial_mean_jump[:,0][radial_mean_jump[:,0]==0] = 1
    y = radial_mean_jump[:,1] / radial_mean_jump[:,0]
    y2 = radial_mean_jump[:,0]
    maxi = np.max(y2)
    y2 /= np.sum(y2)
    y2 /= np.max(y2)
    y2[y2 == 0] = np.nan
    jump_prop_1d[np.isnan(y)] = np.nan

    plt.plot(np.arange(0,int(parameters["ode_gridPoints"])),y,label="mean prolf. dist in times dr (CA)")
    plt.plot(np.arange(0,int(parameters["ode_gridPoints"])),jump_prop_1d,label="mean prolf. dist in times dr (1D)")
    plt.plot(np.arange(0,int(parameters["ode_gridPoints"])),y2,label="prolf. ereignisse times " + str(maxi))
    plt.xlabel('r [*dr]')
    plt.legend()
    plt.title(title)
    #plt.ylabel('mean prolif. dist [*dr]')
    """

    plt.figure(figsize=(20,10))
    plt.suptitle(filename)

    ### Simulation für Volumen Vorgabe

    lmfit_parameters = dictToLmfitParameters(parameters)
    #t = np.linspace(0, parameters["tmax"], int(parameters["ode_max_iter_time"]))

    o2_cell_conc, o2_cell_conc1 = get_o2_from_cell_concentrtation(lmfit_parameters, conc_list)
    dr = parameters["ode_dr"] / parameters["m"]
    radius_o2_cell_conc = np.arange(0,len(o2_cell_conc),1) * dr * 1e6

    print((3.0 / 4 / np.pi * len(cells))**(1.0 / 3) * parameters["ca_dx"] * 1e6)
    c_sum = np.split(conc_list[-1],3)
    c_sum = c_sum[0] + c_sum[1] + c_sum[2]
    v = np.sum(c_sum * ((np.arange(1, len(c_sum) + 1, 1) * dr) ** 3 - (np.arange(0, len(c_sum), 1) * dr) ** 3) * 4. * np.pi / 3.)
    r0 = (3. * v / (4. * np.pi)) ** (1./3.)
    print(r0 * 1e6)

    if not sol is None:
        o2_model, o2_model1 = get_o2_from_cell_concentrtation(lmfit_parameters, sol)
        dr = parameters["ode_dr"] / parameters["m"]
        radius_o2_model = np.arange(0,len(o2_model),1) * dr * 1e6

    n = int(len(conc_list[0])/3.)
    dr = lmfit_parameters["dr"].value * 1e6 / lmfit_parameters["m"].value
    vmax = 4. * np.pi * ((np.arange(1,n,1)*dr)**3 - (np.arange(0,n-1,1)*dr)**3) / 3.
    split = np.array(np.split(np.array(conc_list),3,1))
    r0_cell_conc = np.sum((split[0][:,1:] + split[0][:,:-1] + split[1][:,1:] + split[1][:,:-1] + split[2][:,1:] + split[2][:,:-1]) * vmax / 2.,1)
    r0_cell_conc = ((r0_cell_conc / np.pi)*(3./4.))**(1./3.)

    center = np.array([parameters["ca_centerx"], parameters["ca_centery"], parameters["ca_centerz"]])

    cell_indices = np.array(cell_indices)
    cells_pos = cells_pos[cell_indices]
    cells = cells[cell_indices]

    ncell_list = np.array(ncell_list)
    necrotic_list = np.array(necrotic_list)
    hypoxic_list = np.array(hypoxic_list)

    # ordered grid indices for inward shuffle and c(r)-plots
    # grid of indices to concentration voxels
    concentration_grid = np.indices(concentrations[:, :, :, 0].shape)
    concentration_grid = np.moveaxis(concentration_grid, 0, -1)
    concentration_grid = concentration_grid.reshape((concentrations.shape[0] * concentrations.shape[1] * concentrations.shape[2], 3))
    # distance of each concentration voxel to center
    concentration_dist = np.linalg.norm(concentration_grid - center[np.newaxis, :], axis=1)

    ### Sauerstoffkurve ###
    r0 = (3.0 / 4 / np.pi * len(cells))**(1.0 / 3) * parameters["ca_dx"] * 1e6
    print("R0 (Volume) = " + str(r0))
    # print('R0 (max 100) ='+str(np.mean(np.sort(r0_)[-100:])))
    plt.subplot(221)

    concentration_x = concentration_dist * parameters["ca_dx"] * 1e6
    concentration_y = concentrations[:, :, :, 0].reshape((concentrations.shape[0] * concentrations.shape[1] * concentrations.shape[2]))
    #plt.scatter(concentration_x, concentration_y)
    
    concentration_y = concentration_y[concentration_x<=radius_o2_cell_conc[-1]]
    concentration_y[concentration_y > 100.] = 100.
    concentration_x = concentration_x[concentration_x<=radius_o2_cell_conc[-1]]

    idx = np.sort(random.sample(range(len(concentration_x)), np.min([3000, len(concentration_x)])))
    plt.scatter(concentration_x[idx], concentration_y[idx])
    
    pcrit = parameters["pcrit"]

    if parameters["ca_cmode"] == CMODE_P0_INFTY:
        mode = "c(r->infty) = p0"
    elif parameters["ca_cmode"] == CMODE_P0_R0:
        mode = "c(r = r0) = p0"
    else:
        mode = "c(r = ngrid) = p0"

    ### Suplot 1: Sauerstoffkurve ###
    conc, conc_r0 = get_radial_cell_concentration(cells, parameters, grid)
    rb, cr0 = c_r_solutions(r0, parameters, 200.0 / N_grid * N_grid * 0.5, mode=parameters["ca_cmode"], Nanal=1000, pcrit=0, R=conc_r0)
    cr0[cr0>parameters["p0"]] = parameters["p0"]

    if not sol is None:
        val = np.min(np.abs(rb - radius_o2_model[-1]))
        idx = np.where(np.abs(rb - radius_o2_model[-1]) == val)[0][0]
    else:
        idx = len(rb)

    plt.plot(rb[:idx], cr0[:idx], '-', color=(1, 0, 0), label=f'CA - p')
    plt.plot(radius_o2_cell_conc, o2_cell_conc,"-.",color="black",label="CA - p (from cell conc.)")
    
    if not sol is None:
        plt.plot(radius_o2_model, o2_model,"--",color="orange",label="1D - p")

    #plt.plot(radius_o2_model, o2_model,"--",color="orange",label="1D - p ana3")
    #plt.plot(radius_o2_model, o2_model1,"--",label="1D - p ana1")
    plt.xlabel('r [um]')
    plt.ylabel('pO2 [mmHg]')
    plt.legend()

    ### Wachstumskurve ###
    # simulationsverlauf (blaue linie)
    plt.subplot(222)
    dsim = (3. / 4. / np.pi * ncell_list)**(1. / 3.) * parameters["ca_dx"] * 1e6
    tsim = parameters["ca_dt"] * np.arange(len(ncell_list)) / 3600.0 / 24
    plt.plot(tsim, dsim, color=COLOR_PROLIFERATION, label="CA - r0")

    dr = 0.5 * (np.max(np.linalg.norm(NEIGHBORHOOD_NEUMANN2, axis=1)) + np.max(np.linalg.norm(NEIGHBORHOOD_MOORE2, axis=1)))
    dr = dr * parameters["ca_dx"] * 1e6

    # rote und orange gestrichelte Linie (nekrotisch und hypoxisch)
    necrotic_radius = (3. / 4. / np.pi * necrotic_list)**(1. / 3.) * parameters["ca_dx"] * 1e6
    hypoxic_radius = (3. / 4. / np.pi * (hypoxic_list + necrotic_list))**(1. / 3.) * parameters["ca_dx"] * 1e6

    plt.plot(tsim, necrotic_radius, '-', color=COLOR_ANOXIC, label="CA - rn") # Rot
    #plt.plot(tsim, hypoxic_radius, '-', color=COLOR_HYPOXIC, label="hypoxic radius") # Orange

    if not metrics is None:
        plt.plot(t/24./3600., metrics[3]*1e6, linestyle="dashed", color=COLOR_PROLIFERATION, label='1D - r0', alpha=0.5)
        plt.plot(t/24./3600., metrics[5]*1e6, linestyle="dashed", color=COLOR_ANOXIC, label='1D - rn', alpha=0.5)
    plt.plot(tsim, r0_cell_conc, "-.", color=COLOR_PROLIFERATION,alpha=0.5,label="CA - r0 (No. 2)")

    if not metrics is None:
        y = np.interp(t/24./3600.,tsim,dsim)
        buff = np.abs(t/24./3600. - 4.)
        hit = np.where(np.min(buff) == buff)[0][0]    
        buff = np.abs(y-metrics[3][hit]*1e6)
        hit2 = np.where(np.min(buff) == buff)[0][0]

    #plt.plot(t/24./3600. - (t[hit]/24./3600 - t[hit2]/24./3600.), metrics[3]*1e6, linestyle="dotted", color="blue", label='1D - r0', alpha=0.5)

    plt.xlabel('t [d]')
    plt.ylabel('radius [um]')
    plt.legend()

    if not metrics is None:
        n = int(len(t)/(4.))
        model1 = np.polyfit(t[n:]/24./3600.,y[n:],1)
        model2 = np.polyfit(t[n:]/24./3600.,metrics[3][n:]*1e6,1)
        print("CA: " + str(model1))
        print("1D: " + str(model2))
        print2("Faktor: " + str(model1[0]/model2[0]),filename, logPath=logPath)
    #CA: [  6.03974787 151.1855184 ]
    #1D: [  7.24170534 151.41581366

    """
    ### cells radius-angle scatter plot ###
    plt.subplot(223)
    cells_pos_ = (cells_pos - center[np.newaxis, :]) * parameters["ca_dx"] * 1e6
    r = np.linalg.norm(cells_pos_, axis=1)
    phi = np.arctan2(cells_pos_[:, 1], cells_pos_[:, 0])

    color = ["red", "blue", "green", "black"]
    cellTyp = ["profliferating", "hypoxic", "anoxic", "free"]

    for i in range(CELL_STATE__LEN):
        ind = [j for j in range(len(cells)-1) if cells[j, CELL_ATTR_STATE] == i]
        plt.scatter([r[j] for j in ind], [phi[j] for j in ind], c=color[i], s = 0.5, label=cellTyp[i])

    plt.plot(np.ones(2) * r0, np.array([-np.pi, np.pi]), '--')
    plt.xlabel('r [um]')
    plt.ylabel('phi (theta)')
    plt.legend(loc="upper left")
    """
    """
    ### start cell concentration ###
    plt.subplot(223)

    idx = 0

    res = np.split(conc_list[idx],2)
    c = np.split(sol[idx],2)
    dr = parameters["ode_dr"] / parameters["m"]
    radius = np.arange(0,len(res[0]),1) * dr * 1e6

    plt.hlines(1,0, radius[-1], color="gray", linestyle="dotted")

    plt.plot(radius, res[0], color=COLOR_PROLIFERATION, label="CA - c0")
    plt.plot(radius, res[1], color=COLOR_ANOXIC, label="CA - c1")
    #plt.plot(radius, res[0]+res[1], "cyan", linestyle="dashed", label="sum - CA")

    plt.plot(radius, c[0], color=COLOR_PROLIFERATION,linestyle="dashed",alpha=0.5, label="1D - c0")
    plt.plot(radius, c[1], color=COLOR_ANOXIC,linestyle="dashed",alpha=0.5, label="1D - c1")
    #plt.plot(radius, c[0]+c[1], "cyan", linestyle="dashed", label="sum - 1D")

    plt.xlabel("radius [um]")
    plt.ylabel("concentration")
    plt.legend()
    """
    #"""
    ### radial difference 1d to ca ###
    plt.subplot(223)
    if not metrics is None:
        dsim = (3. / 4. / np.pi * ncell_list)**(1. / 3.) * parameters["ca_dx"] * 1e6
        tsim = parameters["ca_dt"] * np.arange(len(ncell_list)) / 3600.0 / 24
        y = np.interp(t/24./3600., tsim, dsim)

        dr = 0.5 * (np.max(np.linalg.norm(NEIGHBORHOOD_NEUMANN2, axis=1)) + np.max(np.linalg.norm(NEIGHBORHOOD_MOORE2, axis=1)))
        dr = dr * parameters["ca_dx"] * 1e6

        plt.plot(t/24./3600., y - metrics[3]*1e6, linestyle="solid", color="black")

        y = np.interp(t/24./3600., tsim, r0_cell_conc)
        plt.plot(t/24./3600., y - metrics[3]*1e6, linestyle="dashed", color="black")

    plt.xlabel('t [d]')
    plt.ylabel('r0(CA) - r0(1D) radius [um]')
    #"""
    ### Zellanzahlen ###
    """
    plt.subplot(224)
    plt.hist(cells[:, CELL_ATTR_STATE])
    plt.ylabel('N')
    plt.xlabel('type (prolif.,hypoxic,anoxic)')

    plt.suptitle("Simulation: " + parameters["ca_name"],fontsize=20)
    """

    plt.subplot(224)
    idx = -1

    res = np.split(conc_list[idx],3)
    dr = parameters["ode_dr"] / parameters["m"]
    radius = np.arange(0,len(res[0]),1) * dr * 1e6

    plt.hlines(1,0, radius[-1], color="gray", linestyle="dotted")

    plt.plot(radius, res[0], color=COLOR_PROLIFERATION, label="CA - c0")
    plt.plot(radius, res[1], color=COLOR_ANOXIC, label="CA - c1")
    plt.plot(radius, res[2], color=COLOR_MITOTIC_CATASTROPHY, label="CA - c2")
    #plt.plot(radius, res[0]+res[1], "cyan", linestyle="dashed", label="sum - CA")

    if not metrics is None:
        c = np.split(sol[idx],3)

        plt.plot(radius, c[0], color=COLOR_PROLIFERATION,linestyle="dashed",alpha=0.5, label="1D - c0")
        plt.plot(radius, c[1], color=COLOR_ANOXIC,linestyle="dashed",alpha=0.5, label="1D - c1")
        plt.plot(radius, c[2], color=COLOR_MITOTIC_CATASTROPHY,linestyle="dashed",alpha=0.5, label="1D - c2")
        #plt.plot(radius, c[0]+c[1], "cyan", linestyle="dashed", label="sum - 1D")

    plt.xlabel("radius [um]")
    plt.ylabel("concentration")
    plt.legend()

    if parameters["saveFig"]:
        path = getPath()["ca3d"]
        plt.savefig(path + filename + "_4plot.pdf",transparent=True)
    if parameters["plotFig"]:
        plt.show()

# run the cellular automaton
def run_ca(cell_indices, cells, cells_pos, concentrations, free_indices, grid, parameters, therapy_schedule, neighborhood, filename, logPath):
    t0 = time.time()

    # calculate analytical solution -> initialize concentrations
    analytical_stationary_solution(concentrations, len(cell_indices), parameters, only_boundary=False)

    # actually simulate the tumor
    t_list, ncells, ncell_list, necrotic_list, hypoxic_list, conc_list, radial_mean_jump = tumor_simulation(grid, cells_pos, cells, cell_indices, free_indices, concentrations, parameters, therapy_schedule, neighborhood, filename, logPath)

    comp_time = time.time() - t0

    print2(f'Computation Time: {comp_time} seconds',filename,logPath=logPath)
    print2(f'Simulation Time: {t_list[-1] / 24.0 / 3600.0} days',filename,logPath=logPath)
    print2(f'Number of Cells: {ncells} = {len(cell_indices)}',filename,logPath=logPath)

    return t_list, ncell_list, necrotic_list, hypoxic_list, conc_list, comp_time, radial_mean_jump


# initialize the CA
def init_ca(parameters, string_para):
    
    filename = string_para["filename"]
    neighborhood = string_para["ca_neighborhood"]

    logPath = getPath()["ca3d"]+"log/"
    flushLog(filename,logPath=logPath,force=parameters["force_logflush"])
    dictToLog(parameters, filename, logPath=logPath)
    print2("[*] Neighberhood: " + neighborhood, filename, logPath=logPath)
    print2("[*] Starting simulation: " + filename, filename,logPath=logPath)

    N_grid = parameters["ca_N_grid"]
    N_cells = N_grid**3

    # [fraction of occupied space,fraction that consumes,index of occupying cell as float]
    grid = np.zeros((N_grid, N_grid, N_grid, GRID_ATTR__LEN), dtype=float)
    # grid indices of cell
    cells_pos = np.zeros((N_cells, 3), dtype=int)

    # [size, type, time of last division, time of last treatment(BruZieRivOelHaa2019) / compartment(Jung model), arrest time for HT, ...]
    cells = np.zeros((N_cells, CELL_ATTR__LEN), dtype=float)

    # reaction-diffusion concentrations
    concentrations = np.ones((N_grid, N_grid, N_grid, 1), dtype=float)  # [oxygen,...]

    # initialize ~ N0 cells
    Radius0 = (3.0 / 4 / np.pi * parameters["ca_N0"])**(1.0 / 3)
    center = int(N_grid * 0.5) * np.ones(3)  # 3D vector

    # magic
    start_cells_pos = np.indices(grid[:, :, :, 0].shape)  # creates shape (3,100,100,100) # indicies der Reihe, Spalte und Länge
    start_cells_pos = np.moveaxis(start_cells_pos, 0, -1)  # makes shape (100,100,100,3) # e.g. start_cells_pos[0][2][5] = [0 2 5]
    # makes (1000000,3) <=> [[0,0,0], [0,0,1], [0,0,2], ..., [99,99,98], [99,99,99]]
    start_cells_pos = start_cells_pos.reshape((grid[:, :, :, 0].shape[0] * grid[:, :, :, 0].shape[1] * grid[:, :, :, 0].shape[2], 3))
    # keep only indices within Radius0 -> starting cells
    start_cells_pos = start_cells_pos[np.linalg.norm(start_cells_pos - center[np.newaxis, :], axis=-1) < Radius0]
    parameters["ca_start_cells_pos"] = len(start_cells_pos)
    print2(f'Number of starting Cells = {len(start_cells_pos)}',filename,logPath=logPath)

    # initialize cells in grid
    start_cells_indices = np.arange(len(start_cells_pos), dtype=int)  # 0 .. num cells that exist right now

    # init all attributes with 1, except index
    grid[start_cells_pos[:, 0], start_cells_pos[:, 1], start_cells_pos[:, 2], :] = 1
    grid[start_cells_pos[:, 0], start_cells_pos[:, 1], start_cells_pos[:, 2], GRID_ATTR_INDEX] = start_cells_indices

    # lists for efficent handling of cell removal/adding
    cell_indices = list(start_cells_indices)
    free_indices = list(np.arange(N_cells)[::-1][:-len(start_cells_pos)])

    cells_pos[start_cells_indices, :] = start_cells_pos
    cells[start_cells_indices, CELL_ATTR_SIZE] = 1
    cells[start_cells_indices, CELL_ATTR_STATE] = CELL_STATE_PROLIFERATING

    parameters["ca_centerx"], parameters["ca_centery"], parameters["ca_centerz"] = center[0], center[1], center[2]
    parameters["ca_Rmax"] = Radius0
    therapy_schedule = parameters["therapy_schedule"]

    #concentrations = concentrations * parameters["p0"]  # [mmHg]

    # convert to numba types
    tmp_par = typed.Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    for k, v in parameters.items():
        if k != "therapy_schedule":
            tmp_par[k] = v
    parameters = tmp_par

    tmp_par = typed.List()
    for index in cell_indices:
        tmp_par.append(index)
    cell_indices = tmp_par

    tmp_par = typed.List()
    for index in free_indices:
        tmp_par.append(index)
    free_indices = tmp_par

    return N_grid, cell_indices, cells, cells_pos, center, concentrations, free_indices, grid, parameters, therapy_schedule, neighborhood, filename, logPath
