import numpy.random as npr
from numba import jit, typed, types, objmode

from constants import *

import sys
sys.path.insert(0,'../../.')
from tools import *

############################################### BruZieRivOelHaa2019 Modell
@jit(nopython=True)
def Execute(chosen_cell_, grid, cells_pos, cells, cell_indices, free_indices, ncells, c, neighborhood, parameters, t, radial_mean_jump):
    chosen_cell = cell_indices[chosen_cell_]  # get index of chosen cell
    x, y, z = cells_pos[chosen_cell]
    random_variable = npr.uniform(0, 1)

    division_rate = parameters["ca_gamma"]
    dt = parameters["ca_dt"]
    phypoxia = parameters["ph"]
    pcrit = parameters["pcrit"]
    rhypoxiadeath = parameters["ca_epsilon"]
    rclearnecrotic = parameters["ca_delta"]
    diff = 0

    # IF cell state is CELL_STATE_PROLIFERATING && oxygen concentration is below hypoxic concentration && random <= rhypoxiadeath
    # THEN cell becomes necrotic
    #print((cells[chosen_cell, CELL_ATTR_STATE] == CELL_STATE_PROLIFERATING or cells[chosen_cell, CELL_ATTR_STATE] == CELL_STATE_HYPOXIC))
    #print(c[x, y, z, 0] < pcrit)
    #print(random_variable <= rhypoxiadeath * dt)

    if (cells[chosen_cell, CELL_ATTR_STATE] == CELL_STATE_PROLIFERATING or
        cells[chosen_cell, CELL_ATTR_STATE] == CELL_STATE_HYPOXIC or
        cells[chosen_cell, CELL_ATTR_STATE] == CELL_STATE_MITCATA) and c[x, y, z, 0] <= pcrit and random_variable <= rhypoxiadeath * dt:

        # cell death: hypoxia death at division
        cells[chosen_cell, CELL_ATTR_STATE] = CELL_STATE_ANOXIC  # hypoxic cell death
        grid[x, y, z, GRID_ATTR_FRAC_CONSUME] = 0  # necrotic cells do not consume oxygen

    # IF (cell state is CELL_STATE_ANOXIC && random <= clear_probability)
    # THEN cell dies and is removed
    elif cells[chosen_cell, CELL_ATTR_STATE] == CELL_STATE_ANOXIC and random_variable <= rclearnecrotic * dt:
        # necrotic
        ind = cell_indices.pop(chosen_cell_)
        free_indices.append(ind)
        # remove cell volume (Lysis)
        grid[x, y, z, GRID_ATTR_FRAC_SPACE] = grid[x, y, z, GRID_ATTR_FRAC_SPACE] - cells[ind, CELL_ATTR_SIZE]
        grid[x, y, z, GRID_ATTR_FRAC_CONSUME] = 0
        grid[x, y, z, GRID_ATTR_INDEX] = 0
        # remove cell
        cells_pos[ind, :] = 0
        cells[ind, :] = 0
        ncells += -1
        diff = -1

    # IF cell state is CELL_STATE_PROLIFERATING && random <= division_probability
    # THEN cell proliferates
    elif (cells[chosen_cell, CELL_ATTR_STATE] == CELL_STATE_PROLIFERATING
        or cells[chosen_cell, CELL_ATTR_STATE] == CELL_STATE_MITCATA) and random_variable <= division_rate * dt:

        # oxygenated cell division
        # daughter cell same size as mother cell
        size = cells[chosen_cell, CELL_ATTR_SIZE]
        # check for free space in neighborhood
        len_neighborhood = len(neighborhood)
        neigh_ind = np.zeros(len_neighborhood, dtype=np.int64)
        counter = -1
        free_neigh = 0

        prob = np.zeros(len_neighborhood, dtype=np.float64)
        prop_offset = np.zeros(3, dtype=np.float64)

        x_, y_, z_ = cells_pos[chosen_cell]
        dist_ = np.sqrt(((x_ - parameters["ca_centerx"]) ** 2 + (y_ - parameters["ca_centery"]) ** 2 + (z_ - parameters["ca_centerz"]) ** 2)) * parameters["ca_dx"] * parameters["m"]
        r_idx_ = np.floor(dist_/parameters["ode_dr"])

        for neigh in neighborhood:
            x, y, z = cells_pos[chosen_cell] + neigh

            dist = np.sqrt(((x - parameters["ca_centerx"]) ** 2 + (y - parameters["ca_centery"]) ** 2 + (z - parameters["ca_centerz"]) ** 2)) * parameters["ca_dx"] * parameters["m"]
            r_idx = np.floor(dist/parameters["ode_dr"])

            prop_offset[int(r_idx - r_idx_ + 1)] += 1

        #prop_offset[0] = 1
        #prop_offset[1] = 1
        #prop_offset[2] = 1

        for neigh in neighborhood:
            counter += 1
            x, y, z = cells_pos[chosen_cell] + neigh

            dist = np.sqrt(((x - parameters["ca_centerx"]) ** 2 + (y - parameters["ca_centery"]) ** 2 + (z - parameters["ca_centerz"]) ** 2)) * parameters["ca_dx"] * parameters["m"]
            r_idx = np.floor(dist/parameters["ode_dr"])

            if grid[x, y, z, GRID_ATTR_FRAC_SPACE] + size <= 1:
            #if grid[x, y, z, GRID_ATTR_FRAC_SPACE] + size <= 1 and np.abs(r_idx_ - r_idx) <= 1: #range Leq 1  # <- Brauchen wir das noch?
                neigh_ind[free_neigh] = counter

                """
                # dist center
                prob[free_neigh] = np.sqrt((x - parameters["ca_centerx"])**2 +
                                           (y - parameters["ca_centery"])**2 +
                                           (z - parameters["ca_centerz"])**2)
                """

                """
                #dist mother cell
                prob[free_neigh] = np.sqrt((x - x_)**2 +
                                           (y - y_)**2 +
                                           (z - z_)**2)
                #prob[free_neigh] = np.sqrt(neigh[0]**2+neigh[1]**2+neigh[2]**2) # closer to mother cell
                """

                #"""
                #equal Prop
                prob[free_neigh] = 1.
                #"""

                # correction term
                #prob[free_neigh] = 1./prop_offset[int(r_idx - r_idx_ + 1)]

                free_neigh += 1

        # select free position
        neigh_ind = neigh_ind[:free_neigh]
        prob = prob[:free_neigh]

        if free_neigh > 0:
            if cells[chosen_cell, CELL_ATTR_STATE] == CELL_STATE_MITCATA and npr.uniform(0, 1) >= parameters["p_mit_cata"]:

                # necrotic
                ind = cell_indices.pop(chosen_cell_)
                free_indices.append(ind)
                # remove cell volume (Lysis)
                grid[x, y, z, GRID_ATTR_FRAC_SPACE] = grid[x, y, z, GRID_ATTR_FRAC_SPACE] - cells[ind, CELL_ATTR_SIZE]
                grid[x, y, z, GRID_ATTR_FRAC_CONSUME] = 0
                grid[x, y, z, GRID_ATTR_INDEX] = 0
                # remove cell
                cells_pos[ind, :] = 0
                cells[ind, :] = 0
                ncells += -1
                diff = -1

            else:
                prob /= np.sum(prob)
                idx = np.searchsorted(np.cumsum(prob), np.random.random())
                x, y, z = cells_pos[chosen_cell] + neighborhood[neigh_ind[idx]]

                dist = np.sqrt(((x - parameters["ca_centerx"]) ** 2 + (y - parameters["ca_centery"]) ** 2 + (z - parameters["ca_centerz"]) ** 2)) * parameters["ca_dx"] * parameters["m"]
                r_idx = np.floor(dist/parameters["ode_dr"])
                radial_mean_jump[int(r_idx_)][0] += 1
                radial_mean_jump[int(r_idx_)][1] += int(r_idx - r_idx_)

                cells[chosen_cell, CELL_ATTR_T_LAST_DIV] = t

                ind = free_indices.pop()
                # clone to free position
                grid[x, y, z, GRID_ATTR_FRAC_SPACE] = grid[x, y, z, GRID_ATTR_FRAC_SPACE] + size
                x_, y_, z_ = cells_pos[chosen_cell]
                grid[x, y, z, GRID_ATTR_FRAC_CONSUME] = grid[x_, y_, z_, GRID_ATTR_FRAC_CONSUME]
                grid[x, y, z, GRID_ATTR_INDEX] = ind
                cell_indices.append(ind)
                cells_pos[ind, :] = x, y, z
                cells[ind, CELL_ATTR_SIZE] = size  # set size
                cells[ind, 1:] = cells[chosen_cell, 1:]  # clone rest
                ncells += 1
                diff = 1

    return ncells, diff

@jit(nopython=True)
def therapy(therapy, grid, cells_pos, cells, cell_indices, c, parameters, t, filename, logPath):
    sum_cell = 0.
    sum_conv = 0.
    parameters["p_mit_cata"] = therapy[2]

    for chosen_cell in cell_indices:
        x, y, z = cells_pos[chosen_cell]

        if cells[chosen_cell, CELL_ATTR_STATE] != CELL_STATE_ANOXIC:
            sum_cell += 1.
            if c[x, y, z, 0] > 11.:
                oer = 1  # oxygen enhancement ratio
            else:
                oer = 3 - 2 * c[x, y, z, 0] / 11.
            d_oer = therapy[1] / oer

            # linear quadratic model
            s_rt = np.exp(-parameters["gammaR"] * (parameters["alphaR"] * d_oer + parameters["betaR"] * d_oer**2))

            if npr.uniform(0, 1) > s_rt:
                # marked for mitotic catastrophy
                cells[chosen_cell, CELL_ATTR_STATE] = CELL_STATE_MITCATA
                cells[chosen_cell, CELL_ATTR_T_LAST_TREAT] = t
                sum_conv += 1.

    with objmode():
        print2("Percentage of effected cells:",filename,logPath=logPath)
        print2(str(round(100.*sum_conv/sum_cell,2)),filename,logPath=logPath)
