import sys
sys.path.insert(0,'../../../.')
from tools import *
import numpy as np
import matplotlib.pyplot as plt

filename = "CA3D_FaDu__moor3moor4_reproduce_ca3d_1d_v7"
path = getPath()["ca3d"]

print("[*] loading '" + filename + ".npz'")

file = np.load(path + "experiments/" + filename + ".npz", allow_pickle=True)

data = file["data"]
parameters = file["parameters"].item()
N_grid = file["N_grid"]
therapy_schedule = file["therapy_schedule"]
cell_indices = file["cell_indices"]
cells = file["cells"]
cells_pos = file["cells_pos"]
concentrations = file["concentrations"]
free_indices = file["free_indices"]

print("[*] Data Loaded")

t_save = []
t0 = 0
max_iter_time = int(parameters["ode_max_iter_time"])
tmax = parameters["tmax"]

if therapy_schedule[-1][0] > tmax:
        therapy_schedule[-1][0] = tmax
if therapy_schedule[-1][0] != tmax:
    therapy_schedule = np.append(therapy_schedule,[[tmax,0.,0.]],axis=0)

for i in range(len(therapy_schedule)):
        t = np.linspace(0, int(therapy_schedule[i][0])-t0, max_iter_time)
        t0 = therapy_schedule[i][0]
        t_save.append(t)
for i in range(len(t_save)):
        if i == 0:
            t = t_save[i]
        else:
            t = np.append(t,t_save[i]+t[-1])

ncell_list = data[1]
dsim = (3. / 4. / np.pi * ncell_list)**(1. / 3.) * parameters["ca_dx"] * 1e6
tsim = parameters["ca_dt"] * np.arange(len(ncell_list)) / 3600.0 / 24
y = np.interp(t/24./3600., tsim, dsim)

n = int(len(t)/(4.))
model1 = np.polyfit(t[n:]/24./3600.,y[n:],1)

print("[*] Anstieg: " + str(model1[0]))