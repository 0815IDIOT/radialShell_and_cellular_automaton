"""
Content:
Get the relativ errors of the 1D fit to the mean CA3D simulations.

Tags:
ca3d, fak, kappa, relativ error, relativer fehler
"""

import sys
sys.path.insert(0,'../../../.')
from tools import *
import numpy as np
import re
import matplotlib.pyplot as plt
from copy import deepcopy
from ode_setVolume import *
import models

data = {}
neighborhoods = ["neum1", "neum1moor1", "moor1", "neum2", "neum2moor2", "neum3", "neum3moor2", "moor2", "neum3moor3", "moor3"]
path = getPath()["ca3d"]
filename = "CA3D_FaDu__"
model = "slice_model_RT"
logfile = "CA3D_Fak_2_log.txt"
debug = False

def getDatapoints(file):

    data = file["data"]
    parameters = file["parameters"].item()
    therapy_schedule = file["therapy_schedule"]
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

    return t, y

print("[*] Load data from CA3D and mean them")

for gamma in [0.5,1.,2.]:
    data[str(gamma).replace(".","")] = {}
    for n in neighborhoods:
        y_mean = None
        for i in range(5):
            file = np.load(path + "experiments/" + filename + n + "_caliFak_No" + str(i+1) + "_gamma" + str(gamma).replace(".","") + ".npz", allow_pickle=True)
            t, y = getDatapoints(file)
            if y_mean is None:
                y_mean = y
            else:
                y_mean += y
        y_mean /= 5.
        data[str(gamma).replace(".","")][n] = {"ca3d" : y_mean}

print("[*] load 1D Fit results")

dr = []

with open(getPath()["log"] + logfile, "r") as f:
    lines = f.readlines()
    for line in lines:
        x = re.search("dr:[ ]+[0-9]+\.[0-9]+ 1σ in \[[0-9]+\.[0-9]+, [0-9]+\.[0-9]+\]",line)
        if x is not None:
            y = re.findall("[0-9]+\.[0-9]+",x.group(0))
            dr.append(float(y[0]))

i = 0
for gamma in [0.5,1.,2.]:
    j = 0
    for n in neighborhoods:
        data[str(gamma).replace(".","")][n]["1d_dr"] = dr[j + i * len(neighborhoods)]
        j += 1
    i += 1

print("[*] Start resimulating")

values, lmfit_parameters = loadParametersFromLog("slice_model_RT_FaDu_dr_10_1_log")
lmfit_parameters["v0"].value *= 2.
lmfit_parameters["epsilon"].value *= 0.
lmfit_parameters["delta"].value *= 0.
#lmfit_parameters["gamma"].value *= 0.5
lmfit_parameters["tmax"].value = 6.*24.*3600.
lmfit_parameters["dr"].vary = True
lmfit_parameters["dr"].value = lmfit_parameters["og_dr"].value
lmfit_parameters["dr"].min = 0.5*lmfit_parameters["og_dr"].value
lmfit_parameters["dr"].max = 10. * lmfit_parameters["og_dr"].value
lmfit_parameters.add("p_mit_cata",value=0.5,vary=False)

copy_parameter = deepcopy(lmfit_parameters)

c = setVolume_3(lmfit_parameters, t, model)

MyClass = getattr(models,model)
instance = MyClass()

atol = 1e-6 # 1e-6 # 1e-8
rtol = 1e-3 # 1e-3 # 1e-5

rel_error_list = []

for gamma in data:
    print("    [+] gamma: " + gamma)
    for n in data[gamma]:
        print("        [-] neighborhood: " + n)
        lmfit_parameters = deepcopy(copy_parameter)
        lmfit_parameters["gamma"].value *= float(gamma)/10.
        lmfit_parameters["dr"].value = data[gamma][n]["1d_dr"]

        specialParameters={"c":c, "atol":atol, "rtol":rtol}
        arg = instance.getValue(lmfit_parameters, t, specialParameters=specialParameters)
        sol = arg[0]
        metrics = instance.getMetrics(lmfit_parameters, sol)

        r0_1d = metrics[3] * 1e6
        r0_ca = data[gamma][n]["ca3d"]

        """
        print(r0_1d)
        print(r0_ca)
        print(np.abs(r0_1d - r0_ca))
        print(np.min(np.array([r0_1d,r0_ca]),axis=0))
        """
        rel_error = np.max(np.abs(r0_1d - r0_ca) / np.min(np.array([r0_1d,r0_ca]),axis=0))
        rel_error_list.append(rel_error)
        print("            [#] relative error: " + str(rel_error))

print("[*] min. relativ Error: " + str(np.min(rel_error_list))) # 0.0017608535088423371
print("[*] max. relativ Error: " + str(np.max(rel_error_list))) # 0.0094313014253602
