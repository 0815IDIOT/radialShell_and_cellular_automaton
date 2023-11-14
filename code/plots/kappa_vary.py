"""
Content:
Compare the dynamics with adjusted kappa.

Tags:
growth prediction, kappa, comparison 
"""

import sys
sys.path.insert(0,'..')
from tools import *
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from ode_setVolume import *
import models
from CONSTANTS import *

p_mit_cata_fadu = np.array([0.16226254,0.81761204])
alphaR_fadu = 0.34998672
betaR_fadu = 0.07939085

data = {
    "logFile": "slice_model_RT_FaDu_dr_12_2_log",
    "simulate" : True,
    #"tmax" : np.array([3,47])*24*60*60,
    "tmax" : np.array([3,33])*24*60*60,
    "p_mit_cata" : p_mit_cata_fadu,
    "dose" : 20, # Gy
    "alphaR" : alphaR_fadu, # 1/Gy
    "betaR" : betaR_fadu,  # 1/Gy**2
    "gammaR" : 1.0, # unitless
    "max_iter_time": 60,
    #"v0" : 3.07e-11 * 1e15,
    "v0" : 29731.56352162, # slice_model_RT_FaDu_20Gy_dr_14_1_2_log
    "sim_label" : "Brüningk et al., 2019",
    "rmax": 1.1e-3,
    "og_a": 10.64,
    "plate": "110",
    "cellline" : "FaDu",
    "therapy" : "",
    "exp_name" : "",
    "sim_name" : "",
    "d_offset" : 0,
    "4Plots" : True,
    "volumePlot" : False,
    "singlePlots" : False,
    "video": False,
}

values, lmfit_parameters = loadParametersFromLog(data["logFile"])

m = lmfit_parameters["m"].value
rmax = data["rmax"] * m
lmfit_parameters["v0"].value = data["v0"]
dr = lmfit_parameters["dr"].value
max_iter_time = data["max_iter_time"]
tmax = data["tmax"]

t_therapy = data["tmax"] # d
d_therapy = np.array([data["dose"],0]) # Gy
p_therapy = data["p_mit_cata"]
alphaR = data["alphaR"] # 1/Gy
betaR = data["betaR"]  # 1/Gy**2
gammaR = data["gammaR"] # unitless

cellline = data["cellline"]
plate = data["plate"]
model = "slice_model_RT"

filename = "kappa_vary"

if not "alphaR" in lmfit_parameters:
    lmfit_parameters.add("alphaR",value=alphaR)
else:
    lmfit_parameters["alphaR"].value = alphaR

if not "betaR" in lmfit_parameters:
    lmfit_parameters.add("betaR",value=betaR)
else:
    lmfit_parameters["betaR"].value = betaR

if not "gammaR" in lmfit_parameters:
    lmfit_parameters.add("gammaR",value=gammaR)
else:
    lmfit_parameters["gammaR"].value = gammaR

if not "p_mit_cata" in lmfit_parameters:
    lmfit_parameters.add("p_mit_cata",value=0.5)
else:
    lmfit_parameters["p_mit_cata"].value = 0.5

gridPoints = int(round(rmax/dr,2))
t = None
start = 0
for i in range(len(tmax)):
    end = tmax[i]
    t_new = np.linspace(start,end,max_iter_time)
    #if i == 1:
    #    t_new = np.linspace(start,end,max_iter_time*10)
    start = tmax[i]
    t = t_new if t is None else np.append(t, t_new, axis=0)
t = np.unique(t)

exp_parameters = deepcopy(lmfit_parameters)
exp_parameters["a"].value = data["og_a"]

MyClass = getattr(models,model)
instance = MyClass()

linewidth = 1.5
linewidth_err = 1.5
fontsize_legend = 8
fontsize_axis_label = 10
fontsize_axis_tick = 10
capsize = 0#5
capthick = linewidth_err

plt.figure()

faks = [20./30.,1.,40./30.]
linestyle = ["dashed", "solid", "dotted"]

for i in faks:
    
    print("Fak: " + str(i))

    lmfit_parameters_c = deepcopy(lmfit_parameters)
    lmfit_parameters_c["gamma"].value /= i
    lmfit_parameters_c["dr"].value *= i
    kappa = lmfit_parameters_c["dr"].value / lmfit_parameters_c["og_dr"].value

    c = setVolume_3(lmfit_parameters_c, t, model)

    specialParameters = {"c":c, "filename":filename, "t_therapy":t_therapy, "d_therapy":d_therapy, "p_therapy":p_therapy}
    arg = instance.getValue(lmfit_parameters_c, t, specialParameters=specialParameters)
    sol = arg[0]
    metrics = instance.getMetrics(lmfit_parameters_c, sol)

    plt.plot(t / (24.*3600.), metrics[3] * 1e6, label=r"$R_{\mathrm{spheroid}} - \ln(2)/\gamma=" + str(round(i*30.,2)) + r"$h; $\kappa=" + str(round(kappa,2)) + r"$", color=COLOR_PROLIFERATION, linestyle=linestyle[faks.index(i)], linewidth=linewidth)

x_exp_mean, x_exp_var, y_exp_mean, y_exp_var,_ = loadMultiExperiments(cellline, data["exp_name"], plate, data["therapy"])
exp_data_v0 = [x_exp_mean, y_exp_mean, x_exp_var, y_exp_var, cellline]

fak = 3
x_mean = exp_data_v0[0]
y_mean = exp_data_v0[1] * 1e6**fak
x_var = exp_data_v0[2]
y_var = exp_data_v0[3] * (1e6**fak)**2
label = exp_data_v0[4].replace("_"," ")

label = "$R_{\mathrm{spheroid}} -$ Exp."
y_var = ((((y_mean + np.sqrt(y_var)) / np.pi)*(3./4.))**(1./3.) - ((y_mean / np.pi)*(3./4.))**(1./3.))**2
y_mean = ((y_mean / np.pi)*(3./4.))**(1./3.)

plt.errorbar(x_mean, y_mean, np.sqrt(y_var), np.sqrt(x_var), color=COLOR_ERRORBAR, linewidth=linewidth_err, linestyle="none", fmt="^", markersize=6., capsize=capsize, capthick=capthick, label=label, zorder=1)

ax = plt.gca()
plt.xticks(np.arange(min(t/(24.*3600.)), max(t/(24.*3600.))+1, 5.0))
plt.ylabel("radius $R$ [$\mu m$]", fontsize=fontsize_axis_label)
plt.xlabel("time $t$ [d]", fontsize=fontsize_axis_label)
plt.setp(ax.get_xticklabels(), fontsize=fontsize_axis_tick)
plt.setp(ax.get_yticklabels(), fontsize=fontsize_axis_tick)
plt.legend(fontsize=fontsize_legend, frameon=False)

#plt.savefig(getPath()["bilder"] + "kappa_vary" + ".pdf",transparent=True)
plt.savefig(getPath()["bilder"] + "kappa_vary" + ".pdf",transparent=False)
plt.show()