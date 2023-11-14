"""
Content:
Compare two simulations whether the starting condition influences the simulation
outcome or if the behavior transient.

Tags:
starting condition
"""

import sys
sys.path.insert(0,'..')
from tools import *
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from ode_setVolume import *
from o2_models import o2_models
import models
from CONSTANTS import *

data = {
    "logFile": "slice_model_FG_HCT_116_dr_5_1_log",
    "simulate" : True,
    "tmax" : 18*24*60*60,
    "p_mit_cata" : 0.,
    "dose" : 0., # Gy
    "alphaR" : 0., # 1/Gy
    "betaR" : 0.,  # 1/Gy**2
    "gammaR" : 0., # unitless
    "max_iter_time": 60,
    "v0" : 0.85e-10 * 1e15,
    "rmax": 1.1e-3,
    "og_a": 27.92,
    "plate": "000",
    "cellline" : "HCT_116",
    "therapy" : "",
    "exp_name" : "Gri2016_Fig4a_exp",
    "sim_name" : "",
    "d_offset" : 0,
}

values, lmfit_parameters = loadParametersFromLog(data["logFile"])
#prettyPrintLmfitParameters(lmfit_parameters)

m = lmfit_parameters["m"].value
v0 = data["v0"]
lmfit_parameters["v0"].value = data["v0"]
rmax = data["rmax"] * m
dr = lmfit_parameters["dr"].value
max_iter_time = data["max_iter_time"]
tmax = data["tmax"]
name_addon = "" if not "name_addon" in data else data["name_addon"]
datasource = "" if not "datasource" in data else data["datasource"]
if name_addon != "":
    print("    [-] " + name_addon)

t_therapy = np.array([data["tmax"]]) # d
d_therapy = np.array([data["dose"]]) # Gy
p_therapy = np.array([data["p_mit_cata"]])
alphaR = data["alphaR"] # 1/Gy
betaR = data["betaR"]  # 1/Gy**2
gammaR = data["gammaR"] # unitless

cellline = data["cellline"]
plate = data["plate"]
model = "slice_model_RT"
#model = "slice_model_FG"
#filename = "plot_log_" + data["logFile"]
filename = "fit_result_" + name_addon + cellline + "_RT_" + str(data["dose"]) + "_" + data["exp_name"].split("_")[0]

if not "alphaR" in lmfit_parameters:
    lmfit_parameters.add("alphaR",value=alphaR)
if not "betaR" in lmfit_parameters:
    lmfit_parameters.add("betaR",value=betaR)
if not "gammaR" in lmfit_parameters:
    lmfit_parameters.add("gammaR",value=gammaR)
if not "p_mit_cata" in lmfit_parameters:
    lmfit_parameters.add("p_mit_cata",value=0.5)

gridPoints = int(round(rmax/dr,2))
t = np.linspace(0, tmax, max_iter_time)

exp_parameters = deepcopy(lmfit_parameters)
exp_parameters["a"].value = data["og_a"]


MyClass = getattr(models,model)
instance = MyClass()

c = setVolume_3(lmfit_parameters, t, model)

print("[*] Simulation 1")

specialParameters = {"c":c, "filename":filename, "t_therapy":t_therapy, "d_therapy":d_therapy, "p_therapy":p_therapy}
arg1 = instance.getValue(lmfit_parameters, t, specialParameters=specialParameters)
sol1 = arg1[0]
metrics1 = instance.getMetrics(lmfit_parameters, sol1)

################################################################################

copy_parameters = deepcopy(lmfit_parameters)
copy_parameters["v0"].value *= 0.2

fak = 1. if not "og_dr" in copy_parameters else copy_parameters["dr"].value / copy_parameters["og_dr"].value
c0 = setVolume_2(copy_parameters["v0"].value, gridPoints, copy_parameters, fak)
rad = np.array(o2_models.sauerstoffdruck_Gri2016(v0, copy_parameters))
c1 = setVolume_2(4. * rad[0]**3 * np.pi / 3., gridPoints, copy_parameters, fak)
c1 = np.zeros((len(c0)))
c0 -= c1
c = np.append(c0, c1)
c = np.append(c, np.zeros(len(c0)))

print("[*] Simulation 2")

specialParameters = {"c":c, "filename":filename, "t_therapy":t_therapy, "d_therapy":d_therapy, "p_therapy":p_therapy}
arg2 = instance.getValue(copy_parameters, t, specialParameters=specialParameters)
sol2 = arg2[0]
metrics2 = instance.getMetrics(copy_parameters, sol2)

################################################################################
#                               Plotting                                       #
################################################################################

t = t / (24.*3600.)

plt.figure()

plt.plot(t, metrics1[3] * 1e6, label=r"$R_{\mathrm{spheroid}}$", color=COLOR_PROLIFERATION, linestyle="solid",alpha=0.5)
plt.plot(t, metrics1[5] * 1e6, label=r"$R_{\mathrm{necrotic}}$", color=COLOR_ANOXIC, linestyle="solid",alpha=0.5)

plt.plot(t-3.55, metrics2[3] * 1e6, label=r"$R_{\mathrm{spheroid}}$ - 20%", color=COLOR_PROLIFERATION, linestyle="dashed")
plt.plot(t-3.55, metrics2[5] * 1e6, label=r"$R_{\mathrm{necrotic}}$ - 20%", color=COLOR_ANOXIC, linestyle="dashed")


plt.ylabel("radius $R$ [$\mu m$]")
plt.xlabel("time $t$ [d]")
plt.legend()

#plt.savefig(getPath()["bilder"] + "startingcondition" + ".pdf",transparent=True)
plt.savefig(getPath()["bilder"] + "startingcondition" + ".pdf",transparent=False)
plt.show()
