"""
Content:
Compare the growth of the 1D RS model with a simple prediction.

Tags:
growth prediction, comparison
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

#p_mit_cata_fadu = np.array([0.31302445, 0.87420515])
p_mit_cata_fadu = np.array([0.0, 0.87420515])
alphaR_fadu = 0.77807491
betaR_fadu = 0.01190783

data = {
    "logFile": "slice_model_RT_FaDu_dr_12_1_log",#"slice_model_RT_FaDu_dr_11_1_log",#"slice_model_RT_FaDu_dr_10_1_log",#"slice_model_FG_FaDu_dr_5_1_log",
    "simulate" : True,
    "tmax" : np.array([25,55])*24*60*60,#np.array([10,30])*24*60*60,#np.array([3,33])*24*60*60,
    "p_mit_cata" : p_mit_cata_fadu,
    "dose" : 20, # Gy
    "alphaR" : alphaR_fadu, # 1/Gy
    "betaR" : betaR_fadu,  # 1/Gy**2
    "gammaR" : 1.0, # unitless
    "max_iter_time": 60,
    "v0" : 3.07e-11 * 1e15,
    "rmax": 1.1e-3,
    "og_a": 10.64,
    "plate": "110",
    "cellline" : "FaDu",
    "therapy" : "",
    "exp_name" : "",
    "sim_name" : "",
    "d_offset" : 0,
}

values, lmfit_parameters = loadParametersFromLog(data["logFile"])
#prettyPrintLmfitParameters(lmfit_parameters)

m = lmfit_parameters["m"].value
v0 = data["v0"] * 0.001
lmfit_parameters["v0"].value = v0
rmax = data["rmax"] * m
dr = lmfit_parameters["dr"].value
max_iter_time = data["max_iter_time"]
tmax = data["tmax"]
name_addon = "" if not "name_addon" in data else data["name_addon"]
datasource = "" if not "datasource" in data else data["datasource"]
if name_addon != "":
    print("    [-] " + name_addon)

t_therapy = data["tmax"] # d
d_therapy = np.array([data["dose"],0]) # Gy
p_therapy = data["p_mit_cata"]
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
#t = np.linspace(0, tmax[-1], max_iter_time)

exp_parameters = deepcopy(lmfit_parameters)
exp_parameters["a"].value = data["og_a"]


MyClass = getattr(models,model)
instance = MyClass()

c = setVolume_3(lmfit_parameters, t, model)

specialParameters = {"c":c, "filename":filename, "t_therapy":t_therapy, "d_therapy":d_therapy, "p_therapy":p_therapy}
arg = instance.getValue(lmfit_parameters, t, specialParameters=specialParameters)
sol = arg[0]
metrics = instance.getMetrics(lmfit_parameters, sol)
"""
lmfit_parameters2 = deepcopy(lmfit_parameters)
lmfit_parameters2["llambda"].value *= 10.
lmfit_parameters2.add("faktor_logistic",value=100.)
arg2 = instance.getValue(lmfit_parameters2, t, specialParameters=specialParameters)
sol2 = arg2[0]
metrics2 = instance.getMetrics(lmfit_parameters2, sol2)
"""
################################################################################
#                               Plotting                                       #
################################################################################

t = t / (24.*3600.)

t_lin1 = np.logical_and(t<24.,t>14.)
t_lin2 = np.logical_and(t<38.,t>25.)
t_exp1 = np.logical_and(t<12.,t>0.)
t_exp2 = np.logical_and(t<55.,t>40.)
t_exp_nec = np.logical_and(t<55.,t>26.)
t_swap = 10.

y_lin = values["gamma"] * ((1. - p_mit_cata_fadu[0]) - p_mit_cata_fadu[0]) * dr * 1e1 * ((t[t_lin1]-15.) * 24.* 3600.) + 172.
y_lin2 = values["gamma"] * ((1. - p_mit_cata_fadu[1]) - p_mit_cata_fadu[1]) * dr * 1e1 * ((t[t_lin2]-25) * 24.* 3600.) + 459.
y_lin2[y_lin2<0] = None
y_exp1 = 19.4 * np.exp(values["gamma"] * ((1. - p_mit_cata_fadu[0]) - p_mit_cata_fadu[0])* ((t[t_exp1]) * 24.* 3600.) / 3.)
y_exp2 = 114. * np.exp(values["gamma"] * ((1. - p_mit_cata_fadu[1]) - p_mit_cata_fadu[1])* ((t[t_exp2]-40) * 24.* 3600.) / 3.)
y_exp_nec = 128. * np.exp(-values["delta"] * ((t[t_exp_nec]-26) * 24.* 3600.) / 3.)

plt.figure(figsize=(10.,5.))
plt.subplot(121)
plt.title("(a)", loc='left')

plt.plot(t, metrics[3] * 1e6, label=r"$R_{\mathrm{spheroid}}$ - 1D RS model", color=COLOR_PROLIFERATION, linestyle="solid")
plt.plot(t, metrics[5] * 1e6, label=r"$R_{\mathrm{necrotic}}$ - 1D RS model", color=COLOR_ANOXIC, linestyle="solid")
plt.plot(t[t_lin1], y_lin, color="black", linestyle="dashed", label="prediction linear phase")
plt.plot(t[t_exp1], y_exp1, color="black", linestyle="dotted")
plt.plot(t[t_exp2], y_exp2, color="black", linestyle="dotted", label="prediction exponential phase")
plt.plot(t[t_lin2], y_lin2, color="black", linestyle="dashed")
plt.plot(t[t_exp_nec], y_exp_nec, color="black", linestyle="dotted")

plt.axhspan(dr*1e1,3*dr*1e1,0,t[-1],color="gray",alpha=0.4)

plt.text(14.0, 280, r"$\sim\gamma(1-2P^1_{mc})\cdot dr\cdot t$", rotation=75)
plt.text(27.0, 280, r"$\sim\gamma(1-2P^2_{mc})\cdot dr\cdot t$", rotation=-75)

plt.ylabel("radius $R$ [$\mu m$]")
plt.xlabel("time $t$ [d]")
#plt.legend()

################################################################################

plt.subplot(122)
plt.title("(b)", loc='left')

#y_exp[y_exp<20] = None
#y_lin2[y_lin2<10] = None

plt.plot(t, metrics[3] * 1e6, label=r"$R_{\mathrm{spheroid}}$ - RS model", color=COLOR_PROLIFERATION, linestyle="solid")
idx = np.where(metrics[5] == 0)[0][-1]
metrics[5][idx] = metrics[5][-1]
metrics[5][metrics[5]==0] = None
plt.plot(t, metrics[5] * 1e6, label=r"$R_{\mathrm{necrotic}}$ - RS model", color=COLOR_ANOXIC, linestyle="solid")
#plt.plot(t, metrics2[3] * 1e6, label=r"$R_{\mathrm{spheroid}}$ - RS model", color=COLOR_PROLIFERATION, linestyle="dotted")
#plt.plot(t, metrics2[5] * 1e6, label=r"$R_{\mathrm{necrotic}}$ - RS model", color=COLOR_ANOXIC, linestyle="dotted")
plt.plot(t[t_lin1], y_lin, color="black", linestyle="dashed", label="linear prediction")
plt.plot(t[t_exp1], y_exp1, color="black", linestyle="dotted")
plt.plot(t[t_exp2], y_exp2, color="black", linestyle="dotted", label="exponential prediction")
plt.plot(t[t_lin2], y_lin2, color="black", linestyle="dashed")
plt.plot(t[t_exp_nec], y_exp_nec, color="black", linestyle="dotted")

plt.axhspan(dr*1e1,3*dr*1e1,0,t[-1],color="gray",alpha=0.5)

plt.text(-2, 30, r"$\sim\exp((1-2P^1_{mc})\cdot \frac{\gamma}{3}\cdot t)$", rotation=47)
plt.text(34, 45, r"$\sim\exp((1-2P^2_{mc})\cdot \frac{\gamma}{3}\cdot t)$", rotation=-35)
plt.text(35, 0.6, r"$\sim\exp(-\frac{\delta}{3}\cdot t)$", rotation=-63)

plt.ylabel("radius $R$ [$\mu m$]")
plt.xlabel("time $t$ [d]")
plt.yscale("log")
plt.legend()

#plt.savefig(getPath()["bilder"] + "growth_prediction" + ".pdf",transparent=True)
plt.savefig(getPath()["bilder"] + "growth_prediction" + ".pdf",transparent=False)
plt.show()