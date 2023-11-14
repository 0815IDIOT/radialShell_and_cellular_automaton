"""
Content:
Redo o2_validate_1 with sauerstoffdruck_analytical_4+

Tags:
o2, validation
"""

import sys
sys.path.insert(0,'..')
from o2_models import o2_models
import numpy as np
from tools import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ode_setVolume import *
from CONSTANTS import *

simulations = [
    {
        "logFile" : "slice_model_RT_HCT_116_dr_11_1_log",
        "plate": "000",
        "cellline" : "HCT_116",
        "exp_name" : "Gri2016_Fig4a_exp",
    },
    {
        "logFile": "slice_model_RT_HCT_116_Bru2019_Fig5a_dr_11_2_log",
        "plate": "000",
        "cellline" : "HCT_116",
        "exp_name" : "Bru2019_Fig5a",
    },
    {
        "logFile": "slice_model_RT_MDA_MB_468_dr_11_4_log",
        "plate": "000",
        "cellline" : "MDA_MB_468",
        "exp_name" : "Gri2016_Fig4c_exp",
    },
    {
        "logFile": "slice_model_RT_LS_174T_dr_11_2_log",
        "plate": "000",
        "cellline" : "LS_174T",
        "exp_name" : "Gri2016_Fig4b_exp",
    },
    {
        "logFile": "slice_model_RT_SCC_25_dr_11_1_log",
        "plate": "000",
        "cellline" : "SCC_25",
        "exp_name" : "Gri2016_Fig4d_exp",
    },
    {
        "logFile": "slice_model_RT_FaDu_dr_11_1_log",
        "plate": "",
        "cellline" : "FaDu",
        "datasource" : "source2",
        "exp_name" : "",
    },
]

"""
0: HCT Gri
1: HCT Bru
2: MDA
3: LS
4: SCC
5: FaDu
"""

ref_sim = 5

cellline = simulations[ref_sim]["cellline"]
exp_name = simulations[ref_sim]["exp_name"]
plate = simulations[ref_sim]["plate"]
values, lmfit_parameters = loadParametersFromLog(simulations[ref_sim]["logFile"])
lmfit_parameters.add("dose",value=0)
if not "dose" in lmfit_parameters:
    lmfit_parameters.add("dose",value=0)
if not "alphaR" in lmfit_parameters:
    lmfit_parameters.add("alphaR",value=0)
if not "betaR" in lmfit_parameters:
    lmfit_parameters.add("betaR",value=0)
if not "gammaR" in lmfit_parameters:
    lmfit_parameters.add("gammaR",value=0)
if not "p_mit_cata" in lmfit_parameters:
    lmfit_parameters.add("p_mit_cata",value=0)

t = np.linspace(0, lmfit_parameters["tmax"], 60)
lmfit_parameters["v0"].value *= 15.

c = setVolume_3(lmfit_parameters, t, "slice_model_RT")
c = np.array(np.split(c,3))
c0 = c[0]
cn = c[1]

idx = 10

dr = lmfit_parameters["dr"].value
radius = np.arange(0.5, len(c0) + 0.5, 1) * dr * 1e1

################################################################################
linewidth = 2
fontsize_legend = 11
fontsize_axis_label = 12
fontsize_axis_tick = 12
################################################################################

plt.figure(figsize=(8.,10.))
plt.subplot(211)

lgds = plt.plot(radius[:idx], c0[:idx], color=COLOR_PROLIFERATION, linestyle="solid", label = r"$c_p$", linewidth=linewidth)
#lgds += plt.plot(radius[:idx], cn[:idx], color=COLOR_ANOXIC, linestyle="solid", label = r"$c_n$")
#lgds += plt.plot(radius[:idx], (c0 + cn)[:idx], color=COLOR_SUM, linestyle="dotted", label = r"$c_p+c_n$")
#plt.xlabel(r"distance $r$ from spheroid center [$\mu$m]", fontsize=fontsize_axis_label)
plt.ylabel(r"relative cell concentration $c_T(r_i)$", fontsize=fontsize_axis_label)
plt.ylim(-0.1, 1.1)

ax = plt.gca()
ax2 = ax.twinx()
ax2.title.set_color(COLOR_OXYGEN)
ax2.yaxis.label.set_color(COLOR_OXYGEN)
ax2.tick_params(axis='y', colors=COLOR_OXYGEN)
ax2.spines['right'].set_color(COLOR_OXYGEN)
plt.ylabel(r"oxygen level $\rho(r_i)$ [mmHg]", fontsize=fontsize_axis_label)
plt.ylim(-110./11., 110)
plt.xlim(-51, 700)
plt.setp(ax.get_xticklabels(), fontsize=fontsize_axis_tick)
plt.setp(ax.get_yticklabels(), fontsize=fontsize_axis_tick)
plt.setp(ax2.get_xticklabels(), fontsize=fontsize_axis_tick)
plt.setp(ax2.get_yticklabels(), fontsize=fontsize_axis_tick)

r_an = [150,175,200,225,250]

for r in r_an:
    p_ana, rh_ana, rcrit_ana, raw_ana = o2_models.sauerstoffdruck_analytical_4(params=lmfit_parameters, c0 = c0, mode=2, rcrit=r*10**(-1))
    #lgd = plt.plot(radius[:idx], raw_ana[:idx], label=r"$r_{an}=$" + str(r) + r"$\mu$m", linestyle="dashed")
    lgd = plt.plot(p_ana*10, raw_ana, label=r"$r_{an}=$" + str(r) + r"$\mu$m", linestyle="dashed", linewidth=linewidth)
    lgds += lgd
    plt.vlines(r,-5,10,color=lgd[0].get_color(), linewidth=linewidth)

#p_ana, rh_ana, rcrit_ana, raw_ana = o2_models.sauerstoffdruck_analytical_4(params=lmfit_parameters, c0 = c0, mode=2)
#idx = np.where(p_ana == 100)[0][0] + 1
#lgds += plt.plot(radius[:idx], p_ana[:idx], label="O2", color=COLOR_OXYGEN)

plt.title("(a)",loc="left")
plt.hlines(0,0,radius[:idx][-1], color = "gray", alpha=0.5, linewidth=linewidth)
labs = [l.get_label() for l in lgds]
plt.legend(lgds, labs, loc=5, frameon=False, fontsize=fontsize_legend)

ax = plt.gca()
rect = patches.Rectangle((110,-7), 200, 21, linewidth=linewidth, edgecolor="black", facecolor='none')
ax.add_patch(rect)

################################################################################

plt.subplot(212)

lgds = plt.plot(radius[:idx], c0[:idx], color=COLOR_PROLIFERATION, linestyle="solid", label = r"$c_p$", linewidth=linewidth)

plt.xlabel(r"distance $r$ from spheroid center [$\mu$m]", fontsize=fontsize_axis_label)
plt.ylabel(r"relative cell concentration $c_T(r_i)$", fontsize=fontsize_axis_label)
plt.ylim(-0.07, 0.14)

ax = plt.gca()
ax2 = ax.twinx()
ax2.title.set_color(COLOR_OXYGEN)
ax2.yaxis.label.set_color(COLOR_OXYGEN)
ax2.tick_params(axis='y', colors=COLOR_OXYGEN)
ax2.spines['right'].set_color(COLOR_OXYGEN)
plt.ylabel(r"oxygen level $\rho(r_i)$ [mmHg]", fontsize=fontsize_axis_label)
plt.ylim(-7, 14)
plt.xlim(110, 310)
plt.setp(ax.get_xticklabels(), fontsize=fontsize_axis_tick)
plt.setp(ax.get_yticklabels(), fontsize=fontsize_axis_tick)
plt.setp(ax2.get_xticklabels(), fontsize=fontsize_axis_tick)
plt.setp(ax2.get_yticklabels(), fontsize=fontsize_axis_tick)

r_an = [150,175,200,225,250]

for r in r_an:
    p_ana, rh_ana, rcrit_ana, raw_ana = o2_models.sauerstoffdruck_analytical_4(params=lmfit_parameters, c0 = c0, mode=2, rcrit=r*10**(-1))
    lgd = plt.plot(p_ana*10, raw_ana, label=r"$r_{an}=$" + str(r) + r"$\mu$m", linestyle="dashed", linewidth=linewidth)
    lgds += lgd
    plt.vlines(r,-5,10,color=lgd[0].get_color(), linewidth=linewidth)

plt.title("(b)",loc="left")
plt.hlines(0,0,radius[:idx][-1], color = "gray", alpha=0.5)

################################################################################

plt.tight_layout()
#plt.savefig(getPath()["bilder"] + "/validate_o2_3.pdf",transparent=True)
plt.savefig(getPath()["bilder"] + "/validate_o2_7.pdf",transparent=False)
plt.show()