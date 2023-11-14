"""
Content:
Similiar to o2_validate_1 but with more realistic cell concentrations and 
parameters.

Tags:
o2 validation, grimes, hybrid, numeric, analytical
"""

import sys
sys.path.insert(0,'..')
from o2_models import o2_models
import numpy as np
from tools import *
import matplotlib.pyplot as plt
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

def o2_functions(c0, cn, singlePlot = False):
    
    dr = lmfit_parameters["dr"].value
    radius = np.arange(0.5, len(c0) + 0.5, 1) * dr * 1e1

    idx = 10 if singlePlot else len(radius)

    lgds = plt.plot(radius[:idx], c0[:idx], color=COLOR_PROLIFERATION, linestyle="solid", label = r"$c_p$")
    lgd = plt.plot(radius[:idx], cn[:idx], color=COLOR_ANOXIC, linestyle="solid", label = r"$c_n$")
    lgds = lgds + lgd
    if singlePlot:
        lgd = plt.plot(radius[:idx], (c0 + cn)[:idx], color=COLOR_SUM, linestyle="dotted", label = r"$c_p+c_n$")
        lgds = lgds + lgd    
    plt.xlabel(r"distance $r$ from spheroid center [$\mu$m]")
    plt.ylabel(r"relative cell concentration $c_T(r_i)$")
    plt.ylim(-0.1, 1.1)

    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.title.set_color(COLOR_OXYGEN)
    ax2.yaxis.label.set_color(COLOR_OXYGEN)
    ax2.tick_params(axis='y', colors=COLOR_OXYGEN)
    ax2.spines['right'].set_color(COLOR_OXYGEN)
    plt.ylabel(r"oxygen level $\rho(r_i)$ [mmHg]")
    plt.ylim(-110./11., 110)

    v_max = 4. * np.pi * ((np.arange(1,len(c0)+1,1) * dr)**3 - (np.arange(0,len(c0),1) * dr)**3) / 3.
    v = np.sum((c0 + cn) * v_max)
    print("[*] Grimes")
    p_gri = o2_models.sauerstoffdruck_Gri2016_radius(radius/1e1, v, lmfit_parameters)
    idx = np.where(p_gri == 100)[0][0] + 1 if singlePlot else len(p_gri)
    lgd = plt.plot(radius[:idx], p_gri[:idx], label="Grimes et al., (2016)", color="blue")
    lgds = lgds + lgd  

    print("[*] Hybrid-Analytical V4")
    p_ana, rh_ana, rcrit_ana, raw_ana = o2_models.sauerstoffdruck_analytical_4(params=lmfit_parameters, c0 = c0, mode=2)
    idx = np.where(p_ana == 100)[0][0] + 1 if singlePlot else len(p_gri)
    lgd = plt.plot(radius[:idx], p_ana[:idx], label="Hybrid Analytical", color=COLOR_OXYGEN)
    #plt.plot(radius, raw_ana, label="Hybrid Analytical V4 Raw", color="purple")
    lgds = lgds + lgd  

    print("[*] Hybrid-Analytical V6")
    #p_ana, rh_ana, rcrit_ana, raw_ana = o2_models.sauerstoffdruck_analytical_6(params=lmfit_parameters, c0 = c0, cn = cn)
    #plt.plot(radius, p_ana, label="Hybrid Analytical V6", color="cyan")

    print("[*] Numeric")
    #sol_num, p_hist = o2_models.sauerstoffdruck_numeric(lmfit_parameters, c0, mode = 2, p = p_gri)
    #plt.plot(radius, sol_num, label="numeric", color="orange")

    print("[*] Scipy")
    #sol_scipy = o2_models.sauerstoffdruck_scipy(lmfit_parameters, c0, mode = 2, p = p_gri)
    #plt.plot(radius, sol_scipy, label="scipy/numeric", color="purple", linestyle="dashed")

    labs = [l.get_label() for l in lgds]
    plt.legend(lgds, labs, loc=5, frameon=False)

plt.figure(figsize=(20,10))
plt.subplot(321)
plt.title("Original FaDu Fit")

t = np.linspace(0, lmfit_parameters["tmax"], 60)
lmfit_parameters["v0"].value *= 10.
lmfit_parameters["dr"].value *= 0.5
prettyPrintLmfitParameters(lmfit_parameters)
c = setVolume_3(lmfit_parameters, t, "slice_model_RT")
c = np.array(np.split(c,3))
c0 = c[0]
cn = c[1]

#n = 64
#c0[:n] = 1. - cn[:n]

o2_functions(c0, cn)

################################################################################

plt.subplot(322)
plt.title("Dummy Concentracion #1")

D_c = 3.0e-05 # 1/s
D_p = 20.0 # m^2/s
a = 10.30247271 # mmHg/s
gamma = 8.898e-06 # / 3600. # 1/s
delta = 2.684e-06 # rate nec clear
epsilon = 1.126e-04 # rate hypox -> nec
dr = 5.87088013 # m
og_dr = 1.36
pcrit = 0.
phypox = 0.5
p0 = 100.
tmax = 100#10*24*60*60
max_iter_time = 50
rmax = 110.0

gridPoints = int(round(rmax/dr,2))

dict = {
    "D_p": {"value" : D_p},
    "a": {"value" : a},
    "D_c": {"value" : D_c},
    "dr": {"value" : dr},
    "og_dr": {"value" : og_dr},
    "gamma" : {"value" : gamma},
    "delta" : {"value" : delta},
    "epsilon" : {"value" : epsilon},
    "pcrit" : {"value" : pcrit},
    "phypox" : {"value" : phypox},
    "p0" : {"value" : p0},
    "D" : {"value" : D_p},
    "ph" : {"value" : phypox},
    "rd" : {"value" : 1000}
}

lmfit_parameters2 = dictToLmfitParameters(dict)

c5 = np.array([6.51869516e-04, 2.77196620e-02, 9.00562042e-01, 9.99701137e-01,
 9.99796664e-01, 9.99841668e-01, 9.83234270e-01, 5.21790117e-02,
 7.82001570e-05, 1.11437409e-07, 1.59069364e-10, 2.27350634e-13,
 3.25250926e-16, 4.65645805e-19, 6.67018195e-22, 9.55901796e-25,
 1.37038071e-27, 1.95930336e-30])

c5_n = np.array([1.-6.51869516e-04, 1.-2.77196620e-02, 1.-9.00562042e-01,  0.,         0.,         0.,
 0.,         0.,         0.,         0.,         0.,         0.,
 0.,         0.,         0.,         0.,         0.,         0.,        ])

c6 = np.ones((len(c5)))
c6[7:] = 0.
c6[7] = 1.0
c6_n = np.zeros((len(c5)))
c6_n[:3] = 1
c6 = c6 - c6_n

c0 = c6
cn = c6_n

o2_functions(c0, cn)

################################################################################

plt.subplot(323)
plt.title("Dummy Concentracion #2\n(altere necrotic concentration)")

cn[2] = 0.8
c0 = c0 - cn

o2_functions(c0, cn)

################################################################################

plt.subplot(324)
plt.title("Dummy Concentracion #3\n(c0[7]=0.11)")

c0 = c6
c0[7] = 0.11
cn = c6_n

o2_functions(c0, cn)

################################################################################

plt.subplot(326)
plt.title("Dummy Concentracion #5\n(c0[7]=0.1)")

c0 = c6
c0[7] = 0.1
cn = c6_n

o2_functions(c0, cn)

plt.tight_layout()
plt.show()

################################################################################
#------------------------------------------------------------------------------#
################################################################################

plt.figure()
t = np.linspace(0, lmfit_parameters["tmax"], 60)
#lmfit_parameters["v0"].value /= 10.
lmfit_parameters["dr"].value /= 0.5

c = setVolume_3(lmfit_parameters, t, "slice_model_RT")
c = np.array(np.split(c,3))
c0 = c[0]
cn = c[1]

o2_functions(c0, cn, singlePlot=True)

#plt.savefig(getPath()["bilder"] + "o2_validate_6" + ".pdf",transparent=True)
plt.savefig(getPath()["bilder"] + "o2_validate_6" + ".pdf",transparent=False)
plt.show()