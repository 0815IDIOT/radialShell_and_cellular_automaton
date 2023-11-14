"""
Content:
Lade ein eine log datei und erstelle den Plot daraus

Tags:
Log, plot
"""

import sys
sys.path.insert(0,'..')
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from tools import *
import models
from data_loader import data_loader
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from o2_models import o2_models
from ode_setVolume import *
from ode_plots import *

saveFig = True

### HCT ###
p_mit_cata_hct = np.array([0.44, 0.59])  # Brü
p_mit_cata_hct = np.array([0.27, 0.79081012])
p_mit_cata_hct = np.array([0.27294652, 0.79081012]) # slice_model_RT_HCT_116_Bru2019_Fig6_10gy_dr_2_1_1_log, slice_model_RT_HCT_116_Bru2019_Fig6_10gy_dr_2_2_1_log
p_mit_cata_hct = np.array([0.18111667, 0.68971938]) # slice_model_RT_HCT_116_Bru2019_Fig6_10gy_dr_8_1_1_log, slice_model_RT_HCT_116_Bru2019_Fig6_10gy_dr_8_2_2_log
p_mit_cata_hct = np.array([0.19109102,0.68587156]) # slice_model_RT_HCT_116_Bru2019_Fig6_10gy_dr_10_1_1_log, slice_model_RT_HCT_116_Bru2019_Fig6_10gy_dr_10_2_1_log
p_mit_cata_hct = np.array([0.15733116,0.69221963]) # slice_model_RT_HCT_116_Bru2019_Fig6_10gy_dr_11_1_2_log, slice_model_RT_HCT_116_Bru2019_Fig6_10gy_dr_11_2_2_log
p_mit_cata_hct = np.array([0.25932788,0.67875994]) # slice_model_RT_HCT_116_Bru2019_Fig6_10gy_dr_11_6_2_log
p_mit_cata_hct = np.array([0.26707656,0.67051367]) # slice_model_RT_HCT_116_Bru2019_Fig6_10gy_dr_12_1_1
p_mit_cata_hct_p0_80  = np.array([0.28236428,0.66200936]) # slice_model_RT_HCT_116_Bru2019_Fig6_10gy_dr_12_1_1_p0_80
p_mit_cata_hct_p0_120 = np.array([0.25557359,0.67130001]) # slice_model_RT_HCT_116_Bru2019_Fig6_10gy_dr_12_1_1_p0_120

alphaR_hct = 0.5 # 1/Gy # Brü
betaR_hct = 0.042
"""
alphaR_hct = 0.40083979 # 1/Gy # slice_model_RT_HCT_116_Bru2019_Fig6_5gy_dr_5_1_1_log
betaR_hct = 0.03177392

alphaR_hct = 0.73840244 # 1/Gy # slice_model_RT_HCT_116_Bru2019_Fig6_5gy_dr_8_1_1_log
betaR_hct = 0.03148478

alphaR_hct = 0.237763 # 1/Gy # slice_model_RT_HCT_116_Bru2019_Fig6_5gy_dr_10_3_1_log
betaR_hct = 0.14199332

alphaR_hct = 0.55390681 # 1/Gy # slice_model_RT_HCT_116_Bru2019_Fig6_5gy_dr_11_3_2_log
betaR_hct = 0.01055128
"""

### FaDu ###
p_mit_cata_fadu = np.array([0.30313692, 0.85518913]) # slice_model_RT_FaDu__dr_2_1_1_log, slice_model_RT_FaDu__dr_2_2_1_log
p_mit_cata_fadu = np.array([0.31302445, 0.87420515]) # slice_model_RT_FaDu_20Gy_dr_5_1_1_log, slice_model_RT_FaDu_20Gy_dr_5_2_1_log
p_mit_cata_fadu = np.array([0.19795485,0.77249277]) # slice_model_RT_FaDu_20Gy_dr_8_1_1_log, slice_model_RT_FaDu_20Gy_dr_8_2_1_log
p_mit_cata_fadu = np.array([0.21711524,0.76311264]) # slice_model_RT_FaDu_20Gy_dr_10_1_1_log, slice_model_RT_FaDu_20Gy_dr_10_2_1_log
p_mit_cata_fadu = np.array([0.19651351,0.81794283]) # slice_model_RT_FaDu_20Gy_dr_11_1_2_log, slice_model_RT_FaDu_20Gy_dr_11_2_2_log
p_mit_cata_fadu = np.array([0.17390229,0.82416023]) # slice_model_RT_FaDu_20Gy_dr_12_1_2_log
p_mit_cata_fadu = np.array([0.15063646,0.82571004]) # slice_model_RT_FaDu_20Gy_dr_13_1_1_log
p_mit_cata_fadu = np.array([0.15154592,0.82501252]) # slice_model_RT_FaDu_20Gy_dr_14_1_1_log
p_mit_cata_fadu = np.array([0.16226254,0.81761204]) # slice_model_RT_FaDu_20Gy_dr_14_1_2_log
#p_mit_cata_fadu = np.array([0.5, 0.87420515])
#p_mit_cata_fadu = np.array([0.30313692, 0.95518913])
alphaR_fadu = 0.77807491 # 1/Gy # slice_model_RT_FaDu_5Gy_dr_4_1_4_log
betaR_fadu = 0.01190783

alphaR_fadu = 0.60712889 # 1/Gy # slice_model_RT_FaDu_5Gy_dr_5_3_2_log
betaR_fadu = 0.03620404

alphaR_fadu = 0.52112523 # 1/Gy # slice_model_RT_FaDu_5Gy_dr_8_1_2_log
betaR_fadu = 0.03569486

alphaR_fadu = 0.67930258 # 1/Gy # slice_model_RT_FaDu_5Gy_dr_10_3_1_log
betaR_fadu = 0.00453793

alphaR_fadu = 0.51496461 # 1/Gy # slice_model_RT_FaDu_5Gy_dr_11_3_2_log
betaR_fadu = 0.0320455

alphaR_fadu = 0.49613726 # 1/Gy # slice_model_RT_FaDu_5Gy_dr_12_3_1_log
betaR_fadu = 0.03580723

alphaR_fadu = 0.57582494 # 1/Gy # slice_model_RT_FaDu_5Gy_dr_13_1_2_log
betaR_fadu = 0.0309486

alphaR_fadu = 0.34993872 # 1/Gy # slice_model_RT_FaDu_5Gy_dr_14_1_2_log
betaR_fadu = 0.07529264

alphaR_fadu = 0.34998672 # 1/Gy # slice_model_RT_FaDu_5Gy_dr_14_1_3_log
betaR_fadu = 0.07939085

# see CheManLehSorLoeYuDubBauKesStaKun2021, Mokhir
#alphaR_fadu_c = 0.289838 # 1/Gy # control 39 C
#betaR_fadu_c = 0.01984298

alphaR_fadu_c = 0.19239465 # 1/Gy # control 44.5 C
betaR_fadu_c = 0.0689853

#alphaR_fadu_c = 0.27650055 # 1/Gy # control 40.5 C
#betaR_fadu_c = 0.06475292

plots = [
    {
        "logFile": "slice_model_RT_HCT_116_Bru2019_Fig5a_dr_12_3_log",
        "simulate" : True,
        "tmax" : np.array([3,21])*24*60*60,
        "p_mit_cata" : p_mit_cata_hct,
        "dose" : 10, # Gy
        "alphaR" : alphaR_hct, # 1/Gy
        "betaR" : betaR_hct,  # 1/Gy**2
        "gammaR" : 1.0, # unitless
        "max_iter_time": 60,
        #"v0" : 0.95e-11 * 1e15,
        "v0" : 10419.00774889, # slice_model_RT_HCT_116_Bru2019_Fig6_10gy_dr_12_1_1_log
        "rmax": 1.1e-3,
        "og_a": 22.1,
        "plate": "000",
        "cellline" : "HCT_116",
        "therapy" : "",
        "exp_name" : "Bru2019_Fig6_10gy",
        "sim_name" : "Bru2019_Fig6_10gy_sim",#"Bru2019_Fig6_10gy_sim",
        "sim_label" : "Brüningk et al., 2019",
        "sim_RT" : [0.5, 0.042],
        "d_offset" : 0,
        "4Plots" : True,
        "volumePlot" : False,
        "singlePlots" : False,
        "video": False,
    },
    {
        "logFile": "slice_model_RT_HCT_116_Bru2019_Fig5a_dr_12_3_log",
        "simulate" : True,
        "tmax" : np.array([3,21])*24*60*60,
        "p_mit_cata" : p_mit_cata_hct,
        "dose" : 5, # Gy
        "alphaR" : alphaR_hct, # 1/Gy
        "betaR" : betaR_hct,  # 1/Gy**2
        "gammaR" : 1.0, # unitless
        "max_iter_time": 60,
        #"v0" : 1.06e-11 * 1e15,
        "v0" : 10367.22048407, # slice_model_RT_HCT_116_Bru2019_Fig6_5gy_dr_12_1_3_log
        "rmax": 1.1e-3,
        "og_a": 22.1,
        "plate": "000",
        "cellline" : "HCT_116",
        "therapy" : "",
        "exp_name" : "Bru2019_Fig6_5gy",
        "sim_name" : "Bru2019_Fig6_5gy_sim",#"Bru2019_Fig6_5gy_sim",
        "sim_label" : "Brüningk et al., 2019",
        "sim_RT" : [0.5, 0.042],
        "d_offset" : 0,
        "4Plots" : True,
        "volumePlot" : False,
        "singlePlots" : False,
        "video": False,
    },
    {
        "logFile": "slice_model_RT_HCT_116_Bru2019_Fig5a_dr_12_3_log",
        "simulate" : True,
        "tmax" : np.array([3,21])*24*60*60,
        "p_mit_cata" : p_mit_cata_hct,
        "dose" : 2, # Gy
        "alphaR" : alphaR_hct, # 1/Gy
        "betaR" : betaR_hct,  # 1/Gy**2
        "gammaR" : 1.0, # unitless
        "max_iter_time": 60,
        #"v0" : 0.79e-11 * 1e15,
        "v0" : 11459.6659452, # slice_model_RT_HCT_116_Bru2019_Fig6_2gy_dr_12_1_1_log
        "rmax": 1.1e-3,
        "og_a": 22.1,
        "plate": "000",
        "cellline" : "HCT_116",
        "therapy" : "",
        "exp_name" : "Bru2019_Fig6_2gy",
        "sim_name" : "Bru2019_Fig6_2gy_sim",#"Bru2019_Fig6_2gy_sim",
        "sim_label" : "Brüningk et al., 2019",
        "sim_RT" : [0.5, 0.042],
        "d_offset" : 0,
        "4Plots" : True,
        "volumePlot" : False,
        "singlePlots" : False,
        "video": False,
    },
    {
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
    },
    {
        "logFile": "slice_model_RT_FaDu_dr_12_2_log",
        "simulate" : True,
        "tmax" : np.array([3,33])*24*60*60,
        "p_mit_cata" : p_mit_cata_fadu,
        "dose" : 7.5, # Gy
        "alphaR" : alphaR_fadu, # 1/Gy
        "betaR" : betaR_fadu,  # 1/Gy**2
        "gammaR" : 1.0, # unitless
        "max_iter_time": 60,
        "v0" : 2.73e-11 * 1e15,
        "rmax": 1.1e-3,
        "og_a": 10.64,
        "plate": "122",
        "cellline" : "FaDu",
        "therapy" : "",
        "exp_name" : "",
        "sim_name" : "",
        "sim_label" : "",
        "d_offset" : 0,
        "4Plots" : True,
        "volumePlot" : False,
        "singlePlots" : False,
        "video": False,
    },
    {
        "logFile": "slice_model_RT_FaDu_dr_12_2_log",
        "simulate" : True,
        "tmax" : np.array([3,21])*24*60*60,
        "p_mit_cata" : p_mit_cata_fadu,
        "dose" : 5, # Gy
        "alphaR" : alphaR_fadu, # 1/Gy
        "betaR" : betaR_fadu,  # 1/Gy**2
        "gammaR" : 1.0, # unitless
        "max_iter_time": 60,
        #"v0" : 3.07e-11 * 1e15,
        "v0" : 36774.28639716, # slice_model_RT_FaDu_5Gy_dr_14_1_3_log
        "rmax": 1.1e-3,
        "og_a": 10.64,
        "plate": "112",
        "cellline" : "FaDu",
        "therapy" : "",
        "exp_name" : "",
        "sim_name" : "",
        "sim_label" : "",
        "d_offset" : 0,
        "4Plots" : True,
        "volumePlot" : False,
        "singlePlots" : False,
        "video": False,
    },
    {
        "logFile": "slice_model_RT_FaDu_dr_12_2_log",
        "simulate" : True,
        "tmax" : np.array([3,14])*24*60*60,
        "p_mit_cata" : p_mit_cata_fadu,
        "dose" : 2.5, # Gy
        "alphaR" : alphaR_fadu, # 1/Gy
        "betaR" : betaR_fadu,  # 1/Gy**2
        "gammaR" : 1.0, # unitless
        "max_iter_time": 60,
        #"v0" : 3.07e-11 * 1e15,
        "v0" : 39064.92969578, # slice_model_RT_FaDu_2.5Gy_dr_14_1_2_log
        "rmax": 1.1e-3,
        "og_a": 10.64,
        "plate": "111",
        "cellline" : "FaDu",
        "therapy" : "",
        "exp_name" : "",
        "sim_name" : "",
        "sim_label" : "",
        "d_offset" : 0,
        "4Plots" : True,
        "volumePlot" : False,
        "singlePlots" : False,
        "video": False,
    },
    ######################## alpha/beta controll ###############################
    {
        "logFile": "slice_model_RT_FaDu_dr_12_2_log",
        "name_addon" : "controlled_",
        "simulate" : False,
        "tmax" : np.array([3,33])*24*60*60,
        "p_mit_cata" : p_mit_cata_fadu,
        "dose" : 20, # Gy
        "alphaR" : alphaR_fadu_c, # 1/Gy
        "betaR" : betaR_fadu_c,  # 1/Gy**2
        "gammaR" : 1.0, # unitless
        "max_iter_time": 60,
        #"v0" : 3.07e-11 * 1e15,
        "v0" : 29731.56352162, # slice_model_RT_FaDu_20Gy_dr_14_1_2_log
        "rmax": 1.1e-3,
        "og_a": 10.64,
        "plate": "110",
        "cellline" : "FaDu",
        "therapy" : "",
        "exp_name" : "",
        "sim_name" : "",
        "sim_label" : "",
        "d_offset" : 0,
        "4Plots" : True,
        "volumePlot" : False,
        "singlePlots" : False,
        "video": False,
    },
    {
        "logFile": "slice_model_RT_FaDu_dr_12_2_log",
        "name_addon" : "controlled_",
        "simulate" : False,
        "tmax" : np.array([3,21])*24*60*60,
        "p_mit_cata" : p_mit_cata_fadu,
        "dose" : 5, # Gy
        "alphaR" : alphaR_fadu_c, # 1/Gy
        "betaR" : betaR_fadu_c,  # 1/Gy**2
        "gammaR" : 1.0, # unitless
        "max_iter_time": 60,
         #"v0" : 3.07e-11 * 1e15,
        "v0" : 36774.28639716, # slice_model_RT_FaDu_5Gy_dr_14_1_3_log
        "rmax": 1.1e-3,
        "og_a": 10.64,
        "plate": "112",
        "cellline" : "FaDu",
        "therapy" : "",
        "exp_name" : "",
        "sim_name" : "",
        "sim_label" : "",
        "d_offset" : 0,
        "4Plots" : True,
        "volumePlot" : False,
        "singlePlots" : False,
        "video": False,
    },
    {
        "logFile": "slice_model_RT_FaDu_dr_12_2_log",
        "name_addon" : "controlled_",
        "simulate" : False,
        "tmax" : np.array([3,14])*24*60*60,
        "p_mit_cata" : p_mit_cata_fadu,
        "dose" : 2.5, # Gy
        "alphaR" : alphaR_fadu_c, # 1/Gy
        "betaR" : betaR_fadu_c,  # 1/Gy**2
        "gammaR" : 1.0, # unitless
        "max_iter_time": 60,
        #"v0" : 3.07e-11 * 1e15,
        "v0" : 39064.92969578, # slice_model_RT_FaDu_2.5Gy_dr_14_1_2_log
        "rmax": 1.1e-3,
        "og_a": 10.64,
        "plate": "111",
        "cellline" : "FaDu",
        "therapy" : "",
        "exp_name" : "",
        "sim_name" : "",
        "sim_label" : "",
        "d_offset" : 0,
        "4Plots" : True,
        "volumePlot" : False,
        "singlePlots" : False,
        "video": False,
    },
    ############################### p0 vary ####################################
    {
        "logFile": "slice_model_RT_HCT_116_Bru2019_Fig5a_dr_12_3_log",
        "name_addon" : "p0Vary80_",
        "simulate" : True,
        "tmax" : np.array([3,21])*24*60*60,
        "p_mit_cata" : p_mit_cata_hct,
        "dose" : 5, # Gy
        "alphaR" : alphaR_hct, # 1/Gy
        "betaR" : betaR_hct,  # 1/Gy**2
        "gammaR" : 1.0, # unitless
        "max_iter_time": 60,
        #"v0" : 1.06e-11 * 1e15,
        "v0" : 10367.22048407, # slice_model_RT_HCT_116_Bru2019_Fig6_5gy_dr_12_1_3_log
        "rmax": 1.1e-3,
        "og_a": 27.92,
        "plate": "000",
        "cellline" : "HCT_116",
        "therapy" : "",
        "exp_name" : "Bru2019_Fig6_5gy",
        "sim_name" : "Bru2019_Fig6_5gy_sim",
        "sim_label" : "Brüningk et al., 2019",
        "d_offset" : 0,
        "4Plots" : False,
        "volumePlot" : False,
        "singlePlots" : False,
        "video": False,
    },
    {
        "logFile": "slice_model_RT_HCT_116_Bru2019_Fig5a_dr_12_3_log",
        "name_addon" : "p0Vary120_",
        "simulate" : True,
        "tmax" : np.array([3,21])*24*60*60,
        "p_mit_cata" : p_mit_cata_hct,
        "dose" : 5, # Gy
        "alphaR" : alphaR_hct, # 1/Gy
        "betaR" : betaR_hct,  # 1/Gy**2
        "gammaR" : 1.0, # unitless
        "max_iter_time": 60,
        #"v0" : 1.06e-11 * 1e15,
        "v0" : 10367.22048407, # slice_model_RT_HCT_116_Bru2019_Fig6_5gy_dr_12_1_3_log
        "rmax": 1.1e-3,
        "og_a": 27.92,
        "plate": "000",
        "cellline" : "HCT_116",
        "therapy" : "",
        "exp_name" : "Bru2019_Fig6_5gy",
        "sim_name" : "Bru2019_Fig6_5gy_sim",
        "sim_label" : "Brüningk et al., 2019",
        "d_offset" : 0,
        "4Plots" : False,
        "volumePlot" : False,
        "singlePlots" : False,
        "video": False,
    },
    {
        "logFile": "slice_model_RT_HCT_116_Bru2019_Fig5a_dr_12_3_log",
        "name_addon" : "p0Vary80_",
        "simulate" : True,
        "tmax" : np.array([3,21])*24*60*60,
        "p_mit_cata" : p_mit_cata_hct,
        "dose" : 10, # Gy
        "alphaR" : alphaR_hct, # 1/Gy
        "betaR" : betaR_hct,  # 1/Gy**2
        "gammaR" : 1.0, # unitless
        "max_iter_time": 60,
        #"v0" : 0.79e-11 * 1e15,
        "v0" : 10419.00774889, # slice_model_RT_HCT_116_Bru2019_Fig6_10gy_dr_12_1_1_log
        "rmax": 1.1e-3,
        "og_a": 22.1,
        "plate": "000",
        "cellline" : "HCT_116",
        "therapy" : "",
        "exp_name" : "Bru2019_Fig6_10gy",
        "sim_name" : "Bru2019_Fig6_10gy_sim",
        "sim_label" : "Brüningk et al., 2019",
        "d_offset" : 0,
        "4Plots" : False,
        "volumePlot" : False,
        "singlePlots" : False,
        "video": False,
    },
    {
        "logFile": "slice_model_RT_HCT_116_Bru2019_Fig5a_dr_12_3_log",
        "name_addon" : "p0Vary120_",
        "simulate" : True,
        "tmax" : np.array([3,21])*24*60*60,
        "p_mit_cata" : p_mit_cata_hct,
        "dose" : 10, # Gy
        "alphaR" : alphaR_hct, # 1/Gy
        "betaR" : betaR_hct,  # 1/Gy**2
        "gammaR" : 1.0, # unitless
        "max_iter_time": 60,
        #"v0" : 0.79e-11 * 1e15,
        "v0" : 10419.00774889, # slice_model_RT_HCT_116_Bru2019_Fig6_10gy_dr_12_1_1_log
        "rmax": 1.1e-3,
        "og_a": 22.1,
        "plate": "000",
        "cellline" : "HCT_116",
        "therapy" : "",
        "exp_name" : "Bru2019_Fig6_10gy",
        "sim_name" : "Bru2019_Fig6_10gy_sim",
        "sim_label" : "Brüningk et al., 2019",
        "d_offset" : 0,
        "4Plots" : False,
        "volumePlot" : False,
        "singlePlots" : False,
        "video": False,
    },
    ############################### p0 vary V2 #################################
    {
        "logFile": "slice_model_RT_HCT_116_Bru2019_Fig5a_dr_12_3_p0_80_log",
        "name_addon" : "p0_80_",
        "simulate" : True,
        "tmax" : np.array([3,21])*24*60*60,
        "p_mit_cata" : p_mit_cata_hct_p0_80,
        "dose" : 10, # Gy
        "alphaR" : alphaR_hct, # 1/Gy
        "betaR" : betaR_hct,  # 1/Gy**2
        "gammaR" : 1.0, # unitless
        "max_iter_time": 60,
        "v0" : 10458.06388368, # slice_model_RT_HCT_116_Bru2019_Fig6_10gy_dr_12_1_1_p0_80_log
        "rmax": 1.1e-3,
        "og_a": 22.1,
        "plate": "000",
        "cellline" : "HCT_116",
        "therapy" : "",
        "exp_name" : "Bru2019_Fig6_10gy",
        "sim_name" : "Bru2019_Fig6_10gy_sim",#"Bru2019_Fig6_10gy_sim",
        "sim_label" : "Brüningk et al., 2019",
        "sim_RT" : [0.5, 0.042],
        "d_offset" : 0,
        "4Plots" : False,
        "volumePlot" : False,
        "singlePlots" : False,
        "video": False,
    },
    {
        "logFile": "slice_model_RT_HCT_116_Bru2019_Fig5a_dr_12_3_p0_120_log",
        "name_addon" : "p0_120_",
        "simulate" : True,
        "tmax" : np.array([3,21])*24*60*60,
        "p_mit_cata" : p_mit_cata_hct_p0_120,
        "dose" : 10, # Gy
        "alphaR" : alphaR_hct, # 1/Gy
        "betaR" : betaR_hct,  # 1/Gy**2
        "gammaR" : 1.0, # unitless
        "max_iter_time": 60,
        "v0" : 10428.91104297, # slice_model_RT_HCT_116_Bru2019_Fig6_10gy_dr_12_1_1_p0_120_log
        "rmax": 1.1e-3,
        "og_a": 22.1,
        "plate": "000",
        "cellline" : "HCT_116",
        "therapy" : "",
        "exp_name" : "Bru2019_Fig6_10gy",
        "sim_name" : "Bru2019_Fig6_10gy_sim",#"Bru2019_Fig6_10gy_sim",
        "sim_label" : "Brüningk et al., 2019",
        "sim_RT" : [0.5, 0.042],
        "d_offset" : 0,
        "4Plots" : False,
        "volumePlot" : False,
        "singlePlots" : False,
        "video": False,
    },
    {
        "logFile": "slice_model_RT_HCT_116_Bru2019_Fig5a_dr_12_3_p0_80_log",
        "name_addon" : "p0_80_",
        "simulate" : True,
        "tmax" : np.array([3,21])*24*60*60,
        "p_mit_cata" : p_mit_cata_hct_p0_80,
        "dose" : 5, # Gy
        "alphaR" : alphaR_hct, # 1/Gy
        "betaR" : betaR_hct,  # 1/Gy**2
        "gammaR" : 1.0, # unitless
        "max_iter_time": 60,
        "v0" : 10109.17558115, # slice_model_RT_HCT_116_Bru2019_Fig6_5gy_dr_12_1_1_p0_80_log
        "rmax": 1.1e-3,
        "og_a": 22.1,
        "plate": "000",
        "cellline" : "HCT_116",
        "therapy" : "",
        "exp_name" : "Bru2019_Fig6_5gy",
        "sim_name" : "Bru2019_Fig6_5gy_sim",#"Bru2019_Fig6_5gy_sim",
        "sim_label" : "Brüningk et al., 2019",
        "sim_RT" : [0.5, 0.042],
        "d_offset" : 0,
        "4Plots" : False,
        "volumePlot" : False,
        "singlePlots" : False,
        "video": False,
    },
    {
        "logFile": "slice_model_RT_HCT_116_Bru2019_Fig5a_dr_12_3_p0_120_log",
        "name_addon" : "p0_120_",
        "simulate" : True,
        "tmax" : np.array([3,21])*24*60*60,
        "p_mit_cata" : p_mit_cata_hct_p0_120,
        "dose" : 5, # Gy
        "alphaR" : alphaR_hct, # 1/Gy
        "betaR" : betaR_hct,  # 1/Gy**2
        "gammaR" : 1.0, # unitless
        "max_iter_time": 60,
        "v0" : 10835.17097366, # slice_model_RT_HCT_116_Bru2019_Fig6_5gy_dr_12_1_1_p0_120_log
        "rmax": 1.1e-3,
        "og_a": 22.1,
        "plate": "000",
        "cellline" : "HCT_116",
        "therapy" : "",
        "exp_name" : "Bru2019_Fig6_5gy",
        "sim_name" : "Bru2019_Fig6_5gy_sim",#"Bru2019_Fig6_5gy_sim",
        "sim_label" : "Brüningk et al., 2019",
        "sim_RT" : [0.5, 0.042],
        "d_offset" : 0,
        "4Plots" : False,
        "volumePlot" : False,
        "singlePlots" : False,
        "video": False,
    },
]

for data in plots:
    if data["simulate"]:
        print("[*] " + data["logFile"])

        values, lmfit_parameters = loadParametersFromLog(data["logFile"])

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
            if name_addon == "p0Vary80_":
                lmfit_parameters["p0"].value = 80
            elif name_addon == "p0Vary120_":
                lmfit_parameters["p0"].value = 120

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

        if not "alphaR" in lmfit_parameters or lmfit_parameters["alphaR"].value == 0:
            lmfit_parameters.add("alphaR",value=alphaR)
        else:
            lmfit_parameters["alphaR"].value=alphaR
        if not "betaR" in lmfit_parameters or lmfit_parameters["betaR"].value == 0:
            lmfit_parameters.add("betaR",value=betaR)
        else:
            lmfit_parameters["betaR"].value=betaR
        if not "gammaR" in lmfit_parameters or lmfit_parameters["gammaR"].value == 0:
            lmfit_parameters.add("gammaR",value=gammaR)
        else:
            lmfit_parameters["gammaR"].value=gammaR
        if not "p_mit_cata" in lmfit_parameters or lmfit_parameters["p_mit_cata"].value == 0:
            lmfit_parameters.add("p_mit_cata",value=0.5)

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

        x_exp_mean, x_exp_var, y_exp_mean, y_exp_var,_ = loadMultiExperiments(cellline, data["exp_name"], plate, data["therapy"])
        x_exp_mean += data["d_offset"]

        MyClass = getattr(models,model)
        instance = MyClass()

        c = setVolume_3(lmfit_parameters, t, model)

        print("    [+] simulating: " + str(data["dose"]) + " Gy")

        specialParameters = {"c":c, "filename":filename, "t_therapy":t_therapy, "d_therapy":d_therapy, "p_therapy":p_therapy}
        #prettyPrintLmfitParameters(lmfit_parameters)
        arg = instance.getValue(lmfit_parameters, t, specialParameters=specialParameters)
        sol = arg[0]
        metrics = instance.getMetrics(lmfit_parameters, sol)

        y_v0 = np.interp(t/24./3600.,x_exp_mean,y_exp_mean) # m**3
        y_r0 = ((y_v0 / np.pi)*(3./4.))**(1./3.)
        y_rn,_ = getVnFromV0(y_v0,np.zeros(len(y_v0)),exp_parameters,outputFormat="r")
        y_vn = 4. * np.pi * y_rn**3 / 3.
        y_rh,_ = getVhFromV0(y_v0,np.zeros(len(y_v0)),exp_parameters,outputFormat="r")
        y_vh = 4. * np.pi * y_rh**3 / 3.

        metrics_ref = [y_v0,y_vh,y_vn,y_r0,y_rh,y_rn]

        print("    [+] calculating R**2")

        x = deepcopy(x_exp_mean)
        x[x<0] = 0.1
        #t_therapy = np.array([x[-1]*24.*3600.])
        #arg = instance.getValue(lmfit_parameters, x*24.*3600., specialParameters={"c":c,"t_therapy":t_therapy, "d_therapy":d_therapy})
        #sol_r = arg[0]

        #metrics_r = instance.getMetrics(lmfit_parameters, sol_r)
        metrics_r = np.interp(x*24.*3600.,t,metrics[0])
        """
        mean = np.mean(y_exp_mean[1:])
        SSres = np.sum((metrics_r[0][1:] - y_exp_mean[1:])**2)
        SStot = np.sum((y_exp_mean[1:] - mean)**2)
        """
        mean = np.mean(y_exp_mean)
        SSres = np.sum((metrics_r - y_exp_mean)**2)
        SStot = np.sum((y_exp_mean - mean)**2)
        #"""
        r = 1. - SSres / SStot

        print("        [-] R² = " + str(r))

        print("    [+] O2 calculating")

        p = []
        p_raw = []
        p_hypox = []
        dr = lmfit_parameters["dr"].value / m

        for c in sol:
            c0 = np.split(c,3)[0]
            c1 = np.split(c,3)[1]
            c2 = np.split(c,3)[2]
            args_o2 = o2_models.sauerstoffdruck_analytical_4(lmfit_parameters,c0+c2,mode=2)
            p0 = args_o2[0]
            p0_raw = args_o2[3]
            p.append(p0)
            p_raw.append(p0_raw)
            p_hypox.append(args_o2[1])

        print("    [+] get original simulation")

        if data["sim_name"] != "":

            print("        [-] loading data from " + data["sim_name"])
            x_sim, y_sim = loadSimulations(data["sim_name"], data["exp_name"], cellline, plate, data["therapy"], datasource = datasource, smoothing=x_exp_mean)
            sim_data = [x_sim, y_sim, data["sim_label"]]

        elif data["sim_name"] == "" and "sim_RT" in data and data["sim_RT"] != "":

            print("        [-] resimulating")
            sim_parameters = deepcopy(lmfit_parameters)
            sim_parameters["alphaR"].value = data["sim_RT"][0]
            sim_parameters["betaR"].value = data["sim_RT"][1]
            c = setVolume_3(sim_parameters, t, model)
            specialParameters = {"c":c, "filename":filename, "t_therapy":t_therapy, "d_therapy":d_therapy, "p_therapy":p_therapy}
            sim_arg = instance.getValue(sim_parameters, t, specialParameters=specialParameters)
            sim_sol = sim_arg[0]
            sim_metrics = instance.getMetrics(sim_parameters, sim_sol)
            sim_data = [t/(24.*3600.), sim_metrics[0], data["sim_label"]]

        else:
            sim_data = None

        if sim_data != None:
            x = deepcopy(x_exp_mean)
            x[x<0] = 0.1
            metrics_r = np.interp(x*24.*3600.,sim_data[0]*24.*3600.,sim_data[1])
            mean = np.mean(y_exp_mean)
            SSres = np.sum((metrics_r - y_exp_mean)**2)
            SStot = np.sum((y_exp_mean - mean)**2)
            r = 1. - SSres / SStot
            print("        [-] R² = " + str(r))

        print("    [+] ploting")

        exp_data_v0 = [x_exp_mean, y_exp_mean, x_exp_var, y_exp_var, cellline]
        y_exp_mean_vn, y_exp_var_vn = getVnFromV0(y_exp_mean, y_exp_var,exp_parameters)
        exp_data_vn = [x_exp_mean, y_exp_mean_vn, x_exp_var, y_exp_var_vn, cellline]

        t_therapy = t_therapy/(24.*3600.)
        t_therapy = np.array([])
        d_therapy = np.array([])

        ########################################################################

        if data["4Plots"]:

            p = np.array(p_raw)
            p[p>140] = np.nan

            plt.figure(figsize=(11.0,8.25))
            plt.suptitle(cellline.replace("_"," "))

            plt.subplot(221)
            plt.title("(a)", loc='left')
            plotResults_6(t/(24.*3600.), metrics, "", p_raw, dr, filename,
                t_therapy, d_therapy, sol1 = None,
                exp_data_vn = exp_data_vn,
                exp_data_v0 = exp_data_v0,
                metrics2 = metrics_ref,
                sim_data = sim_data,
                mode = "r",
                figSave = False,
                predict = lmfit_parameters,
                subplotMode = True)
            
            plt.subplot(222)
            plt.title("(b)", loc='left')
            animate_6(0, sol, t/(24.*3600.), len(t), dr, p0 = p, metrics1 = metrics, metrics2 = metrics_ref, subplotMode = True)

            plt.subplot(223)
            plt.title("(c)", loc='left')
            plotResults_6(t/(24.*3600.), metrics, "", p_raw, dr, filename,
                t_therapy, d_therapy, sol1 = None,
                exp_data_vn = exp_data_vn,
                exp_data_v0 = exp_data_v0,
                metrics2 = metrics_ref,
                sim_data = sim_data,
                mode = "v",
                figSave = False,
                predict = lmfit_parameters,
                subplotMode = True)

            plt.subplot(224)
            plt.title("(d)", loc='left')
            animate_6(len(t)-1, sol, t/(24.*3600.), len(t), dr, p0 = p, metrics1 = metrics, metrics2 = metrics_ref, subplotMode = True)

            plt.tight_layout()

            if saveFig:
                #plt.savefig(getPath()["bilder"] + "fit_results_dr/" + filename + "_4plot.pdf",transparent=True)
                plt.savefig(getPath()["bilder"] + "fit_results_dr/" + filename + "_4plot.pdf",transparent=False)

        if data["singlePlots"]:
            plotResults_6(t/(24.*3600.), metrics, "", p_raw, dr, filename,
                t_therapy, d_therapy, sol1 = sol,
                exp_data_vn = exp_data_vn,
                exp_data_v0 = exp_data_v0,
                metrics2 = metrics_ref,
                sim_data = sim_data,
                mode = "r",
                figSave = saveFig,
                pathFigSave = getPath()["bilder"] + "fit_results_dr/")
        
        if data["volumePlot"]:
            plotResults_6(t/(24.*3600.), metrics, "", p_raw, dr, filename,
                t_therapy, d_therapy, sol1 = None,
                exp_data_vn = exp_data_vn,
                exp_data_v0 = exp_data_v0,
                metrics2 = metrics_ref,
                sim_data = sim_data,
                mode = "v",
                figSave = saveFig,
                pathFigSave = getPath()["bilder"] + "fit_results_dr/")
        
        if data["video"]:
            anim = animation.FuncAnimation(plt.figure(), animate_6, fargs=[sol, t/(24.*3600.), len(t), dr, p_raw, metrics, metrics_ref], interval=1, frames=len(t), repeat=True)
            #FFwriter = animation.FFMpegWriter(fps=10)
            #var = anim.save(getPath()["bilder"] + filename + ".mp4", writer = FFwriter)
            if saveFig:
                FFwriter = animation.FFMpegWriter(fps=10)
                var = anim.save(getPath()["bilder"] + "fit_results_dr/" + filename + ".mp4", writer = FFwriter)
                plt.close()
            else:
                plt.show()
        
        if not saveFig and (data["4Plots"] or data["singlePlots"] or data["volumePlot"]):
            plt.show()
