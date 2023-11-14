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

plots = [
    {
        "logFile": "slice_model_RT_HCT_116_dr_12_5_log", #"slice_model_RT_HCT_116_dr_11_1_log",#"slice_model_RT_HCT_116_dr_10_2_log", #"", #"slice_model_FG_HCT_116_dr_5_1_log",
        "simulate" : True,
        "tmax" : 17*24*60*60,
        "max_iter_time": 60,
        "rmax": 1.1e-3,
        "og_a": 27.92,
        "plate": "000",
        "cellline" : "HCT_116",
        "exp_name" : "Gri2016_Fig4a_exp",
        "sim_name" : "Gri2016_Fig4a_grun",
        "sim_label" : "Grimes et al., 2016",
        "d_offset" : 0,
        "4Plots" : True,
        "volumePlot" : False,
        "singlePlots" : False,
        "video": False,
    },
    {
        "logFile": "slice_model_RT_HCT_116_Bru2019_Fig5a_dr_12_3_log", #"slice_model_RT_HCT_116_Bru2019_Fig5a_dr_11_2_log",
        "simulate" : True,
        "tmax" : 21*24*60*60,
        "max_iter_time": 60,
        "rmax": 1.1e-3,
        "og_a": 22.1,
        "plate": "000",
        "cellline" : "HCT_116",
        "exp_name" : "Bru2019_Fig5a",
        "sim_name" : "Bru2019_Fig5a_sim",
        "sim_label" : "Brüningk et al., 2019",
        "d_offset" : 0,
        "4Plots" : True,
        "volumePlot" : False,
        "singlePlots" : False,
        "video": True,
    },
    {
        "logFile": "slice_model_RT_MDA_MB_468_dr_12_1_log", #"slice_model_RT_MDA_MB_468_dr_11_4_log",
        "simulate" : True,
        "tmax" : 12*24*60*60,
        "max_iter_time": 60,
        "rmax": 1.1e-3,
        "og_a": 18.09,
        "plate": "000",
        "cellline" : "MDA_MB_468",
        "exp_name" : "Gri2016_Fig4c_exp",
        "sim_name" : "Gri2016_Fig4c_grun",
        "sim_label" : "Grimes et al., 2016",
        "d_offset" : 0,
        "4Plots" : True,
        "volumePlot" : False,
        "singlePlots" : False,
        "video": False,
    },
    {
        "logFile": "slice_model_RT_LS_174T_dr_12_2_log", #"slice_model_RT_LS_174T_dr_11_2_log",
        "simulate" : True,
        "tmax" : 7*24*60*60,
        "max_iter_time": 60,
        "rmax": 1.1e-3,
        "og_a": 20.61,
        "plate": "000",
        "cellline" : "LS_174T",
        "exp_name" : "Gri2016_Fig4b_exp",
        "sim_name" : "Gri2016_Fig4b_grun",
        "sim_label" : "Grimes et al., 2016",
        "d_offset" : 0,
        "4Plots" : True,
        "volumePlot" : False,
        "singlePlots" : False,
        "video": False,
    },
    {
        "logFile": "slice_model_RT_SCC_25_dr_12_1_log",#"slice_model_RT_SCC_25_dr_11_1_log",
        "simulate" : True,
        "tmax" : 17*24*60*60,
        "max_iter_time": 60,
        "rmax": 1.1e-3,
        "og_a": 11.21,
        "plate": "000",
        "cellline" : "SCC_25",
        "exp_name" : "Gri2016_Fig4d_exp",
        "sim_name" : "Gri2016_Fig4d_rot",
        "sim_label" : "Grimes et al., 2016",
        "d_offset" : 0,
        "4Plots" : True,
        "volumePlot" : False,
        "singlePlots" : False,
        "video": False,
    },
    {
        "logFile": "slice_model_RT_FaDu_dr_12_2_log",#"slice_model_RT_FaDu_dr_11_1_log",
        "simulate" : True,
        "tmax" : 14*24*60*60,
        "max_iter_time": 60,
        "rmax": 1.1e-3,
        "og_a": 10.64,
        "plate": "",
        "cellline" : "FaDu",
        "datasource" : "source2",
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
        "logFile": "slice_model_RT_MDA_MB_468_dr_12_1_log",
        "name_addon" : "p0Vary80_",
        "simulate" : True,
        "tmax" : 12*24*60*60,
        "max_iter_time": 60,
        "rmax": 1.1e-3,
        "og_a": 18.09,
        "plate": "000",
        "cellline" : "MDA_MB_468",
        "exp_name" : "Gri2016_Fig4c_exp",
        "sim_name" : "Gri2016_Fig4c_grun",
        "sim_label" : "Grimes et al., 2016",
        "d_offset" : 0,
        "4Plots" : False,
        "volumePlot" : False,
        "singlePlots" : False,
        "video": False,
    },
    {
        "logFile": "slice_model_RT_MDA_MB_468_dr_12_1_log",
        "name_addon" : "p0Vary120_",
        "simulate" : True,
        "tmax" : 12*24*60*60,
        "max_iter_time": 60,
        "rmax": 1.1e-3,
        "og_a": 18.09,
        "plate": "000",
        "cellline" : "MDA_MB_468",
        "exp_name" : "Gri2016_Fig4c_exp",
        "sim_name" : "Gri2016_Fig4c_grun",
        "sim_label" : "Grimes et al., 2016",
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
        "tmax" : 21*24*60*60,
        "max_iter_time": 60,
        "rmax": 1.1e-3,
        "og_a": 22.1,
        "plate": "000",
        "cellline" : "HCT_116",
        "exp_name" : "Bru2019_Fig5a",
        "sim_name" : "",
        "sim_label" : "",
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
        "tmax" : 21*24*60*60,
        "max_iter_time": 60,
        "rmax": 1.1e-3,
        "og_a": 22.1,
        "plate": "000",
        "cellline" : "HCT_116",
        "exp_name" : "Bru2019_Fig5a",
        "sim_name" : "",
        "sim_label" : "",
        "d_offset" : 0,
        "4Plots" : False,
        "volumePlot" : False,
        "singlePlots" : False,
        "video": False,
    },
    ############################### p0 vary V2 #################################
    {
        "logFile": "slice_model_RT_MDA_MB_468_dr_12_1_p0_80_log",
        "name_addon" : "p0_80_",
        "simulate" : True,
        "tmax" : 12*24*60*60,
        "max_iter_time": 60,
        "rmax": 1.1e-3,
        "og_a": 18.09,
        "plate": "000",
        "cellline" : "MDA_MB_468",
        "exp_name" : "Gri2016_Fig4c_exp",
        "sim_name" : "Gri2016_Fig4c_grun",
        "sim_label" : "Grimes et al., 2016",
        "d_offset" : 0,
        "4Plots" : False,
        "volumePlot" : False,
        "singlePlots" : False,
        "video": False,
    },
    {
        "logFile": "slice_model_RT_MDA_MB_468_dr_12_1_p0_120_log",
        "name_addon" : "p0_120_",
        "simulate" : True,
        "tmax" : 12*24*60*60,
        "max_iter_time": 60,
        "rmax": 1.1e-3,
        "og_a": 18.09,
        "plate": "000",
        "cellline" : "MDA_MB_468",
        "exp_name" : "Gri2016_Fig4c_exp",
        "sim_name" : "Gri2016_Fig4c_grun",
        "sim_label" : "Grimes et al., 2016",
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
        "tmax" : 21*24*60*60,
        "max_iter_time": 60,
        "rmax": 1.1e-3,
        "og_a": 22.1,
        "plate": "000",
        "cellline" : "HCT_116",
        "exp_name" : "Bru2019_Fig5a",
        "sim_name" : "",
        "sim_label" : "",
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
        "tmax" : 21*24*60*60,
        "max_iter_time": 60,
        "rmax": 1.1e-3,
        "og_a": 22.1,
        "plate": "000",
        "cellline" : "HCT_116",
        "exp_name" : "Bru2019_Fig5a",
        "sim_name" : "",
        "sim_label" : "",
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

        #prettyPrintLmfitParameters(lmfit_parameters)
        #lmfit_parameters["llambda"].value /= 10.
        """
        lmfit_parameters["v0"].value *= 2.
        lmfit_parameters["epsilon"].value *= 0.
        lmfit_parameters["delta"].value *= 0.
        lmfit_parameters["gamma"].value *= 0.5
        lmfit_parameters["tmax"].value = 6.*24.*3600.
        """
        #lmfit_parameters["llambda"].value *= 3.0
        m = lmfit_parameters["m"].value
        v0 = lmfit_parameters["v0"].value
        rmax = data["rmax"] * m
        dr = lmfit_parameters["dr"].value
        max_iter_time = data["max_iter_time"]
        tmax = data["tmax"]
        cellline = data["cellline"]
        plate = data["plate"]
        model = "slice_model_RT"
        datasource = "" if not "datasource" in data else data["datasource"]
        therapy = "" if not "therapy" in data else data["therapy"]
        name_addon = "" if not "name_addon" in data else data["name_addon"]
        if name_addon != "":
            print("    [-] " + name_addon)
            if name_addon == "p0Vary80_":
                lmfit_parameters["p0"].value = 80
            elif name_addon == "p0Vary120_":
                lmfit_parameters["p0"].value = 120
        #model = "slice_model_FG"
        #filename = "plot_log_" + data["logFile"]
        filename = "fit_result_" + name_addon + cellline + "_" + data["exp_name"].split("_")[0]

        t_therapy = np.array([tmax]) # d
        d_therapy = np.array([0]) # Gy
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

        gridPoints = int(round(rmax/dr,2))
        t = np.linspace(0, tmax, max_iter_time)

        exp_parameters = deepcopy(lmfit_parameters)
        exp_parameters["a"].value = data["og_a"]

        x_exp_mean, x_exp_var, y_exp_mean, y_exp_var,_ = loadMultiExperiments(cellline, data["exp_name"], plate, therapy, datasource = datasource)
        x_exp_mean += data["d_offset"]

        y_v0 = np.interp(t/24./3600., x_exp_mean, y_exp_mean) # m**3
        y_r0 = ((y_v0 / np.pi)*(3./4.))**(1./3.)
        y_rn,_ = getVnFromV0(y_v0, np.zeros(len(y_v0)), exp_parameters, outputFormat="r")
        y_vn = 4. * np.pi * y_rn**3 / 3.
        y_rh,_ = getVhFromV0(y_v0, np.zeros(len(y_v0)), exp_parameters, outputFormat="r")
        y_vh = 4. * np.pi * y_rh**3 / 3.

        metrics_ref = [y_v0,y_vh,y_vn,y_r0,y_rh,y_rn]

        MyClass = getattr(models,model)
        instance = MyClass()

        c = setVolume_3(lmfit_parameters, t, model)

        print("    [+] simulating")

        arg = instance.getValue(lmfit_parameters, t, specialParameters={"c":c,"t_therapy":t_therapy, "d_therapy":d_therapy})
        sol = arg[0]
        metrics = instance.getMetrics(lmfit_parameters, sol)

        np.save(getPath()["datadump"] + data["logFile"][:-4], sol)

        print("    [+] calculating R**2")
        #"""
        x = deepcopy(x_exp_mean)
        x[x<0] = 0.1
        t_therapy = np.array([x[-1]*24.*3600.])
        arg = instance.getValue(lmfit_parameters, x*24.*3600., specialParameters={"c":c,"t_therapy":t_therapy, "d_therapy":d_therapy})
        sol_r = arg[0]

        metrics_r = instance.getMetrics(lmfit_parameters, sol_r)
        #"""
        """
        mean = np.mean(y_exp_mean[1:])
        SSres = np.sum((metrics_r[0][1:] - y_exp_mean[1:])**2)
        SStot = np.sum((y_exp_mean[1:] - mean)**2)
        """
        #"""
        # use this ->
        mean = np.mean(y_exp_mean)
        SSres = np.sum((metrics_r[0] - y_exp_mean)**2)
        SStot = np.sum((y_exp_mean - mean)**2)
        #"""
        #"""
        r = 1. - SSres / SStot

        gamma = (np.log(2) / lmfit_parameters["gamma"].value) / 3600.
        print("        [-] doubling time = " + str(round(gamma,2)) + " [h]")
        print("        [-] gamma = " + str(round(lmfit_parameters["gamma"].value*3600.,2)) + " [1/h]")
        dr_fak = lmfit_parameters["dr"].value / lmfit_parameters["og_dr"].value
        if lmfit_parameters["og_dr"].value != 1.6:
            dr_fak = lmfit_parameters["dr"].value / 1.6
            print("        [-] dr/dr* = " + str(round(dr_fak,2)) + " (Recalculated)")
        else:
            print("        [-] dr/dr* = " + str(round(dr_fak,2)))
        print("        [-] lambda = " + "{:.2e}".format(lmfit_parameters["llambda"].value * 3600.) + " [1/h]")
        llambda = lmfit_parameters["llambda"].value * dr_fak
        print("        [-] lambda = " + "{:.2e}".format(llambda * 3600.) + " [dr*/h]")
        llambda = lmfit_parameters["llambda"].value * lmfit_parameters["dr"].value / lmfit_parameters["m"].value * 1e6
        #print("        [-] lambda = " + "{:.2e}".format(llambda * 3600.) + " [um/h]")
        print("        [-] lambda = " + str(round(llambda * 3600.,1)) + " [um/h]")
        llambda = lmfit_parameters["llambda"].value * lmfit_parameters["dr"].value / lmfit_parameters["m"].value * 1e6
        print("        [-] old lambda = " + "{:.2e}".format(llambda) + " [um/s]")
        print("        [-] epsilon = " + "{:.2e}".format(lmfit_parameters["epsilon"].value * 3600.) + " [1/h]")
        print("        [-] delta = " + "{:.2e}".format(lmfit_parameters["delta"].value * 3600.) + " [1/h]")
        print("        [-] a = " + str(round(lmfit_parameters["a"].value,2)) + " [mmHg/s]")
        print("        [-] R² = " + str(r))

        val1 = str(round(gamma,2))
        val2 = str(round(lmfit_parameters["a"].value,2))
        val3 = "{:.2e}".format(lmfit_parameters["epsilon"].value * 3600.).replace("e-0","e-")
        val4 = "{:.2e}".format(lmfit_parameters["delta"].value * 3600.).replace("e-0","e-")
        val5 = str(round(dr_fak,2))
        val6 = "{:.2e}".format(lmfit_parameters["llambda"].value * dr_fak * 3600.).replace("e-0","e-")
        #val7 = "{:.2e}".format(lmfit_parameters["llambda"].value * 3600. * lmfit_parameters["dr"].value / lmfit_parameters["m"].value * 1e6).replace("e-0","e-")
        val7 = str(round(lmfit_parameters["llambda"].value * 3600. * lmfit_parameters["dr"].value / lmfit_parameters["m"].value * 1e6,1))
        val8 = str(round(r,4))
        #print("    [+] LaTex: $" + val1 + "$ & $" + val2 + "$ & $" + val3 + "$ & $" + val4 + "$ & $" + val5 + "$ & $" + val6 + "$ & $" + val7 + "$ & $" + val8 + "$")
        print("    [+] LaTex: $" + val1 + "$ & $" + val2 + "$ & $" + val3 + "$ & $" + val4 + "$ & $" + val5 + "$ & $" + val7 + "$ & $" + val8 + "$")
        #"""
        print("    [+] O2 calculating")

        p = []
        p_raw = []
        dr = lmfit_parameters["dr"].value / m

        for c in sol:
            c0 = np.split(c,3)[0]
            c1 = np.split(c,3)[1]
            c2 = np.split(c,3)[2]
            c0 += c2
            args_o2 = o2_models.sauerstoffdruck_analytical(params=lmfit_parameters, c0=c0, mode=2)
            p0 = args_o2[0]
            p0_raw = args_o2[3]
            p.append(p0)
            p_raw.append(p0_raw)

        print("    [+] get original simulation")

        if data["sim_name"] != "":
            x_sim, y_sim = loadSimulations(data["sim_name"], data["exp_name"], cellline, plate, therapy, datasource = datasource, smoothing=x_exp_mean)
            sim_data = [x_sim, y_sim, data["sim_label"]]
        else:
            sim_data = None

        print("    [+] ploting")

        exp_data_v0 = [x_exp_mean, y_exp_mean, x_exp_var, y_exp_var, cellline]
        y_exp_mean_vn, y_exp_var_vn = getVnFromV0(y_exp_mean, y_exp_var, exp_parameters)
        exp_data_vn = [x_exp_mean, y_exp_mean_vn, x_exp_var, y_exp_var_vn, cellline]

        ########################################################################

        if data["4Plots"]:

            p = np.array(p_raw)
            p[p>140] = np.nan

            plt.figure(figsize=(11.0,8.25))
            #plt.figure(figsize=(5.8, 3.94))
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
                pathFigSave = getPath()["bilder"] + "fit_results_dr/",
                predict = lmfit_parameters)
        
        if data["volumePlot"]:
            plotResults_6(t/(24.*3600.), metrics, "", p_raw, dr, filename,
                t_therapy, d_therapy, sol1 = None,
                exp_data_vn = exp_data_vn,
                exp_data_v0 = exp_data_v0,
                metrics2 = metrics_ref,
                sim_data = sim_data,
                mode = "v",
                figSave = saveFig,
                pathFigSave = getPath()["bilder"] + "fit_results_dr/",
                predict = lmfit_parameters)

        if data["video"]:
            p = np.array(p_raw)
            p[p>140] = np.nan

            anim = animation.FuncAnimation(plt.figure(), animate_6, fargs=[sol, t/(24.*3600.), len(t), dr, p, None], interval=1, frames=len(t), repeat=True)
            if saveFig:
                FFwriter = animation.FFMpegWriter(fps=10)
                var = anim.save(getPath()["bilder"] + "fit_results_dr/" + filename + ".mp4", writer = FFwriter)
                plt.close()
            else:
                plt.show()
        
        if not saveFig and (data["4Plots"] or data["singlePlots"] or data["volumePlot"]):
            plt.show()
            #pass
