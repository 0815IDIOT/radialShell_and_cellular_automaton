"""
Content:
Verschiedene Funktionen die verschiedene Plots für Paper und Posters mit dem
CA erzeugen

Tags:
bilder, plots, figures, ca, 3D
"""

import sys
sys.path.insert(0,'../../.')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from copy import deepcopy
from constants import *
import models
from CONSTANTS import *
from tools import *
from data_loader import data_loader

def plot_volume_growth(parameters, conc_list):
    print("[*] plot_volume_growth")

    # [RealZeit, Simulationszeit, Volumen]
    #time_ca = [[1,0,0],[2,0,0],[3,0,0],[4,0,0],[5,0,0],[6,0,0],[7,0,0],[8,0,0]]
    time_ca = [[0,0,14363],
                [0.23,17.1,15588],
                [0.47,19.1,16942],
                [0.70, 24.40, 18619],
                [1.17,29.3,22315],
                [1.41,32.5,24024],
                [1.64, 33.3,26166],
                [1.88,39.4,28492],
                [2.11,42.6,30904],
                [2.35,54.2,33500],
                [2.58,61.9,36092],
                [2.81,68.1,38872],
                [3.05,73.8,42152],
                [3.28,75.2,45544],
                [3.52,86.6,48240],
                [3.75,103.3,51824],
                [3.99, 113.6,55816],
                [4.22, 109.9,59119],
                [5.16,137.8,76567],
                [6.01, 181.8, 97293],
                [7.04, 323.0,120769],
                [8.21, 357.5,154699]]
    time_model1 = []
    time_model2 = []

    linewidth = 3
    fontsizeLabel = 15
    fontsizeTick = 12
    fontsizeLegend = 11
    figSave = False

    model1 = "Schichtenmodel"
    model2 = "slice_model_FG"

    MyClass1 = getattr(models,model1)
    instance1 = MyClass1()
    MyClass2 = getattr(models,model2)
    instance2 = MyClass2()

    v0 = (3. / 4. / np.pi * time_ca[0][2])**(1. / 3.) * parameters["ca_dx"] * parameters["m"]
    dr = parameters["ode_dr"]
    m = parameters["m"]
    gridPoints = int(parameters["ode_gridPoints"])
    lmfit_parameters = dictToLmfitParameters(parameters)
    t = np.linspace(0, int(parameters["tmax"]), int(parameters["ode_max_iter_time"]))
    time_model1.append([0,0,v0/m * 1e6])
    time_model2.append([0,0,v0/m * 1e6])

    print("[+] Simulating " + model1)
    """
    lmfit_parameters["gamma"].value /= 7.
    for i in range(len(time_ca)-1):
        print("[" + str(time_ca[i+1][0]) + "]")
        t = np.linspace(0, int(time_ca[i+1][0]*24*60*60), int(parameters["ode_max_iter_time"]))
        t1_1 = time.time()
        metrics1 = instance1.getMetrics(lmfit_parameters, instance1.getValue(lmfit_parameters, t)[0])
        t1_2 = time.time()
        time_model1.append([time_ca[i+1][0], t1_2-t1_1, metrics1[3][-1] * 1e6])
    """
    time_model1 = [[0.00000000e+00, 0.00000000e+00, 1.96032857e+02],
        [2.30000000e-01, 5.54323196e-03, 2.00698683e+02],
        [4.70000000e-01, 6.22820854e-03, 2.05649014e+02],
        [7.00000000e-01, 6.51383400e-03, 2.10514720e+02],
        [1.17000000e+00, 1.13060474e-02, 2.20834238e+02],
        [1.41000000e+00, 6.53433800e-03, 2.26291564e+02],
        [1.64000000e+00, 4.45332527e-02, 2.31729238e+02],
        [1.88000000e+00, 4.07650471e-02, 2.37411449e+02],
        [2.11000000e+00, 4.50153351e-02, 2.42950300e+02],
        [2.35000000e+00, 4.88686562e-02, 2.48820981e+02],
        [2.58000000e+00, 4.71913815e-02, 2.54529199e+02],
        [2.81000000e+00, 4.35147285e-02, 2.60313141e+02],
        [3.05000000e+00, 4.56418991e-02, 2.66425705e+02],
        [3.28000000e+00, 4.58202362e-02, 2.72353836e+02],
        [3.52000000e+00, 3.96330357e-02, 2.78608794e+02],
        [3.75000000e+00, 4.11722660e-02, 2.84666548e+02],
        [3.99000000e+00, 4.55253124e-02, 2.91050617e+02],
        [4.22000000e+00, 4.23364639e-02, 2.97226277e+02],
        [5.16000000e+00, 3.79624367e-02, 3.23001532e+02],
        [6.01000000e+00, 5.55973053e-02, 3.46960900e+02],
        [7.04000000e+00, 4.58841324e-02, 3.76690985e+02],
        [8.21000000e+00, 4.86559868e-02, 4.11244297e+02]]

    print("[+] Simulating " + model2)
    """
    lmfit_parameters["gamma"].value *= 7.
    for i in range(len(time_ca)-1):
        print("[" + str(time_ca[i+1][0]) + "]")
        t = np.linspace(0, int(time_ca[i+1][0]*24*60*60), int(parameters["ode_max_iter_time"]))
        t2_1 = time.time()
        metrics2 = instance2.getMetrics(lmfit_parameters, instance2.getValue(lmfit_parameters, t, specialParameters={"c":conc_list[0]}))
        t2_2 = time.time()
        time_model2.append([time_ca[i+1][0], t2_2-t2_1, metrics2[3][-1] * 1e6])
    """
    time_model2 = [[0.00000000e+00, 0.00000000e+00, 1.96032857e+02],
        [2.30000000e-01, 5.92694283e-02, 1.93402257e+02],
        [4.70000000e-01, 8.02760124e-02, 1.97085051e+02],
        [7.00000000e-01, 9.79378223e-02, 2.01113265e+02],
        [1.17000000e+00, 1.20848656e-01, 2.10791201e+02],
        [1.41000000e+00, 1.54572487e-01, 2.16401880e+02],
        [1.64000000e+00, 2.72361994e-01, 2.22140921e+02],
        [1.88000000e+00, 3.58455181e-01, 2.28455497e+02],
        [2.11000000e+00, 3.93783808e-01, 2.34774436e+02],
        [2.35000000e+00, 4.86966372e-01, 2.41604204e+02],
        [2.58000000e+00, 6.42892599e-01, 2.48340417e+02],
        [2.81000000e+00, 6.27812624e-01, 2.55233697e+02],
        [3.05000000e+00, 7.77290821e-01, 2.62568859e+02],
        [3.28000000e+00, 7.71276474e-01, 2.69713285e+02],
        [3.52000000e+00, 8.89555931e-01, 2.77269108e+02],
        [3.75000000e+00, 1.00422454e+00, 2.84592795e+02],
        [3.99000000e+00, 1.10989738e+00, 2.92308764e+02],
        [4.22000000e+00, 1.04984832e+00, 2.99764400e+02],
        [5.16000000e+00, 1.27670765e+00, 3.30708035e+02],
        [6.01000000e+00, 1.48098469e+00, 3.59138197e+02],
        [7.04000000e+00, 1.80049658e+00, 3.93939196e+02],
        [8.21000000e+00, 2.00754428e+00, 4.33762532e+02]]

    fig = plt.figure(figsize=(8.0,5.0))
    ax = fig.add_subplot(111)

    time_ca = np.array(time_ca)
    time_ca[:,2] =  (3. / 4. / np.pi * time_ca[:,2])**(1. / 3.) * lmfit_parameters["ca_dx"] * 1e6
    time_model2 = np.array(time_model2)
    time_model1 = np.array(time_model1)

    #print(time_ca[:,2])
    #print(time_model2)
    #print(time_model1)

    time_model1[:,1] *= 1000
    time_model2[:,1] *= 100

    plt.plot(time_ca[:,2],time_ca[:,1], color=COLOR_PROLIFERATION, linewidth=linewidth, linestyle="solid", label="agent based (3D CA)")
    plt.plot(time_model2[:,2],time_model2[:,1], color=COLOR_PROLIFERATION, linewidth=linewidth, linestyle="dashed", label="radial Markov model (*100)")
    plt.plot(time_model1[:,2],time_model1[:,1], color=COLOR_PROLIFERATION, linewidth=linewidth, linestyle="-.", label="non spatial model (*1000)")
    #plt.plot(tsim, dsim, color=COLOR_PROLIFERATION, linewidth=linewidth, linestyle="solid", label="CA" + " (t=" + str(round(time_ca,2)) + "s)")
    #plt.plot(t/24./3600., metrics2[3]*1e6, linestyle="dashed", color=COLOR_PROLIFERATION, linewidth=linewidth, label='radial Markov model' + " (t=" + str(round(time_model2,2)) + "s)")
    #plt.plot(t/24./3600., metrics1[3]*1e6, linestyle="-.", color=COLOR_PROLIFERATION, linewidth=linewidth, label='non spatial model' + " (t=" + str(round(time_model1,2)) + "s)")

    #plt.title("computation time")
    #plt.xlim(tsim[1],tsim[-1]*1.1)
    plt.xlabel('radius [um]',fontsize=fontsizeLabel)
    plt.ylabel('simulation time [s]',fontsize=fontsizeLabel)
    plt.legend(fontsize=fontsizeLegend)
    plt.setp(ax.get_xticklabels(), fontsize=fontsizeTick)
    plt.setp(ax.get_yticklabels(), fontsize=fontsizeTick)

    if figSave:
        plt.savefig("../../../bilder/ca3d/plot_volume_growth_1.pdf",transparent=True)
    else:
        plt.show()

def plot_comparison_ca_1d(conc_list, sol, parameters, ncell_list, necrotic_list, metrics, t, filename, plot_para):

    linewidth = 3
    fontsizeLabel = 15
    fontsizeTick = 12
    fontsizeLegend = 11
    figSave = parameters["saveFig"]
    figPlot = parameters["plotFig"]
    text = filename

    path = getPath()["ca3d"]
    plotExp = plot_para["exp_name"]
    cellline = plot_para["cellline"]
    plate = plot_para["plate"]
    datasource = plot_para["datasource"]

    if not (cellline == "" and plotExp == ""):
        data_loader_obj = data_loader()
        m = parameters["m"]
        lmfit_parameters = dictToLmfitParameters(parameters)
        exp_parameters = deepcopy(lmfit_parameters)
        exp_parameters["a"].value = exp_parameters["og_a"].value

        x_exp_mean, x_exp_var, y_exp_mean, y_exp_var,_ = loadMultiExperiments(cellline, plotExp, plate, datasource=datasource)

        y_exp_mean_vn,y_exp_var_vn = getVnFromV0(y_exp_mean, y_exp_var, exp_parameters)
        y_exp_mean_rn,y_exp_var_rn = convertVolumnRadius(y_exp_mean_vn,y_exp_var_vn,inputFormat="v")
        y_exp_mean_r0,y_exp_var_r0 = convertVolumnRadius(y_exp_mean,y_exp_var,inputFormat="v")

        y_exp_mean_rn *= 1e6
        y_exp_var_rn *= (1e6)**2
        y_exp_mean_r0 *= 1e6
        y_exp_var_r0 *= (1e6)**2

    ############################################################################
    #"""
    fig = plt.figure(figsize=(8.0,5.0))
    ax = fig.add_subplot(111)

    idx = -1
    #print(len(conc_list))
    res = np.split(conc_list[idx],3)
    #print(len(res[0]))
    c = np.split(sol[idx],3)
    dr = parameters["ode_dr"] / parameters["m"]
    radius = np.arange(0,len(res[0]),1) * dr * 1e6

    plt.hlines(1,0, radius[-1], color="gray", linestyle="dotted", linewidth=linewidth)

    plt.plot(radius, res[0], color=COLOR_PROLIFERATION, label="CA - c0", linewidth=linewidth)
    plt.plot(radius, res[1], color=COLOR_ANOXIC, label="CA - c1", linewidth=linewidth)
    plt.plot(radius, res[2], color=COLOR_MITOTIC_CATASTROPHY, label="CA - c2", linewidth=linewidth)
    #plt.plot(radius, res[0]+res[1], "cyan", linestyle="dashed", label="sum - CA")

    plt.plot(radius, c[0], color=COLOR_PROLIFERATION,linestyle="dashed",alpha=0.5, label="1D - c0", linewidth=linewidth)
    plt.plot(radius, c[1], color=COLOR_ANOXIC,linestyle="dashed",alpha=0.5, label="1D - c1", linewidth=linewidth)
    plt.plot(radius, c[2], color=COLOR_MITOTIC_CATASTROPHY,linestyle="dashed",alpha=0.5, label="1D - c2", linewidth=linewidth)
    #plt.plot(radius, c[0]+c[1], "cyan", linestyle="dashed", label="sum - 1D")

    plt.xlabel("radius [um]",fontsize=fontsizeLabel)
    plt.ylabel("concentration",fontsize=fontsizeLabel)
    plt.legend(fontsize=fontsizeLegend)
    plt.setp(ax.get_xticklabels(), fontsize=fontsizeTick)
    plt.setp(ax.get_yticklabels(), fontsize=fontsizeTick)

    if figSave:
        plt.savefig(path + text + "_end.pdf",transparent=True)
    #"""
    ############################################################################
    #"""
    fig = plt.figure(figsize=(8.0,5.0))
    ax = fig.add_subplot(111)

    idx = 0
    linewidth = 2

    res = np.split(conc_list[idx],3)
    c = np.split(sol[idx],3)
    dr = parameters["ode_dr"] / parameters["m"]
    radius = np.arange(0,len(res[0]),1) * dr * 1e6

    plt.hlines(1,0, radius[-1], color="gray", linestyle="dotted", linewidth=linewidth)

    plt.plot(radius, res[0], color=COLOR_PROLIFERATION, label="CA - c0", linewidth=linewidth)
    plt.plot(radius, res[1], color=COLOR_ANOXIC, label="CA - c1", linewidth=linewidth)
    plt.plot(radius, res[2], color=COLOR_MITOTIC_CATASTROPHY, label="CA - c2", linewidth=linewidth)
    #plt.plot(radius, res[0]+res[1], "cyan", linestyle="dashed", label="sum - CA")

    plt.plot(radius, c[0], color=COLOR_PROLIFERATION,linestyle="dashed",alpha=0.5, label="1D - c0", linewidth=linewidth)
    plt.plot(radius, c[1], color=COLOR_ANOXIC,linestyle="dashed",alpha=0.5, label="1D - c1", linewidth=linewidth)
    plt.plot(radius, c[2], color=COLOR_MITOTIC_CATASTROPHY,linestyle="dashed",alpha=0.5, label="1D - c2", linewidth=linewidth)
    #plt.plot(radius, c[0]+c[1], "cyan", linestyle="dashed", label="sum - 1D")

    plt.xlabel("radius [um]",fontsize=fontsizeLabel)
    plt.ylabel("concentration",fontsize=fontsizeLabel)
    plt.legend(fontsize=fontsizeLegend)
    plt.setp(ax.get_xticklabels(), fontsize=fontsizeTick)
    plt.setp(ax.get_yticklabels(), fontsize=fontsizeTick)

    if figSave:
        plt.savefig(path + text + "_start.pdf",transparent=True)
    #"""
    ############################################################################
    #"""

    fig = plt.figure(figsize=(8.0,5.0))
    ax = fig.add_subplot(111)

    dsim = (3. / 4. / np.pi * ncell_list)**(1. / 3.) * parameters["ca_dx"] * 1e6
    tsim = parameters["ca_dt"] * np.arange(len(ncell_list)) / 3600.0 / 24
    necrotic_list = np.array(necrotic_list)
    necrotic_radius = (3. / 4. / np.pi * necrotic_list)**(1. / 3.) * parameters["ca_dx"] * 1e6
    #t = np.linspace(0, parameters["tmax"], int(parameters["ode_max_iter_time"]))

    plt.plot(tsim, dsim, color=COLOR_PROLIFERATION, label="CA - r0", linewidth=linewidth)
    plt.plot(tsim, necrotic_radius, '-', color=COLOR_ANOXIC, label="CA - rn", linewidth=linewidth)

    plt.plot(t/24./3600., metrics[3]*1e6, linestyle="dashed", color=COLOR_PROLIFERATION, label='1D - r0', alpha=0.5, linewidth=linewidth)
    plt.plot(t/24./3600., metrics[5]*1e6, linestyle="dashed", color=COLOR_ANOXIC, label='1D - rn', alpha=0.5, linewidth=linewidth)

    if not (cellline == "" and plotExp == ""):
        #"""
        plt.errorbar(x_exp_mean, y_exp_mean_r0, np.sqrt(y_exp_var_r0), np.sqrt(x_exp_var),color=COLOR_ERRORBAR,linewidth=3*0.5, linestyle="none", capsize=5, label="exp. - r0", zorder=1)
        #plt.errorbar(x_exp_mean, y_exp_mean_rn, np.sqrt(y_exp_var_rn), np.sqrt(x_exp_var),color=COLOR_ERRORBAR_PREDICT,linewidth=3*0.5, linestyle="none", capsize=5, label="exp. - rn")
        plt.scatter(x_exp_mean[y_exp_mean_rn==0], y_exp_mean_rn[y_exp_mean_rn==0], color=COLOR_ERRORBAR_PREDICT, marker="|")
        plt.errorbar(x_exp_mean[y_exp_mean_rn!=0] , y_exp_mean_rn[y_exp_mean_rn!=0] , np.sqrt(y_exp_var_rn[y_exp_mean_rn!=0] ), np.sqrt(x_exp_var[y_exp_mean_rn!=0] ), color=COLOR_ERRORBAR_PREDICT, linewidth=3*0.5, linestyle="none", capsize=5, label="exp. - rn", zorder=1)
        #"""
        #pass

    #plt.plot(t/24./3600., t * parameters["gamma"] * parameters["dr"] * 1e1 + metrics[3][0]*1e6, linestyle="dashed", color="black")
    #plt.plot(t[t/24./3600.<2.]/24./3600., metrics[3][0] * 1e6 * np.exp(parameters["gamma"] * t[t/24./3600.<2.] / 3.), linestyle="dashed", color="gray")
    #plt.plot(t/24./3600.,t * parameters["ca_gamma"] * parameters["ca_dx"] * 1e6 * 1.5*0.44109985782089994 + metrics[3][0]*1e6, linestyle="dashed", color="gray")

    plt.xlabel('t [d]',fontsize=fontsizeLabel)
    plt.ylabel('radius [um]',fontsize=fontsizeLabel)
    plt.legend(fontsize=fontsizeLegend)
    plt.setp(ax.get_xticklabels(), fontsize=fontsizeTick)
    plt.setp(ax.get_yticklabels(), fontsize=fontsizeTick)

    if figSave:
        plt.savefig(path + text + "_radius.pdf",transparent=True)
    #"""
    ############################################################################

    if figPlot:
        plt.show()
