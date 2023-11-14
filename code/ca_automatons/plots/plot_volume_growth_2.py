"""
Content:
Measure timings and plot for each model to reach a certain volume/radius size.

Tags:
ca3d, fak, kappa, plot, volume, growth, measurment
"""


import sys
sys.path.insert(0,'../../../.')
sys.path.insert(0,'../.')
import matplotlib.pyplot as plt
from tools import *
import numpy as np
import time
from run_ca import default_parameters 
from cellularautomaton import main
from copy import deepcopy
import models
from ode_setVolume import *
from CONSTANTS import *
from sendNotifications import *

measure_timings = False
plotting = True
saveFig = True

if measure_timings:
    parameters, plot_para, string_para = default_parameters(5, "moor3moor4", "", plotFig = False, saveFig = False, cali_mode = False, run_1d = False, saveCA = True)
    ca_dt = parameters["ca_dt"] # s
    max_times = np.array([1,2,3,4,5,6,7,8,9,10]) * 24.*3600.
    max_times = np.floor(max_times/ca_dt) * ca_dt
    measurements_time = np.zeros((2,10,3))

    MyClass1 = getattr(models, "slice_model_RT")
    instance1 = MyClass1()
    MyClass2 = getattr(models, "Schichtenmodel")
    instance2 = MyClass2()

    for i in range(len(max_times)):
        print("[*] Round No. " + str(i+1))

        string_para_copy = deepcopy(string_para)
        string_para_copy["filename"] += "timings_FaDu_v1_" + str(i+1)
        parameters["tmax"] = max_times[i]
        parameters_1d = deepcopy(parameters)
        parameters_1d.pop("therapy_schedule")
        parameters_1d["rd"] = parameters_1d["ode_rd"]
        parameters_1d["dose"] = parameters_1d["ode_dose"]
        lmfit_parameters = dictToLmfitParameters(parameters_1d)
        
        ### CA ###
        print("   [+] CA")
        t1_ca = time.time()
        main(parameters, plot_para, string_para_copy)
        t2_ca = time.time()

        file = np.load(getPath()["ca3d"] + "experiments/" + string_para_copy["filename"] + ".npz", allow_pickle=True)
        ncell_list = file["data"][1][-1]
        dsim = (3. / 4. / np.pi * ncell_list)**(1. / 3.) * parameters["ca_dx"] * 1e6
        measurements_time[0][i][0] = t2_ca - t1_ca
        measurements_time[1][i][0] = dsim
        
        ### 1D RS ###
        print("   [+] 1D RS")
        t1_1DRS = time.time()
        t = np.linspace(0, max_times[i], parameters["ode_max_iter_time"])
        c = setVolume_3(lmfit_parameters, t, "slice_model_RT")
        t_therapy = np.array([max_times[i]]) # d
        d_therapy = np.array([0]) # Gy
        arg = instance1.getValue(lmfit_parameters, t, specialParameters={"c":c,"t_therapy":t_therapy, "d_therapy":d_therapy})
        t2_1DRS = time.time()
        
        sol = arg[0]
        metrics = instance1.getMetrics(lmfit_parameters, sol)
        measurements_time[0][i][1] = t2_1DRS - t1_1DRS
        measurements_time[1][i][1] = metrics[3][-1] * 1e6
        
        ### Grimes ###
        print("   [+] Grimes")
        #lmfit_parameters["delta"].value /= lmfit_parameters["epsilon"].value
        t1_grimes = time.time()
        t = np.linspace(0, max_times[i], parameters["ode_max_iter_time"])
        arg = instance2.getValue(lmfit_parameters, t)
        t2_grimes = time.time()
    
        metrics2 = instance2.getMetrics(lmfit_parameters, arg[0])
        measurements_time[0][i][2] = t2_grimes - t1_grimes
        measurements_time[1][i][2] = metrics2[3][-1] * 1e6

    np.save(getPath()["ca3d"] + "log/plot_volume_growth_2_v2.npy",measurements_time)

    sendEmail("plot_volume_growth_2 has finished", "plot_volume_growth_2 has finished\n"+str(measurements_time) , "mail@florian-franke.eu", [])

if plotting:

    measurements_time = np.load(getPath()["ca3d"] + "log/plot_volume_growth_2.npy")

    linewidth = 2.5
    fontsizeLabel = 15
    fontsizeTick = 12
    fontsizeLegend = 11

    ca_x = np.array([x[0] for x in measurements_time[1]])
    ca_y = np.array([x[0] for x in measurements_time[0]])
    rs_x = np.array([x[1] for x in measurements_time[1]])
    rs_y = np.array([x[1] for x in measurements_time[0]])
    gr_x = np.array([x[2] for x in measurements_time[1]])
    gr_y = np.array([x[2] for x in measurements_time[0]])

    gr_y = gr_y[gr_x<=ca_x[-1]]
    gr_x = gr_x[gr_x<=ca_x[-1]]

    ###### Ploting ######

    fig = plt.figure(figsize=(8.0,5.0))
    ax = fig.add_subplot(111)

    plt.plot(ca_x, ca_y, color=COLOR_PROLIFERATION, linewidth=linewidth, linestyle="-.", label="agent based")
    plt.plot(rs_x, rs_y, color=COLOR_PROLIFERATION, linewidth=linewidth, linestyle="solid", label="RS-model")
    plt.plot(gr_x, gr_y, color=COLOR_PROLIFERATION, linewidth=linewidth, linestyle="dashed", label="non-spatial model")

    plt.xlabel('radius $r_0$ [$\mu m$]',fontsize=fontsizeLabel)
    plt.ylabel('simulation time [$s$]',fontsize=fontsizeLabel)
    plt.legend(fontsize=fontsizeLegend)
    plt.setp(ax.get_xticklabels(), fontsize=fontsizeTick)
    plt.setp(ax.get_yticklabels(), fontsize=fontsizeTick)
    plt.yscale('log') 

    ###### Scaling Analysis ######

    
    coef2 = np.polyfit(rs_x, rs_y, 1)
    poly1d_fn2 = np.poly1d(coef2)
    plt.plot(rs_x[rs_x > 350], poly1d_fn2(rs_x[rs_x > 350]), color="black",linestyle="dotted")
    plt.text(400,3,r"$\sim 0.01 \cdot r$")
    print(coef2)
    
    coef1 = np.polyfit(gr_x, gr_y, 1)
    poly1d_fn1 = np.poly1d(coef1)
    plt.plot(gr_x[gr_x > 350], poly1d_fn1(gr_x[gr_x > 350]), color="black",linestyle="dotted")
    plt.text(400,0.02,r"$\sim 0.00005 \cdot r$")
    print(coef1)

    coef3 = np.polyfit(np.log(ca_x[ca_x>300]), ca_y[ca_x>300], 1)
    poly1d_fn1 = np.poly1d(coef3)
    plt.plot(ca_x[ca_x > 350], poly1d_fn1(np.log(ca_x[ca_x > 350])), color="black",linestyle="dotted")
    print(coef3)
    plt.text(400,1300,r"$\sim \exp(r)$")
    
    if not saveFig:
        plt.show()
    else:
        plt.savefig(getPath()["bilder"] + "plot_volume_growth_2" + ".pdf",transparent=False)