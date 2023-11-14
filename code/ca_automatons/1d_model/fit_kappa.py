"""
Content:
Fit the kappa for differen ca3d simulations and neighborhoods.

Tags:
ca3d, fak, kappa, 1d, shell
"""

import sys
sys.path.insert(0,'../../../.')
from tools import *
import numpy as np
import models
from ode_setVolume import *
import time
import matplotlib.pyplot as plt
from copy import deepcopy
import time
from sendNotifications import *
from ode_plots import *
import os

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
    tsim = parameters["ca_dt"] * np.arange(len(ncell_list)) / 3600.0 / 24.
    y = np.interp(t/24./3600., tsim, dsim)

    return t, y

################################################################################

data = {}
#neighborhoods = ["neum1", "neum1moor1", "moor1", "neum2", "neum2moor2", "neum3", "neum3moor2", "moor2", "neum3moor3", "moor3"]
#neighborhoods = ["moor3moor4", "moor4"]
neighborhoods = ["neum1", "neum1moor1", "moor1", "neum2", "neum2moor2", "neum3", "neum3moor2", "moor2", "neum3moor3", "moor3", "moor3moor4", "moor4"]
gammas = [0.5,1.,2.]
num = 5
path = getPath()["ca3d"]
filename = "CA3D_FaDu__"
model = "slice_model_RT"
logfile = "CA3D_Fak_9"
debug = False
dryrun = False

if debug:
    neighborhoods = ["neum1moor1"]
    gammas = [1.0]
    num = 5

for gamma in gammas:
    data[str(gamma).replace(".","")] = {}
    for n in neighborhoods:
        y_mean = None
        for i in range(num):
            if dryrun:
                 if not os.path.isfile(path + "experiments/" + filename + n + "_v9_caliFak_No" + str(i+1) + "_gamma" + str(gamma).replace(".","") + ".npz"):
                      print("Missing: " + filename + n + "_v9_caliFak_No" + str(i+1) + "_gamma" + str(gamma).replace(".","") + ".npz")
            else:
                file = np.load(path + "experiments/" + filename + n + "_v9_caliFak_No" + str(i+1) + "_gamma" + str(gamma).replace(".","") + ".npz", allow_pickle=True)
                t, y = getDatapoints(file)
                if y_mean is None:
                    y_mean = y
                else:
                    y_mean += y
        if not dryrun:
            y_mean /= float(num)
            data[str(gamma).replace(".","")][n] = y_mean

################################################################################
if not dryrun:
    x_var = np.zeros(len(y_mean))
    y_var = np.ones(len(y_mean))

    t1 = time.time()

    values, lmfit_parameters = loadParametersFromLog("slice_model_RT_FaDu_dr_11_1_log")
    lmfit_parameters["v0"].value *= 2.5
    lmfit_parameters["epsilon"].value *= 0.
    lmfit_parameters["delta"].value *= 0.
    #lmfit_parameters["gamma"].value *= 0.5
    lmfit_parameters["tmax"].value = 10.*24.*3600.
    lmfit_parameters["dr"].vary = True
    lmfit_parameters["dr"].value = lmfit_parameters["og_dr"].value * 1.1
    lmfit_parameters["dr"].min = 0.5*lmfit_parameters["og_dr"].value
    lmfit_parameters["dr"].max = 10. * lmfit_parameters["og_dr"].value
    lmfit_parameters.add("p_mit_cata",value=0.5,vary=False)

    copy_parameter = deepcopy(lmfit_parameters)

    c = setVolume_3(lmfit_parameters, t, model)

    MyClass = getattr(models,model)
    instance = MyClass()

    flushLog(logfile)

    #"""

    fak = np.zeros((3,len(neighborhoods)))

    i = 0
    for gamma in gammas:
        j = 0
        for n in neighborhoods:

            y_mean = data[str(gamma).replace(".","")][n]

            atol = 1e-6 # 1e-6 # 1e-8
            rtol = 1e-3 # 1e-3 # 1e-5
            t_start = time.time()

            specialParameters={"c":c, "time":t_start, "filename":logfile, "atol":atol, "rtol":rtol}
            #specialParameters={"c":c, "time":t_start, "atol":atol, "rtol":rtol}
            fitParameters = {"resid_func":"residual_r0", "cb_iter_func" : "cb_iter_print"}

            lmfit_parameters = deepcopy(copy_parameter)
            lmfit_parameters["gamma"].value *= gamma
            if not debug:
                optiPara, initPara, metrics, chisqr, lmfitObj = instance.getFit(lmfit_parameters, t, x_var, y_mean/1e6, y_var,
                    fitParameters = fitParameters, specialParameters=specialParameters)
            else:
                optiPara = lmfit_parameters

            #optiPara = lmfit_parameters
            #optiPara["dr"].value = lmfit_parameters["og_dr"].value * 0.8482288386294038
            kappa = optiPara["dr"].value / optiPara["og_dr"].value

            print2("gamma: " + str(gamma),logfile)
            print2("neighborhood: " + n,logfile)
            print2("kappa: " + str(kappa),logfile)
            if not debug:
                printFitResults(lmfit_parameters, optiPara, lmfitObj, logfile, "FaDu")
            else:
                prettyPrintLmfitParameters(lmfit_parameters)
            fak[i][j] = kappa
            j += 1

            arg = instance.getValue(optiPara, t, specialParameters=specialParameters)
            sol = arg[0]
            metrics = instance.getMetrics(optiPara, sol)

            plt.figure()
            plt.title("CA3D_1D_Fit_gamma" + str(gamma).replace(".","") + "_" + n)
            plt.plot(t/24./3600., y_mean, label="CA3D", color="blue")
            plt.plot(t/24./3600., metrics[3]*1e6, label="1D", linestyle="dashed", color="red")
            plt.plot(t/24./3600., t * lmfit_parameters["gamma"].value * lmfit_parameters["dr"].value * 1e1 + metrics[3][0]*1e6, linestyle = "dotted", color = "black", label="prediction")
            plt.legend()
            if not debug:
                plt.savefig(getPath()["ca3d"] + "CA3D_1D_Fit_v4_gamma" + str(gamma).replace(".","") + "_" + n + ".pdf",transparent=True)
                plt.close()

            plt.figure()
            plt.title("CA3D_1D_Fit_gamma" + str(gamma).replace(".","") + "_" + n)
            animate_6(0, sol, t/24./3600., len(t), optiPara["dr"].value * 1e1, p0 = None, metrics1 = metrics, metrics2 = None)
            if not debug:
                plt.savefig(getPath()["ca3d"] + "CA3D_1D_Fit_v4_gamma" + str(gamma).replace(".","") + "_" + n + "_start.pdf",transparent=True)
                plt.close()

            plt.figure()
            plt.title("CA3D_1D_Fit_gamma" + str(gamma).replace(".","") + "_" + n)
            animate_6(-1, sol, t/24./3600., len(t), optiPara["dr"].value * 1e1, p0 = None, metrics1 = metrics, metrics2 = None)
            if not debug:
                plt.savefig(getPath()["ca3d"] + "CA3D_1D_Fit_v4_gamma" + str(gamma).replace(".","") + "_" + n + "_end.pdf",transparent=True)
                plt.close()

            if debug:
                plt.show()

            #break
            #break
        i += 1
    print2(str(fak), logfile)

    t_s = time.time() - t1
    print2(str(round(t_s,1)), logfile)
    t_h = t_s / 3600.
    print2(str(round(t_h,1)), logfile)
    t_d = t_h / 24.
    print2(str(round(t_d,1)), logfile)

    if not debug:
        sendEmail("Simulations for kappa fit has ended", str(fak), "mail@florian-franke.eu", [])
    #"""







    #
