import sys
sys.path.insert(0,'..')
import numpy as np
from tools import *
import matplotlib.pyplot as plt
import models
import time
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from o2_models import o2_models
from data_loader import data_loader
from ode_setVolume import *
from ode_plots import *
import sys
from sendNotifications import *
from copy import deepcopy
from scipy.interpolate import CubicSpline

def modelMappingFit(model, lmfit_parameters, cellline, expFile, plate, therapyType, filename, saveFig, dataDump, sendMail, info, RT_parameters = None, loadFile = "", debugMode = False):

    flushLog(filename)
    tmax = lmfit_parameters["tmax"].value
    max_iter_time = lmfit_parameters["max_iter_time"].value
    m = lmfit_parameters["m"].value
    t_offset_rel = 0 if not "t_offset_rel" in lmfit_parameters else lmfit_parameters["t_offset_rel"].value
    t_offset_abs = 0 if not "t_offset_abs" in lmfit_parameters else lmfit_parameters["t_offset_abs"].value

    if model.endswith("_RT"):
        mode = "RT"
        if RT_parameters == None:
            t_therapy = np.array([tmax])
        else:
            t_therapy = RT_parameters["t_therapy"]
        if RT_parameters == None:
            d_therapy = np.array([0.])
        else:
            d_therapy = RT_parameters["d_therapy"]
        if RT_parameters == None:
            p_therapy = None
        else:
            p_therapy = RT_parameters["p_therapy"]
    elif model.endswith("_FG"):
        mode = "FG"
        t_therapy = np.array([tmax])
    else:
        print("[*] Error: can not find model mode (FG/RT)" )
        sys.exit()

    t = None
    start = 0
    for i in range(len(t_therapy)):
        end = t_therapy[i]
        t_new = np.linspace(start,end,max_iter_time)
        start = t_therapy[i]
        t = t_new if t is None else np.append(t, t_new, axis=0)
    t = np.unique(t)
    #t = np.linspace(0, tmax, max_iter_time)

    ######### init. Cell concentracion ################################################################

    if loadFile == "":
        """
        c0 = setVolume_2(v0, gridPoints, lmfit_parameters)
        rad = np.array(o2_models.sauerstoffdruck_Gri2016(v0, lmfit_parameters))

        c1 = setVolume_2(4. * rad[0]**3 * np.pi / 3., gridPoints, lmfit_parameters)
        c0 -= c1

        c = np.append(c0, c1)
        """
        c = setVolume_3(lmfit_parameters, t, model)

        #if mode == "RT":
            #c = np.append(c, np.zeros(int(len(c)/2.)))
            #c = therapy_RT(lmfit_parameters,c)
    else:
        data = np.load(getPath()["datadump"] + loadFile + ".npy")
        c = data[-1]


    MyClass = getattr(models,model)
    instance = MyClass()

    print2("[*] Starting simulation: " + filename, filename)

    ######### load and prep. exp data #################################################################

    print2("[*] prepare ref data", filename)

    x_exp_mean, x_exp_var, y_exp_mean, y_exp_var,_ = loadMultiExperiments(cellline, expFile, plate, therapyType)
    """
    a = np.arange(0,len(x_exp_mean),1)
    y_exp_mean = y_exp_mean[np.logical_or(a<2,a>4)]
    x_exp_mean = x_exp_mean[np.logical_or(a<2,a>4)]
    y_exp_var = y_exp_var[np.logical_or(a<2,a>4)]
    x_exp_var = x_exp_var[np.logical_or(a<2,a>4)]
    """
    if t_offset_rel > 0:
        x_exp_mean -= x_exp_mean[t_offset_rel]
        x_exp_mean = x_exp_mean[t_offset_rel:]
        x_exp_var = x_exp_var[t_offset_rel:]
        y_exp_mean = y_exp_mean[t_offset_rel:]
        y_exp_var = y_exp_var[t_offset_rel:]
    elif t_offset_rel < 0:
        t_offset_rel = -t_offset_rel
        x_exp_mean -= x_exp_mean[0]
        x_exp_mean = x_exp_mean[:t_offset_rel]
        x_exp_var = x_exp_var[:t_offset_rel]
        y_exp_mean = y_exp_mean[:t_offset_rel]
        y_exp_var = y_exp_var[:t_offset_rel]
    else:
        x_exp_mean -= x_exp_mean[0]
    x_exp_mean += t_offset_abs

    exp_lmfit_parameters = deepcopy(lmfit_parameters)
    exp_lmfit_parameters["a"].value = exp_lmfit_parameters["og_a"].value

    #x = np.linspace(x_exp_mean[0], x_exp_mean[-1], max_iter_time) # d
    #y_v0 = np.interp(t/24./3600.,x_exp_mean,y_exp_mean) # m**3
    y_v0 = CubicSpline(x_exp_mean, y_exp_mean)
    y_v0 = y_v0(t/24./3600.)
    y_r0 = ((y_v0 / np.pi)*(3./4.))**(1./3.)
    y_rn,_ = getVnFromV0(y_v0,np.zeros(len(y_v0)),exp_lmfit_parameters,outputFormat="r")
    y_vn = 4. * np.pi * y_rn**3 / 3.
    y_rh,_ = getVhFromV0(y_v0,np.zeros(len(y_v0)),exp_lmfit_parameters,outputFormat="r")
    y_vh = 4. * np.pi * y_rh**3 / 3.

    #y_v0_var = np.interp(t/24./3600.,x_exp_mean,y_exp_mean + np.sqrt(y_exp_var))
    y_v0_var = CubicSpline(x_exp_mean, y_exp_mean + np.sqrt(y_exp_var))
    y_v0_var = y_v0_var(t/24./3600.)
    y_vn_var = np.array([o2_models.sauerstoffdruck_Gri2016(v*m**3,exp_lmfit_parameters)[0] for v in y_v0_var])/m
    y_v0_var = (y_v0_var - y_v0) ** 2
    y_vn_var = (y_vn_var - y_vn) ** 2

    metrics_ref = [y_v0,y_vh,y_vn,y_r0,y_rh,y_rn]

    #y_mean = np.append(y_v0, y_vn)
    y_mean = np.append(y_r0, y_rn)
    x_var = np.zeros(len(y_mean))
    y_var = np.ones(len(y_mean))
    #y_var = np.append(y_v0_var,y_vn_var)

    if np.sum(y_var) != len(y_var):
        info += "\nFit with y_var"
    if np.sum(x_var) != 0:
        info += "\nFit with x_var"

    ######### start simulation ########################################################################

    print2("[*] model: " + model,filename)
    t_start = time.time()

    atol = 1e-6 # 1e-6 # 1e-8
    rtol = 1e-3 # 1e-3 # 1e-5

    if mode == "FG":
        specialParameters={"c":c, "time":t_start, "filename":filename, "atol":atol, "rtol":rtol}
    elif mode == "RT":
        specialParameters = {"c":c, "time":t_start, "filename":filename, "t_therapy":t_therapy, "d_therapy":d_therapy, "p_therapy":p_therapy, "atol":atol, "rtol":rtol}
    if not debugMode:
        if lmfit_parameters["v0"].vary:
            fitParameters = {"resid_func":"residual_with_rn_v0Vary", "cb_iter_func" : "cb_iter_print2"}
        else:
            #fitParameters = {"resid_func":"residual_with_vn", "cb_iter_func" : "cb_iter_print2"}
            fitParameters = {"resid_func":"residual_with_rn", "cb_iter_func" : "cb_iter_print2"}
        optiPara, initPara, metrics, chisqr, lmfitObj = instance.getFit(lmfit_parameters, t, x_var, y_mean, y_var,
            fitParameters = fitParameters, specialParameters=specialParameters)
        info += "\nFit function: " + fitParameters["resid_func"]
    else:
        optiPara = lmfit_parameters

    ######### after simulation analysis ###############################################################

    t_end = time.time()
    print2("[[ INFO ]] " + info,filename)
    print2("t: " + str(round(t_end-t_start,2)) + " s",filename)
    print2("t: " + str(round((t_end-t_start)/3600.,2)) + " h",filename)
    print2("t: " + str(round((t_end-t_start)/3600./24.,2)) + " d",filename)

    if not debugMode:
        printFitResults(lmfit_parameters, optiPara, lmfitObj, filename, "-")
    else:
        prettyPrintLmfitParameters(lmfit_parameters,filename)

    arg = instance.getValue(optiPara, t, specialParameters=specialParameters)
    sol = arg[0]
    metrics = instance.getMetrics(optiPara, sol)

    p = []
    p_raw = []
    dr = optiPara["dr"].value / m
    spl = 2 if mode == "FG" else 3

    for c in sol:
        c0 = np.split(c,spl)[0]
        c1 = np.split(c,spl)[1]
        if mode == "RT":
            c2 = np.split(c,spl)[2]
        else:
            c2 = np.zeros(len(c0))
        args_o2 = o2_models.sauerstoffdruck_analytical(params=optiPara, c0=c0+c2, mode=2)
        p0 = args_o2[0]
        p0_raw = args_o2[3]
        p.append(p0)
        p_raw.append(p0_raw)

    exp_data_v0 = [x_exp_mean, y_exp_mean, x_exp_var, y_exp_var, cellline]
    y_exp_mean_vn, y_exp_var_vn = getVnFromV0(y_exp_mean, y_exp_var,exp_lmfit_parameters)
    exp_data_vn = [x_exp_mean, y_exp_mean_vn, x_exp_var, y_exp_var_vn, cellline]

    ######### create plots ############################################################################

    plotResults_6(t/(24.*3600.), metrics, filename[:-3], p_raw, dr, filename,
    np.array([]), np.array([]), sol1 = sol,
    exp_data_vn = exp_data_vn,
    exp_data_v0 = exp_data_v0,
    metrics2 = metrics_ref,
    figSave = saveFig,
    mode = "r",
    pathFigSave = getPath()["gif"])

    anim = animation.FuncAnimation(plt.figure(), animate_6, fargs=[sol, t/(24.*3600.), len(t), dr, p_raw, None], interval=1, frames=len(t), repeat=True)
    if saveFig:
        FFwriter = animation.FFMpegWriter(fps=10)
        var = anim.save(getPath()["gif"] + filename + ".mp4", writer = FFwriter)
        plt.close()
    else:
        plt.show()

    ######### data dump and mail notification #########################################################

    if dataDump:
        np.save(getPath()["datadump"] + filename, sol)

    if sendMail:
        f1 = ["gif", filename + ".pdf"]
        f2 = ["gif", filename + "_start.pdf"]
        f3 = ["gif", filename + "_end.pdf"]
        f4 = ["log", filename + "_log.txt"]
        filelist = [f1,f2,f3,f4]
        sendEmail("Simulation '" + filename + "' has ended", "Simulation '" + filename + "' has ended", "mail@florian-franke.eu", filelist)
