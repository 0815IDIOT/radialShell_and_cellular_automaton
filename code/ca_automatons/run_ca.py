import sys
sys.path.insert(0,'../../.')
from tools import *
from cellularautomaton import main
from constants import *
from datetime import datetime
from sendNotifications import *
import os
#import faulthandler; faulthandler.enable()

def default_parameters(ref_sim, neighborhood, filename_addon = "", plotFig = True, saveFig = False, cali_mode=False, run_1d = True, saveCA = False):
    simulations = [
        {
            "logFile" : "slice_model_RT_HCT_116_dr_12_5_log",
            "plate": "000",
            "cellline" : "HCT_116",
            "exp_name" : "Gri2016_Fig4a_exp",
        },
        {
            "logFile": "slice_model_RT_HCT_116_Bru2019_Fig5a_dr_12_3_log",
            "plate": "000",
            "cellline" : "HCT_116",
            "exp_name" : "Bru2019_Fig5a",
        },
        {
            "logFile": "slice_model_RT_MDA_MB_468_dr_12_1_log",
            "plate": "000",
            "cellline" : "MDA_MB_468",
            "exp_name" : "Gri2016_Fig4c_exp",
        },
        {
            "logFile": "slice_model_RT_LS_174T_dr_12_2_log",
            "plate": "000",
            "cellline" : "LS_174T",
            "exp_name" : "Gri2016_Fig4b_exp",
        },
        {
            "logFile": "slice_model_RT_SCC_25_dr_12_1_log",
            "plate": "000",
            "cellline" : "SCC_25",
            "exp_name" : "Gri2016_Fig4d_exp",
        },
        {
            "logFile": "slice_model_RT_FaDu_dr_12_2_log",
            "plate": "",
            "cellline" : "FaDu",
            "datasource" : "source2",
            "exp_name" : "",
        },
    ]

    #ref_sim = 5
    """
    0: HCT Gri
    1: HCT Bru
    2: MDA
    3: LS
    4: SCC
    5: FaDu
    """

    cellline = simulations[ref_sim]["cellline"]
    exp_name = simulations[ref_sim]["exp_name"]
    plate = simulations[ref_sim]["plate"]
    values, lmfit_parameters = loadParametersFromLog(simulations[ref_sim]["logFile"])

    filename = "CA3D_"  + cellline + "_" + exp_name[:3] + "_" + neighborhood
    if filename_addon != "":
        filename += "_" + filename_addon

    # General Parameters
    m = values["m"] # scaling factor
    D_c = values["D_c"] # m^2/s
    D_c2 = values["D_c2"] # m^2/s
    llambda = values["llambda"] # 1/s # migration rate
    D_p = values["D_p"] # m^2/s
    a = values["a"] # mmHg/s
    og_a = values["og_a"] # mmHg/s
    gamma = values["gamma"] # 1/s # division rate
    v0 = values["v0"] # m^3
    if cali_mode:
        v0 *= 2.5
    epsilon = values["epsilon"]  # 1/s # rate hypoxia death
    delta = values["delta"] # 1/s # rate clear necrotic
    pcrit = values["pcrit"] # mmHg
    ph = values["ph"] # mmHg
    p0 = values["p0"] # mmHg
    tmax = values["tmax"]

    therapy_schedule = np.array([[100.*24*3600, 5.,0.5]], dtype=float)
    #therapy_schedule = np.array([[0.004*24*3600, 5.,0.30776227],[3.*24*3600,0.,0.70495827]], dtype=float) # 10Gy_1_9 | 10Gy_2_7
    alphaR = 0.5 # 1/Gy
    betaR = 0.042  # 1/Gy**2
    gammaR = 1.0 # unitless
    p_mit_cata = 0.0 # prop. of mitotic catastrophy

    # 1D Model Parameters
    max_iter_time = 60
    rmax = 1.1e-3 * m
    #dr = values["og_dr"] #13.e-6 * m #7.5e-6 * m
    dr = values["dr"]
    og_dr = values["og_dr"]
    rate_fak = 1.

    if neighborhood == "neum1":
        fak = 0.81 # Neumann1
    elif neighborhood == "neum1moor1":
        fak = 1.11 # Neumann1 Moor1
    elif neighborhood == "moor1":
        fak = 1.39# * 1.116247621623896 # Moor1
    elif neighborhood == "neum2":
        fak = 1.48 # Neumann2
    elif neighborhood == "neum2moor2":
        fak = 2.08 # Neumann2 Moor2
    elif neighborhood == "neum3":
        fak = 2.11 # Neumann3
    elif neighborhood == "neum3moor2":
        fak = 2.42 # Neumann3 Moor2
    elif neighborhood == "moor2":
        fak = 2.58 # Moor2
    elif neighborhood == "neum3moor3":
        fak = 2.94 # Moor3
    elif neighborhood == "moor3":
        fak = 3.71 # Moor3
    elif neighborhood == "moor3moor4":
        fak = 4.26 # Moor3Moor4
    elif neighborhood == "moor4":
        fak = 4.76 # Moor4

    if cali_mode:
        delta = 0
        epsilon = 0

        # Version 1
        #gamma *= 0.5
        dr = og_dr * fak
        rate_fak = 1.

        # Version 2
        #rate_fak = 1./fak
        #dr = og_dr

        # test
        #rate_fak = (dr * fak / og_dr) # 2.198068524949465
        #rate_fak = 1./(dr * fak / og_dr) # 0.06152983934519354
        #rate_fak = 2.749502562653766   -> fak * 1.9756286040820603

        # test2
        #dr = og_dr * 6.
        #rate_fak = 6./fak

        #print("kappa:    " + str((dr / og_dr)))
        #print("fak:      " + str(fak))
        #print("rate_fak: " + str(rate_fak))

    gridPoints = int(round(rmax/dr,2))
    rd = 1000*m

    # CA Parameters
    cdiff = 0.01 # [%] concentration difference below which equilibrium is assumed
    #cmode = CMODE_P0_R0 # mode for analytical c(r)
    cmode = CMODE_P0_GRID
    cperformance = CPERFORMANCE_FULL # [1=True, 0=False] True => only compute in vicinity of spheroid ; False => compute whole grid
    #cperformance = CPERFORMANCE_ANALYTICAL
    dx = values["og_dr"] / values["m"]  # 13. * 1e-6 # m per grid cell (edge length)

    dt = 0.1 / max(np.array([gamma * rate_fak, epsilon, delta])) # s
    N0 = int(v0/ m**3 / (dx**3)) # number of cells for t = 0
    N_grid = 150 # number of cells per dimension

    #date_now = datetime.now()
    #date = date_now.year * 100000000. + date_now.month * 1000000. + date_now.day * 1000. + date_now.hour * 100. + date_now.minute
    date = datetime.now()

    parameters = {
        "m" : m,
        "D_c": D_c,
        "D_c2": D_c2,
        "llambda": llambda,
        "D_p": D_p,
        "D": D_p,
        "a": a,
        "og_a" : og_a,
        "gamma" : gamma,
        "dr": dr,
        "og_dr" : og_dr,
        "v0" : v0,
        "epsilon" : epsilon,
        "delta" : delta,
        "pcrit" : pcrit,
        "ph" : ph,
        "p0" : p0,
        "tmax" : tmax,
        "therapy_schedule" : therapy_schedule,
        "alphaR" : alphaR,
        "betaR" : betaR,
        "gammaR" : gammaR,
        "p_mit_cata" : p_mit_cata,
        "ode_max_iter_time" : max_iter_time,
        "ode_rmax" : rmax,
        "ode_dr" : dr,
        "ode_og_dr" : og_dr,
        "ode_gridPoints" : gridPoints,
        "ode_rd" : rd,
        "ode_fak" : 1., #1., #fak,
        "ode_dose" : 0,
        "ode_cthreshold" : 0.1,
        "ca_a" : a,
        "ca_rate_fak" : rate_fak,
        "ca_fak" : fak,
        "ca_llambda": 1,
        "ca_gamma" : gamma * rate_fak,
        "ca_epsilon" : epsilon,
        "ca_delta" : delta,
        "ca_cdiff" : cdiff,
        "ca_cmode" : cmode,
        "ca_cperformance" : cperformance,
        "ca_dt" : dt,
        "ca_dx" : dx,
        "ca_N0" : N0,
        "ca_N_grid" : N_grid,
        "ca_stop_radius" : 2000.,
        "saveFig" : saveFig,
        "plotFig" : plotFig,
        "force_logflush" : False,
        "run_1d" : run_1d,
        "saveCa" : saveCA,

    }

    plot_para = {
        "plate": simulations[ref_sim]["plate"],
        "cellline" : simulations[ref_sim]["cellline"],
        "exp_name" : simulations[ref_sim]["exp_name"],
        "datasource" : "" if not "datasource" in simulations[ref_sim] else simulations[ref_sim]["datasource"],
    }

    string_para = {
        "filename": filename,
        "ca_neighborhood" : neighborhood,
        "date" : date
    }

    return parameters, plot_para, string_para

if __name__ == "__main__":

    """
    1: single run
    2: multiple run (differ gamma, neighborhood, 5 times)
    3: multiple run (recreate FaDu 10 times)
    """
    option = 1

    ############################################################################

    if option == 1:

        parameters, plot_para, string_para = default_parameters(5, "moor3moor4", "o2_test_FaDu_v1_2", plotFig = False, saveFig = True, cali_mode = False, run_1d = True, saveCA = True)

        parameters["tmax"] = 14.*24.*3600.

        #prettyDictPrint(parameters)
        #parameters["v0"] *= 5.0
        #parameters["dr"] = parameters["og_dr"] * parameters["ca_fak"]
        #parameters["ode_dr"] = parameters["og_dr"] * parameters["ca_fak"]
        #parameters["ode_gridPoints"] = int(round(parameters["ode_rmax"]/parameters["dr"],2))
        #parameters["ca_N0"] = int(parameters["v0"]/ parameters["m"]**3 / (parameters["ca_dx"]**3))
        """
        parameters["gamma"] *= 2. # 247325
        parameters["ca_gamma"] *= 2.
        parameters["delta"] *= 0
        parameters["epsilon"] *= 0
        parameters["ca_delta"] *= 0
        parameters["ca_epsilon"] *= 0
        parameters["ca_dt"] = 0.1 / max(np.array([parameters["gamma"]]))
        """
        #prettyDictPrint(parameters)

        main(parameters, plot_para, string_para)

    ############################################################################
    
    elif option == 2:
    
        parameters_default, _, _ = default_parameters(5, "neum1", "caliFak_No", plotFig = False, saveFig = False, cali_mode = True, run_1d = True, saveCA = True)

        #neighboords = ["neum1", "neum1moor1", "moor1", "neum2", "neum2moor2", "neum3", "neum3moor2", "moor2", "neum3moor3", "moor3"]
        #neighboords = ["moor3moor4", "moor4"]
        neighboords = ["neum1", "neum1moor1", "moor1", "neum2", "neum2moor2", "neum3", "neum3moor2", "moor2", "neum3moor3", "moor3", "moor3moor4", "moor4"]
        gammas = [0.5,1.,2.]
        #gammas = [2.]
        overrideExisting = True

        for gamma in gammas:
            for i in range(5):
                for n in neighboords:
                    parameters, plot_para, string_para = default_parameters(5, n, "v9_caliFak_No"+str(i+1) + "_gamma" + str(gamma).replace(".",""), plotFig = False, saveFig = False, cali_mode = True, run_1d = True, saveCA = True)
                    parameters["tmax"] = 10.*24.*3600.
                    parameters["ca_gamma"] = parameters_default["ca_gamma"] * gamma
                    parameters["gamma"] = parameters_default["gamma"] * gamma
                    parameters["ca_dt"] = 0.1 / max(np.array([parameters["gamma"]]))

                    logFile = getPath()["ca3d"]+"log/" + string_para["filename"] + "_log.txt"
                    if not os.path.isfile(logFile) or overrideExisting:
                        main(parameters, plot_para, string_para)
                    else:
                        print("[*] found existing log file: " + logFile)

        sendEmail("Simulations for CA3D has ended", "Simulations for CA3D has ended", "mail@florian-franke.eu", [])

    ############################################################################

    elif option == 3:

        # FaDu Recreating:

        for i in range(10):
            parameters, plot_para, string_para = default_parameters(5, "moor3moor4", "o2_test_FaDu_v1_1_No" + str(i+1), plotFig = False, saveFig = True, cali_mode = False, run_1d = True, saveCA = True)

            parameters["tmax"] = 14.*24.*3600.

            main(parameters, plot_para, string_para)
        
        sendEmail("Simulations for CA3D has ended", "Simulations for CA3D has ended", "mail@florian-franke.eu", [])

    ############################################################################

    else:
        print("No valid option selected")