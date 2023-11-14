from abc import ABC, abstractmethod
from ode_uni import ode_models
from diskret_uni import diskret_uni
from lmfit_uni import *
from scipy.integrate import solve_ivp
from o2_models import o2_models
from tools import *
import copy
import numpy as np

class cellmodels(ABC):
    def __init__(self,ode,varys,fitParameters={},scipy_methode="LSODA"):
        self.ode = ode
        self.scipy_methode = scipy_methode
        self.varys = varys
        self.fitParameters = fitParameters

    def adjustVary(self, dict, parameter):
        for p in parameter:
            if not p in dict:
                parameter[p].vary = False
        return parameter

    def getValue(self, parameters, x_mean, specialParameters = {}):
        parameters = self.parameterPrep(parameters)
        rn, rh, r0, _, _ = o2_models.sauerstoffdruck_Gri2016(parameters["v0"].value, parameters)
        v0 = [parameters["v0"].value, rn, rh, r0]
        sol = solve_ivp(self.ode, [x_mean[0],x_mean[-1]], v0, args=(parameters,), method=self.scipy_methode, t_eval = x_mean)
        if sol.status == -1:
            print("")
            print("[*] ERROR in solve_ivp")
            print(sol.message)
        sol = np.transpose(sol.y)
        return [sol]

    def getFit(self, parameters, x_mean, x_var, y_mean, y_var, fitParameters={}, specialParameters = {}):

        parameters = self.parameterPrep(parameters)
        fitParameters["specialParameters"] = specialParameters
        fitParameters = {**self.fitParameters, **fitParameters}
        output, mini, res = lmfit_uni_var(x_mean, x_var, y_mean, y_var, parameters, self, **fitParameters)
        optiPara = output.params

        arg = self.getValue(optiPara, x_mean, specialParameters=specialParameters)
        metrics = self.getMetrics(parameters, arg[0]) # v0, vh, vn, r0, rh, rn

        if not "resid_func" in fitParameters:
            residual = getattr(residual_functions(), "residual")
        else:
            residual = getattr(residual_functions(), fitParameters["resid_func"])

        resid = residual(parameters, self, x_mean, x_var, y_mean, y_var, specialParameters)
        chisqr = np.sum(resid**2)

        return optiPara, parameters, metrics, chisqr, [output, mini, res]

    @abstractmethod
    def parameterPrep(self, parameters):
        pass

    @abstractmethod
    def getMetrics(self, parameters, sol):
        # [v0,vh,vn,r0,rh,rn]
        pass

class Schichtenmodel(cellmodels):
    def __init__(self):
        ode = ode_models.ode_Gri2016
        varys = ["D", "a", "delta", "gamma", "p0", "ph", "pcrit", "v0"]
        fitParameters = {}
        scipy_methode="LSODA"

        super().__init__(ode, varys, fitParameters=fitParameters,scipy_methode=scipy_methode)

    def parameterPrep(self, parameters):

        if not "rd" in parameters:
            parameters.add("rd",value=1000,min=0,max=1000,vary=False)
        else:
            parameters["rd"].value = 1000
            parameters["rd"].min = 0
            parameters["rd"].max = 1000
            parameters["rd"].vary = False

        parameters = copy.deepcopy(parameters)
        parameters = super().adjustVary(self.varys, parameters)
        return parameters

    def getMetrics(self, parameters, sol):

        m = 1. if not "m" in parameters else parameters["m"].value

        r0 = sol[:,3] / m
        rh = sol[:,2] / m
        rn = sol[:,1] / m
        v0 = sol[:,0] / m**3
        vh = 4. * np.pi * (rh)**3 / 3.
        vn = 4. * np.pi * (rn)**3 / 3.

        return [v0,vh,vn,r0,rh,rn]

class Schichtenmodel_rd(cellmodels):
    def __init__(self):
        ode = ode_models.ode_Gri2016
        varys = ["D", "a", "delta", "gamma", "p0", "ph", "pcrit", "v0", "rd"]
        fitParameters = {}
        scipy_methode="LSODA"

        super().__init__(ode, varys, fitParameters=fitParameters,scipy_methode=scipy_methode)

    def parameterPrep(self, parameters):

        rl = np.sqrt((6. * parameters["D"].value * (parameters["p0"].value-parameters["pcrit"].value)) / (parameters["a"].min))
        value_max = rl * np.sqrt(1-parameters["ph"].value/parameters["p0"].value)
        value_min = 0
        value = (value_min+value_max)/2. + value_min
        parameters["rd"].min = value_min
        parameters["rd"].max = value_max
        parameters["rd"].value = value
        parameters["rd"].vary = True

        parameters = copy.deepcopy(parameters)
        parameters = super().adjustVary(self.varys, parameters)
        return parameters

    def getMetrics(self, parameters, sol):

        m = 1. if not "m" in parameters else parameters["m"].value

        r0 = sol[:,3] / m
        rh = sol[:,2] / m
        rn = sol[:,1] / m
        v0 = sol[:,0] / m**3
        vh = 4. * np.pi * (rh)**3 / 3.
        vn = 4. * np.pi * (rn)**3 / 3.

        return [v0,vh,vn,r0,rh,rn]

class Gri2016_disk(cellmodels):
    def __init__(self):
        ode = diskret_uni.diskrete_Gri2016
        varys = ["D", "a", "td", "p0", "ph", "pcrit", "v0"]
        fitParameters = {}
        scipy_methode="LSODA"

        super().__init__(ode, varys, fitParameters=fitParameters,scipy_methode=scipy_methode)

    def getValue(self, parameters, x_mean, specialParameters = {}):
        sol = self.ode(x_mean, parameters)
        return [sol]

    def parameterPrep(self, parameters):

        rl = np.sqrt((6. * parameters["D"].value * (parameters["p0"].value-parameters["pcrit"].value)) / (parameters["a"].min))
        value_max = rl * np.sqrt(1-parameters["ph"].value/parameters["p0"].value)
        value_min = 0
        value = (value_min+value_max)/2. + value_min
        parameters["rd"].min = value_min
        parameters["rd"].max = value_max
        parameters["rd"].value = value
        parameters["rd"].vary = True

        parameters = copy.deepcopy(parameters)
        parameters = super().adjustVary(self.varys, parameters)
        return parameters

    def getMetrics(self, parameters, sol):

        m = 1. if not "m" in parameters else parameters["m"].value

        v0 = sol
        vh = np.zeros(len(v0))
        vn = np.zeros(len(v0))
        r0 = (v0 * 3. / 4. / np.pi / m)**(1./3.)
        rh = np.zeros(len(v0))
        rn = np.zeros(len(v0))

        return [v0,vh,vn,r0,rh,rn]

class distrubution_model_debug(cellmodels):
    def __init__(self):
        ode = ode_models.ode_DistrubutionModel
        varys = ["D_p","D_c","D_c2","delta","a","gamma","epsilon","p0","ph","pcrit"]
        fitParameters = {}
        scipy_methode="RK23"

        super().__init__(ode, varys, fitParameters=fitParameters,scipy_methode=scipy_methode)

    def parameterPrep(self, parameters):

        if not "rd" in parameters:
            parameters.add("rd",value=1000,min=0,max=1000,vary=False)
        else:
            parameters["rd"].value = 1000
            parameters["rd"].min = 0
            parameters["rd"].max = 1000
            parameters["rd"].vary = False

        parameters = copy.deepcopy(parameters)
        parameters = super().adjustVary(self.varys, parameters)
        return parameters

    def getValue(self, parameters, x_mean,specialParameters = {}):
        #print("[+] val")
        parameters = self.parameterPrep(parameters)
        c = specialParameters["c"]
        sol = solve_ivp(self.ode, [x_mean[0],x_mean[-1]], c, args=(parameters,), method=self.scipy_methode, t_eval = x_mean, atol = 1e-8, rtol = 1e-5)
        if sol.status == -1:
            print("")
            print("[*] ERROR in solve_ivp")
            print(sol.message)
        sol = np.transpose(sol.y)
        return [sol]

    def getMetrics(self, parameters, sol):

        m = 1. if not "m" in parameters else parameters["m"].value

        dr = parameters["dr"].value / m
        n = int(len(sol[0])/2.)
        vmax = 4. * np.pi * ((np.arange(1,n,1)*dr)**3 - (np.arange(0,n-1,1)*dr)**3) / 3.

        split = np.array(np.split(sol,2,1))
        v0 = np.sum((split[0][:,1:] + split[0][:,:-1] + split[1][:,1:] + split[1][:,:-1]) * vmax / 2.,1)
        vn = np.sum((split[1][:,1:] + split[1][:,:-1]) * vmax / 2.,1)

        r0 = (v0 * 3. / 4. / np.pi)**(1./3.)
        rn = (vn * 3. / 4. / np.pi)**(1./3.)

        rh = []

        for i in range(len(sol)):
            c0 = split[0][i]
            _, rhypox, _ = o2_models.sauerstoffdruck_analytical_3(parameters,c0,mode=2)
            rh.append(rhypox)

        rh = np.array(rh)
        vh = 4. * np.pi * rh**3 / 3.

        return [v0,vh,vn,r0,rh,rn]

class distrubution_model(cellmodels):
    def __init__(self):
        ode = ode_models.distrubution_model_11
        varys = ["D_p","D_c","D_c2","delta","a","gamma","epsilon","p0","ph","pcrit"]
        fitParameters = {}
        scipy_methode="RK23"

        super().__init__(ode, varys, fitParameters=fitParameters,scipy_methode=scipy_methode)

    def parameterPrep(self, parameters):

        if not "rd" in parameters:
            parameters.add("rd",value=1000,min=0,max=1000,vary=False)
        else:
            parameters["rd"].value = 1000
            parameters["rd"].min = 0
            parameters["rd"].max = 1000
            parameters["rd"].vary = False

        parameters = copy.deepcopy(parameters)
        parameters = super().adjustVary(self.varys, parameters)
        return parameters

    def getValue(self, parameters, x_mean,specialParameters = {}):
        #print("[+] val")
        parameters = self.parameterPrep(parameters)
        c = specialParameters["c"]
        sol = solve_ivp(self.ode, [x_mean[0],x_mean[-1]], c, args=(parameters,), method=self.scipy_methode, t_eval = x_mean, atol = 1e-8, rtol = 1e-5)
        if sol.status == -1:
            print("")
            print("[*] ERROR in solve_ivp")
            print(sol.message)
            print(parameters)
        sol = np.transpose(sol.y)
        return [sol]

    def getMetrics(self, parameters, sol):

        m = 1. if not "m" in parameters else parameters["m"].value

        dr = parameters["dr"].value / m
        n = int(len(sol[0])/2.)
        vmax = 4. * np.pi * ((np.arange(1,n,1)*dr)**3 - (np.arange(0,n-1,1)*dr)**3) / 3.

        split = np.array(np.split(sol,2,1))
        v0 = np.sum((split[0][:,1:] + split[0][:,:-1] + split[1][:,1:] + split[1][:,:-1]) * vmax / 2.,1)
        vn = np.sum((split[1][:,1:] + split[1][:,:-1]) * vmax / 2.,1)

        r0 = (v0 * 3. / 4. / np.pi)**(1./3.)
        rn = (vn * 3. / 4. / np.pi)**(1./3.)

        rh = []

        for i in range(len(sol)):
            c0 = split[0][i]
            _, rhypox, _ = o2_models.sauerstoffdruck_analytical_1(parameters,c0,mode=2)
            rh.append(rhypox)

        rh = np.array(rh)
        vh = 4. * np.pi * rh**3 / 3.

        return [v0,vh,vn,r0,rh,rn]

class slice_model_debug(cellmodels):
    def __init__(self):
        ode = ode_models.ode_Slicemodel
        varys = ["D_p","D_c","delta","a","gamma","p0","ph","pcrit","llambda","epsilon","p_mit_cata","alphaR","betaR","gammaR"]
        fitParameters = {}
        scipy_methode="RK23"

        super().__init__(ode, varys, fitParameters=fitParameters,scipy_methode=scipy_methode)

    def parameterPrep(self, parameters):

        if not "rd" in parameters:
            parameters.add("rd",value=1000,min=0,max=1000,vary=False)
        else:
            parameters["rd"].value = 1000
            parameters["rd"].min = 0
            parameters["rd"].max = 1000
            parameters["rd"].vary = False

        parameters = copy.deepcopy(parameters)
        parameters = super().adjustVary(self.varys, parameters)
        return parameters

    def getValue(self, parameters, t, therapy, specialParameters = {}):
        parameters = self.parameterPrep(parameters)
        c = specialParameters["c"]
        sol_sum = None
        t_sum = None
        atol = 1e-6 if "atol" not in specialParameters else specialParameters["atol"] # 1e-10
        rtol = 1e-3 if "rtol" not in specialParameters else specialParameters["rtol"] # 1e-8

        for i in range(len(t)):
            c = therapy_RT(parameters,c, therapy[i])
            #sol = solve_ivp(self.ode, [t[i][0],t[i][-1]], c, args=(parameters,), method=self.scipy_methode, t_eval = t[i], atol = 1e-10, rtol = 1e-8)

            if sol.status == -1:
                print("")
                print("[*] ERROR in solve_ivp")
                print(sol.message)

            sol = np.transpose(sol.y)
            c = sol[-1]

            sol_sum = sol if sol_sum is None else np.append(sol_sum, sol,axis=0)
            t_sum = t[i] if t_sum is None else np.append(t_sum, t[i]+t_sum[-1])
        return [sol_sum, t_sum]

    def getMetrics(self, parameters, sol):

        m = 1. if not "m" in parameters else parameters["m"].value

        dr = parameters["dr"].value / m
        n = int(len(sol[0])/(3.))
        vmax = 4. * np.pi * ((np.arange(1,n,1)*dr)**3 - (np.arange(0,n-1,1)*dr)**3) / 3.

        split = np.array(np.split(sol,3,1))
        v0 = np.sum((split[0][:,1:] + split[0][:,:-1] + split[1][:,1:] + split[1][:,:-1] + split[2][:,1:] + split[2][:,:-1]) * vmax / 2.,1)
        vn = np.sum((split[1][:,1:] + split[1][:,:-1]) * vmax / 2.,1)

        r0 = (v0 * 3. / 4. / np.pi)**(1./3.)
        rn = (vn * 3. / 4. / np.pi)**(1./3.)

        rh = []

        for i in range(len(sol)):
            c0 = split[0][i]
            c2 = split[2][i]
            _, rhypox, _ = o2_models.sauerstoffdruck_analytical_3(parameters,c0+c2,mode=2)
            rh.append(rhypox)

        rh = np.array(rh)
        vh = 4. * np.pi * rh**3 / 3.

        return [v0,vh,vn,r0,rh,rn]

class slice_model_FG_prolifRange(cellmodels):
    def __init__(self):
        ode = ode_models.slice_model_6
        varys = ["D_p","D_c","delta","a","gamma","llambda","epsilon","p0","ph","pcrit", "dr"]
        fitParameters = {}
        scipy_methode="RK23"

        super().__init__(ode, varys, fitParameters=fitParameters,scipy_methode=scipy_methode)

    def parameterPrep(self, parameters):

        if not "rd" in parameters:
            parameters.add("rd",value=1000,min=0,max=1000,vary=False)
        else:
            parameters["rd"].value = 1000
            parameters["rd"].min = 0
            parameters["rd"].max = 1000
            parameters["rd"].vary = False

        parameters = copy.deepcopy(parameters)
        parameters = super().adjustVary(self.varys, parameters)
        return parameters

    def getValue(self, parameters, x_mean, specialParameters = {}):
        parameters = self.parameterPrep(parameters)
        atol = 1e-6 if "atol" not in specialParameters else specialParameters["atol"]
        rtol = 1e-3 if "rtol" not in specialParameters else specialParameters["rtol"]

        if parameters["dr"].vary == True:
            from ode_setVolume import setVolume_2
            from o2_models import o2_models

            dr = parameters["dr"].value
            m = parameters["m"].value
            v0 = parameters["v0"].value
            rmax = 1.1e-3 * m
            gridPoints = int(round(rmax/dr,2))
            c0 = setVolume_2(v0, gridPoints, parameters)
            rad = np.array(o2_models.sauerstoffdruck_Gri2016(v0, parameters))

            c1 = setVolume_2(4. * rad[0]**3 * np.pi / 3., gridPoints, parameters)
            c0 -= c1

            c = np.append(c0, c1)
        else:
            c = specialParameters["c"]

        sol = solve_ivp(self.ode, [x_mean[0],x_mean[-1]], c, args=(parameters,), method=self.scipy_methode, t_eval = x_mean, atol = atol, rtol = rtol)

        if sol.status == -1:
            print("")
            print("[*] ERROR in solve_ivp")
            print(sol.message)
            print(parameters)
        sol = np.transpose(sol.y)
        return [sol]

    def getMetrics(self, parameters, sol):

        m = 1. if not "m" in parameters else parameters["m"].value

        dr = parameters["dr"].value / m
        n = int(len(sol[0])/2.)
        vmax = 4. * np.pi * ((np.arange(1,n,1)*dr)**3 - (np.arange(0,n-1,1)*dr)**3) / 3.

        split = np.array(np.split(sol,2,1))
        v0 = np.sum((split[0][:,1:] + split[0][:,:-1] + split[1][:,1:] + split[1][:,:-1]) * vmax / 2.,1)
        vn = np.sum((split[1][:,1:] + split[1][:,:-1]) * vmax / 2.,1)

        r0 = (v0 * 3. / 4. / np.pi)**(1./3.)
        rn = (vn * 3. / 4. / np.pi)**(1./3.)

        rh = []

        for i in range(len(sol)):
            c0 = split[0][i]
            _, rhypox, _ = o2_models.sauerstoffdruck_analytical_1(parameters,c0,mode=2)
            rh.append(rhypox)

        rh = np.array(rh)
        vh = 4. * np.pi * rh**3 / 3.

        return [v0,vh,vn,r0,rh,rn]

class slice_model_FG(cellmodels):
    def __init__(self):
        #ode = ode_models.slice_model_5
        ode = ode_models.slice_model_7
        varys = ["D_p","D_c","delta","a","gamma","llambda","epsilon","p0","ph","pcrit", "dr"]
        fitParameters = {}
        scipy_methode="RK23"

        super().__init__(ode, varys, fitParameters=fitParameters,scipy_methode=scipy_methode)

    def parameterPrep(self, parameters):

        if not "rd" in parameters:
            parameters.add("rd",value=1000,min=0,max=1000,vary=False)
        else:
            parameters["rd"].value = 1000
            parameters["rd"].min = 0
            parameters["rd"].max = 1000
            parameters["rd"].vary = False

        parameters = copy.deepcopy(parameters)
        parameters = super().adjustVary(self.varys, parameters)
        return parameters

    def getValue(self, parameters, x_mean, specialParameters = {}, event_f = None):
        parameters = self.parameterPrep(parameters)
        atol = 1e-6 if "atol" not in specialParameters else specialParameters["atol"]
        rtol = 1e-3 if "rtol" not in specialParameters else specialParameters["rtol"]
        #atol = 1e-10
        #rtol = 1e-7

        if parameters["dr"].vary == True:
            from ode_setVolume import setVolume_3
            """
            from o2_models import o2_models

            dr = parameters["dr"].value
            m = parameters["m"].value
            v0 = parameters["v0"].value
            fak = 1. if not "og_dr" in parameters else dr / parameters["og_dr"].value

            rmax = parameters["rmax"].value if "rmax" in parameters else 1.1e-3 * m
            gridPoints = int(round(rmax/dr,2))
            c0 = setVolume_2(v0, gridPoints, parameters,fak)
            rad = np.array(o2_models.sauerstoffdruck_Gri2016(v0, parameters))

            c1 = setVolume_2(4. * rad[0]**3 * np.pi / 3., gridPoints, parameters,fak)
            c0 -= c1

            c = np.append(c0, c1)
            """
            c = setVolume_3(parameters, x_mean, self.__class__.__name__)
        else:
            c = specialParameters["c"]

        #dist = np.zeros(int(len(c)/2.))
        #num = np.zeros(int(len(c)/2.))
        dist = None
        num = None

        #sol = solve_ivp(self.ode, [x_mean[0],x_mean[-1]], c, args=(parameters,), method=self.scipy_methode, t_eval = x_mean, atol = 1e-8, rtol = 1e-5)
        sol = solve_ivp(self.ode, [x_mean[0],x_mean[-1]], c, args=(parameters,dist,num), method=self.scipy_methode, t_eval = x_mean, atol = atol, rtol = rtol, events = event_f)

        if sol.status == -1:
            print("")
            print("[*] ERROR in solve_ivp")
            print("[!] parameters:")
            print(parameters)
            print("[!] sol.message:")
            print(sol.message)
        t_eval = sol.t
        if sol.y_events is None:
            y_event= None
        else:
            y_event = sol.y[-1] if len(sol.y_events[0]) == 0 else sol.y_events[0][0]
        sol = np.transpose(sol.y)
        return [sol,t_eval, y_event, dist, num]

    def getMetrics(self, parameters, sol, mode=0):

        m = 1. if not "m" in parameters else parameters["m"].value

        dr = parameters["dr"].value / m
        n = int(len(sol[0])/2. + 1)
        vmax = 4. * np.pi * ((np.arange(1,n,1)*dr)**3 - (np.arange(0,n-1,1)*dr)**3) / 3.

        split = np.array(np.split(sol,2,1))

        if mode == 0:
            v0 = np.sum((split[0] + split[1]) * vmax,1)
            vn = np.sum(split[1] * vmax,1)
        elif mode == 1:
            print("[!] The 'mode' 1 in 'getMetrics' is deprecated!")
            v0 = []
            for row in split[0]:
                idx_0 = np.where(row>0.5)[0]
                if len(idx_0) > 0:
                    idx_0 = idx_0[-1]
                    if idx_0 == len(row)-1:
                        idx_0 -= 1
                    idx_1 = idx_0 + 1
                    slope = (row[idx_1] - row[idx_0])
                    offset = row[idx_0]
                    r0 = dr*((0.5-offset)/slope + idx_0)
                    v0.append(4.*np.pi*r0**3 / 3.)
                else:
                    v0.append(0)
            v0 = np.array(v0)

            vn = []
            for row in split[1]:
                idx_0 = np.where(row>0.5)[0]
                if len(idx_0) > 0:
                    idx_0 = idx_0[-1]
                    idx_1 = idx_0 + 1
                    slope = (row[idx_1] - row[idx_0])
                    offset = row[idx_0]
                    rn = dr*((0.5-offset)/slope + idx_0)
                    vn.append(4.*np.pi*rn**3 / 3.)
                else:
                    vn.append(0)
            vn = np.array(vn)
        else:
            import sys
            print("[!] 'mode' in 'getMetrics' not found")
            sys.exit()

        r0 = (v0 * 3. / 4. / np.pi)**(1./3.)
        rn = (vn * 3. / 4. / np.pi)**(1./3.)

        rh = []

        for i in range(len(sol)):
            c0 = split[0][i]
            _, rhypox, _ , _ = o2_models.sauerstoffdruck_analytical_5(parameters,c0,mode=2)
            rh.append(rhypox/m)

        rh = np.array(rh)
        rh[rh<0] = 0
        vh = 4. * np.pi * rh**3 / 3.

        return [v0,vh,vn,r0,rh,rn]

class slice_model_RT(cellmodels):
    def __init__(self):
        #ode = ode_models.slice_model_5_RT
        ode = ode_models.slice_model_7_RT
        varys = ["D_p","D_c","dr","delta","a","gamma","p0","ph","pcrit","llambda","epsilon","p_mit_cata","alphaR","betaR","gammaR", "p_mit_cata1", "p_mit_cata2", "v0"]
        fitParameters = {}
        scipy_methode="RK23"

        super().__init__(ode, varys, fitParameters=fitParameters,scipy_methode=scipy_methode)

    def parameterPrep(self, parameters):

        if not "rd" in parameters:
            parameters.add("rd",value=1000,min=0,max=1000,vary=False)
        else:
            parameters["rd"].value = 1000
            parameters["rd"].min = 0
            parameters["rd"].max = 1000
            parameters["rd"].vary = False

        parameters = copy.deepcopy(parameters)
        parameters = super().adjustVary(self.varys, parameters)
        return parameters

    def getValue(self, parameters, x_mean, specialParameters = {}, event_f = None):
        parameters = self.parameterPrep(parameters)

        d_therapy = np.array([0]) if not "d_therapy" in specialParameters else specialParameters["d_therapy"]
        t_therapy = np.array([x_mean[-1]]) if not "t_therapy" in specialParameters else specialParameters["t_therapy"]
        p_therapy = np.array([parameters["p_mit_cata"].value]) if not "p_therapy" in specialParameters or specialParameters["p_therapy"] is None else specialParameters["p_therapy"]
        if "p_mit_cata1" in parameters and "p_mit_cata2" in parameters and parameters["p_mit_cata1"].vary and parameters["p_mit_cata2"].vary:
            p_therapy = np.array([parameters["p_mit_cata1"].value, parameters["p_mit_cata2"].value])

        if parameters["dr"].vary or parameters["v0"].vary:
            from ode_setVolume import setVolume_3
            c = setVolume_3(parameters, x_mean, self.__class__.__name__)
        else:
            c = specialParameters["c"]

        atol = 1e-6 if "atol" not in specialParameters else specialParameters["atol"]
        rtol = 1e-3 if "rtol" not in specialParameters else specialParameters["rtol"]

        sol_sum = None
        start = 0.
        end = x_mean[-1]

        for i in range(len(t_therapy)):
            end = t_therapy[i]
            time = x_mean[np.logical_and(x_mean>=start,x_mean<=end)] - start

            parameters["p_mit_cata"].value = p_therapy[i]
            c = therapy_RT(parameters, c, d_therapy[i])
            #sol = solve_ivp(self.ode, [time[0],time[-1]], c, args=(parameters,), method=self.scipy_methode, t_eval = time, atol = 1e-10, rtol = 1e-8)
            sol = solve_ivp(self.ode, [time[0],time[-1]], c, args=(parameters,), method=self.scipy_methode, t_eval = time, atol = atol, rtol = rtol, events = event_f)

            if sol.status == -1:
                logfile = None if not "filename" in specialParameters else specialParameters["filename"]
                print2("", logfile)
                print2("[*] ERROR in solve_ivp", logfile)
                print2("[!] parameters:", logfile)
                prettyPrintLmfitParameters(parameters, logfile)
                print2("[!] sol.message:", logfile)
                print2(sol.message, logfile)

            if sol.y_events is None:
                y_event = None
            else:
                #y_event = sol.y[-1] if len(sol.y_events[0]) == 0 else sol.y_events[0][0]
                y_event = np.transpose(sol.y)[-1] if len(sol.y_events[0]) == 0 else sol.y_events[0][0]

            sol = np.transpose(sol.y)
            c = sol[-1]

            if start == 0:
                sol_sum = sol
            else:
                sol_sum = np.append(sol_sum, sol[1:], axis=0)
            start = t_therapy[i]

        return [sol_sum, x_mean, y_event]

    def getMetrics(self, parameters, sol):

        m = 1. if not "m" in parameters else parameters["m"].value

        dr = parameters["dr"].value / m
        n = int(len(sol[0])/3. + 1)
        vmax = 4. * np.pi * ((np.arange(1,n,1)*dr)**3 - (np.arange(0,n-1,1)*dr)**3) / 3.

        split = np.array(np.split(sol,3,1))
        v0 = np.sum((split[0] + split[1] + split[2]) * vmax,1)
        vn = np.sum(split[1] * vmax,1)

        r0 = (v0 * 3. / 4. / np.pi)**(1./3.)
        rn = (vn * 3. / 4. / np.pi)**(1./3.)

        rh = []

        for i in range(len(sol)):
            c0 = split[0][i]
            c2 = split[2][i]
            _, rhypox, _, _ = o2_models.sauerstoffdruck_analytical_4(parameters,c0+c2,mode=2)
            rh.append(rhypox)

        rh = np.array(rh)
        vh = 4. * np.pi * rh**3 / 3.

        return [v0,vh,vn,r0,rh,rn]
