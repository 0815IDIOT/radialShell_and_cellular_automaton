import lmfit
import numpy as np
import time
from datetime import datetime
from tools import *
import os

class residual_functions:
    @staticmethod
    def residual(params, func, x_mean, x_var, y_mean, y_var, specialParameters):
        sol = func.getMetrics(params, func.getValue(params, x_mean, specialParameters)[0])[0]
        resid = (sol-y_mean) / np.sqrt((y_var==0) + y_var)
        return resid[1:]

    @staticmethod
    def residual_r0(params, func, x_mean, x_var, y_mean, y_var, specialParameters):
        sol = func.getMetrics(params, func.getValue(params, x_mean, specialParameters)[0])[3]
        resid = (sol-y_mean) / np.sqrt((y_var==0) + y_var)
        return resid[1:]

    @staticmethod
    def residual_with_rn(params, func, x_mean, x_var, y_mean, y_var, specialParameters):

        sol = func.getValue(params, x_mean, specialParameters)
        metric = func.getMetrics(params, sol[0])
        r = np.append(metric[3], metric[5])

        resid = (r-y_mean) / np.sqrt((y_var==0) + y_var)
        return resid[1:]

    @staticmethod
    def residual_with_rn_v0Vary(params, func, x_mean, x_var, y_mean, y_var, specialParameters):

        sol = func.getValue(params, x_mean, specialParameters)
        metric = func.getMetrics(params, sol[0])
        r = np.append(metric[3], metric[5])

        resid = (r-y_mean) / np.sqrt((y_var==0) + y_var)
        return resid

    @staticmethod
    def residual_with_vn(params, func, x_mean, x_var, y_mean, y_var, specialParameters):
        sol = func.getValue(params, x_mean, specialParameters)
        metric = func.getMetrics(params, sol[0])
        v = np.append(metric[0], metric[2])

        resid = (v-y_mean) / np.sqrt((y_var==0) + y_var)
        return resid[1:]

    @staticmethod
    def residual_border_with_vn(params, func, x_mean, x_var, y_mean, y_var, specialParameters):
        sol = func.getMetrics(params, func.getValue(params, x_mean, specialParameters)[0],mode=1)
        v = np.append(sol[0],sol[2])
        resid = (v-y_mean) / np.sqrt((y_var==0) + y_var)
        return resid[1:]

    @staticmethod
    def residual_border_plus_with_vn(params, func, x_mean, x_var, y_mean, y_var, specialParameters):
        sol0 = func.getMetrics(params, func.getValue(params, x_mean, specialParameters)[0])
        sol1 = func.getMetrics(params, func.getValue(params, x_mean, specialParameters)[0],mode=1)
        v0 = np.append(sol0[0],sol0[2])
        v1 = np.append(sol1[0],sol1[2])
        resid0 = (v0-y_mean) / np.sqrt((y_var==0) + y_var)
        resid1 = (v1-y_mean) / np.sqrt((y_var==0) + y_var)
        return resid0[1:] + resid1[1:]

class cb_iter_functions:
    @staticmethod
    def cb_iter(params, iter, resid, *args, **kws):
        pass

    @staticmethod
    def cb_iter_print(params, iter, resid, *args, **kws):
        func, x_mean, x_var, y_mean, y_var, specialParameters = args

        t = time.time()
        current_date = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

        print("[" + str(iter) + "] finished at " + current_date)
        print("time needed: " + str(round(t - specialParameters["time"],2)) + "s")
        specialParameters["time"] = t
        print("chisqr: " + str(np.sum(resid**2)))
        for para in params:
            if params[para].vary:
                print(str(para) + ": " + str(params[para].value))
        print("")

    @staticmethod
    def cb_iter_print2(params, iter, resid, *args, **kws):
        func, x_mean, x_var, y_mean, y_var, specialParameters = args

        filename = specialParameters["filename"]
        t = time.time()
        current_date = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

        print2("[" + str(iter) + "] finished at " + current_date,filename)
        print2("time needed: " + str(round(t - specialParameters["time"],2)) + "s",filename)
        specialParameters["time"] = t
        print2("chisqr: " + str(np.sum(resid**2)),filename)
        for para in params:
            if params[para].vary:
                print2(str(para) + ": " + str(params[para].value),filename)
        print2("",filename)

################################################################################

def lmfit_uni_var(x_mean, x_var, y_mean, y_var, para, func, resid_func = "residual", cb_iter_func = "cb_iter", emcee_steps = 0, lokalFit = False, specialParameters={}):

    residual = getattr(residual_functions(), resid_func)
    cb_iter = getattr(cb_iter_functions(), cb_iter_func)
    filename = None if "filename" not in specialParameters else specialParameters["filename"]

    if np.sum(x_var) > 0:
        print2("[INFO] x_var > 0 wird mommentan nicht behandelt",filename)

    if not lokalFit:
        print2("[INFO] None lokalFit",filename)
        mini = lmfit.Minimizer(residual, para, nan_policy='propagate', fcn_args=(func, x_mean, x_var, y_mean, y_var, specialParameters,), iter_cb = cb_iter)
        output = mini.minimize(method='differential_evolution')
        output2, mini2, res = lmfit_uni_var(x_mean, x_var, y_mean, y_var, output.params, func, resid_func, cb_iter_func,
            emcee_steps=emcee_steps,
            lokalFit=True,
            specialParameters=specialParameters)

        output2.method = str(output.method) + " -> " + str(output2.method)
        output2.nfev = str(output.nfev) + " -> " + str(output2.nfev)

        if res != None:
            output2.method += " -> emcee"
            output2.nfev += " -> " + str(emcee_steps)

        if output.errorbars and not output2.errorbars:
            if res == None:
                print2("[INFO] Only globael estimation for std errors.",filename)
            for para in output2.params:
                if output2.params[para].vary:
                    output2.params[para].stderr = output.params[para].stderr
                    output2.params[para].correl = output.params[para].correl

        output = output2
        mini = mini2

    else:
        print2("[INFO] lokalFit",filename)
        mini = lmfit.Minimizer(residual, para, nan_policy='propagate', fcn_args=(func, x_mean, x_var, y_mean, y_var, specialParameters), iter_cb = cb_iter)
        output = mini.minimize(method='leastsq')
        if emcee_steps != 0:
            if emcee_steps > 500:
                burn = 300
                thin = 20
            else:
                burn = 0
                thin = 1
            res = lmfit.minimize(residual, method='emcee', nan_policy='omit',
                params=output.params,
                burn=burn,
                thin=thin,
                steps=emcee_steps,
                progress=True,
                args=(func, x_mean, x_var, y_mean, y_var, specialParameters))

            # https://lmfit.github.io/lmfit-py/fitting.html#lmfit.minimizer.Minimizer.emcee
            # https://lmfit.github.io/lmfit-py/confidence.html#label-confidence-advanced
            # https://www.sternwarte.uni-erlangen.de/wiki/index.php/Emcee#Acceptance_rate
            # https://iopscience.iop.org/article/10.1086/670067/pdf

            # Acceptance_rate: "This is `fraction of proposed steps [of the walkers] that are accepted.".
            # These steps are where the walkers did __not__ move back to its previous position


            acceptance_fraction = np.mean(res.acceptance_fraction)
            if not (acceptance_fraction >= 0.2 and acceptance_fraction <= 0.5):
                print2("[INFO] mean(acceptance_fraction) = " + str(acceptance_fraction),filename)
        else:
            res = None

    return output, mini, res

