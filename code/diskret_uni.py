"""
Content:
Sammlung von diskreten Modellen für die Simulaiton von Tumorwachstum.

Tags:
diskret, Gri2016, funktion, plot
"""

import numpy as np
from ode_uni import ode_models
from scipy.optimize import curve_fit

class diskret_uni:
    def diskrete_Gri2016(x_mean, param):

        def sigmoid(x_mean, L ,x0, k):
            y_mean = L / (1 + np.exp(-k*(x_mean-x0)))
            return (y_mean)

        td = param["td"].value
        v = param["v0"].value

        t = [0]
        y_mean = np.array([v])
        if td == 0:
            return [0 for i in range(len(x_mean))]
        steps = int(np.ceil(x_mean[-1] / td))

        for i in range(steps):
            _, rn, rh, r0 = ode_models.ode_Gri2016(i,[v,0,0,0],param)
            v = 4. * np.pi * (2*r0**3 - rh**3 - rn**3) / 3.
            y_mean = np.append(y_mean,v)
            t.append((1+i) * td)

        try:
            sigPara, _ = curve_fit(sigmoid, t, y_mean, method='lm', maxfev = 10000)
            y_mean = sigmoid(x_mean, sigPara[0], sigPara[1], sigPara[2])
        except Exception as e:
            y_mean = np.zeros(len(x_mean))
        return y_mean
