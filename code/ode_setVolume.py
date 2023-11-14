import numpy as np

def setVolume_2(zielV, gridPoints, params, fak=1.):
    m = 1. if not "m" in params else params["m"].value
    dr = params["dr"].value / m
    zielV /= m**3

    vmax = 4. * np.pi * ((np.arange(1,gridPoints+1,1)*dr)**3 - (np.arange(0,gridPoints,1)*dr)**3) / 3.

    save = 0
    val = 100000
    for offset in np.arange(-gridPoints*10,20*gridPoints,1):
        x = np.arange(0,gridPoints,1) * fak - offset*0.1
        x = np.where(x < 709,x,709)
        c0 = 1. / (1. + np.exp(x))
        v = np.sum(vmax * c0)
        if abs(v-zielV) < val:
            val = abs(v-zielV)
            save = offset*0.1
    x = np.arange(0,fak*gridPoints,fak) - save
    x = np.where(x < 709,x,709)

    return 1. / (1. + np.exp(x))

def setVolume_3(parameters, t, model):

    import models
    from o2_models import o2_models
    from copy import deepcopy
    import sys

    if model.endswith("_RT"):
        mode = "RT"
    elif model.endswith("_FG"):
        mode = "FG"
    else:
        print("[*] Error: can not find model mode (FG/RT)" )
        sys.exit()

    MyClass = getattr(models,model)
    instance = MyClass()

    #parameters["dr"].value = 5.87325181

    dr = parameters["dr"].value
    m = parameters["m"].value
    v0 = parameters["v0"].value * 0.90
    #v0 = parameters["v0"].value * 0.20
    fak = 1. if not "og_dr" in parameters else dr / parameters["og_dr"].value
    rmax = parameters["rmax"].value if "rmax" in parameters else 1.1e-3 * m
    gridPoints = parameters["gridPoints"].value if "gridPoints" in parameters else int(round(rmax/dr,2))

    c0 = setVolume_2(v0, gridPoints, parameters,fak)[:gridPoints]
    rad = np.array(o2_models.sauerstoffdruck_Gri2016(v0, parameters))
    c1 = setVolume_2(4. * rad[0]**3 * np.pi / 3., gridPoints, parameters,fak)[:gridPoints]
    c1 = np.zeros((len(c0)))
    c0 -= c1
    c = np.append(c0, c1)

    parameters = deepcopy(parameters)

    if mode == "RT":
        c = np.append(c, np.zeros(len(c0)))
        if "dose" in parameters:
            parameters["dose"].value = 0
        if "p_mit_cata" in parameters:
            parameters["p_mit_cata"].value = 0

    specialParameters={"c":c}
    parameters["dr"].vary = False
    parameters["v0"].vary = False

    def terminator(t, y, *args):
        val = instance.getMetrics(args[0], np.array([y]))[0]
        ret = args[0]["v0"].value - val[0] * args[0]["m"].value**3
        return ret

    terminator.terminal = True
    terminator.direction = -1
    arg = instance.getValue(parameters, t, specialParameters=specialParameters, event_f=terminator)
    #arg = instance.getValue(parameters, 2.*t, specialParameters=specialParameters,event_f=terminator)
    return arg[2]
