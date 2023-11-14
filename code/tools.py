def getPath():
    import os
    path = os.getcwd()
    path = path.split("/")
    path.reverse()
    root = ""
    for i in path:
        if i == "ca3d":
            break
        else:
            root += "../"

    paths = {
        "root" : root,
        "bilder" : root + "bilder/",
        "gif" : root + "gif/",
        "log" : root + "gif/log/",
        "datadump" : root + "gif/datadump/",
        "code" : root + "code/",
        "ca3d" : root + "ca3d/v2/"
    }

    return paths

def getVhFromV0(y_mean, y_var, lmfit_parameters, outputFormat = "v"):
    from o2_models import o2_models
    import numpy as np

    m = lmfit_parameters["m"].value

    y_var = np.array([o2_models.sauerstoffdruck_Gri2016(((y_mean[i] + np.sqrt(y_var[i]))*m**3),lmfit_parameters)[1] for i in range(len(y_mean))])/m
    y_mean = np.array([o2_models.sauerstoffdruck_Gri2016(v*m**3,lmfit_parameters)[1] for v in y_mean])/m
    if outputFormat == "v":
        y_mean = 4. * np.pi * y_mean**3 / 3.
        y_var = (4. * np.pi * y_var**3 / 3. - y_mean)**2
    elif outputFormat == "r":
        y_var = (y_mean - y_var)**2

    return y_mean, y_var

def getVnFromV0(y_mean, y_var, lmfit_parameters, outputFormat = "v"):
    from o2_models import o2_models
    import numpy as np

    m = lmfit_parameters["m"].value

    y_var = np.array([o2_models.sauerstoffdruck_Gri2016(((y_mean[i] + np.sqrt(y_var[i]))*m**3), lmfit_parameters)[0] for i in range(len(y_mean))])/m
    y_mean = np.array([o2_models.sauerstoffdruck_Gri2016(v*m**3, lmfit_parameters)[0] for v in y_mean])/m
    if outputFormat == "v":
        y_mean = 4. * np.pi * y_mean**3 / 3.
        y_var = (4. * np.pi * y_var**3 / 3. - y_mean)**2
    elif outputFormat == "r":
        y_var = (y_mean - y_var)**2

    return y_mean, y_var

def convertVolumnRadius(y_mean, y_var, inputFormat):
    import sys
    import numpy as np

    if inputFormat == "r":
        y_var = y_mean + np.sqrt(y_var)
        y_var = 4. * y_var**3 * np.pi / 3.
        y_mean = 4. * y_mean**3 * np.pi / 3.
        y_var = (y_var - y_mean)**2

    elif inputFormat == "v":
        y_var = y_mean + np.sqrt(y_var)
        y_var = (3. * y_var / (4. * np.pi)) **(1./3.)
        y_mean = (3. * y_mean / (4. * np.pi)) **(1./3.)
        y_var = (y_var - y_mean)**2

    else:
        print("[!] Funktion 'convertVolumnRadius' hat keine gültiges inputFormat gefunden ('r','v')")
        sys.exit()

    return y_mean, y_var

def loadSimulations(simFileName, expFileName, cellline, plate = "", therapyType = "", datasource = "", smoothing=None):
    import sys
    from data_loader import data_loader
    import numpy as np
    from scipy import interpolate

    experimentsPaths = loadDatasources(datasource=datasource)
    data_loader_obj = data_loader()
    foundOneExp = False

    for expPath in experimentsPaths:
        expData = loadYaml(expPath + 'expConfic.yaml')
        for expFile in expData:
            if expFileName == expFile:
                if expData[expFile]["plate"] == plate or plate == "":
                    if expData[expFile]["cellline"] == cellline and (expData[expFile]["therapy"]["type"] == therapyType or therapyType == ""):
                        foundOneExp = True
                        multipX = expData[expFile]["parameters"]["multipX"]["value"]
                        multipY = expData[expFile]["parameters"]["multipY"]["value"]
                        yFormat = "v0" if "v0" in expData[expFile]["parameters"] else "d0"
                        xFormat = expData[expFile]["parameters"]["tmax"]["unit"]
                        xOffset = 0 if not "offset" in expData[expFile]["parameters"]["tmax"] else expData[expFile]["parameters"]["tmax"]["offset"]
                        break
        if foundOneExp:
            break

    if not foundOneExp:
        print("[!] No experimental data found!")
        print("[!] check if the right plate is used.")
        sys.exit()

    x_mean, y_mean = data_loader_obj.load_numpy_Gri_exp(path = expPath + simFileName, multipX = multipX, multipY = multipY, yFormat=yFormat, xFormat=xFormat)
    x_mean += xOffset

    if not smoothing is None:
        tck = interpolate.splrep(x_mean, y_mean, k=3)
        x_mean = smoothing
        y_mean = interpolate.splev(x_mean, tck)

    return x_mean, y_mean

def loadMultiExperiments(cellline, expFileName = "", plate = "", therapyType = "", datasource = ""):
    import sys
    from data_loader import data_loader
    import numpy as np

    experimentsPaths = loadDatasources(datasource=datasource)
    data_loader_obj = data_loader()

    multipX = np.array([])
    multipY = np.array([])
    filenames = []
    foundOneExp = False

    for expPath in experimentsPaths:
        expData = loadYaml(expPath + 'expConfic.yaml')
        for expFile in expData:
            if expFileName == "" or expFileName == expFile:
                if expData[expFile]["plate"] == plate or plate == "":
                    if expData[expFile]["cellline"] == cellline and (expData[expFile]["therapy"]["type"] == therapyType or therapyType == ""):
                        foundOneExp = True
                        filenames.append(expPath + expFile)
                        multipX = np.append(multipX, expData[expFile]["parameters"]["multipX"]["value"])
                        multipY = np.append(multipY, expData[expFile]["parameters"]["multipY"]["value"])
                        filetype = expData[expFile]["filetyp"]
                        # Dirty: Anpassen für multiple Experiments
                        yFormat = "v0" if "v0" in expData[expFile]["parameters"] else "d0" # Fix Me to unit
                        xFormat = expData[expFile]["parameters"]["tmax"]["unit"]
                        xOffset = 0 if not "offset" in expData[expFile]["parameters"]["tmax"] else expData[expFile]["parameters"]["tmax"]["offset"]

    if not foundOneExp:
        print("[!] No experimental data found!")
        print("[!] check if the right plate is used.")
        sys.exit()

    methode = getattr(data_loader_obj,"load_multiple_" + filetype)
    x_mean, x_var, y_mean, y_var, raw = methode(filenames, multipX, multipY, yFormat=yFormat, xFormat=xFormat)

    return x_mean + xOffset, x_var, y_mean, y_var, raw

def loadParametersFromLog(filename, lmfit_parameters = None):
    import lmfit
    import re

    path = getPath()["log"]

    if filename[-4:] != ".txt":
        filename += ".txt"

    values = {}
    if lmfit_parameters == None:
        lmfit_parameters = lmfit.Parameters()
    file = open(path + filename,"r")
    lines = file.readlines()
    lines.reverse()

    for line in lines:
        x = re.search("    .+: +[0-9]+(\.[0-9]+)?(e[+-]{1}[0-9]{2})?",line)
        if x is not None:
            idx_dp = line.index(":")
            variable = line[4:idx_dp]
            value = float(re.search(" [0-9]+(\.[0-9]+)?(e[+-]{1}[0-9]{2})?", str(x.group())).group()[1:])
            if not variable in lmfit_parameters:
                lmfit_parameters.add(variable,value=value,vary=False)
            else:
                lmfit_parameters[variable].value = value
            values[variable] = value

    return values, lmfit_parameters

def therapy_RT(lmfit_parameters, c, dose, celltypes = 3):
    import numpy as np
    from o2_models import o2_models

    gammaR = lmfit_parameters["gammaR"].value
    alphaR = lmfit_parameters["alphaR"].value
    betaR = lmfit_parameters["betaR"].value
    m = 1. if not "m" in lmfit_parameters else lmfit_parameters["m"].value
    dr = lmfit_parameters["dr"].value / m
    if "dr_og" in lmfit_parameters:
        dr_single = lmfit_parameters["dr_og"].value / m
    else:
        dr_single = lmfit_parameters["dr"].value / m
    v_single = dr_single**3

    n = int(len(c)/3.)
    vmax = 4. * np.pi * ((np.arange(1,n,1)*dr)**3 - (np.arange(0,n-1,1)*dr)**3) / 3.

    c = np.split(c,celltypes)
    c0 = c[0]
    c1 = c[1]
    if celltypes == 3:
        c2 = c[2]
    c_new = np.zeros((3,len(c0)))
    c_new[0] = c0
    c_new[1] = c1

    args_o2 = o2_models.sauerstoffdruck_analytical(params=lmfit_parameters,c0=c0,mode=2)
    p = args_o2[0]

    d_oer = np.ones(len(c0))
    d_oer[p<=11.] = 3. - (2.*p[p<=11.] / 11.)
    d_oer = dose / d_oer
    s_rt = np.exp(-gammaR * (alphaR * d_oer + betaR * d_oer**2))

    c_new[2] = (1.-s_rt) * c_new[0]
    if celltypes == 3:
        c_new[2] += c2
    c_new[0] = s_rt * c_new[0]

    # Remove c0 if remaining volumn < 1 cell
    if dose > 0:
        bool = [vmax*(c_new[0][:-1]+c_new[0][1:])/2.<v_single]
        #bool = np.insert(bool,0,False)
        bool = np.insert(bool,-1,bool[0][-1])
        c_new[2][bool] += c_new[0][bool]
        c_new[0][bool] -= c_new[0][bool]

    return c_new.flatten()

def print2(msg, filename, mode = "a", logPath=None):
    print(msg)
    if logPath is None:
        path = getPath()["log"]
    else:
        path = logPath

    if filename != None:
        #"""
        f = open(path + filename + "_log.txt", mode)
        f.write(msg + "\n")
        f.close()
        #"""
        #with open(path + filename + "_log.txt", mode) as f:
        #    f.write(msg + "\n")
        #with open(path + filename + "_log.txt", "r") as f:
        #    last_line = f.readlines()[-1]
        #    print(last_line)

def flushLog(filename, logPath = None, force = False):
    from os.path import exists
    import sys
    if logPath is None:
        path = getPath()["log"]
    else:
        path = logPath

    if exists(path + filename + "_log.txt") and not force:
        inp = ""
        while not inp in ["Y","N","n","y","J","j"]:
            inp = input("Soll die Datei '" + filename + "_log.txt" + "' überschrieben werden [y/n]: ")
        if inp in ["N", "n"]:
            print("[+] Script aborded!")
            sys.exit()

    if filename != None:
        f = open(path + filename + "_log.txt", "w")
        f.write("")
        f.close()

def loadDatasources(datasource = ""):
    import yaml

    yaml_path = getPath()["code"]
    loader = yaml.SafeLoader

    with open(yaml_path + "experiments.yaml") as f:
        dict = yaml.load(f, Loader=loader)

    return [yaml_path + dict[key]["path"] for key in dict if datasource == "" or key == datasource]

def loadYaml(path):

    import re
    import yaml

    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    with open(path) as f:
        dict = yaml.load(f, Loader=loader)
    return dict

def loadParameters(expName, celllines, expData, vary = {}):
    skipThis = ["multipX", "multipY", "tmax"]
    pathGlobal = getPath()["code"]
    parameters = loadYaml(pathGlobal + "globalParameters.yaml")
    parameters = {**parameters, **celllines[expData[expName]["cellline"]]}
    parameters = {**parameters, **expData[expName]["parameters"]}

    for val in skipThis:
        if val in parameters:
            del parameters[val]

    for para in parameters:
        if "std" in parameters[para]:
            parameters[para]["vary"] = vary[para] if para in vary else True #if len(vary) == 0 else False
            parameters[para]["max"] = parameters[para]["value"] + 2.* parameters[para]["std"]
            parameters[para]["min"] = parameters[para]["min"] if parameters[para]["value"] < 2.* parameters[para]["std"] else parameters[para]["value"] - 2.* parameters[para]["std"]
        else:
            parameters[para]["vary"] = vary[para] if para in vary else False
            parameters[para]["max"] = 1 if parameters[para]["value"] == 0 else 2. * parameters[para]["value"]
            parameters[para]["min"] = 0

    parameters["rd"] = {}
    parameters["rd"]["vary"] = False
    parameters["rd"]["max"] = 1000
    parameters["rd"]["min"] = 0
    parameters["rd"]["value"] = 1000

    return parameters

def parametersToDict(parameters):
    dict = {}
    for para in parameters:
        dict[para] = {
            "value" : parameters[para].value,
            "min" : parameters[para].min,
            "max" : parameters[para].max,
            "vary" : parameters[para].vary
        }
    return dict

def prepareDictToLmfitParameters(dict):
    out = {}
    for k in dict:
        if not isinstance(dict[k], str):
            out[k] = {"value": dict[k]}
    return out

def dictToLmfitParameters(dict, parameters_lmfit = None):
    import lmfit

    if parameters_lmfit is None:
        parameters_lmfit = lmfit.Parameters()

    first_key = list(dict.keys())[0]
    if str(type(dict[first_key])) != "<class 'dict'>":
        dict = prepareDictToLmfitParameters(dict)

    for para in dict:
        if para in parameters_lmfit:
            parameters_lmfit[para].value = dict[para]["value"]
            if "min" in dict[para]:
                parameters_lmfit[para].min = dict[para]["min"]
                parameters_lmfit[para].max = dict[para]["max"]
                parameters_lmfit[para].vary = dict[para]["vary"]
        else:
            if "min" in dict[para]:
                parameters_lmfit.add(para,value=dict[para]["value"],min=dict[para]["min"],max=dict[para]["max"],vary=dict[para]["vary"])
            else:
                parameters_lmfit.add(para,value=dict[para]["value"],vary=False)

    for para in parameters_lmfit:
        if not para in dict:
            parameters_lmfit[para].vary = False

    return parameters_lmfit

def prettyDictPrint(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         prettyDictPrint(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))

def prettyPrintLmfitParameters(para, filename=None, logPath=None):
    for key in para:
        if filename is None:
            print("[" + key + "] " + str(para[key].value))
        else:
            print2("[" + key + "] " + str(para[key].value), filename, logPath=logPath)

def dictToLog(dict,filename, logPath=None):
    print2("[[Variables]]",filename, logPath=logPath)
    for key in dict:
        out = "    " + str(key) + ": " + str(dict[key])
        print2(out,filename, logPath=logPath)

def printFitResults(oldPara, fitPara, lmfitObj, filename, cellline):
    def printSpaces(n):
        out = ""
        for i in range(n):
            out += " "
        return out

    def formateNumbers(num):
        if num < 0.001:
            return "{:.3e}".format(num)
        if len(str(num)) < 8:
            return str(num)
        return str(round(num,8))

    if lmfitObj[2] != None:
        resPara = lmfitObj[2].params
    else:
        resPara = None

    print2("[[Custom Print Out]]",filename)
    print2("    model              = " + str(filename),filename)
    print2("    cellline           = " + str(cellline),filename)
    print2("[[Fit Statistics]]",filename)
    print2("    # fitting method   = " + str(lmfitObj[0].method),filename)
    print2("    # function evals   = " + str(lmfitObj[0].nfev),filename)
    print2("    # data points      = " + str(lmfitObj[0].ndata),filename)
    print2("    # variables        = " + str(lmfitObj[0].nvarys),filename)
    print2("    chi-square         = " + "{:.3e}".format(lmfitObj[0].chisqr),filename)
    print2("    reduced chi-square = " + "{:.3e}".format(lmfitObj[0].redchi),filename)
    print2("    Akaike info crit   = " + formateNumbers(lmfitObj[0].aic),filename)
    print2("    Bayesian info crit = " + formateNumbers(lmfitObj[0].bic),filename)
    print2("[[Variables]]",filename)
    cor = {}
    for para in fitPara:
        out = "    " + str(para) + ":" + printSpaces(8 - len(para)) + formateNumbers(fitPara[para].value)
        if fitPara[para].vary:
            if fitPara[para].stderr == None:
                if resPara == None:
                    val1 = " - "
                    val2 = " - "
                    source = ""
                else:
                    val1 = formateNumbers(resPara[para].value - resPara[para].stderr)
                    val2 = formateNumbers(resPara[para].value + resPara[para].stderr)
                    source = " (emcee)"
            else:
                val1 = formateNumbers(fitPara[para].value - fitPara[para].stderr)
                val2 = formateNumbers(fitPara[para].value + fitPara[para].stderr)
                source = ""

            out += " 1\u03C3 in [" + val1 + ", " + val2 + "]" + source + " "
            out += "(init = " + formateNumbers(oldPara[para].value) + " in [" + formateNumbers(fitPara[para].min) + "," + formateNumbers(fitPara[para].max) + "])"

            if fitPara[para].correl != None:
                corPara = fitPara[para].correl
            else:
                if resPara == None:
                    corPara = {}
                else:
                    corPara = resPara[para].correl

            if not para in cor and len(corPara) != 0:
                cor[para] = {}
            for item in corPara:
                if not (item in cor and para in cor[item]):
                    cor[para][item] = corPara[item]
        else:
            out += " (fixed)"
        print2(out,filename)

    if len(cor) != 0:
        print2("[[Correlations]]",filename)
        for item in cor:
            if len(cor[item]) != 0:
                for item2 in cor[item]:
                    print2("    C(" + str(item) + ", " + str(item2) + ") = " + str(round(cor[item][item2],3)),filename)

    if resPara != None:
        print2("[[emcee Parameters]]",filename)
        for para in resPara:
            if resPara[para].vary:
                print2("    " + str(para) + ":" + printSpaces(8 - len(para)) + formateNumbers(resPara[para].value),filename)
    print2("",filename)
    #print2(str(lmfitObj[0].residual),filename)
