"""
Content:
Trys to fit the parameters of model1 against exp.-data which get interpolated

Tags:
2 models, fit, parameters, interpolated
"""

import sys
sys.path.insert(0,'..')
from modelMappingFit_main import modelMappingFit
import numpy as np
from tools import *

m = 1e5
D_c = 3e-15 * m**2 # m^2/s 1e-10 cm^2/s == 1e-14 m^2/s AmeEdwAkbNad2021
D_c2 = 2e-15 * m**2
llambda = 4e-04/10. # migration rate
D_p = 2e-9 * m**2
a = 18.09 # mmHg/s
og_a = a
gamma = 0. # 1/s
doubt_max = 80. # h
doubt_min = 30. # h
dr = 18.2e-6 * m
og_dr = dr
v0 = 0.65e-10 * m**3# m^3
epsilon = 3.784e-05 # rate hypox -> nec
delta = 4.331e-10 # rate nec clear
pcrit = 0 # mmHg
ph = 0.5 # mmHg
p0 = 100. # mmHg

tmax = 12*24*60*60
max_iter_time = 60
rmax = 1.1e-3 * m

gridPoints = int(round(rmax/dr,2))

gamma = np.log(2) / (doubt_min + (doubt_max - doubt_min) / 2.) / 3600. # 1/s
gamma_min = np.log(2) / doubt_min / 3600. # 1/s
gamma_max = np.log(2) / doubt_max / 3600. # 1/s

#model = "distrubution_model" # Parameter werden gefittet
model = "slice_model_RT" # Parameter werden gefittet
cellline = "MDA_MB_468"
plate = "000" #107 oder 000
expFile = "Gri2016_Fig4c_exp"
therapyType = ""
filename = model + "_" + cellline + "_dr_12_2"
info = "Free Growth Fit of " + cellline + " in Bru2019 et al Data with volumn Fit"
info += "\nactually fitting radius r0 and rn and not v0 and vn"
info += "\ninclude v0 vary into fit"

saveFig = True
sendMail = True
dataDump = False
debugMode = False

dict = {
    "D_p": {"value" : D_p},
    "D": {"value" : D_p},
    "llambda": {"value" : llambda,"min":0.1*llambda,"max":2.*llambda,"vary":True},
    "a": {"value" : a, "min": 13.56, "max":22.61, "vary": True},
    "og_a": {"value" : og_a},
    "D_c": {"value" : D_c,"min":0.1*D_c,"max":10*D_c,"vary":False},
    "D_c2": {"value" : D_c2,"min":0.1*D_c2,"max":10*D_c2,"vary":False},
    "dr": {"value" : dr, "min": 12.e-6 * m, "max": 20. * dr, "vary":True},
    "og_dr" : {"value": og_dr},
    "gamma" : {"value" : gamma, "min":gamma_max, "max":gamma_min, "vary":True},
    "delta" : {"value" : delta,"min":0.1*delta,"max":100*delta,"vary":True},
    "epsilon" : {"value" : epsilon,"min":0.1*epsilon,"max":10*epsilon,"vary":True},
    "v0" : {"value" : v0, "min" : 0.2 * v0, "max": 1.2*v0, "vary" : True},
    "pcrit" : {"value" : pcrit},
    "ph" : {"value" : ph},
    "p0" : {"value" : p0},
    "m" : {"value" : m},
    "rd" : {"value" : 1000*m},
    "tmax" : {"value" : tmax},
    "max_iter_time" : {"value" : max_iter_time},
    "rmax" : {"value" : rmax},
    "gridPoints" : {"value" : gridPoints},
    "p_mit_cata" : {"value" : 0},
    "alphaR" : {"value" : 0},
    "betaR" : {"value" : 0},
    "gammaR" : {"value" : 0},
}

lmfit_parameters = dictToLmfitParameters(dict)

modelMappingFit(model, lmfit_parameters, cellline, expFile, plate, therapyType, filename, saveFig, dataDump, sendMail, info, debugMode=debugMode)
