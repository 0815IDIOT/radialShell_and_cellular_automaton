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
llambda = 5.862e-04 # migration rate
D_p = 2e-9 * m**2
a = 29.56658287 # mmHg/s
og_a = 22.1 # mmHg/s
gamma = 8.103e-06 # 1/s
doubt_max = 100. # h
doubt_min = 17.1 # h
dr = 2.0045619
og_dr = 18.4e-6 * m
v0 = 0.95e-11 * m**3# m^3
epsilon = 3.918e-05 # rate hypox -> nec
delta = 2.666e-06 # rate nec clear
pcrit = 0 # mmHg
ph = 0.5 # mmHg
p0 = 80. # mmHg

dose = 10. # Gy
t_therapy = np.array([3.])*24*60*60 # d
#dose = 0. # Gy
#t_therapy = np.array([18.])*24*60*60 # d
d_therapy = np.array([dose]) # Gy
p_therapy = None
alphaR = 0.5 # 1/Gy
betaR = 0.042  # 1/Gy**2
gammaR = 1.0 # unitless
p_mit_cata = 0.5 # prop. of mitotic catastrophy

#tmax = 3*24*60*60
#t_offset_rel = -4
#t_offset_abs = 0.
tmax = 18*24*60*60
t_offset_rel = 3 
t_offset_abs = 0.
max_iter_time = 60
rmax = 1.1e-3 * m
#"""
dose = 5 # Gy
tmax = 21*24*60*60
v0 = 1.06e-11 * m**3 # 5 Gy
#v0 = 0.79e-11 * 1e15 # 2 Gy
t_therapy = np.array([3,21])*24*60*60 # d
d_therapy = np.array([dose,0]) # Gy
#p_therapy = np.array([0.26707656,0.67051367])
p_therapy = np.array([0.28236428,0.66200936]) # p0 = 80
t_offset_rel = 0
#"""
gridPoints = int(round(rmax/dr,2))

#gamma = np.log(2) / (doubt_min + (doubt_max - doubt_min) / 2.) / 3600. # 1/s
gamma_min = np.log(2) / doubt_min / 3600. # 1/s
gamma_max = np.log(2) / doubt_max / 3600. # 1/s

#model = "distrubution_model" # Parameter werden gefittet
model = "slice_model_RT" # Parameter werden gefittet
cellline = "HCT_116"
plate = "000" #107 oder 000
expFile = "Bru2019_Fig6_5gy"
therapyType = ""
filename = model + "_" + cellline + "_" + expFile + "_dr_12_1_1_p0_80" # dr_9 is skipped! 
info = "Free Growth Fit of " + cellline + " in Bru2019 et al Data with volumn Fit and vary dr"
loadFile = ""
#loadFile = "slice_model_RT_HCT_116_Bru2019_Fig6_10gy_dr_11_1_2"

saveFig = True
sendMail = True
dataDump = True
debugMode = False

dict = {
    "D_p": {"value" : D_p},
    "D": {"value" : D_p},
    "llambda": {"value" : llambda},
    "a": {"value" : a},
    "og_a": {"value" : og_a},
    "D_c": {"value" : D_c},
    "D_c2": {"value" : D_c2},
    "dr": {"value" : dr, "min": 1. * dr, "max": 7.*dr, "vary":False},
    "og_dr" : {"value": og_dr},
    "gamma" : {"value" : gamma, "min":gamma_max, "max":gamma_min, "vary":False},
    "delta" : {"value" : delta},
    "epsilon" : {"value" : epsilon},
    "v0" : {"value" : v0, "min" : 0.8 * v0, "max": 2.0*v0, "vary" : True},
    "pcrit" : {"value" : pcrit},
    "ph" : {"value" : ph},
    "p0" : {"value" : p0},
    "m" : {"value" : m},
    "rd" : {"value" : 1000*m},
    "tmax" : {"value" : tmax},
    "t_offset_rel" : {"value" : t_offset_rel},
    "t_offset_abs" : {"value" : t_offset_abs},
    "max_iter_time" : {"value" : max_iter_time},
    "rmax" : {"value" : rmax},
    "gridPoints" : {"value" : gridPoints},
    "alphaR" : {"value" : alphaR,"min":0, "max":1.5 * alphaR,"vary":False},
    "betaR" : {"value" : betaR,"min":0, "max":6.*betaR,"vary":False},
    "gammaR" : {"value" : gammaR},
    "p_mit_cata" : {"value" : p_mit_cata,"min":0.,"max":1,"vary":False},
    "p_mit_cata1" : {"value" : 0.5,"min":0.,"max":1,"vary":False},
    "p_mit_cata2" : {"value" : 0.5,"min":0.,"max":1,"vary":False},
}

lmfit_parameters = dictToLmfitParameters(dict)
RT_parameters = {"t_therapy" : t_therapy, "d_therapy" : d_therapy, "p_therapy" : p_therapy}

modelMappingFit(model, lmfit_parameters, cellline, expFile, plate, therapyType, filename, saveFig, dataDump, sendMail, info, RT_parameters = RT_parameters, loadFile=loadFile, debugMode=debugMode)
