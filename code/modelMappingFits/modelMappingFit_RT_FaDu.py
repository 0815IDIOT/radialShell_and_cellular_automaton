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
llambda = 2.228e-04 # migration rate
D_p = 2e-9 * m**2
a = 9.84582328 # mmHg/s
og_a = 10.64 # mmHg/s
gamma = 5.350e-06 # 1/s
dr = 6.64271203
og_dr = 16.e-6 * m
v0 = 3.07e-11 * m**3 # m^3 # plate 110
#v0 = 2.62725690e-11 * m**3 # m^3 # plate 108
epsilon = 9.996e-05 # rate hypox -> nec
delta = 8.534e-06 # rate nec clear
pcrit = 0 # mmHg
ph = 0.5 # mmHg
p0 = 100. # mmHg

dose = 2.5 # Gy
#t_therapy = np.array([3])*24*60*60 # d
#d_therapy = np.array([dose]) # Gy
t_therapy = np.array([30])*24*60*60 # d
d_therapy = np.array([0]) # Gy
p_therapy = None
alphaR = 0.34998672 # 1/Gy # Daten von Steffens Fit -> sigma in [+- 0.04]
betaR = 0.07939085 # 1/Gy**2 # Daten von Steffens Fit -> sigma in [+- 0.004]
gammaR = 1.0 # unitless
p_mit_cata = 0.5 # prop. of mitotic catastrophy

#tmax = 3*24*60*60
#t_offset_rel = -2
tmax = 30*24*60*60
t_offset_rel = 1
max_iter_time = 60
rmax = 1.1e-3 * m
#"""
dose = 2.5 # Gy
tmax = 14*24*60*60 # 2.5 Gy
#tmax = 21*24*60*60 # 5 Gy
#tmax = 33*24*60*60 # 20 Gy
t_therapy = np.array([3,14])*24*60*60 # d # 2.5 Gy
#t_therapy = np.array([3,21])*24*60*60 # d # 5 Gy
#t_therapy = np.array([3,33])*24*60*60 # d # 20 Gy
d_therapy = np.array([dose,0]) # Gy
p_therapy = np.array([0.16226254,0.81761204])
t_offset_rel = 0
#"""
gridPoints = int(round(rmax/dr,2))

model = "slice_model_RT"
cellline = "FaDu"
plate = "111"
expFile = ""
therapyType = ""
filename = model + "_" + cellline + "_" + str(dose) + "Gy_dr_14_1_2" 
info = "RT Fit after 14d for " + cellline + " in " + expFile + " with " + str(dose) + "Gy"
loadFile = ""
#loadFile = "slice_model_RT_FaDu_20Gy_dr_11_1_2"

"""
000: Grimes
107: FaDu 0 Gy
111: FaDu 2.5 Gy
112: FaDu 5 Gy
122: FaDu 7.5 Gy
121: FaDu 10 Gy
110: FaDu 20 Gy
108: FaDu 25 Gy ?
"""

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
    "dr": {"value" : dr},
    "og_dr" : {"value": og_dr},
    "gamma" : {"value" : gamma},
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
    "max_iter_time" : {"value" : max_iter_time},
    "rmax" : {"value" : rmax},
    "gridPoints" : {"value" : gridPoints},
    "alphaR" : {"value" : alphaR,"min":0.25, "max":0.35,"vary":False},
    "betaR" : {"value" : betaR,"min":0.015, "max":0.08,"vary":False},
    "gammaR" : {"value" : gammaR},
    "p_mit_cata" : {"value" : p_mit_cata,"min":0.,"max":1,"vary":False},
    "p_mit_cata1" : {"value" : 0.5,"min":0.,"max":1,"vary":False},
    "p_mit_cata2" : {"value" : 0.5,"min":0.,"max":1,"vary":False},
}

lmfit_parameters = dictToLmfitParameters(dict)
RT_parameters = {"t_therapy" : t_therapy, "d_therapy" : d_therapy, "p_therapy" : p_therapy}

modelMappingFit(model, lmfit_parameters, cellline, expFile, plate, therapyType, filename, saveFig, dataDump, sendMail, info, RT_parameters = RT_parameters, loadFile=loadFile, debugMode=debugMode)
