"""
Content:
Simulation time comparison for the 3 different models. 

Tags:
plot, timing
"""
import sys
sys.path.insert(0,'..')
import numpy as np
import matplotlib.pyplot as plt
from tools import *
from CONSTANTS import *

print("[*] plot_volume_growth")

# [RealZeit, Simulationszeit, Volumen]
#time_ca = [[1,0,0],[2,0,0],[3,0,0],[4,0,0],[5,0,0],[6,0,0],[7,0,0],[8,0,0]]
time_ca = [[0,0,14363],
            [0.23,17.1,15588],
            [0.47,19.1,16942],
            [0.70, 24.40, 18619],
            [1.17,29.3,22315],
            [1.41,32.5,24024],
            [1.64, 33.3,26166],
            [1.88,39.4,28492],
            [2.11,42.6,30904],
            [2.35,54.2,33500],
            [2.58,61.9,36092],
            [2.81,68.1,38872],
            [3.05,73.8,42152],
            [3.28,75.2,45544],
            [3.52,86.6,48240],
            [3.75,103.3,51824],
            [3.99, 113.6,55816],
            [4.22, 109.9,59119],
            [5.16,137.8,76567],
            [6.01, 181.8, 97293],
            [7.04, 323.0,120769],
            [8.21, 357.5,154699]]
time_model1 = []
time_model2 = []

linewidth = 2.5
fontsizeLabel = 15
fontsizeTick = 12
fontsizeLegend = 11
figSave = True

model1 = "Schichtenmodel"
model2 = "slice_model_FG"

print("[+] Simulating " + model1)

time_model1 = [[0.00000000e+00, 0.00000000e+00, 1.96032857e+02],
    [2.30000000e-01, 5.54323196e-03, 2.00698683e+02],
    [4.70000000e-01, 6.22820854e-03, 2.05649014e+02],
    [7.00000000e-01, 6.51383400e-03, 2.10514720e+02],
    [1.17000000e+00, 1.13060474e-02, 2.20834238e+02],
    [1.41000000e+00, 6.53433800e-03, 2.26291564e+02],
    [1.64000000e+00, 4.45332527e-02, 2.31729238e+02],
    [1.88000000e+00, 4.07650471e-02, 2.37411449e+02],
    [2.11000000e+00, 4.50153351e-02, 2.42950300e+02],
    [2.35000000e+00, 4.88686562e-02, 2.48820981e+02],
    [2.58000000e+00, 4.71913815e-02, 2.54529199e+02],
    [2.81000000e+00, 4.35147285e-02, 2.60313141e+02],
    [3.05000000e+00, 4.56418991e-02, 2.66425705e+02],
    [3.28000000e+00, 4.58202362e-02, 2.72353836e+02],
    [3.52000000e+00, 3.96330357e-02, 2.78608794e+02],
    [3.75000000e+00, 4.11722660e-02, 2.84666548e+02],
    [3.99000000e+00, 4.55253124e-02, 2.91050617e+02],
    [4.22000000e+00, 4.23364639e-02, 2.97226277e+02],
    [5.16000000e+00, 3.79624367e-02, 3.23001532e+02],
    [6.01000000e+00, 5.55973053e-02, 3.46960900e+02],
    [7.04000000e+00, 4.58841324e-02, 3.76690985e+02],
    [8.21000000e+00, 4.86559868e-02, 4.11244297e+02]]

print("[+] Simulating " + model2)

time_model2 = [[0.00000000e+00, 0.00000000e+00, 1.96032857e+02],
    [2.30000000e-01, 5.92694283e-02, 1.93402257e+02],
    [4.70000000e-01, 8.02760124e-02, 1.97085051e+02],
    [7.00000000e-01, 9.79378223e-02, 2.01113265e+02],
    [1.17000000e+00, 1.20848656e-01, 2.10791201e+02],
    [1.41000000e+00, 1.54572487e-01, 2.16401880e+02],
    [1.64000000e+00, 2.72361994e-01, 2.22140921e+02],
    [1.88000000e+00, 3.58455181e-01, 2.28455497e+02],
    [2.11000000e+00, 3.93783808e-01, 2.34774436e+02],
    [2.35000000e+00, 4.86966372e-01, 2.41604204e+02],
    [2.58000000e+00, 6.42892599e-01, 2.48340417e+02],
    [2.81000000e+00, 6.27812624e-01, 2.55233697e+02],
    [3.05000000e+00, 7.77290821e-01, 2.62568859e+02],
    [3.28000000e+00, 7.71276474e-01, 2.69713285e+02],
    [3.52000000e+00, 8.89555931e-01, 2.77269108e+02],
    [3.75000000e+00, 1.00422454e+00, 2.84592795e+02],
    [3.99000000e+00, 1.10989738e+00, 2.92308764e+02],
    [4.22000000e+00, 1.04984832e+00, 2.99764400e+02],
    [5.16000000e+00, 1.27670765e+00, 3.30708035e+02],
    [6.01000000e+00, 1.48098469e+00, 3.59138197e+02],
    [7.04000000e+00, 1.80049658e+00, 3.93939196e+02],
    [8.21000000e+00, 2.00754428e+00, 4.33762532e+02]]

time_ca = time_ca[1:]
time_model1 = time_model1[1:]
time_model2 = time_model2[1:]

fig = plt.figure(figsize=(8.0,5.0))
ax = fig.add_subplot(111)

time_ca = np.array(time_ca)
time_ca[:,2] =  (3. / 4. / np.pi * time_ca[:,2])**(1. / 3.) * 13.e-6 * 1e6
time_model2 = np.array(time_model2)
time_model1 = np.array(time_model1)

#time_model1[:,1] *= 1000
#time_model2[:,1] *= 100

coef2 = np.polyfit(time_model2[:,2],time_model2[:,1],1)
poly1d_fn2 = np.poly1d(coef2)
plt.plot(time_model2[:,2][time_model2[:,2] > 300], poly1d_fn2(time_model2[:,2][time_model2[:,2] > 300]), color="black",linestyle="dotted")
plt.text(400,3,r"$\sim 0.0087 \cdot r$")

coef1 = np.polyfit(time_model1[:,2],time_model1[:,1],1)
poly1d_fn1 = np.poly1d(coef1)
plt.plot(time_model1[:,2][time_model1[:,2] > 300], poly1d_fn1(time_model1[:,2][time_model1[:,2] > 300]), color="black",linestyle="dotted")
plt.text(400,0.08,r"$\sim 0.0019 \cdot r$")
#print(coef1)

#numpy.polyfit(numpy.log(x), y, 1)
coef3 = np.polyfit(np.log(time_ca[:,2]),time_ca[:,1],1)
poly1d_fn1 = np.poly1d(coef3)
plt.plot(time_ca[:,2][time_ca[:,2] > 300], poly1d_fn1(np.log(time_ca[:,2][time_ca[:,2] > 300])), color="black",linestyle="dotted")
#print(coef3)
plt.text(400,130,r"$\sim \exp(r)$")

plt.plot(time_ca[:,2],time_ca[:,1], color=COLOR_PROLIFERATION, linewidth=linewidth, linestyle="-.", label="agent based")
plt.plot(time_model2[:,2],time_model2[:,1], color=COLOR_PROLIFERATION, linewidth=linewidth, linestyle="solid", label="RS-model")
plt.plot(time_model1[:,2],time_model1[:,1], color=COLOR_PROLIFERATION, linewidth=linewidth, linestyle="dashed", label="non-spatial model")

plt.xlabel('radius $r_0$ [$\mu m$]',fontsize=fontsizeLabel)
plt.ylabel('simulation time [$s$]',fontsize=fontsizeLabel)
plt.legend(fontsize=fontsizeLegend)
plt.setp(ax.get_xticklabels(), fontsize=fontsizeTick)
plt.setp(ax.get_yticklabels(), fontsize=fontsizeTick)
plt.yscale('log') 

if figSave:
    #plt.savefig(getPath()["bilder"] + "plot_volume_growth_1.pdf",transparent=True)
    plt.savefig(getPath()["bilder"] + "plot_volume_growth_1.pdf",transparent=False)
else:
    plt.show()