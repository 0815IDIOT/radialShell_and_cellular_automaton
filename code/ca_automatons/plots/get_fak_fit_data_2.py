"""
Content:
Read the log file and create the plot for the fak.

Tags:
ca3d, fak, kappa
"""

import sys
sys.path.insert(0,'../../../.')
sys.path.insert(0,'../.')
from tools import *
import numpy as np
import re
import matplotlib.pyplot as plt
from constants import *
from copy import deepcopy

logfile = "CA3D_Fak_4_5_log.txt" # FaDu incl. moor4; 6d
#logfile = "CA3D_Fak_6_log.txt" # SCC excl. moor4
#logfile = "CA3D_Fak_9_log.txt" # FaDu excl. moor4; 10d

neighborhoods = ["neum1", "neum1moor1", "moor1", "neum2", "neum2moor2", "neum3", "neum3moor2", "moor2", "neum3moor3", "moor3", "moor3moor4", "moor4"]
#neighborhoods = ["neum1", "neum1moor1", "moor1", "neum2", "neum2moor2", "neum3", "neum3moor2", "moor2", "neum3moor3", "moor3"]
markers = ["o", "v", "P", "p", "X", "D", "h", "^", "s", "*", "d", "<"]

fontsize_axis_label = 13

kappa_mean = []
kappa_max = []
kappa_min = []

with open(getPath()["log"] + logfile, "r") as f:
    lines = f.readlines()
    for line in lines:
        x = re.search("dr:[ ]+[0-9]+\.[0-9]+ 1σ in \[[0-9]+\.[0-9]+, [0-9]+\.[0-9]+\]",line)
        if x is not None:
            y = re.findall("[0-9]+\.[0-9]+",x.group(0))
            mean = float(y[0]) / 1.6
            mini = float(y[1]) / 1.6
            maxi = float(y[2]) / 1.6

            kappa_mean.append(mean)
            kappa_max.append(abs(maxi - mean))
            kappa_min.append(abs(mini - mean))

data = {}
for n in neighborhoods:
    data[n] = {
        "mean" : [],
        "var" : [[],[]]
    }

j = 0
for gamma in [0.5,1.,2.]:
    i = 0
    for n in neighborhoods:
        data[n]["mean"].append(kappa_mean[i + j * len(neighborhoods)])
        data[n]["var"][0].append(kappa_min[i + j * len(neighborhoods)])
        data[n]["var"][1].append(kappa_max[i + j * len(neighborhoods)])
        i += 1
    j += 1

gamma = [3.011e-06, 6.022e-06, 1.204e-05]
colors = []
means = []

plt.figure(figsize=(12.8, 4.8))
#fig = plt.figure()
#size = fig.get_size_inches()
#print(size)
plt.subplot(121)

i = 0
for n in reversed(neighborhoods):
    #print(data[n]["var"])
    #p = plt.errorbar(gamma, data[n]["mean"],data[n]["var"], None, label=n, linestyle="none", linewidth=3*0.5, capsize=5, marker = markers[i])
    p = plt.errorbar(gamma, data[n]["mean"], None, None, label=n, linestyle="none", linewidth=3*0.5, capsize=5, marker = markers[i])
    mean = np.mean(data[n]["mean"])
    var = np.var(data[n]["mean"])
    #print(n + ": " + str(round(mean,2)))
    print(n + ": $" + str(round(mean, 2)) + " \\pm " + str("{:.2e}".format(var)) + "$")
    means.append(mean)
    plt.hlines(mean,gamma[0],gamma[-1],color=p[0].get_color(),alpha=0.5)
    colors.append(p[0].get_color())
    #plt.plot(gamma, data[n]["mean"], label = n, marker = markers[i], linestyle = "none")
    i += 1

#plt.legend(bbox_to_anchor=(0.5, 0.25, 0.5, 0.5))
plt.title("(a)",loc="left")
plt.ylabel(r"scaling factor $\kappa$", fontsize=fontsize_axis_label)
plt.xlabel(r"proliferation rate $\gamma$", fontsize=fontsize_axis_label)

################################################################################

print("")

mean_dist = []

for n in reversed(neighborhoods):
    neigh1 = None
    neigh2 = None
    print(n)
    if "neum1" in n:
        if neigh1 is None:
            neigh1 = NEIGHBORHOOD_NEUMANN1
        else:
            neigh2 = NEIGHBORHOOD_NEUMANN1
    if "neum2" in n:
        if neigh1 is None:
            neigh1 = NEIGHBORHOOD_NEUMANN2
        else:
            neigh2 = NEIGHBORHOOD_NEUMANN2
    if "neum3" in n:
        if neigh1 is None:
            neigh1 = NEIGHBORHOOD_NEUMANN3
        else:
            neigh2 = NEIGHBORHOOD_NEUMANN3
    if "moor1" in n:
        if neigh1 is None:
            neigh1 = NEIGHBORHOOD_MOORE1
        else:
            neigh2 = NEIGHBORHOOD_MOORE1
    if "moor2" in n:
        if neigh1 is None:
            neigh1 = NEIGHBORHOOD_MOORE2
        else:
            neigh2 = NEIGHBORHOOD_MOORE2
    if "moor3" in n:
        if neigh1 is None:
            neigh1 = NEIGHBORHOOD_MOORE3
        else:
            neigh2 = NEIGHBORHOOD_MOORE3
    if "moor4" in n:
        if neigh1 is None:
            neigh1 = NEIGHBORHOOD_MOORE4
        else:
            neigh2 = NEIGHBORHOOD_MOORE4

    dos = [neigh1, neigh2]
    mean_rad = [0., 0.]
    cnt_rad = [0, 0]
    radiis = [[],[]]

    for d in range(2):
        if not dos[d] is None:
            for cell in dos[d]:
                radius = np.sqrt(cell[0]**2 + cell[1]**2 + cell[2]**2)
                radiis[d].append(radius)
                mean_rad[d] += radius
                cnt_rad[d] += 1

    if neigh2 is None:
        mean = mean_rad[0]/cnt_rad[0]
        print(str(mean))
        #print(str(mean_rad[0]) + " / " + str(cnt_rad[0]))
        #print(radiis[0])
        print("")
        mean_dist.append(mean)
    else:
        mean = (mean_rad[0]/cnt_rad[0] + mean_rad[1]/cnt_rad[1])/2.
        print(str(mean))
        #print(str(mean_rad[0]) + " / " + str(cnt_rad[0]) + " | " + str(mean_rad[1]) + " / " + str(cnt_rad[1]))
        #print(radiis[0])
        #print(radiis[1])
        print("")
        mean_dist.append(mean)

plt.subplot(122)

#plt.plot(mean_dist, means, "-", color="black")


fit_mean_dist = deepcopy(mean_dist)
fit_means = deepcopy(means)
w = np.ones(len(fit_mean_dist))
#fit_mean_dist.insert(0,0)
#fit_means.insert(0,0)
#w = np.ones(len(fit_mean_dist)+1) * 0.1
#w[0] = 1


lin_fit = np.polyfit(fit_mean_dist, fit_means, 1, w=w)

plt.plot(mean_dist, np.array(mean_dist) * lin_fit[0] + lin_fit[1], color="black", alpha=1.0)

print(lin_fit)

for i in range(len(means)):
    plt.plot(mean_dist[i], means[i], marker = markers[i], color = colors[i], label = neighborhoods[len(neighborhoods) - i - 1].replace("moor", "moore"))

plt.text(3., 3.5, r"$\kappa\approx" + str(round(lin_fit[0],2)) + r" d_\mathrm{neigh} " + str(round(lin_fit[1],2)) +  r"$", rotation=34)
plt.title("(b)",loc="left")
plt.legend()
plt.ylabel(r"scaling factor $\kappa$", fontsize=fontsize_axis_label)
plt.xlabel(r"mean cell distance in neighborhood $d_\mathrm{neigh}$", fontsize=fontsize_axis_label)
plt.tight_layout()

#plt.savefig(getPath()["bilder"] + "ca3d_fak" + ".pdf",transparent=True)
plt.savefig(getPath()["bilder"] + "ca3d_fak" + ".pdf",transparent=False)
plt.show()
