"""
Content:
Picture showing the different fak values for different gammas under use of 
different neighborhood conditions.

Tags:
ca3d, fak, neighborhood, s, kappa
"""

import sys
sys.path.insert(0,'..')
from tools import *
import matplotlib.pyplot as plt
import numpy as np

gamma = [0.5*6.022e-06,6.022e-06,2.*6.022e-06]

data = [
    {
        "label": "moor2",
        "marker": "X",
        "color" : "gray",
        "values": [2.867569277517932*0.9423965851891873, 2.6854862319295183, 2.867569277517932*0.9501629007657038],
    },{
        "label": "neum2moor2",
        "marker": "p",
        "color" : "green",
        "values": [2.0394729888719265*0.9958915464162437, 2.0394729888719265, 2.0394729888719265*1.0199616357614123],
    },{
        "label": "neum2",
        "marker": "*",
        "color" : "purple",
        "values": [1.569947820463582*0.9935164004928739, 1.569947820463582, 1.569947820463582*0.9556679959089046],
    },{
        "label": "moor1",
        "marker": "s",
        "color" : "red",
        "values": [1.3917102419820815*1.036491425637936, 1.3917102419820815, 1.3917102419820815*1.0247896231659037],
    },{
        "label": "neum1moor1",
        "marker": "v",
        "color" : "blue",
        "values": [1.1761723220696105*1.0251979021397992, 1.1761723220696105, 1.1761723220696105*0.9842698803631841],
    },{
        "label": "neum1",
        "marker": "o",
        "color" : "black",
        "values": [0.8482288386294038*1.0203686520161266, 0.8482288386294038, 0.8482288386294038*0.9588206972126655],
    }
]

plt.figure()

for i in range(len(data)):
    #plt.plot(gamma, data[i]["values"], marker=data[i]["marker"], color=data[i]["color"], label=data[i]["label"], linestyle=None)
    plt.scatter(gamma, data[i]["values"], marker=data[i]["marker"], color=data[i]["color"], label=data[i]["label"])

plt.ylabel(r"scaling factor $\kappa$")
plt.xlabel(r"proliferation rate $\gamma$")
plt.legend(bbox_to_anchor=(0.5, 0., 0.5, 0.5))
plt.savefig(getPath()["bilder"] + "ca3d_fak" + ".pdf",transparent=True)
plt.show()