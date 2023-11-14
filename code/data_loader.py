"""
Content:
Funktionen zum laden der verschiedenen Experimentaldaten.

Tags:
data, load, numpy, npy, npz
"""

import numpy as np
import h5py
from CONSTANTS import *
from datetime import datetime

class data_loader:
    def __init__(self):
        pass

    def load_multiple_hdf5_oncoray(self, plates, multipX = None, multipY = None,yFormat="v0",xFormat="d"):
        if multipX is None:
            multipX = np.ones(len(path))
        if multipY is None:
            multipY = np.ones(len(path))

        resolution = 2.04 # "AxioVision"

        onlyControlled = False
        ignoreDamaged = False
        prefManualLabel = False
        if onlyControlled:
            print("[*] Only controlled selected")
        if ignoreDamaged:
            print("[*] ignoring damaged spheroids")
        if prefManualLabel:
            print("[*] prefere the manual label over label")
            # labels includes manualLabels. Labels has the same numbers of dots and interpolate the manual labels

        def compute_diameter_from_label(label):
            # returns min diameter, max diameter, mean diameter
            center = np.mean(label, axis=0)
            r = np.sqrt((label[:, 0]-center[0])**2+(label[:, 1]-center[1])**2)

            return 2*np.amin(r), 2*np.amax(r), 2*np.mean(r)

        unique_days = []
        datapoints = []
        x_all_raw = []
        y_all_raw = []
        data_labels = []

        for i in range(len(plates)):
            filename = plates[i].split("/")[-1]
            cellline = plates[i].split("/")[-2]
            filename = cellline + "/" + filename
            file_path = plates[i][:-len(filename)]

            with h5py.File(file_path + "ID_20190315.hdf5", "r") as f:
                for well in f.get(filename).keys():
                    data_well = f.get(filename).get(well)
                    x_raw = []
                    y_raw = []

                    if ("controlled" in data_well.attrs.keys() and data_well.attrs["controlled"] == "Y") or not onlyControlled:
                        for day in data_well:
                            data_day = data_well.get(day)

                            if not "label" in data_day.keys() and not "manual_label_pts" in data_day.keys():
                                print("[*] keine labels gefunden at " + well + " " + day)
                                V = 0.
                            else:
                                if "manual_label_pts" in data_day.keys() and prefManualLabel:
                                    label = data_day.get("manual_label_pts")
                                else:
                                    label = data_day.get("label")

                                ################################################

                                _,_,d = compute_diameter_from_label(label)
                                V = 4./3*np.pi*(0.5*d)**3
                                V *= resolution**3

                            ####################################################

                            if "spheroid_existence" in data_day.get("label").attrs.keys() and data_day.get("label").attrs["spheroid_existence"] == "N":
                                V = 0.
                            elif not "spheroid_existence" in data_day.get("label").attrs.keys():
                                print("[*] keine 'spheroid_existence' gefunden at " + well + " " + day)
                            #"""
                            if "damaged_spheroid" in data_day.get("label").attrs.keys() and data_day.get("label").attrs["damaged_spheroid"] == "Y":
                                V = 0.
                            elif not "damaged_spheroid" in data_day.get("label").attrs.keys():
                                print("[*] keine 'damaged_spheroid' gefunden at " + well + " " + day)
                            #"""
                            ####################################################

                            if not ("damaged_spheroid" in data_day.get("label").attrs.keys() and data_day.get("label").attrs["damaged_spheroid"] == "Y" and ignoreDamaged):
                                dt_day = datetime.strptime(day,"%d.%m.%Y")

                                if not dt_day in unique_days:
                                    unique_days.append(dt_day)
                                    datapoints.append(np.array([V*multipY[i]]))
                                else:
                                    idx = unique_days.index(dt_day)
                                    datapoints[idx] = np.append(datapoints[idx],V*multipY[i])

                                x_raw.append(dt_day)
                                y_raw.append(V)
                    x_all_raw.append(x_raw)
                    y_all_raw.append(np.array(y_raw)*multipY[i])
                    data_labels.append(well)

        buff = list(zip(*sorted(zip(unique_days,datapoints))))
        unique_days = list(buff[0])
        datapoints = list(buff[1])

        min_day = unique_days[0]

        for j in range(len(x_all_raw)):
            x_raw = x_all_raw[j]
            for i in range(len(x_raw)):
                x_raw[i] = (x_raw[i] - min_day).days * multipX
            x_all_raw[j] = np.array(x_raw)
            buff = list(zip(*sorted(zip(x_all_raw[j],y_all_raw[j]))))
            x_all_raw[j] = np.array(buff[0])
            y_all_raw[j] = np.array(buff[1])

        unique_days = np.array([(d - min_day).days * multipX[0] for d in unique_days])

        #"""
        if plates[0].endswith("plate110"):
            print("[*] extra fix for plate 110")
            print("[*] FIXME: Daten bis Tag 47 zeigen?")
            idx1 = np.where(unique_days == 17)[0][0]
            idx2 = np.where(datapoints[idx1] == np.max(datapoints[idx1]))[0][0]
            datapoints[idx1][idx2] = 0.

            unique_days = unique_days[:15]
            datapoints = datapoints[:15]
            
            #47 Days
            #unique_days = unique_days[:21]
            #datapoints = datapoints[:21]

        if plates[0].endswith("plate108"):
            print("[*] extra fix for plate 108")
            unique_days = unique_days[:15]
            datapoints = datapoints[:15]
        
        if plates[0].endswith("plate122"):
            print("[*] extra fix for plate 122")
            unique_days = unique_days[:15]
            datapoints = datapoints[:15]
        #"""

        x_mean = unique_days
        x_var = np.ones(len(x_mean)) * 0.25
        y_mean = np.array([np.mean(y) for y in datapoints])
        y_var = np.array([np.var(y) for y in datapoints])

        return x_mean, x_var, y_mean, y_var, [x_all_raw, y_all_raw,data_labels]

    def load_hdf5_oncoray(self, path, multipX = None, multipY = None,yFormat="v0",xFormat="d"):

        if multipY is not None:
            multipY = np.array([multipY])
        if multipX is not None:
            multipX = np.array([multipX])

        x_mean, x_var, y_mean, y_var, raw = self.load_multiple_hdf5_oncoray([path],multipX,multipY)

        return x_mean, x_var, y_mean, y_var, raw

    def load_numpy_Gri_exp(self, path, multipX=1., multipY=1.,yFormat="v0",xFormat="d"):

        data = np.loadtxt(path) * [multipX, multipY]

        x_mean = np.array([g[0] for g in data])
        y_mean = np.array([g[1] for g in data])

        if yFormat == "d0":
            y_mean = 4.*np.pi / 3. * (0.5*y_mean)**3
        if xFormat == "h":
            x_mean = x_mean/24.

        return x_mean, y_mean

    def load_numpy_Gri(self, path, multipX=1., multipY=1.,yFormat="v0",xFormat="d"):
        data = np.loadtxt(path) * [multipX, multipY]
        #x_mean = np.flip(data[:,0][::5])
        #y_mean = np.flip(data[:,1][::5])
        x_mean = data[:,0][::5]
        y_mean = data[:,1][::5]
        #x_var = np.flip(((np.abs(data[:,0][2::5] - data[:,0][::5]) + np.abs(data[:,0][4::5] - data[:,0][::5])) / 2.)**2)
        #y_var = np.flip(((np.abs(data[:,1][1::5] - data[:,1][::5]) + np.abs(data[:,1][3::5] - data[:,1][::5])) / 2.)**2)
        x_var = ((np.abs(data[:,0][2::5] - data[:,0][::5]) + np.abs(data[:,0][4::5] - data[:,0][::5])) / 2.)**2
        y_var = ((np.abs(data[:,1][1::5] - data[:,1][::5]) + np.abs(data[:,1][3::5] - data[:,1][::5])) / 2.)**2

        if yFormat == "d0":
            y_var = (4.*np.pi / 3. * (0.5*(np.sqrt(y_var)+y_mean))**3 - 4.*np.pi / 3. * (0.5*y_mean)**3)**2
            y_mean = 4.*np.pi / 3. * (0.5*y_mean)**3
        if xFormat == "h":
            x_var = ((x_mean + np.sqrt(x_var))/24. - x_mean/24.)**2
            x_mean = x_mean/24.

        return x_mean, x_var, y_mean, y_var, None

    def load_multiple_numpy_Gri(self, path, multipX=None, multipY=None,yFormat="v0",xFormat="d"):
        # just a fake function
        if multipX is None:
            multipX = np.ones(1)
        if multipY is None:
            multipY = np.ones(1)

        return self.load_numpy_Gri(path[0], multipX[0], multipY[0],yFormat=yFormat,xFormat=xFormat)

    def load_numpy_oncoray(self, path, multipX = None, multipY = None):
        return self.load_multiple_numpy_oncoray([path], np.array([multipX]), np.array([multipY]))

    def load_multiple_numpy_oncoray(self, path, multipX = None, multipY = None,yFormat="v0",xFormat="d"):
        import warnings
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")

        if multipX is None:
            multipX = np.ones(len(path))
        if multipY is None:
            multipY = np.ones(len(path))

        maxD = 30 # maximal length of exp. day Data
        data_all = np.ones((len(path),maxD+1,2)) * np.nan
        data_all[:,:,0] = np.arange(0,maxD+1,1)

        for i in range(len(path)):
            data = np.load(path[i] + ".npy")[:,[0,3]] * [multipX[i], multipY[i]]
            data_all[i,np.array(data[:,0],dtype=np.int),1] = data[:,1]

        y_var = np.nanvar(data_all[:,:,1],axis=0)
        y_mean = np.nanmean(data_all[:,:,1],axis=0)
        x_mean = data_all[0,:,0]

        x_mean = x_mean[np.invert(np.isnan(y_mean))]
        x_var = np.ones(len(x_mean)) * 0.25
        y_var = y_var[np.invert(np.isnan(y_mean))]
        y_mean = y_mean[np.invert(np.isnan(y_mean))]

        return x_mean, x_var, y_mean, y_var, None
