import numpy as np
from math import sqrt
from o2_models import o2_models
#import warnings
#warnings.filterwarnings("error")
#except RuntimeWarning:

class ode_models:
    def ode_Gri2016(t, v, params):

        vol = v[0]
        gamma = params["gamma"].value # 1 / s
        delta = params["delta"].value # 1 / s

        rn, rh, r0, rl, rs = o2_models.sauerstoffdruck_Gri2016(vol, params)
        dvdt = 4. * np.pi * (gamma * (r0**3 - rh ** 3) - delta * rn**3) / 3.

        return [dvdt, -v[1]+rn, -v[2]+rh, -v[3]+r0]

    def slice_model_7(t, c,params, dist=None, num=None):

        dr = params["dr"].value
        gamma = params["gamma"].value
        delta = params["delta"].value
        epsilon = params["epsilon"].value
        llambda = params["llambda"].value

        c = np.split(c,2)
        c0 = c[0]
        c1 = c[1]

        radius = dr * (np.arange(1,len(c0)+1,1) + np.arange(0,len(c0),1)) / 2.
        v = 4. * np.pi * ((np.arange(1,len(c0)+1,1) * dr)**3 - (np.arange(0,len(c0),1) * dr)**3) / 3.

        dcdt0 = np.zeros(len(c0))
        dcdt1 = np.zeros(len(c1))

        def migration(c,c_sum):
            buff_c = np.zeros(len(c)+2)
            buff_c[1:-1] = c
            buff_c[0] = 1
            buff_c[len(buff_c)-1] = 0

            buff_c_sum = np.zeros(len(c_sum)+2)
            buff_c_sum[1:-1] = c_sum
            buff_c_sum[0] = 1
            buff_c_sum[len(buff_c_sum)-1] = 0

            buff_v = np.zeros(len(v)+2)
            buff_v[1:-1] = v
            buff_v[0] = 1
            buff_v[len(buff_c)-1] = 1

            buff0 = buff_c[2:] * buff_v[2:] / buff_v[1:-1] * (1 - buff_c_sum[1:-1])
            buff0 -= buff_c[1:-1] * (1 - buff_c_sum[0:-2])
            buff0 *= llambda

            return buff0

        def prolif(c,c_sum, dist, num):
            buff_c = np.zeros(len(c)+4)
            buff_c[2:-2] = c
            buff_c[:2] = 0
            buff_c[-2:] = 0

            buff_c_sum = np.zeros(len(c_sum)+4)
            buff_c_sum[2:-2] = c_sum
            buff_c_sum[:2] = 0
            buff_c_sum[-2:] = 0

            buff_v = np.zeros(len(v)+4)
            buff_v[2:-2] = v
            buff_v[:2] = 0#1e8
            buff_v[-2:] = 0#1e8

            buff0 = gamma * buff_c[2:-2] * ((1 - buff_c_sum[2:-2]) * buff_v[2:-2]) / ((1 - buff_c_sum[1:-3]) * buff_v[1:-3] + (1 - buff_c_sum[2:-2]) * buff_v[2:-2] + (1 - buff_c_sum[3:-1]) * buff_v[3:-1] + ((buff_c_sum[1:-3] == 1) + 1 + (buff_c_sum[2:-2] == 1) + (buff_c_sum[3:-1] == 1) == 4))
            buff0 *= 1 - (buff_v[1:-3] * buff_c_sum[1:-3] + buff_v[2:-2] * buff_c_sum[2:-2] + buff_v[3:-1] * buff_c_sum[3:-1]) / (buff_v[1:-3] + buff_v[2:-2] + buff_v[3:-1])

            buff1 = gamma * buff_c[3:-1] * ((1 - buff_c_sum[2:-2]) * buff_v[2:-2]) / ((1 - buff_c_sum[2:-2]) * buff_v[2:-2] + (1 - buff_c_sum[3:-1]) * buff_v[3:-1] + (1 - buff_c_sum[4:]) * buff_v[4:] + ((buff_c_sum[2:-2] == 1) + 1 + (buff_c_sum[3:-1] == 1) + (buff_c_sum[4:] == 1) == 4))
            buff1 *= 1 - (buff_v[2:-2] * buff_c_sum[2:-2] + buff_v[3:-1] * buff_c_sum[3:-1] + buff_v[4:] * buff_c_sum[4:]) / (buff_v[2:-2] + buff_v[3:-1] + buff_v[4:])

            buff2 = gamma * buff_c[1:-3] * ((1 - buff_c_sum[2:-2]) * buff_v[2:-2]) / ((1 - buff_c_sum[:-4]) * buff_v[:-4] + (1 - buff_c_sum[1:-3]) * buff_v[1:-3] + (1 - buff_c_sum[2:-2]) * buff_v[2:-2] + ((buff_c_sum[:-4] == 1) + 1 + (buff_c_sum[1:-3] == 1) + (buff_c_sum[2:-2] == 1) == 4))
            buff2 *= 1 - (buff_v[:-4] * buff_c_sum[:-4] + buff_v[1:-3] * buff_c_sum[1:-3] + buff_v[2:-2] * buff_c_sum[2:-2]) / (buff_v[:-4] + buff_v[1:-3] + buff_v[2:-2])

            buff3 = gamma * buff_c[2:-2] * ((1 - buff_c_sum[3:-1]) * buff_v[3:-1]) / ((1 - buff_c_sum[1:-3]) * buff_v[1:-3] + (1 - buff_c_sum[2:-2]) * buff_v[2:-2] + (1 - buff_c_sum[3:-1]) * buff_v[3:-1] + ((buff_c_sum[1:-3] == 1) + 1 + (buff_c_sum[2:-2] == 1) + (buff_c_sum[3:-1] == 1) == 4))
            buff3 *= 1 - (buff_v[1:-3] * buff_c_sum[1:-3] + buff_v[2:-2] * buff_c_sum[2:-2] + buff_v[3:-1] * buff_c_sum[3:-1]) / (buff_v[1:-3] + buff_v[2:-2] + buff_v[3:-1])

            buff4 = gamma * buff_c[2:-2] * ((1 - buff_c_sum[1:-3]) * buff_v[1:-3]) / ((1 - buff_c_sum[1:-3]) * buff_v[1:-3] + (1 - buff_c_sum[2:-2]) * buff_v[2:-2] + (1 - buff_c_sum[3:-1]) * buff_v[3:-1] + ((buff_c_sum[1:-3] == 1) + 1 + (buff_c_sum[2:-2] == 1) + (buff_c_sum[3:-1] == 1) == 4))
            buff4 *= 1 - (buff_v[1:-3] * buff_c_sum[1:-3] + buff_v[2:-2] * buff_c_sum[2:-2] + buff_v[3:-1] * buff_c_sum[3:-1]) / (buff_v[1:-3] + buff_v[2:-2] + buff_v[3:-1])

            if num is not None and dist is not None:
                num += buff0 + buff3 + buff4
                dist += buff3 - buff4

            return buff0 + buff1 + buff2

        args_o2 = o2_models.sauerstoffdruck_analytical_5(params,c0,mode=2)
        rhypox = args_o2[1]
        rcrit = args_o2[2]

        ls =  np.arange(0,len(c0),1) * dr
        rs = np.arange(1,len(c0)+1,1) * dr

        rcrit = (rcrit - ls) / dr
        rcrit[rcrit<0] = 0
        rcrit[rcrit>1] = 1

        rhypox = (rhypox - ls) / dr
        rhypox[rhypox<0] = 0
        rhypox[rhypox>1] = 1
        rhypox = 1.-rhypox

        dcdt0 += prolif(c0,c0+c1, dist, num) * rhypox + migration(c0,c0+c1) - epsilon * c0 * rcrit
        dcdt1 += migration(c1,c0+c1) + epsilon * c0 * rcrit - delta * c1

        dcdt = np.append(dcdt0, dcdt1)
        #print(dcdt)
        #print("[" + str(round(t/24./3600.,20)) + "] ")
        #print("\r[" + str(round(t/24./3600.,20)) + "] " + "        ",end="")

        return dcdt

    def slice_model_7_RT(t, c, params, dist=None, num=None):

        dr = params["dr"].value
        gamma = params["gamma"].value
        delta = params["delta"].value
        epsilon = params["epsilon"].value
        llambda = params["llambda"].value
        p_mit_cata = params["p_mit_cata"].value
        faktor_logistic = 1. if not "faktor_logistic" in params else params["faktor_logistic"].value

        c = np.split(c,3)
        c0 = c[0]
        c1 = c[1]
        c2 = c[2]

        radius = dr * (np.arange(1,len(c0)+1,1) + np.arange(0,len(c0),1)) / 2.
        v = 4. * np.pi * ((np.arange(1,len(c0)+1,1) * dr)**3 - (np.arange(0,len(c0),1) * dr)**3) / 3.

        dcdt0 = np.zeros(len(c0))
        dcdt1 = np.zeros(len(c1))
        dcdt2 = np.zeros(len(c2))

        def migration(c, c_sum):
            """
            These boundary conditions lead to those described in the paper. 
            However, they are implemented here in a more technically efficient 
            way, so that only one version of the equation needs to be 
            implemented. This has the consequence that one arrives at the same 
            boundary condition by intellectually setting c_T and sum{c_T} 
            independently. 
            """
            buff_c = np.zeros(len(c)+2)
            buff_c[1:-1] = c
            buff_c[0] = 1
            buff_c[len(buff_c)-1] = 0

            buff_c_sum = np.zeros(len(c_sum)+2)
            buff_c_sum[1:-1] = c_sum
            buff_c_sum[0] = 1
            buff_c_sum[len(buff_c_sum)-1] = 0

            buff_v = np.zeros(len(v)+2)
            buff_v[1:-1] = v
            buff_v[0] = 1
            buff_v[len(buff_c)-1] = 1

            buff0 = buff_c[2:] * buff_v[2:] / buff_v[1:-1] * (1 - buff_c_sum[1:-1])
            buff0 -= buff_c[1:-1] * (1 - buff_c_sum[0:-2])
            buff0 *= llambda
            #buff0 *= 0.00140297
            #buff0 *= 0

            return buff0

        def prolif(c, c_sum, p):
            """
            These boundary conditions lead to those described in the paper. 
            However, they are implemented here in a more technically efficient 
            way, so that only one version of the equation needs to be 
            implemented. This has the consequence that one arrives at the same 
            boundary condition by intellectually setting c_T and sum{c_T} 
            independently. 
            """
            buff_c = np.zeros(len(c)+4)
            buff_c[2:-2] = c
            buff_c[:2] = 0
            buff_c[-2:] = 0

            buff_c_sum = np.zeros(len(c_sum)+4)
            buff_c_sum[2:-2] = c_sum
            buff_c_sum[:2] = 1
            buff_c_sum[-2:] = 0

            buff_v = np.zeros(len(v)+4)
            buff_v[2:-2] = v
            buff_v[:2] = v[0]#1e8
            buff_v[-2:] = 1e8

            overflow0 = (((1 - buff_c_sum[1:-3]) * buff_v[1:-3] + (1 - buff_c_sum[2:-2]) * buff_v[2:-2] + (1 - buff_c_sum[3:-1]) * buff_v[3:-1]) == 0).astype(np.int32)
            neigh0 = buff_c[2:-2] * ((1 - buff_c_sum[2:-2]) * buff_v[2:-2]) / ((1 - buff_c_sum[1:-3]) * buff_v[1:-3] + (1 - buff_c_sum[2:-2]) * buff_v[2:-2] + (1 - buff_c_sum[3:-1]) * buff_v[3:-1] + overflow0)
            log0 = faktor_logistic * (buff_v[1:-3] * (1 - buff_c_sum[1:-3]) + buff_v[2:-2] * (1 - buff_c_sum[2:-2]) + buff_v[3:-1] * (1 - buff_c_sum[3:-1])) / buff_v[2:-2]
            log0[log0>1] = 1.
            log0[log0<0] = 0.
            buff0 = (1. - p) * gamma * neigh0 * log0
            
            overflow1 = (((1 - buff_c_sum[2:-2]) * buff_v[2:-2] + (1 - buff_c_sum[3:-1]) * buff_v[3:-1] + (1 - buff_c_sum[4:]) * buff_v[4:]) == 0).astype(np.int32)
            neigh1 = buff_c[3:-1] * ((1 - buff_c_sum[2:-2]) * buff_v[2:-2]) / ((1 - buff_c_sum[2:-2]) * buff_v[2:-2] + (1 - buff_c_sum[3:-1]) * buff_v[3:-1] + (1 - buff_c_sum[4:]) * buff_v[4:] + overflow1)
            log1 = faktor_logistic * (buff_v[2:-2] * (1 - buff_c_sum[2:-2]) + buff_v[3:-1] * (1 - buff_c_sum[3:-1]) + buff_v[4:] * (1 - buff_c_sum[4:])) / buff_v[3:-1]
            log1[log1>1] = 1.
            log1[log0<0] = 0.
            buff1 = (1. - p) * gamma * buff_v[3:-1] / buff_v[2:-2] * neigh1 * log1
            
            overflow2 = (((1 - buff_c_sum[:-4]) * buff_v[:-4] + (1 - buff_c_sum[1:-3]) * buff_v[1:-3] + (1 - buff_c_sum[2:-2]) * buff_v[2:-2]) == 0).astype(np.int32)
            neigh2 = buff_c[1:-3] * ((1 - buff_c_sum[2:-2]) * buff_v[2:-2]) / ((1 - buff_c_sum[:-4]) * buff_v[:-4] + (1 - buff_c_sum[1:-3]) * buff_v[1:-3] + (1 - buff_c_sum[2:-2]) * buff_v[2:-2] + overflow2)
            log2 = faktor_logistic * (buff_v[:-4] * (1 - buff_c_sum[:-4]) + buff_v[1:-3] * (1 - buff_c_sum[1:-3]) + buff_v[2:-2] * (1 - buff_c_sum[2:-2])) / buff_v[1:-3]
            log2[log2>1] = 1.
            log2[log2<0] = 0.
            buff2 = (1. - p) * gamma * buff_v[1:-3] / buff_v[2:-2] * neigh2 * log2 
            
            log3 = faktor_logistic * (buff_v[2:-2] * (1 - buff_c_sum[2:-2]) + buff_v[2:-2] * (1 - buff_c_sum[2:-2]) + buff_v[3:-1] * (1 - buff_c_sum[3:-1])) / buff_v[2:-2]
            log3[log3>1] = 1.
            log3[log3<0] = 0.
            buff3 = p * gamma * buff_c[2:-2] * log3

            return buff0 + buff1 + buff2, buff3

        args_o2 = o2_models.sauerstoffdruck_analytical(params=params, c0=c0+c2, mode=2)
        rhypox = args_o2[1]
        rcrit = args_o2[2]

        ls =  np.arange(0,len(c0),1) * dr
        rs = np.arange(1,len(c0)+1,1) * dr

        rcrit = (rcrit - ls) / dr
        rcrit[rcrit<0] = 0
        rcrit[rcrit>1] = 1

        rhypox = (rhypox - ls) / dr
        rhypox[rhypox<0] = 0
        rhypox[rhypox>1] = 1
        rhypox = 1.-rhypox
        
        dcdt0 += prolif(c0, c0+c1+c2, 0)[0] * rhypox + migration(c0, c0+c1+c2) - epsilon * c0 * rcrit
        prolif_buff = prolif(c2,c0+c1+c2, p_mit_cata)
        dcdt2 += (prolif_buff[0] - prolif_buff[1]) * rhypox + migration(c2, c0+c1+c2) - epsilon * c2 * rcrit
        dcdt1 += migration(c1, c0+c1+c2) + epsilon * (c0 + c2) * rcrit - delta * c1

        dcdt = np.append(dcdt0,dcdt1)
        dcdt = np.append(dcdt,dcdt2)

        #print("[" + str(round(t/24./3600.,20)) + "] ")
        #print("\r[" + str(round(t/24./3600.,20)) + "] " + "        ",end="")

        return dcdt