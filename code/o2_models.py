import numpy as np
from scipy.integrate import solve_ivp

class o2_models():
    def sauerstoffdruck_Gri2016_radius_2(r, v, params):

        D = params["D"].value # m^2 s^-1
        pinf = params["p0"].value # mmHg
        a = params["a"].value # Kg^-1 m^3 s^-1
        pcrit = params["pcrit"].value # mmHg

        r0 = (3. * v / (4. * np.pi)) ** (1./3.)
        rn = np.sqrt(2.* D * (pcrit - pinf) / a + r0**2)

        p = np.zeros(len(r))
        p[r<rn] == pcrit
        p[np.logical_and(r<r0,rn<=r)] = a * (r[np.logical_and(r<r0,rn<=r)]**2 / 2. + rn**3 / r[np.logical_and(r<r0,rn<=r)]) / (3.*D) + pinf - a * r0**2 / (2. * D)
        p[r>=r0] = -a * (r0**3 - rn**3) / (3. * D * r[r>=r0]) + pinf
        return p

    def sauerstoffdruck_Gri2016_radius(r, v, params):

        D = params["D"].value # m^2 s^-1
        p0 = params["p0"].value # mmHg
        a = params["a"].value # Kg^-1 m^3 s^-1
        omega = 1
        ph = params["ph"].value # mmHg
        pcrit = params["pcrit"].value # mmHg
        rd = params["rd"].value # m

        rn, rh, r0, rl, rs = o2_models.sauerstoffdruck_Gri2016(v, params)
        p = np.zeros(len(r))
        p[r<rn] == pcrit
        p[np.logical_and(r<r0,rn<=r)] = a * (-r0**2/2.-rn**3/r0+r[np.logical_and(r<r0,rn<=r)]**2/2+rn**3/r[np.logical_and(r<r0,rn<=r)]) / (3. * D) + p0
        #p[(r<r0 & rn<=r)] = a * (-r0**2/2.-rn**3/r0+r**2/2+rn**3/r) / (3. * D) + p0
        p[r>=r0] = a * (-r0**2/r[r>=r0] + rn**3/r[r>=r0] + r0**2 - rn**3/r0) / (3. * D) + p0
        p[p>p0] = p0
        p[r>r0] = p0
        return p

    def sauerstoffdruck_Gri2016_2(v, params):
        m = 1. if not "m" in params else params["m"].value
        D = params["D"].value / m**2 # m^2 s^-1
        p0 = params["p0"].value # mmHg
        a = params["a"].value # Kg^-1 m^3 s^-1
        pcrit = params["pcrit"].value # mmHg
        v /= m**3

        if v < 0:
            v = 0

        r0 = (3. * v / (4. * np.pi)) ** (1./3.)

        if r0**2 + 2.* D * (pcrit - p0) / a > 0:
            rn = np.sqrt(2.* D * (pcrit - p0) / a + r0**2)
        else:
            rn = 0.

        return rn*m, None, r0*m, None, None


    def sauerstoffdruck_Gri2016(v, params):
        m = 1. if not "m" in params else params["m"].value
        D = params["D"].value / m**2 # m^2 s^-1
        p0 = params["p0"].value # mmHg
        a = params["a"].value # Kg^-1 m^3 s^-1
        omega = 1
        ph = params["ph"].value # mmHg
        pcrit = params["pcrit"].value # mmHg
        rd = params["rd"].value / m # m
        v /= m**3

        if v < 0:
            v = 0

        r0 = (3. * v / (4. * np.pi)) ** (1./3.)

        rl = np.sqrt((6. * D * (p0-pcrit)) / (a * omega))
        rs = rl * np.sqrt(1-ph/p0)

        if r0 <= rl:
            rn = 0
        else:
            rn = r0 * (0.5 - np.cos((np.arccos(1 - 2. * rl ** 2 / r0 ** 2) - 2. * np.pi) / 3.))

        if r0 <= rs:
            rh = 0
        else:
            phi = r0 ** 2 + 2. * rn ** 3 / r0 + 6. * D * (ph - p0) / (omega * a)
            rh = 2 * np.sqrt(phi/3.) * np.cos(np.arccos(max([-np.sqrt((3. * rn ** 2 / phi)**3),-1]))/3.)
        rh = max([r0-rd,rh])

        return rn*m, rh*m, r0*m, rl*m, rs*m

    def sauerstoffdruck_analytical(**args):
        return o2_models.sauerstoffdruck_analytical_4(**args)

    def sauerstoffdruck_analytical_4(params, c0, mode = 1, rcrit = None, threshold = 0.1, rStart = -1):

        try:
            D_p = params["D_p"].value
            a = params["a"].value
            dr = params["dr"].value
            p0 = params["p0"].value
            ph = params["ph"].value
            pcrit = params["pcrit"].value
            og_dr = params["og_dr"].value if "og_dr" in params else dr

            n = len(c0)
            rmaxi = dr * n
            m = int(round(rmaxi/og_dr,2))
            radius = np.arange(0.5, n+0.5, 1) * dr
            og_radius = np.arange(0.5, m+0.5, 1) * og_dr
            p_radius = np.arange(1., m+1., 1) * og_dr
            og_c0 = np.interp(og_radius,radius,c0)

            def eqn(c1, c2, c3=None):
                if c3 == None:
                    rmax = og_dr * c2
                    c_rmax = c2
                else:
                    rmax = og_dr * c3
                    c_rmax = c3

                radius = np.arange(1, c2 + 1, 1)[:c2] * og_dr
                radius_rmax = np.arange(1, c_rmax + 1, 1)[:c_rmax] * og_dr

                int_c1 = np.sum(radius * og_c0[:c2] * og_dr * (radius < c1))
                int_c12 = np.sum(radius ** 2 * og_c0[:c2] * og_dr * (radius < c1))
                int_rmax = np.sum(radius_rmax * og_c0[:c_rmax] * og_dr)
                int_rmax2 = np.sum(radius_rmax ** 2 * og_c0[:c_rmax] * og_dr)

                out = a * int_c12 - a * int_rmax2 - a * c1 * int_c1
                out += a * rmax * int_rmax + D_p * c1 * pcrit - D_p * rmax * p0
                out = out / (D_p * (c1 - rmax))
                out += a * np.cumsum(radius * og_c0[:c2] * og_dr) / D_p - a * np.cumsum(radius ** 2 * og_c0[:c2] * og_dr) / D_p / radius
                buff = a * c1 * int_rmax2 - a * rmax * int_c12 + a * c1 * rmax * int_c1 - a * c1 * rmax * int_rmax - D_p * c1 * rmax * pcrit + D_p * c1 * rmax * p0
                out = out + buff / (D_p * radius * (c1 - rmax))

                return out

            n = len(og_c0)-1
            if rStart == -1:
                if mode == 2:
                    while og_c0[n] <= threshold and n > 1:
                        n -= 1
            elif rStart == 0:
                pass
            else:
                n = int(round(rStart/og_dr,2))
                og_radius = np.arange(0,n,1) * og_dr
                c0_buff = np.zeros((n))
                c0_buff[:len(og_c0)] = og_c0
                og_c0 = c0_buff

            if rcrit is not None:
                buff = eqn(rcrit, m, c3=n)

                return og_radius, None, rcrit, buff

            rmaxi = og_dr * n
            depth = int(np.log(rmaxi/og_dr)/np.log(2))+1
            rcrit = rmaxi/ 2.
            r = np.arange(0,n,1)*og_dr

            for i in range(depth):

                retOut = eqn(rcrit, n)
                mini = np.min(retOut)
                val = np.where(retOut == mini)[0]
                test = np.interp([rcrit],r,retOut)[0]

                if test > pcrit:
                    rcrit -= rmaxi / 2**(i+2)
                else:
                    rcrit += rmaxi / 2**(i+2)
            
            if test < pcrit:
                rcrit -= rmaxi / 2**(i+2)
            else:
                rcrit += rmaxi / 2**(i+2)

            raw = eqn(rcrit, m, c3=n)
            raw_mini = np.min(raw)
            raw_val = np.where(raw == raw_mini)[0]
            
            test = eqn(0,n)
            if rcrit == rmaxi / 2**(depth) and test[0] > pcrit:
                retOut = test
                mini = np.min(retOut)
                val = np.where(retOut == mini)[0]
                rcrit = -og_dr
            else:
                wheri = np.where(retOut<=pcrit)
                if len(wheri[0]) == 0:
                    rcrit = r[-1]
                else:
                    rcrit = r[np.max(wheri)]

            buff = p0 * np.ones(len(og_c0))
            buff[:n] = retOut
            buff[:val[0]] = pcrit
            buff[buff<pcrit] = pcrit
            # raw data
            raw[:raw_val[0]] = pcrit
            raw[raw<pcrit] = pcrit

            rh = r[np.where(retOut[:n] == min(retOut[:n], key=lambda x:abs(x-ph)))[0][-1]]

            if min(retOut, key=lambda x:abs(x-ph)) > ph:
                rh -= og_dr

            buff = np.interp(radius, p_radius, buff)
            raw = np.interp(radius, p_radius, raw)

            return buff, rh, rcrit, raw

        except Exception as e:
            import sys
            import traceback
            from tools import parametersToDict, prettyDictPrint

            print("")
            print("*** Exception occure ***")
            print("")
            print(traceback.format_exc())
            print(len(retOut))
            print("*** Parameters ***")
            print("")
            print(params)
            #prettyDictPrint(parametersToDict(params))
            print("")
            print("*** Specials ***")
            print("")
            print("np.where: " + str(np.where(retOut == min(retOut, key=lambda x:abs(x-ph)))))
            print("wheri: " + str(wheri))
            print("n: " + str(n))
            print("c0: " + str(c0))
            print("c0[:n]: " + str(c0[:n]))
            print("c0[n]: " + str(c0[n]))
            print("c0[n+1]: " + str(c0[n+1]))
            print("og_c0: " + str(og_c0))
            print("og_c0[:n]: " + str(og_c0[:n]))
            print("og_c0[n]: " + str(og_c0[n]))
            print("og_c0[n+1]: " + str(og_c0[n+1]))
            print("retOut: " + str(retOut))

            sys.exit()

        return c0,None,None
