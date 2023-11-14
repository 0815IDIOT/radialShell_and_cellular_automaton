import matplotlib.pyplot as plt
import numpy as np
from CONSTANTS import *

def animate_6(k, sol, t, max_iter_time, dr, p0 = None, metrics1 = None, metrics2 = None, subplotMode = False):

    if k == -1:
        k = max_iter_time-1
    if not subplotMode:
        plt.clf()
        linewidth = 1
        linewidth_conc = 2
        fontsize = 10
    else:
        """
        linewidth = 0.5
        linewidth_conc = 1
        fontsize = 5
        """
        linewidth = 1
        linewidth_conc = 2
        fontsize = 10

    ax = plt.gca()
    ax.set_zorder(1)  # default zorder is 0 for ax1 and ax2
    ax.patch.set_visible(False)

    c = np.split(sol[k],3)
    r_max = 0
    for i in range(len(sol)):
        c_last = np.split(sol[i],3)
        c_last_sum = np.sum(c_last,axis=0)
        where = np.where(c_last_sum>1e-3)[0]
        if len(where) == 0:
            r_idx = 2
        else:
            r_idx = where[-1] + 2
        if r_idx > r_max:
            r_max = r_idx
    r_idx = r_max
    #r_idx = len(c[0])

    radius = np.arange(0,len(c[0]),1) * dr * 1e6
    #radius_p0 = np.arange(1, len(c[0]) + 1, 1) * dr * 1e6
    radius_p0 = np.arange(0.5, len(c[0]) + 0.5, 1) * dr * 1e6
    radius_bar = np.arange(0.5, len(c[0]) + 0.5, 1) * dr * 1e6
    if not subplotMode:
        plt.title("t = " + str(round(t[k],3)) + " d")

    lgds =  plt.plot(radius_bar[:r_idx], c[0][:r_idx], color=COLOR_PROLIFERATION, label=r"$c_p$", linewidth=linewidth_conc)
    lgd  = plt.plot(radius_bar[:r_idx], c[1][:r_idx], color=COLOR_ANOXIC, label=r"$c_n$", linewidth=linewidth_conc)
    lgds = lgds + lgd
    label_sum = "c_p + c_n"
    if np.sum(c[2]) != 0:
        lgd = plt.plot(radius_bar[:r_idx], c[2][:r_idx], color=COLOR_MITOTIC_CATASTROPHY, label=r"$c_d$", linewidth=linewidth_conc)
        lgds = lgds + lgd
        label_sum += " + c_d"
    label_sum = r"$" + label_sum + "$"
    lgd = plt.plot(radius_bar[:r_idx], c[0][:r_idx]+c[1][:r_idx]+c[2][:r_idx], color=COLOR_SUM, linestyle="dotted", label=label_sum, linewidth=linewidth_conc)
    lgds = lgds + lgd
    if subplotMode:
        lgd = plt.plot([], [], ' ', label="t = " + str(round(t[k],3)) + " d")
        lgds = lgd + lgds 

    #plt.plot(radius, c[0]+c[1], color=COLOR_SUM, linestyle="dotted", label="sum")

    if metrics2 is not None:
        #plt.axvspan(metrics2[4][k] * 1e6,metrics2[3][k] * 1e6,color=COLOR_PROLIFERATION,alpha=0.5,linewidth=0,ymax=1.1/1.2, ymin = 0.1/1.2)
        #plt.axvspan(metrics2[5][k] * 1e6,metrics2[4][k] * 1e6,color=COLOR_HYPOXIC,alpha=0.5,linewidth=0,ymax=1.1/1.2, ymin = 0.1/1.2)
        #plt.axvspan(0,metrics2[5][k] * 1e6,color=COLOR_ANOXIC,alpha=0.5,linewidth=0,ymax=1.1/1.2, ymin = 0.1/1.2)
        plt.arrow(metrics2[3][k] * 1e6, -0.09, 0, 0.07, color=COLOR_ERRORBAR, length_includes_head=True, head_width=10, head_length=0.03, linewidth=linewidth_conc)
        plt.arrow(metrics2[5][k] * 1e6, -0.09, 0, 0.07, color=COLOR_ERRORBAR_PREDICT, length_includes_head=True, head_width=10, head_length=0.03, linewidth=linewidth_conc)

    if metrics1 is not None:
        plt.plot([metrics1[3][k] * 1e6], [0], color=COLOR_PROLIFERATION, marker="x",linewidth=linewidth)
        plt.plot([metrics1[5][k] * 1e6], [0], color=COLOR_ANOXIC, marker="x",linewidth=linewidth)

    plt.ylim(-0.1,1.1)
    plt.xlim(-dr*1e6,radius[:r_idx+1][-1]+dr*1e6)
    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    plt.xlabel("distance $r$ from spheroid center [$\mu$m]",fontsize=fontsize)
    plt.ylabel("relativ cell concentration $c_T(r_i)$",fontsize=fontsize)

    labs = [l.get_label() for l in lgds]

    if p0 is not None:
        ax2 = ax.twinx()
        plt.plot(radius_p0[:r_idx], p0[k][:r_idx], color=COLOR_OXYGEN, linewidth=linewidth_conc/1.5)

        ax2.title.set_color(COLOR_OXYGEN)
        ax2.yaxis.label.set_color(COLOR_OXYGEN)
        ax2.tick_params(axis='y', colors=COLOR_OXYGEN)
        ax2.spines['right'].set_color(COLOR_OXYGEN)

        #ax2.fill(radius, p0[k], COLOR_OXYGEN, alpha=0.5)
        plt.fill_between(radius_p0[:r_idx], p0[k][:r_idx], color=COLOR_OXYGEN, alpha=0.1)
        plt.ylim(-160./11.,160)

        plt.ylabel(r"oxygen level $\rho(r_i)$ [mmHg]",fontsize=fontsize)
        plt.setp(ax2.get_yticklabels(), fontsize=fontsize)
        plt.setp(ax2.get_xticklabels(), fontsize=fontsize)

    #bbox signature (left, bottom, width, height)
    #with 0, 0 being the lower left corner and 1, 1 the upper right corner
    plt.legend(lgds, labs, loc="best", bbox_to_anchor=(0.0, 0.4, 1.0, 0.2), fontsize=fontsize, frameon=False)

def plotResults_6(t, metrics1, model1, p, dr, filename, t_therapy, d_therapy, sol1 = None, exp_data_vn = None, exp_data_v0 = None, metrics2 = None,metric_dot = None, sim_data = None, mode="v", figSave = False, pathFigSave = "../gif/", predict=None, subplotMode = False):

    p = np.array(p)
    p[p>140] = np.nan

    if not subplotMode:
        plt.figure()
        linewidth = 1
        linewidth_err = 1.5
        fontsize_legend = 10
        fontsize_axis_label = 10
        fontsize_axis_tick = 10
        capsize = 5
        capthick = linewidth_err
    else:
        linewidth = 1.5
        linewidth_err = 1.5
        fontsize_legend = 8
        fontsize_axis_label = 10
        fontsize_axis_tick = 10
        capsize = 0#5
        capthick = linewidth_err
    
    if mode == "v":
        plt.plot(t, metrics1[0] * 1e18, label=r"$V_{\mathrm{spheroid}}$ - RS-model", color=COLOR_PROLIFERATION, linestyle="solid", linewidth=linewidth)
        plt.plot(t, metrics1[2] * 1e18, label=r"$V_{\mathrm{necrotic}}$ - RS-model", color=COLOR_ANOXIC, linestyle="solid", linewidth=linewidth)
    elif mode == "r":
        plt.plot(t,metrics1[3] * 1e6, label=r"$R_{\mathrm{spheroid}}$ - RS-model", color=COLOR_PROLIFERATION, linestyle="solid", linewidth=linewidth)
        plt.plot(t,metrics1[5] * 1e6, label=r"$R_{\mathrm{necrotic}}$ - RS-model", color=COLOR_ANOXIC, linestyle="solid", linewidth=linewidth)

    if not metric_dot is None:
        if mode == "v":
            plt.plot(t, metric_dot[0] * 1e18, label=model1 + " V0", color=COLOR_PROLIFERATION, linestyle="dotted", linewidth=linewidth)
            plt.plot(t, metric_dot[2] * 1e18, label=model1 + " Vn", color=COLOR_ANOXIC, linestyle="dotted", linewidth=linewidth)
        elif mode == "r":
            plt.plot(t, metric_dot[3] * 1e6, label=r"$R_{\mathrm{spheroid}}$", color=COLOR_PROLIFERATION, linestyle="dotted", linewidth=linewidth)
            plt.plot(t, metric_dot[5] * 1e6, label=r"$R_{\mathrm{necrotic}}$", color=COLOR_ANOXIC, linestyle="dotted", linewidth=linewidth)

    if not sim_data is None:
        if mode == "v":
            plt.plot(sim_data[0], sim_data[1] * 1e18, color=COLOR_PROLIFERATION, linestyle="dashed", alpha=0.5, label=sim_data[2] + " - model", linewidth=linewidth)
        elif mode == "r":
            plt.plot(sim_data[0], ((sim_data[1] * 1e18 / np.pi)*(3./4.))**(1./3.), linestyle="dashed", color=COLOR_PROLIFERATION, alpha=0.5, label=sim_data[2] + " - model", linewidth=linewidth)

    if mode == "v":
        plt.hlines(2. * metrics1[0][0] * 1e18, t[0], t[-1], color="gray", linestyles="dashed", linewidth=linewidth)
        plt.hlines(5. * metrics1[0][0] * 1e18, t[0], t[-1], color="gray", linestyles="dotted", linewidth=linewidth)#
        plt.text(t[0], 2. * metrics1[0][0] * 1e18 * 1.1, r"$2V_{t=0}$", fontsize=fontsize_axis_label*0.9)
        plt.text(t[0], 5. * metrics1[0][0] * 1e18 * 1.1, r"$5V_{t=0}$", fontsize=fontsize_axis_label*0.9)
        plt.yscale("log")

    if len(d_therapy) > 1:
        if mode == "v":
            maxi = np.max(metrics1[0] * 1e18)
            mini = np.min(metrics1[2] * 1e18)
        elif mode == "r":
            maxi = np.max(metrics1[3] * 1e6)
            mini = np.min(metrics1[5] * 1e6)

        for i in range(len(d_therapy)-1):
            plt.vlines(t_therapy[i], mini, maxi, color="gray", linestyle="dashed", label="therapy " + str(i+1) + ": " + str(d_therapy[i+1]) + " Gy", linewidth=linewidth)

    fak = 3
    if not exp_data_v0 is None:
        x_mean = exp_data_v0[0]
        y_mean = exp_data_v0[1] * 1e6**fak
        x_var = exp_data_v0[2]
        y_var = exp_data_v0[3] * (1e6**fak)**2
        label = exp_data_v0[4].replace("_"," ")

        if mode == "v":
            label = "$V_{\mathrm{spheroid}}$ - Exp."
        elif mode == "r":
            label = "$R_{\mathrm{spheroid}}$ - Exp."
            y_var = ((((y_mean + np.sqrt(y_var)) / np.pi)*(3./4.))**(1./3.) - ((y_mean / np.pi)*(3./4.))**(1./3.))**2
            y_mean = ((y_mean / np.pi)*(3./4.))**(1./3.)

        plt.errorbar(x_mean, y_mean, np.sqrt(y_var), np.sqrt(x_var), color=COLOR_ERRORBAR, linewidth=linewidth_err, linestyle="none", fmt="^", markersize=6., capsize=capsize, capthick=capthick, label=label, zorder=1)

    if not exp_data_vn is None:
        x_mean = exp_data_vn[0]
        y_mean = exp_data_vn[1] * 1e6**fak
        x_var = exp_data_vn[2]
        y_var = exp_data_vn[3] * (1e6**fak)**2
        label = exp_data_vn[4].replace("_"," ")

        if mode == "v":
            label = "$V_{\mathrm{necrotic}}$ - Exp."
        elif mode == "r":
            label = "$R_{\mathrm{necrotic}}$ - Exp."
            y_var =  ((((y_mean + np.sqrt(y_var)) / np.pi)*(3./4.))**(1./3.) - ((y_mean / np.pi)*(3./4.))**(1./3.))**2
            y_mean = ((y_mean / np.pi)*(3./4.))**(1./3.)

        plt.scatter(x_mean[y_mean==0], y_mean[y_mean==0], color=COLOR_ERRORBAR_PREDICT, marker="|", linewidths=linewidth_err)
        plt.scatter(x_mean[y_mean==0], y_mean[y_mean==0], color=COLOR_ERRORBAR_PREDICT, marker="s", s=6**2., linewidths=linewidth_err)
        plt.errorbar(x_mean[y_mean!=0] , y_mean[y_mean!=0] , np.sqrt(y_var[y_mean!=0] ), np.sqrt(x_var[y_mean!=0] ), color=COLOR_ERRORBAR_PREDICT, linewidth=linewidth_err, linestyle="none", fmt="s", markersize=6., capsize=capsize, capthick=capthick, label=label, zorder=1)


    if mode == "v":
        plt.ylabel("volumen $V$ [$\mu m^3$]", fontsize=fontsize_axis_label)
    elif mode == "r":
        plt.ylabel("radius $R$ [$\mu m$]", fontsize=fontsize_axis_label)
    plt.xlabel("time $t$ [d]", fontsize=fontsize_axis_label)
    if mode == "v":
        plt.legend(fontsize=fontsize_legend, frameon=False, bbox_to_anchor=(0.0, 0.2, 1.0, 0.2))
    else:
        plt.legend(fontsize=fontsize_legend, frameon=False)

    ax = plt.gca()
    if mode == "v":
       ylim = list(ax.get_ylim())
       if ylim[1] < 5. * metrics1[0][0] * 1e18 * 1.4:
          ylim[1] =  5. * metrics1[0][0] * 1e18 * 1.4
          ax.set_ylim(ylim[0], ylim[1])

    plt.xticks(np.arange(min(t), max(t)+1, 5.0))
    plt.setp(ax.get_xticklabels(), fontsize=fontsize_axis_tick)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize_axis_tick)

    if figSave:
        if mode == "v":
            plt.savefig(pathFigSave + filename + "_volume.pdf",transparent=True)
        if mode == "r":
            plt.savefig(pathFigSave + filename + ".pdf",transparent=True)
        plt.close()

    # start and end Figure
    if sol1 is not None:
        plt.figure()
        animate_6(0, sol1, t, len(t), dr, p0 = p, metrics1 = metrics1, metrics2 = metrics2)
        if figSave:
            plt.savefig(pathFigSave + filename + "_start.pdf",transparent=True)
            plt.close()

        plt.figure()
        animate_6(len(t)-1, sol1, t, len(t), dr, p0 = p, metrics1 = metrics1, metrics2 = metrics2)
        if figSave:
            plt.savefig(pathFigSave + filename + "_end.pdf",transparent=True)
            plt.close()
