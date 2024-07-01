import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from labellines import labelLines
import os

def main():

    print(os.getcwd())

    b_path = "../../../Plot_Fid_Paper/"

    n_eff = np.linspace(0, 99, 100)
    n_min = n_eff + 3
    q_min = 1 - n_eff/n_min
    eff_max = n_eff/n_min

    #fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))

    #fig = plt.figure(figsize=(10, 5))
    fig1 = plt.figure()
    ax1 = plt.subplot()
    #ax = plt.subplot()
    ax1.plot(n_eff, q_min, color="#E20074")
    ax1.plot(n_eff, eff_max, color="#00305D")
    ax1.vlines(x=40, ymin=0, ymax=1, color="#99ACBE", linestyles="dashed")


    q_min_patch = mpatches.Patch(color="#E20074", label=r"$Q_{min}$")
    eff_max_patch = mpatches.Patch(color="#00305D", label=r"$\eta_{eff,max}$")

    ax1.legend(handles=[q_min_patch, eff_max_patch], loc="upper right")
    ax1.set_xlabel(r"$N_{eff}$")

    plt.savefig(b_path+"qmin_effmax_vs_neff.pdf")
    plt.close()
    #plt.show()

    n_mult = np.linspace(1, 500, 500)

    n_ln = 3*n_mult + 40
    n_ln_down = 3*n_mult + 1
    n_ln_up = 3*n_mult + 100

    q = 1 - 40/n_ln
    q_down = 1 - 1/n_ln_down
    q_up = 1 - 100/n_ln_up
    eff = 40/n_ln
    eff_down = 1 / n_ln_down
    eff_up = 100/n_ln_up

    #fig = plt.figure(figsize=(10, 5))
    #ax = plt.subplot()
    fig2 = plt.figure()
    ax2 = plt.subplot()

    ax2.plot(n_ln_down, q_down, color="#FFDEE6", label=r"$N_{eff}=1$")#, linestyle="-.")
    ax2.plot(n_ln_down, eff_down, color="#B3C1CE", label=r"$N_{eff}=1$")#, linestyle="-.")

    ax2.plot(n_ln_up, q_up, color="#FFDEE6", label=r"$N_{eff}=100$")#, linestyle="--")
    ax2.plot(n_ln_up, eff_up, color="#B3C1CE", label=r"$N_{eff}=100$") #, linestyle="--")

    ax2.plot(n_ln, q, color="#E20074", label=r"$N_{eff}=40$")
    ax2.plot(n_ln, eff, color="#00305D", label=r"$N_{eff}=40$")

    q_patch = mpatches.Patch(color="#E20074", label=r"$Q$")
    eff_patch = mpatches.Patch(color="#00305D", label=r"$\eta_{eff}$")
    ax2.legend(handles=[q_patch, eff_patch], loc="upper right")

    ax2.set_xlabel(r"frame length $N$")
    ax2.set(xlim=(0,800))

    xvals = [500, 500, 500, 500, 500, 500]
    lines = plt.gca().get_lines()
    labelLines(lines, align=True, xvals=xvals)

    """
    n = np.zeros(shape=(len(n_eff), len(n_mult)))

    for i, neff in enumerate(n_eff):
        n_len = 3*n_mult + neff
        n[i][:] = n_len

    n_eff_mtx = np.zeros(shape=(len(n_eff), len(n_mult)))
    for i, _ in enumerate(n_mult):
        n_eff_mtx[:, i] = n_eff

    q = np.zeros(shape=n.shape)
    eff = np.zeros(shape=n.shape)
    for i, neff in enumerate(n_eff):
        eff[i][:] = neff / n[i][:]
        q[i][:] = 1 - eff[i][:]

    fig1, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 12), subplot_kw={'projection': '3d'})

    ax1.plot_wireframe(n_eff_mtx, n, q,rstride=10, cstride=0)
    ax1.set_title("Measuring portion Q")

    ax2.plot_wireframe(n_eff_mtx, n, eff,rstride=10, cstride=0)
    ax2.set_title("frame efficiency")
    """
    plt.tight_layout()
    plt.savefig(b_path+"Q_EFF_vs_N.pdf")
    plt.close()



if __name__=='__main__':
    main()