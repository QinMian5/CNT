# Author: Mian Qin
# Date Created: 8/28/24
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as c
from scipy.optimize import newton
from scipy.interpolate import interp1d
import pandas as pd

T_m = 271  # K
Delta_H_m = 5.60e3  # J/mol

l_PI = 2  # nm
l_box = 4.6  # nm
alpha_list = [0.0, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]
critical_Delta_T_sim = {0.5: 90, 0.6: 85, 0.7: 75, 0.75: 45, 0.8: 10, 0.9: 0, 1.0: 0}


def read_k():
    column_names = ["alpha", "f", "f_err/2", "s", "s_err/2", "h", "h_err/2", "k", "k_err"]
    df = pd.read_csv("pi_cs.txt", sep=r'\s+', header=None, names=column_names, comment="#")
    alpha = df["alpha"].values
    k = df["k"].values
    alpha_to_k = interp1d(alpha, k, kind="linear")
    return alpha, alpha_to_k


def calc_H_m(T):
    H_m = 210368 - 2 * 3.32373e6 / T + 41729.1 * (1 - np.log(T))  # J/mol
    return H_m


def calc_sigma(T):
    sigma = (30.8 - 0.25 * (T_m - T)) * 1e-3
    return sigma


def calc_Delta_mu(T):
    delta_mu_infty = -770
    d = 0
    b = -Delta_H_m / T_m
    a = (delta_mu_infty - 65 * b) / 65 ** 2
    x = T_m - T
    Delta_mu = np.where(T_m - T <= 65, a * x ** 2 + b * x + d, delta_mu_infty)  # J/mol
    return Delta_mu


def calc_rho_i(T):
    rho_i = 0.910  # g/cm^3
    rho_i = rho_i / 18 * 1e6  # Convert into mol/m^3
    return rho_i


def calc_k_eff(alpha, h_lambda):
    assert h_lambda in [0, 1]
    k_w = 1 if h_lambda == 1 else -0.5
    _, alpha_to_k = read_k()
    k_eff = l_PI ** 2 / l_box ** 2 * alpha_to_k(alpha) + (1 - l_PI ** 2 / l_box ** 2) * k_w
    return k_eff


def calc_G(r, k_eff, T):
    sigma = calc_sigma(T)
    rho_i = calc_rho_i(T)
    Delta_mu = calc_Delta_mu(T)

    G_V = np.pi * (2 + k_eff) * (1 - k_eff) ** 2 * Delta_mu * rho_i * r ** 3 / 3
    G_S = np.pi * (2 + k_eff) * (1 - k_eff) ** 2 * sigma * r ** 2
    G = G_V + G_S
    return G


def calculate_deltaTstar(k_eff, n=20):
    if k_eff > 0.99:
        return 0
    r = np.logspace(-11, -7, 1000)

    def f(T):
        G_barr = np.max(calc_G(r, k_eff, T)) - n * c.k * T
        return G_barr

    for Delta_T in range(0, 50, 10):
        x0 = T_m - Delta_T
        T = newton(f, x0)
        Delta_T_star = T_m - T
        if Delta_T_star > 0:
            break
    else:
        raise RuntimeError(f"Fail to calculate deltaT for {k_eff}")
    return Delta_T_star


def main_plot_alpha_to_k():
    plt.style.use("presentation.mplstyle")
    fig, ax = plt.subplots()
    ax.set_title(r"$k$ as a Function of $\alpha$, Linear Interpolation")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$k$")

    alpha, alpha_to_k = read_k()
    alpha_list = np.linspace(0, 1, 100)
    k_list = alpha_to_k(alpha_list)

    ax.plot(alpha_list, k_list, "b-")
    ax.plot(alpha, alpha_to_k(alpha), "ro")

    plt.savefig("figure/alpha_to_k.png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def main_calc_plot_Delta_T_star_k():
    k_list = np.linspace(-1, 1, 41)
    plt.style.use("presentation.mplstyle")
    fig, ax = plt.subplots()
    ax.set_title(r"Critical Supercooling $\Delta T^*$ as a Function of $k$")
    ax.set_xlabel(r"$k \equiv \cos\theta$")
    ax.set_ylabel(r"$\Delta T^*\ ({\rm K})$")
    for n in [1, 5, 10, 30]:
        Delta_T_star_list = []
        for k in k_list:
            Delta_T_star = calculate_deltaTstar(k, n=n)
            Delta_T_star_list.append(Delta_T_star)
        ax.plot(k_list, Delta_T_star_list, "-", label=fr"$n={n}$")

    ax.legend()
    plt.savefig(f"figure/delta_T_star_k_tip4p_ice.png", bbox_inches="tight", pad_inches=0.1)


def main_plot_delta_mu():
    plt.style.use("presentation.mplstyle")
    fig, ax = plt.subplots()
    Delta_T = np.linspace(0, 65, 1000)
    Delta_mu = calc_Delta_mu(T_m - Delta_T)
    ax.plot(Delta_T, Delta_mu / 4.184 / 1000, "b-")
    ax.set_xlim([0, 70])
    ax.set_ylim([-0.4, 0])
    fig.savefig("./figure/delta_mu_tip4p_ice.png", bbox_inches="tight", pad_inches=0.1)


def main():
    data = {0: [], 1: []}
    data_flat = []
    for alpha in alpha_list:
        _, alpha_to_k = read_k()
        k = alpha_to_k(alpha)
        delta_T_star = calculate_deltaTstar(k)
        data_flat.append(delta_T_star)
        for h_lambda in [0, 1]:
            k_eff = calc_k_eff(alpha, h_lambda)
            delta_T_star = calculate_deltaTstar(k_eff)
            data[h_lambda].append(delta_T_star)
            print(f"alpha = {alpha}, h_lambda = {h_lambda}, k_eff = {k_eff}")
            print(f"Delta_T_star = {delta_T_star:.2f}")

    plt.style.use("presentation.mplstyle")
    fig, ax = plt.subplots()
    ax.set_title(r"Critical Supercooling $\Delta T^*$")
    ax.set_xlabel(r"Polarity $\alpha$")
    ax.set_ylabel(r"$\Delta T^*\ ({\rm K})$")
    for h_lambda in [0, 1]:
        ax.plot(alpha_list, data[h_lambda], "o-", label=fr"$h_\lambda = {h_lambda}$")
    # ax.plot([], [])
    # ax.plot([], [])
    ax.plot([0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0], [90, 85, 75, 45, 10, 0, 0], "o-", label=fr"$h_\lambda: 0 \to 1$")

    # critical_T_x = [0.0, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]
    # critical_T_y = data[0][:5] + [45, 10] + data[1][7:]
    critical_T_x = [0.75, 0.8, 0.9, 1.0]
    critical_T_y = [45, 10] + data[1][7:]
    ax.plot(critical_T_x, critical_T_y, "o-", label=r"$\Delta T^*$")
    ax.plot(alpha_list, data_flat, "o-", label=fr"flat")
    ax.legend()
    plt.savefig(f"figure/delta_T_tip4p_ice_compare.png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


if __name__ == "__main__":
    # print(calc_H_m(230))
    # main_plot_delta_mu()
    # main_plot_alpha_to_k()
    # print(calc_Delta_mu(T_m - 65))
    # main_calc_plot_Delta_T_star_k()
    main()
