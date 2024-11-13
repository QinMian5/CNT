# Author: Mian Qin
# Date Created: 8/28/24
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as c
from scipy.optimize import newton


T_m = 273.15  # K

l_PI = 2  # nm
l_box = 4.6  # nm
alpha_to_k_PI = {1: 1, 0.75: 0.5, 0.5: -0.1, 0.25: -0.25, 0.0: -0.5}


def calc_sigma(T):
    Delta_H_m_h = (6.008 + 0.03616 * (T - T_m) - 3.9479e-4 * (T - T_m) ** 2 - 1.6248e-5 * (T - T_m) ** 3 - 3.2563e-7 * (T - T_m) ** 4) * 1e3  # J/mol, Enthalpy of melting for hexagonal ice
    Delta_H_sd_h = 0.155e3  # J/mol, Enthalpy difference between stacking disordered and hexagonal ice
    Delta_H_m_sd = Delta_H_m_h - Delta_H_sd_h
    Delta_H_m_sd_Tr = 4.1776e3  # J/mol, Delta_H_m_sd at Tr
    sigma_sd_l_Tr = 18.505e-3  # J/m^2
    sigma_sd_l = Delta_H_m_sd / Delta_H_m_sd_Tr * sigma_sd_l_Tr
    return sigma_sd_l


def _old_calc_sigma(T):
    sigma = 3.7e-2  # J/m^2
    return sigma


def calc_Delta_mu(T):
    Delta_mu = 210368 + 131.438 * T - 3.32373e6 / T - 41729.1 * np.log(T)  # J/mol
    return Delta_mu


def _old_calc_Delta_mu(T):
    Delta_h = 5.6e3  # J/mol
    Delta_mu = Delta_h / T_m * (T - T_m)
    return Delta_mu


def calc_rho_i(T):
    rho_i = -1.3103e-9 * T ** 3 + 3.8109e-7 * T ** 2 - 9.2592e-5 * T + 0.94040  # g/cm^3
    rho_i = rho_i / 18 * 1e6  # Convert into mol/m^3
    return rho_i


def _old_calc_rho_i(T):
    rho_i = 5e4  # mol/m^3
    return rho_i


def calc_k_eff(alpha, h_lambda):
    assert h_lambda in [0, 1]
    k_w = 1 if h_lambda == 1 else -1
    k_eff = l_PI ** 2 / l_box ** 2 * alpha_to_k_PI[alpha] + (1 - l_PI ** 2 / l_box ** 2) * k_w
    return k_eff


def calc_G(r, k_eff, T):
    sigma = calc_sigma(T)
    rho_i = calc_rho_i(T)
    Delta_mu = calc_Delta_mu(T)

    G_V = np.pi * (2 + k_eff) * (1 - k_eff) ** 2 * Delta_mu * rho_i * r ** 3 / 3
    G_S = np.pi * (2 + k_eff) * (1 - k_eff) ** 2 * sigma * r ** 2
    G = G_V + G_S
    return G


def calculate_deltaTstar(k_eff, n=5):
    if k_eff == 1.0:
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


def main_calc_plot_Delta_T_star_k():
    k_list = np.linspace(-1, 1, 41)
    Delta_T_star_list = []
    for k in k_list:
        Delta_T_star = calculate_deltaTstar(k)
        Delta_T_star_list.append(Delta_T_star)

    plt.style.use("presentation.mplstyle")
    fig, ax = plt.subplots()
    ax.set_title(r"Critical Supercooling $\Delta T^*$ as a Function of $k$")
    ax.set_xlabel(r"$k \equiv \cos\theta$")
    ax.set_ylabel(r"$\Delta T^*\ ({\rm K})$")
    ax.plot(k_list, Delta_T_star_list, "b-")
    plt.savefig(f"figure/delta_T_star_k.png", bbox_inches="tight", pad_inches=0.1)


def main_calc_plot_parameter_difference():
    Delta_T_list = np.linspace(-30, 60, 100)
    old_sigma_list = []
    new_sigma_list = []
    old_rho_list = []
    new_rho_list = []
    old_Delta_mu_list = []
    new_Delta_mu_list = []
    for Delta_T in Delta_T_list:
        T = T_m - Delta_T
        old_sigma_list.append(_old_calc_sigma(T))
        new_sigma_list.append(calc_sigma(T))
        old_rho_list.append(_old_calc_rho_i(T))
        new_rho_list.append(calc_rho_i(T))
        old_Delta_mu_list.append(_old_calc_Delta_mu(T))
        new_Delta_mu_list.append(calc_Delta_mu(T))

    plt.style.use("presentation.mplstyle")

    fig, ax = plt.subplots()
    ax.set_title(r"Comparison of $\sigma(T)$")
    ax.set_xlabel(r"Supercooling $\Delta T$")
    ax.set_ylabel(r"Surface tension $\sigma\ ({\rm J/m^2})$")
    ax.plot(Delta_T_list, old_sigma_list, "b-", label=r"old value")
    ax.plot(Delta_T_list, new_sigma_list, "r-", label=r"new value")
    ax.legend()
    plt.savefig(f"figure/comparison_sigma.png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.set_title(r"Comparison of $\rho_i(T)$")
    ax.set_xlabel(r"Supercooling $\Delta T$")
    ax.set_ylabel(r"Density $\rho_i\ ({\rm mol/m^3})$")
    ax.plot(Delta_T_list, old_rho_list, "b-", label=r"old value")
    ax.plot(Delta_T_list, new_rho_list, "r-", label=r"new value")
    ax.legend()
    plt.savefig(f"figure/comparison_rho.png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.set_title(r"Comparison of $\Delta\mu(T)$")
    ax.set_xlabel(r"Supercooling $\Delta T$")
    ax.set_ylabel(r"Chemical potential $\Delta\mu\ ({\rm J/mol})$")
    ax.plot(Delta_T_list, old_Delta_mu_list, "b-", label=r"old value")
    ax.plot(Delta_T_list, new_Delta_mu_list, "r-", label=r"new value")
    ax.legend()
    plt.savefig(f"figure/comparison_Delta_mu.png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def main():
    data = {0: [], 1: []}
    for alpha in alpha_to_k_PI:
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
        ax.plot(list(alpha_to_k_PI), data[h_lambda], "o-", label=fr"$h_\lambda = {h_lambda}$")
    ax.legend()
    plt.savefig(f"figure/delta_T.png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


if __name__ == "__main__":
    main_calc_plot_parameter_difference()
    # main_calc_plot_Delta_T_star_k()
    # main()
