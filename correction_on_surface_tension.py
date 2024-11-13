# Author: Mian Qin
# Date Created: 8/27/24
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as c


def plot_line(ax, r, E, **kwargs):
    T_m = 271  # K
    ax.plot(r * 1e9, E / (c.k * T_m), **kwargs)


def main():
    Delta_T = 100
    T_m = 271  # K
    Delta_h = 5.6e3  # J/mol
    gamma_0 = 3.7e-2  # J/m^2
    rho_N = 5e4  # mol/m^3
    Delta_mu = -Delta_h * Delta_T / T_m
    r = np.linspace(1e-11, 2e-9, 1000)
    gamma = gamma_0 / (1 + 2 * 1e-10 / r)

    k_eff = -1

    E_V = np.pi * (2 + k_eff) * (1 - k_eff) ** 2 * Delta_mu * rho_N * r ** 3 / 3
    E_S = np.pi * (2 + k_eff) * (1 - k_eff) ** 2 * gamma * r ** 2
    E = E_V + E_S
    E_V_0 = np.pi * (2 + k_eff) * (1 - k_eff) ** 2 * Delta_mu * rho_N * r ** 3 / 3
    E_S_0 = np.pi * (2 + k_eff) * (1 - k_eff) ** 2 * gamma_0 * r ** 2
    E_0 = E_V_0 + E_S_0

    plt.style.use("presentation.mplstyle")
    fig, ax = plt.subplots()
    ax.set_title(fr"$\Delta T = {Delta_T}\ {{\rm K}}, k_{{\rm eff}} = \cos\theta = {k_eff}$")
    ax.set_xlabel("$r$(nm)")
    ax.set_ylabel("$E$(kT)")
    plot_line(ax, r, E_0, label="E")
    plot_line(ax, r, E, label="corrected E")
    ax.legend()
    plt.savefig(f"figure/{Delta_T}_{k_eff}.png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


if __name__ == "__main__":
    main()
