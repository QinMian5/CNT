# Author: Mian Qin
# Date Created: 8/18/24
import numpy as np
import matplotlib.pyplot as plt


def main():
    Delta_T = 10
    T_m = 271  # K
    Delta_h = 5.6  # kJ/mol
    gamma = 3.7e-5  # kJ/m^2
    rho_N = 1e6 / 18  # mol/m^3
    Delta_mu = -Delta_h * Delta_T / T_m
    r = np.linspace(0, 1e-8, 100)
    lambda_ = 4/3 * np.pi * r ** 3
    E = Delta_mu * rho_N * 4/3 * np.pi * r**3 + 4 * np.pi * r**2 * gamma
    dE_dr = 4 * np.pi * r ** 2 * Delta_mu * rho_N + 8 * np.pi * r * gamma
    fig, ax = plt.subplots()
    ax.plot(lambda_, dE_dr)
    plt.show()


if __name__ == "__main__":
    main()
