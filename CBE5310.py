# Author: Mian Qin
# Date Created: 9/15/24
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid


def delta(t, tau):
    # return np.heaviside(t - tau, 0.5)
    sigma = 0.001
    delta_approx = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(- (t - tau) ** 2 / (2 * sigma ** 2))
    return delta_approx


def main():
    t = np.linspace(-4, 4, 80000)
    dt = float(t[1] - t[0])
    f = 3 * delta(t, 1) - 3 * delta(t, -1) - delta(t, 3) + delta(t, -3)
    int1 = cumulative_trapezoid(f, t, initial=0)
    int2 = cumulative_trapezoid(int1, t, initial=0)
    int3 = cumulative_trapezoid(int2, t, initial=0)

    plt.style.use("presentation.mplstyle")
    fig, ax = plt.subplots()
    ax.set_xlabel(rf"$t$")
    ax.set_ylabel(rf"$g(t)$")
    ax.plot(t, int3, "b-")
    plt.show()


if __name__ == "__main__":
    main()
