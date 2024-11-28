import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.stats import norm

import sys

sys.path.append("..")
from analytical_option_formulae.option_types.vanilla_option import VanillaOption


# Define common functions and models
vanilla_option = VanillaOption()


def calculate_stock_prices(
    S: float, r: float, sigma: float, t: npt.NDArray, W: npt.NDArray
) -> float:
    return S * np.exp((r * t - 0.5 * sigma**2 * t) + sigma * W)


def calculate_phi(
    S_t: npt.NDArray, K: float, r: float, sigma: float, T: float, t: npt.NDArray
) -> float:
    d1 = (np.log(S_t / K) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(T - t))
    return norm.cdf(d1)


def calculate_psi_Bt(
    S_t: npt.NDArray, K: float, r: float, sigma: float, T: float, t: npt.NDArray
) -> npt.NDArray:
    d2 = (np.log(S_t / K) + (r - 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(T - t))
    return -K * np.exp(-r * (T - t)) * norm.cdf(d2)


def simulate_brownian_paths(
    n_paths: int, T: float, n_steps: int
) -> tuple[npt.NDArray, npt.NDArray]:
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    X = np.c_[np.zeros((n_paths, 1)), np.random.randn(n_paths, n_steps)]
    return t, np.cumsum(np.sqrt(dt) * X, axis=1)


def compute_hedging_error(
    S_0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    t: npt.NDArray,
    brownian_paths: npt.NDArray,
):
    stock_prices = calculate_stock_prices(S_0, r, sigma, t, brownian_paths)
    phi_values = calculate_phi(stock_prices, K, r, sigma, T, t)
    psi_b_values = calculate_psi_Bt(stock_prices, K, r, sigma, T, t)
    stock_hedging_err = -np.diff(phi_values) * stock_prices[:, 1:]
    bond_hedging_err = (
        psi_b_values[:, :-1] * np.exp(r * T / len(t)) - psi_b_values[:, 1:]
    )
    hedging_err = stock_hedging_err + bond_hedging_err
    hedging_err_sum = np.sum(hedging_err, axis=1)
    return hedging_err_sum


# Parameters
S_0 = 100
K = 100
r = 0.05
sigma = 0.2
T = 1 / 12
num_paths = 50000
hedging_steps = [21, 84]

# Set seed
np.random.seed(42)

hedging_errors = []
# Computation & Visualisation
for hedging_step in hedging_steps:
    t, x = simulate_brownian_paths(num_paths, T, hedging_step)
    simulation_results = compute_hedging_error(S_0, K, r, sigma, T, t, x)
    hedging_errors.append(simulation_results)

print("Mean for N=21: {}".format(np.mean(hedging_errors[0])))
print("Mean for N=84: {}".format(np.mean(hedging_errors[1])))
print("Std for N=21: {}".format(np.std(hedging_errors[0])))
print("Std for N=84: {}".format(np.std(hedging_errors[1])))

plt.figure(figsize=(12, 6))

plt.subplot(121)
for i, hedging_step in enumerate(hedging_steps):
    plt.hist(
        hedging_errors[i],
        bins=50,
        align="mid",
        alpha=0.5,
        label=f"N={hedging_step}",
    )
    plt.xlim(-2, 2)
plt.legend()
plt.title("Frequency Distribution of Hedging Errors (count)")
plt.xlabel("Hedging Error")
plt.ylabel("Frequencies")

plt.subplot(122)
for i, hedging_step in enumerate(hedging_steps):
    bins = np.arange(-2.0, 2.1, 0.1)
    counts, edges = np.histogram(hedging_errors[i], bins=bins)
    # Convert counts to percentage
    percentages = counts / counts.sum() * 100
    plt.bar(
        edges[:-1],
        percentages,
        width=np.diff(edges),
        align="edge",
        alpha=0.5,
        label=f"N={hedging_step}",
    )

plt.legend()
plt.title("Frequency Distribution of Hedging Errors (percentage)")
plt.xlabel("Hedging Error")
plt.ylabel("Frequency (%)")

plt.savefig("Part_4_Hedging_Error.png")
