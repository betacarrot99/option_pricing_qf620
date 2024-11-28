from scipy.stats import norm
import numpy as np
import matplotlib.pylab as plt


def phi(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def psi_Bt(S, K, r, sigma, T):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return -K * np.exp(-r * T) * norm.cdf(d2)


def simulate_Brownian_Motion(paths, steps, T):
    deltaT = T / steps
    t = np.linspace(0, T, steps + 1)
    X = np.c_[np.zeros((paths, 1)), np.random.randn(paths, steps)]
    return t, np.cumsum(np.sqrt(deltaT) * X, axis=1)


paths = 50000
hedging_steps = [21, 84]
maturity = 1.0 / 12

for steps in hedging_steps:
    T, W_T = simulate_Brownian_Motion(paths, steps, maturity)

    r = 0.05
    S0 = 100.0
    K = 100.0
    sigma = 0.2
    dt = maturity / steps
    path_hedging_error = []
    for i in range(paths):
        if i % 1000 == 0 and i > 0:
            print("paths explored: ", i)

        blackscholespath = S0 * np.exp((r - sigma**2 / 2) * T + sigma * W_T[i])

        deltas = []
        stockhedge_errors = []
        bondhedge_errors = []
        hedging_errors = []
        hedged_portfolios = []
        stock_holdings = []
        bond_holdings = []
        for t, S_t in zip(T, blackscholespath):
            stock_pos = phi(S_t, K, r, sigma, maturity - t) * S_t
            bond_pos = psi_Bt(S_t, K, r, sigma, maturity - t)
            V_t = stock_pos + bond_pos
            stock_holdings.append(stock_pos)
            bond_holdings.append(bond_pos)
            hedged_portfolios.append(V_t)
            deltas.append(phi(S_t, K, r, sigma, maturity - t))
            if t == 0.0:
                stockhedge_errors.append(0)
                bondhedge_errors.append(0)
                hedging_errors.append(0)
            else:
                stockhedge_errors.append(prev_phi * S_t - stock_pos)
                bondhedge_errors.append(prev_bond_pos * np.exp(r * dt) - bond_pos)
                hedging_errors.append(
                    (prev_phi * S_t - stock_pos)
                    + (prev_bond_pos * np.exp(r * dt) - bond_pos)
                )

            prev_phi = phi(S_t, K, r, sigma, maturity - t)
            prev_bond_pos = bond_pos

        path_hedging_error.append(sum(hedging_errors))

    bins = np.arange(-2.0, 2.1, 0.1)
    counts, edges = np.histogram(path_hedging_error, bins=bins)
    # Convert counts to percentage
    percentages = counts / counts.sum() * 100

    plt.bar(
        edges[:-1],
        percentages,
        width=np.diff(edges),
        align="edge",
        alpha=0.5,
        label=f"N={steps}",
    )

plt.xlabel("Hedging Error")
plt.ylabel("Frequency (%)")
plt.title("Frequency Distribution of Hedging Errors")
plt.legend()
plt.show()
