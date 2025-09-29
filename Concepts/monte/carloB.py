import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict

# ---- Helper: steady state probabilities for M/M/c/K ----


def mmck_probs(lmbda, mu, c, K):
    """
    Compute steady state probabilities p_n for M/M/c/K queue.
    lmbda: arrival rate
    mu: service rate per server
    c: number of servers
    K: system capacity (including jobs in service)
    """
    rho = lmbda / (c * mu)
    p = np.zeros(K + 1)

    # Unnormalized probabilities
    for n in range(0, K + 1):
        if n < c:
            p[n] = (lmbda / mu)**n / math.factorial(n)
        else:
            p[n] = (lmbda / mu)**n / (math.factorial(c) * (c**(n - c)))

    # Normalize
    p /= np.sum(p)
    return p

# ---- Derived metrics ----


def mmck_metrics(lmbda, mu, c, K):
    p = mmck_probs(lmbda, mu, c, K)
    pK = p[-1]   # blocking prob
    chi = lmbda * (1 - pK)   # carried throughput
    L = np.sum([n * pn for n, pn in enumerate(p)])  # mean number in system

    # Sojourn time for *accepted* jobs
    W = L / chi if chi > 1e-12 else np.inf
    return chi, L, W


if __name__ == "__main__":

    # ---- Sampling domain ----
    np.random.seed(42)
    samples = []
    for _ in range(10000):  # number of random systems
        # Random parameters
        # service rate per server
        mu = 10.0
        # number of servers, at least 1, at most 8
        c = np.random.randint(1, 8)
        # system capacity, at least c, at most c+20
        K = np.random.randint(c + 1, c + 10)
        # arrival rate, between 0.1 and 1.5*c*mu
        # lmbda = np.random.uniform(0.1, 1.5 * c * 1.0)
        lmbda = np.random.exponential(scale=2.0) * c * mu
        # print(lmbda, mu, c, K)

        chi, L, W = mmck_metrics(lmbda, mu, c, K)
        if np.isfinite(W) and chi > 1e-12:
            X = chi / mu          # π2: χ/μ
            Y = lmbda * W / c     # π1: λW/c
            Z = L / K             # π3: L/K
            samples.append((X, Y, Z))

    samples = np.array(samples)

    # ---- Grid for heatmap ----
    grid_size = 50
    Xgrid = np.linspace(0, 2, grid_size)
    Ygrid = np.linspace(0, 5, grid_size)

    Zgrid = np.full((grid_size, grid_size), np.nan)
    counts = defaultdict(list)

    for (X, Y, Z) in samples:
        xi = np.searchsorted(Xgrid, X) - 1
        yi = np.searchsorted(Ygrid, Y) - 1
        if 0 <= xi < grid_size and 0 <= yi < grid_size:
            counts[(xi, yi)].append(Z)

    for (xi, yi), vals in counts.items():
        Zgrid[yi, xi] = np.median(vals)   # median L/K

    # ---- Plot ----
    plt.figure(figsize=(8, 6))
    cs = plt.contourf(Xgrid, Ygrid, Zgrid, levels=15, cmap="viridis")
    cbar = plt.colorbar(cs)
    cbar.set_label("Median L/K (Buffer Utilization)")

    plt.scatter(samples[:, 0], samples[:, 1], s=5, c="white", alpha=0.2)
    plt.xlabel(r"$\chi / \mu$")
    plt.ylabel(r"$\lambda W / c$")
    plt.title(
        "Moody-style Chart for M/M/c/K Queue\nContours show L/K (buffer utilization)")
    plt.show()
