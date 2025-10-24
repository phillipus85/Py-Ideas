import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict

# ---- Helper: steady state probabilities for M/M/c/K ----


def mmck_probs(lmbda, mu, c, K):
    p = np.zeros(K + 1)
    for n in range(0, K + 1):
        if n < c:
            p[n] = (lmbda / mu)**n / math.factorial(n)
        else:
            p[n] = (lmbda / mu)**n / (math.factorial(c) * (c**(n - c)))
    p /= np.sum(p)
    return p

# ---- Derived metrics ----


def mmck_metrics(lmbda, mu, c, K):
    p = mmck_probs(lmbda, mu, c, K)
    pK = p[-1]
    chi = lmbda * (1 - pK)         # carried throughput
    L = np.sum([n * pn for n, pn in enumerate(p)])
    W = L / chi if chi > 1e-12 else np.inf
    return chi, L, W


# ---- Sampling domain ----
samples = []
np.random.seed(42)
for _ in range(3000):
    mu = 1.0
    c = np.random.randint(1, 9)       # servers 1..8
    K = np.random.randint(c + 1, c + 21)  # buffer c+1 .. c+20
    lmbda = np.random.uniform(0.1, 1.5 * c * mu)

    chi, L, W = mmck_metrics(lmbda, mu, c, K)
    if np.isfinite(W) and chi > 1e-12:
        X = chi / mu          # π2
        Y = lmbda * W / c     # π1
        Z = L / K             # π3
        samples.append((X, Y, Z))

samples = np.array(samples)

# ---- Grid for contour ----
grid_size = 60
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
    Zgrid[yi, xi] = np.median(vals)

# ---- Plot as contour lines only ----
plt.figure(figsize=(8, 6))
cs = plt.contour(Xgrid, Ygrid, Zgrid, levels=10, colors="black")
plt.clabel(cs, inline=True, fontsize=8, fmt="%.2f")

plt.xlabel(r"$\chi / \mu$")
plt.ylabel(r"$\lambda W / c$")
plt.title(
    "Moody-style Contour Chart for M/M/c/K Queue\nLines show L/K (buffer utilization)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
