# Prototype: generate a Moody-style chart for M/M/c/K using analytic steady-state formulas.
# X = chi/mu  (carried throughput normalized by single-server capacity)
# Y = lambda * W / c  (normalized occupancy per server)
# Contours = median L/K in grid cells
# We'll sample many primitive parameter sets, compute analytic steady-state p_n, L, chi, W,
# then build a 2D grid and compute median L/K per cell for plotting.
import numpy.ma as ma
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from math import factorial
import warnings

warnings.filterwarnings("ignore")


def mmck_steady_probs(lam, mu, c, K):
    """Compute steady-state probabilities p_n for M/M/c/K birth-death using offered load a=lam/mu.
    Returns p (array length K+1) and p0.
    """
    a = lam / mu
    # compute normalization denominator
    terms = []
    # n = 0..c-1
    for n in range(0, min(c, K) + 1):
        terms.append(a**n / factorial(n))
    # n = c..K if K >= c
    if K >= c+1:
        for n in range(c+1, K+1):
            # careful: for n=c use a**c / c!
            # but loop starts at c+1 so:
            terms.append(a**n / (factorial(c) * (c**(n-c))))
    # if K < c, we already handled up to K in first loop
    denom = sum(terms)
    p0 = 1.0 / denom
    # build p_n
    p = np.zeros(K+1)
    for n in range(0, K+1):
        if n <= c:
            p[n] = p0 * (a**n / factorial(n))
        else:
            p[n] = p0 * (a**n / (factorial(c) * (c**(n-c))))
    return p, p0


def analyze_mmck(lam, mu, c, K):
    """Return tuple (chi, L, W) for M/M/c/K system based on steady-state probabilities.
    - chi = effective throughput (carried rate) = lam * (1 - p_K)
    - L = mean number in system = sum n p_n
    - W = mean time in system for accepted customers = L / chi (Little's law for accepted)
    If chi==0 (all blocked) returns None.
    """
    p, p0 = mmck_steady_probs(lam, mu, c, K)
    pK = p[-1]
    chi = lam * (1 - pK)
    if chi <= 0:
        return None
    L = sum(n * p[n] for n in range(len(p)))
    W = L / chi  # mean sojourn for accepted customers
    return chi, L, W, pK


# Sample many parameter sets
rng = np.random.default_rng(42)
N = 50000

mus = 1.0  # fix mu = 1 for normalization (units)
samples = []
for _ in range(N):
    # sample number of servers c between 1 and 8 (integer)
    c = rng.integers(1, 9)
    # choose K between c and c+20
    K = c + rng.integers(0, 21)
    # sample offered lambda: uniform between 0.01 and c*mu*1.5 (allow overload)
    lam = rng.uniform(0.01, c * mus * 1.5)
    res = analyze_mmck(lam, mus, c, K)
    if res is None:
        continue
    chi, L, W, p_block = res
    # compute pi groups
    pi_x = chi / mus           # chi/mu
    pi_y = (lam * W) / c       # lambda * W / c
    pi_cont = L / K            # L/K
    samples.append((pi_x, pi_y, pi_cont, lam, mus, c, K, chi, L, W, p_block))

samples = np.array(samples, dtype=float)
print("Generated samples:", samples.shape[0])

# Build grid for plotting: choose ranges based on sampled data
x = samples[:, 0]  # chi/mu
y = samples[:, 1]  # lambda*W/c
z = samples[:, 2]  # L/K

# Define a function to ensure limits are valid (not NaN/Inf)
def safe_limit(value, default=1.0):
    if np.isnan(value) or np.isinf(value) or value <= 0:
        return default
    return value

# define grid
nx, ny = 120, 120

# Calculate limits with safety checks
x_lower = safe_limit(np.percentile(x, 1), 0.1)
x_upper = safe_limit(min(np.percentile(x, 99), x.max()*1.05), 1.0)
y_lower = safe_limit(np.percentile(y, 1), 0.1)
y_upper = safe_limit(min(np.percentile(y, 99), y.max()*1.05), 1.0)

x_min, x_max = max(0.01, x_lower), x_upper
y_min, y_max = max(0.01, y_lower), y_upper

# Ensure min < max
if x_min >= x_max:
    x_min, x_max = 0.01, 1.0
if y_min >= y_max:
    y_min, y_max = 0.01, 1.0

xi = np.linspace(x_min, x_max, nx)
yi = np.linspace(y_min, y_max, ny)
Xi, Yi = np.meshgrid(xi, yi)

# compute median z in each cell
Z_med = np.full_like(Xi, np.nan)
counts = np.zeros_like(Xi, dtype=int)

# bin samples
x_idx = np.searchsorted(xi, x) - 1
y_idx = np.searchsorted(yi, y) - 1
# clamp indices
x_idx = np.clip(x_idx, 0, nx-1)
y_idx = np.clip(y_idx, 0, ny-1)

# accumulate lists per cell (to compute median)
cell_vals = defaultdict(list)
for sx, sy, sz, *_ in samples:
    ix = int(np.searchsorted(xi, sx) - 1)
    iy = int(np.searchsorted(yi, sy) - 1)
    if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
        continue
    cell_vals[(iy, ix)].append(sz)

for (iy, ix), vals in cell_vals.items():
    counts[iy, ix] = len(vals)
    # Only calculate median if we have data points
    if len(vals) > 0:
        Z_med[iy, ix] = np.median(vals)

# Plot heatmap of median L/K
fig, ax = plt.subplots(figsize=(8, 6))

# Print the axis limits to check for any issues
print(f"Plotting with x range: {x_min} to {x_max}")
print(f"Plotting with y range: {y_min} to {y_max}")

# Filter out any NaN or inf values for scatter plot
valid_mask = ~np.isnan(x) & ~np.isnan(y) & ~np.isinf(x) & ~np.isinf(y)
x_valid = x[valid_mask]
y_valid = y[valid_mask]

# show samples as scatter for context (light markers)
sc = ax.scatter(x_valid, y_valid, s=6, alpha=0.15, edgecolors='none')

# Replace any remaining NaN values in Z_med with 0
Z_med = np.nan_to_num(Z_med, nan=0.0)

# plot median heatmap using imshow (extent mapping)
# prepare image array: flip vertically because imshow's origin is top
img = np.flipud(Z_med)
extent = [x_min, x_max, y_min, y_max]
im = ax.imshow(img, extent=extent, aspect='auto', origin='lower')

# add contour lines for some levels
levels = [0.1, 0.25, 0.5, 0.75, 0.9]
# mask nan for contour
Z_mask = ma.masked_invalid(Z_med)
CS = ax.contour(Xi, Yi, Z_mask, levels=levels)
ax.clabel(CS, inline=True, fontsize=8, fmt="%.2f")

ax.set_xlabel(r'$\chi/\mu$')
ax.set_ylabel(r'$\lambda W / c$')
ax.set_title(
    'Prototype: median L/K (contours) — M/M/c/K analytic approximation')

plt.colorbar(im, ax=ax, label='median L/K (heatmap)')
plt.tight_layout()

# # Save to file in the current directory
# out_path = "mmck_pi_chart.png"
# fig.savefig(out_path, dpi=150)
# print(f"Plot saved to: {out_path}")

# Explicitly show the plot window
plt.show()
