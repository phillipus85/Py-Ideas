"""
Quick Example: Using Both Exponential and Poisson Models

This demonstrates how to fit and compare both models with your own data.
"""

import numpy as np
from fitness_occ_eff_stall import (
    fit_exponential_model,
    fit_poisson_model,
    print_fitness_report,
    print_poisson_fitness_report,
    compare_models,
    plot_comprehensive_analysis
)
import matplotlib.pyplot as plt

# Example: Your own data
# Replace these with your actual measurements
occ = np.array([0.3, 0.5, 0.7, 0.4, 0.6, 0.8, 0.2, 0.5, 0.9, 0.4])
eff = np.array([0.8, 0.7, 0.6, 0.9, 0.7, 0.5, 0.95, 0.75, 0.4, 0.85])
stall = np.array([0.5, 1.2, 2.8, 0.3, 1.5, 3.5, 0.1, 0.9, 5.2, 0.6])

print("=" * 70)
print("FITTING YOUR DATA TO EXPONENTIAL AND POISSON MODELS")
print("=" * 70)

# Method 1: Exponential Model (for continuous stall values)
print("\n1. EXPONENTIAL MODEL")
print("-" * 70)
alpha_exp, beta_exp, gamma_exp = fit_exponential_model(occ, eff, stall)
print_fitness_report(occ, eff, stall, alpha_exp, beta_exp, gamma_exp)

# Method 2: Poisson Model (for count data)
print("\n\n2. POISSON REGRESSION MODEL")
print("-" * 70)
alpha_pois, beta_pois, gamma_pois = fit_poisson_model(occ, eff, stall)
print_poisson_fitness_report(occ, eff, stall, alpha_pois, beta_pois, gamma_pois)

# Compare both models
compare_models(occ, eff, stall, alpha_exp, beta_exp, gamma_exp,
               alpha_pois, beta_pois, gamma_pois)

# Visualize (using exponential model)
fig = plot_comprehensive_analysis(occ, eff, stall, alpha_exp, beta_exp, gamma_exp)
fig.suptitle('Your Data Analysis', fontsize=14, y=0.995)

plt.show()

print("\n" + "=" * 70)
print("WHICH MODEL TO USE?")
print("=" * 70)
print("""
EXPONENTIAL MODEL:
  ✓ Stall is continuous (time, probability, rate)
  ✓ Always positive values
  ✓ Multiplicative effects
  ✗ Not for count data

POISSON MODEL:
  ✓ Stall is count data (number of events)
  ✓ Non-negative integers
  ✓ Handles rare events well
  ✗ Assumes variance = mean (check overdispersion)

TIPS:
  • If overdispersed (dispersion > 1.5), try Negative Binomial
  • Compare AIC/BIC - lower is better
  • Check residual plots for patterns
  • Consider domain knowledge about your data type
""")
