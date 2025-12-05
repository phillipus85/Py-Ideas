# Proof of Concept: Occupation & Efficiency → Stall (Exponential Model)

## Overview

This proof of concept demonstrates how to test the fitness of an exponential PDF model where two coefficients (Occupation and Efficiency) predict a third variable (Stall).

## Model Description

**Mathematical Model:**
```
Stall = exp(α × Occupation + β × Efficiency + γ)
```

Where:
- **Occupation (Occ)**: Resource occupation level [0, 1]
- **Efficiency (Eff)**: System efficiency [0, 1]  
- **Stall**: System stall probability or time (positive real number)
- **α**: Occupation coefficient (expected positive - more occupation → more stall)
- **β**: Efficiency coefficient (expected negative - more efficiency → less stall)
- **γ**: Baseline intercept

## Files

- `fitness_occ_eff_stall.py`: Main proof of concept script

## Key Features

### 1. Data Generation
- Generates synthetic data following exponential relationship
- Configurable noise levels to test robustness
- Uses realistic parameter ranges

### 2. Model Fitting
- Maximum Likelihood Estimation (MLE) for parameters
- Log-transform for linear regression approach
- Robust optimization using BFGS method

### 3. Fitness Testing
Comprehensive tests to validate exponential model:
- **R² Score**: Measures predictive power
- **RMSE/MAE/MAPE**: Error metrics
- **Shapiro-Wilk Test**: Tests normality of log-residuals
- **Kolmogorov-Smirnov Test**: Distribution comparison
- **Anderson-Darling Test**: More sensitive normality test

### 4. Visualization
Six comprehensive plots:
1. 3D scatter of actual data (Occ, Eff, Stall)
2. Predicted vs Actual scatter plot
3. Log-residuals histogram with normal overlay
4. Q-Q plot for normality assessment
5. Residuals vs Occupation plot
6. Parameter comparison (true vs fitted)

## Usage

```python
# Generate data
occ, eff, stall, true_params = generate_system_data(
    n_samples=1000,
    noise_level=0.1,
    seed=42
)

# Fit model
alpha, beta, gamma = fit_exponential_model(occ, eff, stall)

# Print fitness report
print_fitness_report(occ, eff, stall, alpha, beta, gamma, true_params)

# Visualize
fig = plot_comprehensive_analysis(occ, eff, stall, alpha, beta, gamma, true_params)
plt.show()
```

## Interpreting Results

### Good Fit Indicators:
- ✓ R² > 0.9 (excellent) or > 0.7 (good)
- ✓ Log-residuals pass normality tests (p-value > 0.05)
- ✓ Q-Q plot points near diagonal line
- ✓ Residuals randomly scattered around zero
- ✓ Low MAPE (< 20%)

### Poor Fit Indicators:
- ✗ R² < 0.5
- ✗ Log-residuals fail normality tests
- ✗ Q-Q plot shows systematic deviations
- ✗ Residuals show patterns (not random)
- ✗ High MAPE (> 50%)

## Expected Behavior

For a well-functioning system:
- **α > 0**: Higher occupation leads to more stalls
- **β < 0**: Higher efficiency reduces stalls
- **γ**: Baseline stall rate (typically negative)

## Examples Included

1. **Low Noise**: Near-perfect exponential relationship
2. **High Noise**: Tests robustness with 20% noise contamination

## Requirements

```
numpy
matplotlib
scipy
```

## Extensions

To use with your own data:

```python
# Load your data
occ = your_occupation_data  # Array of occupation values
eff = your_efficiency_data  # Array of efficiency values
stall = your_stall_data     # Array of stall measurements

# Fit and test
alpha, beta, gamma = fit_exponential_model(occ, eff, stall)
print_fitness_report(occ, eff, stall, alpha, beta, gamma)
plot_comprehensive_analysis(occ, eff, stall, alpha, beta, gamma)
plt.show()
```

## Statistical Background

The exponential model is appropriate when:
1. The response variable (Stall) is always positive
2. The relationship is multiplicative rather than additive
3. Log-residuals are normally distributed
4. The effect of predictors is exponential rather than linear

By testing log-residuals for normality, we validate that the exponential model is the right choice.
