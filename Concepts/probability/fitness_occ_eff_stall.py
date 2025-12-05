"""
Proof of Concept: Testing Exponential PDF Fitness for Predicting Stall
from Occupation and Efficiency Coefficients

This script models a system where:
- Occupation (Occ): Resource occupation level [0, 1]
- Efficiency (Eff): System efficiency [0, 1]
- Stall: System stall probability/time (predicted from Occ and Eff)

The relationship follows an exponential model:
    Stall = exp(α * Occ + β * Eff + γ)

We'll:
1. Generate synthetic data with exponential relationship
2. Fit the model parameters (α, β, γ)
3. Test fitness of the exponential model
4. Visualize the predictions vs actual values
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
# from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings("ignore")


def generate_system_data(n_samples=1000, noise_level=0.1, seed=42):
    """
    Generate synthetic system data with exponential relationship.

    Stall = exp(α * Occ + β * Eff + γ) + noise

    Args:
        n_samples: Number of data points to generate
        noise_level: Standard deviation of multiplicative noise
        seed: Random seed for reproducibility

    Returns:
        occ, eff, stall arrays
    """
    np.random.seed(seed)

    # True parameters for the exponential model
    alpha_true = 2.5   # Occupation coefficient (positive = more occ → more stall)
    beta_true = -1.8   # Efficiency coefficient (negative = more eff → less stall)
    gamma_true = -1.0  # Baseline intercept

    # Generate occupation and efficiency values
    # Use beta distribution to keep values in [0, 1]
    occ = np.random.beta(2, 2, n_samples)  # Centered around 0.5
    eff = np.random.beta(3, 2, n_samples)  # Slightly higher efficiency

    # Calculate stall using exponential relationship
    linear_combination = alpha_true * occ + beta_true * eff + gamma_true
    stall_true = np.exp(linear_combination)

    # Add multiplicative noise (log-normal)
    noise = np.random.lognormal(0, noise_level, n_samples)
    stall = stall_true * noise

    return occ, eff, stall, (alpha_true, beta_true, gamma_true)


def exponential_model(occ, eff, alpha, beta, gamma):
    """
    Exponential model: Stall = exp(α * Occ + β * Eff + γ)

    Args:
        occ: Occupation values
        eff: Efficiency values
        alpha, beta, gamma: Model parameters

    Returns:
        Predicted stall values
    """
    return np.exp(alpha * occ + beta * eff + gamma)


def fit_exponential_model(occ, eff, stall):
    """
    Fit exponential model parameters using maximum likelihood estimation.

    We minimize: sum((log(stall_pred) - log(stall_actual))^2)

    Args:
        occ: Occupation values
        eff: Efficiency values
        stall: Actual stall values

    Returns:
        Fitted parameters (alpha, beta, gamma)
    """
    # Take log transform for linear regression
    log_stall = np.log(stall)

    # Objective function: sum of squared residuals
    def objective(params):
        alpha, beta, gamma = params
        log_stall_pred = alpha * occ + beta * eff + gamma
        residuals = log_stall - log_stall_pred
        return np.sum(residuals**2)

    # Initial guess
    x0 = [0.0, 0.0, 0.0]

    # Optimize
    result = minimize(objective, x0, method='BFGS')

    return result.x  # Returns (alpha, beta, gamma)


def fit_poisson_model(occ, eff, stall):
    """
    Fit Poisson regression model using maximum likelihood estimation.

    For Poisson regression: λ = exp(α * Occ + β * Eff + γ)
    where λ is the expected rate/count

    The log-likelihood for Poisson:
    LL = sum(stall * log(λ) - λ - log(stall!))

    We maximize the log-likelihood (minimize negative log-likelihood)

    Args:
        occ: Occupation values
        eff: Efficiency values
        stall: Actual stall counts (should be non-negative integers or treated as counts)

    Returns:
        Fitted parameters (alpha, beta, gamma)
    """
    # Negative log-likelihood for Poisson regression
    def neg_log_likelihood(params):
        alpha, beta, gamma = params
        # Calculate lambda (rate parameter)
        lambda_pred = np.exp(alpha * occ + beta * eff + gamma)

        # Avoid numerical issues with very small or large values
        lambda_pred = np.clip(lambda_pred, 1e-10, 1e10)

        # Negative log-likelihood (we minimize this)
        # Note: we drop the log(stall!) term as it doesn't depend on parameters
        nll = -np.sum(stall * np.log(lambda_pred) - lambda_pred)

        return nll

    # Initial guess - use least squares as starting point
    x0 = fit_exponential_model(occ, eff, stall)

    # Optimize using BFGS
    result = minimize(neg_log_likelihood, x0, method='BFGS')

    return result.x  # Returns (alpha, beta, gamma)


def test_poisson_fitness(occ, eff, stall, alpha, beta, gamma):
    """
    Test goodness of fit for Poisson regression model.

    Args:
        occ, eff, stall: Data arrays
        alpha, beta, gamma: Model parameters

    Returns:
        Dictionary with test results
    """
    # Predicted lambda values
    lambda_pred = exponential_model(occ, eff, alpha, beta, gamma)

    results = {}

    # 1. Deviance (measure of goodness of fit)
    # Saturated model has perfect fit (deviance = 0)
    # Higher deviance = worse fit
    saturated_loglik = np.sum(stall * np.log(np.maximum(stall, 1e-10)) - stall)
    model_loglik = np.sum(stall * np.log(lambda_pred) - lambda_pred)
    deviance = 2 * (saturated_loglik - model_loglik)

    results['deviance'] = deviance
    results['deviance_df'] = len(stall) - 3  # degrees of freedom (n - number of parameters)

    # 2. Pearson chi-square statistic
    pearson_residuals = (stall - lambda_pred) / np.sqrt(lambda_pred)
    pearson_chi2 = np.sum(pearson_residuals**2)

    results['pearson_chi2'] = pearson_chi2
    results['pearson_chi2_df'] = len(stall) - 3

    # 3. Overdispersion test
    # For Poisson, variance = mean. If variance > mean, we have overdispersion
    dispersion_param = pearson_chi2 / results['pearson_chi2_df']
    results['dispersion_parameter'] = dispersion_param
    results['is_overdispersed'] = dispersion_param > 1.5  # Rule of thumb

    # 4. AIC (Akaike Information Criterion) - lower is better
    k = 3  # number of parameters
    aic = -2 * model_loglik + 2 * k
    results['aic'] = aic

    # 5. BIC (Bayesian Information Criterion) - lower is better
    n = len(stall)
    bic = -2 * model_loglik + k * np.log(n)
    results['bic'] = bic

    return results


def compute_residuals(occ, eff, stall, alpha, beta, gamma):
    """
    Compute residuals and their statistics.

    Args:
        occ, eff, stall: Data arrays
        alpha, beta, gamma: Model parameters

    Returns:
        Dictionary with residual statistics
    """
    stall_pred = exponential_model(occ, eff, alpha, beta, gamma)

    # Regular residuals
    residuals = stall - stall_pred

    # Log-space residuals (for exponential model, these should be more normally distributed)
    log_residuals = np.log(stall) - np.log(stall_pred)

    # Relative errors
    relative_errors = np.abs(residuals) / stall

    return {
        'residuals': residuals,
        'log_residuals': log_residuals,
        'relative_errors': relative_errors,
        'stall_pred': stall_pred,
        'rmse': np.sqrt(np.mean(residuals**2)),
        'mae': np.mean(np.abs(residuals)),
        'mape': np.mean(relative_errors) * 100,  # Mean Absolute Percentage Error
        'r_squared': 1 - np.sum(residuals**2) / np.sum((stall - np.mean(stall))**2)
    }


def test_exponential_fitness(log_residuals):
    """
    Test if log residuals follow normal distribution (implies exponential model fitness).

    Args:
        log_residuals: Residuals in log space

    Returns:
        Dictionary with test results
    """
    results = {}

    # Shapiro-Wilk test for normality
    stat_sw, p_sw = stats.shapiro(log_residuals)
    results['shapiro_wilk'] = {
        'statistic': stat_sw,
        'p_value': p_sw,
        'is_normal': p_sw > 0.05
    }

    # Kolmogorov-Smirnov test against normal distribution
    stat_ks, p_ks = stats.kstest(log_residuals, 'norm',
                                 args=(np.mean(log_residuals),
                                       np.std(log_residuals)))
    results['ks_test'] = {
        'statistic': stat_ks,
        'p_value': p_ks,
        'is_normal': p_ks > 0.05
    }

    # Anderson-Darling test
    result_ad = stats.anderson(log_residuals, dist='norm')
    critical_5pct = result_ad.critical_values[2] if len(result_ad.critical_values) > 2 else None
    results['anderson_darling'] = {
        'statistic': result_ad.statistic,
        'critical_value_5pct': critical_5pct,
        'is_normal': result_ad.statistic < critical_5pct if critical_5pct else None
    }

    return results


def plot_comprehensive_analysis(occ,
                                eff,
                                stall,
                                alpha,
                                beta,
                                gamma,
                                true_params=None):
    """
    Create comprehensive visualization of the model fit.

    Args:
        occ, eff, stall: Data arrays
        alpha, beta, gamma: Fitted parameters
        true_params: True parameters (if known) for comparison
    """
    residuals_dict = compute_residuals(occ, eff, stall, alpha, beta, gamma)
    stall_pred = residuals_dict['stall_pred']
    log_residuals = residuals_dict['log_residuals']

    fig = plt.figure(figsize=(16, 10))

    # 1. 3D scatter plot: Actual data
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    scatter = ax1.scatter(occ,
                          eff,
                          stall,
                          c=stall,
                          cmap='viridis',
                          alpha=0.6,
                          s=20)
    ax1.set_xlabel('Occupation')
    ax1.set_ylabel('Efficiency')
    ax1.set_zlabel('Stall')
    ax1.set_title('Actual Data: Stall vs Occ & Eff')
    plt.colorbar(scatter, ax=ax1, label='Stall', shrink=0.5)

    # 2. Predicted vs Actual
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(stall, stall_pred, alpha=0.5, s=10)
    
    # Perfect prediction line
    min_val = min(stall.min(), stall_pred.min())
    max_val = max(stall.max(), stall_pred.max())
    ax2.plot([min_val, max_val],
             [min_val, max_val],
             'r--',
             linewidth=2,
             label='Perfect fit')

    ax2.set_xlabel('Actual Stall')
    ax2.set_ylabel('Predicted Stall')
    ax2.set_title(f'Predicted vs Actual (R²={residuals_dict["r_squared"]:.4f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Residuals histogram
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.hist(log_residuals,
             bins=50,
             density=True,
             alpha=0.7,
             edgecolor='black')

    # Overlay normal distribution
    x_range = np.linspace(log_residuals.min(), log_residuals.max(), 100)
    normal_pdf = stats.norm.pdf(x_range, np.mean(log_residuals), np.std(log_residuals))
    ax3.plot(x_range, normal_pdf, 'r-', linewidth=2, label='Normal PDF')

    ax3.set_xlabel('Log Residuals')
    ax3.set_ylabel('Density')
    ax3.set_title('Log Residuals Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Q-Q plot for log residuals
    ax4 = fig.add_subplot(2, 3, 4)
    stats.probplot(log_residuals, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Log Residuals)')
    ax4.grid(True, alpha=0.3)

    # 5. Residuals vs Occupation
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(occ, residuals_dict['residuals'], alpha=0.5, s=10)
    ax5.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax5.set_xlabel('Occupation')
    ax5.set_ylabel('Residuals')
    ax5.set_title('Residuals vs Occupation')
    ax5.grid(True, alpha=0.3)

    # 6. Parameter comparison (if true params known)
    ax6 = fig.add_subplot(2, 3, 6)

    if true_params is not None:
        alpha_true, beta_true, gamma_true = true_params
        params_true = [alpha_true, beta_true, gamma_true]
        params_fitted = [alpha, beta, gamma]
        param_names = ['α (Occ)', 'β (Eff)', 'γ (Int)']

        x_pos = np.arange(len(param_names))
        width = 0.35

        ax6.bar(x_pos - width / 2,
                params_true,
                width,
                label='True',
                alpha=0.8)
        ax6.bar(x_pos + width / 2,
                params_fitted,
                width,
                label='Fitted',
                alpha=0.8)

        ax6.set_xlabel('Parameters')
        ax6.set_ylabel('Value')
        ax6.set_title('Parameter Comparison')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(param_names)
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
    else:
        # Just show fitted parameters
        params_fitted = [alpha, beta, gamma]
        param_names = ['α (Occ)', 'β (Eff)', 'γ (Int)']
        ax6.bar(param_names, params_fitted, alpha=0.8)
        ax6.set_xlabel('Parameters')
        ax6.set_ylabel('Value')
        ax6.set_title('Fitted Parameters')
        ax6.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


def print_fitness_report(occ, eff, stall, alpha, beta, gamma, true_params=None):
    """
    Print comprehensive fitness report.

    Args:
        occ, eff, stall: Data arrays
        alpha, beta, gamma: Fitted parameters
        true_params: True parameters (if known)
    """
    print("=" * 70)
    print("EXPONENTIAL MODEL FITNESS REPORT")
    print("=" * 70)

    # Model equation
    print(f"\nModel: Stall = exp({alpha:.4f} * Occ + {beta:.4f} * Eff + {gamma:.4f})")

    if true_params:
        alpha_true, beta_true, gamma_true = true_params
        print(f"True:  Stall = exp({alpha_true:.4f} * Occ + {beta_true:.4f} * Eff + {gamma_true:.4f})")
        print("\nParameter Errors:")
        _msg = f"\tα error: {abs(alpha - alpha_true):.4f} "
        _msg += f"({abs(alpha - alpha_true) / abs(alpha_true)*100:.2f}%)"
        print(_msg)
        print(f"\tβ error: {abs(beta - beta_true):.4f} ({abs(beta - beta_true) / abs(beta_true)*100:.2f}%)")
        print(f"  γ error: {abs(gamma - gamma_true):.4f} ({abs(gamma - gamma_true) / abs(gamma_true)*100:.2f}%)")

    # Compute residuals
    residuals_dict = compute_residuals(occ, eff, stall, alpha, beta, gamma)
    
    print("\n" + "=" * 70)
    print("GOODNESS OF FIT METRICS")
    print("=" * 70)
    print(f"R² (coefficient of determination): {residuals_dict['r_squared']:.4f}")
    print(f"RMSE (Root Mean Square Error):     {residuals_dict['rmse']:.4f}")
    print(f"MAE (Mean Absolute Error):         {residuals_dict['mae']:.4f}")
    print(f"MAPE (Mean Abs Percentage Error):  {residuals_dict['mape']:.2f}%")
    
    # Test exponential fitness
    fitness_tests = test_exponential_fitness(residuals_dict['log_residuals'])
    
    print("\n" + "=" * 70)
    print("NORMALITY TESTS (Log Residuals)")
    print("=" * 70)
    
    sw = fitness_tests['shapiro_wilk']
    print(f"\n1. Shapiro-Wilk Test:")
    print(f"   Statistic: {sw['statistic']:.4f}")
    print(f"   p-value:   {sw['p_value']:.4f}")
    print(f"   Result:    {'Normal ✓' if sw['is_normal'] else 'Not Normal ✗'}")
    
    ks = fitness_tests['ks_test']
    print(f"\n2. Kolmogorov-Smirnov Test:")
    print(f"   Statistic: {ks['statistic']:.4f}")
    print(f"   p-value:   {ks['p_value']:.4f}")
    print(f"   Result:    {'Normal ✓' if ks['is_normal'] else 'Not Normal ✗'}")
    
    ad = fitness_tests['anderson_darling']
    print(f"\n3. Anderson-Darling Test:")
    print(f"   Statistic: {ad['statistic']:.4f}")
    if ad['critical_value_5pct']:
        print(f"   Critical:  {ad['critical_value_5pct']:.4f} (5% level)")
        print(f"   Result:    {'Normal ✓' if ad['is_normal'] else 'Not Normal ✗'}")
    
    # Overall assessment
    normal_count = sum([
        sw['is_normal'],
        ks['is_normal'],
        ad['is_normal'] if ad['is_normal'] is not None else False
    ])
    
    print("\n" + "=" * 70)
    print(f"OVERALL ASSESSMENT: {normal_count}/3 tests indicate normality")
    print("=" * 70)
    
    if residuals_dict['r_squared'] > 0.9 and normal_count >= 2:
        print("✓ EXCELLENT FIT: Exponential model is appropriate")
    elif residuals_dict['r_squared'] > 0.7 and normal_count >= 2:
        print("✓ GOOD FIT: Exponential model is acceptable")
    elif residuals_dict['r_squared'] > 0.5:
        print("⚠ MODERATE FIT: Consider model refinement")
    else:
        print("✗ POOR FIT: Exponential model may not be appropriate")
    
    print("=" * 70)


def print_poisson_fitness_report(occ, eff, stall, alpha, beta, gamma, true_params=None):
    """
    Print comprehensive Poisson regression fitness report.
    
    Args:
        occ, eff, stall: Data arrays
        alpha, beta, gamma: Fitted parameters
        true_params: True parameters (if known)
    """
    print("=" * 70)
    print("POISSON REGRESSION MODEL FITNESS REPORT")
    print("=" * 70)
    
    # Model equation
    print(f"\nModel: λ = exp({alpha:.4f} * Occ + {beta:.4f} * Eff + {gamma:.4f})")
    print("       Stall ~ Poisson(λ)")
    
    if true_params:
        alpha_true, beta_true, gamma_true = true_params
        print(f"True:  λ = exp({alpha_true:.4f} * Occ + {beta_true:.4f} * Eff + {gamma_true:.4f})")
        print(f"\nParameter Errors:")
        print(f"  α error: {abs(alpha - alpha_true):.4f} ({abs(alpha - alpha_true)/abs(alpha_true)*100:.2f}%)")
        print(f"  β error: {abs(beta - beta_true):.4f} ({abs(beta - beta_true)/abs(beta_true)*100:.2f}%)")
        print(f"  γ error: {abs(gamma - gamma_true):.4f} ({abs(gamma - gamma_true)/abs(gamma_true)*100:.2f}%)")
    
    # Compute residuals
    residuals_dict = compute_residuals(occ, eff, stall, alpha, beta, gamma)
    
    # Poisson-specific tests
    poisson_tests = test_poisson_fitness(occ, eff, stall, alpha, beta, gamma)
    
    print("\n" + "=" * 70)
    print("GOODNESS OF FIT METRICS")
    print("=" * 70)
    print(f"R² (pseudo R-squared):              {residuals_dict['r_squared']:.4f}")
    print(f"RMSE (Root Mean Square Error):     {residuals_dict['rmse']:.4f}")
    print(f"MAE (Mean Absolute Error):         {residuals_dict['mae']:.4f}")
    print(f"MAPE (Mean Abs Percentage Error):  {residuals_dict['mape']:.2f}%")
    
    print("\n" + "=" * 70)
    print("POISSON-SPECIFIC TESTS")
    print("=" * 70)
    
    print(f"\n1. Deviance Test:")
    print(f"   Deviance:      {poisson_tests['deviance']:.4f}")
    print(f"   Degrees of freedom: {poisson_tests['deviance_df']}")
    print(f"   Deviance/df:   {poisson_tests['deviance']/poisson_tests['deviance_df']:.4f}")
    print(f"   (Values close to 1 indicate good fit)")
    
    print(f"\n2. Pearson Chi-Square Test:")
    print(f"   Chi-square:    {poisson_tests['pearson_chi2']:.4f}")
    print(f"   Degrees of freedom: {poisson_tests['pearson_chi2_df']}")
    print(f"   Chi-square/df: {poisson_tests['pearson_chi2']/poisson_tests['pearson_chi2_df']:.4f}")
    
    print(f"\n3. Overdispersion Test:")
    print(f"   Dispersion parameter: {poisson_tests['dispersion_parameter']:.4f}")
    print(f"   Status: {'Overdispersed ⚠' if poisson_tests['is_overdispersed'] else 'Appropriate ✓'}")
    print(f"   (Value > 1.5 suggests overdispersion; consider Negative Binomial)")
    
    print(f"\n4. Information Criteria:")
    print(f"   AIC (Akaike):  {poisson_tests['aic']:.2f} (lower is better)")
    print(f"   BIC (Bayesian): {poisson_tests['bic']:.2f} (lower is better)")

    # Overall assessment
    print("\n" + "=" * 70)
    print("OVERALL ASSESSMENT")
    print("=" * 70)
    
    deviance_ratio = poisson_tests['deviance'] / poisson_tests['deviance_df']
    
    if residuals_dict['r_squared'] > 0.9 and not poisson_tests['is_overdispersed'] and 0.8 < deviance_ratio < 1.2:
        print("✓ EXCELLENT FIT: Poisson model is highly appropriate")
    elif residuals_dict['r_squared'] > 0.7 and deviance_ratio < 1.5:
        print("✓ GOOD FIT: Poisson model is acceptable")
    elif poisson_tests['is_overdispersed']:
        print("⚠ OVERDISPERSION DETECTED: Consider Negative Binomial model")
    elif residuals_dict['r_squared'] > 0.5:
        print("⚠ MODERATE FIT: Model may need refinement")
    else:
        print("✗ POOR FIT: Poisson model may not be appropriate")
    
    print("=" * 70)


def compare_models(occ, eff, stall, alpha_exp, beta_exp, gamma_exp,
                   alpha_pois, beta_pois, gamma_pois):
    """
    Compare Exponential and Poisson model fits.

    Args:
        occ, eff, stall: Data arrays
        alpha_exp, beta_exp, gamma_exp: Exponential model parameters
        alpha_pois, beta_pois, gamma_pois: Poisson model parameters
    """
    print("\n" + "=" * 70)
    print("MODEL COMPARISON: EXPONENTIAL vs POISSON")
    print("=" * 70)

    # Compute metrics for both models
    residuals_exp = compute_residuals(occ, eff, stall, alpha_exp, beta_exp, gamma_exp)
    residuals_pois = compute_residuals(occ, eff, stall, alpha_pois, beta_pois, gamma_pois)
    poisson_tests = test_poisson_fitness(occ, eff, stall, alpha_pois, beta_pois, gamma_pois)
    
    print("\n{:<30} {:>15} {:>15}".format("Metric", "Exponential", "Poisson"))
    print("-" * 70)
    print("{:<30} {:>15.4f} {:>15.4f}".format("R²", residuals_exp['r_squared'], residuals_pois['r_squared']))
    print("{:<30} {:>15.4f} {:>15.4f}".format("RMSE", residuals_exp['rmse'], residuals_pois['rmse']))
    print("{:<30} {:>15.4f} {:>15.4f}".format("MAE", residuals_exp['mae'], residuals_pois['mae']))
    print("{:<30} {:>15.2f}% {:>14.2f}%".format("MAPE", residuals_exp['mape'], residuals_pois['mape']))
    print("{:<30} {:>15} {:>15.2f}".format("AIC", "N/A", poisson_tests['aic']))
    print("{:<30} {:>15} {:>15.2f}".format("BIC", "N/A", poisson_tests['bic']))
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    
    if residuals_exp['r_squared'] > residuals_pois['r_squared'] + 0.05:
        print("→ Exponential model provides better fit (higher R²)")
    elif residuals_pois['r_squared'] > residuals_exp['r_squared'] + 0.05:
        print("→ Poisson model provides better fit (higher R²)")
    else:
        print("→ Both models provide similar fit quality")
    
    if poisson_tests['is_overdispersed']:
        print("⚠ Poisson shows overdispersion - consider Negative Binomial")
    
    print("=" * 70)


# Main execution
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PROOF OF CONCEPT: OCC & EFF → STALL")
    print("Comparing Exponential and Poisson Models")
    print("=" * 70)
    
    # Example 1: Low noise data - fits both models
    print("\n\nEXAMPLE 1: Low Noise Data")
    print("-" * 70)
    
    occ1, eff1, stall1, true_params1 = generate_system_data(
        n_samples=1000, noise_level=0.05, seed=42
    )
    
    # Fit Exponential model
    print("\nFitting Exponential Model...")
    alpha1_exp, beta1_exp, gamma1_exp = fit_exponential_model(occ1, eff1, stall1)
    print_fitness_report(occ1, eff1, stall1, alpha1_exp, beta1_exp, gamma1_exp, true_params1)
    
    # Fit Poisson model
    print("\n\nFitting Poisson Regression Model...")
    alpha1_pois, beta1_pois, gamma1_pois = fit_poisson_model(occ1, eff1, stall1)
    print_poisson_fitness_report(occ1, eff1, stall1, alpha1_pois, beta1_pois, gamma1_pois, true_params1)
    
    # Compare models
    compare_models(occ1, eff1, stall1, alpha1_exp, beta1_exp, gamma1_exp,
                   alpha1_pois, beta1_pois, gamma1_pois)
    
    # Plot
    fig1 = plot_comprehensive_analysis(occ1, eff1, stall1, alpha1_exp, beta1_exp, gamma1_exp,
                                        true_params1)
    fig1.suptitle('Example 1: Low Noise Data (Exponential Model)', fontsize=14, y=0.995)
    
    # Example 2: Higher noise
    print("\n\nEXAMPLE 2: Higher Noise Data")
    print("-" * 70)
    
    occ2, eff2, stall2, true_params2 = generate_system_data(
        n_samples=1000, noise_level=0.2, seed=123
    )
    
    # Fit both models
    print("\nFitting Exponential Model...")
    alpha2_exp, beta2_exp, gamma2_exp = fit_exponential_model(occ2, eff2, stall2)
    print_fitness_report(occ2, eff2, stall2, alpha2_exp, beta2_exp, gamma2_exp, true_params2)
    
    print("\n\nFitting Poisson Regression Model...")
    alpha2_pois, beta2_pois, gamma2_pois = fit_poisson_model(occ2, eff2, stall2)
    print_poisson_fitness_report(occ2, eff2, stall2, alpha2_pois, beta2_pois, gamma2_pois, true_params2)
    
    # Compare models
    compare_models(occ2, eff2, stall2, alpha2_exp, beta2_exp, gamma2_exp,
                   alpha2_pois, beta2_pois, gamma2_pois)
    
    # Plot
    fig2 = plot_comprehensive_analysis(occ2, eff2, stall2, alpha2_exp, beta2_exp, gamma2_exp,
                                        true_params2)
    fig2.suptitle('Example 2: Higher Noise Data (Exponential Model)', fontsize=14, y=0.995)
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("\nEXPONENTIAL MODEL (Continuous Data):")
    print("  • Use when: Stall is a continuous positive variable (time, probability)")
    print("  • Interpretation: Log-linear relationship")
    print("  • Test: Log-residuals should be normally distributed")
    print("")
    print("POISSON MODEL (Count Data):")
    print("  • Use when: Stall represents counts (number of stall events)")
    print("  • Interpretation: Rate parameter follows log-linear relationship")
    print("  • Test: Check deviance and overdispersion")
    print("")
    print("PARAMETER MEANINGS:")
    print("  • α > 0: Higher Occupation → More Stalls (resource contention)")
    print("  • β < 0: Higher Efficiency → Fewer Stalls (better performance)")
    print("  • γ: Baseline log-rate/log-value")
    print("")
    print("MODEL SELECTION:")
    print("  • Compare R², AIC, BIC")
    print("  • Check residual patterns")
    print("  • Consider data type (continuous vs count)")
    print("  • If overdispersed: Try Negative Binomial for counts")
    print("=" * 70)
    
    plt.show()
