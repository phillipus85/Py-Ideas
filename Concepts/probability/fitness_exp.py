"""
Proof of Concept: Testing Fitness Against an Exponential PDF

This script demonstrates different methods to test how well data fits
an exponential distribution:
1. Visual comparison (histogram vs theoretical PDF)
2. Q-Q plot (quantile-quantile plot)
3. Statistical tests (Kolmogorov-Smirnov, Anderson-Darling, Chi-square)
4. Maximum Likelihood Estimation (MLE) parameter fitting
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")


def generate_exponential_data(lambda_param, size=1000, noise_level=0.0):
    """
    Generate sample data from exponential distribution with optional noise.

    Args:
        lambda_param: Rate parameter (1/mean) of exponential distribution
        size: Number of samples to generate
        noise_level: Fraction of data to contaminate with noise (0 to 1)

    Returns:
        Array of generated data
    """
    # Generate exponential data
    data = np.random.exponential(scale=1 / lambda_param, size=size)

    # Add noise if specified (contamination with uniform distribution)
    if noise_level > 0:
        n_noise = int(size * noise_level)
        noise_indices = np.random.choice(size, n_noise, replace=False)
        data[noise_indices] = np.random.uniform(0, data.max() * 2, n_noise)

    return data


def fit_exponential_mle(data):
    """
    Fit exponential distribution using Maximum Likelihood Estimation.

    Args:
        data: Sample data

    Returns:
        Estimated lambda parameter (rate)
    """
    # MLE for exponential: lambda = 1 / mean
    return 1 / np.mean(data)


def exponential_pdf(x, lambda_param):
    """
    Exponential probability density function.

    Args:
        x: Input values
        lambda_param: Rate parameter

    Returns:
        PDF values
    """
    return lambda_param * np.exp(-lambda_param * x)


def plot_visual_fitness(data, lambda_param, title="Visual Fitness Test"):
    """
    Create visual comparison between data histogram and theoretical PDF.

    Args:
        data: Sample data
        lambda_param: Rate parameter for exponential distribution
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram with PDF overlay
    ax1.hist(data, bins=50, density=True, alpha=0.6, color='blue',
             edgecolor='black', label='Data histogram')

    x_range = np.linspace(0, data.max(), 1000)
    theoretical_pdf = exponential_pdf(x_range, lambda_param)
    ax1.plot(x_range, theoretical_pdf, 'r-', linewidth=2,
             label=f'Exponential PDF (λ={lambda_param:.3f})')

    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Q-Q plot
    # Generate theoretical quantiles
    sorted_data = np.sort(data)
    n = len(sorted_data)
    theoretical_quantiles = -np.log(1 - np.arange(1, n + 1) / (n + 1)) / lambda_param

    ax2.scatter(theoretical_quantiles, sorted_data, alpha=0.5, s=10)

    # Add reference line
    min_val = min(theoretical_quantiles.min(), sorted_data.min())
    max_val = max(theoretical_quantiles.max(), sorted_data.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
             label='Perfect fit')

    ax2.set_xlabel('Theoretical Quantiles')
    ax2.set_ylabel('Sample Quantiles')
    ax2.set_title('Q-Q Plot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def kolmogorov_smirnov_test(data, lambda_param):
    """
    Perform Kolmogorov-Smirnov test for exponential distribution.

    Args:
        data: Sample data
        lambda_param: Rate parameter

    Returns:
        Dictionary with test results
    """
    # KS test using scipy
    statistic, p_value = stats.kstest(data, 'expon', args=(0, 1 / lambda_param))

    return {
        'statistic': statistic,
        'p_value': p_value,
        'reject_null': p_value < 0.05,
        'interpretation': 'Good fit' if p_value >= 0.05 else 'Poor fit'
    }


def anderson_darling_test(data, lambda_param):
    """
    Perform Anderson-Darling test for exponential distribution.

    Args:
        data: Sample data (should be already scaled by lambda_param)
        lambda_param: Rate parameter

    Returns:
        Dictionary with test results
    """
    # Scale data to standard exponential
    scaled_data = data * lambda_param

    # Anderson-Darling test
    result = stats.anderson(scaled_data, dist='expon')

    # Check against 5% significance level (usually index 2)
    critical_value_5pct = result.critical_values[2] if len(result.critical_values) > 2 else None

    return {
        'statistic': result.statistic,
        'critical_values': result.critical_values,
        'significance_levels': result.significance_level,
        'reject_null_5pct': result.statistic > critical_value_5pct if critical_value_5pct else None,
        'interpretation': 'Poor fit' if (critical_value_5pct and result.statistic > critical_value_5pct) else 'Good fit'
    }


def chi_square_test(data, lambda_param, n_bins=10):
    """
    Perform Chi-square goodness-of-fit test.

    Args:
        data: Sample data
        lambda_param: Rate parameter
        n_bins: Number of bins to use

    Returns:
        Dictionary with test results
    """
    # Create bins
    bins = np.linspace(0, data.max(), n_bins + 1)
    observed_freq, _ = np.histogram(data, bins=bins)

    # Calculate expected frequencies
    expected_prob = np.diff(stats.expon.cdf(bins, scale=1 / lambda_param))
    expected_freq = expected_prob * len(data)

    # Remove bins with too few expected counts
    mask = expected_freq >= 5
    observed_freq = observed_freq[mask]
    expected_freq = expected_freq[mask]

    # Chi-square test
    chi2_stat = np.sum((observed_freq - expected_freq)**2 / expected_freq)
    dof = len(observed_freq) - 2  # -1 for bins, -1 for estimated parameter
    p_value = 1 - stats.chi2.cdf(chi2_stat, dof)

    return {
        'statistic': chi2_stat,
        'degrees_of_freedom': dof,
        'p_value': p_value,
        'reject_null': p_value < 0.05,
        'interpretation': 'Good fit' if p_value >= 0.05 else 'Poor fit'
    }


def comprehensive_fitness_test(data, lambda_param=None, verbose=True):
    """
    Perform comprehensive fitness testing against exponential distribution.

    Args:
        data: Sample data
        lambda_param: Rate parameter (if None, will estimate from data)
        verbose: Whether to print detailed results

    Returns:
        Dictionary with all test results
    """
    # Estimate lambda if not provided
    if lambda_param is None:
        lambda_param = fit_exponential_mle(data)
        if verbose:
            print(f"Estimated λ using MLE: {lambda_param:.4f}")
            print(f"Estimated mean: {1 / lambda_param:.4f}")
            print(f"Sample mean: {np.mean(data):.4f}\n")

    results = {
        'lambda_param': lambda_param,
        'sample_size': len(data),
        'sample_mean': np.mean(data),
        'sample_std': np.std(data),
        'theoretical_mean': 1 / lambda_param,
        'theoretical_std': 1 / lambda_param
    }

    # Perform tests
    if verbose:
        print("=" * 60)
        print("FITNESS TEST RESULTS")
        print("=" * 60)

    # Kolmogorov-Smirnov test
    ks_result = kolmogorov_smirnov_test(data, lambda_param)
    results['ks_test'] = ks_result

    if verbose:
        print("\n1. Kolmogorov-Smirnov Test:")
        print(f"   Statistic: {ks_result['statistic']:.4f}")
        print(f"   p-value: {ks_result['p_value']:.4f}")
        print(f"   Result: {ks_result['interpretation']}")

    # Anderson-Darling test
    ad_result = anderson_darling_test(data, lambda_param)
    results['ad_test'] = ad_result

    if verbose:
        print("\n2. Anderson-Darling Test:")
        print(f"   Statistic: {ad_result['statistic']:.4f}")
        print(f"   Result: {ad_result['interpretation']}")

    # Chi-square test
    chi2_result = chi_square_test(data, lambda_param)
    results['chi2_test'] = chi2_result

    if verbose:
        print("\n3. Chi-square Test:")
        print(f"   Statistic: {chi2_result['statistic']:.4f}")
        print(f"   Degrees of freedom: {chi2_result['degrees_of_freedom']}")
        print(f"   p-value: {chi2_result['p_value']:.4f}")
        print(f"   Result: {chi2_result['interpretation']}")

    # Overall assessment
    good_fits = sum([
        ks_result['interpretation'] == 'Good fit',
        ad_result['interpretation'] == 'Good fit',
        chi2_result['interpretation'] == 'Good fit'
    ])

    if verbose:
        print("\n" + "=" * 60)
        print(f"OVERALL ASSESSMENT: {good_fits}/3 tests indicate good fit")
        print("=" * 60)

    results['overall_good_fits'] = good_fits
    results['overall_assessment'] = 'Good fit' if good_fits >= 2 else 'Poor fit'

    return results


# Example usage and demonstrations
if __name__ == "__main__":
    print("EXPONENTIAL DISTRIBUTION FITNESS TEST - PROOF OF CONCEPT\n")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Example 1: Perfect exponential data
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Perfect Exponential Data")
    print("=" * 70)

    true_lambda = 2.0
    data_perfect = generate_exponential_data(true_lambda,
                                             size=1000, noise_level=0.0)

    results_perfect = comprehensive_fitness_test(data_perfect, verbose=True)
    fig1 = plot_visual_fitness(data_perfect, results_perfect['lambda_param'],
                               "Perfect Exponential Data")

    # Example 2: Exponential data with noise
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Exponential Data with 10% Noise")
    print("=" * 70)

    data_noisy = generate_exponential_data(true_lambda,
                                           size=1000, noise_level=0.1)

    results_noisy = comprehensive_fitness_test(data_noisy, verbose=True)
    fig2 = plot_visual_fitness(data_noisy, results_noisy['lambda_param'],
                               "Exponential Data with 10% Noise")

    # Example 3: Non-exponential data (normal distribution)
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Non-Exponential Data (Normal Distribution)")
    print("=" * 70)

    data_normal = np.abs(np.random.normal(2, 1, 1000))  # Use absolute to keep positive

    results_normal = comprehensive_fitness_test(data_normal, verbose=True)
    fig3 = plot_visual_fitness(data_normal, results_normal['lambda_param'],
                               "Non-Exponential Data (Normal)")

    # Show all plots
    plt.show()

    print("\n" + "=" * 70)
    print("PROOF OF CONCEPT COMPLETE")
    print("=" * 70)
    print("\nKey takeaways:")
    print("1. Use visual inspection (histogram + Q-Q plot) first")
    print("2. Multiple statistical tests provide robust assessment")
    print("3. MLE provides good parameter estimates")
    print("4. p-value > 0.05 suggests good fit (for KS and Chi-square)")
    print("5. Q-Q plot should show points near the diagonal line for good fit")
