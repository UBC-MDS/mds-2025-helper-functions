import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t, chi2

def htv(test_output, test_type="z", alpha=0.05):
    """
    Visualize Type I and Type II errors for a hypothesis test based on its output.

    Parameters:
        test_output (dict): Output from the hypothesis test containing:
                            - 'mu0': Mean under null hypothesis (H0)
                            - 'mu1': Mean under alternative hypothesis (H1)
                            - 'sigma': Standard deviation (z or t test)
                            - 'sample_size': Sample size (for z and t tests)
                            - 'df1': Degrees of freedom 1 (for F tests, optional)
                            - 'df2': Degrees of freedom 2 (for F tests, optional)
                            - 'df': Degrees of freedom (for t and chi-squared tests, optional)
        test_type (str): Type of test ('z', 't', 'chi2', 'anova').
        alpha (float): Significance level (Type I error rate).

    Returns:
        tuple: (fig, ax) Matplotlib figure and axes objects for the plot.
    """
    mu0 = test_output.get("mu0", 0)
    mu1 = test_output.get("mu1", 1)
    sigma = test_output.get("sigma", 1)
    sample_size = test_output.get("sample_size", 30)
    df = test_output.get("df", None)
    df1 = test_output.get("df1", None)
    df2 = test_output.get("df2", None)

    # Define critical value and distribution
    if test_type == "z":
        critical_value = norm.ppf(1 - alpha, loc=mu0, scale=sigma / np.sqrt(sample_size))
        dist_null = lambda x: norm.pdf(x, loc=mu0, scale=sigma / np.sqrt(sample_size))
        dist_alt = lambda x: norm.pdf(x, loc=mu1, scale=sigma / np.sqrt(sample_size))
    elif test_type == "t":
        if df is None:
            df = sample_size - 1  # Default degrees of freedom
        critical_value = t.ppf(1 - alpha, df=df)
        dist_null = lambda x: t.pdf(x, df=df)
        dist_alt = lambda x: t.pdf(x, df=df, loc=mu1 - mu0)  # Shifted alternative
    elif test_type == "chi2":
        if df is None:
            raise ValueError("Degrees of freedom (df) must be specified for chi-squared tests.")
        critical_value = chi2.ppf(1 - alpha, df=df)
        dist_null = lambda x: chi2.pdf(x, df=df)
        dist_alt = lambda x: chi2.pdf(x, df=df + 1)  # Shifted alternative
    elif test_type == "anova":
        if df1 is None or df2 is None:
            raise ValueError("Degrees of freedom (df1, df2) must be specified for ANOVA tests.")
        critical_value = t.ppf(1 - alpha, df=df1)
        dist_null = lambda x: t.pdf(x, df=df1)
        dist_alt = lambda x: t.pdf(x, df=df2, loc=mu1 - mu0)  # Shifted alternative
    else:
        raise ValueError("Invalid test type. Choose 'z', 't', 'chi2', or 'anova'.")

    # Generate x values
    x = np.linspace(mu0 - 4 * sigma, mu1 + 4 * sigma, 1000)

    # Null and alternative distributions
    y_null = dist_null(x)
    y_alt = dist_alt(x)

    # Plot distributions
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, y_null, label="Null Distribution (H0)", color="blue")
    ax.plot(x, y_alt, label="Alternative Distribution (H1)", color="red")

    # Fill Type I and Type II errors
    ax.fill_between(x, 0, y_null, where=(x > critical_value), color="orange", alpha=0.5, label="Type I Error (α)")
    ax.fill_between(x, 0, y_alt, where=(x <= critical_value), color="green", alpha=0.5, label="Type II Error (β)")

    # Add annotations
    ax.axvline(x=critical_value, color="black", linestyle="--", label="Critical Value")
    ax.text(critical_value, max(y_null) * 0.6, f"Critical Value = {critical_value:.2f}", rotation=90, fontsize=10)

    # Labels and legend
    ax.set_title(f"Type I and Type II Errors for {test_type.upper()} Test", fontsize=16)
    ax.set_xlabel("Test Statistic", fontsize=14)
    ax.set_ylabel("Probability Density", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    # Return figure and axes
    return fig, ax

