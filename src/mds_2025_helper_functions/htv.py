import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t, chi2, f

def htv(test_output, test_type="z", alpha=0.05, tail="two-tailed"):
    """
    Visualize Type I (\u03b1) and Type II (\u03b2) errors in hypothesis testing.

    Parameters:
        test_output (dict): Dictionary containing hypothesis test parameters:
                            - 'mu0': Mean under the null hypothesis (H0)
                            - 'mu1': Mean under the alternative hypothesis (H1)
                            - 'sigma': Standard deviation (for z or t tests)
                            - 'sample_size': Sample size (for z and t tests)
                            - 'df1': Degrees of freedom 1 (for F tests, optional)
                            - 'df2': Degrees of freedom 2 (for F tests, optional)
                            - 'df': Degrees of freedom (for t and chi-squared tests, optional)
        test_type (str): Type of test ('z', 't', 'chi2', 'anova').
        alpha (float): Significance level (Type I error rate).
        tail (str): One-tailed or two-tailed test ("one-tailed" or "two-tailed").

    Returns:
        tuple: (fig, ax) Matplotlib figure and axes objects.
    """
    mu0 = test_output.get("mu0", 0)
    mu1 = test_output.get("mu1", 1)
    sigma = test_output.get("sigma", 1)
    sample_size = test_output.get("sample_size", 30)
    df = test_output.get("df", None)
    df1 = test_output.get("df1", None)
    df2 = test_output.get("df2", None)

    # Define critical values and distributions based on the test type
    if test_type == "z":
        std_error = sigma / np.sqrt(sample_size)
        if tail == "two-tailed":
            critical_value_low = norm.ppf(alpha / 2, loc=mu0, scale=std_error)
            critical_value_high = norm.ppf(1 - alpha / 2, loc=mu0, scale=std_error)
        else:
            critical_value = norm.ppf(1 - alpha, loc=mu0, scale=std_error)
        dist_null = lambda x: norm.pdf(x, loc=mu0, scale=std_error)
        dist_alt = lambda x: norm.pdf(x, loc=mu1, scale=std_error)

    elif test_type == "t":
        if df is None:
            df = sample_size - 1  # Default degrees of freedom
        if tail == "two-tailed":
            critical_value_low = t.ppf(alpha / 2, df=df)
            critical_value_high = t.ppf(1 - alpha / 2, df=df)
        else:
            critical_value = t.ppf(1 - alpha, df=df)
        dist_null = lambda x: t.pdf(x, df=df)
        dist_alt = lambda x: t.pdf(x, df=df, loc=mu1 - mu0)

    elif test_type == "chi2":
        if df is None:
            raise ValueError("Degrees of freedom (df) must be specified for chi-squared tests.")
        if tail == "two-tailed":
            critical_value_low = chi2.ppf(alpha / 2, df=df)
            critical_value_high = chi2.ppf(1 - alpha / 2, df=df)
        else:
            critical_value = chi2.ppf(1 - alpha, df=df)
        dist_null = lambda x: chi2.pdf(x, df=df)
        dist_alt = lambda x: chi2.pdf(x, df=df + 1)  # Alternative hypothesis

    elif test_type == "anova":
        if df1 is None or df2 is None:
            raise ValueError("Degrees of freedom (df1 and df2) must be specified for ANOVA tests.")
        if tail == "two-tailed":
            critical_value_low = f.ppf(alpha / 2, dfn=df1, dfd=df2)
            critical_value_high = f.ppf(1 - alpha / 2, dfn=df1, dfd=df2)
        else:
            critical_value = f.ppf(1 - alpha, dfn=df1, dfd=df2)
        dist_null = lambda x: f.pdf(x, dfn=df1, dfd=df2)
        dist_alt = lambda x: f.pdf(x, dfn=df1, dfd=df2 + 1)  # Alternative hypothesis

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

    # Fill Type I and Type II error regions
    if tail == "two-tailed":
        ax.fill_between(x, 0, y_null, where=(x <= critical_value_low) | (x >= critical_value_high), 
                        color="orange", alpha=0.5, label="Type I Error (α)")
        ax.fill_between(x, 0, y_alt, where=(x > critical_value_low) & (x < critical_value_high), 
                        color="green", alpha=0.5, label="Type II Error (β)")
        ax.axvline(x=critical_value_low, color="black", linestyle="--", label=f"Critical Value (Low) = {critical_value_low:.2f}")
        ax.axvline(x=critical_value_high, color="black", linestyle="--", label=f"Critical Value (High) = {critical_value_high:.2f}")
    else:
        ax.fill_between(x, 0, y_null, where=(x >= critical_value), color="orange", alpha=0.5, label="Type I Error (α)")
        ax.fill_between(x, 0, y_alt, where=(x < critical_value), color="green", alpha=0.5, label="Type II Error (β)")
        ax.axvline(x=critical_value, color="black", linestyle="--", label=f"Critical Value = {critical_value:.2f}")

    # Add legend and grid
    ax.set_title(f"Type I and Type II Errors for {test_type.upper()} Test", fontsize=16)
    ax.set_xlabel("Test Statistic", fontsize=14)
    ax.set_ylabel("Probability Density", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    return fig, ax

