import pytest
import matplotlib.pyplot as plt
from hypothesis_test_viz import visualize_hypothesis_test_output_extended

def test_visualize_hypothesis_test_output_extended():
    # Test cases for z-test
    test_output_z = {
        "mu0": 0,
        "mu1": 1,
        "sigma": 1,
        "sample_size": 30
    }
    
    fig, ax = visualize_hypothesis_test_output_extended(test_output_z, test_type="z", alpha=0.05, tail="two-tailed")
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    fig, ax = visualize_hypothesis_test_output_extended(test_output_z, test_type="z", alpha=0.05, tail="one-tailed")
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    # Test cases for t-test
    test_output_t = {
        "mu0": 0,
        "mu1": 2,
        "sigma": 2,
        "sample_size": 25,
        "df": 24
    }
    
    fig, ax = visualize_hypothesis_test_output_extended(test_output_t, test_type="t", alpha=0.05, tail="two-tailed")
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    fig, ax = visualize_hypothesis_test_output_extended(test_output_t, test_type="t", alpha=0.1, tail="one-tailed")
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    # Test cases for chi2-test
    test_output_chi2 = {
        "mu0": 0,
        "mu1": 0,
        "sigma": 1,
        "sample_size": 1,
        "df": 10
    }
    
    fig, ax = visualize_hypothesis_test_output_extended(test_output_chi2, test_type="chi2", alpha=0.05)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    # Test cases for anova
    test_output_anova = {
        "mu0": 0,
        "mu1": 0,
        "sigma": 1,
        "sample_size": 1,
        "df1": 10,
        "df2": 20
    }
    
    fig, ax = visualize_hypothesis_test_output_extended(test_output_anova, test_type="anova", alpha=0.05)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

    # Edge cases
    with pytest.raises(ValueError):
        visualize_hypothesis_test_output_extended({}, test_type="unknown")

    with pytest.raises(ValueError):
        visualize_hypothesis_test_output_extended({}, test_type="chi2")
