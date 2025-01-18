import pytest
import matplotlib.pyplot as plt
from mds_2025_helper_functions.htv import htv

def test_htv_z_test_two_tailed():
    test_output = {
        "mu0": 0,
        "mu1": 1,
        "sigma": 1,
        "sample_size": 30
    }
    fig, ax = htv(test_output, test_type="z", alpha=0.05, tail="two-tailed")
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

def test_htv_z_test_one_tailed():
    test_output = {
        "mu0": 0,
        "mu1": 1,
        "sigma": 1,
        "sample_size": 30
    }
    fig, ax = htv(test_output, test_type="z", alpha=0.05, tail="one-tailed")
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

def test_htv_t_test_two_tailed():
    test_output = {
        "mu0": 0,
        "mu1": 2,
        "sigma": 2,
        "sample_size": 25,
        "df": 24
    }
    fig, ax = htv(test_output, test_type="t", alpha=0.05, tail="two-tailed")
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

def test_htv_t_test_one_tailed():
    test_output = {
        "mu0": 0,
        "mu1": 2,
        "sigma": 2,
        "sample_size": 25,
        "df": 24
    }
    fig, ax = htv(test_output, test_type="t", alpha=0.1, tail="one-tailed")
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

def test_htv_chi2_test():
    test_output = {
        "mu0": 0,
        "mu1": 0,
        "sigma": 1,
        "sample_size": 1,
        "df": 10
    }
    fig, ax = htv(test_output, test_type="chi2", alpha=0.05)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

def test_htv_anova_test():
    test_output = {
        "mu0": 0,
        "mu1": 0,
        "sigma": 1,
        "sample_size": 1,
        "df1": 10,
        "df2": 20
    }
    fig, ax = htv(test_output, test_type="anova", alpha=0.05)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)

def test_htv_invalid_test_type():
    with pytest.raises(ValueError):
        htv({}, test_type="unknown")

def test_htv_missing_parameters():
    with pytest.raises(ValueError):
        htv({}, test_type="chi2")
