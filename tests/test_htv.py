import unittest
import numpy as np
from scipy.stats import norm, t, chi2
import matplotlib.pyplot as plt
from src.mds_2025_helper_functions.htv import htv

class TestHypothesisVisualizer(unittest.TestCase):

    def test_z_test_plot(self):
        """Test visualization for Z-test"""
        z_test_output = {
            "mu0": 0,
            "mu1": 1,
            "sigma": 1,
            "sample_size": 30
        }
        fig, ax = htv(z_test_output, test_type="z")
        self.assertIsNotNone(fig, "Z-test plot figure is None")
        self.assertIsNotNone(ax, "Z-test plot axis is None")
        plt.close(fig)

    def test_t_test_plot(self):
        """Test visualization for T-test"""
        t_test_output = {
            "mu0": 0,
            "mu1": 1,
            "sigma": 1,
            "sample_size": 30,
            "df": 29
        }
        fig, ax = htv(t_test_output, test_type="t")
        self.assertIsNotNone(fig, "T-test plot figure is None")
        self.assertIsNotNone(ax, "T-test plot axis is None")
        plt.close(fig)

    def test_chi_squared_test_plot(self):
        """Test visualization for Chi-squared test"""
        chi2_test_output = {
            "mu0": 0,
            "mu1": 1,
            "sigma": 1,
            "sample_size": 30,
            "df": 10
        }
        fig, ax = htv(chi2_test_output, test_type="chi2")
        self.assertIsNotNone(fig, "Chi-squared plot figure is None")
        self.assertIsNotNone(ax, "Chi-squared plot axis is None")
        plt.close(fig)

    def test_anova_test_plot(self):
        """Test visualization for ANOVA test"""
        anova_test_output = {
            "mu0": 0,
            "mu1": 1,
            "sigma": 1,
            "sample_size": 30,
            "df1": 2,
            "df2": 27
        }
        fig, ax = htv(anova_test_output, test_type="anova")
        self.assertIsNotNone(fig, "ANOVA plot figure is None")
        self.assertIsNotNone(ax, "ANOVA plot axis is None")
        plt.close(fig)

    def test_invalid_test_type(self):
        """Test handling of invalid test type"""
        invalid_test_output = {
            "mu0": 0,
            "mu1": 1,
            "sigma": 1,
            "sample_size": 30
        }
        with self.assertRaises(ValueError):
            htv(invalid_test_output, test_type="invalid")

if __name__ == "__main__":
    unittest.main()
