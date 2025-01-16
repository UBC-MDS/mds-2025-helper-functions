import pandas as pd
import numpy as np
import pytest
from mds_2025_helper_functions.eda import perform_eda

def test_perform_eda_invalid_input():
    """Test that the function raises a TypeError when input is not a DataFrame."""
    with pytest.raises(TypeError, match="Input must be a pandas DataFrame."):
        perform_eda([]) 

@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_perform_eda_basic_summary(capfd, mocker):
    """Test that the function prints basic information about the DataFrame."""
    mocker.patch("matplotlib.pyplot.show")
    

    data = {
        "numeric": [1, 2, 3, 4, 5],
        "categorical": ["A", "B", "A", "B", "C"],
        "datetime": pd.date_range("2022-01-01", periods=5)
    }
    df = pd.DataFrame(data)

    perform_eda(df)

    captured = capfd.readouterr()

    assert "===== Dataset Overview =====" in captured.out
    assert "===== Basic Statistics =====" in captured.out
    assert "===== Missing Values Report =====" in captured.out
    assert "numeric" in captured.out
    assert "categorical" in captured.out
    assert "datetime" in captured.out

def test_perform_eda_missing_values_report(capfd, mocker):
    """Test the missing values report."""
    mocker.patch("matplotlib.pyplot.show")

    data = {
        "col1": [1, 2, np.nan, 4, 5],
        "col2": [np.nan, 2, 3, np.nan, 5],
    }
    df = pd.DataFrame(data)

    perform_eda(df)

    captured = capfd.readouterr()

    assert "===== Missing Values Report =====" in captured.out
    assert "col1" in captured.out
    assert "col2" in captured.out

def test_perform_eda_no_missing_values(capfd, mocker):
    """Test behavior when no missing values exist."""
    mocker.patch("matplotlib.pyplot.show")

    df = pd.DataFrame({
        "col1": [1, 2, 3, 4],
        "col2": [5, 6, 7, 8]
    })

    perform_eda(df)

    captured = capfd.readouterr()

    assert "No missing values in the dataset." in captured.out

def test_perform_eda_correlation_heatmap(mocker):
    """Test that correlation heatmap is generated for numeric columns."""
    mocker.patch("matplotlib.pyplot.show")

    df = pd.DataFrame({
        "num1": [1, 2, 3, 4, 5],
        "num2": [2, 4, 6, 8, 10],
        "num3": [5, 3, 2, 8, 7]
    })

    perform_eda(df)

def test_perform_eda_feature_visualizations(mocker):
    """Test that feature visualizations are generated."""
    mocker.patch("matplotlib.pyplot.show")

    df = pd.DataFrame({
        "num1": [1, 2, 3, 4, 5],
        "cat1": ["A", "B", "A", "C", "A"],
        "date1": pd.date_range("2022-01-01", periods=5)
    })

    perform_eda(df)


def test_perform_eda_outliers_report(capfd, mocker):
    """Test that the outliers report works correctly."""
    mocker.patch("matplotlib.pyplot.show")
    df = pd.DataFrame({
        "num1": [1, 2, 100, 3, 4],
        "num2": [5, 6, 7, 8, 500]
    })

    perform_eda(df)

    captured = capfd.readouterr()

    assert "===== Outliers Report =====" in captured.out
    assert "num1: 1 potential outliers" in captured.out
    assert "num2: 1 potential outliers" in captured.out

def test_perform_eda_no_numeric_columns(capfd, mocker):
    """Test behavior when no numeric columns exist."""
    mocker.patch("matplotlib.pyplot.show")

    df = pd.DataFrame({
        "cat1": ["A", "B", "C"],
        "cat2": ["X", "Y", "Z"]
    })

    perform_eda(df)

    captured = capfd.readouterr()
    assert "Not enough numeric columns for correlation heatmap." in captured.out