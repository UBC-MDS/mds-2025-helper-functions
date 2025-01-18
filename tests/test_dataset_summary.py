import pytest
import pandas as pd
import numpy as np
from mds_2025_helper_functions.dataset_summary import dataset_summary

def test_dataset_summary_invalid_input():
    """
    Test that the function raises a TypeError when the input is not a pandas DataFrame.
    """
    invalid_inputs = [None, [1, 2, 3], {"a": 1, "b": 2}, "not a DataFrame", 123]
    for invalid_input in invalid_inputs:
        with pytest.raises(TypeError, match="Input must be a pandas DataFrame"):
            dataset_summary(invalid_input)

def test_dataset_summary_empty_dataframe():
    """
    Test the function with an empty DataFrame.
    """
    df = pd.DataFrame()
    result = dataset_summary(df)

    assert isinstance(result, dict)
    assert result["missing_values"].empty
    assert result["feature_types"]["numerical_features"] == 0
    assert result["feature_types"]["categorical_features"] == 0
    assert result["duplicates"] == 0
    assert result["numerical_summary"].empty
    assert result["categorical_summary"].empty

def test_dataset_summary_with_missing_values():
    """
    Test the function with a DataFrame containing missing values.
    """
    df = pd.DataFrame({
        "num_col": [1, np.nan, 3, 4],
        "cat_col": ["A", "B", None, "A"]
    })
    result = dataset_summary(df)

    missing_values = result["missing_values"]
    assert "num_col" in missing_values["column"].values
    assert "cat_col" in missing_values["column"].values
    assert missing_values["missing_count"].sum() == 2

def test_dataset_summary_with_duplicates():
    """
    Test the function with a DataFrame containing duplicate rows.
    """
    df = pd.DataFrame({
        "num_col": [1, 2, 3, 1],
        "cat_col": ["A", "B", "C", "A"]
    })
    result = dataset_summary(df)

    assert result["duplicates"] == 1

def test_dataset_summary_no_categorical_features():
    """
    Test the function with a DataFrame containing only numerical features.
    """
    df = pd.DataFrame({
        "num1": [1, 2, 3, 4],
        "num2": [5, 6, 7, 8]
    })
    result = dataset_summary(df)

    assert result["feature_types"]["numerical_features"] == 2
    assert result["feature_types"]["categorical_features"] == 0
    assert result["categorical_summary"].empty

def test_dataset_summary_no_numeric_features():
    """
    Test the function with a DataFrame containing only categorical features.
    """
    df = pd.DataFrame({
        "cat1": ["A", "B", "C"],
        "cat2": ["X", "Y", "Z"]
    })
    result = dataset_summary(df)

    assert result["feature_types"]["numerical_features"] == 0
    assert result["feature_types"]["categorical_features"] == 2
    assert result["numerical_summary"].empty