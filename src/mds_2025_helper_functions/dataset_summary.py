import pandas as pd

def dataset_summary(data):
    """
    Generates a comprehensive summary of a dataset.

    This function analyzes the input DataFrame and provides insights, including:
    - Missing value statistics for each column
    - Counts of numerical and categorical features
    - Number of duplicate rows
    - Descriptive statistics for numerical features
    - Unique value counts for categorical features

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to analyze. Must be a pandas DataFrame.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        
        - 'missing_values' (pd.DataFrame): 
            Summary of missing values, including counts and percentages for each column.
        - 'feature_types' (dict): 
            Counts of numerical and categorical features in the dataset.
            Format: {'numerical_features': int, 'categorical_features': int}.
        - 'duplicates' (int): 
            The number of duplicate rows in the dataset.
        - 'numerical_summary' (pd.DataFrame): 
            Descriptive statistics for numerical columns.
        - 'categorical_summary' (pd.DataFrame): 
            Unique value counts for categorical columns.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.
    ValueError
        If the DataFrame is empty or contains unsupported data types.
    """
    
    # Check input type
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    # Handle empty DataFrame
    if data.empty:
        return {
            "missing_values": pd.DataFrame(columns=["column", "missing_count", "missing_percentage"]),
            "feature_types": {"numerical_features": 0, "categorical_features": 0},
            "duplicates": 0,
            "numerical_summary": pd.DataFrame(),
            "categorical_summary": pd.DataFrame(),
        }

    # Missing value statistics
    missing_values = data.isnull().sum().reset_index()
    missing_values.columns = ["column", "missing_count"]
    missing_values["missing_percentage"] = (missing_values["missing_count"] / len(data)) * 100

    # Count feature types
    numerical_features = data.select_dtypes(include="number").shape[1]
    categorical_features = data.select_dtypes(exclude="number").shape[1]
    feature_types = {"numerical_features": numerical_features, "categorical_features": categorical_features}

    # Duplicate rows
    duplicates = data.duplicated().sum()

    # Descriptive statistics for numerical features
    if numerical_features > 0:
        numerical_summary = data.describe(include="number").transpose()
    else:
        numerical_summary = pd.DataFrame()

    # Unique value counts for categorical features
    categorical_columns = data.select_dtypes(include=["object", "category"]).columns
    if not categorical_columns.empty:
        categorical_summary = data[categorical_columns].nunique().reset_index()
        categorical_summary.columns = ["column", "unique_values"]
    else:
        categorical_summary = pd.DataFrame()

    return {
        "missing_values": missing_values,
        "feature_types": feature_types,
        "duplicates": duplicates,
        "numerical_summary": numerical_summary,
        "categorical_summary": categorical_summary
    }