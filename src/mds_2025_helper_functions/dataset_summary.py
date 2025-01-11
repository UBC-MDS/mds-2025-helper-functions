def dataset_summary(data):
    """
    Generates a comprehensive summary of a dataset, including:
    - Missing value statistics
    - Counts of numerical and categorical features
    - Duplicate row count
    - Descriptive statistics for numerical features
    - Unique value counts for categorical features

    Parameters
    ----------
    data : pandas.DataFrame
        The input dataset to analyze and summarize.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - 'missing_values': pd.DataFrame
            A DataFrame summarizing the number and percentage of missing values for each column.
        - 'feature_types': dict
            A dictionary with counts of numerical and categorical features:
            {'numerical_features': int, 'categorical_features': int}.
        - 'duplicates': int
            The number of duplicate rows in the dataset.
        - 'numerical_summary': pd.DataFrame
            Basic descriptive statistics for numerical features.
        - 'categorical_summary': pd.DataFrame
            A summary of unique value counts for categorical features.
    """
    pass