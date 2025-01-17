import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(dataframe, rows=5, cols=2):
    """
    A universal EDA function to generate data summaries and visualize features.

    Parameters:
        dataframe (pd.DataFrame): The input dataset for EDA.
        rows (int): Number of rows in the grid layout for visualizations.
        cols (int): Number of columns in the grid layout for visualizations.

    Returns:
        None
    """
    
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    print("===== Dataset Overview =====")
    print(dataframe.info())

    print("\n===== Basic Statistics =====")
    print(dataframe.describe(include='all').transpose())

    # Missing value report
    print("\n===== Missing Values Report =====")
    missing_values = dataframe.isnull().sum()
    print(missing_values[missing_values > 0])

    # Plot missing value heatmap (if missing values exist)
    if dataframe.isnull().values.any():
        plt.figure(figsize=(10, 6))
        sns.heatmap(dataframe.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.show()
    else:
        print("No missing values in the dataset.")
    
    # Correlation heatmap for numeric features
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(dataframe[numeric_cols].corr(), dtype=bool))
        sns.heatmap(dataframe[numeric_cols].corr(), mask=mask, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Correlation Heatmap")
        plt.show()
    else:
        print("Not enough numeric columns for correlation heatmap.")

    # Dynamic Grid Plot for All Features
    print("\n===== Feature Visualizations =====")
    total_features = len(dataframe.columns)
    total_plots = rows * cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), tight_layout=True)
    axes = axes.ravel()

    for i, feature in enumerate(dataframe.columns):
        if dataframe[feature].dtype in [np.float64, np.int64]:  # Numeric columns
            sns.histplot(dataframe[feature], kde=True, bins=20, ax=axes[i])
            axes[i].set_title(f"Distribution of {feature}")
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel("Frequency")
        elif pd.api.types.is_datetime64_any_dtype(dataframe[feature]):  # Datetime columns
            dataframe[feature].value_counts().sort_index().plot(kind="line", marker="o", ax=axes[i])
            axes[i].set_title(f"Time Series of {feature}")
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel("Count")
        else: 
            sns.countplot(
                x=dataframe[feature], 
                ax=axes[i], 
                order=dataframe[feature].value_counts().index, 
                palette="viridis", 
                hue=None, 
                legend=False
            )
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].set_title(f"Count Plot for {feature}")
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel("Count")

    for j in range(total_features, total_plots):
        fig.delaxes(axes[j])

    plt.show()

    # Scatterplots for Numeric Feature Pairs
    print("\n===== Scatterplots for Numeric Features =====")
    if len(numeric_cols) > 1:
        num_pairs = len(numeric_cols) * (len(numeric_cols) - 1) // 2  # Total number of unique pairs
        rows_scatter = (num_pairs // cols) + (1 if num_pairs % cols != 0 else 0)  # Dynamic row count
        fig, axes = plt.subplots(rows_scatter, cols, figsize=(cols * 6, rows_scatter * 4), tight_layout=True)
        axes = axes.ravel()

        pair_idx = 0
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1:]:
                if pair_idx >= len(axes):
                    break
                sns.scatterplot(x=dataframe[col1], y=dataframe[col2], ax=axes[pair_idx], alpha=0.7)
                axes[pair_idx].set_title(f"{col1} vs {col2}")
                axes[pair_idx].set_xlabel(col1)
                axes[pair_idx].set_ylabel(col2)
                pair_idx += 1

        for j in range(pair_idx, len(axes)):
            fig.delaxes(axes[j])

        plt.show()
    else:
        print("Not enough numeric columns for scatterplots.")

    # Outliers Detection Report
    print("\n===== Outliers Report =====")
    for col in numeric_cols:
        Q1 = dataframe[col].quantile(0.25)
        Q3 = dataframe[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = dataframe[(dataframe[col] < Q1 - 1.5 * IQR) | (dataframe[col] > Q3 + 1.5 * IQR)]
        print(f"{col}: {len(outliers)} potential outliers")