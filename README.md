# mds_2025_helper_functions

[![Documentation Status](https://readthedocs.org/projects/mds-2025-helper-functions/badge/?version=latest)](https://mds-2025-helper-functions.readthedocs.io/en/latest/?badge=latest)
[![Python CI/CD](https://github.com/UBC-MDS/mds-2025-helper-functions/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/UBC-MDS/mds-2025-helper-functions/actions/workflows/ci-cd.yml)
[![codecov](https://codecov.io/gh/UBC-MDS/mds-2025-helper-functions/branch/main/graph/badge.svg)](https://codecov.io/gh/UBC-MDS/mds-2025-helper-functions)

A package to streamline common code chunks executed by students in the UBC MDS program circa 2025. 

## Functions
- compare_model_scores() - a function that takes multiple models and returns a table of mean CV scores for each for easy comparison.
- perform_eda() - a function to perform exploratory data analysis on a dataset
- dataset_summary() - a function that generates a comprehensive summary of a dataset, including missing value statistics, feature counts, duplicate rows, and descriptive statistics.
- htv() - (Hypothesis Test Visualization) provide good plots for user's hypothesis test result, easier to understand what happend in test rather than just number.

## Similar packages
- While this package extends cross-validation from [scikit-learn](https://scikit-learn.org/stable/), there are no known packages that provide CV score comparison similar to compare_model_scores(). The most similar is the summary_cv() function in the [CrossPy](https://github.com/UBC-MDS/CrossPy) package, which summarizes CV scores for a single model.

- While the `ProfileReport` class from the [ydata-profiling](https://github.com/ydataai/ydata-profiling) package provides automated exploratory data analysis and reporting, there are no known packages that offer the same level of flexible, on-demand visualizations and insights as the `perform_eda()` function. The most similar functionality is available in pandas-profiling, which generates detailed HTML reports but lacks the modular, interactive approach that `perform_eda()` provides for tailoring EDA to specific datasets and workflows.

- The `dataset_summary()` function combines essential dataset insights—missing values, feature types, duplicates, and basic statistics—into one comprehensive and easy-to-use tool. While similar functionality exists in libraries like [pandas-profiling](https://github.com/ydataai/pandas-profiling) and [missingno](https://github.com/ResidentMario/missingno), these tools focus on specific aspects or full-scale exploratory analysis. No single function consolidates all these features in one place, making `dataset_summary()` a uniquely efficient solution for preprocessing workflows.

- There is no similar function could provide plot for hypothesis test output. Data Scientist do it manually, but it is not friendly for learner.

## Installation

```bash
$ pip install mds_2025_helper_functions
```

## Usage

# Function Documentation and Usage

## 1. `compare_model_scores`

### Description:
This function compares the mean cross-validation scores of multiple ML models and produces a summary table.

### Parameters:
- `*args` (BaseEstimator): Models to evaluate (e.g., `LogisticRegression`, `RandomForestClassifier`, etc.).
- `X` (array-like): Training dataset of features with shape `(n_samples, n_features)`.
- `y` (array-like, optional): Target values for supervised learning tasks.
- `scoring` (string or callable, optional): Evaluation metrics (e.g., `"accuracy"`). Refer to the [Scikit-learn scoring documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter).
- `return_train_scores` (bool): Whether to include training scores in addition to test scores. Default is `False`.
- `**kwargs`: Additional arguments for `sklearn.model_selection.cross_validate`.

### Returns:
A `pandas.DataFrame` comparing the performance of the models.

### Example Usage:
```python
from mds_2025_helper_functions.scores import compare_model_scores
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasetsimport warnings
warnings.filterwarning.import warningsfilterwarnings de```

```  from sklearn.datasets import load iris
It seems the last few messages got jumbled up. Let me correct it and present the complete optimized README usage documentation without unnecessary comments.

---

```markdown
# Function Documentation and Usage

## 1. `compare_model_scores`

### Description:
This function compares the mean cross-validation scores of multiple ML models and produces a summary table.

### Parameters:
- `*args` (BaseEstimator): Models to evaluate (e.g., `LogisticRegression`, `RandomForestClassifier`, etc.).
- `X` (array-like): Training dataset of features with shape `(n_samples, n_features)`.
- `y` (array-like, optional): Target values for supervised learning tasks.
- `scoring` (string or callable, optional): Evaluation metrics (e.g., `"accuracy"`). Refer to the [Scikit-learn scoring documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter).
- `return_train_scores` (bool): Whether to include training scores in addition to test scores. Default is `False`.
- `**kwargs`: Additional arguments for `sklearn.model_selection.cross_validate`.

### Returns:
A `pandas.DataFrame` comparing the performance of the models.

### Example Usage:
```python
from mds_2025_helper_functions.scores import compare_model_scores
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Load sample dataset
data = load_iris()
X, y = data["data"], data["target"]

# Compare models
compare_model_scores(
    LogisticRegression(),
    DecisionTreeClassifier(),
    X=X,
    y=y,
    scoring="accuracy"
)
```

---

## 2. `perform_eda`

### Description:
A one-stop Exploratory Data Analysis (EDA) function to generate data summaries, spot missing values, visualize feature distributions, and detect outliers.

### Parameters:
- `dataframe` (pd.DataFrame): The input dataset for analysis.
- `rows` (int): Number of rows in the grid layout for visualizations. Default is 5.
- `cols` (int): Number of columns in the grid layout for visualizations. Default is 2.

### Returns:
- Prints dataset statistics, missing values report, and an outlier summary.
- Generates plots and visualizations using Matplotlib and Seaborn.

### Example Usage:
```python
from mds_2025_helper_functions.eda import perform_eda
import pandas as pd

data = {
    'Age': [25, 32, 47, 51, 62],
    'Salary': [50000, 60000, 120000, 90000, 85000],
    'Department': ['HR', 'Finance', 'IT', 'Finance', 'HR'],
}
df = pd.DataFrame(data)

perform_eda(df, rows=2, cols=2)
```

---

## 3. `dataset_summary`

### Description:
Generates a summary of a dataset including missing values, feature types, duplicate rows, and other descriptive statistics.

### Parameters:
- `data` (pd.DataFrame): The dataset to summarize.

### Returns:
A dictionary containing:
- `'missing_values'`: DataFrame of missing value counts and percentages.
- `'feature_types'`: Counts of numerical and categorical features.
- `'duplicates'`: Number of duplicate rows.
- `'numerical_summary'`: Descriptive statistics for numerical columns.
- `'categorical_summary'`: Unique value counts for categorical features.

### Example Usage:
```python
from mds_2025_helper_functions.dataset_summary import dataset_summary
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Alice', None],
    'Age': [25, 32, 47, None, 29],
    'Salary': [50000, 60000, 120000, None, 80000],
    'Department': ['HR', 'Finance', 'IT', 'HR', 'Finance']
}
df = pd.DataFrame(data)

summary = dataset_summary(df)
print(summary['missing_values'])
print(summary['numerical_summary'])
print(summary['categorical_summary'])
```

---

## 4. `htv`

### Description:
Visualizes Type I (α) and Type II (β) errors in hypothesis tests.

### Parameters:
- `test_output` (dict): Dictionary containing hypothesis test parameters:
  - `'mu0'` (float): Mean under the null hypothesis (H₀).
  - `'mu1'` (float): Mean under the alternative hypothesis (H₁).
  - `'sigma'` (float): Standard deviation.
  - `'sample_size'` (int): Sample size.
  - `'df'` (int, optional): Degrees of freedom, required for `'t'` or `'chi2'` tests.
  - `'df1'`, `'df2'` (int, optional): For F-tests (`anova`).
- `test_type` (str): Type of test (`'z'`, `'t'`, `'chi2'`, or `'anova'`).
- `alpha` (float): Significance level for Type I error. Default is `0.05`.
- `tail` (str): `'one-tailed'` or `'two-tailed'`. Default is `'two-tailed'`.

### Returns:
- A tuple of `(fig, ax)` for plotting the visualization.

### Example Usage:
```python
from mds_2025_helper_functions.htv import htv
import matplotlib.pyplot as plt

test_params = {
    'mu0': 100,
    'mu1': 105,
    'sigma': 15,
    'sample_size': 30
}

fig, ax = htv(test_params, test_type="z", alpha=0.05, tail="two-tailed")
plt.show()
```

```python
test_params_t = {
    'mu0': 0,
    'mu1': 1.5,
    'sigma': 1,
    'sample_size': 25
}

fig, ax = htv(test_params_t, test_type="t", alpha=0.01, tail="one-tailed")
plt.show()
```

---

### Notes:
- Required imports:
  ```python
  from mds_2025_helper_functions.scores import compare_model_scores
  from mds_2025_helper_functions.eda import perform_eda
  from mds_2025_helper_functions.dataset_summary import dataset_summary
  from mds_2025_helper_functions.htv import htv
  from sklearn.datasets import load_iris, load_diabetes
  from sklearn.dummy import DummyRegressor, DummyClassifier
  from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
  import matplotlib.pyplot as plt
  import pandas as pd
  import warnings
  warnings.filterwarnings('ignore')
  ```


## Testing Commands

### 1. Run All Tests
```bash
pytest
```

---

### 2. Run a Specific Test File
```bash
pytest test_dataset_summary.py
pytest test_eda.py
pytest test_htv.py
pytest test_scores.py
```

---

### 3. Run a Specific Test Function
```bash
pytest test_dataset_summary.py::test_function_name
```

---

### 4. Run Tests with Verbose Output
```bash
pytest -v
```

---

### 5. Run Tests with Coverage
```bash
pytest --cov=.
```

Generate an HTML coverage report:
```bash
pytest --cov=. --cov-report=html
```

---

### 6. Run Tests in Parallel (Optional)
Run tests with 4 parallel workers:
```bash
pytest -n 4
```

---

### 7. Clear Pytest Cache
```bash
pytest --cache-clear
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## Contributors

Karlygash Zhakupbayeva, Samuel Adetsi, Xi Cu, Michael Hewlett

## License

`mds_2025_helper_functions` was created by Karlygash Zhakupbayeva, Samuel Adetsi, Xi Cu, Michael Hewlett. It is licensed under the terms of the MIT license.

## Credits

`mds_2025_helper_functions` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
