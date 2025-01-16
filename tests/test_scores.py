from mds_2025_helper_functions.scores import compare_model_scores
import pytest
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import pandas as pd
import numpy as np

def compare_model_scores(*args, X, y=None, scoring=None, return_train_scores=False, **kwargs):
    """Creates a table comparing mean CV scores of multiple models.
    Parameters
    ----------
    *args: model objects implementing 'fit'
        The model objects to compare
    
    X: {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data
    
    y: array-like of shape (n_samples) or (n_samples, n_outputs), default=None
        Target values to try to predict in the case of supervised learning.

    scoring: str, callable, list, tuple, or dict, default=None
        Metrics to evaluate models on

    return_train_scores: bool, default=False
        Whether to include training scores in addition to test scores

    **kwargs:
        Additional arguments passed to sklearn.model_selection_metrics.cross_validate
        See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html
    
    Returns
    -------
    pd.DataFrame
        Dataframe comparing model performance with:
        - Rows: Different models
        - Columns: Score metrics
        - Index: Model names
    """
    return

# write pseudo code - desc, input (objects), output (objects)
"""

Tests my function should cover
- outputs a dataframe with score metrics as columns and model names as index names
- only 1 model is given
- a mix of model types is given (e.g. regression model and classification model)

Tests cross-validate should already cover
- no models are given
- a non-model object is given as a model
- improper mix of model and target type (e.g. regression model and categorical target)
- X is wrong type
- y is wrong type

"""
# write empty test blocks
# write data for tests
x = pd.DataFrame(
    {
        "feature_numeric": [1, 2, 3, 10],
        "feature_categorical": ["apple", "banana", "orange", "apple"]
    }
)
y_regression = pd.DataFrame(
    {
        "target_numeric": [5, 10, 14, 2]
    }
)
y_classification = pd.DataFrame(
    {
        "target_categorical": ["olive", "cypress", "cypress", "oak"]
    }
)
dummy_r = DummyRegressor()
tree_r = DecisionTreeRegressor()
dummy_c = DummyClassifier()
tree_c = DecisionTreeClassifier()

output_col_names = ["fit_time", "score_time", "test_score", "train_score"]
output_index_names_regression = ["dummy_r", "tree_r"]
output_index_names_classification = ["dummy_c", "tree_c"]

# write tests
def test_compare_model_scores_success():
    """Returns a pandas dataframe with the metrics as column names and models as index names"""
    result_r = compare_model_scores(dummy_r, tree_r, x, y_regression, return_train_scores=True)
    result_r_index = result_r.index.tolist()
    result_r_colnames = result_r.columns.tolist()

    result_c = compare_model_scores(dummy_c, tree_c, x, y_classification, return_train_scores=True)
    result_c_index = result_c.index.tolist()
    result_c_colnames = result_c.columns.tolist()
    
    assert isinstance(result_r, pd.DataFrame)
    assert set(result_r_index) == set(output_index_names_regression)
    assert set(result_r_colnames) == set(output_col_names)

    assert isinstance(result_c, pd.DataFrame)
    assert set(result_c_index) == set(output_index_names_classification)
    assert set(result_c_colnames) == set(output_col_names)

def test_compare_model_scores_edge():
    pass

def test_compare_model_scores_error():
    """Raises an error if only 1 model is passed as an argument"""
    result = compare_model_scores(dummy_r, x, y_regression, return_train_scores=True)
    with pytest.raises(ValueError):
        result
    
    """Raises an error if a mix of classification and regression models are provided"""
    result = compare_model_scores(dummy_r, dummy_c, x, y_regression)
    with pytest.raises(TypeError):
        result
# write code