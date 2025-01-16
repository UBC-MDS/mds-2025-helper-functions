from mds_2025_helper_functions.scores import compare_model_scores
import pytest

def test_compare_model_scores_success():
    """Test that basic function works"""
    pass

def test_compare_model_scores_edge():
    pass

def test_compare_model_scores_error():
    pass

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

X missing
only 1 argument specified
element in args not a model object
mixed model types - some cat, some regress - used

basic - takes in 2 models, an X, and a y - outputs a table




"""
# write empty test blocks
# write data for tests
# write tests
# write code