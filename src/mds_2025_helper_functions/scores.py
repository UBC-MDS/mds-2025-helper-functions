from sklearn.model_selection import cross_validate
import numpy as np
import pandas as pd

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
    results = []
    for model in args:
        cv_results = cross_validate(
            model,
            X=X,
            y=y,
            scoring=scoring,
            return_train_score=return_train_scores,
            **kwargs
        )

        mean_scores = {key: np.mean(val) for key, val in cv_results.items()}
        mean_scores['model'] = model.__class__.__name__
        results.append(mean_scores)

    return pd.Dataframe(results).set_index('model')