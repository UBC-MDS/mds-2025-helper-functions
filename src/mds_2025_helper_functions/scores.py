from sklearn.model_selection import cross_validate
from sklearn.base import BaseEstimator
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
    if len(args) <= 1:
        raise TypeError(
            "compare_model_scores() requires at least 2 models. "
            f"You provided {len(args)}."
        )
    
    for model in args:
        if not isinstance(model, BaseEstimator):
            raise TypeError(
                "All models must be sklearn models. "
                f"The following argument is not an sklearn model: {model}"
                )
    
    model_types = {model._estimator_type for model in args}
    
    if len(model_types) > 1:
        raise ValueError(
            "All models must be of the same type. "
            f"Found multiple types: {', '.join(sorted(model_types))}"
        )
    
    results = []
    model_counts = {}

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
        model_name = model.__class__.__name__
        if model_name in model_counts:
            model_counts[model_name] += 1
            model_name = f"{model_name}_{model_counts[model_name]}"
        else:
            model_counts[model_name] = 1

        mean_scores['model'] = model_name
        results.append(mean_scores)

    return pd.DataFrame(results).set_index('model')