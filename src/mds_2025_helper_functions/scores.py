from sklearn.model_selection import cross_validate
from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd

def compare_model_scores(*args, X, y=None, scoring=None, return_train_scores=False, **kwargs):
    """
    Creates a table comparing mean cross-validation scores of multiple models.

    Parameters
    ----------
    *args : sklearn.base.BaseEstimator
        Model objects implementing the `fit` method. At least two models are required.
    
    X : array-like of shape (n_samples, n_features)
        Training data.
    
    y : array-like of shape (n_samples,) or (n_samples, n_outputs), optional
        Target values for supervised learning tasks.
    
    scoring : str, callable, list, tuple, or dict, optional
        Metrics to evaluate models. Refer to `scikit-learn` scoring documentation:
        https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter.
    
    return_train_scores : bool, default=False
        Whether to include training scores in addition to test scores.

    **kwargs : dict
        Additional arguments passed to `sklearn.model_selection.cross_validate`.

    Returns
    -------
    pd.DataFrame
        A DataFrame comparing model performance:
        - Rows represent different models.
        - Columns include metrics from cross-validation.
        - Index contains model names.
    
    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> compare_model_scores(LogisticRegression(), RandomForestClassifier(), X=X_train, y=y_train, scoring="accuracy")
    """
    # Check for at least 2 models to compare
    if len(args) <= 1:
        raise TypeError(
            "compare_model_scores() requires at least 2 models. "
            f"You provided {len(args)}."
        )
    
    # Check that objects passed as arguments are models (not e.g. lists, strings, etc)
    for model in args:
        if not isinstance(model, BaseEstimator):
            raise TypeError(
                "All models must be sklearn models. "
                f"The following argument is not an sklearn model: {model}"
                )
    
    # Check that all models are either classifiers or regressors
    model_types = {model._estimator_type for model in args}
    
    if len(model_types) > 1:
        raise ValueError(
            "All models must be of the same type. "
            f"Found multiple types: {', '.join(sorted(model_types))}"
        )
    
    # Main code
    results = []
    model_counts = {}

    for model in args:
        # Get CV scores
        cv_results = cross_validate(
            model,
            X=X,
            y=y,
            scoring=scoring,
            return_train_score=return_train_scores,
            **kwargs
        )

        # Calculate mean of scores
        mean_scores = {key: np.mean(val) for key, val in cv_results.items()}

        # Give model a unique name
        model_name = model.__class__.__name__
        if model_name in model_counts:
            model_counts[model_name] += 1
            model_name = f"{model_name}_{model_counts[model_name]}"
        else:
            model_counts[model_name] = 1

        mean_scores['model'] = model_name

        # Add model scores and model name to list
        results.append(mean_scores)

    # Return model list as DataFrame
    return pd.DataFrame(results).set_index('model')