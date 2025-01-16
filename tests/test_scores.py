from mds_2025_helper_functions.scores import compare_model_scores
import pytest
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import pandas as pd
import numpy as np
from unittest.mock import patch

# Mock data
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
tree_r2 = DecisionTreeRegressor(max_depth=2)
dummy_c = DummyClassifier()
tree_c = DecisionTreeClassifier()
non_sklearn_object = []

output_col_names = ["fit_time", "score_time", "test_score", "train_score"]
output_index_names_regression = ["dummy_r", "tree_r"]
output_index_names_classification = ["dummy_c", "tree_c"]

mock_cv_results = {
    'test_score': np.array([0.7, 0.8, 0.9]),
    'train_score': np.array([0.75, 0.85, 0.95]),
    'fit_time': np.array([0.1, 0.1, 0.1]),
    'score_time': np.array([0.05, 0.05, 0.05])
}

# Success tests
def test_compare_model_scores_regression():
    """For regression models, returns a pandas dataframe with the metrics as column names and models as index names"""
    with patch('sklearn.model_selection.cross_validate', return_value = mock_cv_results):
        result_r = compare_model_scores(dummy_r, tree_r, x, y_regression, return_train_scores=True)
        result_r_index = result_r.index.tolist()
        result_r_colnames = result_r.columns.tolist()
        
        assert isinstance(result_r, pd.DataFrame)
        assert result_r.shape[0] == 2
        assert set(result_r_index) == set(output_index_names_regression)
        assert set(result_r_colnames) == set(output_col_names)
        assert np.isclose(result_r.loc['DummyRegressor', 'test_score'], 0.8)

def test_compare_model_scores_classification():
    """For classification models, returns a pandas dataframe with the metrics as column names and models as index names"""
    with patch('sklearn.model_selection.cross_validate', return_value = mock_cv_results):
        result_c = compare_model_scores(dummy_c, tree_c, x, y_classification, return_train_scores=True)
        result_c_index = result_c.index.tolist()
        result_c_colnames = result_c.columns.tolist()

        assert isinstance(result_c, pd.DataFrame)
        assert result_c.shape[0] == 2
        assert set(result_c_index) == set(output_index_names_classification)
        assert set(result_c_colnames) == set(output_col_names)
        assert np.isclose(result_c.loc['DummyClassifier', 'test_score'], 0.8)

# Edge tests
def test_compare_model_scores_return_train_scores_false():
    """Train scores are excluded when return_train_scores=False"""
    with patch('sklearn.model_selection.cross_validate', return_value = mock_cv_results):
        result = compare_model_scores(dummy_r, tree_r, x, y_regression, return_train_scores=False)
        assert 'train_score' not in result.columns
        assert result.shape[1] == 3

def test_compare_model_scores_multiple_scoring_metrics():
    """DataFrame has correct columns for multiple scoring metrics"""
    with patch('sklearn.model_selection.cross_validate', return_value = mock_cv_results):
        result = compare_model_scores(
            dummy_r, 
            tree_r, 
            x, 
            y_regression, 
            scoring=['accuracy', 'precision'], 
            return_train_scores=True
        )

        assert 'test_accuracy' in result.columns
        assert 'test_precision' in result.columns

def test_compare_model_scores_same_model_type():
    """Function handles multiple instances of same model type"""
    with patch('sklearn.model_selection.cross_validate', return_value = mock_cv_results):
        result = compare_model_scores(tree_r, tree_r2, x, y_classification, return_train_scores=True)
    
    assert result.index[0] != result.index[1]

# Error tests
def test_compare_model_scores_single_model():
    """Raises an error if only 1 model is passed as an argument"""
    expected_error = "compare_model_scores() requires at least 2 models. You provided 1."
    
    with pytest.raises(TypeError) as e:
        compare_model_scores(dummy_r, x, y_regression, return_train_scores=True)
    
    actual_error = str(e.value)
    assert actual_error == expected_error

def test_compare_model_scores_invalid_model():
    """Raises an error if a non sklearn model is passed as a model argument"""
    expected_error = "All models must be sklearn models. The following argument is not an sklearn model: []"

    with pytest.raises(TypeError) as e:
        compare_model_scores(non_sklearn_object, x, y_regression, return_train_scores=True)
        
    actual_error = str(e.value)
    assert actual_error == expected_error

def test_compare_model_scores_mixed_models():
    """Raises an error if a mix of classification and regression models are provided"""
    expected_error = "All models must be of the same type. Found multiple types: regressor, classifier"
    
    with pytest.raises(ValueError) as e:
        compare_model_scores(dummy_r, dummy_c, x, y_regression)

    actual_error = str(e.value)
    assert actual_error == expected_error