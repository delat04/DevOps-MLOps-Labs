import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
import pandas as pd
from data_loader import (
    load_iris_data,
    get_feature_names,
    get_target_names,
    load_iris_as_dataframe,
    get_dataset_info
)


class TestDataLoader:
    """Test suite for data_loader module"""

    def test_load_iris_data_returns_correct_shapes(self):
        """Test that load_iris_data returns correct array shapes"""
        X_train, X_test, y_train, y_test = load_iris_data(test_size=0.2, random_state=42)

        # Check that we have 4 features
        assert X_train.shape[1] == 4
        assert X_test.shape[1] == 4

        # Check that train and test sets have corresponding shapes
        assert X_train.shape[0] == y_train.shape[0]
        assert X_test.shape[0] == y_test.shape[0]

        # Check total samples is 150 (Iris dataset size)
        assert X_train.shape[0] + X_test.shape[0] == 150

    def test_load_iris_data_split_ratio(self):
        """Test that test_size parameter correctly splits the data"""
        test_size = 0.3
        X_train, X_test, y_train, y_test = load_iris_data(test_size=test_size, random_state=42)

        total_samples = X_train.shape[0] + X_test.shape[0]
        actual_test_ratio = X_test.shape[0] / total_samples

        # Allow small tolerance for rounding
        assert abs(actual_test_ratio - test_size) < 0.02

    def test_load_iris_data_class_distribution(self):
        """Test that all three classes are present in train and test sets"""
        X_train, X_test, y_train, y_test = load_iris_data(test_size=0.2, random_state=42)

        # Check that we have all 3 classes
        train_classes = np.unique(y_train)
        test_classes = np.unique(y_test)

        assert len(train_classes) == 3
        assert len(test_classes) == 3
        assert all(c in [0, 1, 2] for c in train_classes)
        assert all(c in [0, 1, 2] for c in test_classes)

    def test_get_feature_names_returns_list(self):
        """Test that get_feature_names returns correct feature list"""
        feature_names = get_feature_names()

        assert isinstance(feature_names, list)
        assert len(feature_names) == 4
        assert all(isinstance(name, str) for name in feature_names)
        assert 'sepal' in feature_names[0].lower()
        assert 'petal' in feature_names[2].lower()

    def test_get_target_names_returns_species(self):
        """Test that get_target_names returns correct species names"""
        target_names = get_target_names()

        assert isinstance(target_names, (list, np.ndarray))
        assert len(target_names) == 3
        assert 'setosa' in target_names
        assert 'versicolor' in target_names
        assert 'virginica' in target_names

    def test_load_iris_as_dataframe_structure(self):
        """Test that DataFrame has correct structure and columns"""
        df = load_iris_as_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 150  # Iris dataset has 150 samples
        assert 'target' in df.columns
        assert 'species' in df.columns

        # Check that we have 6 columns (4 features + target + species)
        assert len(df.columns) == 6

        # Check data types
        assert df['target'].dtype in [np.int32, np.int64]
        assert df['species'].dtype == object

    def test_load_iris_as_dataframe_target_mapping(self):
        """Test that target values correctly map to species names"""
        df = load_iris_as_dataframe()

        # Check that target 0 maps to setosa
        setosa_rows = df[df['target'] == 0]
        assert all(setosa_rows['species'] == 'setosa')

        # Check that target 1 maps to versicolor
        versicolor_rows = df[df['target'] == 1]
        assert all(versicolor_rows['species'] == 'versicolor')

        # Check that target 2 maps to virginica
        virginica_rows = df[df['target'] == 2]
        assert all(virginica_rows['species'] == 'virginica')

    def test_get_dataset_info_completeness(self):
        """Test that dataset info contains all expected fields"""
        info = get_dataset_info()

        assert isinstance(info, dict)
        assert 'feature_names' in info
        assert 'target_names' in info
        assert 'n_samples' in info
        assert 'n_features' in info
        assert 'n_classes' in info
        assert 'class_distribution' in info

        # Verify values
        assert info['n_samples'] == 150
        assert info['n_features'] == 4
        assert info['n_classes'] == 3
        assert len(info['class_distribution']) == 3

    def test_get_dataset_info_balanced_classes(self):
        """Test that Iris dataset has balanced class distribution"""
        info = get_dataset_info()

        class_counts = info['class_distribution']

        # Each class should have 50 samples
        assert all(count == 50 for count in class_counts.values())

    def test_reproducibility_with_random_state(self):
        """Test that same random_state produces identical splits"""
        X_train1, X_test1, y_train1, y_test1 = load_iris_data(test_size=0.2, random_state=42)
        X_train2, X_test2, y_train2, y_test2 = load_iris_data(test_size=0.2, random_state=42)

        # Check arrays are identical
        assert np.array_equal(X_train1, X_train2)
        assert np.array_equal(X_test1, X_test2)
        assert np.array_equal(y_train1, y_train2)
        assert np.array_equal(y_test1, y_test2)


@pytest.mark.parametrize("test_size", [0.1, 0.2, 0.3, 0.4])
def test_various_split_sizes(test_size):
    """Test that different test_size values work correctly"""
    X_train, X_test, y_train, y_test = load_iris_data(test_size=test_size, random_state=42)

    total = X_train.shape[0] + X_test.shape[0]
    assert total == 150
    assert X_test.shape[0] > 0
    assert X_train.shape[0] > 0


def test_data_types():
    """Test that loaded data has correct numpy dtypes"""
    X_train, X_test, y_train, y_test = load_iris_data()

    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(y_test, np.ndarray)

    # Features should be float
    assert X_train.dtype in [np.float32, np.float64]

    # Targets should be int
    assert y_train.dtype in [np.int32, np.int64]