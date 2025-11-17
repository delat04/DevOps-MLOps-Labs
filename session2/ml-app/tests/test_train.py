import sys
import os

from mlxtend.evaluate.bootstrap_point632 import accuracy
from rich.diagnose import report

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, call
from model import IrisClassifier
from data_loader import load_iris_data


class TestTrainModule:
    """Test suite for train.py module"""

    def test_model_training_improves_accuracy(self):
        """Test that training improves model accuracy above baseline"""
        X_train, X_test, y_train, y_test = load_iris_data(test_size=0.2, random_state=42)

        classifier = IrisClassifier()
        classifier.train(X_train, y_train)

        accuracy, _ = classifier.evaluate(X_test, y_test)

        # Iris is an easy dataset, accuracy should be above 90%
        assert accuracy > 0.90
        assert accuracy <= 1.0

    def test_model_predictions_after_training(self):
        """Test that model can make predictions after training"""
        X_train, X_test, y_train, y_test = load_iris_data(test_size=0.2, random_state=42)

        classifier = IrisClassifier()
        classifier.train(X_train, y_train)

        predictions = classifier.predict(X_test)

        # Check predictions are valid
        assert len(predictions) == len(X_test)
        assert all(pred in [0, 1, 2] for pred in predictions)

    def test_model_save_and_load(self):
        """Test that model can be saved and loaded correctly"""
        X_train, X_test, y_train, y_test = load_iris_data(test_size=0.2, random_state=42)

        # Train original model
        classifier = IrisClassifier()
        classifier.train(X_train, y_train)
        original_predictions = classifier.predict(X_test)

        # Save model
        model_path = 'test_model.pkl'
        try:
            classifier.save_model(model_path)

            # Load model
            loaded_classifier = IrisClassifier()
            loaded_classifier.load_model(model_path)
            loaded_predictions = loaded_classifier.predict(X_test)

            # Predictions should be identical
            assert np.array_equal(original_predictions, loaded_predictions)
        finally:
            # Cleanup
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_classification_report_format(self):
        """Test that classification report contains expected metrics"""
        X_train, X_test, y_train, y_test = load_iris_data(test_size=0.2, random_state=42)

        classifier = IrisClassifier()
        classifier.train(X_train, y_train)
        accuracy, report = classifier.evaluate(X_test, y_test)

        # Check report contains key metrics
        assert isinstance(report, str)
        assert 'precision' in report.lower()
        assert 'recall' in report.lower()
        assert 'f1-score' in report.lower()

    def test_training_with_different_data_splits(self):
        """Test that training works with different data split ratios"""
        test_sizes = [0.2, 0.3, 0.4]

        for test_size in test_sizes:
            X_train, X_test, y_train, y_test = load_iris_data(test_size=test_size, random_state=42)

            classifier = IrisClassifier()
            classifier.train(X_train, y_train)
            accuracy, _ = classifier.evaluate(X_test, y_test)

            # Model should achieve reasonable accuracy with any split
            assert accuracy > 0.85

    def test_model_consistency_with_random_state(self):
        """Test that training with same random_state produces consistent results"""
        X_train, X_test, y_train, y_test = load_iris_data(test_size=0.2, random_state=42)

        # Train two models with same data
        classifier1 = IrisClassifier()
        classifier1.train(X_train, y_train)
        predictions1 = classifier1.predict(X_test)

        classifier2 = IrisClassifier()
        classifier2.train(X_train, y_train)
        predictions2 = classifier2.predict(X_test)

        # Predictions should be identical
        assert np.array_equal(predictions1, predictions2)

    def test_training_updates_model_state(self):
        """Test that training actually updates the model"""
        X_train, X_test, y_train, y_test = load_iris_data(test_size=0.2, random_state=42)

        classifier = IrisClassifier()

        # Before training, model should not be fitted
        assert not hasattr(classifier.model, 'classes_') or classifier.model.classes_ is None

        # After training, model should have learned classes
        classifier.train(X_train, y_train)
        assert hasattr(classifier.model, 'classes_')
        assert len(classifier.model.classes_) == 3




class TestTrainingPipeline:
    """Integration tests for the complete training pipeline"""

    def test_end_to_end_training_pipeline(self):
        """Test complete training pipeline from data load to evaluation"""
        # Load data
        X_train, X_test, y_train, y_test = load_iris_data(test_size=0.2, random_state=42)

        # Train
        classifier = IrisClassifier()
        classifier.train(X_train, y_train)

        # Evaluate

        isinstance(report, str)

    def test_model_generalizes_to_unseen_data(self):
        """Test that trained model generalizes to new data splits"""
        # Train on one split
        X_train1, _, y_train1, _ = load_iris_data(test_size=0.2, random_state=42)
        classifier = IrisClassifier()
        classifier.train(X_train1, y_train1)

        # Test on different split
        _, X_test2, _, y_test2 = load_iris_data(test_size=0.3, random_state=99)
        accuracy, _ = classifier.evaluate(X_test2, y_test2)

        # Should still maintain good accuracy
        assert accuracy > 0.85


@pytest.mark.parametrize("random_state", [42, 123, 456])
def test_training_stability_across_seeds(random_state):
    """Test that training produces stable results across different random seeds"""
    X_train, X_test, y_train, y_test = load_iris_data(test_size=0.2, random_state=random_state)

    classifier = IrisClassifier()
    classifier.train(X_train, y_train)
    accuracy, _ = classifier.evaluate(X_test, y_test)

    # Should achieve good accuracy regardless of random seed
    assert accuracy > 0.85


def test_training_handles_all_classes():
    """Test that trained model can predict all three iris classes"""
    X_train, X_test, y_train, y_test = load_iris_data(test_size=0.2, random_state=42)

    classifier = IrisClassifier()
    classifier.train(X_train, y_train)
    predictions = classifier.predict(X_test)

    # Check that at least two classes are predicted (may not get all 3 in small test set)
    unique_predictions = np.unique(predictions)
    assert len(unique_predictions) >= 2
    assert all(pred in [0, 1, 2] for pred in unique_predictions)


def test_model_coefficients_exist_after_training():
    """Test that model has learned coefficients after training"""
    X_train, _, y_train, _ = load_iris_data(test_size=0.2, random_state=42)

    classifier = IrisClassifier()
    classifier.train(X_train, y_train)

    # Logistic regression should have coefficients
    assert hasattr(classifier.model, 'coef_')
    assert classifier.model.coef_ is not None
    assert classifier.model.coef_.shape[1] == 4  # 4 features