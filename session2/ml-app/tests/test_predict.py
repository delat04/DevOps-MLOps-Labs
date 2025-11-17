import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from io import StringIO
from model import IrisClassifier
from data_loader import load_iris_data


class TestPredictModule:
    def setup_method(self):
        """Setup method that runs before each test"""
        # Create and train a classifier for testing
        X_train, X_test, y_train, y_test = load_iris_data(test_size=0.3, random_state=42)
        self.classifier = IrisClassifier()
        self.classifier.train(X_train, y_train)

    def test_prediction_output_format(self):
        """Test that predictions return correct data types and shapes"""
        test_features = [[5.1, 3.5, 1.4, 0.2]]
        prediction = self.classifier.predict(test_features)

        assert isinstance(prediction, np.ndarray)
        assert len(prediction) == 1
        assert prediction[0] in [0, 1, 2]

    def test_prediction_probabilities(self):
        """Test that probability predictions sum to 1"""
        test_features = [[5.1, 3.5, 1.4, 0.2]]
        probabilities = self.classifier.model.predict_proba(test_features)[0]

        assert len(probabilities) == 3
        assert np.isclose(np.sum(probabilities), 1.0)
        assert all(0 <= p <= 1 for p in probabilities)

    def test_multiple_predictions(self):
        """Test batch prediction functionality"""
        test_features = [
            [5.1, 3.5, 1.4, 0.2],
            [6.7, 3.0, 5.2, 2.3],
            [5.9, 3.0, 4.2, 1.5]
        ]
        predictions = self.classifier.predict(test_features)

        assert len(predictions) == 3
        assert all(pred in [0, 1, 2] for pred in predictions)

    def test_prediction_consistency(self):
        """Test that same input produces same prediction"""
        test_features = [[5.1, 3.5, 1.4, 0.2]]

        pred1 = self.classifier.predict(test_features)[0]
        pred2 = self.classifier.predict(test_features)[0]

        assert pred1 == pred2

    def test_invalid_feature_count(self):
        """Test that prediction fails with wrong number of features"""
        invalid_features = [[5.1, 3.5, 1.4]]  # Only 3 features instead of 4

        with pytest.raises(ValueError):
            self.classifier.predict(invalid_features)


@pytest.mark.parametrize("features,expected_class", [
    ([5.1, 3.5, 1.4, 0.2], 0),  # Typical Setosa
    ([6.7, 3.0, 5.2, 2.3], 2),  # Typical Virginica
    ([5.9, 3.0, 4.2, 1.5], 1),  # Typical Versicolor
])
def test_known_examples_prediction(features, expected_class):
    """Test predictions on known examples"""
    X_train, X_test, y_train, y_test = load_iris_data(test_size=0.3, random_state=42)
    classifier = IrisClassifier()
    classifier.train(X_train, y_train)

    prediction = classifier.predict([features])[0]
    assert prediction == expected_class