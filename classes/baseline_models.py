from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import numpy as np
import pandas as pd
import logging

LOGGER = logging.getLogger(__name__)

class MeanBaselineModel:
    def fit(self, X, y):
        """Fits the model by calculating the mean of the target variable."""
        self.mean_value = np.mean(y)

    def predict(self, X):
        """Predicts the mean value for all instances."""
        return np.full(len(X), self.mean_value)

    def evaluate(self, y_true, y_pred):
        """Evaluates the model using regression metrics."""
        metrics = {
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": mean_squared_error(y_true, y_pred, squared=False),
            "R2": r2_score(y_true, y_pred)
        }
        return pd.DataFrame.from_dict(metrics, orient='index', columns=['Score'])


class MajorityBaselineModel:
    def fit(self, X, y):
        """Fits the model by determining the majority class."""
        self.majority_class = y.value_counts().idxmax()

    def predict(self, X):
        """Predicts the majority class for all instances."""
        return np.full(len(X), self.majority_class)

    def evaluate(self, y_true, y_pred):
        """Evaluates the model using classification metrics."""
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        return pd.DataFrame.from_dict(metrics, orient='index', columns=['Score'])


class RandomBaselineModel:
    def fit(self, X, y):
        """Fits the model by storing the unique classes."""
        self.classes = y.unique()

    def predict(self, X):
        """Predicts a random class for each instance."""
        return np.random.choice(self.classes, len(X))

    def evaluate(self, y_true, y_pred):
        """Evaluates the model using classification metrics."""
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        return pd.DataFrame.from_dict(metrics, orient='index', columns=['Score'])

