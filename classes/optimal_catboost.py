from collections import Counter
from typing import Any, Dict, List, Optional

import joblib
import matplotlib.pyplot as plt
import optuna
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target
import logging




class OptimalCatBoostClassifier(CatBoostClassifier):
    """
    A streamlined wrapper for CatBoostClassifier with Optuna optimization,
    hardcoded evaluation metrics, feature importance plotting, and optional SQLite logging.

    Args:
        features (List[str]): List of feature columns.
        param_grid (Dict): Parameter grid for Optuna optimization.
        n_trials (int): Number of Optuna trials.
        cat_features (Optional[List[str]]): List of categorical feature names.
        use_class_weights (bool): Whether to compute and apply class weights. Defaults to True.
        cache_path (Optional[str]): Path to SQLite database for storing Optuna results.
        study_name (Optional[str]): Name for the study in the cache database
    """

    def __init__(
        self,
        features: List[str],
        param_grid: Dict,
        n_trials: int = 10,
        cat_features: Optional[List[str]] = None,
        use_class_weights: bool = True,
        cache_path: Optional[str] = None,
        study_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.features = features
        self.param_grid = param_grid
        self.n_trials = n_trials
        self.cat_features = cat_features
        self.use_class_weights = use_class_weights
        self.cache_path = cache_path
        self.study_name = study_name
        self._LOGGER = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def compute_class_weights(y: pd.Series) -> List[float]:
        """Compute class weights based on the target distribution."""
        class_counts = Counter(y)
        total_samples = len(y)
        return [total_samples / (len(class_counts) * class_counts[label]) for label in sorted(class_counts.keys())]

    def _initialize_target_properties(self, y: pd.Series):
        """Set up target type and class weights."""
        self.target_type_ = type_of_target(y)
        if self.use_class_weights:
            self.class_weights_ = self.compute_class_weights(y)

    def _optimize_parameters(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series
    ) -> Optional[Dict[str, Any]]:
        """Optimize parameters using Optuna."""

        def objective(trial):
            params = {
                k: trial.suggest_categorical(k, v) if isinstance(v, list) else trial.suggest_float(k, *v)
                for k, v in self.param_grid.items()
            }
            model = CatBoostClassifier(
                **params, cat_features=self.cat_features, class_weights=self.class_weights_, verbose=0
            )
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=100, verbose=0)
            preds = model.predict(X_valid)
            return f1_score(y_valid, preds, average="binary" if self.target_type_ == "binary" else "weighted")

        study = optuna.create_study(
            direction="maximize",
            storage=f"sqlite:///{self.cache_path}" if self.cache_path else None,
            study_name=self.study_name if self.study_name is not None else "catboost_optimization",
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Optimize parameters and train the CatBoost model."""
        self._initialize_target_properties(y)
        
        X_train, X_valid, y_train, y_valid = train_test_split(
            X[self.features], y, test_size=0.2, random_state=42, stratify=y
        )
        self.best_params_ = self._optimize_parameters(X_train, y_train, X_valid, y_valid)

        if not self.best_params_:
            raise ValueError("Optuna failed to find optimal parameters.")

        self.set_params(**self.best_params_, cat_features=self.cat_features, class_weights=self.class_weights_, verbose=0)
        super().fit(X[self.features], y)

        y_pred = self.predict(X_valid)
        self.training_results_ = pd.DataFrame(
            {
                "Metric": ["Accuracy", "F1 Score", "Precision", "Recall"],
                "Score": [
                    accuracy_score(y_valid, y_pred),
                    f1_score(y_valid, y_pred, average="binary" if self.target_type_ == "binary" else "weighted"),
                    precision_score(y_valid, y_pred, average="binary" if self.target_type_ == "binary" else "weighted"),
                    recall_score(y_valid, y_pred, average="binary" if self.target_type_ == "binary" else "weighted"),
                ],
            }
        )
        self._LOGGER.info("Training Results: %s", self.training_results_.to_dict(orient="records"))

    def plot_feature_importance(self):
        """Plot feature importance."""
        if not self.is_fitted():
            raise ValueError("Model is not trained yet. Call fit() first.")

        feature_importances = self.get_feature_importance()
        plt.figure(figsize=(10, 8))
        plt.barh(self.feature_names_, feature_importances)
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Feature Importance")
        plt.show()

    def save_model(self, path: str):
        """Save the trained model to disk."""
        joblib.dump(self, path)
        self._LOGGER.info("Model saved to %s", path)

    def load_model(self, path: str):
        """Load the trained model from disk."""
        loaded_model = joblib.load(path)
        self.__dict__.update(loaded_model.__dict__)
        self._LOGGER.info("Model loaded from %s", path)


class OptimalCatBoostRegressor(CatBoostRegressor):
    """
    A streamlined wrapper for CatBoostRegressor with Optuna optimization,
    hardcoded evaluation metrics, feature importance plotting, and optional SQLite logging.

    Args:
        features (List[str]): List of feature columns.
        param_grid (Dict): Parameter grid for Optuna optimization.
        n_trials (int): Number of Optuna trials.
        cat_features (Optional[List[str]]): List of categorical feature names.
        cache_path (Optional[str]): Path to SQLite database for storing Optuna results.
        study_name (Optional[str]): Custom name for the Optuna study. Defaults to None.
    """

    def __init__(
        self,
        features: List[str],
        param_grid: Dict,
        n_trials: int = 10,
        cat_features: Optional[List[str]] = None,
        cache_path: Optional[str] = None,
        study_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.features = features
        self.param_grid = param_grid
        self.n_trials = n_trials
        self.cat_features = cat_features
        self.cache_path = cache_path
        self.study_name = study_name
        self.target_type_ = None
        self.best_params_ = None
        self.training_results_ = None
        self._LOGGER = logging.getLogger(self.__class__.__name__)

    def _initialize_target_properties(self, y: pd.Series):
        """Set up target type and validate."""
        self.target_type_ = type_of_target(y)
        if self.target_type_ not in ["continuous"]:
            raise ValueError(f"Target type '{self.target_type_}' is not supported for regression.")

    def _optimize_parameters(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series
    ) -> Optional[Dict[str, Any]]:
        """Optimize parameters using Optuna."""

        def objective(trial):
            params = {
                k: trial.suggest_categorical(k, v) if isinstance(v, list) else trial.suggest_float(k, *v)
                for k, v in self.param_grid.items()
            }
            model = CatBoostRegressor(**params, cat_features=self.cat_features, verbose=0)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=100, verbose=0)
            preds = model.predict(X_valid)
            return root_mean_squared_error(y_valid, preds)

        study = optuna.create_study(
            direction="minimize",  # Minimizing RMSE
            storage=f"sqlite:///{self.cache_path}" if self.cache_path else None,
            study_name=self.study_name if self.study_name else "catboost_regressor_optimization",
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Optimize parameters and train the CatBoost model."""
        self._initialize_target_properties(y)

        X_train, X_valid, y_train, y_valid = train_test_split(
            X[self.features], y, test_size=0.2, random_state=42
        )
        self.best_params_ = self._optimize_parameters(X_train, y_train, X_valid, y_valid)

        if not self.best_params_:
            raise ValueError("Optuna failed to find optimal parameters.")

        self.set_params(**self.best_params_, cat_features=self.cat_features, verbose=0)
        super().fit(X[self.features], y)

        y_pred = self.predict(X_valid)
        self.training_results_ = pd.DataFrame(
            {
                "Metric": ["Mean Absolute Error", "Mean Squared Error", "Root Mean Squared Error", "R2 Score"],
                "Score": [
                    mean_absolute_error(y_valid, y_pred),
                    mean_squared_error(y_valid, y_pred),
                    root_mean_squared_error(y_valid, y_pred),  # RMSE
                    r2_score(y_valid, y_pred),
                ],
            }
        )
        self._LOGGER.info("Training Results: %s", self.training_results_.to_dict(orient="records"))

    def plot_feature_importance(self):
        """Plot feature importance."""
        if not hasattr(self, "feature_importances_"):
            raise ValueError("Model is not trained yet. Call fit() first.")

        feature_importances = self.get_feature_importance()
        plt.figure(figsize=(10, 8))
        plt.barh(self.features, feature_importances)
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Feature Importance")
        plt.show()

    def save_model(self, path: str):
        """Save the trained model to disk."""
        joblib.dump(self, path)
        self._LOGGER.info("Model saved to %s", path)

    def load_model(self, path: str):
        """Load the trained model from disk."""
        loaded_model = joblib.load(path)
        self.__dict__.update(loaded_model.__dict__)
        self._LOGGER.info("Model loaded from %s", path)

    def get_params(self, deep=True):
        """Return estimator parameters for cloning."""
        params = super(CatBoostRegressor, self).get_params(deep=deep)
        params.update({
            "features": self.features,
            "param_grid": self.param_grid,
            "n_trials": self.n_trials,
            "cat_features": self.cat_features,
            "cache_path": self.cache_path,
            "study_name": self.study_name,
        })
        return params


    def set_params(self, **params):
        """Set estimator parameters for cloning."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        super(CatBoostRegressor, self).set_params(**params)
        return self