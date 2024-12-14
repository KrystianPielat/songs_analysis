from collections import Counter
import uuid
from typing import Any, Dict, List, Optional
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target
import logging
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer
import numpy as np
from optuna.integration import OptunaSearchCV
from sklearn.model_selection import StratifiedKFold, KFold

class OptimalCatBoostClassifier(CatBoostClassifier):
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
        self.study_name = study_name or f'catboost_optimization_{uuid.uuid4()}'
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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
 
    def _cross_val_metrics(self, model: CatBoostClassifier, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold):
        """Compute cross-validated metrics."""
        metrics = {"Accuracy": [], "F1 Score": [], "Precision": [], "Recall": []}
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics["Accuracy"].append(accuracy_score(y_test, y_pred))
            metrics["F1 Score"].append(f1_score(y_test, y_pred, average="weighted"))
            metrics["Precision"].append(precision_score(y_test, y_pred, average="weighted"))
            metrics["Recall"].append(recall_score(y_test, y_pred, average="weighted"))
        return {metric: np.mean(scores) for metric, scores in metrics.items()}

    def compute_class_weights(self, y: pd.Series) -> List[float]:
        class_counts = Counter(y)
        total_samples = len(y)
        return [total_samples / (len(class_counts) * class_counts[label]) for label in sorted(class_counts.keys())]

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.target_type_ = type_of_target(y)
        if self.target_type_ not in ["binary", "multiclass"]:
            raise ValueError(f"Unsupported target type: {self.target_type_}")

        class_weights = self.compute_class_weights(y) if self.use_class_weights else None
        params = {
            key: optuna.distributions.CategoricalDistribution(value) if isinstance(value, list) else optuna.distributions.FloatDistribution(*value)
            for key, value in self.param_grid.items()
        }
        
        study = optuna.create_study(study_name=self.study_name, storage=f"sqlite:///{self.cache_path}", load_if_exists=True, direction="maximize")

        optuna_search = OptunaSearchCV(
            estimator=CatBoostClassifier(cat_features=self.cat_features, class_weights=class_weights, verbose=0),
            param_distributions=params,
            cv=self.cv,
            n_trials=self.n_trials,
            scoring="f1_weighted",
            refit=False,
            random_state=42,
            n_jobs=-1,
            study=study
        )

        optuna_search.fit(X[self.features], y)

        # Update self with the best parameters
        self.best_params_ = optuna_search.best_params_
        self.set_params(**self.best_params_, cat_features=self.cat_features, class_weights=class_weights, verbose=0)
        
        # Compute training results
        self._LOGGER.info("Computing training results with cross validation...")
        crossval_model = CatBoostClassifier(cat_features=self.cat_features, class_weights=class_weights, verbose=0).set_params(**self.best_params_)
        self.training_results_ = self._cross_val_metrics(crossval_model, X[self.features], y, self.cv)
        
        # Fit the current instance with the optimized parameters on the whole dataset
        self._LOGGER.info("Fitting the final model on the whole dataset...")
        super().fit(X[self.features], y)

        self._LOGGER.info("Training completed with results: %s", self.training_results_)

    @property
    def training_results(self):
        check_is_fitted(self, msg='First fit the model')
        return pd.DataFrame([(k, round(v, 3)) for k, v in self.training_results_.items()], columns=["Metric", "Score"])

    def get_params(self, deep=True):
        """Return estimator parameters for Scikit-learn compatibility."""
        params = super(CatBoostClassifier, self).get_params(deep=deep)
        params.update({
            "features": self.features,
            "param_grid": self.param_grid,
            "n_trials": self.n_trials,
            "cat_features": self.cat_features,
            "use_class_weights": self.use_class_weights,
            "cache_path": self.cache_path,
            "study_name": self.study_name,
        })
        return params

    def set_params(self, **params):
        """Set estimator parameters for Scikit-learn compatibility."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        super().set_params(**params)
        return self
        
    def plot_feature_importance(self):
        """Plot the top 10 most important features using Seaborn."""
        if not self.is_fitted():
            raise ValueError("Model is not trained yet. Call fit() first.")

        feature_importances = self.get_feature_importance()
        feature_names = self.feature_names_
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": feature_importances
        }).sort_values(by="Importance", ascending=False).head(10)  # Top 10 features

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
        plt.title("Top 10 Feature Importances", fontsize=16)
        plt.xlabel("Importance", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.tight_layout()
        plt.show()

class OptimalCatBoostRegressor(CatBoostRegressor):
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
        self.study_name = study_name or f'catboost_optimization_{uuid.uuid4()}'
        self.cv = KFold(n_splits=5, shuffle=True, random_state=42)
        self._LOGGER = logging.getLogger(self.__class__.__name__)

    def _cross_val_metrics(self, model: CatBoostClassifier, X: pd.DataFrame, y: pd.Series, cv: KFold):
        """Compute cross-validated metrics."""
        metrics = {"MAE": [], "MSE": [], "RMSE": [], "R2": []}
        y = pd.Series(y)

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train, verbose=0)
            y_pred = model.predict(X_test)
            metrics["MAE"].append(mean_absolute_error(y_test, y_pred))
            metrics["MSE"].append(mean_squared_error(y_test, y_pred))
            metrics["RMSE"].append(mean_squared_error(y_test, y_pred, squared=False))
            metrics["R2"].append(r2_score(y_test, y_pred))
        return {metric: np.mean(scores) for metric, scores in metrics.items()}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        params = {
            key: optuna.distributions.CategoricalDistribution(value) if isinstance(value, list) else optuna.distributions.FloatDistribution(*value)
            for key, value in self.param_grid.items()
        }
        study = optuna.create_study(study_name=self.study_name, storage=f"sqlite:///{self.cache_path}", load_if_exists=True, direction="maximize")
        optuna_search = OptunaSearchCV(
            estimator=CatBoostRegressor(cat_features=self.cat_features, verbose=0),
            param_distributions=params,
            cv=self.cv,
            n_trials=self.n_trials,
            scoring=make_scorer(mean_squared_error, greater_is_better=False),
            refit=False,
            random_state=42,
            n_jobs=-1,
            study=study
        )

        optuna_search.fit(X[self.features], y)

        # Update self with the best parameters
        self.best_params_ = optuna_search.best_params_
        self.set_params(**self.best_params_, cat_features=self.cat_features, verbose=0)

        # Compute training results
        self._LOGGER.info("Computing training results with cross validation...")
        crossval_model = CatBoostRegressor(cat_features=self.cat_features, verbose=0).set_params(**self.best_params_)
        self.training_results_ = self._cross_val_metrics(crossval_model, X[self.features], y, self.cv)
        
        # Fit the current instance with the optimized parameters on the whole dataset
        self._LOGGER.info("Fitting the final model on the whole dataset...")
        super().fit(X[self.features], y)

        self._LOGGER.info("Training completed with results: %s", self.training_results_)

    @property
    def training_results(self):
        check_is_fitted(self, msg='First fit the model')
        return pd.DataFrame([(k, round(v, 3)) for k, v in self.training_results_.items()], columns=["Metric", "Score"])

    def plot_feature_importance(self):
        """Plot the top 10 most important features using Seaborn."""
        if not self.is_fitted():
            raise ValueError("Model is not trained yet. Call fit() first.")

        feature_importances = self.get_feature_importance()
        feature_names = self.feature_names_
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": feature_importances
        }).sort_values(by="Importance", ascending=False).head(10)  # Top 10 features

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
        plt.title("Top 10 Feature Importances", fontsize=16)
        plt.xlabel("Importance", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.tight_layout()
        plt.show()


    def get_params(self, deep=True):
        """Return estimator parameters for Scikit-learn compatibility."""
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
        """Set estimator parameters for Scikit-learn compatibility."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        super().set_params(**params)
        return self
