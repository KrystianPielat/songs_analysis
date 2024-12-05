import os
import numpy as np
from typing import List, Dict, Optional
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from tqdm.auto import tqdm
from .optimal_catboost import OptimalCatBoostClassifier, OptimalCatBoostRegressor
import logging
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE


LOGGER = logging.getLogger(__name__)

class BasePipeline:
    """Base class for a machine learning pipeline to handle data preprocessing, training, evaluation, and SHAP analysis."""

    def __init__(self, df: pd.DataFrame, target_column: str, num_features: List[str], cat_features: List[str]) -> None:
        """
        Initializes the pipeline with dataset, target column, and feature columns.

        Args:
            df (pd.DataFrame): The dataset.
            target_column (str): Name of the target column.
            num_features (List[str]): List of numerical feature names.
            cat_features (List[str]): List of categorical feature names.
        """
        self.df = df
        self.target_column = target_column
        self.num_features = num_features
        self.cat_features = cat_features
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.shap_values = None
        self.X_test_transformed = None  # To store transformed test data
        LOGGER.info(f"Initializing pipeline for target variable: {target_column}")

    def get_feature_preprocessing_pipeline(self, encode_categorical: bool = True) -> ColumnTransformer:
        """Sets up a preprocessing pipeline for numerical and categorical features.

        Returns:
            ColumnTransformer: Preprocessing pipeline for the feature columns.
        """
        LOGGER.info("Setting up the pipeline...")
        
        # Preprocessing for numerical and categorical data
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        if encode_categorical:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
        else:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent'))
            ])

        # Combine both numeric and categorical preprocessors
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.num_features),
                ('cat', categorical_transformer, self.cat_features)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        )
        preprocessor.set_output(transform="pandas")
        return preprocessor

    def split(self) -> None:
        """Splits the data into training and testing sets."""
        LOGGER.info("Splitting the data into training and testing sets...")
        X = self.df[self.num_features + self.cat_features]
        y = self.df[self.target_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        LOGGER.info("Data splitting completed.")

    def train_model(self) -> None:
        """Trains the model using the training data."""
        LOGGER.info("Training the model...")
        self.pipeline.fit(self.X_train, self.y_train)
        LOGGER.info("Model training completed.")


    def save_results(self, save_dir: str = 'results') -> None:
        """Saves SHAP plots and evaluation table to the specified directory.

        Args:
            save_dir (str): Directory to save results, default is 'results'.
        """
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.info(f"Saving results to {save_dir}...")

        evaluation_df = self.evaluate_model()
        evaluation_file = os.path.join(save_dir, 'evaluation_results.csv')
        evaluation_df.to_csv(evaluation_file, index=False)
        LOGGER.info(f"Evaluation results saved to {evaluation_file}")

        if self.shap_values is not None and self.X_test_transformed is not None:
            # Convert transformed data to DataFrame if needed
            if not isinstance(self.X_test_transformed, pd.DataFrame):
                feature_names = self.pipeline.named_steps['preprocessor'].get_feature_names_out()
                self.X_test_transformed = pd.DataFrame(self.X_test_transformed, columns=feature_names)
        
            # Check SHAP values dimensionality
            if len(self.shap_values.values.shape) > 2:  # Multi-class SHAP values
                for class_idx in range(self.shap_values.values.shape[2]):  # Iterate over classes
                    class_shap_values = self.shap_values.values[..., class_idx]
        
                    # Save SHAP summary plot for each class
                    plt.figure()
                    shap.summary_plot(class_shap_values, self.X_test_transformed, plot_type="bar", show=False)
                    summary_plot_file = os.path.join(save_dir, f'shap_summary_plot_class_{class_idx}.png')
                    plt.savefig(summary_plot_file, bbox_inches='tight')
                    plt.close()
                    LOGGER.info(f"SHAP summary plot for class {class_idx} saved to {summary_plot_file}")
        
                    # Save SHAP beeswarm plot for each class
                    plt.figure(figsize=(10, 8))
                    shap.summary_plot(class_shap_values, self.X_test_transformed, show=False)
                    beeswarm_plot_file = os.path.join(save_dir, f'shap_beeswarm_plot_class_{class_idx}.png')
                    plt.savefig(beeswarm_plot_file, bbox_inches='tight')
                    plt.close()
                    LOGGER.info(f"SHAP beeswarm plot for class {class_idx} saved to {beeswarm_plot_file}")
            else:  # Binary classification or regression
                plt.figure()
                shap.summary_plot(self.shap_values.values, self.X_test_transformed, plot_type="bar", show=False)
                summary_plot_file = os.path.join(save_dir, 'shap_summary_plot.png')
                plt.savefig(summary_plot_file, bbox_inches='tight')
                plt.close()
                LOGGER.info(f"SHAP summary plot saved to {summary_plot_file}")
        
                plt.figure(figsize=(10, 8))
                shap.summary_plot(self.shap_values.values, self.X_test_transformed, show=False)
                beeswarm_plot_file = os.path.join(save_dir, 'shap_beeswarm_plot.png')
                plt.savefig(beeswarm_plot_file, bbox_inches='tight')
                plt.close()
                LOGGER.info(f"SHAP beeswarm plot saved to {beeswarm_plot_file}")


    def run_pipeline(self, save_dir: str = 'dashboard/results') -> None:
        """Runs the full pipeline including splitting, training, evaluation, SHAP analysis, and result saving.

        Args:
            save_dir (str): Directory to save results, default is 'dashboard/results'.
        """
        with tqdm(total=4, desc="Pipeline Progress", unit="step") as pbar:
            LOGGER.info(f"Starting pipeline execution for prediction of {self.target_column}...")

            pbar.set_description("Splitting data")
            self.split()
            pbar.update(1)

            pbar.set_description("Training model")
            self.train_model()
            pbar.update(1)

            pbar.set_description("Performing SHAP analysis")
            self.perform_shap_analysis()
            pbar.update(1)

            pbar.set_description("Saving results")
            self.save_results(save_dir)
            pbar.update(1)

            LOGGER.info("Pipeline execution completed.")

class OptimalClassificationPipeline(BasePipeline):
    """Pipeline for classification models using OptimalCatBoostClassifier with SHAP analysis."""

    def __init__(
        self, 
        df: pd.DataFrame, 
        target_column: str, 
        num_features: List[str], 
        cat_features: List[str], 
        param_grid: Dict, 
        handle_imbalance: bool = True, 
        n_trials: int = 10, 
        cache_path: Optional[str] = None
    ) -> None:
        """
        Initializes the classification pipeline with OptimalCatBoostClassifier.

        Args:
            df (pd.DataFrame): The dataset.
            target_column (str): Name of the target column.
            num_features (List[str]): List of numerical feature names.
            cat_features (List[str]): List of categorical feature names.
            param_grid (Dict): Parameter grid for Optuna optimization.
            handle_imbalance (bool, optional): Whether to handle class imbalance. Defaults to True.
            n_trials (int): Number of trials for Optuna optimization. Defaults to 10.
            cache_path (Optional[str]): Path to cache Optuna study results.
        """
        super().__init__(df, target_column, num_features, cat_features)
        self.model_type = "classification"
        self.param_grid = param_grid
        self.handle_imbalance = handle_imbalance
        self.n_trials = n_trials
        self.cache_path = cache_path
        self.class_weights = None
        self.pipeline = self.get_pipeline()

    def get_pipeline(self) -> Pipeline:
        """Sets up the classification pipeline with preprocessing and OptimalCatBoostClassifier.

        Returns:
            Pipeline: Classification pipeline.
        """
        preprocessor = self.get_feature_preprocessing_pipeline(encode_categorical=False)

        model = OptimalCatBoostClassifier(
            features=self.num_features + self.cat_features,
            param_grid=self.param_grid,
            n_trials=self.n_trials,
            cat_features=self.cat_features,
            use_class_weights=self.handle_imbalance,
            cache_path=self.cache_path,
            study_name=f"catboost_{self.target_column}"
        )

        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
    def train_model(self):
        """Trains the OptimalCatBoostClassifier using the full pipeline."""
        LOGGER.info("Training the OptimalCatBoostClassifier through the pipeline...")
        self.pipeline.fit(self.X_train, self.y_train)
        LOGGER.info("Pipeline training completed.")
    

    def evaluate_model(self) -> pd.DataFrame:
        """Evaluates the classification model and generates a metrics DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing classification metrics (accuracy, precision, recall, F1 score).
        """
        return self.pipeline.named_steps['model'].training_results_

    def perform_shap_analysis(self):
        """Performs SHAP analysis for feature importance on the OptimalCatBoostClassifier."""
        LOGGER.info("Performing SHAP analysis for OptimalCatBoostClassifier...")

        # Transform the test set using the preprocessing pipeline
        self.X_test_transformed = self.pipeline.named_steps["preprocessor"].transform(self.X_test)

        # Create and compute SHAP explainer
        explainer = shap.TreeExplainer(self.pipeline.named_steps['model'])
        self.shap_values = explainer(self.X_test_transformed, check_additivity=False)

        LOGGER.info("SHAP analysis for classification completed.")

class OptimalRegressionPipeline(BasePipeline):
    """Pipeline for regression models using OptimalCatBoostRegressor with SHAP analysis."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str,
        num_features: List[str],
        cat_features: List[str],
        param_grid: Dict,
        n_trials: int = 10,
        cache_path: Optional[str] = None,
    ) -> None:
        """
        Initializes the regression pipeline with OptimalCatBoostRegressor.

        Args:
            df (pd.DataFrame): The dataset.
            target_column (str): Name of the target column.
            num_features (List[str]): List of numerical feature names.
            cat_features (List[str]): List of categorical feature names.
            param_grid (Dict): Parameter grid for Optuna optimization.
            n_trials (int): Number of trials for Optuna optimization. Defaults to 10.
            cache_path (Optional[str]): Path to cache Optuna study results.
        """
        super().__init__(df, target_column, num_features, cat_features)
        self.model_type = "regression"
        self.param_grid = param_grid
        self.n_trials = n_trials
        self.cache_path = cache_path
        self.pipeline = self.get_pipeline()

    def get_pipeline(self) -> TransformedTargetRegressor:
        """Sets up the regression pipeline with preprocessing, model, and target transformation.
    
        Returns:
            TransformedTargetRegressor: Regression pipeline with target transformation.
        """
        preprocessor = self.get_feature_preprocessing_pipeline(encode_categorical=False)
    
        target_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        model = OptimalCatBoostRegressor(
            features=self.num_features + self.cat_features,
            param_grid=self.param_grid,
            n_trials=self.n_trials,
            cat_features=self.cat_features,
            cache_path=self.cache_path,
        )
        
        regression_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        return TransformedTargetRegressor(
            regressor=regression_pipeline,
            transformer=target_transformer
        )


    def train_model(self):
        """Trains the OptimalCatBoostRegressor using the full pipeline."""
        LOGGER.info("Training the OptimalCatBoostRegressor through the pipeline...")
        self.pipeline.fit(self.X_train, self.y_train)
        LOGGER.info("Pipeline training completed.")

    def evaluate_model(self) -> pd.DataFrame:
        """Evaluates the regression model and generates a metrics DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing regression metrics (MAE, MSE, RMSE, R2 Score).
        """
        y_pred = self.pipeline.predict(self.X_test)
        evaluation_results = {
            "Metric": ["Mean Absolute Error", "Mean Squared Error", "Root Mean Squared Error", "R2 Score"],
            "Score": [
                mean_absolute_error(self.y_test, y_pred),
                mean_squared_error(self.y_test, y_pred),
                mean_squared_error(self.y_test, y_pred, squared=False),  # RMSE
                r2_score(self.y_test, y_pred),
            ],
        }
        return pd.DataFrame(evaluation_results)

    def perform_shap_analysis(self):
        """Performs SHAP analysis for feature importance on the OptimalCatBoostRegressor."""
        LOGGER.info("Performing SHAP analysis for OptimalCatBoostRegressor...")
    
        # Access the internal pipeline of TransformedTargetRegressor
        regression_pipeline = self.pipeline.regressor_
    
        # Transform the test set using the preprocessing pipeline
        self.X_test_transformed = regression_pipeline.named_steps["preprocessor"].transform(self.X_test)
    
        # Create and compute SHAP explainer
        model = regression_pipeline.named_steps['model']
        explainer = shap.TreeExplainer(model)
        self.shap_values = explainer(self.X_test_transformed, check_additivity=False)
    
        LOGGER.info("SHAP analysis for regression completed.")
