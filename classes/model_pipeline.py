import os
import numpy as np
from typing import List, Dict, Optional, Any
import pandas as pd
import seaborn as sns
import shap
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, OrdinalEncoder
from tqdm.auto import tqdm
from .optimal_catboost import OptimalCatBoostClassifier, OptimalCatBoostRegressor
import logging
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier



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
        self.cat_features = [ c for c in cat_features if c in df.columns]
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.shap_values = None
        self.X_test_transformed = None  # To store transformed test data
        LOGGER.info(f"Initializing pipeline for target variable: {target_column}")
    
    def get_feature_preprocessing_pipeline(self, encode_categorical: bool = True) -> ColumnTransformer:
        """
        Sets up a preprocessing pipeline for numerical and categorical features.
        Includes feature selection based on importance or correlation.
        
        Args:
            encode_categorical (bool): Whether to encode categorical features. Defaults to True.
    
        Returns:
            ColumnTransformer: Preprocessing pipeline for the feature columns.
        """
        LOGGER.info("Setting up the pipeline with feature selection...")
    
        # Dynamically choose model for feature selection based on task type
        if self.model_type == "regression":
            selection_model = RandomForestRegressor(random_state=42)
        elif self.model_type == "classification":
            selection_model = RandomForestClassifier(random_state=42)
        else:
            raise ValueError("Invalid model_type. Must be 'regression' or 'classification'.")


        feature_selector = SelectFromModel(selection_model, threshold="mean", prefit=False)

    
        # Preprocessing for numerical data
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
    
        # Preprocessing for categorical data
        if encode_categorical:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
                # ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype='object'))
            ])
        else:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ])
    
        # Combine numerical and categorical preprocessors
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, list(set(self.num_features))),
                ('cat', categorical_transformer, list(set(self.cat_features))),
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        )

    
        # preprocessor.set_output(transform="pandas")
        # Feature selection pipeline with preprocessor
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('feature_selection', feature_selector)
        ])
    
        pipeline.set_output(transform="pandas")
        return pipeline


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

        
    def save_results(self, save_dir: str = "results") -> None:
        """
        Saves evaluation metrics, SHAP plots, and classification report to the specified directory.
    
        Args:
            save_dir (str): Directory to save results. Defaults to 'results'.
        """
        LOGGER.info(f"Saving results to {save_dir}...")
        os.makedirs(save_dir, exist_ok=True)
    
        # Save evaluation metrics and plots
        LOGGER.info("Saving evaluation metrics...")
        self.evaluate_model(save_dir=save_dir)
    
        # Save SHAP analysis and plots
        LOGGER.info("Saving SHAP analysis results...")
        self.perform_shap_analysis(save_dir=save_dir)
    
        LOGGER.info("All results saved successfully.")


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


    def perform_shap_analysis(self, save_dir: Optional[str] = None):
        """
        Performs SHAP analysis for feature importance.
        - Generates beeswarm plots for multiclass classification (per class and aggregated).
        - Generates bar charts for per-class and global (aggregated) feature importance.
        - Optionally saves the plots to the specified directory.
        
        Args:
            save_dir (Optional[str]): Directory to save SHAP plots. Defaults to None.
        """
        LOGGER.info("Performing SHAP analysis...")
        
        # Transform the test set using the preprocessing pipeline
        self.X_test_transformed = self.preprocessor.transform(self.X_test)
        
        # Create and compute SHAP explainer
        if not self.shap_values:
            explainer = shap.TreeExplainer(self.model)
            self.shap_values = explainer(self.X_test_transformed, check_additivity=False)
        
        # Feature names
        feature_names = self.X_test_transformed.columns
        
        if len(self.shap_values.values.shape) == 3:  # Multiclass classification
            num_classes = self.shap_values.values.shape[2]
            LOGGER.info(f"Multiclass classification detected with {num_classes} classes.")
            
            # Per-class feature importance
            for class_idx in range(num_classes):
                LOGGER.info(f"Generating SHAP feature importance and beeswarm plot for Class {class_idx}")
                
                # Extract SHAP values for the current class
                class_shap_values = self.shap_values.values[:, :, class_idx]
                
                # Beeswarm plot for the class
                plt.figure(figsize=(10, 8))
                shap.summary_plot(class_shap_values, self.X_test_transformed, show=False)
                plt.title(f"SHAP Beeswarm Plot for Class {class_idx}")
                
                if save_dir:
                    beeswarm_file = os.path.join(save_dir, f"shap_beeswarm_class_{class_idx}.png")
                    plt.savefig(beeswarm_file, bbox_inches="tight")
                    LOGGER.info(f"SHAP beeswarm plot for Class {class_idx} saved to {beeswarm_file}")
                    plt.close()
                else:
                    plt.show()
                
                # Calculate mean absolute SHAP values for this class
                shap_values_abs_mean_class = np.abs(class_shap_values).mean(axis=0)
                
                # Create feature importance DataFrame
                feature_importance_df_class = pd.DataFrame({
                    "Feature": feature_names,
                    "Mean SHAP Value": shap_values_abs_mean_class
                }).sort_values(by="Mean SHAP Value", ascending=False).head(15)
                
                # Plot per-class feature importance
                plt.figure(figsize=(10, 8))
                sns.barplot(x="Mean SHAP Value", y="Feature", data=feature_importance_df_class, palette="viridis")
                plt.title(f"Feature Importance for Class {class_idx}")
                plt.xlabel("Mean Absolute SHAP Value")
                plt.ylabel("Feature")
                plt.tight_layout()
                
                if save_dir:
                    importance_file = os.path.join(save_dir, f"shap_feature_importance_class_{class_idx}.png")
                    plt.savefig(importance_file, bbox_inches="tight")
                    LOGGER.info(f"Feature importance bar plot for Class {class_idx} saved to {importance_file}")
                    plt.close()
                else:
                    plt.show()
            
            # Global (aggregated) feature importance
            LOGGER.info("Generating global feature importance (aggregated across classes).")
            shap_values_abs_mean_global = np.abs(self.shap_values.values).mean(axis=(0, 2))  # Mean across samples and classes
            
            # Beeswarm plot for global SHAP values
            plt.figure(figsize=(10, 8))
            shap.summary_plot(self.shap_values.values.mean(axis=2), self.X_test_transformed, show=False)
            plt.title("SHAP Beeswarm Plot (Aggregated Across Classes)")
            
            if save_dir:
                beeswarm_file = os.path.join(save_dir, "shap_beeswarm_global.png")
                plt.savefig(beeswarm_file, bbox_inches="tight")
                LOGGER.info(f"Global beeswarm plot saved to {beeswarm_file}")
                plt.close()
            else:
                plt.show()
            
            # Create feature importance DataFrame
            feature_importance_df_global = pd.DataFrame({
                "Feature": feature_names,
                "Mean SHAP Value": shap_values_abs_mean_global
            }).sort_values(by="Mean SHAP Value", ascending=False).head(15)
            
            # Plot global feature importance
            plt.figure(figsize=(10, 8))
            sns.barplot(x="Mean SHAP Value", y="Feature", data=feature_importance_df_global, palette="viridis")
            plt.title("Global Feature Importance (Aggregated Across Classes)")
            plt.xlabel("Mean Absolute SHAP Value")
            plt.ylabel("Feature")
            plt.tight_layout()
            
            if save_dir:
                importance_file = os.path.join(save_dir, "shap_feature_importance_global.png")
                plt.savefig(importance_file, bbox_inches="tight")
                LOGGER.info(f"Global feature importance bar plot saved to {importance_file}")
                plt.close()
            else:
                plt.show()
        
        else:  # Binary classification or regression
            LOGGER.info("Binary classification or regression detected.")
            
            # Beeswarm plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(self.shap_values.values, self.X_test_transformed, show=False)
            plt.title("SHAP Beeswarm Plot")
            
            if save_dir:
                beeswarm_file = os.path.join(save_dir, "shap_beeswarm.png")
                plt.savefig(beeswarm_file, bbox_inches="tight")
                LOGGER.info(f"SHAP beeswarm plot saved to {beeswarm_file}")
                plt.close()
            else:
                plt.show()
            
            # Global feature importance
            LOGGER.info("Generating feature importance bar plot...")
            shap_values_abs_mean = np.abs(self.shap_values.values).mean(axis=0)
            
            feature_importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Mean SHAP Value": shap_values_abs_mean
            }).sort_values(by="Mean SHAP Value", ascending=False).head(15)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(x="Mean SHAP Value", y="Feature", data=feature_importance_df, palette="viridis")
            plt.title("Global Feature Importance")
            plt.xlabel("Mean Absolute SHAP Value")
            plt.ylabel("Feature")
            plt.tight_layout()
            
            if save_dir:
                importance_file = os.path.join(save_dir, "shap_feature_importance_global.png")
                plt.savefig(importance_file, bbox_inches="tight")
                LOGGER.info(f"Global feature importance bar plot saved to {importance_file}")
                plt.close()
            else:
                plt.show()
    




class OptimalClassificationPipeline(BasePipeline):
    """Pipeline for classification models using OptimalCatBoostClassifier with SHAP analysis."""

    def __init__(
        self, 
        df: pd.DataFrame, 
        target_column: str, 
        num_features: List[str], 
        cat_features: List[str], 
        param_grid: Dict, 
        class_weights: Dict[Any, str] = None,
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
            n_trials (int): Number of trials for Optuna optimization. Defaults to 10.
            cache_path (Optional[str]): Path to cache Optuna study results.
        """
        super().__init__(df, target_column, num_features, cat_features)
        self.model_type = "classification"
        self.param_grid = param_grid
        self.n_trials = n_trials
        self.cache_path = cache_path
        self.class_weights = class_weights
        self.pipeline = self.get_pipeline()

    def get_pipeline(self) -> Pipeline:
        """Sets up the classification pipeline with preprocessing and OptimalCatBoostClassifier.

        Returns:
            Pipeline: Classification pipeline.
        """
        preprocessor = self.get_feature_preprocessing_pipeline(encode_categorical=True)

        model = OptimalCatBoostClassifier(
            # features=self.num_features + self.cat_features,
            param_grid=self.param_grid,
            n_trials=self.n_trials,
            cat_features=self.cat_features,
            cache_path=self.cache_path,
            study_name=f"catboost_{self.target_column}",
            class_weights=self.class_weights
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
    

    @property
    def training_results(self) -> pd.DataFrame:
        """Evaluates the classification model and generates a metrics DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing classification metrics (accuracy, precision, recall, F1 score).
        """
        return self.pipeline.named_steps['model'].training_results
        

    def evaluate_model(self, save_dir: Optional[str] = None, X_eval: Optional[pd.DataFrame] = None, y_eval: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Evaluates the classification model and generates a metrics DataFrame.
        
        Args:
            save_dir (Optional[str]): Directory to save the classification report and heatmap. Defaults to None.
            X_eval (Optional[pd.DataFrame]): Optional custom features for evaluation. Defaults to None.
            y_eval (Optional[pd.Series]): Optional custom target for evaluation. Defaults to None.
        
        Returns:
            pd.DataFrame: DataFrame containing classification metrics (accuracy, precision, recall, F1 score).
        """
        LOGGER.info("Evaluating the model...")

        # Use the provided evaluation set, or fallback to the default test set
        X_eval = X_eval if X_eval is not None else self.X_test
        y_eval = y_eval if y_eval is not None else self.y_test

        y_pred = self.pipeline.predict(X_eval)
        
        # Generate classification report as a dictionary
        report_dict = classification_report(y_eval, y_pred, output_dict=True)
        
        # Convert to DataFrame
        report_df = pd.DataFrame(report_dict).transpose()

        class_metrics = report_df.drop(index=["macro avg", "weighted avg", "accuracy"], errors="ignore")

        # Create a heatmap for the class metrics
        plt.figure(figsize=(10, 6))
        sns.heatmap(class_metrics.iloc[:, :-1], annot=True, cmap="YlGnBu", fmt=".2f")  # Exclude 'support'
        plt.title("Classification Report Heatmap (Per Class Metrics)")
        plt.tight_layout()
        
        if save_dir:
            # Save the heatmap
            heatmap_file = os.path.join(save_dir, "classification_report_heatmap.png")
            plt.savefig(heatmap_file, bbox_inches="tight")
            LOGGER.info(f"Classification report heatmap saved to {heatmap_file}")
            plt.close()
            report_csv_file = os.path.join(save_dir, "classification_report.csv")
            report_df.to_csv(report_csv_file, index=True)
            LOGGER.info(f"Classification report saved to {report_csv_file}")
        else:
            plt.show()
    
        return report_df


    @property
    def model(self):
        return self.pipeline.named_steps['model']
        
    @property
    def preprocessor(self):
        return self.pipeline.named_steps['preprocessor']



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
        preprocessor = self.get_feature_preprocessing_pipeline(encode_categorical=True)
    
        target_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        model = OptimalCatBoostRegressor(
            # features=self.num_features + self.cat_features,
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

    def evaluate_model(self, save_dir: Optional[str] = None, X_eval: Optional[pd.DataFrame] = None, y_eval: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Evaluates the regression model and generates a metrics DataFrame.
        Includes residual and predicted vs actual plots.

        Args:
            save_dir (Optional[str]): Directory to save evaluation metrics and plots. Defaults to None.
            X_eval (Optional[pd.DataFrame]): Optional custom features for evaluation. Defaults to None.
            y_eval (Optional[pd.Series]): Optional custom target for evaluation. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing regression metrics (MAE, MSE, RMSE, R2 Score).
        """
        LOGGER.info("Evaluating the regression model...")

        # Use the provided evaluation set, or fallback to the default test set
        X_eval = X_eval if X_eval is not None else self.X_test
        y_eval = y_eval if y_eval is not None else self.y_test

        y_pred = self.pipeline.predict(X_eval)

        # Compute metrics
        evaluation_results = {
            "Metric": ["Mean Absolute Error", "Mean Squared Error", "Root Mean Squared Error", "R2 Score"],
            "Score": [
                mean_absolute_error(y_eval, y_pred),
                mean_squared_error(y_eval, y_pred),
                mean_squared_error(y_eval, y_pred, squared=False),  # RMSE
                r2_score(y_eval, y_pred),
            ],
        }
        metrics_df = pd.DataFrame(evaluation_results)

        # Create residual plot
        residuals = y_eval - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, bins=30, color="blue")
        plt.title("Residual Distribution")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.tight_layout()

        if save_dir:
            residual_plot_file = os.path.join(save_dir, "residual_distribution.png")
            plt.savefig(residual_plot_file, bbox_inches="tight")
            LOGGER.info(f"Residual distribution plot saved to {residual_plot_file}")
        if not save_dir:
            plt.show()
        else:
            plt.close()

        # Create predicted vs actual plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_eval, y=y_pred, alpha=0.6, edgecolor=None)
        plt.plot([y_eval.min(), y_eval.max()], [y_eval.min(), y_eval.max()], color="red", linestyle="--")
        plt.title("Predicted vs Actual Values")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.tight_layout()

        if save_dir:
            pred_vs_actual_file = os.path.join(save_dir, "predicted_vs_actual.png")
            plt.savefig(pred_vs_actual_file, bbox_inches="tight")
            LOGGER.info(f"Predicted vs actual plot saved to {pred_vs_actual_file}")
        if not save_dir:
            plt.show()
        else:
            plt.close()

        # Save metrics as a CSV
        if save_dir:
            metrics_file = os.path.join(save_dir, "regression_metrics.csv")
            metrics_df.to_csv(metrics_file, index=False)
            LOGGER.info(f"Regression metrics saved to {metrics_file}")

        return metrics_df
        
    @property
    def model(self):
        try:
            check_is_fitted(self.pipeline)
            return self.pipeline.regressor_['model']
        except NotFittedError as exc:
            pass
        return self.pipeline.regressor['model']
        
    @property
    def preprocessor(self):
        try:
            check_is_fitted(self.pipeline)
            return self.pipeline.regressor_['preprocessor']
        except NotFittedError as exc:
            pass
        return self.pipeline.regressor['preprocessor']
