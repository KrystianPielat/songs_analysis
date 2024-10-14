import os
import numpy as np
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
import logging
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE


LOGGER = logging.getLogger(__name__)

class BasePipeline:
    def __init__(self, df, target_column, num_features, cat_features):
        self.df = df
        self.target_column = target_column
        self.num_features = num_features
        self.cat_features = cat_features
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.shap_values = None
        self.X_test_transformed = None  # To store transformed test data
        LOGGER.info(f"Initializing pipeline for target variable: {target_column}")

    def get_feature_preprocessing_pipeline(self):
        LOGGER.info("Setting up the pipeline...")
        
        # Preprocessing for numerical and categorical data
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Combine both numeric and categorical preprocessors
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.num_features),
                ('cat', categorical_transformer, self.cat_features)
            ]
        )

        return preprocessor

    def split(self):
        LOGGER.info("Splitting the data into training and testing sets...")
        X = self.df[self.num_features + self.cat_features]
        y = self.df[self.target_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        LOGGER.info("Data splitting completed.")

    def train_model(self):
        LOGGER.info("Training the model...")
        self.pipeline.fit(self.X_train, self.y_train)
        LOGGER.info("Model training completed.")

    def save_results(self, save_dir='results'):
        """Saves SHAP plots and evaluation table to a specified directory, creating directories if necessary."""
        os.makedirs(save_dir, exist_ok=True)
        LOGGER.info(f"Saving results to {save_dir}...")

        evaluation_df = self.evaluate_model()
        evaluation_file = os.path.join(save_dir, 'evaluation_results.csv')
        evaluation_df.to_csv(evaluation_file, index=False)
        LOGGER.info(f"Evaluation results saved to {evaluation_file}")

        if self.shap_values is not None and self.X_test_transformed is not None:
            plt.figure()
            shap.summary_plot(self.shap_values, self.X_test_transformed, plot_type="bar", show=False)
            summary_plot_file = os.path.join(save_dir, 'shap_summary_plot.png')
            plt.savefig(summary_plot_file, bbox_inches='tight')
            plt.close()
            LOGGER.info(f"SHAP summary plot saved to {summary_plot_file}")

            plt.figure(figsize=(10, 8))
            shap.plots.beeswarm(self.shap_values, show=False)
            beeswarm_plot_file = os.path.join(save_dir, 'shap_beeswarm_plot.png')
            plt.savefig(beeswarm_plot_file, bbox_inches='tight')
            plt.close()
            LOGGER.info(f"SHAP beeswarm plot saved to {beeswarm_plot_file}")

    def run_pipeline(self, save_dir='dashboard/results'):
        with tqdm(total=5, desc="Pipeline Progress", unit="step") as pbar:
            LOGGER.info("Starting pipeline execution...")

            pbar.set_description("Splitting data")
            self.split()
            pbar.update(1)

            pbar.set_description("Training model")
            self.train_model()
            pbar.update(1)

            pbar.set_description("Evaluating model")
            self.evaluate_model()
            pbar.update(1)

            pbar.set_description("Performing SHAP analysis")
            self.perform_shap_analysis()
            pbar.update(1)

            pbar.set_description("Saving results")
            self.save_results(save_dir)
            pbar.update(1)

            LOGGER.info("Pipeline execution completed.")

class ClassificationPipeline(BasePipeline):
    def __init__(self, df, target_column, num_features, cat_features, handle_imbalance=True):
        super().__init__(df, target_column, num_features, cat_features)
        self.model_type = 'classification'
        self.handle_imbalance = handle_imbalance
        self.model = RandomForestClassifier(random_state=42, class_weight='balanced')  # Class weight balancing
        self.pipeline = self.get_classification_pipeline()

    def get_classification_pipeline(self):
        preprocessor = self.get_feature_preprocessing_pipeline()
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', self.model)
        ])


    def evaluate_model(self):
        LOGGER.info("Evaluating classification model...")
        y_pred = self.pipeline.predict(self.X_test)

        # Compute classification metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        LOGGER.info("Classification evaluation completed.")
    
        # Create a metrics DataFrame
        metrics = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Score': [accuracy, precision, recall, f1]
        }
        
        df_metrics = pd.DataFrame(metrics)
        df_conf_matrix = pd.DataFrame(conf_matrix)
    
        return df_metrics
    
    def train_model(self):
        # Apply the preprocessing step on the DataFrame (ensure X_train is a DataFrame)
        preprocessor_step = self.pipeline.named_steps['preprocessor']
        self.X_train = pd.DataFrame(self.X_train, columns=self.num_features + self.cat_features)  # Ensure it's a DataFrame
        self.X_train = preprocessor_step.fit_transform(self.X_train)
        
        if self.handle_imbalance:
            LOGGER.info("Fixing class balance...")
            # Adjust the number of neighbors for SMOTE dynamically based on the class size
            min_class_count = self.y_train.value_counts().min()
            n_neighbors = min(5, min_class_count - 1)  # Ensure k_neighbors is less than the size of the smallest class
    
            if n_neighbors < 1:
                LOGGER.warning("Cannot apply SMOTE as the minority class has only one sample. Skipping SMOTE.")
            else:
                smote = SMOTE(random_state=42, k_neighbors=n_neighbors)
                self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
                LOGGER.info(f"Applied SMOTE with k_neighbors={n_neighbors} to fix class imbalance.")

        # Train the model with the resampled data
        self.pipeline.named_steps['model'].fit(self.X_train, self.y_train)
        LOGGER.info("Model training completed.")
        
    def perform_shap_analysis(self):
        LOGGER.info("Performing SHAP analysis for classification...")
        model_pipeline = self.pipeline
        
        # Transform the test set using the preprocessing pipeline
        self.X_test_transformed = model_pipeline.named_steps['preprocessor'].transform(self.X_test)
        
        numeric_features = self.num_features
        categorical_features = list(model_pipeline.named_steps['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out(self.cat_features))
        selected_feature_names = numeric_features + categorical_features
        
        explainer = shap.Explainer(model_pipeline.named_steps['model'], self.X_test_transformed, feature_names=selected_feature_names)
        shap_values = explainer(self.X_test_transformed, check_additivity=False)
    
        # For classification, shap_values is a list of arrays (one per class), select class 1
        self.shap_values = shap_values[..., 1]  # Select class 1 SHAP values
        LOGGER.info("SHAP analysis for classification completed.")


class RegressionPipeline(BasePipeline):
    def __init__(self, df, target_column, num_features, cat_features):
        super().__init__(df, target_column, num_features, cat_features)
        self.model_type = 'regression'
        self.model = RandomForestRegressor(random_state=42)
        self.pipeline = self.get_regression_pipeline()

    def get_regression_pipeline(self):
        preprocessor = self.get_feature_preprocessing_pipeline()
        target_transformer = Pipeline(steps=[
            # ('log_transformer', FunctionTransformer(np.log1p, inverse_func=np.expm1, check_inverse=False)),
            ('scaler', StandardScaler())
        ])
        return TransformedTargetRegressor(
            regressor=Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', self.model)
            ]), transformer=target_transformer
        )

    def evaluate_model(self):
        LOGGER.info("Evaluating regression model...")
        y_pred = self.pipeline.predict(self.X_test)
        
        # Calculate regression metrics
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = mean_squared_error(self.y_test, y_pred, squared=False)
        r2 = r2_score(self.y_test, y_pred)
        
        # Create a DataFrame with rounded metrics
        results = pd.DataFrame({
            'Metric': ['Mean Absolute Error (MAE)', 'Mean Squared Error (MSE)', 'Root Mean Squared Error (RMSE)', 'R-squared'],
            'Value': [round(mae, 3), round(mse, 3), round(rmse, 3), round(r2, 3)]
        })
    
        LOGGER.info("Regression evaluation completed.")
        return results

    def perform_shap_analysis(self):
        LOGGER.info("Performing SHAP analysis for regression...")
        model_pipeline = self.pipeline.regressor_
        
        # Transform the test set using the preprocessing pipeline
        self.X_test_transformed = model_pipeline.named_steps['preprocessor'].transform(self.X_test)
        
        numeric_features = self.num_features
        categorical_features = list(model_pipeline.named_steps['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out(self.cat_features))
        selected_feature_names = numeric_features + categorical_features
        
        explainer = shap.Explainer(model_pipeline.named_steps['model'], self.X_test_transformed, feature_names=selected_feature_names)
        self.shap_values = explainer(self.X_test_transformed, check_additivity=False)
        LOGGER.info("SHAP analysis for regression completed.")