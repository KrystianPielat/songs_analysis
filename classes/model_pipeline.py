import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pandas.api.types import is_numeric_dtype


# NLP and Text Processing
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import cmudict
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Scikit-learn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
from typing import Literal

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap
from sklearn.inspection import permutation_importance

from tqdm.auto import tqdm

import logging

LOGGER = logging.getLogger(__name__)



class ModelPipeline:
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

        self.model_type = 'regression' if is_numeric_dtype(df[target_column]) else 'classification'
        self.model = RandomForestClassifier(random_state=42) if self.model_type == 'classification' else RandomForestRegressor(random_state=42)
        self.pipeline = self.get_pipeline()

        LOGGER.info(f"ModelPipeline initialized for {self.model_type} task.")

    def get_pipeline(self):
        LOGGER.info("Setting up the pipeline...")

        # Preprocessing for numerical data
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Preprocessing pipeline combining both numeric and categorical
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.num_features),
                ('cat', categorical_transformer, self.cat_features)
            ]
        )

        # Model pipeline with preprocessing
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', self.model)
        ])

        LOGGER.info("Pipeline setup completed.")

        # If regression, use TransformedTargetRegressor for target transformation
        if self.model_type == 'regression':
            target_transformer = Pipeline(steps=[
                ('log_transformer', FunctionTransformer(np.log1p, inverse_func=np.expm1, check_inverse=False)),
                ('scaler', StandardScaler())
            ])
            LOGGER.info("TransformedTargetRegressor applied for regression task.")
            return TransformedTargetRegressor(regressor=model_pipeline, transformer=target_transformer)
        
        # If classification, no target transformation is needed
        return model_pipeline

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

    @staticmethod
    def evaluate_regression_model(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        
        results = {
            'Metric': ['Mean Absolute Error', 'Mean Squared Error', 'Root Mean Squared Error', 'R-squared'],
            'Value': [mae, mse, rmse, r2]
        }

        LOGGER.info("Regression evaluation completed.")
        return pd.DataFrame(results)

    @staticmethod
    def evaluate_classification_model(y_true, y_pred):
        """Evaluates the classification model and returns a dataframe with metrics"""
        accuracy = accuracy_score(y_true, y_pred)
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        report_df['accuracy'] = accuracy

        LOGGER.info("Classification evaluation completed.")
        return report_df

    def evaluate_model(self):
        LOGGER.info("Evaluating the model...")
        if self.model_type == 'regression':
            return ModelPipeline.evaluate_regression_model(self.y_test, self.pipeline.predict(self.X_test))
        return ModelPipeline.evaluate_classification_model(self.y_test, self.pipeline.predict(self.X_test))

    def perform_shap_analysis(self):
        LOGGER.info("Performing SHAP analysis...")
        if self.model_type == 'regression':
            model_pipeline = self.pipeline.regressor_
        else:
            model_pipeline = self.pipeline
    
        self.X_test_transformed = model_pipeline.named_steps['preprocessor'].transform(self.X_test)
        
        numeric_features = self.num_features
        categorical_features = list(model_pipeline.named_steps['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out(self.cat_features))
        selected_feature_names = numeric_features + categorical_features
        
        explainer = shap.Explainer(model_pipeline.named_steps['model'], self.X_test_transformed, feature_names=selected_feature_names)
        self.shap_values = explainer(self.X_test_transformed, check_additivity=False)

        LOGGER.info("SHAP analysis completed.")

    def show_summary(self):
        LOGGER.info("Displaying model evaluation and SHAP summary...")
        display(self.evaluate_model())
        
        if self.shap_values is not None and self.X_test_transformed is not None:
            shap.summary_plot(self.shap_values, self.X_test_transformed, plot_type="bar")
            shap.plots.beeswarm(self.shap_values, max_display=15)

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
        with tqdm(total=6, desc="Pipeline Progress", unit="step") as pbar:
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

            pbar.set_description("Showing summary")
            self.show_summary()
            pbar.update(1)

            pbar.set_description("Saving results")
            self.save_results(save_dir)
            pbar.update(1)

            LOGGER.info("Pipeline execution completed.")
