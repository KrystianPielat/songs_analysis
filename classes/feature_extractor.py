from abc import ABC, abstractmethod
import pandas as pd
from typing import Any

class FeatureExtractor(ABC):
    """Abstract base class for feature extraction, outlining the methods
    required for extracting and adding features to a DataFrame."""

    @abstractmethod
    def extract_features(self, data: Any) -> Any:
        """Extract features from a given data input.

        Args:
            data (Any): The data input from which to extract features.
        
        Returns:
            Any: Extracted features, with format depending on the implementation.
        """
        pass

    @abstractmethod
    def add_features(self, df: pd.DataFrame, data_column: str) -> pd.DataFrame:
        """Add features to the DataFrame based on the specified data column.

        Args:
            df (pd.DataFrame): DataFrame to which the features will be added.
            data_column (str): Column in the DataFrame containing data for feature extraction.
        
        Returns:
            pd.DataFrame: The DataFrame with added features.
        """
        pass
