from abc import ABC, abstractmethod


class FeatureExtractor(ABC):
    
    @abstractmethod
    def extract_features(self, data):
        """Extract features from a given data input"""
        pass

    @abstractmethod
    def add_features(self, df, data_column):
        """Add features to the DataFrame based on the data column"""
        pass