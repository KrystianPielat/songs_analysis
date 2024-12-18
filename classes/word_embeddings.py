import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
from tqdm.auto import tqdm
from typing import Optional
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from classes.feature_extractor import FeatureExtractor
import logging

class Word2VecFeatureExtractor(FeatureExtractor):
    def __init__(self, model: Optional[Word2Vec] = None, n_pca_components: Optional[int] = None) -> None:
        """
        Initializes the Word2Vec Feature Extractor.

        Args:
            model (Optional[Word2Vec]): Pre-trained Word2Vec model. If None, it will train a model.
            n_pca_components (Optional[int]): Number of PCA components for dimensionality reduction.
        """
        self.model = model
        self.n_pca_components = n_pca_components
        self.pca = PCA(n_components=n_pca_components) if n_pca_components else None
        self._logger = logging.getLogger(self.__class__.__name__)

    def train_model(self, data: pd.Series, vector_size: int = 100, window: int = 5, min_count: int = 1) -> None:
        """
        Trains a Word2Vec model on the provided text data.

        Args:
            data (pd.Series): Text data to train on.
            vector_size (int): Size of word vectors.
            window (int): Maximum distance between words for context.
            min_count (int): Ignores words with frequency lower than this.
        """
        tokenized_data = [word_tokenize(text.lower()) for text in data]
        self.model = Word2Vec(sentences=tokenized_data, vector_size=vector_size, window=window, min_count=min_count)
        self._logger.info("Word2Vec model trained.")

    def text_to_vector(self, text: str) -> np.ndarray:
        """
        Converts a text string into a vector by averaging its Word2Vec word embeddings.

        Args:
            text (str): The input text.

        Returns:
            np.ndarray: Averaged Word2Vec vector for the input text.
        """
        tokens = word_tokenize(text.lower())
        vectors = [self.model.wv[word] for word in tokens if word in self.model.wv]
        if not vectors:  # If no words have embeddings, return a zero vector
            return np.zeros(self.model.vector_size)
        return np.mean(vectors, axis=0)

    def extract_features(self, data: pd.Series) -> pd.DataFrame:
        """
        Extracts Word2Vec features from a pandas Series.

        Args:
            data (pd.Series): The text data to process.

        Returns:
            pd.DataFrame: DataFrame containing Word2Vec features.
        """
        if self.model is None:
            raise ValueError("Word2Vec model is not initialized. Train or provide a model first.")

        # Convert each text to its averaged Word2Vec vector
        vectors = np.array([self.text_to_vector(text) for text in tqdm(data, desc="Extracting Word2Vec Features")])

        # Apply PCA if required
        if self.pca:
            vectors = self.pca.fit_transform(vectors)

        # Create a DataFrame with Word2Vec features
        feature_columns = [f"w2v_{i}" for i in range(vectors.shape[1])]
        return pd.DataFrame(vectors, columns=feature_columns)

    def add_features(self, df: pd.DataFrame, text_column: str = 'lyrics', vector_size: int = 100) -> pd.DataFrame:
        """
        Adds Word2Vec features to the given DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing text data.
            text_column (str): Column containing the text data.
            vector_size (int): Size of Word2Vec word vectors if training a new model.

        Returns:
            pd.DataFrame: The original DataFrame with added Word2Vec features.
        """
        if self.model is None:
            self._logger.info("No pre-trained model provided. Training Word2Vec model...")
            self.train_model(df[text_column], vector_size=vector_size)

        # Extract Word2Vec features
        word2vec_features = self.extract_features(df[text_column])

        # Align and combine with the original DataFrame
        df = df.drop([c for c in df.columns if c.startswith('w2v_')], axis=1)
        return pd.concat([df.reset_index(drop=True), word2vec_features], axis=1)



class TfidfFeatureExtractor(FeatureExtractor):
    def __init__(self, n_pca_components: int = 100) -> None:
        """Initializes the TF-IDF Feature Extractor with PCA."""
        self.tfidf = TfidfVectorizer()
        self.pca = PCA(n_components=n_pca_components)

    def extract_features(self, data: pd.Series) -> pd.DataFrame:
        """Extracts TF-IDF features and applies PCA for dimensionality reduction.

        Args:
            data (pd.Series): The text data to process.

        Returns:
            pd.DataFrame: The PCA-reduced TF-IDF features as a DataFrame.
        """
        # TF-IDF Transformation
        tfidf_matrix = self.tfidf.fit_transform(data).toarray()

        # Apply PCA for dimensionality reduction
        tfidf_pca = self.pca.fit_transform(tfidf_matrix)

        # Create a DataFrame for PCA-reduced TF-IDF features
        pca_columns = [f'tfidf_{i}' for i in range(tfidf_pca.shape[1])]
        return pd.DataFrame(tfidf_pca, columns=pca_columns)

    def add_features(self, df: pd.DataFrame, text_column: str = 'lyrics') -> pd.DataFrame:
        """Adds TF-IDF PCA-reduced features to the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing text data.
            text_column (str, optional): Column containing the text data. Defaults to 'text'.

        Returns:
            pd.DataFrame: The original DataFrame with added TF-IDF PCA features.
        """
        # Extract TF-IDF features
        tfidf_features = self.extract_features(df[text_column])

        # Align and overwrite columns in the original DataFrame
        df = df.drop([c for c in df.columns if c.startswith('tfidf_') ], axis=1)
        
        # Combine with the original DataFrame
        return pd.concat([df.reset_index(drop=True), tfidf_features], axis=1)
