import warnings
import librosa
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from classes.feature_extractor import FeatureExtractor


class AudioFeatureExtractor(FeatureExtractor):
    """Class to handle the extraction of audio features from MP3 files."""

    def extract_features(self, audio_path):
        """Extracts audio features from a given audio file."""
        y, sr = librosa.load(audio_path, sr=None)

        # Extract features
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)

        # Suppress FutureWarning for librosa.beat.tempo
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            tempo = librosa.beat.tempo(y=y, sr=sr)[0]

        zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)

        # Combine features into a single array
        features = np.hstack((mfccs, chroma, spectral_contrast, tempo, zcr))
        return features

    def add_features(self, df, audio_column='mp3_path'):
        """
        Adds audio features to the given DataFrame.
    
        Parameters:
        - df: DataFrame containing audio file paths.
        - audio_column: Name of the column containing the path to audio files.
    
        Returns:
        - DataFrame: The original DataFrame with added audio features.
        """
        # Ensure the audio_column exists in the DataFrame
        if audio_column not in df.columns:
            raise ValueError(f"Column '{audio_column}' not found in DataFrame.")
    
        # Define feature columns
        feature_columns = [
            f'mfcc_{i+1}' for i in range(13)] + \
            [f'chroma_{i+1}' for i in range(12)] + \
            [f'spectral_contrast_{i+1}' for i in range(7)] + \
            ['tempo_extracted', 'zcr']
    
        # Identify missing columns
        missing_columns = [col for col in feature_columns if col not in df.columns]
    
        # If all features are already present, return the original DataFrame
        if not missing_columns:
            print("All audio features already present.")
            return df
    
        all_features = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Audio Features", unit="file"):
            audio_path = row[audio_column]
            try:
                features = self.extract_features(audio_path)
                all_features.append(features)
            except Exception as e:
                print(f"Error processing file {audio_path}: {e}")
                all_features.append([None] * 34)  # Add placeholder for failed extractions
    
        # Create DataFrame for missing features
        features_df = pd.DataFrame(all_features, columns=feature_columns)
    
        # Merge the features into the original DataFrame
        df_with_features = pd.concat([df.reset_index(drop=True), features_df[missing_columns]], axis=1)
        return df_with_features