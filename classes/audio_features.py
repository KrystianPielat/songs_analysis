import warnings
import librosa
import numpy as np
import pandas as pd
import logging
from tqdm.auto import tqdm
from classes.feature_extractor import FeatureExtractor
from multiprocessing import Pool, cpu_count
import gc
import psutil

LOGGER = logging.getLogger(__name__)

class AudioFeatureExtractor:
    """Class to handle the extraction of audio features from MP3 files."""

    @property
    def output_columns(self) -> list:
        """Defines the list of output feature columns."""
        return [
            f'mfcc_{i+1}' for i in range(13)
        ] + [
            f'chroma_{i+1}' for i in range(12)
        ] + [
            f'spectral_contrast_{i+1}' for i in range(7)
        ] + [
            'tempo_extracted', 'zcr'
        ]

    @staticmethod
    def extract_features(audio_path: str) -> np.ndarray:
        """Static method to extract audio features for multiprocessing."""
        try:
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
            return np.hstack((mfccs, chroma, spectral_contrast, tempo, zcr))
        except Exception as e:
            LOGGER.error(f"Error processing file {audio_path}: {e}")
            return [None] * 34  # Placeholder for failed extractions

    @staticmethod
    def _process_batch_static(args):
        """Static method to process a batch of rows for multiprocessing."""
        batch, audio_column = args
        results = []
        for _, row in batch.iterrows():
            audio_path = row[audio_column]
            features = AudioFeatureExtractor.extract_features(audio_path)
            results.append(features)
        return results

    def add_features(self, df: pd.DataFrame, audio_column: str = 'mp3_path', batch_size: int = None) -> pd.DataFrame:
        """Adds audio features to the given DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing audio file paths.
            audio_column (str): Name of the column containing the path to audio files.
            batch_size (int, optional): Size of batches for parallel processing.

        Returns:
            pd.DataFrame: The original DataFrame with added audio features.
        """
        if audio_column not in df.columns:
            raise ValueError(f"Column '{audio_column}' not found in DataFrame.")

        feature_columns = self.output_columns
        missing_columns = [col for col in feature_columns if col not in df.columns]

        if not missing_columns:
            LOGGER.info("All audio features already present.")
            return df

        all_features = []

        # Process in parallel if batch_size is provided
        if batch_size:
            batches = [(df.iloc[i:i + batch_size], audio_column) for i in range(0, len(df), batch_size)]
            with Pool(max(cpu_count() // 2, 1)) as pool:
                with tqdm(total=len(df), desc="Extracting Audio Features", unit="file") as pbar:
                    for batch_result in pool.imap(AudioFeatureExtractor._process_batch_static, batches):
                        all_features.extend(batch_result)
                        pbar.update(len(batch_result))
                        gc.collect()
                        memory_info = psutil.virtual_memory()
                        LOGGER.info(f"Memory Usage: {memory_info.percent}%")

        else:
            # Process sequentially
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Audio Features", unit="file"):
                audio_path = row[audio_column]
                features = self.extract_features(audio_path)
                all_features.append(features)

        # Create a DataFrame with the extracted features
        features_df = pd.DataFrame(all_features, columns=feature_columns)

        # Add missing columns only
        df_with_features = pd.concat([df.reset_index(drop=True), features_df[missing_columns]], axis=1)
        return df_with_features
