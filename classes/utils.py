import pandas as pd
import os
import numpy as np
import re
import logging
from typing import Callable, List
from mutagen.mp3 import MP3
from mutagen import MutagenError
from classes.constants import GENRE_MAPPING

_LOGGER = logging.getLogger(__name__)

def winsorize_series(series, lower_percentile, upper_percentile):
    """
    Winsorizes a pandas Series by capping values at specified percentiles.
    
    Args:
        series (pd.Series): The input series to be winsorized.
        lower_percentile (float): The lower percentile (e.g., 0.05 for the 5th percentile).
        upper_percentile (float): The upper percentile (e.g., 0.95 for the 95th percentile).
        
    Returns:
        pd.Series: Winsorized series.
    """
    lower_bound = np.percentile(series, lower_percentile * 100)
    upper_bound = np.percentile(series, upper_percentile * 100)
    return series.clip(lower=lower_bound, upper=upper_bound)

def variance_based_empath_cleaning(df, variance_threshold: float = 0.0001):
    empath_df = df[[col for col in df.columns if col.startswith("empath_")]]
    feature_variances = empath_df.var()
    low_variance_features = feature_variances[feature_variances < 0.0001].index
    _LOGGER.info(f"Total empath features: {empath_df.shape[1]}")
    _LOGGER.info(f"Low-variance features to remove: {len(low_variance_features)}")
    return list(empath_df.drop(columns=low_variance_features).columns)

def gather_data_from_folders(playlists_dir: str) -> pd.DataFrame:
    """Concatenates all CSV files from subfolders of a specified directory into one DataFrame.

    Args:
        playlists_dir (str): Directory containing subfolders with CSV files.

    Returns:
        pd.DataFrame: Combined DataFrame from all CSV files in subdirectories.
    """
    all_dataframes = []

    for root, dirs, files in os.walk(playlists_dir):
        for file in files:
            if file.endswith('.csv') and '.ipynb' not in root and not file.startswith("_") and 'faulty' not in file:
                csv_path = os.path.join(root, file)
                _LOGGER.info(f"Loading CSV file: {csv_path}")
                df = pd.read_csv(csv_path, index_col=None)

                if 'csv_path' not in df.columns:
                    _LOGGER.warning(f"'csv_path' column missing in {file}. Adding the column...")
                    df['csv_path'] = csv_path
                    df.to_csv(csv_path, index=False)
                    _LOGGER.info(f"'csv_path' column added and saved to {csv_path}")

                if 'Unnamed: 0' in df.columns:
                    _LOGGER.warning(f"Found 'Unnamed: 0' in {file}, dropping the column.")
                    df = df.drop(columns=['Unnamed: 0'])

                all_dataframes.append(df)

    combined_df = pd.concat(all_dataframes, ignore_index=True)
    return combined_df


def apply_function_to_csvs(playlists_dir: str, modify_function: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
    """Applies a specified function to all CSVs in subfolders of a given directory, modifies the DataFrames,
    and saves them back to their respective files.

    Args:
        playlists_dir (str): Directory containing subfolders with CSV files.
        modify_function (Callable[[pd.DataFrame], pd.DataFrame]): Function that takes a DataFrame as input and returns a modified DataFrame.
    """
    for root, dirs, files in os.walk(playlists_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(root, file)
                df = pd.read_csv(csv_path)
                modified_df = modify_function(df)
                modified_df.to_csv(csv_path, index=False)
                _LOGGER.info(f"Modified and saved CSV at: {csv_path}")


def example_modify_function(df: pd.DataFrame) -> pd.DataFrame:
    """Example function to modify a DataFrame by setting 'No lyrics found' values to None in the lyrics column.

    Args:
        df (pd.DataFrame): DataFrame to modify.

    Returns:
        pd.DataFrame: Modified DataFrame.
    """
    df.loc[df.lyrics == "No lyrics found", 'lyrics'] = None
    return df


def is_mp3_valid(file_path: str) -> bool:
    """Verifies if the MP3 file at the given path is not corrupted.

    Args:
        file_path (str): The path to the MP3 file.

    Returns:
        bool: True if the file is valid, False if it's corrupted or unreadable.
    """
    try:
        if not os.path.exists(file_path):
            return False
        audio = MP3(file_path)
        _ = audio.info.length
        return True
    except (MutagenError, IOError, ValueError):
        return False


def find_songs_to_drop(df_to_process: pd.DataFrame, allow_nan_cols: List[str]) -> pd.DataFrame:
    """Identifies songs to drop based on missing values in critical columns, invalid 'mp3_path', or lyrics shorter than 180 characters.

    Args:
        df_to_process (pd.DataFrame): DataFrame containing song metadata.
        allow_nan_cols (List[str]): List of columns allowed to have NaN values.

    Returns:
        pd.DataFrame: Rows that are identified as invalid or incomplete.
    """
    columns_to_check = [c for c in df_to_process.columns if c not in allow_nan_cols]

    rows_with_nan = df_to_process.index.difference(
        df_to_process.dropna(subset=columns_to_check).index
    )

    invalid_mp3_rows = df_to_process.index[
        ~df_to_process['mp3_path'].apply(lambda x: is_mp3_valid(x) if pd.notna(x) else False)
    ]
    
    df_to_process['genre'] = df_to_process['genres'].apply(lambda x: reduce_genres_with_regex(eval(x), GENRE_MAPPING))
    incorrect_genre_rows = df_to_process[(df_to_process.genre.isna()) | (df_to_process.genre == 'None')].index

    short_lyrics_rows = df_to_process.index[
        df_to_process['lyrics'].apply(lambda x: len(x) < 180 if pd.notna(x) else True)
    ]

    rows_to_drop = rows_with_nan.union(invalid_mp3_rows).union(short_lyrics_rows).union(incorrect_genre_rows)

    for idx in rows_to_drop:
        row = df_to_process.loc[idx]
        reason = []
        if idx in rows_with_nan:
            reason.append("missing critical data")
        if idx in invalid_mp3_rows:
            reason.append("invalid MP3 file")
        if idx in short_lyrics_rows:
            reason.append("short lyrics")
        if idx in incorrect_genre_rows:
            reason.append("incorrect or missing genre")
        _LOGGER.warning(f"Dropping song '{row['title']}' by '{row['artist']}': {', '.join(reason)}.")

    return df_to_process.loc[rows_to_drop]


def remove_mp3_files(mp3_paths: pd.Series) -> None:
    """Removes MP3 files specified in a pandas Series of file paths.

    Args:
        mp3_paths (pd.Series): A pandas Series containing file paths to MP3 files.
    """
    for path in mp3_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                _LOGGER.info(f"Removed MP3 file: {path}")
            else:
                _LOGGER.warning(f"File does not exist: {path}")
        except Exception as e:
            _LOGGER.error(f"Error while removing file {path}: {e}")


def clean_songs_to_drop(songs_to_drop: pd.DataFrame) -> None:
    """Removes MP3 files and corresponding rows from CSV files.

    Args:
        songs_to_drop (pd.DataFrame): DataFrame containing `mp3_path` and `csv_path` columns.
    """
    if not {'mp3_path', 'csv_path', 'id'}.issubset(songs_to_drop.columns):
        raise ValueError("The DataFrame must contain 'mp3_path' and 'csv_path' columns.")

    for _, row in songs_to_drop.iterrows():
        mp3_path = row['mp3_path']
        if pd.notna(mp3_path) and os.path.exists(mp3_path):
            try:
                os.remove(mp3_path)
                _LOGGER.info(f"Removed MP3 file: {mp3_path}")
            except Exception as e:
                _LOGGER.error(f"Error removing MP3 file {mp3_path}: {e}")

        csv_path = row['csv_path']
        if pd.notna(csv_path) and os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                row_conditions = (df['id'] == row['id'])
                updated_df = df[~row_conditions]
                updated_df.to_csv(csv_path, index=False)
                _LOGGER.info(f"Updated CSV file: {csv_path}")
            except Exception as e:
                _LOGGER.error(f"Error updating CSV file {csv_path}: {e}")

def reduce_genres_with_regex(genre_list: list, mapping: dict):
    """
    Dynamically match genres using regex patterns and assign them based on GENRE_MAPPING.
    
    Args:
        genre_list (list): List of genres for a song.
        mapping (dict): Dictionary mapping general genres to specific subgenres.

    Returns:
        str: The generalized genre if found, otherwise None.
    """
    for genre in genre_list:
        for general, specifics in mapping.items():
            # Create a dynamic regex pattern to match subgenres (e.g., 'hip hop' in 'desi hip hop')
            pattern = r'\b(?:' + '|'.join(re.escape(subgenre) for subgenre in specifics) + r')\b'
            if re.search(pattern, genre, re.IGNORECASE):
                return general
    return None