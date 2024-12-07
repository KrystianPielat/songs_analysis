import pandas as pd
import os
from typing import Callable, List
from mutagen.mp3 import MP3
from mutagen import MutagenError



def gather_data_from_folders(playlists_dir: str) -> pd.DataFrame:
    """Concatenates all CSV files from subfolders of a specified directory into one DataFrame.

    Args:
        playlists_dir (str): Directory containing subfolders with CSV files.

    Returns:
        pd.DataFrame: Combined DataFrame from all CSV files in subdirectories.
    """
    all_dataframes = []

    # Iterate over all subdirectories in the playlists directory
    for root, dirs, files in os.walk(playlists_dir):
        for file in files:
            if file.endswith('.csv') and '.ipynb' not in root and not file.startswith("_"):
                csv_path = os.path.join(root, file)
                
                print(f"Loading CSV file: {csv_path}")  # Debugging statement to see the file being loaded
                df = pd.read_csv(csv_path, index_col=None)

                # Add 'csv_path' column if it doesn't exist
                if 'csv_path' not in df.columns:
                    print(f"'csv_path' column missing in {file}. Adding the column...")
                    df['csv_path'] = csv_path

                    # Save the updated DataFrame back to the CSV
                    df.to_csv(csv_path, index=False)
                    print(f"'csv_path' column added and saved to {csv_path}")

                # Check if the 'Unnamed: 0' column exists, if so, print the issue and drop it
                if 'Unnamed: 0' in df.columns:
                    print(f"Found 'Unnamed: 0' in {file}, dropping the column")
                    df = df.drop(columns=['Unnamed: 0'])

                all_dataframes.append(df)

    # Concatenate all DataFrames
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    return combined_df


def apply_function_to_csvs(playlists_dir: str, modify_function: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
    """
    Applies a specified function to all CSVs in subfolders of a given directory, modifies the DataFrames,
    and saves them back to their respective files.

    Args:
        playlists_dir (str): Directory containing subfolders with CSV files.
        modify_function (Callable[[pd.DataFrame], pd.DataFrame]): Function that takes a DataFrame as input and returns a modified DataFrame.
    """
    # Iterate over all subdirectories in the playlists directory
    for root, dirs, files in os.walk(playlists_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(root, file)
                
                # Load the CSV into a DataFrame
                df = pd.read_csv(csv_path)
                
                # Apply the given function to the DataFrame
                modified_df = modify_function(df)
                
                # Save the modified DataFrame back to the same CSV file
                modified_df.to_csv(csv_path, index=False)
                print(f"Modified and saved CSV at: {csv_path}")

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
    """
    Verifies if the MP3 file at the given path is not corrupted.

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
    """
    Identifies songs to drop based on missing values in critical columns.
    invalid 'mp3_path', or lyrics shorter than 180 characters.

    Args:
        df_to_process (pd.DataFrame): DataFrame containing song metadata.

    Returns:
        pd.DataFrame: Rows that are identified as invalid or incomplete.
    """
    # Identify columns to check for NaN values, excluding 'popularity'
    columns_to_check = [c for c in df_to_process.columns if c not in allow_nan_cols]

    # Find rows with missing values in critical columns
    rows_with_nan = df_to_process.index.difference(
        df_to_process.dropna(subset=columns_to_check).index
    )

    # Find rows with invalid 'mp3_path'
    invalid_mp3_rows = df_to_process.index[
        ~df_to_process['mp3_path'].apply(lambda x: is_mp3_valid(x) if pd.notna(x) else False)
    ]

    # Find rows with lyrics shorter than 180 characters
    short_lyrics_rows = df_to_process.index[
        df_to_process['lyrics'].apply(lambda x: len(x) < 180 if pd.notna(x) else True)
    ]

    # Combine all conditions
    rows_to_drop = rows_with_nan.union(invalid_mp3_rows).union(short_lyrics_rows)

    # Return the rows to drop
    return df_to_process.loc[rows_to_drop]



def remove_mp3_files(mp3_paths: pd.Series) -> None:
    """
    Removes MP3 files specified in a pandas Series of file paths.

    Args:
        mp3_paths (pd.Series): A pandas Series containing file paths to MP3 files.

    Returns:
        None
    """
    for path in mp3_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                print(f"Removed: {path}")
            else:
                print(f"File does not exist: {path}")
        except Exception as e:
            print(f"Error while removing file {path}: {e}")


def clean_songs_to_drop(songs_to_drop: pd.DataFrame) -> None:
    """
    Removes MP3 files and corresponding rows from CSV files.

    Args:
        songs_to_drop (pd.DataFrame): DataFrame containing `mp3_path` and `csv_path` columns.

    Returns:
        None
    """
    # Ensure required columns exist
    if not {'mp3_path', 'csv_path', 'id'}.issubset(songs_to_drop.columns):
        raise ValueError("The DataFrame must contain 'mp3_path' and 'csv_path' columns.")

    for _, row in songs_to_drop.iterrows():
        # Remove MP3 file if it exists
        mp3_path = row['mp3_path']
        if pd.notna(mp3_path) and os.path.exists(mp3_path):
            try:
                os.remove(mp3_path)
                print(f"Removed MP3 file: {mp3_path}")
            except Exception as e:
                print(f"Error removing MP3 file {mp3_path}: {e}")

        # Remove row from corresponding CSV
        csv_path = row['csv_path']
        if pd.notna(csv_path) and os.path.exists(csv_path):
            try:
                # Load CSV
                df = pd.read_csv(csv_path)

                # Identify the row to remove (using `id`, `title`, or other unique identifiers if available)
                row_conditions = (df['id'] == row['id'])

                # Drop the row and save back to the CSV
                updated_df = df[~row_conditions]
                updated_df.to_csv(csv_path, index=False)
                print(f"Updated CSV file: {csv_path}")
            except Exception as e:
                print(f"Error updating CSV file {csv_path}: {e}")
