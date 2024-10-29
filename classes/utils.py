import pandas as pd
import os
from typing import Callable


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
