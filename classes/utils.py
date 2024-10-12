import pandas as pd
import os


def gather_data_from_folders(playlists_dir: str):
    """Concatenates all CSVs from subfolders of a given playlists directory into one DataFrame."""
    all_dataframes = []

    # Iterate over all subdirectories in the playlists directory
    for root, dirs, files in os.walk(playlists_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_path = os.path.join(root, file)
                df = pd.read_csv(csv_path)
                all_dataframes.append(df)

    # Concatenate all DataFrames
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    return combined_df


def apply_function_to_csvs(playlists_dir, modify_function):
    """
    Applies a function to all CSVs from subfolders of a given playlists directory, 
    modifies the DataFrame, and saves them back in their respective folders.

    Parameters:
    playlists_dir (str): The root directory where the playlists and CSVs are located.
    modify_function (function): A function that takes a DataFrame as input and returns a modified DataFrame.
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

def example_modify_function(df: pd.DataFrame):
    df.loc[df.lyrics == "No lyrics found", 'lyrics'] = None
    return df
