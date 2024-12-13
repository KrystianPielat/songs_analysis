import pandas as pd
from typing import Optional
import os
import logging
from multiprocessing import Pool, cpu_count
from tqdm.auto import tqdm
from classes.feature_extractor import FeatureExtractor



class FeatureProcessor:
    def __init__(self, extractor: FeatureExtractor, input_file: str, output_file: str, batch_size: Optional[int] = 100, skip_processed: bool = False):
        """
        Initializes the FeatureProcessor class.

        Args:
            extractor (FeatureExtractor): Feature extractor for processing data.
            input_file (str): Path to the input pickle file.
            output_file (str): Path to the output pickle file.
            batch_size (Optional[int]): Number of records to process per batch. If None, process all in one batch.
            skip_processed (bool): Whether to overwrite existing records.
        """
        self.input_file = input_file
        self.output_file = output_file
        self.batch_size = batch_size
        self.extractor = extractor
        self.skip_processed = skip_processed
        self._logger = logging.getLogger(self.__class__.__name__)

    def load_existing_ids(self) -> set:
        """
        Load existing IDs from the output file to skip already processed songs.

        Returns:
            set: A set of already processed song IDs.
        """
        if os.path.exists(self.output_file):
            existing_df = pd.read_pickle(self.output_file)
            return set(existing_df["id"].tolist())  # Assuming the DataFrame has an 'id' column
        return set()

    def process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """
        Process a batch of songs by adding audio features.

        Args:
            batch (pd.DataFrame): Batch of songs to process.

        Returns:
            pd.DataFrame: Processed batch with audio features.
        """
        return self.extractor.add_features(batch)

    def save_batch_to_output(self, batch: pd.DataFrame) -> None:
        """
        Save the processed batch to the output file by appending or overwriting.

        Args:
            batch (pd.DataFrame): Processed batch to save.

        Returns:
            None
        """
        if not batch.empty:
            if os.path.exists(self.output_file):
                # Read the existing data
                existing_data = pd.read_pickle(self.output_file)

                # Ensure indices are reset for both DataFrames
                existing_data = existing_data.reset_index(drop=True)
                batch = batch.reset_index(drop=True)

                # Concatenate with the adjusted batch
                updated_data = pd.concat([existing_data, batch], ignore_index=True)
            else:
                # No existing data, just use the new batch with reset index
                updated_data = batch.reset_index(drop=True)

            # Save the combined DataFrame back to the file
            updated_data.to_pickle(self.output_file)
            self._logger.info(f"Saved batch with {len(batch)} records to {self.output_file}")

    def process_batches(self) -> None:
        """
        Process the DataFrame either in a single batch or multiple batches.

        Returns:
            None
        """
        # Load input data
        df = pd.read_pickle(self.input_file)

        # Identify already processed IDs
        if not self.skip_processed:
            existing_ids = self.load_existing_ids()
            df_to_process = df[~df["id"].isin(existing_ids)]  # Filter unprocessed songs
        else:
            df_to_process = df

        self._logger.info(f"Total records to process: {len(df_to_process)}")
        if df_to_process.empty:
            self._logger.info("All records have already been processed.")
            return

        # Process all in a single batch if batch_size is None
        if self.batch_size is None:
            self._logger.info("Processing in a single batch without multiprocessing...")
            processed_data = self.process_batch(df_to_process)
            self.save_batch_to_output(processed_data)
        else:
            # Process in multiple batches with tqdm
            self._logger.info(f"Processing in batches of size {self.batch_size}...")
            for start in tqdm(range(0, len(df_to_process), self.batch_size), desc="Processing Batches", unit="batch"):
                batch = df_to_process.iloc[start:start + self.batch_size]
                processed_data = self.process_batch(batch)
                self.save_batch_to_output(processed_data)


