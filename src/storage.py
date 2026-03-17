import pandas as pd
from pathlib import Path

class DataStorage:
    """
    Handles storage and loading of market data 
    using the Parquet format for efficient disk I/O and memory usage.

    Note: We don't use csv files for storage because they are inefficient for large datasets, 
    especially when it comes to read/write performance and storage space.
    Parquet files, on the other hand, are designed for high performance and can handle large volumes 
    of data more efficiently, which is crucial for backtesting and live trading scenarios 
    where speed and resource management are critical.
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save_to_parquet(self, df: pd.DataFrame, filename: str):
        """
        Save a DataFrame to a Parquet file for efficient storage and retrieval.
        """
        file_path = self.data_dir / f"{filename}.parquet"
        # use the 'pyarrow' engine for better performance and compression
        # the 'snappy' compression is the standard for Parquet and offers a good balance between speed and compression ratio
        df.to_parquet(file_path, engine='pyarrow', compression='snappy')
        print(f"[Storage] Data saved to {file_path}")


    def load_from_parquet(self, filename: str) -> pd.DataFrame:
        """
        Load a DataFrame from a Parquet file.
        """
        file_path = self.data_dir / f"{filename}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        df = pd.read_parquet(file_path, engine='pyarrow')
        print(f"[Storage] Data loaded from {file_path}")
        return df

# -- Modular Testing -- 
if __name__ == "__main__":
    print("Testing DataStorage module...")
    # Create a sample DataFrame
    sample_data = {
        'KO': [50.0, 51.0, 52.0],
        'PEP': [150.0, 151.0, 152.0],
    }, index=pd.date_range(start='2024-01-01', periods=3)

    storage = DataStorage(data_dir='../data')
    storage.save_to_parquet(sample_data, filename='test_data')
    loaded_data = storage.load_from_parquet(filename='test_data')

    print("Original DataFrame:")
    print(sample_data)
    print("\nLoaded DataFrame:")
    print(loaded_data)

    # Assert they are the same
    assert loaded_data.equals(sample_data), "Loaded data does not match original data!"
    print("DataStorage module test passed successfully.")