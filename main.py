import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import DataLoader
from src.storage import DataStorage

def main():
    print("=== Pipeline Initialization ===")

    TICKERS = ['PEP', 'KO']  # Note: Y (Dependent) first, X (Independent) second
    START_DATE = '2015-01-01'
    END_DATE = '2024-01-01'
    DATA_FILE = 'pep_ko_daily.parquet'
    
    # 1. Initialize DataLoader and Storage
    loader = DataLoader(tickers=TICKERS, start_date=START_DATE, end_date=END_DATE)
    storage = DataStorage(data_dir="data")
    
    # 2. Pipeline Execution
    filepath = os.path.join(storage.data_dir, DATA_FILE)
    
    # Check if data already exists to avoid re-downloading (Standard Quant Practice)
    if os.path.exists(filepath):
        print("\n--- Data found locally. Loading from Parquet ---")
        df_raw = storage.load_from_parquet(DATA_FILE)
        loader.raw_data = df_raw # Inject data into loader
    else:
        print("\n--- Downloading Data from Web ---")
        df_raw = loader.fetch_data()
        storage.save_to_parquet(df_raw, DATA_FILE)
    
    # 3. Data Processing (Cleaning & C-Contiguous alignment)
    print("\n--- Processing Data for C++ / Cython Backend ---")
    y_prices, x_prices, dates = loader.clean_and_align()
    
    # 4. Validation Checks (The 'Quant Sanity Check')
    print("\n=== Validation Results ===")
    print(f"Total trading days: {len(dates)}")
    print(f"Missing values after clean: {np.isnan(y_prices).sum() + np.isnan(x_prices).sum()}")
    print(f"C-Contiguous Memory (Ready for C++): {y_prices.flags['C_CONTIGUOUS']}")
    
    log_ret_y = loader.get_log_returns(y_prices)
    print(f"Sample Log Return (PEP) on day 1: {log_ret_y[1]:.6f}")
    
    print("\nPipeline ready. Next step: Feed y_prices and x_prices into Kalman Filter.")

if __name__ == "__main__":
    main()