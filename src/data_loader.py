import numpy as np
import pandas as pd
import yfinance as yf
from typing import List, Tuple

class DataLoader:
    """
    Handles data ingestion, cleaning, and alignment for Statical Arbitrage strategies.
    Prepares memory-contiguos arrays suitable for high-performance computations in C++.
    """
    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.raw_data = None

    def fetch_data(self) -> pd.DataFrame:
        """
        Download OHLCV data from Yahoo Finance for the specified tickers and date range.
        """
        print(f"Fetching data for {self.tickers} from {self.start_date} to {self.end_date} ... ")

        # Download adjusted close prices for the specified tickers
        # In this version of yfinance (>=0.2.18) 'Close' contains the adjusted prices 
        df = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Close']
        self.raw_data = df
        print("Data fetching complete.")
        return df
    
    def clean_and_align(self) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Clean missing data and align time series across all tickers.
        Returns strictly aligned C-contiguous arrays for the dependent and independent variables, along with the corresponding dates
        for fast processing in C++/Cython.
        """

        if self.raw_data is None:
            raise ValueError("Data not fetched. Call fetch_data() before clean_and_align().")

        # 1. Forward fill missing values to handle NaNs
        # If a stock doesn't trade on a given day (e.g., due to holidays),
        # we assume the price remains unchanged from the last available price.
        # WARNING: Never use backward fill as it can introduce look-ahead bias in backtesting.
        cleaned_data = self.raw_data.ffill()

        # 2. Drop initial rows with NaNs (if any) after forward filling
        cleaned_data = cleaned_data.dropna()

        # 3. Extract strictly aligned C-contiguous arrays for C++ processing
        # Asset Y (Dependent variable) is the first ticker, and Asset X (Independent variable) is the second ticker.
        y_prices = np.ascontiguousarray(cleaned_data[self.tickers[0]].values)
        x_prices = np.ascontiguousarray(cleaned_data[self.tickers[1]].values)
        dates = cleaned_data.index

        return y_prices, x_prices, dates

    def get_log_returns(self, prices: np.ndarray) -> np.ndarray:
        """
        Calculate log returns from risk management metrics
        Formula: r_t = ln(Pt / Pt-1)
        """
        log_returns = np.zeros_like(prices)
        log_returns[1:] = np.log(prices[1:] / prices[:-1])
        return log_returns


if __name__ == "__main__":
    print("Testing DataLoader module independently...")
    loader = DataLoader(tickers=['KO', 'PEP'], start_date='2020-01-01', end_date='2023-01-01')
    df = loader.fetch_data()
    y, x, dates = loader.clean_and_align()
    
    print(f"Data shape: {df.shape}")
    print(f"Y array (PEP) type/shape: {type(y)}, {y.shape}")
    print(f"X array (KO) type/shape: {type(x)}, {x.shape}")
    print(f"Are arrays C-contiguous? Y: {y.flags['C_CONTIGUOUS']}, X: {x.flags['C_CONTIGUOUS']}")