import numpy as np  
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from typing import Tuple

class StationarityAnalyzer:
    """
    Analyzes time series for mean-reverting properties.
    Used during the In-Sample (training) period to validate pairs
    and extract the dynamic window size for Z-Score calculation.
    """

    @staticmethod
    def check_stationarity(spread: pd.Series, significance_level: float=0.05) -> Tuple[bool, float]:
        """
        Perform the Augmented Dickey-Fuller test to check for stationarity.
        
        Args:
            spread: The time series of the spread.
            significance_level: The p-value threshold (default 5%).
            
        Returns:
            Tuple containing:
            - is_stationary (bool): True if p-value < significance_level.
            - p_value (float): The actual p-value from the ADF test.
        """
        # maxlag=1 is often used in StatArb to avoid over-complicating the AR structure
        # but letting AIC choose is standard practice
        adf_result = adfuller(spread.dropna(), autolag='AIC')
        p_value = adf_result[1]
        
        is_stationary = p_value < significance_level
        return is_stationary, p_value


    @staticmethod
    def calculate_half_life(spread: pd.Series, min_samples: int=30) -> float:
        """
        Calculates the half-life of mean reversion for a given spread.
        
        Formula: dS(t) = lambda * S(t-1) + mu + epsilon
        Half-life = -ln(2) / lambda
        
        Args:
            spread (pd.Series): The time series of the spread.
            min_samples (int): Minimum number of data points required.
            
        Returns:
            float: The half-life in periods (e.g., days). Returns np.inf if 
                   the series is not mean-reverting.
        """
        spread_clean = spread.dropna()

        if len(spread_clean) < min_samples:
            raise ValueError(f"Not enough data points to calculate half-life. Required: {min_samples}, Available: {len(spread_clean)}")
        
        # 1. S_{t-1} (lagged spread)
        lagged_spread = spread_clean.shift(1).dropna()

        # 2. dS(t) (delta spread)
        delta_spread = spread_clean.diff().dropna()

        # Align the series for regression
        aligned_index = lagged_spread.index.intersection(delta_spread.index) # Ensure we only use rows where both lagged_spread and delta_spread have values
        lagged_spread_aligned = lagged_spread.loc[aligned_index]
        delta_spread_aligned = delta_spread.loc[aligned_index]

        # 3. Linear regression to estimate lambda
        # We add a constant to capture the drift (mu) the non-zero mean of the spread
        X = sm.add_constant(lagged_spread_aligned)
        y = delta_spread_aligned

        model = sm.OLS(y, X).fit()

        lambda_coef = model.params.iloc[1]  # The coefficient of S(t-1)

        # 4. Check for mean-reverting property
        # if lambda >= 0, the series diverges or is a random walk, so we return infinite half-life
        if lambda_coef >= 0:
            return np.inf
        
        # 5. Calculate half-life
        half_life = -np.log(2) / lambda_coef
        return half_life
    
# --- MODULAR TESTING ---
if __name__ == "__main__":
    print("=== Testing StationarityAnalyzer ===")
    
    # Generate a synthetic mean-reverting spread: S_t = 0.85 * S_{t-1} + noise
    np.random.seed(42)
    n = 500
    synthetic_spread = np.zeros(n)
    for t in range(1, n):
        synthetic_spread[t] = 0.85 * synthetic_spread[t-1] + np.random.normal(0, 1)
        
    spread_series = pd.Series(synthetic_spread)
    
    # 1. Test Stationarity
    is_stat, p_val = StationarityAnalyzer.check_stationarity(spread_series)
    print(f"ADF Test p-value: {p_val:.4f} -> Stationary: {is_stat}")
    
    # 2. Test Half-Life
    hl = StationarityAnalyzer.calculate_half_life(spread_series)
    
    # Expected lambda = (0.85 - 1) = -0.15. Expected HL = -ln(2)/-0.15 = 4.62
    print(f"Calculated Half-Life: {hl:.2f} periods")
    print("If it's around 4.62, the math is spot on.")  