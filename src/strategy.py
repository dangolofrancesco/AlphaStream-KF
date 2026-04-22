import numpy as np
import pandas as pd

class StatArbStrategy:
    """
    Transaltes the dynamic spread from Kalman Filter into actionable trading signals for pairs trading
    using the filter Z-score z_t = e_t / sqrt(S_t).
    """

    def __init__(self, entry_z: float = 2.0, exit_z: float = 0.0):
        """
        Initializes the strategy parameters.

        Args:
            entry_z: Z-score threshold for entering a trade (e.g., 2.0 means enter when spread is 2 std devs away from mean).
            exit_z: Z-score threshold for exiting a trade (e.g., 0.0 means exit when spread reverts to mean).
        """
        self.entry_z = entry_z
        self.exit_z = exit_z

    def _get_size_multiplier(self, abs_z: float) -> float:
        """
        Determines the position size multiplier based on the absolute Z-score.
        This allows for a more aggressive position when the spread is further from the mean, and a
        more conservative position when it's closer.
        """
        if abs_z <= 1.0:   return 0.0
        elif abs_z < 1.5:  return 0.5
        elif abs_z < 2.0:  return 0.75
        elif abs_z < 2.5:  return 0.9
        else:              return 1.0

    def generate_signals(self, innovations: np.ndarray, innovation_var: np.ndarray) -> pd.DataFrame:
        """
        Calculates the Z-Score and generates the target position for each time step.
        
        Args:
            innovations: 1D numpy array containing the KF innovation e_t.
            innovation_var: 1D numpy array containing innovation variance S_t.

        Returns:
            A pandas DataFrame with columns:
                - 'spread': the input spread array
                - 'z_score': the calculated Z-Score of the spread
                - 'position': the target position (1 for long, -1 for short, 0 for no position)
        """

        # Z-score of the filter 
        z_score = pd.Series(innovations / np.sqrt(innovation_var))

        n = len(z_score)
        position = np.zeros(n, dtype=float)  # pre-allocate position array
        current_position = 0  # 0 = flat, 1 = long spread, -1 = short spread
        entry_multiplier = 0.0  # This will be adjusted based on the Z-score distance
        hard_stop_z = 3.5  # Structural-break hard stop threshold

        for t in range(n):
            z = z_score.iloc[t]

            # If Z-Score is NaN, we stay flat.
            if pd.isna(z):
                position[t] = 0
                continue

            # Entry Conditions
            if current_position == 0:
                if z >= self.entry_z and z < hard_stop_z:
                    current_position = -1  # spread is too high ->  short the spread
                    entry_multiplier = self._get_size_multiplier(abs(z))
                elif z <= -self.entry_z and z > -hard_stop_z:
                    current_position = 1   # spread is too low -> long the spread
                    entry_multiplier = self._get_size_multiplier(abs(z))

            # Exit & Hard Stop Conditions
            elif current_position == 1:
                # Stop out on structural break or exit on mean reversion.
                if z <= -hard_stop_z or z >= -self.exit_z:
                    current_position = 0
                    entry_multiplier = 0.0
            elif current_position == -1:
                # Stop out on structural break or exit on mean reversion.
                if z >= hard_stop_z or z <= self.exit_z:
                    current_position = 0
                    entry_multiplier = 0.0

            # Multiplier FIXED on entry 
            position[t] = current_position * entry_multiplier

        # 4. Compile results into a DataFrame
        results = pd.DataFrame({
            'Innovation': pd.Series(innovations),
            'Innovation_Var': pd.Series(innovation_var),
            'Z_Score': z_score,
            'Target_Position': position
        })

        return results
    
    # --- MODULAR TESTING ---
if __name__ == "__main__":
    print("=== Testing Signal Generation Strategy ===")
    
    # 1. Create a synthetic spread that oscillates
    np.random.seed(42)
    t = np.linspace(0, 100, 500)
    # A sine wave simulating mean reversion, plus some noise
    synthetic_spread = np.sin(t) * 2.5 + np.random.normal(0, 0.2, 500)
    
    # 2. Initialize and run strategy
    strategy = StatArbStrategy(entry_z=2.0, exit_z=0.0)
    synthetic_var = np.ones_like(synthetic_spread)
    signals_df = strategy.generate_signals(synthetic_spread, synthetic_var)
    
    # 3. Analyze Results
    long_days = (signals_df['Target_Position'] == 1).sum()
    short_days = (signals_df['Target_Position'] == -1).sum()
    flat_days = (signals_df['Target_Position'] == 0).sum()
    
    print(f"Total trading periods: {len(signals_df)}")
    print(f"Days Long the Spread: {long_days}")
    print(f"Days Short the Spread: {short_days}")
    print(f"Days Flat (Out of market): {flat_days}")
    
    # Verify the logic: Check the Z-score at the first Long entry
    long_entries = signals_df[(signals_df['Target_Position'] == 1) & (signals_df['Target_Position'].shift(1) == 0)]
    if not long_entries.empty:
        first_long_z = long_entries['Z_Score'].iloc[0]
        print(f"\nZ-Score at first Long Entry: {first_long_z:.2f} (Should be <= -2.0)")
        assert first_long_z <= -2.0, "Entry logic failed!"
        print("Signal Generation logic works perfectly!")
