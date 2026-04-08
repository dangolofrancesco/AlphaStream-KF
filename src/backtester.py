import numpy as np
import pandas as pd

class Backtester:
    """
    Backtesting engine for the pairs trading strategy using the Kalman Filter and Z-Score based signals.
    Applies execution delays (shifting), transcation costs, and calculates risk metrics. 
    """

    def __init__(self, initial_capital: float = 100000.0, transaction_cost_bps: float = 0.5):
        """
        Args:
            initial_capital: The starting capital for the backtest in dollars
            transaction_cost_bps: Transcation costs + slippage in basis points (1 bps = 0.01%)
                                    5.0 bps means 0.05% cost per trade, which is a reasonable estimate for liquid stocks.
        """
        self.initial_capital = initial_capital
        self.tc_rate = transaction_cost_bps / 10000.0  # convert bps to decimal


    def run_backtest(self, signals_df: pd.DataFrame, y_prices: np.ndarray, x_prices: np.ndarray, betas: np.ndarray) -> pd.DataFrame:
        """
        Executes the backtest by applying the generated signals to the price data,

        Args:
            signals_df: DataFrame containing 'Target_Position' from strategy (1, -1, 0)
            y_prices: 1D array of the dependent variable prices (e.g., PEP)
            x_prices: 1D array of the independent variable prices (e.g., KO)
            betas: Array of dynamic hedge ratios (Beta) from the Kalman Filter
        """

        df = signals_df.copy()

        # Calculate the daily returns of the two stocks
        # Simple returns for P&L calculation: r_t = (P_t - P_{t-1}) / P_{t-1}
        df['Y_Return'] = pd.Series(y_prices).pct_change().fillna(0) # percentage return for Y daily
        df['X_Return'] = pd.Series(x_prices).pct_change().fillna(0)
        df['Beta'] = betas

        # Calculate the Portfolio Return (Spread Return) 
        # If we are long the spread (+1), we are Long 1 unit of Y and Short 'Beta' units of X
        df['Portfolio_Return'] = df['Y_Return'] - (df['Beta'] * df['X_Return'])

        # CRITICAL: Shift the position to avoid Look-Ahead Bias
        # Position taken at t-1 earns the return of t 
        df['Actual_Position'] = df['Target_Position'].shift(1).fillna(0)

        # Calculate Gross Strategy Return
        df['Gross_Strategy_Return'] = df['Actual_Position'] * df['Portfolio_Return']

        # Apply Transaction Costs
        # We only pay transaction costs when we change our position
        df['Trade_Executed'] = df['Actual_Position'].diff().abs()
        df['TC_Penalty'] = df['Trade_Executed'] * self.tc_rate

        # Calculate Net Strategy Return after Transaction Costs
        df['Net_Strategy_Return'] = df['Gross_Strategy_Return'] - df['TC_Penalty'].fillna(0)

        # Calculate Equity Curve
        df['Equity_Curve'] = self.initial_capital * (1 + df['Net_Strategy_Return'].cumprod())

        return df

    def calculate_metrics(self, df: pd.DataFrame) -> dict:
        """
        Calculate risk metrics.
        """
        net_returns = df['Net_Strategy_Return'].dropna()
        equity = df['Equity_Curve'].dropna()

        if len(net_returns) == 0 or net_returns.std() == 0:
            return {
                'Total Return': 0.0,
                'Annualized Return': 0.0,
                'Annualized Volatility': 0.0,
                'Sharpe Ratio': 0.0
            }
        
        # Basic Metrics
        total_return = (df['Equity_Curve'].iloc[-1] / self.initial_capital) - 1
        ann_return = (1 + total_return) ** (252 / len(df)) - 1
        ann_vol = net_returns.std() * np.sqrt(252)


        # Annualized Sharpe Ratio (assuming 252 trading days, risk-free rate = 0)
        daily_mean = net_returns.mean()
        daily_vol = net_returns.std()
        sharpe_ratio = (daily_mean / daily_vol) * np.sqrt(252) if daily_vol > 0 else 0.0

        # Sortino Ratio
        downside_returns = net_returns[net_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0
        sortino_ratio = (daily_mean / downside_vol) * np.sqrt(252) if downside_vol > 0 else 0.0

        # Maximum Drawdown
        rolling_max = equity.cummax()
        drawdown = equity / rolling_max - 1.0
        max_drawdown = drawdown.min()

        # Calman Ratio
        calmar = ann_return / abs(max_drawdown) if max_drawdown < 0 else np.inf

        # Value at Risk (VaR) at 95% confidence level
        var_95 = np.percentile(net_returns, 5)
        cvar_95 = net_returns[net_returns <= var_95].mean() if len(net_returns[net_returns <= var_95]) > 0 else var_95

        # Win Rate
        active_days = net_returns[net_returns != 0]
        if len(active_days) > 0:
            win_rate = len(active_days[active_days > 0]) / len(active_days)
        else:
            win_rate = 0.0

        # Moments of the Return Distribution
        skewness = net_returns.skew()
        kurtosis = net_returns.kurtosis()

        print("\n" + "="*30)
        print(" Backtest Performance Metrics ")
        print("="*30)
        print(f"Total Net Return:   {total_return*100:.2f}%")
        print(f"Annualized Return:   {ann_return*100:.2f}%")
        print(f"Annualized Vol:      {ann_vol*100:.2f}%")
        print("-" * 30)
        print(f"Annualized Sharpe:  {sharpe_ratio:.2f}")
        print(f"Annualized Sortino: {sortino_ratio:.2f}")
        print(f"Calmar Ratio:       {calmar:.2f}")
        print("-" * 30)
        print(f"VaR (95%):          {var_95*100:.2f}%")
        print(f"CVaR (95%):         {cvar_95*100:.2f}%")
        print(f"Maximum Drawdown:   {max_drawdown*100:.2f}%")
        print(f"Daily Win Rate:     {win_rate*100:.2f}%")
        print(f"Total Trades:       {df['Trade_Executed'].sum():.0f}")
        print("-" * 30)
        print(f"Skewness:           {skewness:.2f}")
        print(f"Kurtosis:           {kurtosis:.2f}")
        print("="*30)
        


# --- MODULAR TESTING ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=== Testing Backtester Engine ===")

    np.random.seed(42)
    n_days = 252
    
    x_prices = np.cumsum(np.random.normal(0, 1, n_days)) + 100
    y_prices = 1.5 * x_prices + np.cumsum(np.random.normal(0, 0.5, n_days)) + 20
    
    # betas from Kalman Filter
    betas = np.full(n_days, 1.5)
    
    # signal generation: let's create a simple signal that goes long the spread from day 50 to 100 and short the spread from day 150 to 200
    signals = np.zeros(n_days)
    signals[50:100] = 1   # Long Spread dal giorno 50 al 100
    signals[150:200] = -1 # Short Spread dal giorno 150 al 200
    
    signals_df = pd.DataFrame({'Target_Position': signals})
    
    
    # backtest execution
    bt = Backtester(initial_capital=100000.0, transaction_cost_bps=5.0)
    
    print("\nRunning backtest...")
    results_df = bt.run_backtest(signals_df, y_prices, x_prices, betas)
    
    # Calculate and print performance metrics
    bt.calculate_metrics(results_df)
    
    # Anti cheat test to verify that the position is correctly shifted by one day to avoid look-ahead bias
    # Let's check the first long signal at day 50 and verify that the actual position is 0 on day 50 and becomes 1 on day 51
    assert results_df['Target_Position'].iloc[50] == 1.0, "Error in signal generation: Target_Position at day 50 should be 1.0 (long)."
    assert results_df['Actual_Position'].iloc[50] == 0.0, "Look-Ahead Bias observed: Actual_Position at day 50 should be 0.0 due to shift, but it's not."
    assert results_df['Actual_Position'].iloc[51] == 1.0, "Execution delay observed: Actual_Position at day 51 should be 1.0 (long) due to shift, but it's not."
    print("\nLook-Ahead Bias test passed: Position is correctly shifted by one day.")


