import numpy as np
import pandas as pd
from typing import Tuple

class Backtester:
    """
    Backtesting engine for the pairs trading strategy using the Kalman Filter and Z-Score based signals.
    Applies execution delays (shifting), transcation costs, and calculates risk metrics. 
    """

    def __init__(self, 
                initial_capital: float = 100000.0, 
                transaction_cost_bps: float = 5.0, # 5 bps commission per trade (0.05%)
                baseline_slippage: float = 2.0, # 2 bps baseline slippage
                high_slippage_bps: float = 15.0,
                liquidity_window: int = 30, # Look-back window for liquidity assessment
                stop_loss_pct: float = 0.05, # -5% of maximum loss
                take_profit_pct: float = 0.10 # +10% of maximum gain
                ):
        
        self.initial_capital = initial_capital
        self.base_tc = transaction_cost_bps / 10000.0  # bps -> decimal (5 bps = 0.0005)
        self.baseline_slippage = baseline_slippage / 10000.0
        self.high_slippage_rate = high_slippage_bps / 10000.0
        self.liquidity_window = liquidity_window
        self.stop_loss_rate = stop_loss_pct
        self.take_profit_rate = take_profit_pct


    def _calculate_dynamic_thresholds(self, volumes: np.ndarray) -> Tuple[float, float]:
        """
        Calculate precentiles 25 and 75 of volumes using a rolling window to determine dynamic thresholds for slippage modeling.
        """
        vol_series = pd.Series(volumes)
        p25 = vol_series.rolling(window=self.liquidity_window, min_periods=1).quantile(0.25).values
        p75 = vol_series.rolling(window=self.liquidity_window, min_periods=1).quantile(0.75).values

        return p25, p75
    
    def _calculate_execution_cost(self, trade_value: float, shares_traded: float,
                                volume_today: float, volume_25: float, volume_75: float) -> float:
        """
        Calculate the total execution cost = commission + market-impact slippage

        Cost model:
            - commission = base_tc * trade_value * adj_factor
            - slippage = baseline_slippage * (1 + proportion_of_trade * impact_scale) * trade_value
        """
        # If volume is 0 or NaN, we apply the highest rate to penalize the trade 
        if volume_today == 0 or np.isnan(volume_today):
            return (self.base_tc + self.baseline_slippage * 3) * trade_value

        # Liquidity adjustment on commission only 
        if volume_today < volume_25:
            adj_factor = 1.5 # Low liquidity: wider spread -> +50% commission
        elif volume_today > volume_75:
            adj_factor = 0.75 # High liquidity: tighter spread -> -25% commission
        else:
            adj_factor = 1.0 # Normal liquidity, base slippage

        commission_cost = self.base_tc * trade_value * adj_factor

        # Dynamic slippage 
        # which percentage of the market volume we are trading? The higher, the more slippage we pay
        proportion_of_trade = abs(shares_traded) / volume_today
        proportion_of_trade = min(proportion_of_trade, 1.0)  # Cap at 100% to avoid extreme slippage

        impact_scale = 10.0 # Amplifier: 1% of daily volume -> 10x baseline slippage

        slippage_rate = self.baseline_slippage * (1.0 + proportion_of_trade * impact_scale)
        slippage_cost = slippage_rate * trade_value

        return commission_cost + slippage_cost


    def run_backtest(self, signals_df: pd.DataFrame, 
                     y_prices: np.ndarray, x_prices: np.ndarray, betas: np.ndarray,
                     y_volumes: np.ndarray = None, x_volumes: np.ndarray = None) -> pd.DataFrame:
        """
        Dollar-based iterative backtest with explicit cash accounting for long and short legs.
 
        Args:
            signals_df: DataFrame containing 'Target_Position' from strategy (1, -1, 0)

            y_prices: 1D array of the dependent variable prices 
            x_prices: 1D array of the independent variable prices 
            betas: Array of dynamic hedge ratios (Beta) from the Kalman Filter
            y_volumes: Daily volume for Y (optional, used for slippage modeling)
            x_volumes: Daily volume for X (optional)
        """
        n_days = len(y_prices)

        if y_volumes is None: y_volumes = np.full(n_days, np.nan)  
        if x_volumes is None: x_volumes = np.full(n_days, np.nan)

        # Compute liquidity percentiles for slippage modeling
        y_p25, y_p75 = self._calculate_dynamic_thresholds(y_volumes)
        x_p25, x_p75 = self._calculate_dynamic_thresholds(x_volumes)
        targets = signals_df['Target_Position'].fillna(0).values  

        # Portfolio state variables
        cash = self.initial_capital
        shares_y = 0.0
        shares_x = 0.0
        current_position = 0  # 1 for long spread, -1 for short spread, 0 for flat
        trade_entry_equity = 0.0

        # Arrays for final dataframe
        equity_curve = np.zeros(n_days)
        tc_paid = np.zeros(n_days)

        for t in range(n_days):
            # Current equity
            current_equity = cash + (shares_y * y_prices[t]) + (shares_x * x_prices[t])
            equity_curve[t] = current_equity

            daily_tc = 0.0

            # Circuit breakers (SL and TP)
            circuit_break = False
            if current_position != 0 and trade_entry_equity > 0:
                trade_pnl_pct = (current_equity / trade_entry_equity) - 1.0
                if trade_pnl_pct <= -self.stop_loss_rate or trade_pnl_pct >= self.take_profit_rate:
                    circuit_break = True
            
            # Order execution logic (if target changes, we operate at t+1)
            # To avoid look-ahead bias, we execute trades of day t based on the previous day's signal (t-1)
            exec_target = targets[t-1] if t > 0 else 0
            if circuit_break:
                exec_target = 0 # Force flat regardeless of stretegy signal
            
            if exec_target != current_position:
                # 1. Close existing position if any
                if current_position != 0:
                    # Sell Y and Buy X to close the spread
                    val_y_traded = abs(shares_y * y_prices[t])
                    val_x_traded = abs(shares_x * x_prices[t])

                    # Slippage modeling: higher slippage for low volume days
                    tc_y = self._calculate_execution_cost(val_y_traded, abs(shares_y), y_volumes[t], y_p25[t], y_p75[t])
                    tc_x = self._calculate_execution_cost(val_x_traded, abs(shares_x), x_volumes[t], x_p25[t], x_p75[t])

                    daily_tc += tc_y + tc_x 

                    cash += (shares_y * y_prices[t]) + (shares_x * x_prices[t]) - daily_tc
                    shares_y = 0.0
                    shares_x = 0.0

                # 2. Open new position if target is not flat
                if exec_target != 0:
                    trade_entry_equity = current_equity

                    # We want to allocate about 50% of the entire capital 
                    alloc_y = 0.5 * current_equity

                    if exec_target == 1:  # Long Spread: Long Y, Short X
                        shares_y = alloc_y / y_prices[t]
                        shares_x = -(betas[t] * shares_y)  # Short X according to the hedge ratio to market neutrality 
                    elif exec_target == -1:  # Short Spread: Short Y, Long X
                        shares_y = -(alloc_y / y_prices[t])
                        shares_x = -(betas[t] * shares_y)  # Long X according to the hedge ratio to market neutrality

                    # Opening costs 
                    val_y_traded = abs(shares_y * y_prices[t])
                    val_x_traded = abs(shares_x * x_prices[t])

                    tc_y = self._calculate_execution_cost(val_y_traded, abs(shares_y), y_volumes[t], y_p25[t], y_p75[t])
                    tc_x = self._calculate_execution_cost(val_x_traded, abs(shares_x), x_volumes[t], x_p25[t], x_p75[t])

                    daily_tc += tc_y + tc_x
                    cash -= (shares_y * y_prices[t]) + (shares_x * x_prices[t]) + daily_tc

                current_position = exec_target
                tc_paid[t] = daily_tc
                
                # Update equity at the end of the day after trade execution
                current_equity = cash + (shares_y * y_prices[t]) + (shares_x * x_prices[t])
                equity_curve[t] = current_equity
        
        # Compile results into a DataFrame
        df = signals_df.copy()
        df['Equity_Curve'] = equity_curve
        df['Strategy_Net_Return'] = pd.Series(equity_curve).pct_change().fillna(0)
        df['Transaction_Cost'] = tc_paid
        df['Actual_Position'] = [targets[i-1] if i > 0 else 0 for i in range(n_days)]  # Shifted position to reflect execution delay
        # Mark a trade when the executed position changes from the prior day.
        actual_pos = pd.Series(df['Actual_Position']).fillna(0)
        df['Trade_Executed'] = actual_pos.diff().fillna(actual_pos).ne(0).astype(int)
        
        return df

    def calculate_metrics(self, df: pd.DataFrame, pair_label: str = "", period_label: str = "") -> dict:
        """
        Calculate risk metrics.
        """
        net_returns = df['Strategy_Net_Return'].dropna()
        equity = df['Equity_Curve'].dropna()

        if len(net_returns) == 0 or net_returns.std() == 0:
            terminal_wealth = float(df['Equity_Curve'].iloc[-1]) if len(df) > 0 else float(self.initial_capital)
            metrics = {
                'Total Return': 0.0,
                'Annualized Return': 0.0,
                'Annualized Volatility': 0.0,
                'Sharpe Ratio': 0.0,
                'Terminal Wealth': terminal_wealth,
            }
            return metrics
        
        # Basic Metrics
        terminal_wealth = float(df['Equity_Curve'].iloc[-1])
        total_return = (terminal_wealth / self.initial_capital) - 1
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

        if 'Trade_Executed' in df.columns:
            total_trades = float(df['Trade_Executed'].fillna(0).sum())
        elif 'Actual_Position' in df.columns:
            pos = pd.Series(df['Actual_Position']).fillna(0)
            total_trades = float(pos.diff().fillna(pos).ne(0).sum())
        else:
            total_trades = 0.0

        title_parts = ["BACKTEST PERFORMANCE METRICS"]
        if pair_label:
            title_parts.append(pair_label)
        if period_label:
            title_parts.append(period_label)
        title = " | ".join(title_parts)

        print("\n" + "="*40)
        print(f" {title} ")
        print("="*40)
        print("[RETURNS]")
        print(f"  Terminal Wealth:       ${terminal_wealth:,.2f}")
        print(f"  Total Net Return:      {total_return*100:.2f}%")
        print(f"  Annualized Return:     {ann_return*100:.2f}%")
        print(f"  Daily Mean Return:     {daily_mean*100:.4f}%")
        print("-" * 40)
        print("[RISK & VOLATILITY]")
        print(f"  Daily Volatility:      {daily_vol*100:.2f}%")
        print(f"  Annualized Vol:        {ann_vol*100:.2f}%")
        print(f"  Annualized Downside Vol: {downside_vol*100:.2f}%")
        print(f"  Maximum Drawdown:      {max_drawdown*100:.2f}%")
        print("-" * 40)
        print("[RISK-ADJUSTED RATIOS]")
        print(f"  Sharpe Ratio:          {sharpe_ratio:.2f}")
        print(f"  Sortino Ratio:         {sortino_ratio:.2f}")
        print(f"  Calmar Ratio:          {calmar:.2f}")
        print("-" * 40)
        print("[VALUE AT RISK]")
        print(f"  VaR (95%):             {var_95*100:.2f}%")
        print(f"  CVaR (95%):            {cvar_95*100:.2f}%")
        print("-" * 40)
        print("[DISTRIBUTION & TRADING]")
        print(f"  Skewness:              {skewness:.4f}")
        print(f"  Excess Kurtosis:       {kurtosis:.4f}")
        print(f"  Daily Win Rate:        {win_rate*100:.2f}%")
        print(f"  Total Trades:          {total_trades:.0f}")
        print("="*40)
        metrics = {
            'Terminal Wealth': terminal_wealth,
            'Total Return': float(total_return),
            'Annualized Return': float(ann_return),
            'Daily Mean Return': float(daily_mean),
            'Daily Volatility': float(daily_vol),
            'Annualized Volatility': float(ann_vol),
            'Annualized Downside Volatility': float(downside_vol),
            'Maximum Drawdown': float(max_drawdown),
            'Sharpe Ratio': float(sharpe_ratio),
            'Sortino Ratio': float(sortino_ratio),
            'Calmar Ratio': float(calmar),
            'VaR 95': float(var_95),
            'CVaR 95': float(cvar_95),
            'Skewness': float(skewness),
            'Excess Kurtosis': float(kurtosis),
            'Daily Win Rate': float(win_rate),
            'Total Trades': float(total_trades),
        }
        return metrics


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

