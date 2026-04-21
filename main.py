# main.py
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import yfinance as yf

# Imports from the updated modules
from src.backtester import Backtester as PairBacktester
from src.half_life import StationarityAnalyzer
from src.kalman_filter import KalmanFilterPairs
from src.storage import DataStorage
from src.strategy import StatArbStrategy

class BacktestOrchestrator:
    """Orchestrates pair selection, signal generation, and portfolio backtesting."""

    def __init__(
        self,
        data_dir: str = "data",
        tickers_file: str = "data/Tickers.json",
        ticker_group: str = "sample_tickers",
        train_start: str = "2011-01-01",
        train_end: str = "2021-12-31",
        test_start: str = "2022-01-01",
        test_end: str = "2024-03-01",
        top_n_pairs: int = 5,
        half_life_min: int = 5,
        half_life_max: int = 25,
        min_train_days: int = 200,
        entry_z: float = 1.0,
        exit_z: float = 0.0,
        transaction_bps: float = 5.0,
        initial_capital: float = 100_000.0,
    ):
        self.data_dir = data_dir
        self.tickers_file = tickers_file
        self.ticker_group = ticker_group
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.top_n_pairs = top_n_pairs
        self.half_life_min = half_life_min
        self.half_life_max = half_life_max
        self.min_train_days = min_train_days
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.transaction_bps = transaction_bps
        self.initial_capital = float(initial_capital)
        self.storage = DataStorage(data_dir=self.data_dir)

        self._tickers_cache = None
        self._train_cache = None
        self._test_cache = None

    def load_tickers(self) -> list[str]:
        cfg_path = Path(self.tickers_file)
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        tickers = cfg.get(self.ticker_group) or cfg.get("tickers")
        self._tickers_cache = [str(t).strip() for t in tickers if str(t).strip()]
        return self._tickers_cache

    def fetch_with_volume(self, tickers: list[str], start: str, end: str):
        raw = yf.download(tickers, start=start, end=end, progress=False)
        prices = raw["Close"].ffill().dropna(how="all")
        volumes = raw["Volume"].ffill().fillna(0)
        return prices, volumes

    def select_top_pairs(self, top_n: int = None, enforce_disjoint: bool = True) -> list[dict]:
        """
        Select pairs with a disjoint-set style rule to avoid overexposure to single tickers.
        """
        train_prices, _ = self.fetch_with_volume(self._tickers_cache, self.train_start, self.train_end)
        available = train_prices.columns[train_prices.notna().sum() > self.min_train_days].tolist()
        
        # Initial scan for stationarity and half-life
        candidates = []
        for y_t, x_t in itertools.combinations(available, 2):
            pair_data = train_prices[[y_t, x_t]].dropna()
            if len(pair_data) < self.min_train_days: continue
            
            res_ols = sm.OLS(pair_data[y_t], sm.add_constant(pair_data[x_t])).fit()
            is_stat, p_val = StationarityAnalyzer.check_stationarity(res_ols.resid)
            if is_stat:
                hl = StationarityAnalyzer.calculate_half_life(res_ols.resid)
                if self.half_life_min < hl < self.half_life_max:
                    candidates.append({"pair": (y_t, x_t), "p_value": p_val, "half_life": hl, "beta_init": res_ols.params.iloc[1]})

        candidates.sort(key=lambda x: x["p_value"])
        
        # Greedy selection with disjoint tickers
        selected = []
        used_tickers = set()
        target = top_n or self.top_n_pairs

        for c in candidates:
            y, x = c["pair"]
            if enforce_disjoint:
                if y in used_tickers or x in used_tickers: continue
                used_tickers.update([y, x])
            
            selected.append(c)
            if len(selected) == target: break
            
        return selected

    def _calculate_cointegration_guardian(
        self,
        y_test: np.ndarray,
        x_test: np.ndarray,
        lookback_window: int = 126,
        step_days: int = 21,
        p_value_threshold: float = 0.10,
    ) -> np.ndarray:
        """
        Generate an authorization mask (1 = allowed, 0 = suspended) by re-running
        the ADF test on rolling windows.

        - lookback_window: 126 days (~6 months of data)
        - step_days: 21 days (~1 trading month)
        """
        n_days = len(y_test)
        auth_mask = np.ones(n_days, dtype=float)

        # Start checking only after enough observations are available in the test set.
        for i in range(lookback_window, n_days, step_days):
            y_window = y_test[i - lookback_window : i]
            x_window = x_test[i - lookback_window : i]

            try:
                res_ols = sm.OLS(y_window, sm.add_constant(x_window)).fit()
                adf_result = adfuller(res_ols.resid, maxlag=1, autolag=None)
                p_val = adf_result[1]
                end_idx = min(i + step_days, n_days)

                if p_val > p_value_threshold:
                    # Kill switch: cointegration appears broken, suspend trading.
                    auth_mask[i:end_idx] = 0.0
                else:
                    auth_mask[i:end_idx] = 1.0
            except Exception:
                # Fail-safe behavior: suspend trading for the next interval.
                end_idx = min(i + step_days, n_days)
                auth_mask[i:end_idx] = 0.0

        return auth_mask

    def run_single_pair(self, pair_info: dict, test_prices: pd.DataFrame, test_volumes: pd.DataFrame, pair_capital: float):
        asset_y, asset_x = pair_info["pair"]
        test_pair = test_prices[[asset_y, asset_x]].dropna()
        
        y_test, x_test = test_pair[asset_y].values, test_pair[asset_x].values
        y_vol = test_volumes[asset_y].reindex(test_pair.index).values
        x_vol = test_volumes[asset_x].reindex(test_pair.index).values

        # New KF setup: separate variances for Beta (slow) and Alpha (fast)
        kf = KalmanFilterPairs()
        state_means, true_spread = kf.filter(y_test, x_test, beta_init=pair_info["beta_init"])
        
        strategy = StatArbStrategy(entry_z=self.entry_z, exit_z=self.exit_z)
        signals_df = strategy.generate_signals(true_spread, half_life=pair_info["half_life"])

        # Cointegration guardian kill switch.
        auth_mask = self._calculate_cointegration_guardian(
            y_test,
            x_test,
            lookback_window=126,
            step_days=21,
            p_value_threshold=0.10,
        )
        signals_df["Target_Position"] = signals_df["Target_Position"] * auth_mask
        signals_df["Auth_Mask"] = auth_mask

        # Institutional-style backtester (dollar-based + liquidity adjustment)
        #bt = PairBacktester(initial_capital=pair_capital, transaction_cost_bps=self.transaction_bps)
        bt = PairBacktester(
        initial_capital=pair_capital, 
        transaction_cost_bps=5.0,
        stop_loss_pct=0.10,  # Dagli più respiro (10%)
        take_profit_pct=0.20 # Cerca vincite più grandi
        )
        results_df = bt.run_backtest(signals_df, y_test, x_test, state_means[:, 0], y_vol, x_vol)
        results_df.index = test_pair.index
        
        metrics = bt.calculate_metrics(results_df, pair_label=f"{asset_y}/{asset_x}")
        return results_df, metrics

    def run(self):
        self.load_tickers()
        top_pairs = self.select_top_pairs(enforce_disjoint=True)

        if not top_pairs:
            print("No valid pairs found with current filters.")
            return None
        
        test_tickers = list({t for p in top_pairs for t in p["pair"]})
        test_prices, test_volumes = self.fetch_with_volume(test_tickers, self.test_start, self.test_end)
        
        per_pair_capital = self.initial_capital / len(top_pairs)
        results_list = []

        for p in top_pairs:
            res_df, _ = self.run_single_pair(p, test_prices, test_volumes, per_pair_capital)
            if res_df is not None:
                results_list.append(res_df)
                y_t, x_t = p["pair"]
                filename = f"backtest_{y_t}_{x_t}".replace(".", "_").replace("^", "")
                self.storage.save_to_parquet(res_df, filename)

        if not results_list:
            print("No pair backtest results generated.")
            return None

        # Portfolio aggregation
        equity_series = [df["Equity_Curve"] for df in results_list]
        portfolio_equity = pd.concat(equity_series, axis=1).ffill().sum(axis=1)
        portfolio_df = pd.DataFrame({"Portfolio_Equity": portfolio_equity})
        self.storage.save_to_parquet(portfolio_df, "portfolio_combined")
        
        print(f"\n[FINALE] Terminal Wealth Portafoglio: ${portfolio_equity.iloc[-1]:,.2f}")
        return portfolio_equity

if __name__ == "__main__":
    orchestrator = BacktestOrchestrator(ticker_group="sample_tickers")
    orchestrator.run()