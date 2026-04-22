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

    DEFAULT_SECTOR_GROUPS = {
        "tech_platforms": ["AAPL", "MSFT", "GOOGL", "META"],
        "tech_hardware_semis": ["CSCO", "JNPR", "INTC", "AMD", "NVDA"],
        "consumer_staples": ["KO", "PEP", "PG", "CL", "PM", "MO"],
        "autos": ["F", "GM"],
        "apparel_lifestyle": ["NKE", "ADDYY"],
        "energy_integrated": ["XOM", "CVX", "BP", "SHEL"],
        "energy_upstream_services": ["COP", "OXY", "MRO", "DVN", "SLB", "HAL"],
    }

    def __init__(
        self,
        data_dir: str = "data",
        tickers_file: str = "data/Tickers.json",
        ticker_group: str = "new_sample",
        train_start: str = "2011-01-01",
        train_end: str = "2016-12-31",
        test_start: str = "2017-01-01",
        test_end: str = "2018-12-31",
        top_n_pairs: int = 5,
        half_life_min: int = 5,
        half_life_max: int = 30,
        min_train_days: int = 200,
        entry_z: float = 2.0,
        exit_z: float = 0.0,
        transaction_bps: float = 5.0,
        initial_capital: float = 100_000.0,
        adf_selection_p_value_threshold: float = 0.05,
        enable_half_life_fallback: bool = True,
        fallback_half_life_min: int = 2,
        fallback_half_life_max: int = 45,
        use_log_prices: bool = True,
        kf_r: float = 1e-4,
        kf_q_beta: float = 1e-6,
        kf_q_alpha: float = 1e-4,
        enable_adf_guardian: bool = False,
        adf_guardian_p_value_threshold: float = 0.10,
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
        self.adf_selection_p_value_threshold = float(adf_selection_p_value_threshold)
        self.enable_half_life_fallback = enable_half_life_fallback
        self.fallback_half_life_min = int(fallback_half_life_min)
        self.fallback_half_life_max = int(fallback_half_life_max)
        self.use_log_prices = bool(use_log_prices)
        self.kf_r = float(kf_r)
        self.kf_q_beta = float(kf_q_beta)
        self.kf_q_alpha = float(kf_q_alpha)
        self.enable_adf_guardian = enable_adf_guardian
        self.adf_guardian_p_value_threshold = float(adf_guardian_p_value_threshold)
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

        close = raw["Close"]
        volume = raw["Volume"]

        # Normalize single-ticker downloads to DataFrame.
        if isinstance(close, pd.Series):
            col_name = tickers[0] if tickers else "TICKER"
            close = close.to_frame(name=col_name)
        if isinstance(volume, pd.Series):
            col_name = tickers[0] if tickers else "TICKER"
            volume = volume.to_frame(name=col_name)

        requested = [str(t) for t in tickers]
        present = set(map(str, close.columns.tolist()))

        missing = [t for t in requested if t not in present]
        empty = [t for t in requested if t in present and close[t].dropna().empty]
        failed = sorted(set(missing + empty))
        if failed:
            print(f"[Data] Excluding tickers with failed/empty download: {', '.join(failed)}")

        prices = close.drop(columns=[t for t in failed if t in close.columns], errors="ignore")
        volumes = volume.drop(columns=[t for t in failed if t in volume.columns], errors="ignore")

        prices = prices.ffill().dropna(how="all")
        volumes = volumes.reindex(columns=prices.columns).ffill().fillna(0)
        return prices, volumes

    def _build_same_sector_pairs(self, available: list[str]) -> list[tuple[str, str]]:
        """Build candidate pairs using only tickers that belong to the same sector group."""
        sector_by_ticker = {
            ticker: sector
            for sector, tickers in self.DEFAULT_SECTOR_GROUPS.items()
            for ticker in tickers
        }

        grouped: dict[str, list[str]] = {}
        unknown = []
        for ticker in available:
            sector = sector_by_ticker.get(ticker)
            if sector is None:
                unknown.append(ticker)
                continue
            grouped.setdefault(sector, []).append(ticker)

        if unknown:
            print(f"[Pair Selection] Excluding unclassified tickers: {', '.join(sorted(unknown))}")

        pairs = []
        for tickers in grouped.values():
            tickers = sorted(tickers)
            if len(tickers) < 2:
                continue
            pairs.extend(itertools.combinations(tickers, 2))

        return pairs

    def scan_pairs(self, df_prices: pd.DataFrame, enforce_half_life: bool = True) -> list[dict]:
        """
        Scan candidate pairs on training prices using recent dynamics only.

        - Pairs are tested only within the same sector group.
        - OLS + ADF use the most recent 252 training bars when available.
        - Pairs with fewer than 100 bars in the scan window are skipped.
        """
        available = df_prices.columns[df_prices.notna().sum() > self.min_train_days].tolist()
        pairs_to_test = self._build_same_sector_pairs(available)

        print(f"[Pair Selection] Available tickers: {len(available)}")
        print(f"[Pair Selection] Candidate pairs same sector: {len(pairs_to_test)}")

        tested = []
        lookback_window = 252

        for asset_y, asset_x in pairs_to_test:
            pair_data = df_prices[[asset_y, asset_x]].dropna()

            # Use only the most recent 252 training bars to capture current dynamics.
            if len(pair_data) > lookback_window:
                pair_data = pair_data.iloc[-lookback_window:]

            # Minimum viable sample size for reliable ADF testing.
            if len(pair_data) < 100:
                continue

            try:
                if self.use_log_prices:
                    y_train = np.log(pair_data[asset_y])
                    x_train = np.log(pair_data[asset_x])
                else:
                    y_train = pair_data[asset_y]
                    x_train = pair_data[asset_x]

                res_ols = sm.OLS(y_train, sm.add_constant(x_train)).fit()
                spread_train = res_ols.resid
                _, p_val = StationarityAnalyzer.check_stationarity(spread_train)

                candidate = {
                    "pair": (asset_y, asset_x),
                    "p_value": float(p_val),
                    "beta_init": float(res_ols.params.iloc[1]),
                    "alpha_init": float(res_ols.params.iloc[0]),
                    "residuals": spread_train,
                }

                if enforce_half_life:
                    hl = StationarityAnalyzer.calculate_half_life(spread_train)
                    candidate["half_life"] = hl
                    strict_ok = self.half_life_min < hl < self.half_life_max
                    relaxed_ok = self.fallback_half_life_min < hl < self.fallback_half_life_max
                    candidate["strict_half_life_ok"] = strict_ok
                    candidate["relaxed_half_life_ok"] = relaxed_ok

                tested.append(candidate)
            except Exception:
                continue

        if not tested:
            print("[Pair Selection] No valid pairs to test after data-quality filters.")
            return []

        adf_candidates = [
            c for c in tested if c["p_value"] <= self.adf_selection_p_value_threshold
        ]
        print(
            f"[Pair Selection] Tested pairs: {len(tested)} | "
            f"ADF-passing pairs (p <= {self.adf_selection_p_value_threshold:.2f}): {len(adf_candidates)}"
        )

        if not enforce_half_life:
            adf_candidates.sort(key=lambda x: x["p_value"])
            return adf_candidates

        hl_candidates = [c for c in adf_candidates if c.get("strict_half_life_ok", False)]
        hl_relaxed_candidates = [c for c in adf_candidates if c.get("relaxed_half_life_ok", False)]

        print(
            f"[Pair Selection] Half-life pass (strict {self.half_life_min}-{self.half_life_max}): "
            f"{len(hl_candidates)}"
        )

        if not hl_candidates and self.enable_half_life_fallback:
            hl_candidates = hl_relaxed_candidates
            print(
                "[Pair Selection] WARNING: 0 strict half-life pairs. "
                f"Using relaxed half-life band {self.fallback_half_life_min}-{self.fallback_half_life_max}: "
                f"{len(hl_candidates)} pairs."
            )

        hl_candidates.sort(key=lambda x: x["p_value"])
        return hl_candidates

    def select_top_pairs(self, top_n: int = None, enforce_disjoint: bool = True) -> list[dict]:
        """
        Select pairs with a disjoint-set style rule to avoid overexposure to single tickers.

        Pipeline:
            1. Same-sector pair generation
            2. OLS + ADF on last 252 bars of training set (no FDR correction)
            3. Half-life filter on survivors
            4. Greedy disjoint-ticker selection
            5. Fixed Q/R assignment for selected pairs
        """
        train_prices, _ = self.fetch_with_volume(self._tickers_cache, self.train_start, self.train_end)
        hl_candidates = self.scan_pairs(train_prices, enforce_half_life=True)
        target = top_n or self.top_n_pairs

        if not hl_candidates:
            return []

        # Stage 3: greedy disjoint-ticker selection.
        pre_selected = []
        used_tickers = set()
        for c in hl_candidates:
            y, x = c["pair"]
            if enforce_disjoint:
                if y in used_tickers or x in used_tickers:
                    continue
                used_tickers.update([y, x])
            pre_selected.append(c)
            if len(pre_selected) == target:
                break

        print(f"[Pair Selection] Pre-selected pairs after disjoint rule: {len(pre_selected)}")

        # Stage 4: assign fixed Q/R for the selected pairs.
        fixed_q = np.diag([self.kf_q_beta, self.kf_q_alpha])
        selected = []
        for c in pre_selected:
            y_t, x_t = c["pair"]
            print(f"\n[KF Params] {y_t} / {x_t} | R={self.kf_r:.6g} Q=diag([{self.kf_q_beta:.6g}, {self.kf_q_alpha:.6g}])")

            selected.append(
                {
                    "pair": c["pair"],
                    "p_value": c["p_value"],
                    "half_life": c["half_life"],
                    "beta_init": c["beta_init"],
                    "alpha_init": c["alpha_init"],
                    "R": self.kf_r,
                    "Q": fixed_q.copy(),
                }
            )

        return selected

    def _calculate_cointegration_guardian(
        self,
        y_test: np.ndarray,
        x_test: np.ndarray,
        lookback_window: int = 252,
        step_days: int = 21,
        adf_p_value_threshold: float = 0.10,
    ) -> np.ndarray:
        """
        Generate an authorization mask (1 = allowed, 0 = suspended) by re-running
        rolling residual-based stationarity tests.

        - lookback_window: 252 days (~12 months of data)
        - step_days: 21 days (~1 trading month)
        - ADF: if p-value is too high, residuals are likely not stationary.
        """
        n_days = len(y_test)
        auth_mask = np.ones(n_days, dtype=float)

        # Start checking only after enough observations are available in the test set.
        for i in range(lookback_window, n_days, step_days):
            y_window = y_test[i - lookback_window : i]
            x_window = x_test[i - lookback_window : i]

            try:
                if self.use_log_prices:
                    y_window = np.log(y_window)
                    x_window = np.log(x_window)
                res_ols = sm.OLS(y_window, sm.add_constant(x_window)).fit()
                adf_result = adfuller(res_ols.resid, maxlag=1, autolag=None)
                adf_p_val = adf_result[1]
                end_idx = min(i + step_days, n_days)

                is_broken = adf_p_val > adf_p_value_threshold

                if is_broken:
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

        # Instantiate the Kalman Filter with fixed low-noise params for log-prices.
        kf = KalmanFilterPairs(
            observation_variance=1e-4,  # R low measurement noise for log-prices
            delta_beta=1e-6,            # Q_beta very slow
            delta_alpha=1e-6,           # Q_alpha very slow (reduced from 1e-4)
        )
        state_means, innovations, innovation_vars = kf.filter(
            y_test,
            x_test,
            beta_init=pair_info["beta_init"],
            alpha_init=pair_info.get("alpha_init"),
            use_log_prices=self.use_log_prices,
        )

        strategy = StatArbStrategy(entry_z=self.entry_z, exit_z=self.exit_z)
        signals_df = strategy.generate_signals(innovations, innovation_vars)
        raw_active_days = int(np.count_nonzero(signals_df["Target_Position"].values))

        # Statistical guardian (rolling ADF 252d), optional.
        if self.enable_adf_guardian:
            auth_mask_adf = self._calculate_cointegration_guardian(
                y_test,
                x_test,
                lookback_window=252,
                step_days=21,
                adf_p_value_threshold=self.adf_guardian_p_value_threshold,
            )
        else:
            auth_mask_adf = np.ones_like(innovations, dtype=float)

        # Authorization mask from ADF guardian only.
        auth_mask = auth_mask_adf

        signals_df["Target_Position"] = signals_df["Target_Position"] * auth_mask
        signals_df["Auth_Mask"] = auth_mask
        signals_df["Auth_Mask_ADF"] = auth_mask_adf

        masked_active_days = int(np.count_nonzero(signals_df["Target_Position"].values))
        print(
            f"    [{asset_y}/{asset_x}] Active signal days raw/masked: "
            f"{raw_active_days}/{masked_active_days} | mask mean={np.mean(auth_mask):.2f}"
        )
        print(
            f"    [{asset_y}/{asset_x}] Mask coverage ADF={np.mean(auth_mask_adf):.2f}"
        )

        # Dollar-based backtester with liquidity adjustment.
        bt = PairBacktester(
            initial_capital=pair_capital,
            transaction_cost_bps=self.transaction_bps,
            stop_loss_pct=0.10,
            take_profit_pct=0.20,
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

        # Persist pair metadata (Q is a numpy array, so convert for JSON).
        metadata_path = Path(self.data_dir) / "top_pairs_metadata.json"
        serializable = []
        for p in top_pairs:
            entry = {
                "pair": list(p["pair"]),
                "p_value": float(p["p_value"]),
                "half_life": float(p["half_life"]),
                "beta_init": float(p["beta_init"]),
                "alpha_init": float(p["alpha_init"]),
                "R": float(p["R"]),
                "Q_diag": [float(p["Q"][0, 0]), float(p["Q"][1, 1])],
            }
            serializable.append(entry)
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

        test_tickers = list({t for p in top_pairs for t in p["pair"]})
        test_prices, test_volumes = self.fetch_with_volume(test_tickers, self.test_start, self.test_end)
        
        per_pair_capital = self.initial_capital / len(top_pairs)
        results_list = []
        pair_summaries = []

        for p in top_pairs:
            res_df, metrics = self.run_single_pair(p, test_prices, test_volumes, per_pair_capital)
            if res_df is not None:
                results_list.append(res_df)
                y_t, x_t = p["pair"]
                filename = f"backtest_{y_t}_{x_t}".replace(".", "_").replace("^", "")
                self.storage.save_to_parquet(res_df, filename)

                pair_summaries.append(
                    {
                        "pair": f"{y_t}/{x_t}",
                        "coverage": float(np.mean(res_df["Auth_Mask"])),
                        "trades": int(metrics.get("Total Trades", 0)),
                        "terminal": float(metrics.get("Terminal Wealth", per_pair_capital)),
                    }
                )

        if not results_list:
            print("No pair backtest results generated.")
            return None

        # Portfolio aggregation
        equity_series = [df["Equity_Curve"] for df in results_list]
        portfolio_equity = pd.concat(equity_series, axis=1).ffill().sum(axis=1)
        portfolio_df = pd.DataFrame({"Portfolio_Equity": portfolio_equity})
        self.storage.save_to_parquet(portfolio_df, "portfolio_combined")

        if pair_summaries:
            print("\n[Coverage Summary]")
            for s in pair_summaries:
                print(
                    f"  {s['pair']}: coverage={s['coverage']:.2f}, "
                    f"trades={s['trades']}, terminal=${s['terminal']:,.2f}"
                )
        
        print(f"\n[FINALE] Terminal Wealth Portafoglio: ${portfolio_equity.iloc[-1]:,.2f}")
        return portfolio_equity

if __name__ == "__main__":
    orchestrator = BacktestOrchestrator(ticker_group="new_sample")
    orchestrator.run()