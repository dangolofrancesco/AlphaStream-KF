import os
import json
import itertools
import statsmodels.api as sm
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

from src.data_loader import DataLoader
from src.storage import DataStorage
from src.half_life import StationarityAnalyzer
from src.kalman_filter import KalmanFilterPairs
from src.strategy import StatArbStrategy
from src.backtester import Backtester


DATA_DIR   = "data"
TICKERS    = [
        "VNO", "REG", "SPG", "JNJ", "ORCL", "GS", "UPS", "CHTR", "GILD", 
        "LOW", "CVS", "AVGO", "F", "NEE", "DHR", "BKNG", "AAPL", 
        "MSFT", "AMZN", "GOOGL", "GOOG", "TSLA", "JPM", "V", "PG", "UNH", 
        "INTC", "HD", "MA", "DIS", "XOM", "VZ", "KO", "MRK", "PFE", "NFLX", 
        "CSCO", "BA", "CVX", "IBM", "MMM", "CAT", "RTX", "MCD", "WFC", 
        "SBUX", "PEP", "NKE", "C", "QCOM", "COST", "T", "MS", "HON", "LLY", 
        "ACN", "CRM", "TXN", "MDLZ", "ADP", "MO", "GE", "COP", "AMGN", "SO", 
        "KMI", "INTU", "AIG", "SPGI", "MU", "VLO", "CME", "ECL", "GM", "SLB", 
        "CMCSA", "FDX", "AAL", "HAL", "LMT", "WM", "MDT", "ISRG", "ILMN", 
        "ABT", "UNP", "BMY", "LIN", "TCEHY", "NVDA", "BAC", "WMT", "^HSI", 
        "^GSPC", "^VIX", "XOM", "SPY", "QQQ", "VTI", "IWM", "GLD", "VWO", 
        "EFA", "XLF", "XLU", "USO", "UNG", "UGA", "GLTR", "ADBE", "CTSH", 
        "FTNT", "PGR", "ROL", "TTWO", "IJR", "MDY", "VOE", "VOT", "IWP", 
        "IWS", "JKH", "IVOO", "MDYG", "PSA", "PLD", "ICE", "DFS", "MCO", 
        "EQR", "NTES", "HDB", "ITUB", "EEM", "JCI", "ADSK", "SCHW", "CINF", 
        "RJF", "AMAT", "AMT", "AAP", "ADM", "APH", "AEP", "AIV", "ALB", 
        "ALGN", "BIIB", "BLK", "ANSS", "AOS", "APA", "APD", "AXP", "AZO", 
        "BAX", "BDX", "CHRW", "CI", "CL", "CLX", "CMG", "CMI", "CNA", "CNP", 
        "CNX", "COF", "CPB", "CPRI", "CF", "CHD", "BSX", "BWA", "BXP", "CBRE"
    ]

TRAIN_START = "2011-01-01"   
TRAIN_END   = "2021-12-31"
TEST_START  = "2022-01-01"
TEST_END    = "2024-03-01"

TOP_N_PAIRS        = 5        
HALF_LIFE_MIN      = 5        
HALF_LIFE_MAX      = 45       
MIN_TRAIN_DAYS     = 200      
ENTRY_Z            = 1.0      # enter when |z| >= 1σ
EXIT_Z             = 0.5      # exit when |z| <= 0.5σ (avoids whipsawing at zero)
TRANSACTION_BPS    = 5.0      # commission (bps)
INITIAL_CAPITAL    = 100_000  # total portfolio capital across all selected pairs


def _fetch_with_volume(tickers: list, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download adjusted close prices and volumes in one call."""
    raw = yf.download(tickers, start=start, end=end, progress=False)
    prices = raw["Close"].ffill().dropna(how="all")
    volumes = raw["Volume"].ffill().fillna(0)
    return prices, volumes


def _scan_pairs(df_prices: pd.DataFrame, enforce_half_life: bool = True) -> list[dict]:
    """
    Engle-Granger two-step cointegration scan over all ticker combinations.
    Returns a list of candidate pairs sorted by p-value (ascending).
    """
    available = df_prices.columns[df_prices.notna().sum() > MIN_TRAIN_DAYS].tolist()
    candidates = []

    for asset_y, asset_x in itertools.combinations(available, 2):
        pair_data = df_prices[[asset_y, asset_x]].dropna()
        if len(pair_data) < MIN_TRAIN_DAYS:
            continue
        try:
            y = pair_data[asset_y]
            x = pair_data[asset_x]

            # Step 1: OLS regression to get residuals
            res_ols = sm.OLS(y, sm.add_constant(x)).fit()
            spread_train = res_ols.resid

            # Step 2: ADF test on the residuals
            is_stat, p_val = StationarityAnalyzer.check_stationarity(spread_train)
            if not is_stat:
                continue

            hl = StationarityAnalyzer.calculate_half_life(spread_train)
            hl_ok = HALF_LIFE_MIN < hl < HALF_LIFE_MAX
            if enforce_half_life and not hl_ok:
                continue

            candidates.append({
                "pair":      (asset_y, asset_x),
                "p_value":   p_val,
                "half_life": hl,
                "beta_init": float(res_ols.params.iloc[1]),
                "half_life_ok": hl_ok,
            })
        except Exception:
            continue

    candidates.sort(key=lambda d: d["p_value"])
    return candidates


def _backtest_pair(pair_info: dict, train_prices: pd.DataFrame, train_volumes: pd.DataFrame,
                   test_prices: pd.DataFrame, test_volumes: pd.DataFrame,
                   backtester: Backtester, pair_idx: int) -> tuple[pd.DataFrame | None, dict | None]:
    """
    Run the full KF → signal → backtest pipeline for a single pair on the test set.
    Returns (results_df, metrics_dict) or (None, None) if data is insufficient.
    """
    asset_y, asset_x = pair_info["pair"]

    test_pair = test_prices[[asset_y, asset_x]].dropna()
    if len(test_pair) < 50:
        print(f"  [SKIP] {asset_y} vs {asset_x}: insufficient test data ({len(test_pair)} rows)")
        return None, None

    y_test = test_pair[asset_y].values
    x_test = test_pair[asset_x].values

    # Align volumes to the same index
    y_vol = (test_volumes[asset_y].reindex(test_pair.index).ffill().fillna(0).values
             if asset_y in test_volumes.columns else None)
    x_vol = (test_volumes[asset_x].reindex(test_pair.index).ffill().fillna(0).values
             if asset_x in test_volumes.columns else None)

    # Kalman Filter 
    # R=1.0, Q=1e-3 → Q/R = 0.001
    # beta_init from OLS avoids the burn-in period on the test set.
    kf = KalmanFilterPairs(observation_variance=1.0, process_variance=1e-3)
    state_means, dynamic_spread = kf.filter(y_test, x_test, beta_init=pair_info["beta_init"])
    dynamic_betas = state_means[:, 0]

    # Signal generation 
    strategy = StatArbStrategy(entry_z=ENTRY_Z, exit_z=EXIT_Z)
    signals_df = strategy.generate_signals(dynamic_spread, half_life=pair_info["half_life"])

    # Backtest
    results_df = backtester.run_backtest(
        signals_df, y_test, x_test, dynamic_betas,
        y_volumes=y_vol, x_volumes=x_vol,
    )
    results_df.index = test_pair.index

    pair_label = f"{asset_y} vs {asset_x}"
    period_label = f"{TEST_START} to {TEST_END}"
    metrics = backtester.calculate_metrics(results_df, pair_label=pair_label, period_label=period_label)

    return results_df, metrics


def _combined_portfolio_metrics(all_results: list[pd.DataFrame], initial_capital_total: float,
                                cash_buffer: float = 0.0) -> None:
    """
    Sum the equity curves of all pairs to produce a combined portfolio equity curve,
    then print the aggregate performance metrics.

    instead of evaluating each pair in isolation,
    the portfolio P&L is the sum of all concurrent positions.  Diversification across
    uncorrelated pairs reduces kurtosis and stabilises the equity curve.
    """
    print("\n" + "=" * 50)
    print(f"  COMBINED PORTFOLIO — top {len(all_results)} pairs")
    print("=" * 50)

    # Align all equity curves on a common date index, forward-fill any gaps
    equity_series = [df["Equity_Curve"].rename(f"pair_{i}") for i, df in enumerate(all_results)]
    combined = pd.concat(equity_series, axis=1).ffill().bfill()
    portfolio_equity = combined.sum(axis=1) + float(cash_buffer)

    net_returns = portfolio_equity.pct_change().fillna(0)
    terminal_wealth = float(portfolio_equity.iloc[-1])
    total_return = (terminal_wealth / initial_capital_total) - 1
    ann_return = (1 + total_return) ** (252 / len(portfolio_equity)) - 1
    ann_vol = net_returns.std() * np.sqrt(252)
    daily_mean = net_returns.mean()
    daily_vol = net_returns.std()
    sharpe = (daily_mean / daily_vol) * np.sqrt(252) if daily_vol > 0 else 0.0

    downside = net_returns[net_returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0.0
    sortino = (daily_mean / downside_vol) * np.sqrt(252) if downside_vol > 0 else 0.0

    rolling_max = portfolio_equity.cummax()
    max_dd = (portfolio_equity / rolling_max - 1.0).min()
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.inf

    active = net_returns[net_returns != 0]
    win_rate = len(active[active > 0]) / len(active) if len(active) > 0 else 0.0
    kurtosis = net_returns.kurtosis()

    print(f"  Total capital budget:    ${initial_capital_total:,.0f}")
    if cash_buffer > 0:
        print(f"  Uninvested cash buffer:  ${cash_buffer:,.2f}")
    print(f"  Terminal Wealth:         ${terminal_wealth:,.2f}")
    print(f"  Total Net Return:        {total_return*100:.2f}%")
    print(f"  Annualized Return:       {ann_return*100:.2f}%")
    print(f"  Annualized Volatility:   {ann_vol*100:.2f}%")
    print(f"  Sharpe Ratio:            {sharpe:.2f}")
    print(f"  Sortino Ratio:           {sortino:.2f}")
    print(f"  Calmar Ratio:            {calmar:.2f}")
    print(f"  Maximum Drawdown:        {max_dd*100:.2f}%")
    print(f"  Daily Win Rate:          {win_rate*100:.2f}%")
    print(f"  Excess Kurtosis:         {kurtosis:.2f}")
    print("=" * 50)

    return portfolio_equity


def main():
    storage = DataStorage(data_dir=DATA_DIR)

    # STEP 1: fetch training data 
    print(f"\n[STEP 1] Fetching training data ({TRAIN_START} to {TRAIN_END})...")
    train_prices, train_volumes = _fetch_with_volume(TICKERS, TRAIN_START, TRAIN_END)

    available = train_prices.columns[train_prices.notna().sum() > MIN_TRAIN_DAYS].tolist()
    print(f"  Tickers with sufficient data: {len(available)} / {len(TICKERS)}")

    # STEP 2: cointegration scan on training set 
    print(f"\n[STEP 2] Scanning {len(available)*(len(available)-1)//2} pairs for cointegration...")
    candidates = _scan_pairs(train_prices[available], enforce_half_life=True)

    if not candidates:
        print("No cointegrated pairs found with the current filters. Exiting.")
        return

    top_pairs = candidates[:TOP_N_PAIRS]

    print(f"\n  TOP {len(top_pairs)} PAIRS SELECTED:")
    for i, p in enumerate(top_pairs, 1):
        print(f"  {i}. {p['pair'][0]} vs {p['pair'][1]}"
              f"  |  p={p['p_value']:.5f}  |  HL={p['half_life']:.1f}d"
              f"  |  β_init={p['beta_init']:.3f}"
              f"{'' if p.get('half_life_ok', True) else '  |  [HL fallback]'}")

    # STEP 3: fetch test data for all selected tickers 
    test_tickers = list({t for p in top_pairs for t in p["pair"]})
    print(f"\n[STEP 3] Fetching test data ({TEST_START} to {TEST_END}) "
          f"for {len(test_tickers)} tickers...")
    test_prices, test_volumes = _fetch_with_volume(test_tickers, TEST_START, TEST_END)

    # STEP 4: KF + strategy + backtest for each pair
    # Portfolio-level allocation: split total capital across selected pairs.
    # Each pair gets its own Backtester with per_pair_capital.
    print(f"\n[STEP 4] Running Kalman Filter, signals and backtest for each pair...")
    selected_pairs_count = len(top_pairs)
    per_pair_capital = INITIAL_CAPITAL / selected_pairs_count
    print(f"  Portfolio capital budget: ${INITIAL_CAPITAL:,.0f}")
    print(f"  Capital per selected pair (equal-weight): ${per_pair_capital:,.2f}")
    print("  Pair sizing rule: ~50% of each pair capital to Y leg, X leg scaled by dynamic β.")

    all_results: list[pd.DataFrame] = []
    all_metrics: list[dict] = []

    for i, pair_info in enumerate(top_pairs, 1):
        asset_y, asset_x = pair_info["pair"]
        print(f"\n  [{i}/{len(top_pairs)}] {asset_y} vs {asset_x}")

        bt = Backtester(
            initial_capital=per_pair_capital,
            transaction_cost_bps=TRANSACTION_BPS,
        )
        results_df, metrics = _backtest_pair(
            pair_info, train_prices, train_volumes,
            test_prices, test_volumes, bt, pair_idx=i,
        )

        if results_df is not None:
            all_results.append(results_df)
            all_metrics.append(metrics)
            storage.save_to_parquet(
                results_df,
                f"backtest_{asset_y}_{asset_x}".replace(".", "_").replace("^", ""),
            )

    if not all_results:
        print("\nNo pairs produced valid backtest results.")
        return

    # STEP 5: combined portfolio 
    print(f"\n[STEP 5] Aggregating combined portfolio ({len(all_results)} pairs)...")
    deployed_capital = per_pair_capital * len(all_results)
    cash_buffer = max(0.0, INITIAL_CAPITAL - deployed_capital)
    if cash_buffer > 0:
        print(f"  [INFO] {len(top_pairs) - len(all_results)} pair(s) skipped; keeping ${cash_buffer:,.2f} in cash.")
    portfolio_equity = _combined_portfolio_metrics(
        all_results,
        initial_capital_total=INITIAL_CAPITAL,
        cash_buffer=cash_buffer,
    )

    portfolio_df = pd.DataFrame({"Portfolio_Equity": portfolio_equity})
    storage.save_to_parquet(portfolio_df, "portfolio_combined")

    print(f"\nDone. Results saved to '{DATA_DIR}/'.")
    print("Open the Jupyter Notebook to visualise equity curves and Z-scores.")


if __name__ == "__main__":
    main()
