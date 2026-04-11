# Statistical Arbitrage with Kalman Filter

An implementation of a **pairs trading** strategy applied to US equities, combining Engle-Granger cointegration analysis with a streaming Kalman Filter for dynamic hedge ratio estimation. The system runs a fully automated pipeline from data ingestion to portfolio-level backtesting, with explicit handling of transaction costs, liquidity-adjusted slippage, and per-trade risk controls.

## Results

Backtested on a universe of ~170 US large-cap equities over the period **January 2022 – March 2024** (out-of-sample), after training the cointegration model on **January 2011 – December 2021**:

| Metric | Value |
|---|---|
| Total net return | +1.36% |
| Annualized return | +0.63% |
| Annualized volatility | 2.26% |
| Sharpe ratio | 0.29 |
| Maximum drawdown | −2.51% |
| Total trades | 75 |

The strategy remained profitable through a period in which the S&P 500 experienced a drawdown exceeding 25% (2022), demonstrating the market-neutral character of the approach. The low drawdown relative to directional benchmarks reflects the effectiveness of the hedge ratio calibration and the stop-loss mechanism.

## Methodology

### Pair Selection — Engle-Granger Two-Step

Pair selection is based on **cointegration** rather than correlation. Correlation captures short-term co-movement but says nothing about long-term equilibrium, while cointegration implies that two non-stationary price series form a stationary linear combination, which is the economic precondition for a mean-reverting spread to exist and persist.

The procedure follows Engle and Granger. In the first step, the dependent asset Y is regressed on the independent asset X via OLS to estimate the static hedge ratio and extract the residual spread. In the second step, an Augmented Dickey-Fuller test is applied to the residuals, where a p-value below 0.05 rejects the null of a unit root, confirming that the spread is stationary and therefore mean-reverting. To further constrain the opportunity set to pairs that revert at a tradeable frequency, a half-life filter is applied: only pairs with a mean-reversion half-life between 5 and 45 days are retained.

The entire scan runs on the training set (2011–2021) across all combinations of available tickers, and the top N pairs ranked by p-value are selected for out-of-sample trading.

### Hedge Ratio Estimation

The OLS beta from the first step is a static estimate valid at the time of estimation, but the true relationship between two assets drifts over time as business conditions, capital structure, and macro exposures evolve. A static hedge ratio applied rigidly in the test set will therefore produce a spread that is not truly stationary, generating false signals and unintended directional exposure.

To address this, the hedge ratio is estimated dynamically using a **Kalman Filter** applied in streaming mode over the test period. The state vector contains two components: the hedge ratio `β_t` and the intercept `α_t`, and the observation equation at each time step is `y_t = β_t · x_t + α_t + ε_t`. The filter follows the standard predict-update recursion:

- **Predict**: `θ̂_{t|t−1} = θ̂_{t−1}`, `P_{t|t−1} = P_{t−1} + Q`
- **Update**: the Kalman gain `K_t` weights the innovation `e_t = y_t − H_t θ̂_{t|t−1}` proportionally to the ratio of process noise to observation noise

The noise parameters are set to `R = 1.0` and `Q = 10⁻³ · I`, giving a Q/R ratio of 0.001 — meaning the filter allows the hedge ratio to drift slowly and smoothly rather than chasing daily price fluctuations. The filter is warm-started with the OLS beta from the training period, eliminating the convergence delay that would otherwise affect the first weeks of the test set. The innovation sequence `e_t` produced by the filter is used directly as the trading spread.

### Signal Generation

The spread is standardized into a **Z-score** using a rolling mean and standard deviation computed over a window equal to the half-life of the pair estimated in the training period. This calibrates the entry threshold to the natural oscillation frequency of each specific spread, rather than applying a uniform window across all pairs.

Trading signals follow a state-machine logic with hysteresis:
- **Long spread** (long Y, short X): when the Z-score falls below `−1.0σ`, the spread is deemed unusually compressed and expected to widen back toward the mean.
- **Short spread** (short Y, long X): when the Z-score rises above `+1.0σ`.
- **Exit**: when the Z-score reverts to within `±0.5σ` of the mean. The asymmetric exit threshold (0.5σ rather than 0) avoids whipsawing around the mean while still capturing the majority of the mean-reversion profit.

Position sizing is adaptive across four tiers based on the absolute Z-score magnitude, following the approach of Pole (2007):


| Z-score range | Size multiplier |
|---|---|
| (1.0, 1.5) | 0.50× |
| [1.5, 2.0) | 0.75× |
| [2.0, 2.5) | 0.90× |
| ≥ 2.5      | 1.00× |



Larger positions are taken only when the signal is stronger, which reduces exposure on marginal entries and concentrates capital on the highest-conviction opportunities. All signals are shifted forward by one day before execution to prevent look-ahead bias, the signal computed at the close of day `t` is executed at the open of day `t+1`.

### Backtesting

The backtester simulates **dollar-based P&L** with explicit accounting for both legs of each trade. Cash and share quantities are tracked separately, with the X leg sized as `shares_x = −β_t · shares_y` at the time of entry to ensure dynamic market neutrality. The allocation per pair is 50% of the assigned capital to the Y leg, with the X leg sized proportionally.

Transaction costs are modelled as a percentage of notional value (5 bps base commission), adjusted by a liquidity factor derived from rolling 30-day volume percentiles: low-volume days (below P25) incur a 1.5× multiplier, high-volume days (above P75) a 0.75× multiplier. An additional market-impact slippage term scales with position size as a fraction of daily volume, capturing the price impact of larger trades.

Per-trade risk controls are enforced via circuit breakers: a **stop-loss** at −5% of trade entry equity and a **take-profit** at +10%, both evaluated daily against the current mark-to-market. When triggered, the position is fully unwound at the day's price.

The combined portfolio equity is the equal-weighted sum of the five individual pair equity curves, with capital allocated uniformly across selected pairs.

## Architecture

The project is implemented as a set of independent modules, each with a single responsibility:

```
AlphaStream-KF/
├── main.py                  # End-to-end pipeline: scan → KF → backtest → portfolio
├── src/
│   ├── data_loader.py       # yfinance ingestion, forward-fill, C-contiguous arrays
│   ├── half_life.py         # ADF stationarity test, half-life estimation via OLS
│   ├── kalman_filter.py     # Streaming Kalman Filter with warm-start
│   ├── strategy.py          # Z-score computation, state-machine signal generation
│   ├── backtester.py        # Dollar-based P&L, slippage model, circuit breakers
│   ├── storage.py           # Parquet I/O via PyArrow (snappy compression)
│   └── plotting.py          # Visualisation utilities for the analysis notebook
├── notebooks/
│   └── analysis.ipynb       # Interactive results and analysis notebook
└── data/
    └── Tickers.json         # Asset universe
```

## Repository Structure

- `main.py` — orchestrates the full pipeline: cointegration scan, Kalman Filter estimation, signal generation, backtesting, and portfolio aggregation. Writes `pairs_metadata.json` at completion.
- `src/data_loader.py` — downloads adjusted close prices and volumes from Yahoo Finance, applies forward-fill to handle non-trading days, and returns memory-contiguous NumPy arrays.
- `src/half_life.py` — implements the ADF test via `statsmodels` and estimates the half-life via OLS regression on first differences.
- `src/kalman_filter.py` — manual NumPy implementation of the predict-update Kalman loop, supporting warm-start initialisation from the OLS prior.
- `src/strategy.py` — computes the rolling Z-score and runs the state-machine signal generator with adaptive position sizing.
- `src/backtester.py` — simulates dollar-based P&L with per-leg cash accounting, liquidity-adjusted slippage, and stop-loss/take-profit circuit breakers.
- `src/storage.py` — handles Parquet serialisation and deserialisation using PyArrow with Snappy compression.
- `src/plotting.py` — provides a `Plotter` class with functions for cointegration heatmaps, dynamic beta charts, spread/Z-score panels, trading signal overlays, equity curves, drawdown charts, and the performance summary table.
- `notebooks/analysis.ipynb` — loads results from `data/` and renders the full analysis.

## Requirements

```
python >= 3.11
numpy
pandas
matplotlib
scipy
statsmodels
yfinance
pyarrow
```

Install all dependencies:

```bash
pip install numpy pandas matplotlib scipy statsmodels yfinance pyarrow
```


## Usage

Run the full pipeline (cointegration scan + Kalman Filter + backtest + portfolio):

```bash
python main.py
```

The script will print progress at each step and save results to the `data/` directory. Once complete, open the analysis notebook:

```bash
jupyter notebook notebooks/analysis.ipynb
```

## Known Limitations

- **Short-selling constraints**: borrowing costs for the short leg are not modelled. For US large-cap equities these are typically 10–50 bps annually, but they can be significantly higher for hard-to-borrow names.
- **Settlement delays**: T+2 settlement is not explicitly enforced. In practice this introduces a two-day lag in available capital after closing a position.
- **Statistical significance of backtest results**: with 75 total trades across a 26-month test period, the distribution of Sharpe ratios has a wide confidence interval. The results are directionally meaningful but should not be interpreted as statistically conclusive without walk-forward validation across multiple non-overlapping test windows.
- **Spurious cointegration**: the ADF test identifies statistical properties of the training sample but cannot guarantee that the relationship persists out-of-sample. Pairs with no fundamental economic linkage may pass the test by coincidence, particularly in long training windows.

## License

MIT License.