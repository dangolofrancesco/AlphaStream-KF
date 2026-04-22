# Statistical Arbitrage with Kalman Filter

In this repository, we delve into the application of institutional-grade statistical arbitrage (StatArb) within the dynamic and highly efficient landscape of large-cap US Equities. Acknowledging the distinct challenges that modern markets present—such as rapid regime shifts, changing interest rate cycles, and spurious cointegration. This investigation employs a robust dataset focused on tightly coupled intra-sector assets (e.g., Mega-cap Tech, Energy Supermajors, Systemic Banks). The code elucidates how an adaptive, mathematically rigorous statistical arbitrage engine can navigate market inefficiencies while strictly managing risk and execution costs.

## Methodology
The methodology is comprehensive, moving beyond naive mean-reversion models to implement a "Pure Kalman" adaptive framework. It emphasizes economic pre-filtering, rigorous cointegration testing, and advanced tools like the Kalman Filter with decoupled process variances to identify stable relationships between assets. The approach is further refined by a filter-theoretic z-score, disjoint portfolio selection to ensure true market neutrality, and fractional position sizing locked at entry. This repo bridges theoretical econometrics with practical hedge-fund execution tactics, rendering it a crucial tool for quantitative research in fast-evolving financial markets.

## Data Split
The dataset was divided into two distinct subsets: training (in-sample) and testing (out-of-sample) periods. The training set spans from 2011 to 2016, providing a substantial amount of data for model building and establishing baseline historical relationships. The testing set covers the 2017-2018 period. This approach provides a substantial amount of data for model building while reserving a significant portion for out-of-sample testing to validate the model’s predictive power.

## Pair Selection
Pair selection for the statistical arbitrage strategy was based on the concept of cointegration, rather than correlation. While correlation measures the strength and direction of a linear relationship between two variables, it does not imply a stable, long-term relationship. Cointegration, on the other hand, suggests that two or more time series, despite being non-stationary individually, can form a stationary combination, indicating a long-term equilibrium relationship. Pairs were chosen based on their cointegration following a multi-layered pipeline:
1. Economic Pre-filtering: Assets are rigorously grouped by sector and fundamental business models (e.g., Payment Processors, Semicondutors) to ensure that statistical cointegration is backed by true economic linkage.
2. Recent-Window Stationarity: Rather than falling into the "10-year OLS trap" where long-term averages mask short-term divergences, the Augmented Dickey-Fuller (ADF) test is applied only to the most recent 252 trading days of the training set. A p-value threshold of less than $0.05$ confirms a statistically significant.
3. Half-Life Filtering: The speed of mean reversion is calculated. Pairs must exhibit a half-life between 5 and 30 days to ensure they are tradable at a realistic frequency.
4. Disjoint Sets (Risk Management): A greedy algorithm selects the top pairs while strictly enforcing disjoint sets (no single ticker can appear in more than one pair). This prevents portfolio over-exposure and risk concentration (e.g., inadvertently building a massive net-long position on a single asset).

## Spread Calculation
A detailed process was implemented to calculate the spread between asset pairs, utilizing Kalman-filtered beta values and the half-life of mean reversion. First, all asset prices are transformed into log-prices to normalize volatility and ensure parameters scale correctly across assets of different monetary values. The strategy utilizes a Kalman Filter to dynamically estimate the hedge ratio ($\beta$) and the intercept ($\alpha$). Crucially, the process noise covariance matrix ($Q$) is decoupled: $\beta$ is assigned an extremely slow variance ($10^{-6}$) to reflect the sticky fundamental relationship between companies, while $\alpha$ is tuned ($10^{-6}$) to prevent "model catch-up" where the filter simply forgives market deviations.The trading signal is derived from the pre-update innovation ($e_t$), representing the true "surprise" or prediction error of the market against yesterday's state, rather than the post-fit residual which algebraically absorbs the anomaly.

## Signal Generation
A z-score-based approach was incorporated for making trading decisions. Instead of relying on static rolling windows that fail during volatility shocks, the z-score is calculated directly within the Kalman Filter as $Z_t = e_t / \sqrt{S_t}$, standardizing the innovation by the filter's own dynamic measurement of uncertainty. Position sizing is dynamically adjusted according to the magnitude of the z-scores (e.g., multipliers of 0.5, 0.75, or 1.0 based on anomaly severity) where larger positions are taken when higher z-scores indicate significant deviations from the mean, and lower z-scores lead to more conservative positions. Crucially, the position size is locked at entry. This prevents the pathology of scaling out of a winning trade as it reverts to the mean, allowing the strategy to capture the full economic edge of the Ornstein-Uhlenbeck process. Trading signals are generated based on these scores: buying the spread when the z-score drops below $-2.0$ and shorting it when it exceeds $+2.0$. Exits are strictly targeted at the absolute mean ($0.0$) to maximize profit per trade. Furthermore, a Structural Break Hard-Stop is implemented: if the z-score explodes beyond $\pm 3.5$, the system recognizes a permanent regime shift and liquidates the position to prevent catastrophic drawdowns.

## Back-Testing
A pairs trading approach was simulated using historical data, considering market realities such as transaction costs, liquidity, stop-loss, and take-profit thresholds. The strategy starts with a specified capital of $100,000, dynamically allocating capital based on the signal's strength multiplier to ensure dollar-neutrality across the long and short legs. To strictly prevent Look-Ahead Bias, execution is delayed by one period ($T+1$). Transaction costs are aggressively factored in at a base rate of 5 basis points. Furthermore, a dynamic slippage model evaluates the relative liquidity of the trading day: slippage is penalized heavily if the requested share size exceeds the 25th to 75th percentile of the asset's rolling historical trading volume. The simulation continuously updates cash positions, applying circuit breakers such as a 10% portfolio-level stop-loss to mitigate tail risks, and a 20% take-profit. The final out-of-sample evaluation assesses the terminal wealth and risk-adjusted metrics (Sharpe Ratio, Maximum Drawdown, Excess Kurtosis). Despite the friction of realistic costs and a hostile market regime, the refined disjoint portfolio demonstrated severe drawdown reduction (e.g., mitigating a potential -46% loss on semiconductor pairs down to -12%) and generated consistent, uncorrelated absolute returns.


## Repository Structure
- `main.py` — orchestrates the full pipeline: cointegration scan, Kalman Filter estimation, signal generation, backtesting, and portfolio aggregation.
- `src/data_loader.py` — downloads adjusted close prices and volumes from Yahoo Finance, applies forward-fill to handle non-trading days, and returns memory-contiguous NumPy arrays.
- `src/half_life.py` — implements the ADF test and estimates the half-life via OLS regression on first differences.
- `src/kalman_filter.py` — manual NumPy implementation of the predict-update Kalman loop, supporting warm-start initialization from the OLS prior.
- `src/strategy.py` — computes the Z-score and runs the signal generator with adaptive position sizing.
- `src/backtester.py` — simulates dollar-based P&L with per-leg cash accounting, liquidity-adjusted slippage, and stop-loss/take-profit circuit breakers.
- `src/storage.py` — handles Parquet serialization and deserialization using PyArrow with Snappy compression.
- `src/plotting.py` — provides a Plotter class with functions for cointegration heatmaps, dynamic beta charts, spread/Z-score panels, trading signal overlays, equity curves, drawdown charts, and the performance summary table.
- `notebooks/analysis.ipynb` — loads results from data/ and renders the full analysis.

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

## Usage

Run the full pipeline (cointegration scan + Kalman Filter + backtest + portfolio):

```bash
python main.py
```

The script will print progress at each step and save results to the `data/` directory. Once complete, open the analysis notebook:

```bash
jupyter notebook notebooks/analysis.ipynb
```

## License

MIT License.
