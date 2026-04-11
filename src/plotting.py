"""
Visualization utilities for the Statistical Arbitrage project.
All functions accept pre-computed data (prices, spreads, signals, backtest results)
and return matplotlib Figure objects so they can be displayed in a Jupyter notebook
or saved to disk without any side effects.

Usage:
    from src.plotting import Plotter
    fig = Plotter.equity_curves(results_list, labels, initial_capital)
    fig.savefig("equity.png", dpi=150)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
from typing import Optional


# Colour palette — consistent across all plots
_PURPLE  = "#534AB7"
_TEAL    = "#0F6E56"
_AMBER   = "#BA7517"
_CORAL   = "#D85A30"
_GRAY    = "#888780"
_RED     = "#E24B4A"
_BLUE    = "#185FA5"

_PAIR_COLORS = [_PURPLE, _TEAL, _AMBER, _CORAL, _BLUE]

_BASE_STYLE = {
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#D3D1C7",
    "axes.linewidth":    0.6,
    "axes.grid":         True,
    "grid.color":        "#F1EFE8",
    "grid.linewidth":    0.5,
    "xtick.color":       "#5F5E5A",
    "ytick.color":       "#5F5E5A",
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.labelsize":    10,
    "axes.titlesize":    11,
    "axes.titleweight":  "medium",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "legend.frameon":    False,
    "legend.fontsize":   9,
    "font.family":       "sans-serif",
}

def _apply_style():
    plt.rcParams.update(_BASE_STYLE)

def _fmt_pct(ax, axis="y"):
    """Format an axis as percentages."""
    fmt = mticker.FuncFormatter(lambda x, _: f"{x*100:.1f}%")
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)

def _fmt_dollar(ax):
    """Format y-axis as dollar amounts."""
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )

def _date_axis(ax):
    """Apply a clean date format to the x-axis."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")


# SECTION 1 — Pair Selection & Cointegration

class Plotter:

    @staticmethod
    def cointegration_heatmap(
        pvalue_matrix: np.ndarray,
        tickers: list[str],
        threshold: float = 0.05,
        figsize: tuple = (12, 10),
    ) -> plt.Figure:
        """
        Heatmap of Engle-Granger p-values for all ticker pairs.
        Cells above `threshold` are masked (grey) — only significant pairs are coloured.
        The colour scale runs from dark (p ≈ 0, strong cointegration) to light (p ≈ threshold).
        """
        _apply_style()
        fig, ax = plt.subplots(figsize=figsize)

        n = len(tickers)
        masked = np.where(pvalue_matrix < threshold, pvalue_matrix, np.nan)

        im = ax.imshow(masked, cmap="Blues_r", vmin=0, vmax=threshold, aspect="auto")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(tickers, fontsize=8)

        # Annotate cells with p-values where significant
        for i in range(n):
            for j in range(n):
                val = pvalue_matrix[i, j]
                if val < threshold and i != j:
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=7, color="white" if val < threshold / 2 else "#3C3489")

        cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("ADF p-value", fontsize=9)
        ax.set_title(f"Cointegration p-value matrix  (shaded = p < {threshold})", pad=12)
        fig.tight_layout()
        return fig

    @staticmethod
    def pairs_summary_table(
        candidates: list[dict],
        figsize: tuple = (10, 0.5),
    ) -> plt.Figure:
        """
        Styled table summarising the top cointegrated pairs selected for trading.
        Columns: rank, pair, p-value, half-life (days), OLS beta (initial KF prior).
        """
        _apply_style()
        rows = []
        for i, c in enumerate(candidates, 1):
            y, x = c["pair"]
            rows.append([
                i,
                f"{y} / {x}",
                f"{c['p_value']:.5f}",
                f"{c['half_life']:.1f}",
                f"{c['beta_init']:.4f}",
            ])

        cols = ["Rank", "Pair (Y / X)", "ADF p-value", "Half-life (days)", "β OLS (init)"]
        height = max(2.0, 0.5 + 0.4 * len(rows))
        fig, ax = plt.subplots(figsize=(figsize[0], height))
        ax.axis("off")

        tbl = ax.table(
            cellText=rows, colLabels=cols, loc="center", cellLoc="center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1, 1.6)

        for (r, c_), cell in tbl.get_celld().items():
            cell.set_edgecolor("#D3D1C7")
            if r == 0:
                cell.set_facecolor(_PURPLE)
                cell.set_text_props(color="white", fontweight="medium")
            elif r % 2 == 0:
                cell.set_facecolor("#F1EFE8")
            else:
                cell.set_facecolor("white")

        ax.set_title("Selected cointegrated pairs — training period", pad=8, fontsize=11, fontweight="medium")
        fig.tight_layout()
        return fig

    @staticmethod
    def normalized_prices(
        y_prices: np.ndarray,
        x_prices: np.ndarray,
        dates: pd.DatetimeIndex,
        asset_y: str,
        asset_x: str,
        figsize: tuple = (12, 4),
    ) -> plt.Figure:
        """
        Normalised price series (base 100) for both assets in a pair.
        Visual divergence between the two lines is what creates the spread we exploit.
        """
        _apply_style()
        fig, ax = plt.subplots(figsize=figsize)

        y_norm = 100 * y_prices / y_prices[0]
        x_norm = 100 * x_prices / x_prices[0]

        ax.plot(dates, y_norm, color=_PURPLE, linewidth=1.2, label=asset_y)
        ax.plot(dates, x_norm, color=_TEAL,   linewidth=1.2, label=asset_x, linestyle="--")
        ax.axhline(100, color=_GRAY, linewidth=0.5, linestyle=":")
        ax.set_ylabel("Normalised price (base = 100)")
        ax.set_title(f"Normalised price series — {asset_y} vs {asset_x}")
        ax.legend()
        _date_axis(ax)
        fig.tight_layout()
        return fig


# SECTION 2 — Kalman Filter & Spread

    @staticmethod
    def dynamic_beta(
        betas: np.ndarray,
        beta_ols: float,
        dates: pd.DatetimeIndex,
        asset_y: str,
        asset_x: str,
        figsize: tuple = (12, 4),
    ) -> plt.Figure:
        """
        Kalman Filter beta estimate over the test period vs the static OLS beta.
        The KF beta drifts over time — this shows why a dynamic hedge ratio is needed.
        A stable KF beta close to the OLS value suggests a robust cointegration relationship.
        """
        _apply_style()
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(dates, betas, color=_PURPLE, linewidth=1.2, label="KF beta (dynamic)")
        ax.axhline(beta_ols, color=_AMBER, linewidth=1.0, linestyle="--",
                   label=f"OLS beta (static) = {beta_ols:.3f}")
        ax.set_ylabel("Hedge ratio β")
        ax.set_title(f"Dynamic hedge ratio — {asset_y} vs {asset_x}")
        ax.legend()
        _date_axis(ax)
        fig.tight_layout()
        return fig

    @staticmethod
    def spread_and_zscore(
        spread: np.ndarray,
        zscore: np.ndarray,
        dates: pd.DatetimeIndex,
        asset_y: str,
        asset_x: str,
        entry_z: float = 1.0,
        exit_z: float = 0.5,
        figsize: tuple = (12, 6),
    ) -> plt.Figure:
        """
        Two-panel chart: (top) KF innovation spread; (bottom) rolling Z-score with
        entry/exit bands. Shaded regions highlight when the strategy is in a position.
        The spread is the raw innovation e_t = y_t - (β_t · x_t + α_t) from the KF.
        The Z-score normalises it using a rolling window equal to the half-life.
        """
        _apply_style()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Spread
        ax1.plot(dates, spread, color=_PURPLE, linewidth=0.8, alpha=0.9)
        ax1.axhline(0, color=_GRAY, linewidth=0.5, linestyle=":")
        ax1.set_ylabel("KF spread (innovation)")
        ax1.set_title(f"Spread and Z-score — {asset_y} vs {asset_x}")

        # Z-score with bands
        ax2.plot(dates, zscore, color=_TEAL, linewidth=0.9, alpha=0.9, label="Z-score")
        ax2.axhline( entry_z, color=_CORAL, linewidth=0.8, linestyle="--", label=f"+{entry_z}σ entry")
        ax2.axhline(-entry_z, color=_CORAL, linewidth=0.8, linestyle="--", label=f"-{entry_z}σ entry")
        ax2.axhline( exit_z,  color=_GRAY,  linewidth=0.6, linestyle=":",  label=f"±{exit_z}σ exit")
        ax2.axhline(-exit_z,  color=_GRAY,  linewidth=0.6, linestyle=":")
        ax2.axhline(0, color=_GRAY, linewidth=0.4)
        ax2.fill_between(dates, entry_z, zscore,
                         where=(zscore >= entry_z), alpha=0.12, color=_CORAL)
        ax2.fill_between(dates, -entry_z, zscore,
                         where=(zscore <= -entry_z), alpha=0.12, color=_BLUE)
        ax2.set_ylabel("Z-score")
        ax2.legend(loc="upper right", ncol=2)
        _date_axis(ax2)
        fig.tight_layout()
        return fig


# SECTION 3 — Strategy & Signals

    @staticmethod
    def trading_signals(
        y_prices: np.ndarray,
        signals_df: pd.DataFrame,
        dates: pd.DatetimeIndex,
        asset_y: str,
        asset_x: str,
        figsize: tuple = (12, 4),
    ) -> plt.Figure:
        """
        Price series of asset Y with buy/sell entry markers and exit markers.
        Long-spread entries (long Y) are green triangles up.
        Short-spread entries (short Y) are red triangles down.
        Exits are grey dots. This makes it easy to see whether entries precede profitable moves.
        """
        _apply_style()
        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(dates, y_prices, color=_GRAY, linewidth=0.8, alpha=0.7, label=asset_y, zorder=1)

        pos = signals_df["Actual_Position"].values if "Actual_Position" in signals_df.columns \
              else signals_df["Target_Position"].values

        pos_series = pd.Series(pos, index=dates)
        changes = pos_series.diff().fillna(pos_series)

        long_entries  = dates[changes == 1]
        short_entries = dates[changes == -1]
        exits         = dates[(changes != 0) & (pos_series == 0)]

        prices_s = pd.Series(y_prices, index=dates)

        ax.scatter(long_entries,  prices_s.loc[long_entries],
                   marker="^", color=_TEAL, s=60, zorder=3, label="Long entry (long Y)")
        ax.scatter(short_entries, prices_s.loc[short_entries],
                   marker="v", color=_CORAL, s=60, zorder=3, label="Short entry (short Y)")
        ax.scatter(exits, prices_s.loc[exits],
                   marker="o", color=_GRAY, s=25, zorder=2, label="Exit", alpha=0.7)

        ax.set_ylabel(f"{asset_y} price")
        ax.set_title(f"Trading signals — {asset_y} vs {asset_x}")
        ax.legend(ncol=3)
        _date_axis(ax)
        fig.tight_layout()
        return fig

    @staticmethod
    def position_and_zscore(
        zscore: np.ndarray,
        signals_df: pd.DataFrame,
        dates: pd.DatetimeIndex,
        asset_y: str,
        asset_x: str,
        entry_z: float = 1.0,
        figsize: tuple = (12, 4),
    ) -> plt.Figure:
        """
        Z-score line with the active position shown as a coloured background.
        Green shading = long spread (long Y / short X).
        Red shading = short spread (short Y / long X).
        Flat periods have no shading. Useful for checking signal timing vs spread extremes.
        """
        _apply_style()
        fig, ax = plt.subplots(figsize=figsize)

        pos = signals_df["Actual_Position"].values if "Actual_Position" in signals_df.columns \
              else signals_df["Target_Position"].values

        ax.fill_between(dates, -entry_z * 2, entry_z * 2,
                        where=(pos > 0), alpha=0.15, color=_TEAL, label="Long spread")
        ax.fill_between(dates, -entry_z * 2, entry_z * 2,
                        where=(pos < 0), alpha=0.15, color=_CORAL, label="Short spread")

        ax.plot(dates, zscore, color=_PURPLE, linewidth=0.9, zorder=2)
        ax.axhline( entry_z, color=_CORAL, linewidth=0.7, linestyle="--")
        ax.axhline(-entry_z, color=_CORAL, linewidth=0.7, linestyle="--")
        ax.axhline(0, color=_GRAY, linewidth=0.4)

        ax.set_ylabel("Z-score")
        ax.set_title(f"Position timing vs Z-score — {asset_y} vs {asset_x}")
        ax.legend(loc="upper right")
        _date_axis(ax)
        fig.tight_layout()
        return fig


# SECTION 4 — Backtest Performance

    @staticmethod
    def equity_curves(
        results_list: list[pd.DataFrame],
        labels: list[str],
        initial_capital: float,
        portfolio_equity: Optional[pd.Series] = None,
        figsize: tuple = (12, 5),
    ) -> plt.Figure:
        """
        Equity curves for each individual pair plus the combined portfolio.
        The portfolio equity (sum of all pairs) is shown as a thicker black line.
        All curves are re-based to start at the same value so relative performance is comparable.
        The horizontal dashed line marks the initial capital (break-even reference).
        """
        _apply_style()
        fig, ax = plt.subplots(figsize=figsize)

        for df, label, color in zip(results_list, labels, _PAIR_COLORS):
            ax.plot(df.index, df["Equity_Curve"], color=color,
                    linewidth=0.9, alpha=0.7, label=label)

        if portfolio_equity is not None:
            ax.plot(portfolio_equity.index, portfolio_equity,
                    color="black", linewidth=1.8, label="Combined portfolio", zorder=5)

        ax.axhline(initial_capital, color=_GRAY, linewidth=0.6,
                   linestyle="--", label="Initial capital")
        _fmt_dollar(ax)
        ax.set_ylabel("Portfolio value ($)")
        ax.set_title("Equity curves — individual pairs and combined portfolio")
        ax.legend(ncol=2)
        _date_axis(ax)
        fig.tight_layout()
        return fig

    @staticmethod
    def drawdown_chart(
        portfolio_equity: pd.Series,
        figsize: tuple = (12, 4),
    ) -> plt.Figure:
        """
        Drawdown from peak for the combined portfolio equity curve.
        Drawdown is defined as (equity / rolling_max) - 1.
        The shaded area is coloured by severity — shallow drawdowns are amber, deep are red.
        Maximum drawdown is annotated on the chart.
        """
        _apply_style()
        fig, ax = plt.subplots(figsize=figsize)

        rolling_max = portfolio_equity.cummax()
        drawdown = (portfolio_equity / rolling_max - 1.0) * 100

        ax.fill_between(portfolio_equity.index, drawdown, 0,
                        color=_RED, alpha=0.35, label="Drawdown")
        ax.plot(portfolio_equity.index, drawdown, color=_RED, linewidth=0.7)

        mdd_val = drawdown.min()
        mdd_date = drawdown.idxmin()
        ax.annotate(f"Max DD: {mdd_val:.1f}%",
                    xy=(mdd_date, mdd_val),
                    xytext=(10, 10), textcoords="offset points",
                    fontsize=9, color=_RED,
                    arrowprops=dict(arrowstyle="->", color=_RED, lw=0.8))

        ax.axhline(0, color=_GRAY, linewidth=0.5)
        ax.set_ylabel("Drawdown (%)")
        ax.set_title("Portfolio drawdown from peak")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
        _date_axis(ax)
        fig.tight_layout()
        return fig

    @staticmethod
    def return_distribution(
        portfolio_equity: pd.Series,
        figsize: tuple = (10, 5),
    ) -> plt.Figure:
        """
        Histogram of daily portfolio returns with a kernel density estimate (KDE) overlay
        and a theoretical normal distribution fitted to the same mean and std.
        Fat tails (excess kurtosis) appear as histogram bars extending beyond the normal curve.
        VaR at 5% is marked with a vertical line.
        """
        _apply_style()
        fig, ax = plt.subplots(figsize=figsize)

        returns = portfolio_equity.pct_change().dropna() * 100  # in %

        ax.hist(returns, bins=60, color=_PURPLE, alpha=0.55, density=True, label="Daily returns")

        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 300)
        ax.plot(x, norm.pdf(x, mu, sigma),
                color=_CORAL, linewidth=1.5, linestyle="--", label="Normal fit")

        # KDE via Gaussian smoothing (manual, no scipy dependency)
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(returns)
        ax.plot(x, kde(x), color=_TEAL, linewidth=1.5, label="KDE")

        var_5 = float(np.percentile(returns, 5))
        ax.axvline(var_5, color=_RED, linewidth=1.0, linestyle=":",
                   label=f"VaR (95%) = {var_5:.2f}%")

        kurt = returns.kurtosis()
        ax.set_xlabel("Daily return (%)")
        ax.set_ylabel("Density")
        ax.set_title(f"Return distribution  |  μ={mu:.3f}%  σ={sigma:.3f}%  kurtosis={kurt:.1f}")
        ax.legend()
        fig.tight_layout()
        return fig

    @staticmethod
    def performance_table(
        results_list: list[pd.DataFrame],
        labels: list[str],
        initial_capital_per_pair: float,
        portfolio_equity: Optional[pd.Series] = None,
        total_capital: Optional[float] = None,
        figsize: tuple = (14, 0.5),
    ) -> plt.Figure:
        """
        Summary performance table: one row per pair plus a combined portfolio row.
        Metrics: Total Return, Ann. Return, Ann. Vol, Sharpe, Sortino, Calmar, Max DD,
        VaR (95%), CVaR (95%), Win Rate, Total Trades.
        Green/red colouring on the Return column highlights profitable vs losing pairs.
        """
        _apply_style()

        def _compute_row(equity: pd.Series, capital: float, label: str) -> list:
            net_ret = equity.pct_change().fillna(0)
            total_r  = (equity.iloc[-1] / capital) - 1
            ann_r    = (1 + total_r) ** (252 / len(equity)) - 1
            ann_vol  = net_ret.std() * np.sqrt(252)
            dm, dv   = net_ret.mean(), net_ret.std()
            sharpe   = (dm / dv * np.sqrt(252)) if dv > 0 else 0
            ds_vol   = net_ret[net_ret < 0].std() * np.sqrt(252)
            sortino  = (dm / ds_vol * np.sqrt(252)) if ds_vol > 0 else 0
            mdd      = ((equity / equity.cummax()) - 1).min()
            calmar   = ann_r / abs(mdd) if mdd < 0 else float("inf")
            var95    = np.percentile(net_ret, 5)
            cvar95   = net_ret[net_ret <= var95].mean() if (net_ret <= var95).any() else var95
            active   = net_ret[net_ret != 0]
            win_rate = (active > 0).sum() / len(active) if len(active) > 0 else 0
            trades   = int(equity.pct_change().ne(0).sum())
            return [
                label,
                f"{total_r*100:+.2f}%",
                f"{ann_r*100:+.2f}%",
                f"{ann_vol*100:.2f}%",
                f"{sharpe:.2f}",
                f"{sortino:.2f}",
                f"{calmar:.2f}",
                f"{mdd*100:.2f}%",
                f"{var95*100:.2f}%",
                f"{cvar95*100:.2f}%",
                f"{win_rate*100:.1f}%",
                str(trades),
            ]

        cols = ["Pair", "Total ret.", "Ann. ret.", "Ann. vol",
                "Sharpe", "Sortino", "Calmar",
                "Max DD", "VaR 95%", "CVaR 95%", "Win rate", "Trades"]

        rows = [_compute_row(df["Equity_Curve"], initial_capital_per_pair, lbl)
                for df, lbl in zip(results_list, labels)]

        if portfolio_equity is not None and total_capital is not None:
            rows.append(_compute_row(portfolio_equity, total_capital, "PORTFOLIO"))

        height = max(2.0, 0.55 + 0.42 * len(rows))
        fig, ax = plt.subplots(figsize=(figsize[0], height))
        ax.axis("off")

        tbl = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1, 1.55)

        for (r, c_), cell in tbl.get_celld().items():
            cell.set_edgecolor("#D3D1C7")
            if r == 0:
                cell.set_facecolor(_PURPLE)
                cell.set_text_props(color="white", fontweight="medium")
            elif r == len(rows):           # portfolio row — bold
                cell.set_facecolor("#F1EFE8")
                cell.set_text_props(fontweight="medium")
            elif r % 2 == 0:
                cell.set_facecolor("#F8F7F4")
            else:
                cell.set_facecolor("white")
            # Colour the return column
            if r > 0 and c_ == 1:
                val = rows[r - 1][1]
                if val.startswith("+"):
                    cell.set_text_props(color=_TEAL)
                elif val.startswith("-"):
                    cell.set_text_props(color=_RED)

        ax.set_title("Performance metrics summary", pad=8, fontsize=11, fontweight="medium")
        fig.tight_layout()
        return fig
