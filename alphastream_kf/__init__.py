"""AlphaStream-KF: Kalman Filter Statistical Arbitrage.

A state-space framework for statistical arbitrage that replaces static OLS
regression with a Kalman Filter to track the latent, time-varying hedge ratio
β_t in non-stationary price series.
"""

from .kalman_filter import KalmanFilter
from .stat_arb import KalmanFilterStatArb

__all__ = ["KalmanFilter", "KalmanFilterStatArb"]
