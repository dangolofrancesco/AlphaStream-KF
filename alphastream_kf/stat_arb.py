"""Kalman Filter-based Statistical Arbitrage.

Replaces the static OLS regression used in classical pairs-trading with a
Kalman Filter (state-space model) that continuously re-estimates the latent
hedge ratio β_t and intercept α_t as they drift over time.

Model
-----
The log-price (or price) of asset *y* is modelled as a linear function of
asset *x* with time-varying coefficients::

    y_t = β_t · x_t + α_t + ε_t,    ε_t ~ N(0, R)

The hidden state **θ_t = [β_t, α_t]^T** follows a random walk::

    θ_t = θ_{t-1} + w_t,             w_t ~ N(0, Q)

This yields a Kalman Filter with:
* Transition matrix  F = I₂
* Observation matrix H_t = [x_t, 1]  (updated at every step)
* Process noise      Q = q · I₂
* Observation noise  R = r  (scalar)

The synthetic spread is the observation residual after KF update::

    s_t = y_t − β_t · x_t − α_t

Trading signals are generated from the z-score of s_t, normalised by
√S_t (the square root of the innovation variance as a proxy for spread
volatility), following standard z-score mean-reversion rules.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .kalman_filter import KalmanFilter


class KalmanFilterStatArb:
    """State-space Statistical Arbitrage via a Kalman Filter hedge ratio.

    Parameters
    ----------
    process_noise : float
        Diagonal value of the process noise covariance **Q** (controls how
        quickly β_t and α_t are allowed to drift).  Smaller values produce
        smoother, slower-adapting estimates; larger values track faster.
    observation_noise : float
        Scalar observation noise variance **R**.
    initial_state_cov : float
        Diagonal value of the initial state covariance **P₀**.  A large
        value expresses high initial uncertainty and lets the filter
        converge quickly from cold-start.
    entry_threshold : float
        |z-score| level at which a new position is opened.  Default 2.0.
    exit_threshold : float
        |z-score| level at which an existing position is closed.  Default 0.5.

    Attributes
    ----------
    betas\_ : np.ndarray, shape (T,)
        Filtered hedge-ratio estimates β_{t|t}.
    alphas\_ : np.ndarray, shape (T,)
        Filtered intercept estimates α_{t|t}.
    spreads\_ : np.ndarray, shape (T,)
        Synthetic spread  s_t = y_t − β_t · x_t − α_t.
    zscores\_ : np.ndarray, shape (T,)
        Normalised spread  z_t = e_t / √S_t.
    """

    def __init__(
        self,
        process_noise: float = 1e-4,
        observation_noise: float = 1e-2,
        initial_state_cov: float = 1.0,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
    ) -> None:
        if process_noise <= 0:
            raise ValueError("process_noise must be strictly positive.")
        if observation_noise <= 0:
            raise ValueError("observation_noise must be strictly positive.")
        if initial_state_cov <= 0:
            raise ValueError("initial_state_cov must be strictly positive.")
        if entry_threshold <= exit_threshold:
            raise ValueError("entry_threshold must be greater than exit_threshold.")

        self.process_noise = process_noise
        self.observation_noise = observation_noise
        self.initial_state_cov = initial_state_cov
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

        # Fitted attributes — populated by fit()
        self.betas_: Optional[np.ndarray] = None
        self.alphas_: Optional[np.ndarray] = None
        self.spreads_: Optional[np.ndarray] = None
        self.zscores_: Optional[np.ndarray] = None
        self._innovations: Optional[np.ndarray] = None
        self._innovation_vars: Optional[np.ndarray] = None
        self._kf: Optional[KalmanFilter] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, y: np.ndarray, x: np.ndarray) -> "KalmanFilterStatArb":
        """Run the Kalman Filter over the full price history.

        Parameters
        ----------
        y : array-like, shape (T,)
            Price (or log-price) series of the *dependent* asset.
        x : array-like, shape (T,)
            Price (or log-price) series of the *independent* asset.

        Returns
        -------
        self
            The fitted estimator (enables method chaining).
        """
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)

        if y.ndim != 1 or x.ndim != 1:
            raise ValueError("y and x must be one-dimensional arrays.")
        if y.shape != x.shape:
            raise ValueError(
                f"y and x must have the same length; got {len(y)} and {len(x)}."
            )

        T = len(y)
        kf = self._build_kalman_filter()

        betas = np.empty(T)
        alphas = np.empty(T)
        spreads = np.empty(T)
        innovations = np.empty(T)
        innovation_vars = np.empty(T)

        for t in range(T):
            kf.predict()
            H = np.array([x[t], 1.0])
            _, _, innov, innov_var = kf.update(y[t], H)

            betas[t] = kf.state[0]
            alphas[t] = kf.state[1]
            spreads[t] = y[t] - kf.state[0] * x[t] - kf.state[1]
            innovations[t] = innov
            innovation_vars[t] = innov_var

        self._kf = kf
        self.betas_ = betas
        self.alphas_ = alphas
        self.spreads_ = spreads
        self._innovations = innovations
        self._innovation_vars = innovation_vars
        self.zscores_ = innovations / np.sqrt(np.maximum(innovation_vars, 1e-12))

        return self

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def get_hedge_ratios(self) -> np.ndarray:
        """Return a copy of the filtered hedge-ratio series β_{t|t}."""
        self._check_fitted()
        return self.betas_.copy()

    def get_intercepts(self) -> np.ndarray:
        """Return a copy of the filtered intercept series α_{t|t}."""
        self._check_fitted()
        return self.alphas_.copy()

    def get_spread(self) -> np.ndarray:
        """Return the synthetic spread  s_t = y_t − β_t · x_t − α_t."""
        self._check_fitted()
        return self.spreads_.copy()

    def get_zscore(self) -> np.ndarray:
        """Return the normalised spread z-score  z_t = e_t / √S_t."""
        self._check_fitted()
        return self.zscores_.copy()

    def generate_signals(self) -> np.ndarray:
        """Generate mean-reversion trading signals from the spread z-score.

        Rules
        -----
        * Open **long** spread  (+1) when z_t < −entry_threshold
        * Open **short** spread (−1) when z_t >  entry_threshold
        * Close position (0)        when |z_t| < exit_threshold

        Returns
        -------
        signals : np.ndarray of int, shape (T,)
            +1 = long spread, −1 = short spread, 0 = flat.
        """
        self._check_fitted()
        z = self.zscores_
        T = len(z)
        signals = np.zeros(T, dtype=int)
        position = 0

        for t in range(T):
            if position == 0:
                if z[t] < -self.entry_threshold:
                    position = 1   # long spread
                elif z[t] > self.entry_threshold:
                    position = -1  # short spread
            else:
                if abs(z[t]) < self.exit_threshold:
                    position = 0
                elif position == 1 and z[t] > self.entry_threshold:
                    position = -1  # flip
                elif position == -1 and z[t] < -self.entry_threshold:
                    position = 1   # flip
            signals[t] = position

        return signals

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_kalman_filter(self) -> KalmanFilter:
        """Construct and return a fresh :class:`KalmanFilter` instance."""
        n = 2  # state: [beta, alpha]
        return KalmanFilter(
            transition_matrix=np.eye(n),
            process_noise_cov=self.process_noise * np.eye(n),
            observation_noise_var=self.observation_noise,
            initial_state=np.zeros(n),
            initial_state_cov=self.initial_state_cov * np.eye(n),
        )

    def _check_fitted(self) -> None:
        """Raise :class:`RuntimeError` if :meth:`fit` has not been called."""
        if self.betas_ is None:
            raise RuntimeError(
                "This KalmanFilterStatArb instance is not fitted yet. "
                "Call fit(y, x) before accessing results."
            )
