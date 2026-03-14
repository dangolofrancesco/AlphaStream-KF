"""Kalman Filter for state-space tracking of time-varying parameters.

This module implements the standard linear Kalman Filter predict-update cycle,
suitable for estimating latent states (e.g. a time-varying hedge ratio β_t) in
non-stationary time series.

State-space formulation
-----------------------
State equation (process model):
    θ_t = F · θ_{t-1} + w_t,    w_t ~ N(0, Q)

Observation equation:
    y_t = H_t · θ_t + v_t,      v_t ~ N(0, R)

For pairs-trading with state θ_t = [β_t, α_t]^T:
    F   = I₂              (random-walk prior — no drift)
    H_t = [x_t, 1]        (observation matrix changes each step)
    Q   = diag(q, q)      (isotropic process noise)
    R   = r               (scalar observation noise)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class KalmanFilter:
    """Linear Kalman Filter for tracking a vector-valued latent state.

    Parameters
    ----------
    transition_matrix : array-like, shape (n, n)
        State transition matrix **F**.
    process_noise_cov : array-like, shape (n, n)
        Process noise covariance **Q**.
    observation_noise_var : float
        Scalar observation noise variance **R** (single-output model).
    initial_state : array-like, shape (n,)
        Initial state estimate **θ₀**.
    initial_state_cov : array-like, shape (n, n)
        Initial state covariance **P₀**.
    """

    def __init__(
        self,
        transition_matrix: np.ndarray,
        process_noise_cov: np.ndarray,
        observation_noise_var: float,
        initial_state: np.ndarray,
        initial_state_cov: np.ndarray,
    ) -> None:
        self.F = np.array(transition_matrix, dtype=float)
        self.Q = np.array(process_noise_cov, dtype=float)
        self.R = float(observation_noise_var)
        self.state = np.array(initial_state, dtype=float)
        self.P = np.array(initial_state_cov, dtype=float)

        n = self.state.shape[0]
        if self.F.shape != (n, n):
            raise ValueError(
                f"transition_matrix must have shape ({n}, {n}), "
                f"got {self.F.shape}."
            )
        if self.Q.shape != (n, n):
            raise ValueError(
                f"process_noise_cov must have shape ({n}, {n}), "
                f"got {self.Q.shape}."
            )
        if self.P.shape != (n, n):
            raise ValueError(
                f"initial_state_cov must have shape ({n}, {n}), "
                f"got {self.P.shape}."
            )
        if self.R <= 0:
            raise ValueError("observation_noise_var must be strictly positive.")

    # ------------------------------------------------------------------
    # Core predict / update steps
    # ------------------------------------------------------------------

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Propagate state and covariance forward one time step.

        Returns
        -------
        state_pred : np.ndarray, shape (n,)
            A priori state estimate θ_{t|t-1}.
        P_pred : np.ndarray, shape (n, n)
            A priori state covariance P_{t|t-1}.
        """
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state.copy(), self.P.copy()

    def update(
        self,
        observation: float,
        observation_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Incorporate a new scalar observation and update the state.

        Parameters
        ----------
        observation : float
            Scalar measurement y_t.
        observation_matrix : array-like, shape (n,)
            Row vector H_t for the current time step.

        Returns
        -------
        state : np.ndarray, shape (n,)
            A posteriori state estimate θ_{t|t}.
        P : np.ndarray, shape (n, n)
            A posteriori state covariance P_{t|t}.
        innovation : float
            Prediction error  e_t = y_t − H_t · θ_{t|t-1}.
        innovation_var : float
            Innovation variance  S_t = H_t · P_{t|t-1} · H_t^T + R.
        """
        H = np.array(observation_matrix, dtype=float).ravel()  # shape (n,)

        # Innovation: e_t = y_t - H_t · θ_{t|t-1}
        innovation = float(observation) - float(np.dot(H, self.state))

        # Innovation variance: S_t = H_t · P_{t|t-1} · H_t^T + R
        S = float(np.dot(H, self.P @ H)) + self.R

        # Kalman gain: K_t = P_{t|t-1} · H_t^T / S_t  → shape (n,)
        K = (self.P @ H) / S

        # State update: θ_{t|t} = θ_{t|t-1} + K_t · e_t
        self.state = self.state + K * innovation

        # Covariance update — Joseph stabilised form for numerical robustness
        n = self.state.shape[0]
        I_KH = np.eye(n) - np.outer(K, H)
        self.P = I_KH @ self.P @ I_KH.T + self.R * np.outer(K, K)

        return self.state.copy(), self.P.copy(), innovation, S
