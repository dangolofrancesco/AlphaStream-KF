import numpy as np
import pandas as pd
from typing import Tuple

class KalmanFilterPairs:
    """
    Dynamic estimation of the Hedge Ratio (Beta) using a Kalman Filter for pairs trading.
    Designed for fast execution using NumPy contiguous arrays. 

    Q (2x2) and R (scalar) are provided by the orchestrator
    and kept fixed during the whole backtest run.
    """

    def __init__(self, 
                 R: float = None, 
                 Q: np.ndarray = None,
                 observation_variance: float = 1e-3, 
                 delta_beta: float = 1e-5,    # Slow mean reversion for Beta
                 delta_alpha: float = 1e-3):  # Fast adaptation for Intercept (Drift)
        """
        Initializes the covariance matrices.

        Args:
            observation_variance (R): Noise in the daily price observation. 
            process_variance (Q):     Daily variability of the true hedge ratio. 
            observation_varaiance: Fallback for R if not provided directly.
            delta_beta: Fallback for Q[0, 0] if Q not provided (controls how quickly Beta can change).
            delta_alpha: Fallback for Q[1, 1] if Q not provided (controls how quickly Alpha can change).
        """
        self.state_dim = 2 # state = [beta, alpha]

        self.R = R if R is not None else observation_variance
        self.Q = Q if Q is not None else np.diag([delta_beta, delta_alpha])


    def filter(
        self,
        y_prices: np.ndarray,
        x_prices: np.ndarray,
        beta_init: float = None,
        alpha_init: float = None,
        use_log_prices: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies the recursive Predict-Update loop of the Kalman Filter
        
        Args:
            y_prices: 1D countigous array of the dependent variable (e.g., KO prices).
            x_prices: 1D contiguous array of the independent variable (e.g., PEP prices).
            beta_init: Optional initial value for Beta (hedge ratio) to speed up convergence.
            alpha_init: Optional initial value for Alpha (intercept) to speed up convergence.
            use_log_prices: If True, the filter will operate on log-prices instead of raw

        Returns:
            Tupe containing:
                - state_means: 2D array of shape (T, 2) with [beta_t, alpha_t] estimates at each time step.
                - innovations: 1D array (T,)containing the innovation e_t (signal for trading) at each time step.
                - innovation_vars: 1D array (T,) containing the variance S_t of the innovation at each time step
        """

        y = np.asarray(y_prices, dtype=np.float64).flatten()
        x = np.asarray(x_prices, dtype=np.float64).flatten()

        if use_log_prices:
            y = np.log(y)
            x = np.log(x)   

        T = len(y)

        # Pre-allocate memory for outputs for better performance
        state_means = np.zeros((T, self.state_dim))  
        innovations = np.zeros(T)  
        innovation_vars = np.zeros(T) 

        # 1. Initialization (t=0)
        if beta_init is not None:
            a0 = alpha_init if alpha_init is not None else 0.0
            theta = np.array([[beta_init], [a0]])  # [beta_OLS, alpha_init]
            P = np.eye(self.state_dim) * 1.0        # moderate uncertainty with warm-start
        else:
            theta = np.zeros((self.state_dim, 1))
            P = np.eye(self.state_dim) * 10.0       # high uncertainty → fast initial convergence

        I = np.eye(self.state_dim)

        # 2. Recursive Kalman Filter loop
        for t in range(T):
            # Observation matrix H_t = [x_t, 1]
            H_t = np.array([[x[t], 1.0]])  

            # Predict Step
            theta_pred = theta  
            P_pred = P + self.Q  

            # Innovation (before the update)
            # e_t = y_t - H_t @ theta_pred
            y_pred = np.dot(H_t, theta_pred)
            e_t = y[t] - y_pred[0, 0]  # the result of np.dot is [[beta*x + alpha]], we want the scalar value for the innovation so we use [0, 0] to extract it

            
            # Kalman Gain (Update)
            # F_t = H_t @ P_pred @ H_t.T + R
            # K = P_pred @ H_t.T / F_t
            S_t = np.dot(np.dot(H_t, P_pred), H_t.T)[0, 0] + self.R  # Innovation variance (scalar)
            K = np.dot(P_pred, H_t.T) / S_t

            # Update estimates
            theta = theta_pred + (K * e_t)
            P = np.dot(I - np.dot(K, H_t), P_pred)

            # Store results
            state_means[t, :] = theta.flatten()  
            innovations[t] = e_t
            innovation_vars[t] = S_t

        return state_means, innovations, innovation_vars


# ------------------------------------------------------------------
# MODULAR TESTING
# ------------------------------------------------------------------
if __name__ == "__main__":
    import time
 
    print("=== Test KalmanFilterPairs con calibrazione ===\n")
 
    np.random.seed(42)
    n_days = 1000
    X = np.cumsum(np.random.normal(0, 1, n_days)) + 100
    true_beta = np.linspace(1.5, 2.0, n_days)
    Y = true_beta * X + 10.0 + np.random.normal(0, 0.5, n_days)
 
    # Test 1: con default (backward-compatible)
    kf_default = KalmanFilterPairs()
    t0 = time.time()
    states, innov, innov_var = kf_default.filter(Y, X)
    dt = (time.time() - t0) * 1000
    print(f"[Default Q/R] {n_days} days in {dt:.1f}ms")
    print(f"  Final beta: {states[-1, 0]:.4f} (true: {true_beta[-1]:.4f})")
    print(f"  Mean NIS: {np.mean(innov**2 / innov_var):.3f} (should be ~1.0)\n")
 
    # Test 2: con Q/R calibrati
    import statsmodels.api as sm
    res = sm.OLS(Y[:700], sm.add_constant(X[:700])).fit()
    R_cal = float(np.var(res.resid))
    Q_cal = np.diag([1e-4 * R_cal, 0.01 * R_cal])
 
    kf_cal = KalmanFilterPairs(R=R_cal, Q=Q_cal)
    states2, innov2, innov_var2 = kf_cal.filter(
        Y, X, beta_init=res.params[1], alpha_init=res.params[0]
    )
    print(f"[Calibrated Q/R] R={R_cal:.4f}")
    print(f"  Final beta: {states2[-1, 0]:.4f} (true: {true_beta[-1]:.4f})")
    print(f"  Mean NIS: {np.mean(innov2**2 / innov_var2):.3f} (should be ~1.0)")