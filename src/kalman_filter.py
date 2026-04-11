import numpy as np
import pandas as pd
from typing import Tuple

class KalmanFilterPairs:
    """
    Dynamic estimation of the Hedge Ratio (Beta) using a Kalman Filter for pairs trading.
    Designed for fast execution using NumPy contiguous arrays. 
    """

    def __init__(
            self, 
            observation_variance: float = 1.0, 
            process_variance: float = 1e-3):
        """
        Initializes the covariance matrices.

        Args:
            observation_variance (R): Noise in the daily price observation. Set to 1.0,
                                      prices are noisy and the filter should not over-trust
                                      a single day's reading.
            process_variance (Q):     Daily variability of the true hedge ratio. Set to 1e-3
                                      so that Q/R = 0.001, meaning beta adapts slowly and
                                      smoothly rather than chasing every daily fluctuation.
                                      This matches the delta=1e-3 convention used in the
                                      pykalman literature for pairs trading.
        """
        self.R = observation_variance  
        self.state_dim = 2 # state = [beta, alpha]
        self.Q = np.eye(self.state_dim) * process_variance


    def filter(self, y_prices: np.ndarray, x_prices: np.ndarray, beta_init=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies the recursive Predict-Update loop of the Kalman Filter
        
        Args:
            y_prices: 1D countigous array of the dependent variable (e.g., KO prices).
            x_prices: 1D contiguous array of the independent variable (e.g., PEP prices).

        Returns:
            Tupe containing:
                - state_means: 2D array of shape (T, 2) with [beta_t, alpha_t] estimates at each time step.
                - spread: 1D array of shape (T,) containing the innovation (spread) at each time step.
        """

        y = np.asarray(y_prices).flatten()
        x = np.asarray(x_prices).flatten()
        T = len(y)

        # First, pre-allocate memory for outputs for better performance
        state_means = np.zeros((T, self.state_dim))  
        spread = np.zeros(T)

        # 1. Initialization (t=0)
        if beta_init is not None:
            theta = np.array([[beta_init], [0.0]])  # [beta_OLS, alpha=0]
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

            # Update Step
            # Compute the innovation (spread)
            # e_t = y_t - H_t @ theta_pred
            y_pred = np.dot(H_t, theta_pred)
            e_t = y[t] - y_pred[0, 0]  # the result of np.dot is [[beta*x + alpha]], we want the scalar value for the innovation so we use [0, 0] to extract it

            # Kalman Gain
            # F_t = H_t @ P_pred @ H_t.T + R
            # K = P_pred @ H_t.T / F_t
            F_t = np.dot(np.dot(H_t, P_pred), H_t.T) + self.R  
            K = np.dot(P_pred, H_t.T) / F_t 

            # Update estimates
            theta = theta_pred + (K * e_t)
            P = np.dot(I - np.dot(K, H_t), P_pred)

            # Store results
            state_means[t, :] = theta.flatten()  
            spread[t] = e_t

        return state_means, spread



# --- MODULAR TESTING ---
if __name__ == "__main__":
    print("=== Testing Kalman Filter Optimization ===")
    
    # Generate dummy data
    np.random.seed(42)
    n_days = 1000
    X = np.cumsum(np.random.normal(0, 1, n_days)) + 100
    
    # Simulate a true beta drifting from 1.5 to 2.0
    true_beta = np.linspace(1.5, 2.0, n_days)
    Y = true_beta * X + 10.0 + np.random.normal(0, 0.5, n_days)
    
    # Initialize and run
    kf = KalmanFilterPairs(observation_variance=0.5, process_variance=1e-5)
    
    # Measuring performance (Python loop speed)
    import time
    start_time = time.time()
    states, spread = kf.filter(Y, X)
    end_time = time.time()
    
    estimated_beta = states[:, 0]
    
    print(f"Processed {n_days} days in {(end_time - start_time)*1000:.2f} milliseconds.")
    print(f"Final True Beta: {true_beta[-1]:.4f}")
    print(f"Final Estimated Beta: {estimated_beta[-1]:.4f}")


