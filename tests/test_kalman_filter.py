"""Tests for the KalmanFilter class."""

import numpy as np
import pytest

from alphastream_kf.kalman_filter import KalmanFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_scalar_kf(q: float = 1e-4, r: float = 1e-2, p0: float = 1.0) -> KalmanFilter:
    """Return a 1-D KF with a scalar state (useful for simple tests)."""
    return KalmanFilter(
        transition_matrix=np.array([[1.0]]),
        process_noise_cov=np.array([[q]]),
        observation_noise_var=r,
        initial_state=np.array([0.0]),
        initial_state_cov=np.array([[p0]]),
    )


def make_2d_kf(q: float = 1e-4, r: float = 1e-2, p0: float = 1.0) -> KalmanFilter:
    """Return a 2-D KF for [beta, alpha] state tracking."""
    n = 2
    return KalmanFilter(
        transition_matrix=np.eye(n),
        process_noise_cov=q * np.eye(n),
        observation_noise_var=r,
        initial_state=np.zeros(n),
        initial_state_cov=p0 * np.eye(n),
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestKalmanFilterConstruction:
    def test_valid_construction_1d(self):
        kf = make_scalar_kf()
        assert kf.state.shape == (1,)
        assert kf.P.shape == (1, 1)

    def test_valid_construction_2d(self):
        kf = make_2d_kf()
        assert kf.state.shape == (2,)
        assert kf.P.shape == (2, 2)

    def test_shape_mismatch_F(self):
        with pytest.raises(ValueError, match="transition_matrix"):
            KalmanFilter(
                transition_matrix=np.eye(3),          # wrong size
                process_noise_cov=np.eye(2),
                observation_noise_var=0.01,
                initial_state=np.zeros(2),
                initial_state_cov=np.eye(2),
            )

    def test_shape_mismatch_Q(self):
        with pytest.raises(ValueError, match="process_noise_cov"):
            KalmanFilter(
                transition_matrix=np.eye(2),
                process_noise_cov=np.eye(3),           # wrong size
                observation_noise_var=0.01,
                initial_state=np.zeros(2),
                initial_state_cov=np.eye(2),
            )

    def test_non_positive_R(self):
        with pytest.raises(ValueError, match="observation_noise_var"):
            KalmanFilter(
                transition_matrix=np.eye(2),
                process_noise_cov=np.eye(2),
                observation_noise_var=-1.0,
                initial_state=np.zeros(2),
                initial_state_cov=np.eye(2),
            )

    def test_zero_R(self):
        with pytest.raises(ValueError, match="observation_noise_var"):
            KalmanFilter(
                transition_matrix=np.eye(2),
                process_noise_cov=np.eye(2),
                observation_noise_var=0.0,
                initial_state=np.zeros(2),
                initial_state_cov=np.eye(2),
            )


# ---------------------------------------------------------------------------
# Predict step
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_returns_copies(self):
        kf = make_2d_kf()
        s, p = kf.predict()
        # Modifying returned arrays must not change internal state
        s[:] = 999
        p[:] = 999
        assert not np.allclose(kf.state, 999)
        assert not np.allclose(kf.P, 999)

    def test_covariance_grows_on_predict(self):
        """With non-zero Q, predicted covariance must exceed prior covariance."""
        kf = make_2d_kf(q=1e-3, p0=1.0)
        P_before = kf.P.copy()
        kf.predict()
        assert np.all(np.diag(kf.P) > np.diag(P_before))

    def test_state_unchanged_with_zero_mean_noise(self):
        """Random-walk prediction (F=I) does not change the state mean."""
        kf = make_2d_kf()
        kf.state = np.array([1.5, -0.3])
        kf.predict()
        np.testing.assert_allclose(kf.state, [1.5, -0.3])


# ---------------------------------------------------------------------------
# Update step
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_update_returns_four_values(self):
        kf = make_2d_kf()
        kf.predict()
        result = kf.update(observation=1.0, observation_matrix=np.array([2.0, 1.0]))
        assert len(result) == 4

    def test_innovation_is_scalar(self):
        kf = make_2d_kf()
        kf.predict()
        _, _, innov, innov_var = kf.update(1.0, np.array([2.0, 1.0]))
        assert np.isscalar(innov) or innov.shape == ()
        assert innov_var > 0

    def test_covariance_decreases_after_update(self):
        """Observing data reduces state uncertainty."""
        kf = make_2d_kf(p0=10.0, r=0.01)
        kf.predict()
        P_pred = kf.P.copy()
        kf.update(1.0, np.array([1.0, 1.0]))
        assert np.all(np.diag(kf.P) < np.diag(P_pred))

    def test_update_returns_copies(self):
        kf = make_2d_kf()
        kf.predict()
        s, p, _, _ = kf.update(1.0, np.array([2.0, 1.0]))
        s[:] = 999
        p[:] = 999
        assert not np.allclose(kf.state, 999)
        assert not np.allclose(kf.P, 999)

    def test_covariance_is_symmetric_after_update(self):
        kf = make_2d_kf()
        rng = np.random.default_rng(0)
        for _ in range(10):
            kf.predict()
            H = rng.uniform(-2, 2, size=2)
            kf.update(float(rng.normal()), H)
        np.testing.assert_allclose(kf.P, kf.P.T, atol=1e-12)

    def test_innovation_zero_when_observation_matches_prediction(self):
        """If the observation exactly matches the prior prediction the
        innovation should be zero (up to floating point)."""
        kf = make_scalar_kf(q=0.0, r=0.01)
        kf.state = np.array([5.0])
        kf.P = np.array([[0.0]])  # zero uncertainty in state
        kf.predict()
        # Predicted observation: H @ state = [1] @ [5] = 5
        _, _, innov, _ = kf.update(observation=5.0, observation_matrix=np.array([1.0]))
        assert abs(innov) < 1e-10


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------

class TestConvergence:
    def test_tracks_constant_signal(self):
        """The KF should converge to a constant hidden state."""
        rng = np.random.default_rng(42)
        TRUE_BETA = 2.0
        kf = make_scalar_kf(q=1e-5, r=0.01, p0=1.0)
        obs = TRUE_BETA + rng.normal(0, 0.1, size=500)
        for y in obs:
            kf.predict()
            kf.update(y, np.array([1.0]))
        np.testing.assert_allclose(kf.state[0], TRUE_BETA, atol=0.05)

    def test_tracks_slowly_drifting_parameter(self):
        """KF should follow a slowly varying β that shifts mid-series."""
        rng = np.random.default_rng(7)
        T = 400
        true_betas = np.concatenate([np.full(200, 1.5), np.full(200, 3.0)])
        x = rng.uniform(1, 5, size=T)
        y = true_betas * x + rng.normal(0, 0.05, size=T)

        kf = make_2d_kf(q=1e-3, r=0.01, p0=1.0)
        betas = []
        for t in range(T):
            kf.predict()
            kf.update(y[t], np.array([x[t], 1.0]))
            betas.append(kf.state[0])

        # Estimate at end should be close to the new regime value
        np.testing.assert_allclose(np.mean(betas[-50:]), 3.0, atol=0.1)
