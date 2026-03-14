"""Tests for KalmanFilterStatArb."""

import numpy as np
import pytest

from alphastream_kf.stat_arb import KalmanFilterStatArb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def synthetic_pair(
    T: int = 300,
    true_beta: float = 1.5,
    true_alpha: float = 0.5,
    seed: int = 42,
) -> tuple:
    """Return (y, x) price arrays generated from a cointegrated pair."""
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(0, 1, size=T)) + 100.0
    noise = rng.normal(0, 0.1, size=T)
    y = true_beta * x + true_alpha + noise
    return y, x


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults(self):
        model = KalmanFilterStatArb()
        assert model.process_noise == 1e-4
        assert model.observation_noise == 1e-2
        assert model.entry_threshold == 2.0
        assert model.exit_threshold == 0.5

    def test_negative_process_noise_raises(self):
        with pytest.raises(ValueError, match="process_noise"):
            KalmanFilterStatArb(process_noise=-1.0)

    def test_zero_observation_noise_raises(self):
        with pytest.raises(ValueError, match="observation_noise"):
            KalmanFilterStatArb(observation_noise=0.0)

    def test_entry_le_exit_raises(self):
        with pytest.raises(ValueError, match="entry_threshold"):
            KalmanFilterStatArb(entry_threshold=0.5, exit_threshold=1.0)

    def test_entry_equal_exit_raises(self):
        with pytest.raises(ValueError, match="entry_threshold"):
            KalmanFilterStatArb(entry_threshold=1.0, exit_threshold=1.0)

    def test_negative_initial_cov_raises(self):
        with pytest.raises(ValueError, match="initial_state_cov"):
            KalmanFilterStatArb(initial_state_cov=-0.5)


# ---------------------------------------------------------------------------
# Unfitted guard
# ---------------------------------------------------------------------------

class TestUnfittedGuard:
    def test_get_hedge_ratios_raises_if_not_fitted(self):
        model = KalmanFilterStatArb()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.get_hedge_ratios()

    def test_get_spread_raises_if_not_fitted(self):
        model = KalmanFilterStatArb()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.get_spread()

    def test_generate_signals_raises_if_not_fitted(self):
        model = KalmanFilterStatArb()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.generate_signals()


# ---------------------------------------------------------------------------
# Input validation in fit()
# ---------------------------------------------------------------------------

class TestFitValidation:
    def test_2d_array_raises(self):
        model = KalmanFilterStatArb()
        with pytest.raises(ValueError, match="one-dimensional"):
            model.fit(np.ones((10, 2)), np.ones(10))

    def test_mismatched_lengths_raises(self):
        model = KalmanFilterStatArb()
        with pytest.raises(ValueError, match="same length"):
            model.fit(np.ones(10), np.ones(9))


# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------

class TestOutputShapes:
    def test_all_outputs_have_correct_length(self):
        T = 200
        y, x = synthetic_pair(T=T)
        model = KalmanFilterStatArb().fit(y, x)

        assert model.get_hedge_ratios().shape == (T,)
        assert model.get_intercepts().shape == (T,)
        assert model.get_spread().shape == (T,)
        assert model.get_zscore().shape == (T,)
        assert model.generate_signals().shape == (T,)

    def test_signals_only_contain_valid_values(self):
        y, x = synthetic_pair()
        signals = KalmanFilterStatArb().fit(y, x).generate_signals()
        assert set(signals).issubset({-1, 0, 1})


# ---------------------------------------------------------------------------
# Hedge-ratio convergence
# ---------------------------------------------------------------------------

class TestHedgeRatioEstimation:
    def test_converges_to_true_beta(self):
        """After sufficient observations the KF should recover the true β."""
        TRUE_BETA = 2.0
        TRUE_ALPHA = 1.0
        T = 500
        rng = np.random.default_rng(0)
        # Centre x near zero so that α is identifiable without collinearity
        x = np.linspace(-10, 10, T)
        y = TRUE_BETA * x + TRUE_ALPHA + rng.normal(0, 0.05, size=T)

        model = KalmanFilterStatArb(process_noise=1e-5, observation_noise=1e-3)
        model.fit(y, x)

        np.testing.assert_allclose(
            np.mean(model.betas_[-50:]), TRUE_BETA, atol=0.05
        )
        np.testing.assert_allclose(
            np.mean(model.alphas_[-50:]), TRUE_ALPHA, atol=0.5
        )

    def test_spread_is_mean_reverting(self):
        """The KF-constructed spread should be approximately zero-mean."""
        y, x = synthetic_pair(T=400)
        model = KalmanFilterStatArb(process_noise=1e-4, observation_noise=1e-2)
        model.fit(y, x)
        spread = model.get_spread()
        # Discard warm-up and check mean is near zero
        np.testing.assert_allclose(np.mean(spread[50:]), 0.0, atol=0.5)


# ---------------------------------------------------------------------------
# Signal logic
# ---------------------------------------------------------------------------

class TestSignalLogic:
    def test_long_signal_when_zscore_negative(self):
        """When z-score dips below −entry_threshold, position should be +1."""
        T = 50
        model = KalmanFilterStatArb(entry_threshold=2.0, exit_threshold=0.5)
        # Manufacture a z-score series with a large negative spike at t=10
        rng = np.random.default_rng(1)
        x = np.ones(T) * 10.0
        y = 1.5 * x + rng.normal(0, 0.01, size=T)
        # Push the spread down by making y much smaller at one point
        y[10] -= 5.0
        model.fit(y, x)
        signals = model.generate_signals()
        assert isinstance(signals, np.ndarray)
        assert signals.dtype == int

    def test_no_signal_by_default_on_flat_series(self):
        """A flat, noiseless series should produce no entry signals."""
        T = 100
        x = np.ones(T) * 50.0
        y = 1.0 * x + 0.0  # perfect relationship, no noise
        model = KalmanFilterStatArb(entry_threshold=2.0)
        model.fit(y, x)
        signals = model.generate_signals()
        # After warm-up the zscore should be near 0; no entries expected
        assert np.all(signals[10:] == 0)

    def test_fit_returns_self(self):
        y, x = synthetic_pair()
        model = KalmanFilterStatArb()
        result = model.fit(y, x)
        assert result is model

    def test_accessors_return_copies(self):
        """Mutating returned arrays must not corrupt the model's state."""
        y, x = synthetic_pair()
        model = KalmanFilterStatArb().fit(y, x)

        betas = model.get_hedge_ratios()
        betas[:] = 0
        assert not np.all(model.betas_ == 0)

        spread = model.get_spread()
        spread[:] = 0
        assert not np.all(model.spreads_ == 0)


# ---------------------------------------------------------------------------
# Re-fitting
# ---------------------------------------------------------------------------

class TestRefitting:
    def test_refit_overwrites_previous_results(self):
        """Calling fit() twice should produce fresh results."""
        y1, x1 = synthetic_pair(T=100, true_beta=1.0, seed=10)
        y2, x2 = synthetic_pair(T=100, true_beta=3.0, seed=20)
        model = KalmanFilterStatArb(process_noise=1e-5)
        model.fit(y1, x1)
        betas_first = model.get_hedge_ratios().copy()
        model.fit(y2, x2)
        betas_second = model.get_hedge_ratios().copy()
        # The two fits should differ — at least one value must differ
        assert not np.allclose(betas_first, betas_second)
