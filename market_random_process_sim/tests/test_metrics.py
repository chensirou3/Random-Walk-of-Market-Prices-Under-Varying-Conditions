import numpy as np

from analysis.metrics import (
    compute_autocorrelation,
    compute_curve_roughness,
    compute_max_drawdown,
    compute_mean_reversion_slope,
    compute_returns,
    compute_variance_ratio,
    gini_coefficient,
    mean_return_t_statistic,
)
import pandas as pd


def test_compute_returns_uses_log_returns():
    prices = np.array([100.0, 110.0, 121.0])
    returns = compute_returns(prices)

    np.testing.assert_allclose(returns, [np.log(1.1), np.log(1.1)])


def test_compute_max_drawdown():
    prices = np.array([100.0, 120.0, 90.0, 110.0])

    assert compute_max_drawdown(prices) == -0.25


def test_compute_autocorrelation():
    series = np.array([1.0, 2.0, 3.0, 4.0])

    assert np.isclose(compute_autocorrelation(series, lag=1), 1.0)


def test_mean_reversion_slope_returns_numbers():
    prices = np.array([100.0, 101.0, 99.0, 100.5, 98.5, 100.0])
    beta, alpha, r2 = compute_mean_reversion_slope(prices, initial_price=100.0)

    assert np.isfinite(beta)
    assert np.isfinite(alpha)
    assert np.isfinite(r2)


def test_variance_ratio_returns_number():
    returns = np.array([0.01, -0.01, 0.02, -0.02, 0.01, 0.0, -0.01])

    assert np.isfinite(compute_variance_ratio(returns, lag=3))


def test_curve_roughness_lower_for_smoother_curve():
    rough_curve = pd.DataFrame(
        {
            "demand": [1.0, 0.2, 0.9, 0.1],
            "supply": [0.1, 0.8, 0.3, 1.0],
        }
    )
    smooth_curve = pd.DataFrame(
        {
            "demand": [1.0, 0.8, 0.6, 0.4],
            "supply": [0.2, 0.4, 0.6, 0.8],
        }
    )

    assert compute_curve_roughness(smooth_curve) < compute_curve_roughness(rough_curve)


def test_mean_return_t_statistic_zero_mean_series():
    rng = np.random.default_rng(0)
    returns = rng.normal(loc=0.0, scale=0.01, size=2000)
    result = mean_return_t_statistic(returns)

    assert result["n"] == 2000
    assert np.isfinite(result["t_stat"])
    assert abs(result["t_stat"]) < 3.0
    assert 0.0 <= result["p_value_two_sided"] <= 1.0
    assert 0.0 <= result["p_value_one_sided_positive"] <= 1.0


def test_mean_return_t_statistic_detects_positive_drift():
    rng = np.random.default_rng(1)
    returns = rng.normal(loc=0.005, scale=0.01, size=2000)
    result = mean_return_t_statistic(returns)

    assert result["mean_return"] > 0
    assert result["t_stat"] > 5.0
    assert result["p_value_one_sided_positive"] < 1e-6


def test_mean_return_t_statistic_small_sample_returns_nan():
    result = mean_return_t_statistic([0.01])

    assert result["n"] == 1.0
    assert np.isnan(result["t_stat"])
    assert np.isnan(result["p_value_one_sided_positive"])


def test_gini_coefficient_equal_distribution_is_zero():
    assert gini_coefficient([5.0, 5.0, 5.0, 5.0]) == 0.0


def test_gini_coefficient_maximally_unequal_distribution():
    values = [0.0] * 99 + [100.0]
    gini = gini_coefficient(values)

    assert gini > 0.9
    assert gini <= 1.0


def test_gini_coefficient_zero_total_returns_zero():
    assert gini_coefficient([0.0, 0.0, 0.0]) == 0.0


def test_gini_coefficient_empty_input_returns_nan():
    assert np.isnan(gini_coefficient([]))
