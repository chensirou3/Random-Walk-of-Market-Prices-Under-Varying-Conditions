"""Statistical metrics for simulated market price processes."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def _finite_array(series: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(series, dtype=float)
    return arr[np.isfinite(arr)]


def compute_returns(price_series: Iterable[float] | np.ndarray) -> np.ndarray:
    """Compute daily log returns from a positive price series."""
    prices = np.asarray(price_series, dtype=float)
    if prices.ndim != 1:
        raise ValueError("price_series must be one-dimensional.")
    if np.any(prices <= 0):
        raise ValueError("price_series must be strictly positive.")
    return np.diff(np.log(prices))


def compute_basic_metrics(
    price_series: Iterable[float] | np.ndarray,
    return_series: Iterable[float] | np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the core metric set used by experiment tables."""
    prices = _finite_array(price_series)
    returns = compute_returns(prices) if return_series is None else _finite_array(return_series)
    distribution_stats = compute_distribution_stats(returns)
    return {
        "mean_return": float(np.mean(returns)) if returns.size else float("nan"),
        "std_return": float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0,
        "skewness": distribution_stats["skewness"],
        "kurtosis": distribution_stats["kurtosis"],
        "return_autocorr_lag1": compute_autocorrelation(returns, lag=1),
        "variance_ratio_lag5": compute_variance_ratio(returns, lag=5),
        "positive_return_pct": compute_positive_return_pct(returns),
        "price_min": float(np.min(prices)) if prices.size else float("nan"),
        "price_max": float(np.max(prices)) if prices.size else float("nan"),
        "final_price": float(prices[-1]) if prices.size else float("nan"),
        "max_drawdown": compute_max_drawdown(prices),
        "realized_volatility": compute_realized_volatility(returns),
    }


def compute_autocorrelation(series: Iterable[float] | np.ndarray, lag: int = 1) -> float:
    """Compute sample autocorrelation at a positive lag."""
    if lag < 1:
        raise ValueError("lag must be at least 1.")
    arr = _finite_array(series)
    if arr.size <= lag:
        return float("nan")
    x = arr[:-lag]
    y = arr[lag:]
    if np.std(x) <= 1e-15 or np.std(y) <= 1e-15:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def compute_max_drawdown(price_series: Iterable[float] | np.ndarray) -> float:
    """Compute the most negative peak-to-trough drawdown."""
    prices = _finite_array(price_series)
    if prices.size == 0:
        return float("nan")
    running_max = np.maximum.accumulate(prices)
    drawdowns = prices / running_max - 1.0
    return float(np.min(drawdowns))


def compute_realized_volatility(return_series: Iterable[float] | np.ndarray) -> float:
    """Compute realized volatility as sqrt(sum of squared log returns)."""
    returns = _finite_array(return_series)
    if returns.size == 0:
        return float("nan")
    return float(np.sqrt(np.sum(returns**2)))


def compute_variance_ratio(return_series: Iterable[float] | np.ndarray, lag: int = 5) -> float:
    """Compute a simple variance ratio for log returns.

    For a random walk with uncorrelated increments, the variance of lag-period
    returns should be close to ``lag`` times the one-period return variance, so
    the ratio should be near 1.
    """
    if lag < 2:
        raise ValueError("lag must be at least 2.")
    returns = _finite_array(return_series)
    if returns.size <= lag:
        return float("nan")
    one_period_var = float(np.var(returns, ddof=1))
    if one_period_var <= 1e-15:
        return 0.0
    rolling_lag_returns = np.convolve(returns, np.ones(lag), mode="valid")
    lag_var = float(np.var(rolling_lag_returns, ddof=1))
    return float(lag_var / (lag * one_period_var))


def compute_positive_return_pct(return_series: Iterable[float] | np.ndarray) -> float:
    """Return the percentage of strictly positive return observations."""
    returns = _finite_array(return_series)
    if returns.size == 0:
        return float("nan")
    return float(np.mean(returns > 0.0))


def compute_max_abs_return(return_series: Iterable[float] | np.ndarray) -> float:
    """Return the largest absolute log return."""
    returns = _finite_array(return_series)
    if returns.size == 0:
        return float("nan")
    return float(np.max(np.abs(returns)))


def compute_corr(x: Iterable[float] | np.ndarray, y: Iterable[float] | np.ndarray) -> float:
    """Compute correlation between two finite one-dimensional arrays."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n = min(x_arr.size, y_arr.size)
    if n < 2:
        return float("nan")
    x_arr = x_arr[:n]
    y_arr = y_arr[:n]
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size < 2 or np.std(x_arr) <= 1e-15 or np.std(y_arr) <= 1e-15:
        return 0.0
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def compute_distribution_stats(return_series: Iterable[float] | np.ndarray) -> dict[str, float]:
    """Compute skewness and excess kurtosis for a return series."""
    returns = _finite_array(return_series)
    if returns.size == 0:
        return {"skewness": float("nan"), "kurtosis": float("nan")}
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=0))
    if std <= 1e-15:
        return {"skewness": 0.0, "kurtosis": 0.0}
    centered = returns - mean
    skewness = float(np.mean(centered**3) / std**3)
    kurtosis = float(np.mean(centered**4) / std**4 - 3.0)
    return {"skewness": skewness, "kurtosis": kurtosis}


def compute_mean_reversion_slope(
    price_series: Iterable[float] | np.ndarray,
    initial_price: float,
) -> tuple[float, float, float]:
    """Regress next-day return on current log deviation from initial price.

    The model is ``return[t+1] = alpha + beta * log(P[t] / P[0]) + error``.
    A negative beta indicates that positive deviations tend to be followed by
    lower returns.
    """
    if initial_price <= 0:
        raise ValueError("initial_price must be positive.")
    prices = _finite_array(price_series)
    if prices.size < 3:
        return 0.0, 0.0, 0.0
    returns = compute_returns(prices)
    deviation = np.log(prices[:-1] / initial_price)
    mask = np.isfinite(deviation) & np.isfinite(returns)
    x = deviation[mask]
    y = returns[mask]
    if x.size < 2:
        return 0.0, float(np.mean(y)) if y.size else 0.0, 0.0
    if np.var(x) <= 1e-15:
        return 0.0, float(np.mean(y)), 0.0

    x_design = np.column_stack([np.ones_like(x), x])
    alpha, beta = np.linalg.lstsq(x_design, y, rcond=None)[0]
    y_hat = alpha + beta * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 0.0 if ss_tot <= 1e-15 else 1.0 - ss_res / ss_tot
    return float(beta), float(alpha), float(r2)


def compute_curve_roughness(auction_curve: pd.DataFrame) -> float:
    """Compute normalized auction-curve roughness.

    Roughness is defined as the average first absolute difference of normalized
    demand and supply curves. Demand and supply are each normalized by their own
    maximum quantity, so curves from different market depths are comparable. A
    smoother curve has lower roughness.
    """
    required = {"demand", "supply"}
    if auction_curve.empty or not required.issubset(auction_curve.columns):
        return float("nan")

    values: list[float] = []
    for column in ["demand", "supply"]:
        curve = np.asarray(auction_curve[column], dtype=float)
        if curve.size < 2:
            continue
        scale = float(np.nanmax(np.abs(curve)))
        if scale <= 1e-15:
            continue
        normalized = curve / scale
        values.append(float(np.mean(np.abs(np.diff(normalized)))))

    return float(np.mean(values)) if values else float("nan")


def confidence_interval_95(values: Iterable[float] | np.ndarray) -> tuple[float, float]:
    """Return a normal-approximation 95% confidence interval for the mean."""
    arr = _finite_array(values)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        mean = float(arr[0])
        return mean, mean
    mean = float(np.mean(arr))
    half_width = 1.96 * float(np.std(arr, ddof=1)) / np.sqrt(arr.size)
    return mean - half_width, mean + half_width


def mean_return_t_statistic(
    return_series: Iterable[float] | np.ndarray,
) -> dict[str, float]:
    """Compute a one-sample t-statistic that the mean return exceeds zero.

    The two-sided p-value is approximated using the standard normal CDF, which is
    accurate for the long return series used in this project and avoids a scipy
    dependency. ``p_value_one_sided_positive`` tests ``H1: mean > 0``.
    """
    returns = _finite_array(return_series)
    if returns.size < 2:
        return {
            "n": float(returns.size),
            "mean_return": float(np.mean(returns)) if returns.size else float("nan"),
            "std_return": float("nan"),
            "t_stat": float("nan"),
            "p_value_two_sided": float("nan"),
            "p_value_one_sided_positive": float("nan"),
        }
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1))
    n = returns.size
    if std <= 1e-15:
        t_stat = float("inf") if mean > 0 else (float("-inf") if mean < 0 else 0.0)
    else:
        t_stat = mean / (std / math.sqrt(n))
    abs_t = abs(t_stat) if math.isfinite(t_stat) else float("inf")
    p_two = math.erfc(abs_t / math.sqrt(2.0)) if math.isfinite(abs_t) else 0.0
    p_one_positive = 0.5 * math.erfc(t_stat / math.sqrt(2.0)) if math.isfinite(t_stat) else (0.0 if t_stat > 0 else 1.0)
    return {
        "n": float(n),
        "mean_return": mean,
        "std_return": std,
        "t_stat": float(t_stat),
        "p_value_two_sided": float(p_two),
        "p_value_one_sided_positive": float(p_one_positive),
    }


def gini_coefficient(values: Iterable[float] | np.ndarray) -> float:
    """Compute the Gini coefficient of a non-negative distribution.

    Negative entries are clipped to zero before computation so the result is
    always in ``[0, 1]``. Returns ``nan`` for empty input and ``0`` for a
    distribution whose total mass is zero.
    """
    arr = _finite_array(values)
    if arr.size == 0:
        return float("nan")
    arr = np.clip(arr, 0.0, None)
    total = float(np.sum(arr))
    if total <= 1e-15:
        return 0.0
    sorted_arr = np.sort(arr)
    n = sorted_arr.size
    cumulative = np.cumsum(sorted_arr)
    gini = (n + 1 - 2 * float(np.sum(cumulative)) / total) / n
    return float(max(0.0, min(1.0, gini)))


def summarize_multiple_runs(results: list[dict[str, float]] | list[object]) -> pd.DataFrame:
    """Summarize metrics across multiple simulation runs.

    ``results`` can be a list of metric dictionaries or objects with ``prices``
    and ``returns`` attributes, such as ``MarketRunResult``.
    """
    rows: list[dict[str, float]] = []
    for result in results:
        if isinstance(result, dict):
            rows.append(result)
        else:
            prices = getattr(result, "prices")
            returns = getattr(result, "returns")
            rows.append(compute_basic_metrics(prices, returns))
    return pd.DataFrame(rows)
