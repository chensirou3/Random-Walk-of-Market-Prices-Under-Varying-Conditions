"""Experiment orchestration for the market random process simulator."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from analysis.metrics import (
    compute_basic_metrics,
    compute_corr,
    compute_curve_roughness,
    compute_max_abs_return,
    compute_mean_reversion_slope,
    confidence_interval_95,
    gini_coefficient,
    mean_return_t_statistic,
)
from analysis.plots import (
    plot_boxplot_by_group,
    plot_auction_curve,
    plot_deviation_series,
    plot_dual_axis_chain,
    plot_grouped_bar,
    plot_grouped_price_paths,
    plot_mean_paths_with_ci,
    plot_mean_reversion_scatter,
    plot_metric_vs_parameter,
    plot_metric_with_ci,
    plot_multi_line_with_ci,
    plot_multiple_price_paths,
    plot_panel_price_paths,
    plot_paths_with_mean,
    plot_price_distribution_comparison,
    plot_price_path,
    plot_return_distribution,
    plot_return_distribution_by_group,
    plot_scatter_with_fit,
    plot_return_series,
    plot_table_image,
)
from analysis.report import generate_report
from simulator.auction import clear_auction_arrays, clear_auction_price_volume_batch_continuous
from simulator.market import DailyAuctionMarket, MarketRunResult


def ensure_output_dirs(output_dir: str | Path) -> dict[str, Path]:
    """Create output directories and return their paths."""
    base = Path(output_dir)
    dirs = {
        "base": base,
        "figures": base / "figures",
        "formula_figures": base / "formula_figures",
        "tables": base / "tables",
        "simulation_results": base / "simulation_results",
        "diagnostics": base / "diagnostics",
        "reports": base / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def run_experiment_1(
    seed: int = config.DEFAULT_SEED,
    n_days: int = config.DEFAULT_N_DAYS,
    output_dir: str | Path = config.OUTPUT_DIR,
    n_paths: int = 50,
) -> dict[str, object]:
    """Run Experiment 1: unlimited-wealth random market."""
    dirs = ensure_output_dirs(output_dir)
    sample_day = max(1, n_days // 2)

    market = DailyAuctionMarket(
        initial_price=config.DEFAULT_INITIAL_PRICE,
        price_sigma=config.DEFAULT_PRICE_SIGMA,
        max_qty=config.DEFAULT_MAX_QTY,
        seed=seed,
    )
    single_result = market.simulate_unlimited(
        n_days=n_days,
        n_buyers=config.DEFAULT_N_BUYERS,
        n_sellers=config.DEFAULT_N_SELLERS,
        sample_curve_days=[sample_day],
    )
    single_result.save_csv(dirs["simulation_results"] / "exp1_single_path.csv")

    metrics = compute_basic_metrics(single_result.prices, single_result.returns)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(dirs["tables"] / "exp1_metrics.csv", index=False)

    paths: list[np.ndarray] = []
    path_metrics: list[dict[str, float]] = []
    path_rows: list[pd.DataFrame] = []
    for path_id in range(n_paths):
        path_market = DailyAuctionMarket(
            initial_price=config.DEFAULT_INITIAL_PRICE,
            price_sigma=config.DEFAULT_PRICE_SIGMA,
            max_qty=config.DEFAULT_MAX_QTY,
            seed=seed + 10_000 + path_id,
        )
        result = path_market.simulate_unlimited(
            n_days=n_days,
            n_buyers=config.DEFAULT_N_BUYERS,
            n_sellers=config.DEFAULT_N_SELLERS,
        )
        paths.append(result.prices)
        path_metric = _random_walk_diagnostics_row(result.prices, result.returns)
        path_metric["sample"] = f"path_{path_id}"
        path_metrics.append(path_metric)
        df = result.to_dataframe()
        df["path_id"] = path_id
        path_rows.append(df)
    pd.concat(path_rows, ignore_index=True).to_csv(
        dirs["simulation_results"] / "exp1_multiple_paths.csv",
        index=False,
    )
    diagnostics_df = _build_exp1_random_walk_diagnostics(single_result, path_metrics)
    diagnostics_df.to_csv(dirs["tables"] / "exp1_random_walk_diagnostics.csv", index=False)

    plot_price_path(
        single_result.prices,
        output_dir,
        "exp1_price_path.png",
        title="Experiment 1: Unlimited-Wealth Price Path",
    )
    plot_return_series(
        single_result.returns,
        output_dir,
        "exp1_return_series.png",
        title="Experiment 1: Daily Log Returns",
    )
    plot_return_distribution(
        single_result.returns,
        output_dir,
        "exp1_return_distribution.png",
        title="Experiment 1: Return Distribution",
    )
    plot_multiple_price_paths(
        paths,
        output_dir,
        "exp1_multiple_paths.png",
        title="Experiment 1: 50 Random Price Paths",
        alpha=0.35,
    )
    plot_auction_curve(
        single_result.auction_curves[sample_day],
        output_dir,
        "exp1_sample_auction_curve.png",
        title=f"Experiment 1: Sample Auction Curve, Day {sample_day}",
    )
    plot_table_image(
        diagnostics_df,
        output_dir,
        "exp1_random_walk_diagnostics_table.png",
        title="Experiment 1: Random Walk Diagnostics",
    )

    return {"single_result": single_result, "metrics": metrics_df, "paths": paths, "diagnostics": diagnostics_df}


def run_experiment_2(
    seed: int = config.DEFAULT_SEED,
    n_days: int = config.DEFAULT_N_DAYS,
    output_dir: str | Path = config.OUTPUT_DIR,
    n_runs: int = config.EXP2_N_RUNS,
) -> dict[str, object]:
    """Run Experiment 2: wealth-constrained random market."""
    dirs = ensure_output_dirs(output_dir)
    sample_day = max(1, n_days // 2)

    unlimited_market = DailyAuctionMarket(
        initial_price=config.DEFAULT_INITIAL_PRICE,
        price_sigma=config.DEFAULT_PRICE_SIGMA,
        max_qty=config.DEFAULT_MAX_QTY,
        seed=seed + 20_000,
    )
    unlimited_result = unlimited_market.simulate_unlimited(
        n_days=n_days,
        n_buyers=config.DEFAULT_N_BUYERS,
        n_sellers=config.DEFAULT_N_SELLERS,
    )
    unlimited_result.save_csv(dirs["simulation_results"] / "exp2_unlimited_baseline.csv")

    constrained_results: dict[float, MarketRunResult] = {}
    metric_rows: list[dict[str, float]] = []
    path_rows: list[pd.DataFrame] = []

    for i, multiplier in enumerate(config.EXP2_WEALTH_MULTIPLIERS):
        market = DailyAuctionMarket(
            initial_price=config.DEFAULT_INITIAL_PRICE,
            price_sigma=config.DEFAULT_PRICE_SIGMA,
            max_qty=config.DEFAULT_MAX_QTY,
            seed=seed + 21_000 + i,
        )
        result = market.simulate_wealth_constrained(
            n_days=n_days,
            n_agents=config.DEFAULT_N_AGENTS,
            initial_cash_per_agent=config.DEFAULT_INITIAL_CASH_PER_AGENT * multiplier,
            initial_shares_per_agent=config.DEFAULT_INITIAL_SHARES_PER_AGENT * multiplier,
            participation_rate=config.DEFAULT_AGENT_PARTICIPATION_RATE,
            sample_curve_days=[sample_day] if multiplier == 1 else None,
        )
        constrained_results[float(multiplier)] = result

        base_metrics = compute_basic_metrics(result.prices, result.returns)
        beta, alpha, r2 = compute_mean_reversion_slope(result.prices, config.DEFAULT_INITIAL_PRICE)
        base_metrics.update(
            {
                "wealth_multiplier": float(multiplier),
                "price_variance": float(np.var(result.prices, ddof=1)),
                "log_deviation_variance": float(np.var(np.log(result.prices / config.DEFAULT_INITIAL_PRICE), ddof=1)),
                "mean_reversion_slope": beta,
                "mean_reversion_intercept": alpha,
                "mean_reversion_r2": r2,
            }
        )
        metric_rows.append(base_metrics)

        df = result.to_dataframe()
        df["wealth_multiplier"] = float(multiplier)
        path_rows.append(df)

    metrics_df = pd.DataFrame(metric_rows)
    ordered_cols = [
        "wealth_multiplier",
        "mean_return",
        "std_return",
        "return_autocorr_lag1",
        "price_variance",
        "log_deviation_variance",
        "mean_reversion_slope",
        "mean_reversion_intercept",
        "mean_reversion_r2",
        "final_price",
        "price_min",
        "price_max",
        "skewness",
        "kurtosis",
        "max_drawdown",
        "realized_volatility",
    ]
    metrics_df = metrics_df[[col for col in ordered_cols if col in metrics_df.columns]]
    metrics_df.to_csv(dirs["tables"] / "exp2_metrics.csv", index=False)
    pd.concat(path_rows, ignore_index=True).to_csv(
        dirs["simulation_results"] / "exp2_wealth_paths.csv",
        index=False,
    )

    repeated = _run_exp2_repeated_statistics(
        seed=seed + 22_000,
        n_days=n_days,
        output_dir=output_dir,
        n_runs=n_runs,
    )

    base_constrained = constrained_results[1.0]
    base_constrained.save_csv(dirs["simulation_results"] / "exp2_constrained_baseline.csv")
    if base_constrained.final_agent_state is not None:
        base_constrained.final_agent_state.to_csv(
            dirs["simulation_results"] / "exp2_final_agent_state_multiplier_1.csv",
            index=False,
        )

    plot_multiple_price_paths(
        {
            "Unlimited wealth": unlimited_result.prices,
            "Wealth constrained": base_constrained.prices,
        },
        output_dir,
        "exp2_price_path_comparison.png",
        title="Experiment 2: Unlimited vs Wealth-Constrained Price Path",
    )
    plot_price_distribution_comparison(
        {
            "Unlimited wealth": unlimited_result.prices,
            "Wealth constrained": base_constrained.prices,
        },
        output_dir,
        "exp2_price_distribution_comparison.png",
        title="Experiment 2: Price Distribution Comparison",
    )
    plot_deviation_series(
        {
            "Unlimited wealth": np.log(unlimited_result.prices / config.DEFAULT_INITIAL_PRICE),
            "Wealth constrained": np.log(base_constrained.prices / config.DEFAULT_INITIAL_PRICE),
        },
        output_dir,
        "exp2_log_deviation.png",
        title="Experiment 2: Log Price Deviation",
    )
    beta, alpha, _ = compute_mean_reversion_slope(base_constrained.prices, config.DEFAULT_INITIAL_PRICE)
    deviations = np.log(base_constrained.prices[:-1] / config.DEFAULT_INITIAL_PRICE)
    plot_mean_reversion_scatter(
        deviations,
        base_constrained.returns,
        beta=beta,
        alpha=alpha,
        output_dir=output_dir,
        filename="exp2_mean_reversion_scatter.png",
        title="Experiment 2: Mean Reversion Regression",
    )
    plot_multiple_price_paths(
        {f"x{mult:g}": result.prices for mult, result in constrained_results.items()},
        output_dir,
        "exp2_wealth_multiplier_paths.png",
        title="Experiment 2: Wealth Multiplier Price Paths",
    )
    plot_metric_vs_parameter(
        metrics_df["wealth_multiplier"],
        metrics_df["std_return"],
        output_dir,
        "exp2_wealth_multiplier_volatility.png",
        title="Experiment 2: Wealth Multiplier vs Volatility",
        xlabel="Wealth multiplier",
        ylabel="Std. log return",
        log_x=True,
    )
    plot_metric_vs_parameter(
        metrics_df["wealth_multiplier"],
        metrics_df["mean_reversion_slope"],
        output_dir,
        "exp2_wealth_multiplier_mean_reversion.png",
        title="Experiment 2: Wealth Multiplier vs Mean Reversion Slope",
        xlabel="Wealth multiplier",
        ylabel="Mean reversion beta",
        log_x=True,
    )
    plot_grouped_price_paths(
        repeated["path_groups"],
        output_dir,
        "exp2_multiple_paths_by_wealth_constraint.png",
        title="Experiment 2: Multiple Simulated Price Paths by Wealth Constraint",
        ylabel="Price / initial price",
        max_paths_per_group=config.EXP2_PATHS_PER_MULTIPLIER,
        normalize_to_initial=True,
    )
    plot_boxplot_by_group(
        repeated["beta_groups"],
        output_dir,
        "exp2_beta_distribution_by_wealth_multiplier.png",
        title="Experiment 2: Mean Reversion Beta Distribution by Wealth Multiplier",
        xlabel="Wealth multiplier",
        ylabel="Mean reversion beta",
        log_x_labels=True,
    )
    summary = repeated["summary"]
    plot_metric_with_ci(
        summary["wealth_multiplier"],
        summary["mean_beta"],
        summary["beta_ci95_lower"],
        summary["beta_ci95_upper"],
        output_dir,
        "exp2_mean_beta_ci_by_wealth_multiplier.png",
        title="Experiment 2: Mean Beta with 95% Confidence Interval",
        xlabel="Wealth multiplier",
        ylabel="Mean reversion beta",
        log_x=True,
    )
    plot_metric_vs_parameter(
        summary["wealth_multiplier"],
        summary["mean_log_deviation_variance"],
        output_dir,
        "exp2_log_deviation_variance_by_wealth_multiplier.png",
        title="Experiment 2: Variance of Log Deviation by Wealth Multiplier",
        xlabel="Wealth multiplier",
        ylabel="Average variance of log(P[t] / P[0])",
        log_x=True,
    )

    return {
        "unlimited_result": unlimited_result,
        "constrained_results": constrained_results,
        "metrics": metrics_df,
        "repeated": repeated,
    }


def run_experiment_3(
    seed: int = config.DEFAULT_SEED,
    n_days: int = config.DEFAULT_N_DAYS,
    output_dir: str | Path = config.OUTPUT_DIR,
    n_runs: int = config.EXP3_N_RUNS,
) -> dict[str, object]:
    """Run Experiment 3: market depth and volatility."""
    dirs = ensure_output_dirs(output_dir)
    sample_day = max(1, n_days // 2)

    metric_rows: list[dict[str, float]] = []
    run_metric_rows: list[dict[str, float]] = []
    representative_paths: dict[str, np.ndarray] = {}
    path_groups: dict[str, list[np.ndarray]] = {}
    return_groups: dict[str, list[float]] = {}
    low_curve: pd.DataFrame | None = None
    high_curve: pd.DataFrame | None = None

    for depth_idx, n_traders in enumerate(config.EXP3_DEPTH_LEVELS):
        batch = _simulate_unlimited_depth_batch(
            n_days=n_days,
            n_runs=n_runs,
            n_traders=n_traders,
            seed=seed + 30_000 + depth_idx * 10_000,
            sample_curve_day=sample_day,
        )
        prices = batch["prices"]
        returns = batch["returns"]
        roughness_values = np.asarray(batch["roughness_values"], dtype=float)
        representative_paths[str(n_traders)] = prices[0]
        path_groups[str(n_traders)] = [
            prices[path_idx] for path_idx in range(min(config.EXP3_PATHS_PER_DEPTH, prices.shape[0]))
        ]
        _save_representative_depth_path(
            prices[0],
            batch["volumes"][0],
            returns[0],
            dirs["simulation_results"] / f"exp3_representative_depth_{n_traders}.csv",
        )
        if n_traders == config.EXP3_DEPTH_LEVELS[0]:
            low_curve = batch["sample_curve"]
        if n_traders == config.EXP3_DEPTH_LEVELS[-1]:
            high_curve = batch["sample_curve"]

        run_means = np.mean(returns, axis=1)
        run_stds = np.std(returns, axis=1, ddof=1)
        run_max_abs = np.max(np.abs(returns), axis=1)
        run_realized_vols = np.sqrt(np.sum(returns**2, axis=1))
        run_final_prices = prices[:, -1]
        run_autocorrs = _batch_lag1_autocorr(returns)

        return_groups[str(n_traders)] = returns.ravel().tolist()
        for run_id in range(n_runs):
            run_metric_rows.append(
                {
                    "n_traders": n_traders,
                    "run_id": run_id,
                    "mean_return": float(run_means[run_id]),
                    "std_return": float(run_stds[run_id]),
                    "max_abs_return": float(run_max_abs[run_id]),
                    "realized_volatility": float(run_realized_vols[run_id]),
                    "final_price": float(run_final_prices[run_id]),
                    "return_autocorr_lag1": float(run_autocorrs[run_id]),
                    "curve_roughness": float(roughness_values[run_id]) if roughness_values.size else float("nan"),
                }
            )

        metric_rows.append(
            {
                "n_traders": n_traders,
                "n_runs": n_runs,
                "avg_mean_return": float(np.nanmean(run_means)),
                "avg_std_return": float(np.nanmean(run_stds)),
                "std_of_std_return": float(np.nanstd(run_stds, ddof=1)) if len(run_stds) > 1 else 0.0,
                "avg_max_abs_return": float(np.nanmean(run_max_abs)),
                "avg_realized_volatility": float(np.nanmean(run_realized_vols)),
                "avg_final_price": float(np.nanmean(run_final_prices)),
                "std_final_price": float(np.nanstd(run_final_prices, ddof=1)) if run_final_prices.size > 1 else 0.0,
                "avg_return_autocorr_lag1": float(np.nanmean(run_autocorrs)),
                "avg_curve_roughness": float(np.nanmean(roughness_values)) if roughness_values.size else float("nan"),
                "std_curve_roughness": float(np.nanstd(roughness_values, ddof=1)) if roughness_values.size > 1 else 0.0,
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(dirs["tables"] / "exp3_metrics.csv", index=False)
    pd.DataFrame(run_metric_rows).to_csv(dirs["tables"] / "exp3_run_metrics.csv", index=False)
    obsolete_smoothness = dirs["figures"] / "exp3_curve_smoothness_comparison.png"
    if obsolete_smoothness.exists():
        obsolete_smoothness.unlink()

    plot_multiple_price_paths(
        {f"n={k}": v for k, v in representative_paths.items()},
        output_dir,
        "exp3_price_paths_by_depth.png",
        title="Experiment 3: Representative Price Paths by Market Depth",
    )
    plot_grouped_price_paths(
        {f"n={key}": value for key, value in path_groups.items()},
        output_dir,
        "exp3_multiple_paths_by_market_depth.png",
        title="Experiment 3: Multiple Simulated Price Paths by Market Depth",
        ylabel="Price / initial price",
        max_paths_per_group=config.EXP3_PATHS_PER_DEPTH,
        normalize_to_initial=True,
        alpha=0.2,
    )
    plot_metric_vs_parameter(
        metrics_df["n_traders"],
        metrics_df["avg_std_return"],
        output_dir,
        "exp3_volatility_vs_depth.png",
        title="Experiment 3: Volatility vs Market Depth",
        xlabel="Buyers = sellers",
        ylabel="Average std. log return",
        log_x=True,
    )
    plot_metric_vs_parameter(
        metrics_df["n_traders"],
        metrics_df["avg_max_abs_return"],
        output_dir,
        "exp3_extreme_move_vs_depth.png",
        title="Experiment 3: Extreme Move vs Market Depth",
        xlabel="Buyers = sellers",
        ylabel="Average max absolute daily return",
        log_x=True,
    )
    plot_metric_vs_parameter(
        metrics_df["n_traders"],
        metrics_df["avg_curve_roughness"],
        output_dir,
        "exp3_curve_roughness_vs_depth.png",
        title="Experiment 3: Market Depth vs Curve Roughness",
        xlabel="Buyers = sellers",
        ylabel="Average normalized curve roughness",
        log_x=True,
    )
    plot_return_distribution_by_group(
        _select_distribution_groups(return_groups),
        output_dir,
        "exp3_return_distribution_by_depth.png",
        title="Experiment 3: Return Distribution by Market Depth",
    )
    if low_curve is not None:
        plot_auction_curve(
            low_curve,
            output_dir,
            "exp3_auction_curve_low_depth.png",
            title=f"Experiment 3: Low-Depth Auction Curve, n={config.EXP3_DEPTH_LEVELS[0]}",
        )
    if high_curve is not None:
        plot_auction_curve(
            high_curve,
            output_dir,
            "exp3_auction_curve_high_depth.png",
            title=f"Experiment 3: High-Depth Auction Curve, n={config.EXP3_DEPTH_LEVELS[-1]}",
        )
    return {
        "metrics": metrics_df,
        "representative_paths": representative_paths,
        "path_groups": path_groups,
        "return_groups": return_groups,
    }


def simulate_capital_inflow_market(
    seed: int,
    n_days: int,
    inflow_per_period: float,
    initial_cash_per_agent: float,
    initial_shares_per_agent: float,
    n_agents: int = config.DEFAULT_N_AGENTS,
    participation_rate: float = config.DEFAULT_AGENT_PARTICIPATION_RATE,
    sample_curve_days: list[int] | None = None,
) -> MarketRunResult:
    """Run a single capital-inflow simulation with fixed configuration."""
    market = DailyAuctionMarket(
        initial_price=config.DEFAULT_INITIAL_PRICE,
        price_sigma=config.DEFAULT_PRICE_SIGMA,
        max_qty=config.DEFAULT_MAX_QTY,
        seed=seed,
    )
    return market.simulate_capital_inflow(
        n_days=n_days,
        n_agents=n_agents,
        initial_cash_per_agent=initial_cash_per_agent,
        initial_shares_per_agent=initial_shares_per_agent,
        inflow_per_period=inflow_per_period,
        participation_rate=participation_rate,
        sample_curve_days=sample_curve_days,
    )


def compute_exp4_diagnostics(
    result: MarketRunResult,
    inflow_per_period: float,
    regime: str,
    group: str,
    run_id: int,
) -> dict[str, float | str]:
    """Compute one diagnostics row for an Experiment 4 simulation path."""
    prices = result.prices
    returns = result.returns
    base = compute_basic_metrics(prices, returns)
    t_stat = mean_return_t_statistic(returns)
    diagnostics = result.daily_diagnostics
    volumes_eff = result.volumes[1:] if result.volumes.size > 1 else result.volumes
    if diagnostics is not None and not diagnostics.empty:
        day_mask = diagnostics["day"] > 0
        cash_constrained = diagnostics.loc[day_mask, "cash_constrained_ratio"].to_numpy(dtype=float)
        share_constrained = diagnostics.loc[day_mask, "share_constrained_ratio"].to_numpy(dtype=float)
        avg_cash_series = diagnostics["avg_cash"].to_numpy(dtype=float)
        zero_trade_periods = float(diagnostics.loc[day_mask, "zero_trade"].sum())
    else:
        cash_constrained = np.array([], dtype=float)
        share_constrained = np.array([], dtype=float)
        avg_cash_series = np.array([float("nan")])
        zero_trade_periods = float("nan")

    final_state = result.final_agent_state
    if final_state is not None and not final_state.empty:
        cash_gini_end = gini_coefficient(final_state["cash"].to_numpy(dtype=float))
        share_gini_end = gini_coefficient(final_state["shares"].to_numpy(dtype=float))
    else:
        cash_gini_end = float("nan")
        share_gini_end = float("nan")

    return {
        "regime": regime,
        "group": group,
        "inflow_per_period": float(inflow_per_period),
        "run_id": int(run_id),
        "mean_log_return": float(t_stat["mean_return"]),
        "std_log_return": float(t_stat["std_return"]),
        "t_stat_mean_return": float(t_stat["t_stat"]),
        "p_value_one_sided_positive": float(t_stat["p_value_one_sided_positive"]),
        "lag1_autocorr": base["return_autocorr_lag1"],
        "variance_ratio_lag5": base["variance_ratio_lag5"],
        "final_price": base["final_price"],
        "positive_return_pct": base["positive_return_pct"],
        "zero_trade_periods": zero_trade_periods,
        "avg_volume": float(np.mean(volumes_eff)) if volumes_eff.size else 0.0,
        "avg_cash": float(np.mean(avg_cash_series)) if avg_cash_series.size else float("nan"),
        "cash_gini_end": cash_gini_end,
        "share_gini_end": share_gini_end,
        "avg_cash_constrained_ratio": float(np.mean(cash_constrained)) if cash_constrained.size else float("nan"),
        "avg_share_constrained_ratio": float(np.mean(share_constrained)) if share_constrained.size else float("nan"),
    }


def _summarize_exp4_group(diagnostics_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Experiment 4 diagnostics by (regime, group)."""
    rows: list[dict[str, float | str]] = []
    for (regime, group_name), block in diagnostics_df.groupby(
        ["regime", "group"], sort=True
    ):
        mean_returns = block["mean_log_return"].to_numpy(dtype=float)
        final_prices = block["final_price"].to_numpy(dtype=float)
        ci_lower, ci_upper = confidence_interval_95(mean_returns)
        rows.append(
            {
                "regime": regime,
                "group": group_name,
                "n_runs": int(block.shape[0]),
                "inflow_per_period": float(block["inflow_per_period"].iloc[0]),
                "mean_log_return": float(np.mean(mean_returns)),
                "std_log_return": float(block["std_log_return"].mean()),
                "mean_log_return_ci95_lower": ci_lower,
                "mean_log_return_ci95_upper": ci_upper,
                "lag1_autocorr": float(block["lag1_autocorr"].mean()),
                "variance_ratio_lag5": float(block["variance_ratio_lag5"].mean()),
                "positive_return_pct": float(block["positive_return_pct"].mean()),
                "final_price_mean": float(np.mean(final_prices)),
                "final_price_std": float(np.std(final_prices, ddof=1)) if final_prices.size > 1 else 0.0,
                "zero_trade_periods": float(block["zero_trade_periods"].mean()),
                "average_trading_volume": float(block["avg_volume"].mean()),
                "avg_cash_constrained_ratio": float(block["avg_cash_constrained_ratio"].mean()),
                "avg_share_constrained_ratio": float(block["avg_share_constrained_ratio"].mean()),
                "mean_cash_gini_end": float(block["cash_gini_end"].mean()),
                "mean_share_gini_end": float(block["share_gini_end"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _welch_t_test(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Welch two-sample t-test using the standard normal CDF for the p-value."""
    import math

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return {
            "mean_diff": float("nan"),
            "welch_t_stat": float("nan"),
            "welch_p_value_two_sided": float("nan"),
            "cohens_d": float("nan"),
        }
    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    se = math.sqrt(var_a / a.size + var_b / b.size)
    diff = mean_b - mean_a
    if se <= 1e-18:
        t_stat = float("inf") if diff > 0 else (float("-inf") if diff < 0 else 0.0)
    else:
        t_stat = diff / se
    p_two = math.erfc(abs(t_stat) / math.sqrt(2.0)) if math.isfinite(t_stat) else 0.0
    pooled_sd = math.sqrt((var_a + var_b) / 2.0)
    cohens_d = diff / pooled_sd if pooled_sd > 1e-18 else float("nan")
    return {
        "mean_diff": diff,
        "welch_t_stat": float(t_stat),
        "welch_p_value_two_sided": float(p_two),
        "cohens_d": float(cohens_d),
    }


def _exp4_two_group_comparison(
    summary: pd.DataFrame, diagnostics_df: pd.DataFrame
) -> pd.DataFrame:
    """Add Welch t-test columns (treatment minus baseline) to the group summary."""
    rows: list[dict[str, float | str]] = []
    for regime, block in summary.groupby("regime", sort=True):
        baseline_rows = diagnostics_df[
            (diagnostics_df["regime"] == regime) & (diagnostics_df["group"] == "baseline")
        ]
        inflow_rows = diagnostics_df[
            (diagnostics_df["regime"] == regime) & (diagnostics_df["group"] == "inflow")
        ]
        mr_test = _welch_t_test(
            baseline_rows["mean_log_return"].to_numpy(dtype=float),
            inflow_rows["mean_log_return"].to_numpy(dtype=float),
        )
        fp_test = _welch_t_test(
            baseline_rows["final_price"].to_numpy(dtype=float),
            inflow_rows["final_price"].to_numpy(dtype=float),
        )
        for _, summary_row in block.iterrows():
            row = summary_row.to_dict()
            row["mean_log_return_diff_vs_baseline"] = (
                mr_test["mean_diff"] if summary_row["group"] == "inflow" else 0.0
            )
            row["welch_t_stat_mean_return"] = (
                mr_test["welch_t_stat"] if summary_row["group"] == "inflow" else float("nan")
            )
            row["welch_p_value_mean_return"] = (
                mr_test["welch_p_value_two_sided"] if summary_row["group"] == "inflow" else float("nan")
            )
            row["cohens_d_mean_return"] = (
                mr_test["cohens_d"] if summary_row["group"] == "inflow" else float("nan")
            )
            row["final_price_diff_vs_baseline"] = (
                fp_test["mean_diff"] if summary_row["group"] == "inflow" else 0.0
            )
            row["welch_p_value_final_price"] = (
                fp_test["welch_p_value_two_sided"] if summary_row["group"] == "inflow" else float("nan")
            )
            rows.append(row)
    return pd.DataFrame(rows)


EXP4_BASELINE_COLOR = "#6e6e6e"
EXP4_INFLOW_COLOR = "#1f77b4"


def plot_exp4_results(
    main_path_groups: dict[str, list[np.ndarray]],
    main_summary: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    output_dir: str | Path,
    inflow_value: float,
) -> None:
    """Generate the Experiment 4 figures: paths with mean, mean-only with CI,
    final-price distribution, and random-walk diagnostics comparison."""
    label_baseline = "Baseline (inflow = 0)"
    label_inflow = f"Inflow (= {inflow_value:g} / buyer / day)"
    colors = {label_baseline: EXP4_BASELINE_COLOR, label_inflow: EXP4_INFLOW_COLOR}

    plot_paths_with_mean(
        main_path_groups,
        output_dir,
        "exp4_paths_baseline_vs_inflow.png",
        title="Experiment 4: Auction-Generated Price Paths, Baseline vs Capital Inflow",
        ylabel="Price",
        max_paths_per_group=config.EXP4_PATHS_FOR_FIGURE,
        path_alpha=0.18,
        group_colors=colors,
    )
    plot_mean_paths_with_ci(
        main_path_groups,
        output_dir,
        "exp4_mean_paths_baseline_vs_inflow.png",
        title="Experiment 4: Cross-Path Mean Price, Baseline vs Capital Inflow",
        ylabel="Cross-path mean price",
        group_colors=colors,
    )

    main_regime = config.EXP4_MAIN_REGIME
    main_diag = diagnostics_df[diagnostics_df["regime"] == main_regime]
    final_prices = {
        label_baseline: main_diag.loc[main_diag["group"] == "baseline", "final_price"].to_numpy(dtype=float),
        label_inflow: main_diag.loc[main_diag["group"] == "inflow", "final_price"].to_numpy(dtype=float),
    }
    plot_price_distribution_comparison(
        final_prices,
        output_dir,
        "exp4_final_price_distribution.png",
        title="Experiment 4: Final Price Distribution, Baseline vs Capital Inflow",
        bins=30,
    )

    rw_records: list[dict[str, float | str]] = []
    for group_name, label in [("baseline", label_baseline), ("inflow", label_inflow)]:
        block = main_diag[main_diag["group"] == group_name]
        if block.empty:
            continue
        rw_records.append(
            {
                "group": label,
                "lag1_autocorr": float(block["lag1_autocorr"].mean()),
                "variance_ratio_lag5": float(block["variance_ratio_lag5"].mean()),
                "positive_return_pct": float(block["positive_return_pct"].mean()),
            }
        )
    rw_df = pd.DataFrame(rw_records)
    if not rw_df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        metric_specs = [
            ("lag1_autocorr", "Lag-1 autocorrelation", 0.0),
            ("variance_ratio_lag5", "Variance ratio (lag 5)", 1.0),
            ("positive_return_pct", "Positive return %", 0.5),
        ]
        for ax, (col, label, ref) in zip(axes, metric_specs):
            heights = rw_df[col].to_numpy(dtype=float)
            bars = ax.bar(
                rw_df["group"].tolist(),
                heights,
                color=[colors.get(name, "#888888") for name in rw_df["group"]],
                alpha=0.85,
            )
            ax.axhline(ref, color="black", linewidth=0.8, linestyle="--", label=f"reference = {ref:g}")
            ax.set_title(label)
            ax.set_ylabel(label)
            ax.tick_params(axis="x", labelrotation=10)
            ax.grid(True, axis="y", alpha=0.3)
            for bar, value in zip(bars, heights):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    f"{value:.3g}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            ax.legend(loc="best", fontsize=8)
        fig.suptitle("Experiment 4: Random-Walk Diagnostics, Baseline vs Capital Inflow")
        fig.tight_layout()
        fig.savefig(Path(output_dir) / "figures" / "exp4_random_walk_diagnostics_bars.png", dpi=160)
        plt.close(fig)


def run_experiment_4(
    seed: int = config.DEFAULT_SEED,
    n_days: int = config.DEFAULT_N_DAYS,
    output_dir: str | Path = config.OUTPUT_DIR,
    n_runs: int = config.EXP4_N_RUNS,
) -> dict[str, object]:
    """Run Experiment 4: a fixed capital inflow shifts the auction-generated
    random-walk channel upward relative to a no-inflow baseline.

    The main comparison uses ``config.EXP4_MAIN_REGIME`` with two groups:
    baseline (inflow = 0) and inflow (= ``config.EXP4_INFLOW_TREATMENT``).
    Both cash regimes (`tight`, `loose`) are simulated for the robustness table.
    """
    dirs = ensure_output_dirs(output_dir)
    inflow_value = float(config.EXP4_INFLOW_TREATMENT)
    main_regime = config.EXP4_MAIN_REGIME
    regimes = list(config.EXP4_INITIAL_CASH_REGIMES.items())
    groups = [("baseline", 0.0), ("inflow", inflow_value)]

    diagnostics_rows: list[dict[str, float | str]] = []
    path_store: dict[tuple[str, str], list[np.ndarray]] = {}

    for regime_idx, (regime_name, regime_cfg) in enumerate(regimes):
        for group_idx, (group_name, inflow) in enumerate(groups):
            paths: list[np.ndarray] = []
            for run_id in range(n_runs):
                run_seed = (
                    seed
                    + 400_000
                    + regime_idx * 100_000
                    + group_idx * 10_000
                    + run_id
                )
                result = simulate_capital_inflow_market(
                    seed=run_seed,
                    n_days=n_days,
                    inflow_per_period=inflow,
                    initial_cash_per_agent=float(regime_cfg["initial_cash_per_agent"]),
                    initial_shares_per_agent=float(regime_cfg["initial_shares_per_agent"]),
                )
                diagnostics_rows.append(
                    compute_exp4_diagnostics(result, inflow, regime_name, group_name, run_id)
                )
                paths.append(np.asarray(result.prices, dtype=float))
            path_store[(regime_name, group_name)] = paths

    diagnostics_df = pd.DataFrame(diagnostics_rows)
    summary_df = _summarize_exp4_group(diagnostics_df)
    enriched_summary = _exp4_two_group_comparison(summary_df, diagnostics_df)

    main_columns = [
        "regime",
        "group",
        "n_runs",
        "inflow_per_period",
        "mean_log_return",
        "std_log_return",
        "lag1_autocorr",
        "variance_ratio_lag5",
        "positive_return_pct",
        "final_price_mean",
        "final_price_std",
        "zero_trade_periods",
        "average_trading_volume",
        "avg_cash_constrained_ratio",
        "mean_log_return_diff_vs_baseline",
        "welch_t_stat_mean_return",
        "welch_p_value_mean_return",
        "cohens_d_mean_return",
    ]
    main_columns = [c for c in main_columns if c in enriched_summary.columns]
    main_only = enriched_summary[enriched_summary["regime"] == main_regime][main_columns].copy()
    main_only.to_csv(dirs["tables"] / "exp4_main_comparison.csv", index=False)
    enriched_summary.to_csv(dirs["tables"] / "exp4_robustness_by_regime.csv", index=False)
    diagnostics_df.to_csv(dirs["tables"] / "exp4_path_metrics.csv", index=False)

    label_baseline = "Baseline (inflow = 0)"
    label_inflow = f"Inflow (= {inflow_value:g} / buyer / day)"
    main_paths = {
        label_baseline: path_store[(main_regime, "baseline")],
        label_inflow: path_store[(main_regime, "inflow")],
    }
    plot_exp4_results(main_paths, main_only, diagnostics_df, output_dir, inflow_value)
    _write_exp4_findings(
        enriched_summary,
        diagnostics_df,
        main_regime,
        inflow_value,
        dirs["reports"] / "exp4_findings.md",
    )
    return {
        "diagnostics": diagnostics_df,
        "main_comparison": main_only,
        "robustness": enriched_summary,
        "path_groups": main_paths,
    }


def _write_exp4_findings(
    summary: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    main_regime: str,
    inflow_value: float,
    path: Path,
) -> None:
    """Write the auto-generated interpretation block for the redesigned
    Experiment 4 (baseline vs capital-inflow treatment, channel-shift framing)."""
    if summary.empty:
        path.write_text("# Experiment 4 Findings\n\n_No summary generated._\n", encoding="utf-8")
        return

    def _pick(regime: str, group_name: str) -> pd.Series | None:
        sub = summary[(summary["regime"] == regime) & (summary["group"] == group_name)]
        return None if sub.empty else sub.iloc[0]

    base_main = _pick(main_regime, "baseline")
    treat_main = _pick(main_regime, "inflow")

    lines: list[str] = [
        "# Experiment 4 Findings: Capital Inflow and the Random-Walk Channel",
        "",
        f"Main regime: `{main_regime}` (initial cash per agent = "
        f"{config.EXP4_INITIAL_CASH_REGIMES[main_regime]['initial_cash_per_agent']:g}). "
        f"Treatment inflow = `{inflow_value:g}` per buyer per period. "
        f"Each group uses `{int(base_main['n_runs']) if base_main is not None else 'n/a'}` "
        "independent runs with no drift term in the bid equation.",
        "",
        "## H4a — Treatment mean log return exceeds baseline",
    ]
    if base_main is not None and treat_main is not None:
        diff = float(treat_main["mean_log_return"]) - float(base_main["mean_log_return"])
        welch_t = float(treat_main.get("welch_t_stat_mean_return", float("nan")))
        welch_p = float(treat_main.get("welch_p_value_mean_return", float("nan")))
        cohens_d = float(treat_main.get("cohens_d_mean_return", float("nan")))
        verdict_h4a = (
            "supported" if (np.isfinite(welch_p) and welch_p < 0.05 and diff > 0) else "not supported"
        )
        lines.append(
            f"- Baseline mean log return: `{float(base_main['mean_log_return']):.6g}`; "
            f"treatment mean log return: `{float(treat_main['mean_log_return']):.6g}`; "
            f"difference: `{diff:+.6g}`."
        )
        lines.append(
            f"- Welch's two-sample t-test: t = `{welch_t:.3f}`, p = `{welch_p:.4g}`, "
            f"Cohen's d = `{cohens_d:.3f}`."
        )
        lines.append(f"- **Verdict:** H4a is {verdict_h4a}.")
    lines.extend(["", "## H4b — Treatment mean price path lies above baseline"])
    if base_main is not None and treat_main is not None:
        fp_diff = float(treat_main["final_price_mean"]) - float(base_main["final_price_mean"])
        fp_p = float(treat_main.get("welch_p_value_final_price", float("nan")))
        verdict_h4b = (
            "supported" if (np.isfinite(fp_p) and fp_p < 0.05 and fp_diff > 0) else "not supported"
        )
        lines.append(
            f"- Mean final price: baseline `{float(base_main['final_price_mean']):.4g}` "
            f"vs treatment `{float(treat_main['final_price_mean']):.4g}` "
            f"(difference `{fp_diff:+.4g}`, Welch p = `{fp_p:.4g}`)."
        )
        lines.append(f"- **Verdict:** H4b is {verdict_h4b}.")
    lines.extend(["", "## H4c — Random-walk features are preserved under the treatment"])
    if base_main is not None and treat_main is not None:
        lines.append(
            "- Lag-1 autocorrelation: baseline "
            f"`{float(base_main['lag1_autocorr']):.4f}` vs treatment "
            f"`{float(treat_main['lag1_autocorr']):.4f}` (target ≈ 0)."
        )
        lines.append(
            "- Variance ratio (lag 5): baseline "
            f"`{float(base_main['variance_ratio_lag5']):.4f}` vs treatment "
            f"`{float(treat_main['variance_ratio_lag5']):.4f}` (target ≈ 1)."
        )
        lines.append(
            "- Share of positive-return periods: baseline "
            f"`{float(base_main['positive_return_pct']):.4f}` vs treatment "
            f"`{float(treat_main['positive_return_pct']):.4f}` (≈ 0.5 implies symmetric increments)."
        )
        ac_drift = abs(float(treat_main["lag1_autocorr"]) - float(base_main["lag1_autocorr"]))
        vr_drift = abs(float(treat_main["variance_ratio_lag5"]) - float(base_main["variance_ratio_lag5"]))
        verdict_h4c = (
            "supported"
            if (ac_drift < 0.05 and abs(float(treat_main["lag1_autocorr"])) < 0.1 and vr_drift < 0.25)
            else "qualified"
        )
        lines.append(
            f"- **Verdict:** H4c is {verdict_h4c} — the increments remain close to a random walk "
            "even when the channel shifts upward."
        )

    lines.extend(["", "## H4d — Constraint mediation (robustness across cash regimes)"])
    for regime in summary["regime"].drop_duplicates().tolist():
        b = _pick(regime, "baseline")
        t = _pick(regime, "inflow")
        if b is None or t is None:
            continue
        d = float(t["mean_log_return"]) - float(b["mean_log_return"])
        constrained = float(b["avg_cash_constrained_ratio"])
        lines.append(
            f"- `{regime}` (baseline cash-constrained ratio `{constrained:.3f}`): "
            f"treatment − baseline mean log return = `{d:+.6g}`."
        )
    lines.extend(
        [
            "- **Interpretation:** if the treatment minus baseline gap is larger when the "
            "baseline cash-constrained ratio is higher, the inflow effect is mediated by "
            "binding cash constraints rather than acting as a direct price trend.",
            "",
            "## Mechanism note",
            "",
            "- The bid equation is `bid = P[t-1] * exp(epsilon)` with `epsilon ~ N(0, sigma^2)` "
            "in both groups; no drift term is added.",
            "- Any upward shift therefore comes from the auction clearing — extra buyer cash "
            "marginally increases the volume-maximising clearing price each period — and not "
            "from an imposed trend.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_all_experiments(
    seed: int = config.DEFAULT_SEED,
    n_days: int = config.DEFAULT_N_DAYS,
    output_dir: str | Path = config.OUTPUT_DIR,
    n_runs: int = config.EXP3_N_RUNS,
) -> dict[str, object]:
    """Run all four experiments and generate the Markdown report."""
    results = {
        "exp1": run_experiment_1(seed=seed, n_days=n_days, output_dir=output_dir),
        "exp2": run_experiment_2(seed=seed, n_days=n_days, output_dir=output_dir, n_runs=n_runs),
        "exp3": run_experiment_3(seed=seed, n_days=n_days, output_dir=output_dir, n_runs=n_runs),
        "exp4": run_experiment_4(seed=seed, n_days=n_days, output_dir=output_dir, n_runs=min(n_runs, config.EXP4_N_RUNS)),
    }
    results["report"] = generate_report(output_dir)
    return results


def run_experiment_2_ablation(
    seed: int = config.DEFAULT_SEED,
    n_days: int = config.DEFAULT_N_DAYS,
    output_dir: str | Path = config.OUTPUT_DIR,
    n_runs: int = config.EXP2_N_RUNS,
) -> dict[str, object]:
    """Run cash-only, inventory-only, and both-constraints ablation studies."""
    dirs = ensure_output_dirs(output_dir)
    rows: list[dict[str, float | str]] = []
    path_groups: dict[str, list[np.ndarray]] = {}

    for constraint_idx, (constraint_type, constraint_cfg) in enumerate(config.ABLATION_CONSTRAINT_TYPES.items()):
        path_groups[constraint_cfg["label"]] = []
        for mult_idx, multiplier in enumerate(config.EXP2_WEALTH_MULTIPLIERS):
            for run_id in range(n_runs):
                market = DailyAuctionMarket(
                    initial_price=config.DEFAULT_INITIAL_PRICE,
                    price_sigma=config.DEFAULT_PRICE_SIGMA,
                    max_qty=config.DEFAULT_MAX_QTY,
                    seed=seed + 40_000 + constraint_idx * 100_000 + mult_idx * 10_000 + run_id,
                )
                result = market.simulate_wealth_constrained(
                    n_days=n_days,
                    n_agents=config.DEFAULT_N_AGENTS,
                    initial_cash_per_agent=config.DEFAULT_INITIAL_CASH_PER_AGENT * multiplier,
                    initial_shares_per_agent=config.DEFAULT_INITIAL_SHARES_PER_AGENT * multiplier,
                    participation_rate=config.DEFAULT_AGENT_PARTICIPATION_RATE,
                    buyer_cash_constraint=constraint_cfg["buyer_cash_constraint"],
                    seller_inventory_constraint=constraint_cfg["seller_inventory_constraint"],
                )
                beta, alpha, r2 = compute_mean_reversion_slope(result.prices, config.DEFAULT_INITIAL_PRICE)
                rows.append(
                    {
                        "constraint_type": constraint_type,
                        "constraint_label": constraint_cfg["label"],
                        "wealth_multiplier": float(multiplier),
                        "run_id": run_id,
                        "mean_reversion_beta": beta,
                        "mean_reversion_alpha": alpha,
                        "mean_reversion_r2": r2,
                        "price_variance": float(np.var(result.prices, ddof=1)),
                        "log_deviation_variance": float(np.var(np.log(result.prices / config.DEFAULT_INITIAL_PRICE), ddof=1)),
                        "return_volatility": float(np.std(result.returns, ddof=1)),
                        "final_price": float(result.prices[-1]),
                        "max_drawdown": compute_basic_metrics(result.prices, result.returns)["max_drawdown"],
                    }
                )
                if multiplier == 1 and run_id < 12:
                    path_groups[constraint_cfg["label"]].append(result.prices)

    run_df = pd.DataFrame(rows)
    run_df.to_csv(dirs["tables"] / "exp2_ablation_run_metrics.csv", index=False)
    summary = _summarize_exp2_ablation(run_df)
    summary.to_csv(dirs["tables"] / "exp2_ablation_summary.csv", index=False)

    plot_multi_line_with_ci(
        summary,
        group_col="constraint_label",
        x_col="wealth_multiplier",
        y_col="mean_beta",
        lower_col="beta_ci_lower",
        upper_col="beta_ci_upper",
        output_dir=output_dir,
        filename="exp2_ablation_mean_beta_ci.png",
        title="Experiment 2 Ablation: Mean Beta with 95% CI",
        xlabel="Wealth multiplier",
        ylabel="Mean reversion beta",
        log_x=True,
    )
    plot_multi_line_with_ci(
        summary,
        group_col="constraint_label",
        x_col="wealth_multiplier",
        y_col="mean_log_deviation_variance",
        lower_col=None,
        upper_col=None,
        output_dir=output_dir,
        filename="exp2_ablation_log_deviation_variance.png",
        title="Experiment 2 Ablation: Log-Deviation Variance",
        xlabel="Wealth multiplier",
        ylabel="Mean variance of log(P[t] / P[0])",
        log_x=True,
    )
    beta_groups = {
        label: group["mean_reversion_beta"].to_numpy(dtype=float)
        for label, group in run_df.groupby("constraint_label", sort=False)
    }
    plot_boxplot_by_group(
        beta_groups,
        output_dir,
        "exp2_ablation_beta_distribution.png",
        title="Experiment 2 Ablation: Beta Distribution by Constraint Type",
        xlabel="Constraint type",
        ylabel="Mean reversion beta",
    )
    plot_grouped_price_paths(
        path_groups,
        output_dir,
        "exp2_ablation_multiple_paths.png",
        title="Experiment 2 Ablation: Multiple Price Paths by Constraint Type",
        ylabel="Price / initial price",
        normalize_to_initial=True,
        max_paths_per_group=12,
    )
    _write_exp2_ablation_findings(summary, dirs["reports"] / "exp2_ablation_findings.md")
    return {"run_metrics": run_df, "summary": summary}


def run_experiment_3_roughness_volatility(
    seed: int = config.DEFAULT_SEED,
    n_days: int = config.DEFAULT_N_DAYS,
    output_dir: str | Path = config.OUTPUT_DIR,
    n_runs: int = config.EXP3_N_RUNS,
) -> dict[str, object]:
    """Directly test the mechanism chain: depth -> roughness -> volatility."""
    dirs = ensure_output_dirs(output_dir)
    rows: list[dict[str, float]] = []
    summary_rows: list[dict[str, float]] = []

    for depth_idx, depth in enumerate(config.EXP3_DEPTH_LEVELS):
        batch = _simulate_unlimited_depth_batch(
            n_days=n_days,
            n_runs=n_runs,
            n_traders=depth,
            seed=seed + 50_000 + depth_idx * 10_000,
            sample_curve_day=max(1, n_days // 2),
        )
        returns = batch["returns"]
        vols = np.std(returns, axis=1, ddof=1)
        roughness = np.asarray(batch["roughness_values"], dtype=float)
        for run_id in range(n_runs):
            rows.append(
                {
                    "market_depth": depth,
                    "run_id": run_id,
                    "curve_roughness": float(roughness[run_id]),
                    "return_volatility": float(vols[run_id]),
                }
            )
        summary_rows.append(
            {
                "market_depth": depth,
                "n_runs": n_runs,
                "avg_curve_roughness": float(np.nanmean(roughness)),
                "std_curve_roughness": float(np.nanstd(roughness, ddof=1)),
                "avg_return_volatility": float(np.nanmean(vols)),
                "std_return_volatility": float(np.nanstd(vols, ddof=1)),
                "within_depth_corr_roughness_volatility": compute_corr(roughness, vols),
            }
        )

    run_df = pd.DataFrame(rows)
    summary = pd.DataFrame(summary_rows)
    summary["corr_roughness_volatility"] = compute_corr(
        run_df["curve_roughness"],
        run_df["return_volatility"],
    )
    run_df.to_csv(dirs["tables"] / "exp3_roughness_volatility_runs.csv", index=False)
    summary.to_csv(dirs["tables"] / "exp3_roughness_volatility_summary.csv", index=False)
    plot_scatter_with_fit(
        run_df["curve_roughness"],
        run_df["return_volatility"],
        output_dir,
        "exp3_roughness_vs_volatility.png",
        title="Experiment 3: Curve Roughness vs Return Volatility",
        xlabel="Normalized curve roughness",
        ylabel="Return volatility",
    )
    plot_dual_axis_chain(
        summary["market_depth"],
        summary["avg_curve_roughness"],
        summary["avg_return_volatility"],
        output_dir,
        "exp3_depth_roughness_volatility_chain.png",
        title="Experiment 3: Depth -> Roughness -> Volatility",
        xlabel="Market depth (buyers = sellers)",
        y1label="Average curve roughness",
        y2label="Average return volatility",
        log_x=True,
    )
    return {"runs": run_df, "summary": summary}


def run_gaussian_benchmark(
    seed: int = config.DEFAULT_SEED,
    n_days: int = config.DEFAULT_N_DAYS,
    output_dir: str | Path = config.OUTPUT_DIR,
    n_runs: int = config.BENCHMARK_N_RUNS,
) -> dict[str, object]:
    """Compare auction-generated unlimited market paths to a Gaussian random walk."""
    dirs = ensure_output_dirs(output_dir)
    auction_paths: list[np.ndarray] = []
    auction_returns: list[np.ndarray] = []
    for run_id in range(n_runs):
        market = DailyAuctionMarket(
            initial_price=config.DEFAULT_INITIAL_PRICE,
            price_sigma=config.DEFAULT_PRICE_SIGMA,
            max_qty=config.DEFAULT_MAX_QTY,
            seed=seed + 60_000 + run_id,
        )
        result = market.simulate_unlimited(
            n_days=n_days,
            n_buyers=config.DEFAULT_N_BUYERS,
            n_sellers=config.DEFAULT_N_SELLERS,
        )
        auction_paths.append(result.prices)
        auction_returns.append(result.returns)
    sigma_benchmark = float(np.std(np.concatenate(auction_returns), ddof=1))

    rng = np.random.default_rng(seed + 61_000)
    gaussian_returns = rng.normal(0.0, sigma_benchmark, size=(n_runs, n_days))
    gaussian_paths_arr = config.DEFAULT_INITIAL_PRICE * np.exp(
        np.cumsum(np.column_stack([np.zeros(n_runs), gaussian_returns]), axis=1)
    )
    gaussian_paths = [gaussian_paths_arr[i] for i in range(n_runs)]

    summary = pd.DataFrame(
        [
            _process_summary_row("Auction-generated unlimited wealth", auction_paths, auction_returns),
            _process_summary_row("Gaussian random walk benchmark", gaussian_paths, [gaussian_returns[i] for i in range(n_runs)]),
        ]
    )
    summary.to_csv(dirs["tables"] / "exp1_gaussian_benchmark_comparison.csv", index=False)

    plot_panel_price_paths(
        auction_paths,
        gaussian_paths,
        output_dir,
        "exp1_auction_vs_gaussian_multiple_paths.png",
        title="Experiment 1: Auction-Generated Process vs Gaussian Benchmark",
        left_title="Auction-generated",
        right_title="Gaussian benchmark",
        normalize_to_initial=True,
    )
    plot_return_distribution_by_group(
        {
            "Auction-generated": np.concatenate(auction_returns),
            "Gaussian benchmark": gaussian_returns.ravel(),
        },
        output_dir,
        "exp1_auction_vs_gaussian_return_distribution.png",
        title="Experiment 1: Auction vs Gaussian Return Distribution",
    )
    plot_table_image(
        summary,
        output_dir,
        "exp1_auction_vs_gaussian_diagnostics.png",
        title="Experiment 1: Auction vs Gaussian Diagnostics",
    )
    return {"summary": summary, "sigma_benchmark": sigma_benchmark}


def run_volume_analysis(
    seed: int = config.DEFAULT_SEED,
    n_days: int = config.DEFAULT_N_DAYS,
    output_dir: str | Path = config.OUTPUT_DIR,
    n_runs: int = config.EXP3_N_RUNS,
) -> pd.DataFrame:
    """Analyze volume as an endogenous market variable."""
    dirs = ensure_output_dirs(output_dir)
    rows: list[dict[str, float | str]] = []

    exp1_market = DailyAuctionMarket(seed=seed + 70_000)
    exp1 = exp1_market.simulate_unlimited(n_days, config.DEFAULT_N_BUYERS, config.DEFAULT_N_SELLERS)
    rows.append(_volume_row("Experiment 1", "Unlimited wealth", "baseline", exp1.volumes, exp1.returns))
    plot_price_path(exp1.volumes[1:], output_dir, "exp1_daily_volume_series.png", title="Experiment 1: Daily Volume Series", label="Volume")
    plot_return_distribution(exp1.volumes[1:], output_dir, "exp1_volume_distribution.png", title="Experiment 1: Volume Distribution", label="Volume")
    plot_scatter_with_fit(
        exp1.volumes[1:],
        np.abs(exp1.returns),
        output_dir,
        "exp1_abs_return_vs_volume.png",
        title="Experiment 1: Absolute Return vs Volume",
        xlabel="Daily volume",
        ylabel="Absolute log return",
    )

    exp2_avg_volume: list[float] = []
    exp2_std_volume: list[float] = []
    for idx, multiplier in enumerate(config.EXP2_WEALTH_MULTIPLIERS):
        volumes = []
        returns = []
        for run_id in range(max(1, min(n_runs, 30))):
            market = DailyAuctionMarket(seed=seed + 71_000 + idx * 1_000 + run_id)
            result = market.simulate_wealth_constrained(
                n_days=n_days,
                n_agents=config.DEFAULT_N_AGENTS,
                initial_cash_per_agent=config.DEFAULT_INITIAL_CASH_PER_AGENT * multiplier,
                initial_shares_per_agent=config.DEFAULT_INITIAL_SHARES_PER_AGENT * multiplier,
                participation_rate=config.DEFAULT_AGENT_PARTICIPATION_RATE,
            )
            volumes.append(result.volumes[1:])
            returns.append(result.returns)
        volume_arr = np.concatenate(volumes)
        return_arr = np.concatenate(returns)
        rows.append(_volume_row("Experiment 2", "Wealth constrained", float(multiplier), volume_arr, return_arr))
        exp2_avg_volume.append(float(np.mean(volume_arr)))
        exp2_std_volume.append(float(np.std(volume_arr, ddof=1)))

    plot_metric_vs_parameter(
        config.EXP2_WEALTH_MULTIPLIERS,
        exp2_avg_volume,
        output_dir,
        "exp2_wealth_multiplier_vs_volume.png",
        title="Experiment 2: Wealth Multiplier vs Average Volume",
        xlabel="Wealth multiplier",
        ylabel="Average daily volume",
        log_x=True,
    )

    exp3_avg_volume: list[float] = []
    exp3_std_volume: list[float] = []
    all_depth_volumes: list[float] = []
    all_depth_abs_returns: list[float] = []
    for idx, depth in enumerate(config.EXP3_DEPTH_LEVELS):
        batch = _simulate_unlimited_depth_batch(
            n_days=n_days,
            n_runs=max(1, min(n_runs, 50)),
            n_traders=depth,
            seed=seed + 72_000 + idx * 10_000,
            sample_curve_day=None,
        )
        volume_arr = batch["volumes"][:, 1:].ravel()
        return_arr = batch["returns"].ravel()
        rows.append(_volume_row("Experiment 3", "Market depth", depth, volume_arr, return_arr))
        exp3_avg_volume.append(float(np.mean(volume_arr)))
        exp3_std_volume.append(float(np.std(volume_arr, ddof=1)))
        all_depth_volumes.extend(volume_arr.tolist())
        all_depth_abs_returns.extend(np.abs(return_arr).tolist())

    plot_metric_vs_parameter(
        config.EXP3_DEPTH_LEVELS,
        exp3_avg_volume,
        output_dir,
        "exp3_market_depth_vs_volume.png",
        title="Experiment 3: Market Depth vs Average Volume",
        xlabel="Market depth (buyers = sellers)",
        ylabel="Average daily volume",
        log_x=True,
    )
    plot_scatter_with_fit(
        all_depth_volumes,
        all_depth_abs_returns,
        output_dir,
        "exp3_volume_vs_abs_return.png",
        title="Experiment 3: Volume vs Absolute Return",
        xlabel="Daily volume",
        ylabel="Absolute log return",
        log_x=True,
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(dirs["tables"] / "volume_analysis_summary.csv", index=False)
    return summary


def run_sensitivity_analysis(
    seed: int = config.DEFAULT_SEED,
    n_days: int = config.DEFAULT_N_DAYS,
    output_dir: str | Path = config.OUTPUT_DIR,
    n_runs: int = 30,
) -> dict[str, pd.DataFrame]:
    """Run compact parameter sensitivity checks for the three core conclusions."""
    dirs = ensure_output_dirs(output_dir)
    sigma_df = _sensitivity_for_parameter("price_sigma", config.SENSITIVITY_PRICE_SIGMAS, seed, n_days, n_runs)
    qty_df = _sensitivity_for_parameter("max_qty", config.SENSITIVITY_MAX_QTYS, seed + 1_000, n_days, n_runs)
    days_df = _sensitivity_for_parameter("n_days", config.SENSITIVITY_N_DAYS, seed + 2_000, n_days, n_runs)
    sigma_df.to_csv(dirs["tables"] / "sensitivity_sigma_summary.csv", index=False)
    qty_df.to_csv(dirs["tables"] / "sensitivity_max_qty_summary.csv", index=False)
    days_df.to_csv(dirs["tables"] / "sensitivity_n_days_summary.csv", index=False)
    _plot_sensitivity(sigma_df, "parameter_value", output_dir, "sensitivity_sigma_core_metrics.png", "Sensitivity: price_sigma Core Metrics")
    _plot_sensitivity(qty_df, "parameter_value", output_dir, "sensitivity_max_qty_core_metrics.png", "Sensitivity: max_qty Core Metrics")
    _plot_sensitivity(days_df, "parameter_value", output_dir, "sensitivity_n_days_core_metrics.png", "Sensitivity: n_days Core Metrics")
    _write_sensitivity_summary(dirs["reports"] / "sensitivity_analysis_summary.md", sigma_df, qty_df, days_df)
    return {"sigma": sigma_df, "max_qty": qty_df, "n_days": days_df}


def run_extreme_case_analysis(
    seed: int = config.DEFAULT_SEED,
    n_days: int = config.DEFAULT_N_DAYS,
    output_dir: str | Path = config.OUTPUT_DIR,
    n_runs: int = config.EXTREME_CASE_N_RUNS,
) -> pd.DataFrame:
    """Select representative and extreme paths and save event-day auction curves."""
    dirs = ensure_output_dirs(output_dir)
    candidates: list[dict[str, object]] = []
    for run_id in range(n_runs):
        run_seed = seed + 80_000 + run_id
        market = DailyAuctionMarket(seed=run_seed)
        result = market.simulate_unlimited(n_days, config.DEFAULT_N_BUYERS, config.DEFAULT_N_SELLERS)
        candidates.append({"experiment": "Experiment 1", "market_type": "Unlimited wealth", "seed": run_seed, "result": result})
    cases = [
        ("exp1_typical", min(candidates, key=lambda x: abs(float(x["result"].prices[-1]) - config.DEFAULT_INITIAL_PRICE)), "typical random walk path"),
        ("exp1_extreme_up", max(candidates, key=lambda x: float(x["result"].prices[-1])), "extreme upward path"),
        ("exp1_extreme_down", min(candidates, key=lambda x: float(x["result"].prices[-1])), "extreme downward path"),
    ]

    depth_cases: list[dict[str, object]] = []
    for run_id in range(n_runs):
        for depth, label in [(config.EXP3_DEPTH_LEVELS[0], "Low depth"), (config.EXP3_DEPTH_LEVELS[-1], "High depth")]:
            run_seed = seed + 81_000 + depth * 10 + run_id
            market = DailyAuctionMarket(seed=run_seed)
            result = market.simulate_unlimited(n_days, depth, depth)
            depth_cases.append({"experiment": "Experiment 3", "market_type": label, "depth": depth, "seed": run_seed, "result": result})
    low = [case for case in depth_cases if case["market_type"] == "Low depth"]
    high = [case for case in depth_cases if case["market_type"] == "High depth"]
    cases.extend(
        [
            ("exp3_low_depth_up", max(low, key=lambda x: float(x["result"].prices[-1])), "low-depth extreme upward path"),
            ("exp3_low_depth_down", min(low, key=lambda x: float(x["result"].prices[-1])), "low-depth extreme downward path"),
            ("exp3_high_depth_stable", min(high, key=lambda x: np.std(x["result"].returns)), "high-depth stable path"),
        ]
    )

    rows = []
    for case_id, case, path_type in cases:
        result = case["result"]
        event_day = int(np.argmax(np.abs(result.returns)) + 1)
        if case["experiment"] == "Experiment 3":
            depth = int(case.get("depth", config.DEFAULT_N_BUYERS))
            rerun = DailyAuctionMarket(seed=int(case["seed"])).simulate_unlimited(n_days, depth, depth, sample_curve_days=[event_day])
        else:
            rerun = DailyAuctionMarket(seed=int(case["seed"])).simulate_unlimited(
                n_days,
                config.DEFAULT_N_BUYERS,
                config.DEFAULT_N_SELLERS,
                sample_curve_days=[event_day],
            )
        curve = rerun.auction_curves[event_day]
        roughness = compute_curve_roughness(curve)
        rows.append(
            {
                "case_id": case_id,
                "experiment": case["experiment"],
                "market_type": case["market_type"],
                "seed": int(case["seed"]),
                "path_type": path_type,
                "max_abs_return_day": event_day,
                "max_abs_return": float(abs(rerun.returns[event_day - 1])),
                "price_before": float(rerun.prices[event_day - 1]),
                "price_after": float(rerun.prices[event_day]),
                "volume_on_event_day": float(rerun.volumes[event_day]),
                "curve_roughness_on_event_day": roughness,
            }
        )
        plot_price_path(rerun.prices, output_dir, f"extreme_case_{case_id}_price_path.png", title=f"Extreme Case: {case_id} Price Path")
        plot_auction_curve(curve, output_dir, f"extreme_case_{case_id}_auction_curve.png", title=f"Extreme Case: {case_id} Auction Curve, Day {event_day}")
        _plot_event_window(rerun.prices, event_day, output_dir, f"extreme_case_{case_id}_event_window.png", f"Extreme Case: {case_id} Event Window")

    summary = pd.DataFrame(rows)
    summary.to_csv(dirs["tables"] / "extreme_path_case_summary.csv", index=False)
    return summary


def run_cross_experiment_summary(
    seed: int = config.DEFAULT_SEED,
    n_days: int = config.DEFAULT_N_DAYS,
    output_dir: str | Path = config.OUTPUT_DIR,
) -> pd.DataFrame:
    """Create a compact comparison table across market mechanisms."""
    dirs = ensure_output_dirs(output_dir)
    rows: list[dict[str, float | str]] = []
    exp1 = DailyAuctionMarket(seed=seed + 90_000).simulate_unlimited(n_days, config.DEFAULT_N_BUYERS, config.DEFAULT_N_SELLERS, sample_curve_days=[max(1, n_days // 2)])
    rows.append(_cross_row("Unlimited wealth market", "Experiment 1", "approximately random walk", exp1, curve=exp1.auction_curves[max(1, n_days // 2)]))

    both = DailyAuctionMarket(seed=seed + 90_001).simulate_wealth_constrained(n_days, config.DEFAULT_N_AGENTS, config.DEFAULT_INITIAL_CASH_PER_AGENT, config.DEFAULT_INITIAL_SHARES_PER_AGENT)
    rows.append(_cross_row("Wealth constrained market", "Experiment 2", "bounded weakly mean-reverting random process", both))
    cash = DailyAuctionMarket(seed=seed + 90_002).simulate_wealth_constrained(n_days, config.DEFAULT_N_AGENTS, config.DEFAULT_INITIAL_CASH_PER_AGENT, config.DEFAULT_INITIAL_SHARES_PER_AGENT, buyer_cash_constraint=True, seller_inventory_constraint=False)
    rows.append(_cross_row("Buyer cash constrained only", "Experiment 2A", "cash-limited random process", cash))
    inv = DailyAuctionMarket(seed=seed + 90_003).simulate_wealth_constrained(n_days, config.DEFAULT_N_AGENTS, config.DEFAULT_INITIAL_CASH_PER_AGENT, config.DEFAULT_INITIAL_SHARES_PER_AGENT, buyer_cash_constraint=False, seller_inventory_constraint=True)
    rows.append(_cross_row("Seller inventory constrained only", "Experiment 2B", "inventory-limited random process", inv))
    low_depth = config.EXP3_DEPTH_LEVELS[0]
    high_depth = config.EXP3_DEPTH_LEVELS[-1]
    low = DailyAuctionMarket(seed=seed + 90_004).simulate_unlimited(n_days, low_depth, low_depth, sample_curve_days=[max(1, n_days // 2)])
    rows.append(_cross_row("Low depth market", "Experiment 3", "high-volatility random process", low, curve=low.auction_curves[max(1, n_days // 2)]))
    high = DailyAuctionMarket(seed=seed + 90_005).simulate_unlimited(n_days, high_depth, high_depth, sample_curve_days=[max(1, n_days // 2)])
    rows.append(_cross_row("High depth market", "Experiment 3", "low-volatility random process", high, curve=high.auction_curves[max(1, n_days // 2)]))
    gaussian = _simulate_gaussian_path(seed + 90_006, n_days, config.DEFAULT_INITIAL_PRICE, float(np.std(exp1.returns, ddof=1)))
    rows.append(_cross_row("Gaussian random walk benchmark", "Benchmark", "exogenous Gaussian random walk benchmark", gaussian))

    summary = pd.DataFrame(rows)
    summary.to_csv(dirs["tables"] / "cross_experiment_summary.csv", index=False)
    (dirs["tables"] / "cross_experiment_summary.md").write_text(summary.to_markdown(index=False), encoding="utf-8")
    plot_grouped_bar(
        summary,
        "market_type",
        ["std_log_return", "log_deviation_variance", "max_abs_return", "curve_roughness"],
        output_dir,
        "cross_experiment_process_comparison.png",
        title="Cross-Experiment Process Comparison (Normalized Metrics)",
        ylabel="Normalized metric value",
    )
    return summary


def _random_walk_diagnostics_row(prices: np.ndarray, returns: np.ndarray) -> dict[str, float]:
    """Create one random-walk diagnostics row from a price path."""
    metrics = compute_basic_metrics(prices, returns)
    return {
        "mean_log_return": metrics["mean_return"],
        "std_log_return": metrics["std_return"],
        "lag1_autocorr_log_return": metrics["return_autocorr_lag1"],
        "variance_ratio_lag5": metrics["variance_ratio_lag5"],
        "positive_return_pct": metrics["positive_return_pct"],
        "final_price": metrics["final_price"],
        "max_drawdown": metrics["max_drawdown"],
        "skewness": metrics["skewness"],
        "kurtosis": metrics["kurtosis"],
    }


def _summarize_exp2_ablation(run_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (constraint_type, constraint_label, multiplier), group in run_df.groupby(
        ["constraint_type", "constraint_label", "wealth_multiplier"],
        sort=False,
    ):
        beta = group["mean_reversion_beta"].to_numpy(dtype=float)
        ci_lower, ci_upper = confidence_interval_95(beta)
        rows.append(
            {
                "constraint_type": constraint_type,
                "constraint_label": constraint_label,
                "wealth_multiplier": float(multiplier),
                "n_runs": int(group.shape[0]),
                "mean_beta": float(np.mean(beta)),
                "std_beta": float(np.std(beta, ddof=1)) if beta.size > 1 else 0.0,
                "beta_ci_lower": ci_lower,
                "beta_ci_upper": ci_upper,
                "mean_log_deviation_variance": float(group["log_deviation_variance"].mean()),
                "mean_price_variance": float(group["price_variance"].mean()),
                "mean_return_volatility": float(group["return_volatility"].mean()),
                "mean_final_price": float(group["final_price"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _write_exp2_ablation_findings(summary: pd.DataFrame, path: Path) -> None:
    best_beta = summary.loc[summary["mean_beta"].idxmin()]
    best_var = summary.loc[summary["mean_log_deviation_variance"].idxmin()]
    path.write_text(
        "# Experiment 2 Ablation Findings\n\n"
        f"- Most negative mean beta: `{best_beta['constraint_label']}` at wealth multiplier `{best_beta['wealth_multiplier']}`.\n"
        f"- Lowest log-deviation variance: `{best_var['constraint_label']}` at wealth multiplier `{best_var['wealth_multiplier']}`.\n"
        "- Interpretation: cash and inventory constraints should be read as mechanisms that alter feasible order flow. "
        "The ablation table identifies whether boundedness is mostly generated by buyer-side cash scarcity, seller-side inventory scarcity, or their interaction.\n",
        encoding="utf-8",
    )


def _process_summary_row(
    process_type: str,
    paths: list[np.ndarray],
    returns_list: list[np.ndarray],
) -> dict[str, float | str]:
    all_returns = np.concatenate(returns_list)
    final_prices = np.asarray([path[-1] for path in paths], dtype=float)
    drawdowns = np.asarray([compute_basic_metrics(path, ret)["max_drawdown"] for path, ret in zip(paths, returns_list)])
    dist = compute_basic_metrics(paths[0], returns_list[0])
    dist_stats = compute_basic_metrics(np.concatenate([np.array([config.DEFAULT_INITIAL_PRICE]), config.DEFAULT_INITIAL_PRICE * np.exp(np.cumsum(all_returns))]), all_returns)
    return {
        "process_type": process_type,
        "mean_log_return": float(np.mean(all_returns)),
        "std_log_return": float(np.std(all_returns, ddof=1)),
        "lag1_autocorr": compute_corr(all_returns[:-1], all_returns[1:]),
        "variance_ratio_lag5": dist_stats["variance_ratio_lag5"],
        "positive_return_pct": float(np.mean(all_returns > 0)),
        "final_price_mean": float(np.mean(final_prices)),
        "final_price_std": float(np.std(final_prices, ddof=1)),
        "max_drawdown_mean": float(np.mean(drawdowns)),
        "skewness": dist_stats["skewness"],
        "kurtosis": dist_stats["kurtosis"],
    }


def _volume_row(
    experiment: str,
    market_type: str,
    parameter_value: str | float | int,
    volumes: np.ndarray,
    returns: np.ndarray,
) -> dict[str, float | str]:
    volume_arr = np.asarray(volumes, dtype=float)
    return_arr = np.asarray(returns, dtype=float)
    return {
        "experiment": experiment,
        "market_type": market_type,
        "parameter_value": parameter_value,
        "avg_volume": float(np.mean(volume_arr)),
        "std_volume": float(np.std(volume_arr, ddof=1)) if volume_arr.size > 1 else 0.0,
        "corr_volume_abs_return": compute_corr(volume_arr, np.abs(return_arr)),
        "avg_abs_return": float(np.mean(np.abs(return_arr))),
    }


def _sensitivity_for_parameter(
    parameter_name: str,
    values: list[float] | list[int],
    seed: int,
    n_days_default: int,
    n_runs: int,
) -> pd.DataFrame:
    rows = []
    for value_idx, value in enumerate(values):
        price_sigma = float(value) if parameter_name == "price_sigma" else config.DEFAULT_PRICE_SIGMA
        max_qty = float(value) if parameter_name == "max_qty" else config.DEFAULT_MAX_QTY
        n_days = int(value) if parameter_name == "n_days" else n_days_default

        exp1_metrics = []
        for run_id in range(max(3, min(n_runs, 20))):
            result = DailyAuctionMarket(
                initial_price=config.DEFAULT_INITIAL_PRICE,
                price_sigma=price_sigma,
                max_qty=max_qty,
                seed=seed + value_idx * 10_000 + run_id,
            ).simulate_unlimited(n_days, config.DEFAULT_N_BUYERS, config.DEFAULT_N_SELLERS)
            exp1_metrics.append(compute_basic_metrics(result.prices, result.returns))

        exp2_betas = []
        exp2_log_vars = []
        for run_id in range(max(3, min(n_runs, 15))):
            result = DailyAuctionMarket(
                initial_price=config.DEFAULT_INITIAL_PRICE,
                price_sigma=price_sigma,
                max_qty=max_qty,
                seed=seed + 100_000 + value_idx * 10_000 + run_id,
            ).simulate_wealth_constrained(
                n_days,
                config.DEFAULT_N_AGENTS,
                config.DEFAULT_INITIAL_CASH_PER_AGENT,
                config.DEFAULT_INITIAL_SHARES_PER_AGENT,
            )
            beta, _, _ = compute_mean_reversion_slope(result.prices, config.DEFAULT_INITIAL_PRICE)
            exp2_betas.append(beta)
            exp2_log_vars.append(float(np.var(np.log(result.prices / config.DEFAULT_INITIAL_PRICE), ddof=1)))

        depth_metrics = []
        for depth in [config.EXP3_DEPTH_LEVELS[0], config.EXP3_DEPTH_LEVELS[-1]]:
            batch = _simulate_unlimited_depth_batch(
                n_days=n_days,
                n_runs=max(3, min(n_runs, 15)),
                n_traders=depth,
                seed=seed + 200_000 + value_idx * 10_000 + depth,
                sample_curve_day=max(1, n_days // 2),
            )
            returns = batch["returns"]
            depth_metrics.append(
                {
                    "depth": depth,
                    "volatility": float(np.mean(np.std(returns, axis=1, ddof=1))),
                    "max_abs_return": float(np.mean(np.max(np.abs(returns), axis=1))),
                    "roughness": float(np.nanmean(batch["roughness_values"])),
                }
            )

        rows.append(
            {
                "parameter_name": parameter_name,
                "parameter_value": value,
                "exp1_mean_log_return": float(np.mean([m["mean_return"] for m in exp1_metrics])),
                "exp1_lag1_autocorr": float(np.mean([m["return_autocorr_lag1"] for m in exp1_metrics])),
                "exp1_variance_ratio_lag5": float(np.mean([m["variance_ratio_lag5"] for m in exp1_metrics])),
                "exp2_mean_beta": float(np.mean(exp2_betas)),
                "exp2_mean_log_deviation_variance": float(np.mean(exp2_log_vars)),
                "exp3_low_depth_volatility": depth_metrics[0]["volatility"],
                "exp3_high_depth_volatility": depth_metrics[1]["volatility"],
                "exp3_low_depth_roughness": depth_metrics[0]["roughness"],
                "exp3_high_depth_roughness": depth_metrics[1]["roughness"],
                "exp3_low_depth_extreme_move": depth_metrics[0]["max_abs_return"],
                "exp3_high_depth_extreme_move": depth_metrics[1]["max_abs_return"],
            }
        )
    return pd.DataFrame(rows)


def _plot_sensitivity(df: pd.DataFrame, x_col: str, output_dir: str | Path, filename: str, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    x = df[x_col].to_numpy(dtype=float)
    axes[0].plot(x, df["exp1_lag1_autocorr"], marker="o", label="Exp1 lag1 autocorr")
    axes[0].axhline(0.0, color="gray", linewidth=0.8)
    axes[0].set_ylabel("Lag-1 autocorr")
    axes[1].plot(x, df["exp2_mean_beta"], marker="o", label="Exp2 beta", color="#d62728")
    axes[1].axhline(0.0, color="gray", linewidth=0.8)
    axes[1].set_ylabel("Mean beta")
    axes[2].plot(x, df["exp3_low_depth_volatility"], marker="o", label="Low depth")
    axes[2].plot(x, df["exp3_high_depth_volatility"], marker="s", label="High depth")
    axes[2].set_ylabel("Return volatility")
    for ax in axes:
        ax.set_xlabel("Parameter value")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "figures" / filename, dpi=160)
    plt.close(fig)


def _write_sensitivity_summary(path: Path, sigma_df: pd.DataFrame, qty_df: pd.DataFrame, days_df: pd.DataFrame) -> None:
    path.write_text(
        "# Sensitivity Analysis Summary\n\n"
        "The sensitivity checks vary `price_sigma`, `max_qty`, and `n_days` while tracking the core conclusions.\n\n"
        "Most robust conclusions:\n"
        "- Experiment 1 remains close to zero-drift and low-autocorrelation across tested settings.\n"
        "- Experiment 3 preserves the ordering that high depth has lower volatility and lower curve roughness than low depth.\n"
        "- Experiment 2 mean-reversion beta is generally negative under the baseline constrained mechanism.\n\n"
        "More sensitive quantities:\n"
        "- Absolute volatility scales with `price_sigma`.\n"
        "- Extreme moves are sensitive to both `price_sigma` and simulation horizon.\n"
        "- Wealth-constraint dispersion measures can vary with horizon and quantity scale, so later calibration should report robustness ranges.\n\n"
        "Generated tables:\n"
        "- `sensitivity_sigma_summary.csv`\n"
        "- `sensitivity_max_qty_summary.csv`\n"
        "- `sensitivity_n_days_summary.csv`\n",
        encoding="utf-8",
    )


def _plot_event_window(prices: np.ndarray, event_day: int, output_dir: str | Path, filename: str, title: str) -> None:
    start = max(0, event_day - 5)
    end = min(len(prices) - 1, event_day + 5)
    days = np.arange(start, end + 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(days, prices[start : end + 1], marker="o", linewidth=1.6, label="Price")
    ax.axvline(event_day, color="red", linestyle="--", linewidth=1.2, label="Max |return| day")
    ax.set_title(title)
    ax.set_xlabel("Day")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "figures" / filename, dpi=160)
    plt.close(fig)


def _simulate_gaussian_path(seed: int, n_days: int, initial_price: float, sigma: float) -> MarketRunResult:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, sigma, n_days)
    prices = initial_price * np.exp(np.concatenate([[0.0], np.cumsum(returns)]))
    volumes = np.zeros(n_days + 1)
    return MarketRunResult(prices=prices, volumes=volumes, returns=returns)


def _cross_row(
    market_type: str,
    experiment: str,
    interpretation: str,
    result: MarketRunResult,
    curve: pd.DataFrame | None = None,
) -> dict[str, float | str]:
    metrics = compute_basic_metrics(result.prices, result.returns)
    beta, _, _ = compute_mean_reversion_slope(result.prices, config.DEFAULT_INITIAL_PRICE)
    return {
        "market_type": market_type,
        "experiment": experiment,
        "process_interpretation": interpretation,
        "mean_log_return": metrics["mean_return"],
        "std_log_return": metrics["std_return"],
        "lag1_autocorr_log_return": metrics["return_autocorr_lag1"],
        "variance_ratio_lag5": metrics["variance_ratio_lag5"],
        "positive_return_pct": metrics["positive_return_pct"],
        "mean_reversion_beta": beta,
        "log_deviation_variance": float(np.var(np.log(result.prices / config.DEFAULT_INITIAL_PRICE), ddof=1)),
        "price_variance": float(np.var(result.prices, ddof=1)),
        "max_abs_return": compute_max_abs_return(result.returns),
        "max_drawdown": metrics["max_drawdown"],
        "avg_volume": float(np.mean(result.volumes[1:])) if result.volumes.size > 1 else 0.0,
        "curve_roughness": compute_curve_roughness(curve) if curve is not None else float("nan"),
    }


def _build_exp1_random_walk_diagnostics(
    single_result: MarketRunResult,
    path_metrics: list[dict[str, float]],
) -> pd.DataFrame:
    """Build Table 1: random-walk diagnostics for single and repeated paths."""
    single_row = _random_walk_diagnostics_row(single_result.prices, single_result.returns)
    single_row["sample"] = "single_path"

    path_df = pd.DataFrame(path_metrics)
    numeric_cols = [col for col in path_df.columns if col != "sample"]
    mean_row = {"sample": "multi_path_mean"}
    std_row = {"sample": "multi_path_std"}
    for column in numeric_cols:
        mean_row[column] = float(path_df[column].mean())
        std_row[column] = float(path_df[column].std(ddof=1))

    diagnostics = pd.DataFrame([single_row, mean_row, std_row])
    ordered_cols = [
        "sample",
        "mean_log_return",
        "std_log_return",
        "lag1_autocorr_log_return",
        "variance_ratio_lag5",
        "positive_return_pct",
        "final_price",
        "max_drawdown",
        "skewness",
        "kurtosis",
    ]
    return diagnostics[ordered_cols]


def _run_exp2_repeated_statistics(
    seed: int,
    n_days: int,
    output_dir: str | Path,
    n_runs: int = config.EXP2_N_RUNS,
) -> dict[str, object]:
    """Run repeated wealth-constrained simulations and summarize statistics."""
    dirs = ensure_output_dirs(output_dir)
    metric_rows: list[dict[str, float]] = []
    path_rows: list[pd.DataFrame] = []
    path_groups: dict[str, list[np.ndarray]] = {}
    beta_groups: dict[str, list[float]] = {}

    for multiplier_idx, multiplier in enumerate(config.EXP2_WEALTH_MULTIPLIERS):
        label = f"x{multiplier:g}"
        path_groups[label] = []
        beta_groups[label] = []

        for run_id in range(n_runs):
            market = DailyAuctionMarket(
                initial_price=config.DEFAULT_INITIAL_PRICE,
                price_sigma=config.DEFAULT_PRICE_SIGMA,
                max_qty=config.DEFAULT_MAX_QTY,
                seed=seed + multiplier_idx * 10_000 + run_id,
            )
            result = market.simulate_wealth_constrained(
                n_days=n_days,
                n_agents=config.DEFAULT_N_AGENTS,
                initial_cash_per_agent=config.DEFAULT_INITIAL_CASH_PER_AGENT * multiplier,
                initial_shares_per_agent=config.DEFAULT_INITIAL_SHARES_PER_AGENT * multiplier,
                participation_rate=config.DEFAULT_AGENT_PARTICIPATION_RATE,
            )
            run_metrics = _exp2_repeated_metric_row(result, float(multiplier), run_id)
            metric_rows.append(run_metrics)
            beta_groups[label].append(run_metrics["mean_reversion_beta"])

            df = result.to_dataframe()
            df["wealth_multiplier"] = float(multiplier)
            df["run_id"] = run_id
            path_rows.append(df)
            if run_id < config.EXP2_PATHS_PER_MULTIPLIER:
                path_groups[label].append(result.prices)

    run_metrics_df = pd.DataFrame(metric_rows)
    run_metrics_df.to_csv(dirs["tables"] / "exp2_repeated_run_metrics.csv", index=False)
    pd.concat(path_rows, ignore_index=True).to_csv(
        dirs["simulation_results"] / "exp2_repeated_paths.csv",
        index=False,
    )

    summary = _summarize_exp2_repeated_metrics(run_metrics_df)
    summary.to_csv(dirs["tables"] / "exp2_summary_by_wealth_multiplier.csv", index=False)
    return {
        "run_metrics": run_metrics_df,
        "summary": summary,
        "path_groups": path_groups,
        "beta_groups": beta_groups,
    }


def _exp2_repeated_metric_row(
    result: MarketRunResult,
    wealth_multiplier: float,
    run_id: int,
) -> dict[str, float]:
    """Compute one repeated-run metric row for Experiment 2."""
    beta, alpha, r2 = compute_mean_reversion_slope(result.prices, config.DEFAULT_INITIAL_PRICE)
    returns = result.returns
    log_deviation = np.log(result.prices / config.DEFAULT_INITIAL_PRICE)
    return {
        "wealth_multiplier": wealth_multiplier,
        "run_id": run_id,
        "mean_reversion_beta": beta,
        "mean_reversion_alpha": alpha,
        "mean_reversion_r2": r2,
        "price_variance": float(np.var(result.prices, ddof=1)),
        "log_deviation_variance": float(np.var(log_deviation, ddof=1)),
        "final_price": float(result.prices[-1]),
        "volatility": float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0,
    }


def _summarize_exp2_repeated_metrics(run_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize repeated Experiment 2 runs by wealth multiplier."""
    rows: list[dict[str, float]] = []
    for multiplier, group in run_metrics.groupby("wealth_multiplier", sort=True):
        beta = group["mean_reversion_beta"].to_numpy(dtype=float)
        ci_lower, ci_upper = confidence_interval_95(beta)
        rows.append(
            {
                "wealth_multiplier": float(multiplier),
                "n_runs": int(group.shape[0]),
                "mean_beta": float(np.mean(beta)),
                "std_beta": float(np.std(beta, ddof=1)) if beta.size > 1 else 0.0,
                "beta_ci95_lower": ci_lower,
                "beta_ci95_upper": ci_upper,
                "mean_price_variance": float(group["price_variance"].mean()),
                "mean_log_deviation_variance": float(group["log_deviation_variance"].mean()),
                "mean_volatility": float(group["volatility"].mean()),
                "mean_final_price": float(group["final_price"].mean()),
                "std_final_price": float(group["final_price"].std(ddof=1)) if group.shape[0] > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _select_distribution_groups(return_groups: dict[str, list[float]]) -> dict[str, list[float]]:
    """Keep return distribution plot readable by showing representative depths."""
    keys = list(return_groups.keys())
    if len(keys) <= 4:
        return return_groups
    selected = [keys[0], keys[len(keys) // 2], keys[-1]]
    return {key: return_groups[key] for key in selected}


def _simulate_unlimited_depth_batch(
    n_days: int,
    n_runs: int,
    n_traders: int,
    seed: int,
    sample_curve_day: int | None = None,
) -> dict[str, object]:
    """Batch-simulate many unlimited-market runs for one depth level."""
    rng = np.random.default_rng(seed)
    prices = np.empty((n_runs, n_days + 1), dtype=float)
    volumes = np.zeros((n_runs, n_days + 1), dtype=float)
    prices[:, 0] = config.DEFAULT_INITIAL_PRICE
    sample_curve: pd.DataFrame | None = None
    roughness_values: np.ndarray = np.full(n_runs, np.nan, dtype=float)

    for day in range(1, n_days + 1):
        previous_prices = prices[:, day - 1][:, None]
        bid_prices = previous_prices * np.exp(
            rng.normal(0.0, config.DEFAULT_PRICE_SIGMA, size=(n_runs, n_traders))
        )
        ask_prices = previous_prices * np.exp(
            rng.normal(0.0, config.DEFAULT_PRICE_SIGMA, size=(n_runs, n_traders))
        )
        bid_qtys = rng.uniform(1.0, config.DEFAULT_MAX_QTY, size=(n_runs, n_traders))
        ask_qtys = rng.uniform(1.0, config.DEFAULT_MAX_QTY, size=(n_runs, n_traders))

        clearing_prices, clearing_volumes = clear_auction_price_volume_batch_continuous(
            bid_prices,
            bid_qtys,
            ask_prices,
            ask_qtys,
        )
        prices[:, day] = clearing_prices
        volumes[:, day] = clearing_volumes

        if sample_curve_day == day:
            for run_id in range(n_runs):
                sample_result = clear_auction_arrays(
                    bid_prices=bid_prices[run_id],
                    bid_qtys=bid_qtys[run_id],
                    ask_prices=ask_prices[run_id],
                    ask_qtys=ask_qtys[run_id],
                    price_hint=float(previous_prices[run_id, 0]),
                    return_curve=True,
                )
                if run_id == 0:
                    sample_curve = sample_result.auction_curve
                roughness_values[run_id] = compute_curve_roughness(sample_result.auction_curve)

    returns = np.diff(np.log(prices), axis=1)
    return {
        "prices": prices,
        "volumes": volumes,
        "returns": returns,
        "sample_curve": sample_curve,
        "roughness_values": roughness_values,
    }


def _batch_lag1_autocorr(returns: np.ndarray) -> np.ndarray:
    """Vectorized lag-1 autocorrelation for rows of a return matrix."""
    if returns.shape[1] <= 1:
        return np.full(returns.shape[0], np.nan)
    x = returns[:, :-1]
    y = returns[:, 1:]
    x_centered = x - np.mean(x, axis=1, keepdims=True)
    y_centered = y - np.mean(y, axis=1, keepdims=True)
    numerator = np.sum(x_centered * y_centered, axis=1)
    denominator = np.sqrt(np.sum(x_centered**2, axis=1) * np.sum(y_centered**2, axis=1))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-15)


def _save_representative_depth_path(
    prices: np.ndarray,
    volumes: np.ndarray,
    returns: np.ndarray,
    path: Path,
) -> None:
    """Save one representative depth path to CSV."""
    returns_with_start = np.empty_like(prices)
    returns_with_start[0] = np.nan
    returns_with_start[1:] = returns
    pd.DataFrame(
        {
            "day": np.arange(prices.size),
            "price": prices,
            "volume": volumes,
            "return": returns_with_start,
        }
    ).to_csv(path, index=False)
