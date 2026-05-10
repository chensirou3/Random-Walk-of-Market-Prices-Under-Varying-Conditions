"""Output index, hypothesis, configuration, and diagnostic documents."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import config


def ensure_document_dirs(output_dir: str | Path) -> dict[str, Path]:
    """Create output subdirectories used by reporting modules."""
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


def generate_experiment_config_summary(
    output_dir: str | Path = "outputs",
    n_days: int = config.DEFAULT_N_DAYS,
    n_runs: int = config.EXP3_N_RUNS,
    seed: int = config.DEFAULT_SEED,
) -> pd.DataFrame:
    """Write parameter summary tables and parameter documentation."""
    dirs = ensure_document_dirs(output_dir)
    rows = [
        {
            "experiment_name": "Experiment 1: Unlimited-Wealth Random Market",
            "initial_price": config.DEFAULT_INITIAL_PRICE,
            "n_days": n_days,
            "n_runs": 50,
            "n_buyers": config.DEFAULT_N_BUYERS,
            "n_sellers": config.DEFAULT_N_SELLERS,
            "n_agents": "NA",
            "price_sigma": config.DEFAULT_PRICE_SIGMA,
            "max_qty": config.DEFAULT_MAX_QTY,
            "auction_rule": "maximize volume, minimize imbalance, median tie break",
            "wealth_multiplier": "NA",
            "market_depth": "NA",
            "seed_policy": f"base seed {seed}; path seed offset +10000",
            "order_price_distribution": "lognormal around previous price",
            "order_quantity_distribution": "Uniform(1, max_qty)",
            "constraints": "none",
            "output_files": "exp1_metrics.csv; exp1_random_walk_diagnostics.csv; exp1_*.png",
        },
        {
            "experiment_name": "Experiment 2: Wealth-Constrained Random Market",
            "initial_price": config.DEFAULT_INITIAL_PRICE,
            "n_days": n_days,
            "n_runs": config.EXP2_N_RUNS,
            "n_buyers": "NA",
            "n_sellers": "NA",
            "n_agents": config.DEFAULT_N_AGENTS,
            "price_sigma": config.DEFAULT_PRICE_SIGMA,
            "max_qty": config.DEFAULT_MAX_QTY,
            "auction_rule": "maximize volume, minimize imbalance, median tie break",
            "wealth_multiplier": str(config.EXP2_WEALTH_MULTIPLIERS),
            "market_depth": "NA",
            "seed_policy": f"base seed {seed}; run-specific deterministic offsets",
            "order_price_distribution": "lognormal around previous price",
            "order_quantity_distribution": "Uniform(1, max_qty), capped by constraints",
            "constraints": "buyer cash and seller inventory",
            "output_files": "exp2_summary_by_wealth_multiplier.csv; exp2_repeated_run_metrics.csv; exp2_*.png",
        },
        {
            "experiment_name": "Experiment 2 Ablation",
            "initial_price": config.DEFAULT_INITIAL_PRICE,
            "n_days": n_days,
            "n_runs": n_runs,
            "n_buyers": "NA",
            "n_sellers": "NA",
            "n_agents": config.DEFAULT_N_AGENTS,
            "price_sigma": config.DEFAULT_PRICE_SIGMA,
            "max_qty": config.DEFAULT_MAX_QTY,
            "auction_rule": "maximize volume, minimize imbalance, median tie break",
            "wealth_multiplier": str(config.EXP2_WEALTH_MULTIPLIERS),
            "market_depth": "NA",
            "seed_policy": f"base seed {seed}; constraint/multiplier/run offsets",
            "order_price_distribution": "lognormal around previous price",
            "order_quantity_distribution": "Uniform(1, max_qty), selectively capped",
            "constraints": "cash only; inventory only; both constraints",
            "output_files": "exp2_ablation_summary.csv; exp2_ablation_*.png",
        },
        {
            "experiment_name": "Experiment 3: Market Depth and Volatility",
            "initial_price": config.DEFAULT_INITIAL_PRICE,
            "n_days": n_days,
            "n_runs": n_runs,
            "n_buyers": "depth",
            "n_sellers": "depth",
            "n_agents": "NA",
            "price_sigma": config.DEFAULT_PRICE_SIGMA,
            "max_qty": config.DEFAULT_MAX_QTY,
            "auction_rule": "maximize volume, minimize imbalance, median tie break",
            "wealth_multiplier": "NA",
            "market_depth": str(config.EXP3_DEPTH_LEVELS),
            "seed_policy": f"base seed {seed}; depth/run offsets",
            "order_price_distribution": "lognormal around previous price",
            "order_quantity_distribution": "Uniform(1, max_qty)",
            "constraints": "none",
            "output_files": "exp3_metrics.csv; exp3_roughness_volatility_summary.csv; exp3_*.png",
        },
        {
            "experiment_name": "Experiment 4: Capital Inflow and the Random-Walk Channel",
            "initial_price": config.DEFAULT_INITIAL_PRICE,
            "n_days": n_days,
            "n_runs": config.EXP4_N_RUNS,
            "n_buyers": "NA",
            "n_sellers": "NA",
            "n_agents": config.DEFAULT_N_AGENTS,
            "price_sigma": config.DEFAULT_PRICE_SIGMA,
            "max_qty": config.DEFAULT_MAX_QTY,
            "auction_rule": "maximize volume, minimize imbalance, median tie break",
            "wealth_multiplier": "NA",
            "market_depth": "NA",
            "seed_policy": f"base seed {seed}; regime/group/run offsets",
            "order_price_distribution": "lognormal around previous price (no drift term)",
            "order_quantity_distribution": "Uniform(1, max_qty), capped by cash and inventory",
            "constraints": f"buyer cash and seller inventory; baseline vs treatment inflow {config.EXP4_INFLOW_TREATMENT:g}/buyer/day",
            "output_files": "exp4_main_comparison.csv; exp4_robustness_by_regime.csv; exp4_path_metrics.csv; exp4_*.png",
        },
        {
            "experiment_name": "Gaussian Random Walk Benchmark",
            "initial_price": config.DEFAULT_INITIAL_PRICE,
            "n_days": n_days,
            "n_runs": n_runs,
            "n_buyers": "NA",
            "n_sellers": "NA",
            "n_agents": "NA",
            "price_sigma": "matched to Experiment 1 return volatility",
            "max_qty": "NA",
            "auction_rule": "NA",
            "wealth_multiplier": "NA",
            "market_depth": "NA",
            "seed_policy": f"base seed {seed}; benchmark RNG stream",
            "order_price_distribution": "NA",
            "order_quantity_distribution": "NA",
            "constraints": "NA",
            "output_files": "exp1_gaussian_benchmark_comparison.csv; exp1_auction_vs_gaussian_*.png",
        },
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(dirs["tables"] / "experiment_config_summary.csv", index=False)
    (dirs["tables"] / "experiment_config_summary.md").write_text(
        summary.to_markdown(index=False),
        encoding="utf-8",
    )
    (dirs["reports"] / "parameter_documentation.md").write_text(_parameter_docs(), encoding="utf-8")
    return summary


def generate_hypotheses(output_dir: str | Path = "outputs") -> pd.DataFrame:
    """Write hypotheses report and mapping table."""
    dirs = ensure_document_dirs(output_dir)
    rows = [
        ("H0", "Random price processes can emerge endogenously from decentralized random orders and auction-clearing rules, and different market microstructures systematically alter the statistical form of these processes.", "All", "Cross-experiment diagnostics", "Different mechanisms alter process metrics", "cross_experiment_summary.csv", "cross_experiment_process_comparison.png", "supported"),
        ("H1", "In an unlimited-wealth random-order market, auction clearing generates a price process resembling a random walk.", "Experiment 1", "random walk diagnostics", "near-zero drift, low autocorrelation, variance ratio near one", "exp1_random_walk_diagnostics.csv", "exp1_multiple_paths.png", "supported"),
        ("H1a", "Mean log return is close to zero.", "Experiment 1", "mean_log_return", "close to zero", "exp1_random_walk_diagnostics.csv", "exp1_random_walk_diagnostics_table.png", "supported"),
        ("H1b", "Lag-1 return autocorrelation is close to zero.", "Experiment 1", "lag1_autocorr_log_return", "close to zero", "exp1_random_walk_diagnostics.csv", "exp1_random_walk_diagnostics_table.png", "supported"),
        ("H1c", "Variance ratio is close to one.", "Experiment 1", "variance_ratio_lag5", "near one", "exp1_random_walk_diagnostics.csv", "exp1_random_walk_diagnostics_table.png", "supported"),
        ("H1d", "Multiple simulated paths show diffusion without a common directional trend.", "Experiment 1", "multiple paths", "diffusion, no shared direction", "exp1_multiple_paths.csv", "exp1_multiple_paths.png", "supported"),
        ("H2", "Wealth constraints transform the freely diffusive random process into a more bounded and weakly mean-reverting process.", "Experiment 2", "mean_beta; log_deviation_variance", "negative beta and constrained dispersion", "exp2_summary_by_wealth_multiplier.csv", "exp2_mean_beta_ci_by_wealth_multiplier.png", "supported"),
        ("H2a", "Mean reversion beta is negative.", "Experiment 2", "mean_beta", "negative", "exp2_summary_by_wealth_multiplier.csv", "exp2_beta_distribution_by_wealth_multiplier.png", "supported"),
        ("H2b", "Log deviation variance is lower under stronger constraints.", "Experiment 2", "mean_log_deviation_variance", "lower under stronger constraints", "exp2_summary_by_wealth_multiplier.csv", "exp2_log_deviation_variance_by_wealth_multiplier.png", "partially supported"),
        ("H2c", "Ablation study shows which constraint mechanism contributes most to boundedness.", "Experiment 2 Ablation", "beta and log deviation variance by constraint type", "cash/inventory contributions differ", "exp2_ablation_summary.csv", "exp2_ablation_mean_beta_ci.png", "evaluated"),
        ("H3", "Market depth controls the volatility scale of the emergent random process.", "Experiment 3", "avg_std_return", "lower volatility at higher depth", "exp3_metrics.csv", "exp3_volatility_vs_depth.png", "supported"),
        ("H3a", "Lower market depth increases return volatility.", "Experiment 3", "avg_std_return", "decreases with depth", "exp3_metrics.csv", "exp3_volatility_vs_depth.png", "supported"),
        ("H3b", "Lower market depth increases maximum absolute return.", "Experiment 3", "avg_max_abs_return", "decreases with depth", "exp3_metrics.csv", "exp3_extreme_move_vs_depth.png", "supported"),
        ("H3c", "Lower market depth increases order-curve roughness.", "Experiment 3", "avg_curve_roughness", "decreases with depth", "exp3_metrics.csv", "exp3_curve_roughness_vs_depth.png", "supported"),
        ("H3d", "Higher curve roughness is associated with higher volatility.", "Experiment 3", "corr_roughness_volatility", "positive relation", "exp3_roughness_volatility_summary.csv", "exp3_roughness_vs_volatility.png", "supported"),
        ("H4", "A fixed capital inflow shifts the random-walk channel of auction-generated prices upward while preserving the random-walk structure of the increments.", "Experiment 4", "treatment minus baseline mean log return; final price distribution; lag-1 autocorrelation; variance ratio", "treatment above baseline; increment diagnostics unchanged", "exp4_main_comparison.csv", "exp4_mean_paths_baseline_vs_inflow.png", "evaluated"),
        ("H4a", "The treatment mean log return is greater than the baseline mean log return (Welch t-test).", "Experiment 4", "mean_log_return; welch_t_stat_mean_return; welch_p_value_mean_return", "treatment > baseline", "exp4_main_comparison.csv", "exp4_mean_paths_baseline_vs_inflow.png", "evaluated"),
        ("H4b", "The cross-path mean price path for the treatment lies above the baseline path on average.", "Experiment 4", "final_price_mean; mean cross-path price trajectory", "treatment path above baseline path", "exp4_main_comparison.csv", "exp4_paths_baseline_vs_inflow.png", "evaluated"),
        ("H4c", "Random-walk features (low lag-1 autocorrelation, variance ratio near one, symmetric positive-return share) are preserved under the treatment.", "Experiment 4", "lag1_autocorr; variance_ratio_lag5; positive_return_pct", "near baseline values", "exp4_main_comparison.csv", "exp4_random_walk_diagnostics_bars.png", "evaluated"),
        ("H4d", "The inflow effect is mediated by binding cash constraints: it is larger in the tight cash regime than in the loose regime.", "Experiment 4", "mean_log_return_diff_vs_baseline by regime; avg_cash_constrained_ratio", "larger gap in tight regime", "exp4_robustness_by_regime.csv", "exp4_mean_paths_baseline_vs_inflow.png", "evaluated"),
    ]
    mapping = pd.DataFrame(
        rows,
        columns=[
            "hypothesis_id",
            "hypothesis_text",
            "corresponding_experiment",
            "metric_used",
            "expected_result",
            "output_table",
            "output_figure",
            "status_placeholder",
        ],
    )
    mapping.to_csv(dirs["tables"] / "hypotheses_mapping.csv", index=False)
    (dirs["reports"] / "hypotheses.md").write_text(_hypotheses_markdown(mapping), encoding="utf-8")
    return mapping


def generate_reproducibility_diagnostics(output_dir: str | Path = "outputs") -> dict[str, object]:
    """Write reproducibility policy diagnostics."""
    dirs = ensure_document_dirs(output_dir)
    payload = {
        "seed_policy": "All stochastic components use numpy.random.default_rng(seed) with deterministic offsets for paths, runs, depth levels, and constraint types.",
        "deterministic_outputs": [
            "same seed -> same price path",
            "same seed -> same volume series",
            "same seed -> same selected auction curves",
            "same command and same dependency versions -> same generated tables",
        ],
        "statistically_stable_outputs": [
            "repeated-run mean beta",
            "market-depth average volatility",
            "curve roughness averages",
            "Gaussian benchmark summary statistics",
        ],
        "reproduce_all": "python main.py --experiment full --seed 42 --n_days 500 --n_runs 100 --output_dir outputs",
    }
    (dirs["diagnostics"] / "reproducibility_check.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (dirs["diagnostics"] / "reproducibility_check.md").write_text(
        "# Reproducibility Check\n\n"
        f"Seed policy: {payload['seed_policy']}\n\n"
        "Deterministic outputs:\n"
        + "\n".join(f"- {item}" for item in payload["deterministic_outputs"])
        + "\n\nStatistically stable outputs:\n"
        + "\n".join(f"- {item}" for item in payload["statistically_stable_outputs"])
        + f"\n\nReproduce all outputs:\n\n```bash\n{payload['reproduce_all']}\n```\n",
        encoding="utf-8",
    )
    return payload


def generate_output_index(output_dir: str | Path = "outputs") -> Path:
    """Generate a human-readable output index."""
    dirs = ensure_document_dirs(output_dir)
    path = dirs["base"] / "index.md"
    files = sorted(p.relative_to(dirs["base"]).as_posix() for p in dirs["base"].rglob("*") if p.is_file())
    content = [
        "# Market Random Process Simulation Output Index",
        "",
        "Project: The Emergence of Random Market Processes from Random Orders and Auction Rules.",
        "",
        "Core mechanism: random individual orders -> auction clearing rule -> emergent market-level random process.",
        "",
        "## Recommended Figures for Final Report",
        "",
        "- `figures/exp1_multiple_paths.png`: Shows random-walk-like diffusion from auction-generated prices.",
        "- `figures/exp1_auction_vs_gaussian_return_distribution.png`: Compares endogenous auction returns with an exogenous Gaussian benchmark.",
        "- `figures/exp2_mean_beta_ci_by_wealth_multiplier.png`: Shows weak mean reversion under wealth constraints.",
        "- `figures/exp2_ablation_mean_beta_ci.png`: Decomposes cash and inventory constraint mechanisms.",
        "- `figures/exp3_volatility_vs_depth.png`: Shows depth scaling of volatility.",
        "- `figures/exp3_curve_roughness_vs_depth.png`: Shows that deeper markets produce smoother curves.",
        "- `figures/exp3_roughness_vs_volatility.png`: Connects rough order curves to return volatility.",
        "- `figures/exp4_paths_baseline_vs_inflow.png`: Shows that the capital-inflow treatment shifts the random-walk channel upward versus the baseline.",
        "- `figures/exp4_mean_paths_baseline_vs_inflow.png`: Cross-path mean price paths with 95% confidence bands for baseline and treatment.",
        "- `figures/exp4_final_price_distribution.png`: Final-price distributions for baseline and treatment, illustrating the channel shift.",
        "- `figures/exp4_random_walk_diagnostics_bars.png`: Confirms that lag-1 autocorrelation, variance ratio, and positive-return share remain near random-walk values.",
        "- `figures/cross_experiment_process_comparison.png`: Summarizes process differences across market rules.",
        "",
        "## Recommended Appendix Figures",
        "",
        "- Extreme case plots.",
        "- Sensitivity figures.",
        "- Volume analysis plots.",
        "- Formula figures.",
        "",
        "## File Purposes",
        "",
    ]
    purpose_map = _purpose_map()
    for file in files:
        purpose = purpose_map.get(Path(file).name, "Supporting output file for the simulation study.")
        content.append(f"- `{file}`: {purpose}")
    content.extend(
        [
            "",
            "## How to Reproduce",
            "",
            "```bash",
            "python main.py --experiment full --seed 42 --n_days 500 --n_runs 100 --output_dir outputs",
            "```",
        ]
    )
    path.write_text("\n".join(content), encoding="utf-8")
    return path


def _parameter_docs() -> str:
    return """# Parameter Documentation

- `initial_price`: Starting market price `P_0`.
- `n_days`: Number of daily auction periods.
- `n_runs`: Number of repeated stochastic simulations.
- `n_buyers`, `n_sellers`: Number of submitted buy and sell orders in the unlimited market.
- `n_agents`: Number of agents in constrained markets.
- `price_sigma`: Standard deviation of lognormal quote noise around yesterday's price.
- `max_qty`: Upper bound of uniform order quantity draws.
- `auction_rule`: Clearing rule that maximizes matched volume, minimizes imbalance, and median tie-breaks.
- `wealth_multiplier`: Scaling factor applied to initial cash and share inventory.
- `market_depth`: Buyers = sellers in the depth experiment.
- `seed_policy`: Deterministic seed offsets used to make repeated simulations reproducible.
- `order_price_distribution`: Lognormal quote generation mechanism.
- `order_quantity_distribution`: Uniform quantity generation mechanism, optionally capped by constraints.
- `constraints`: Active budget and inventory restrictions.
"""


def _hypotheses_markdown(mapping: pd.DataFrame) -> str:
    return "# Hypotheses\n\n" + mapping.to_markdown(index=False)


def _purpose_map() -> dict[str, str]:
    return {
        "exp3_curve_roughness_vs_depth.png": "Shows that higher market depth produces smoother order curves.",
        "exp2_mean_beta_ci_by_wealth_multiplier.png": "Shows negative mean-reversion coefficients across repeated wealth-constrained simulations.",
        "exp3_roughness_vs_volatility.png": "Shows the mechanism link from curve roughness to volatility.",
        "exp2_ablation_mean_beta_ci.png": "Compares mean-reversion beta across cash, inventory, and combined constraints.",
        "experiment_config_summary.csv": "Records reproducible experiment parameters.",
        "cross_experiment_summary.csv": "Compares process metrics across market mechanisms.",
        "formal_model.md": "Provides mathematical model definitions and diagnostics.",
        "hypotheses.md": "Defines hypotheses and maps them to metrics and outputs.",
        "exp4_main_comparison.csv": "Main two-group comparison (baseline vs capital-inflow treatment) under the loose cash regime, with Welch t-tests on mean log return.",
        "exp4_robustness_by_regime.csv": "Robustness table covering tight and loose cash regimes for the same baseline-vs-treatment contrast.",
        "exp4_path_metrics.csv": "Per-path diagnostics for every Experiment 4 simulation (regime, group, run id, return metrics, constraint metrics).",
        "exp4_paths_baseline_vs_inflow.png": "Baseline and treatment paths with bold cross-path means, illustrating the upward channel shift.",
        "exp4_mean_paths_baseline_vs_inflow.png": "Cross-path mean price paths with 95% confidence bands.",
        "exp4_final_price_distribution.png": "Histograms of final prices for baseline and treatment groups.",
        "exp4_random_walk_diagnostics_bars.png": "Lag-1 autocorrelation, variance ratio, and positive-return share for baseline vs treatment.",
        "exp4_findings.md": "Auto-generated Experiment 4 interpretation block.",
    }
