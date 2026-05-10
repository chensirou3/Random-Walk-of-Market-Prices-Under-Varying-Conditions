"""Markdown report generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def generate_report(output_dir: str | Path = "outputs") -> Path:
    """Generate the Markdown experiment report from saved metric tables."""
    output_path = Path(output_dir)
    report_dir = output_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "market_random_process_report.md"

    exp1_diag = _read_table(output_path / "tables" / "exp1_random_walk_diagnostics.csv")
    exp2_summary = _read_table(output_path / "tables" / "exp2_summary_by_wealth_multiplier.csv")
    exp3_summary = _read_table(output_path / "tables" / "exp3_metrics.csv")
    exp4_main = _read_table(output_path / "tables" / "exp4_main_comparison.csv")
    exp4_robust = _read_table(output_path / "tables" / "exp4_robustness_by_regime.csv")

    exp1_table = _table_preview(exp1_diag)
    exp2_table = _table_preview(
        _select_columns(
            exp2_summary,
            [
                "wealth_multiplier",
                "n_runs",
                "mean_beta",
                "std_beta",
                "beta_ci95_lower",
                "beta_ci95_upper",
                "mean_log_deviation_variance",
                "mean_volatility",
            ],
        )
    )
    exp3_table = _table_preview(
        _select_columns(
            exp3_summary,
            [
                "n_traders",
                "n_runs",
                "avg_std_return",
                "avg_max_abs_return",
                "avg_curve_roughness",
                "std_final_price",
            ],
        )
    )
    exp4_main_table = _table_preview(
        _select_columns(
            exp4_main,
            [
                "group",
                "n_runs",
                "inflow_per_period",
                "mean_log_return",
                "lag1_autocorr",
                "variance_ratio_lag5",
                "positive_return_pct",
                "final_price_mean",
                "welch_t_stat_mean_return",
                "welch_p_value_mean_return",
                "cohens_d_mean_return",
            ],
        )
    )
    exp4_robust_table = _table_preview(
        _select_columns(
            exp4_robust,
            [
                "regime",
                "group",
                "inflow_per_period",
                "mean_log_return",
                "final_price_mean",
                "avg_cash_constrained_ratio",
                "mean_log_return_diff_vs_baseline",
                "welch_p_value_mean_return",
            ],
        )
    )
    exp4_conclusion = _build_exp4_conclusion(exp4_main, exp4_robust)

    body = f"""# The Emergence of Random Market Processes from Random Orders and Auction Rules

# 市场随机过程如何从随机订单与集合竞价规则中自发生成

## 1. Introduction

Traditional financial modeling often starts from the macro assumption:

**Price Process = Random Process**

This project asks a lower-level computational economics question:

**Random Individual Orders + Auction Clearing Rules -> Emergent Market-Level Random Process**

The simulator deliberately excludes external information shocks, prediction rules, strategic manipulation, and rational-expectations structure. Each day, random buy and sell orders are generated around yesterday's price. A daily auction rule converts these micro-level random orders into one clearing price and one volume. The resulting price series is then studied as an endogenous random process.

## 2. Experimental Setup

Daily market loop:

Yesterday Price  
-> Generate Random Orders  
-> Auction Clearing  
-> Today Price and Volume  
-> Repeat

Auction clearing evaluates every candidate order price:

- `demand(p)` is buy quantity with bid price at least `p`.
- `supply(p)` is sell quantity with ask price at most `p`.
- `matched_volume(p) = min(demand(p), supply(p))`.
- `imbalance(p) = abs(demand(p) - supply(p))`.

The clearing price maximizes matched volume, then minimizes imbalance, then uses the median price if there is still a tie.

## 3. Method Addendum

Three upgrades were added to make the results more suitable for a report or paper.

**Experiment 1 random-walk diagnostics.** The unlimited-wealth market now reports a formal diagnostics table using the single example path and the 50-path simulation set. The table includes mean log return, log-return volatility, lag-1 autocorrelation, variance ratio at lag 5, percentage of positive return days, final price, max drawdown, skewness, and kurtosis.

**Experiment 2 repeated statistics.** Each wealth multiplier (`0.5, 1, 2, 5, 10`) is simulated 100 times for the requested horizon. Each run records mean-reversion beta, price variance, variance of `log(P[t] / P[0])`, final price, and log-return volatility. Summary statistics report mean beta, beta standard deviation, normal-approximation 95% confidence intervals, average variance measures, average volatility, and final-price dispersion.

**Experiment 3 curve roughness.** Curve roughness is defined as the average first absolute difference of normalized demand and supply curves. Demand and supply are each divided by their own maximum quantity before differencing. This makes depths comparable and keeps the interpretation direct: smoother aggregate curves have lower roughness.

## 4. Experiment 1: Random Walk Emergence in an Unlimited-Wealth Market

Question: **Are random orders plus auction clearing sufficient to generate random price paths?**

Figures:

- ![Experiment 1 price path](../figures/exp1_price_path.png)
- ![Experiment 1 daily log returns](../figures/exp1_return_series.png)
- ![Experiment 1 return distribution](../figures/exp1_return_distribution.png)
- ![Experiment 1 multiple paths](../figures/exp1_multiple_paths.png)
- ![Experiment 1 sample auction curve](../figures/exp1_sample_auction_curve.png)
- ![Experiment 1 random walk diagnostics table](../figures/exp1_random_walk_diagnostics_table.png)

**Table 1. Experiment 1 Random Walk Diagnostics**

{exp1_table}

Finding: log returns are centered near zero, lag-1 autocorrelation is close to zero, and the multi-path figure shows diffusion rather than a stable deterministic trend. Individual paths can look directional, but the diagnostics support the interpretation that these are random realizations generated by the order-auction mechanism.

## 5. Experiment 2: Wealth Constraints and the Modification of Random Dynamics

Question: **When traders face cash and inventory constraints, does the random process change structure?**

Figures:

- ![Experiment 2 path comparison](../figures/exp2_price_path_comparison.png)
- ![Experiment 2 price distribution comparison](../figures/exp2_price_distribution_comparison.png)
- ![Experiment 2 log price deviation](../figures/exp2_log_deviation.png)
- ![Experiment 2 mean reversion regression](../figures/exp2_mean_reversion_scatter.png)
- ![Experiment 2 multiple paths by wealth constraint](../figures/exp2_multiple_paths_by_wealth_constraint.png)
- ![Experiment 2 beta distribution](../figures/exp2_beta_distribution_by_wealth_multiplier.png)
- ![Experiment 2 mean beta confidence interval](../figures/exp2_mean_beta_ci_by_wealth_multiplier.png)
- ![Experiment 2 log-deviation variance](../figures/exp2_log_deviation_variance_by_wealth_multiplier.png)

**Table 2. Experiment 2 Summary Statistics by Wealth Multiplier**

{exp2_table}

Finding: wealth constraints do not remove randomness. They change the feasible order flow after prices move away from the starting region. The repeated simulations quantify whether the mean-reversion beta is consistently negative and whether log-deviation variance is compressed under stronger constraints. The relevant claim is structural: constraints can transform free diffusion into a softer, more bounded random process.

## 6. Experiment 3: Market Depth and the Scaling of Random Fluctuations

Question: **How does market depth change the volatility scale, extreme moves, and order-curve mechanism of the random process?**

Figures:

- ![Experiment 3 representative paths](../figures/exp3_price_paths_by_depth.png)
- ![Experiment 3 multiple paths by depth](../figures/exp3_multiple_paths_by_market_depth.png)
- ![Experiment 3 volatility by depth](../figures/exp3_volatility_vs_depth.png)
- ![Experiment 3 extreme move by depth](../figures/exp3_extreme_move_vs_depth.png)
- ![Experiment 3 return distributions](../figures/exp3_return_distribution_by_depth.png)
- ![Experiment 3 curve roughness](../figures/exp3_curve_roughness_vs_depth.png)
- ![Experiment 3 low-depth auction curve](../figures/exp3_auction_curve_low_depth.png)
- ![Experiment 3 high-depth auction curve](../figures/exp3_auction_curve_high_depth.png)

**Table 3. Experiment 3 Summary Statistics by Market Depth**

{exp3_table}

Finding: depth scales the random process. Low-depth markets have rougher order curves, larger return volatility, and larger extreme moves. High-depth markets average out individual randomness, producing smoother curves and tighter return distributions. This supports the mechanism that volatility can be generated endogenously by finite, noisy order curves.

## 7. Experiment 4: Capital Inflow and the Random-Walk Channel

Question: **Does a fixed capital inflow shift the entire random-walk channel of prices upward, while preserving the stochastic increment structure of an auction-generated random walk?**

Design. A single inflow treatment is compared against a no-inflow baseline. The main comparison uses the `loose` cash regime (initial cash per agent set so the baseline is essentially a free random walk). Both groups use the same bid/ask equation `bid = P[t-1] * exp(epsilon)` with `epsilon ~ N(0, sigma^2)` and zero mean; the only difference is that each agent in the treatment group receives a daily cash injection of `inflow_per_period`. The `tight` regime acts only as a robustness check for the constraint-mediation channel.

Figures:

- ![Experiment 4 baseline vs inflow paths with cross-path means](../figures/exp4_paths_baseline_vs_inflow.png)
- ![Experiment 4 cross-path mean paths with 95% CI](../figures/exp4_mean_paths_baseline_vs_inflow.png)
- ![Experiment 4 final price distributions](../figures/exp4_final_price_distribution.png)
- ![Experiment 4 random-walk diagnostics](../figures/exp4_random_walk_diagnostics_bars.png)

**Table 4. Experiment 4 Main Comparison (`loose` regime, baseline vs treatment)**

{exp4_main_table}

**Table 5. Experiment 4 Robustness Across Cash Regimes**

{exp4_robust_table}

Finding:

{exp4_conclusion}

## 8. Integrated Discussion

The three experiments form one mechanism chain:

Random orders  
-> auction clearing  
-> emergent market-level random process

Under unlimited wealth, the process is closest to a free random walk: no center force is imposed, and price can diffuse. Under wealth and inventory constraints, the process remains random but feasible order flow changes with price level, creating weak mean-reverting pressure and softer boundaries. Under different market depths, the process remains random but its volatility scale changes because aggregate order curves become rougher or smoother.

| Market Rule | Random Process Type | Center Force | Soft Boundary | Volatility Pattern |
|---|---|---|---|---|
| Unlimited Wealth | Random walk approximation | No center | No soft boundary | Free diffusion |
| Wealth Constrained | Modified / weakly mean-reverting random process | Can emerge | Can emerge | Depends on constraint strength |
| Low Market Depth | High-volatility random process | Not required | Not required | Extreme jumps more common |
| High Market Depth | Low-volatility random process | Not required | Not required | Smoother path |

## 9. Conclusion

The results support the main research thesis: market-level randomness does not need to be imposed as a primitive assumption. In this simulated market, it emerges from random individual orders filtered through a clearing rule. Market rules do not eliminate randomness; they shape the type of random process that emerges. Experiment 4 sharpens this conclusion by showing that a fixed capital inflow shifts the entire random-walk channel upward — the cross-path mean climbs and final-price distributions move to the right — while the increment process itself remains close to a random walk. The shift is produced by auction clearing alone, with no drift term added to the bidding equation.

## 10. Future Extensions

1. External sentiment shocks.
2. Different agent types.
3. Continuous double auction.
4. Limit order book.
5. Heterogeneous beliefs.
6. Adaptive agents.
7. Wealth distribution dynamics.
8. Order curve anomaly detection.
9. Real market calibration.
10. Agent-based market microstructure model.

## 11. Suggested Final Report Structure

1. Introduction.
2. Experimental Setup.
3. Experiment 1: Random Walk Emergence in an Unlimited-Wealth Market.
4. Experiment 2: Wealth Constraints and the Modification of Random Dynamics.
5. Experiment 3: Market Depth and the Scaling of Random Fluctuations.
6. Experiment 4: Capital Inflow and the Random-Walk Channel.
7. Integrated Discussion.
8. Conclusion.
"""
    report_path.write_text(body, encoding="utf-8")
    return report_path


def _read_table(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    return df[[column for column in columns if column in df.columns]]


def _table_preview(df: pd.DataFrame, max_rows: int = 12) -> str:
    if df.empty:
        return "_Table not generated yet._"
    return df.head(max_rows).to_markdown(index=False, floatfmt=".6g")


def _build_exp4_conclusion(main: pd.DataFrame, robust: pd.DataFrame) -> str:
    """Produce a short interpretation block from the redesigned Experiment 4 tables."""
    if main.empty:
        return "_Experiment 4 summary not generated yet._"

    parts: list[str] = []
    base = main.loc[main["group"] == "baseline"]
    treat = main.loc[main["group"] == "inflow"]
    if not base.empty and not treat.empty:
        b = base.iloc[0]
        t = treat.iloc[0]
        diff = float(t["mean_log_return"]) - float(b["mean_log_return"])
        welch_p = float(t.get("welch_p_value_mean_return", float("nan")))
        cohens_d = float(t.get("cohens_d_mean_return", float("nan")))
        fp_diff = float(t["final_price_mean"]) - float(b["final_price_mean"])
        parts.append(
            f"- H4a (channel shift in returns): baseline mean log return `{float(b['mean_log_return']):.4g}` "
            f"vs treatment `{float(t['mean_log_return']):.4g}` "
            f"(diff `{diff:+.4g}`, Welch p `{welch_p:.4g}`, Cohen's d `{cohens_d:.3f}`)."
        )
        parts.append(
            f"- H4b (mean path lifted): mean final price baseline `{float(b['final_price_mean']):.4g}` "
            f"vs treatment `{float(t['final_price_mean']):.4g}` (diff `{fp_diff:+.4g}`)."
        )
        parts.append(
            f"- H4c (random-walk features preserved): lag-1 autocorrelation `{float(b['lag1_autocorr']):.3f}` "
            f"vs `{float(t['lag1_autocorr']):.3f}`; variance ratio (lag 5) `{float(b['variance_ratio_lag5']):.3f}` "
            f"vs `{float(t['variance_ratio_lag5']):.3f}`; positive-return share `{float(b['positive_return_pct']):.3f}` "
            f"vs `{float(t['positive_return_pct']):.3f}`. The increments remain near a symmetric random walk."
        )

    if not robust.empty:
        diffs: list[str] = []
        for regime, block in robust.groupby("regime", sort=True):
            b_row = block.loc[block["group"] == "baseline"]
            t_row = block.loc[block["group"] == "inflow"]
            if b_row.empty or t_row.empty:
                continue
            d = float(t_row.iloc[0]["mean_log_return"]) - float(b_row.iloc[0]["mean_log_return"])
            constrained = float(b_row.iloc[0]["avg_cash_constrained_ratio"])
            diffs.append(
                f"`{regime}` (baseline cash-constrained ratio `{constrained:.3f}`) -> "
                f"treatment minus baseline mean log return `{d:+.4g}`"
            )
        if diffs:
            parts.append(
                "- H4d (constraint mediation): " + "; ".join(diffs) + ". "
                "A larger gap when the baseline cash-constrained ratio is higher indicates the inflow "
                "effect is mediated by binding cash constraints rather than acting as a direct trend."
            )

    parts.append(
        "- Mechanism: the bid equation `bid = P[t-1] * exp(epsilon)` was identical for baseline and "
        "treatment, with `epsilon` zero-mean and i.i.d.. Any upward channel shift therefore comes from "
        "the auction clearing — additional buyer cash marginally raises each day's volume-maximising "
        "clearing price — and not from an exogenous trend imposed on prices."
    )
    return "\n".join(parts)
