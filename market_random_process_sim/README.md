# market_random_process_sim

This project is an educational and research-oriented market simulator. It studies how a market-level stochastic price process can emerge from individual random orders and market rules.

Traditional financial models often start with:

**Price Process = Random Process**

This project starts from:

**Random Individual Orders + Market Rules -> Emergent Market-Level Random Process**

The simulator is not a trading strategy, prediction system, manipulation model, or pump-and-dump optimizer. It is a modular market microstructure experiment built around daily auction clearing.

## Installation

Use Python 3.10+.

```bash
pip install -r requirements.txt
```

## Run Experiments

Run all experiments:

```bash
python main.py --experiment all --seed 42 --n_days 500
```

The default command also runs all experiments:

```bash
python main.py
```

Run a single experiment:

```bash
python main.py --experiment exp1
python main.py --experiment exp2
python main.py --experiment exp3
python main.py --experiment exp4
```

Run supplemental analyses:

```bash
python main.py --experiment ablation
python main.py --experiment roughness
python main.py --experiment benchmark
python main.py --experiment volume
python main.py --experiment extreme_cases
python main.py --experiment sensitivity
python main.py --experiment formal_model
python main.py --experiment hypotheses
python main.py --experiment diagnostics
python main.py --experiment index
```

Run the full research output pipeline:

```bash
python main.py --experiment full --seed 42 --n_days 500 --n_runs 100 --output_dir outputs
```

For a faster smoke run, reduce `--n_days` and `--n_runs`.

Use a custom output directory:

```bash
python main.py --experiment all --output_dir outputs
```

## Outputs

The program creates output folders automatically:

- `outputs/figures/`: experiment figures.
- `outputs/tables/`: metrics tables.
- `outputs/simulation_results/`: raw simulation paths and representative runs.
- `outputs/reports/market_random_process_report.md`: generated Markdown report.

## Market Design

Each day has one auction:

Yesterday Price  
-> Generate Random Orders  
-> Auction Clearing  
-> Today Price and Volume  
-> Repeat

Order prices are sampled around yesterday's price with a lognormal form:

```text
log_price = log(P[t-1]) + epsilon
epsilon ~ Normal(0, sigma)
```

The auction evaluates all candidate prices and computes:

- `demand(p)`: buy quantity with bid price >= `p`.
- `supply(p)`: sell quantity with ask price <= `p`.
- `matched_volume(p) = min(demand(p), supply(p))`.
- `imbalance(p) = abs(demand(p) - supply(p))`.

The clearing rule is:

1. Maximize matched volume.
2. Minimize imbalance.
3. Use the median candidate price if there is still a tie.

## Experiment 1: Unlimited-Wealth Random Market

This experiment removes wealth and inventory constraints. Buyers can buy any quantity, sellers can sell any quantity, and all order prices are random around yesterday's market price.

Research question: can a Gaussian random walk-like process emerge from purely random orders and an auction rule?

Expected interpretation: because each new order distribution is centered on the previous clearing price, price direction is not predictable from the model design. Apparent trends in one path are stochastic realizations, not strategy or information.

The upgraded diagnostics table `outputs/tables/exp1_random_walk_diagnostics.csv` summarizes:

- mean log return
- log-return volatility
- lag-1 return autocorrelation
- lag-5 variance ratio
- percentage of positive return days
- final price
- max drawdown
- skewness and kurtosis

## Experiment 2: Wealth-Constrained Random Market

Agents have cash and share inventory:

- Buy quantity is capped by `cash / bid_price`.
- Sell quantity is capped by current shares.
- Executable auction orders are filled pro rata.
- Agent cash and share balances are updated after each daily clearing.

Research question: do constraints turn a freely diffusing random process into a process with soft boundaries or mean reversion?

The report estimates:

```text
return[t+1] = alpha + beta * log(P[t] / P[0]) + error
```

A negative `beta` is evidence that positive deviations tend to be followed by lower returns.

The upgraded repeated experiment runs each wealth multiplier 100 times and writes:

- `outputs/tables/exp2_repeated_run_metrics.csv`
- `outputs/tables/exp2_summary_by_wealth_multiplier.csv`
- `outputs/simulation_results/exp2_repeated_paths.csv`

The summary table reports mean beta, beta standard deviation, 95% confidence intervals, average price variance, average log-deviation variance, average volatility, and final-price dispersion. The key question is whether wealth constraints systematically modify the random process rather than whether one path happens to revert.

## Experiment 3: Market Depth and Volatility

This experiment changes the number of buyers and sellers:

```text
20, 50, 100, 200, 500, 1000, 5000
```

Each depth level is run repeatedly. The key question is whether market depth changes volatility by smoothing aggregate demand and supply curves.

Expected interpretation: low-depth markets have rough order curves and larger jumps. High-depth markets average out individual randomness and produce smoother paths. Depth changes volatility intensity, not the existence of randomness.

The upgraded depth experiment computes auction-curve roughness:

```text
roughness = average of mean(abs(diff(normalized demand curve)))
            and mean(abs(diff(normalized supply curve)))
```

Demand and supply are each normalized by their own maximum quantity before differencing, so different depths are comparable. Lower roughness means smoother aggregate order curves. The summary table `outputs/tables/exp3_metrics.csv` includes average curve roughness alongside volatility and extreme-move statistics.

## Experiment 4: Capital Inflow and the Random-Walk Channel

This experiment tests whether a fixed capital inflow shifts the entire random-walk channel of auction-generated prices upward — and whether the random-walk structure of the increments is preserved while it shifts.

The bidding rule remains zero-drift lognormal around the previous price:

```text
bid[i, t] = P[t-1] * exp(epsilon_b)
ask[i, t] = P[t-1] * exp(epsilon_s)
epsilon_b, epsilon_s ~ Normal(0, sigma)
```

Two groups are compared with identical bid equations:

- **Baseline**: each agent receives zero daily inflow.
- **Treatment**: each agent receives `EXP4_INFLOW_TREATMENT` (default `10.0`) dollars in cash at the start of each day before orders are generated.

The main comparison runs both groups under the `loose` cash regime (so the baseline is essentially a free random walk) and applies Welch's two-sample t-test to the mean log return. The `tight` regime is run as a robustness check for the constraint-mediation channel.

Key recorded variables (per simulation path):

- Mean log return, its standard deviation, and per-path t-statistic.
- Lag-1 autocorrelation, lag-5 variance ratio, positive-return percentage.
- Final price, average trading volume, zero-trade periods.
- Average cash- and share-constrained agent ratios.

Outputs:

- `outputs/tables/exp4_main_comparison.csv` — main loose-regime baseline vs treatment table with Welch t/p and Cohen's d.
- `outputs/tables/exp4_robustness_by_regime.csv` — same contrast in both `tight` and `loose` regimes.
- `outputs/tables/exp4_path_metrics.csv` — per-path diagnostics for every run.
- `outputs/figures/exp4_paths_baseline_vs_inflow.png` — spaghetti paths with bold cross-path means.
- `outputs/figures/exp4_mean_paths_baseline_vs_inflow.png` — cross-path means with 95% CI bands.
- `outputs/figures/exp4_final_price_distribution.png` — final-price histograms by group.
- `outputs/figures/exp4_random_walk_diagnostics_bars.png` — lag-1 autocorr, variance ratio, positive-return share by group.

## Main Report Figures and Tables

Key added figures:

- `outputs/figures/exp1_random_walk_diagnostics_table.png`
- `outputs/figures/exp2_multiple_paths_by_wealth_constraint.png`
- `outputs/figures/exp2_beta_distribution_by_wealth_multiplier.png`
- `outputs/figures/exp2_mean_beta_ci_by_wealth_multiplier.png`
- `outputs/figures/exp2_log_deviation_variance_by_wealth_multiplier.png`
- `outputs/figures/exp3_multiple_paths_by_market_depth.png`
- `outputs/figures/exp3_curve_roughness_vs_depth.png`
- `outputs/figures/exp4_paths_baseline_vs_inflow.png`
- `outputs/figures/exp4_mean_paths_baseline_vs_inflow.png`

Key added tables:

- `outputs/tables/exp1_random_walk_diagnostics.csv`
- `outputs/tables/exp2_summary_by_wealth_multiplier.csv`
- `outputs/tables/exp2_repeated_run_metrics.csv`
- `outputs/tables/exp3_metrics.csv`
- `outputs/tables/exp4_main_comparison.csv`

## Tests

Run tests from the project directory:

```bash
pytest
```

The tests cover auction clearing, market simulation reproducibility, positive prices, returns, max drawdown, autocorrelation, and mean reversion regression output.

## Future Extensions

The current code intentionally avoids manipulation logic and focuses on the first three stochastic-process layers. The structure leaves room for:

1. `SentimentShock`: external news or sentiment shifts that move order distribution centers.
2. `HeterogeneousAgents`: different sigmas, wealth levels, risk preferences, and trading frequencies.
3. `AdaptiveAgents`: agents that update quotes from past prices.
4. `ContinuousDoubleAuction`: continuous matching instead of one daily auction.
5. `LimitOrderBook`: persistent order book state.
6. `MarketImpact`: large orders affecting prices.
7. `AnomalyDetection`: detecting unusual order curves without optimizing manipulation.
8. `Calibration`: fitting sigma, depth, and volume parameters to real data.
9. Wealth distribution dynamics.
10. Agent-based market microstructure models.
