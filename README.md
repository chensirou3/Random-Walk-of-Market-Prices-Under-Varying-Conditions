# Random Walk of Market Prices Under Varying Conditions

An agent-based market simulator that studies how a market-level stochastic price
process **emerges** from individual random orders cleared by a uniform-price
auction. Instead of postulating

> Price process = random process

the project starts from

> Random individual orders + market rules &nbsp;⟶&nbsp; emergent market-level random process

and runs four controlled experiments that progressively add wealth constraints,
market depth, and capital inflow on top of a free-diffusion baseline.

The simulator is **not** a trading strategy, prediction system, manipulation
model, or pump-and-dump optimizer. It is a modular market-microstructure
laboratory built around one daily auction.

---

## Table of Contents

1. [Repository layout](#repository-layout)
2. [Installation](#installation)
3. [Quick start](#quick-start)
4. [Command-line interface](#command-line-interface)
5. [Default parameters](#default-parameters)
6. [Market design](#market-design)
7. [Experiment 1 — Random-walk emergence](#experiment-1--random-walk-emergence-unlimited-wealth)
8. [Experiment 2 — Wealth constraints](#experiment-2--wealth-constraints-and-mean-reversion)
9. [Experiment 3 — Market depth](#experiment-3--market-depth-and-volatility-scaling)
10. [Experiment 4 — Capital inflow](#experiment-4--capital-inflow-and-the-random-walk-channel)
11. [Supplemental analyses](#supplemental-analyses)
12. [Output directory layout](#output-directory-layout)
13. [Hypotheses map](#hypotheses-map)
14. [Tests and reproducibility](#tests-and-reproducibility)
15. [Future extensions](#future-extensions)

---

## Repository layout

```
.
├── README.md                       # this file
├── .gitignore
└── market_random_process_sim/
    ├── README.md                   # project-internal README
    ├── requirements.txt
    ├── config.py                   # all global parameters
    ├── main.py                     # CLI entry point
    ├── simulator/
    │   ├── orders.py               # lognormal order generation
    │   ├── auction.py              # uniform-price auction clearing
    │   ├── agents.py               # cash + share inventory bookkeeping
    │   ├── market.py               # day-by-day market loop
    │   └── experiments.py          # four experiments + supplementals
    ├── analysis/
    │   ├── metrics.py              # log-returns, autocorr, VR, Welch t, Cohen's d, Gini
    │   ├── plots.py                # all figure helpers
    │   ├── report.py               # markdown report writer
    │   ├── formal_model.py         # formula + per-experiment method figures
    │   └── index.py                # output index, hypotheses, config summary
    ├── tests/                      # pytest suite (auction, market, metrics, repro)
    ├── newoutput/                  # latest committed full run (n_runs=100, n_days=500)
    └── outputs/                    # earlier full-run artefacts (kept for reference)
```

---

## Installation

Python 3.10+ is required. From the project subdirectory:

```bash
cd market_random_process_sim
pip install -r requirements.txt
```

Dependencies: `numpy>=1.24`, `pandas>=2.0`, `matplotlib>=3.7`, `pytest>=7.0`,
`tabulate>=0.9`.

---

## Quick start

Run all four core experiments with the default configuration
(`seed=42`, `n_days=500`, `n_runs=100`):

```bash
cd market_random_process_sim
python main.py --experiment all
```

Run the **complete research pipeline** (core + ablation + roughness + benchmark
+ volume + extreme cases + sensitivity + cross-experiment summary + formal-model
figures + hypotheses + report + index):

```bash
python main.py --experiment full --seed 42 --n_days 500 --n_runs 100 --output_dir newoutput
```

A fast smoke run (about a minute):

```bash
python main.py --experiment exp4 --seed 42 --n_days 60 --n_runs 8 --output_dir outputs_smoke
```

---

## Command-line interface

`python main.py [--experiment NAME] [--seed INT] [--n_days INT] [--n_runs INT] [--output_dir PATH]`

| `--experiment` value | What it runs |
|---|---|
| `all` *(default)* | Experiments 1–4 + cross-experiment comparison + report |
| `full` | `all` + ablation + roughness + benchmark + volume + extreme cases + sensitivity + formal-model figures + hypotheses + reproducibility diagnostics + index |
| `exp1` … `exp4` | One core experiment + report |
| `ablation` | Constraint ablation around Experiment 2 (buyer-cash only / seller-inventory only / both) |
| `roughness` | Roughness-vs-volatility scan around Experiment 3 |
| `benchmark` | Auction-clearing path vs Gaussian random-walk benchmark |
| `volume` | Volume vs absolute-return relationship |
| `extreme_cases` | Extreme-up / extreme-down event windows from Experiment 1 paths |
| `sensitivity` | Sweep over `sigma`, `q_max`, and `n_days` |
| `cross` | Side-by-side cross-experiment metric table |
| `formal_model` | Regenerate formula + per-experiment **method** figures (no simulation) |
| `hypotheses` | Regenerate `hypotheses.md` and `hypotheses_mapping.csv` |
| `diagnostics` | Reproducibility diagnostics (seed strategy, file-by-file determinism) |
| `index` | Build `index.md` listing every artefact |
| `config` | Dump the resolved parameter grid to markdown + CSV |

CLI flags:

* `--seed` — global random seed (default `42`).
* `--n_days` — simulated trading days (default `500`).
* `--n_runs` — repeated runs per parameter cell for the supplementals (default `100`; Experiment 4 caps at `EXP4_N_RUNS=100`).
* `--output_dir` — root folder for all artefacts (default `outputs`).

---

## Default parameters

All numbers below live in `market_random_process_sim/config.py` and are
shared by the four core experiments unless an experiment explicitly overrides
them.

| Symbol | Variable | Default | Meaning |
|---|---|---|---|
| $P_0$ | `DEFAULT_INITIAL_PRICE` | `100.0` | Starting price |
| $\sigma$ | `DEFAULT_PRICE_SIGMA` | `0.03` | Std-dev of the lognormal bid noise |
| $q_{\max}$ | `DEFAULT_MAX_QTY` | `100.0` | Upper bound on the per-order quantity |
| $N_b, N_s$ | `DEFAULT_N_BUYERS`, `DEFAULT_N_SELLERS` | `200, 200` | Buyers and sellers per day (Exp 1) |
| $T$ | `DEFAULT_N_DAYS` | `500` | Days per path |
| $N$ | `DEFAULT_N_AGENTS` | `500` | Persistent agents (Exp 2 / Exp 4) |
| $W_0$ | `DEFAULT_INITIAL_CASH_PER_AGENT` | `20 000.0` | Baseline cash endowment |
| $H_0$ | `DEFAULT_INITIAL_SHARES_PER_AGENT` | `200.0` | Baseline share endowment |
| — | `DEFAULT_AGENT_PARTICIPATION_RATE` | `0.8` | Fraction of agents trading per day |
| — | `EXP2_WEALTH_MULTIPLIERS` | `[0.5, 1, 2, 5, 10]` | Cash & share scaling factors |
| — | `EXP3_DEPTH_LEVELS` | `[20, 50, 100, 200, 500, 1000, 5000]` | Number of buyers = sellers per day |
| $c$ | `EXP4_INFLOW_TREATMENT` | `10.0` | Inflow per agent per day in the treatment group |
| — | `EXP4_MAIN_REGIME` | `"loose"` | Cash regime used for the main two-group contrast |
| — | `EXP4_PATHS_FOR_FIGURE` | `50` | Paths drawn in the spaghetti figure |
| seed | `DEFAULT_SEED` | `42` | Master RNG seed |


---

## Market design

Each day has **one** uniform-price auction. The day-by-day loop is

```
P[t-1]
  ├─ generate random buy orders  {(p_i^b, q_i^b)}
  ├─ generate random sell orders {(p_j^s, q_j^s)}
  ├─ build aggregate D(p), S(p)
  ├─ pick clearing price P[t] = argmax V(p), then argmin |D(p)-S(p)|, median tie-break
  └─ record P[t], V[t], constraint diagnostics
```

**Order submission rule** (zero-drift lognormal around the previous price):

$$
p_i^b = P_{t-1} \exp(\epsilon_i^b), \qquad p_j^s = P_{t-1} \exp(\epsilon_j^s),
\qquad \epsilon \sim \mathcal{N}(0, \sigma^2), \qquad q \sim U(1, q_{\max})
$$

**Budget and inventory constraints** (active in Exp 2/4, off in Exp 1/3):

$$
q_i^b \cdot p_i^b \le W_i, \qquad q_j^s \le H_j
$$

**Auction clearing rule**:

$$
D_t(p)=\sum_i q_i^b\,\mathbf{1}(p_i^b\ge p),\quad
S_t(p)=\sum_j q_j^s\,\mathbf{1}(p_j^s\le p),
$$

$$
V_t(p)=\min(D_t(p),S_t(p)),\quad
I_t(p)=|D_t(p)-S_t(p)|,
$$

$$
P_t=\arg\max_p V_t(p),\ \ \min_p I_t(p),\ \text{median tie-break}.
$$

Compact reference figures are auto-generated under
`<output_dir>/formula_figures/`:

* `formal_model_order_submission.png`
* `formal_model_auction_clearing.png`
* `formal_model_price_dynamics_metrics.png`
* `formal_model_roughness_metric.png`
* `exp1_method_formulas.png`, `exp2_method_formulas.png`, `exp3_method_formulas.png`, `exp4_method_formulas.png`

---

## Experiment 1 — Random-walk emergence (unlimited wealth)

**Question.** Can a Gaussian random-walk-like price process emerge from purely
random orders cleared by an auction, without any drift built into the bid
equation?

**Setup.** No cash or inventory constraints. 200 fresh buyers + 200 fresh
sellers per day, $P_0=100$, $\sigma=0.03$, $q_{\max}=100$, $T=500$, plus 50
independent replication paths.

**Diagnostics** (saved to `tables/exp1_random_walk_diagnostics.csv`):

* mean log-return $\bar r$
* return volatility $\sigma_r$
* lag-1 autocorrelation $\rho_1$
* variance ratio $VR(5)$
* positive-return-day percentage
* final price, max drawdown, skewness, kurtosis

**Hypothesis H1.** The auction mechanism produces $\rho_1\approx 0$,
$VR(k)\approx 1$, and a return distribution close to $\mathcal{N}(0,\sigma_r^2)$.

**Headline outputs.**

| File | Content |
|---|---|
| `figures/exp1_price_path.png` | Single representative price path |
| `figures/exp1_return_distribution.png` | Empirical return histogram |
| `figures/exp1_multiple_paths.png` | 50 replication paths |
| `figures/exp1_random_walk_diagnostics_table.png` | Compact diagnostics card |
| `figures/exp1_auction_vs_gaussian_*.png` | Auction-clearing vs Gaussian benchmark |
| `tables/exp1_metrics.csv` | Single-path metrics |
| `tables/exp1_random_walk_diagnostics.csv` | Replication diagnostics |

---

## Experiment 2 — Wealth constraints and mean reversion

**Question.** Do binding cash and inventory constraints turn a freely diffusing
random process into a process with soft boundaries or weak mean reversion?

**Setup.** 500 persistent agents (participation rate 0.8) endowed with cash
$W_i=k\cdot W_0$ and shares $H_j=k\cdot H_0$, where the wealth multiplier
$k\in\{0.5,1,2,5,10\}$. Each multiplier is replicated 100 times.

**Constrained order generation.** Buy quantity is capped by
$\lfloor W_i / p_i^b \rfloor$ and sell quantity by $H_j$; orders that would
violate either cap are clipped.

**Mean-reversion test.** Let $x_t=\log(P_t/P_0)$ be the log deviation. The
report estimates the OLS regression

$$ r_{t+1} = \alpha + \beta\,x_t + \varepsilon_t $$

per path; $\beta<0$ is evidence of weak mean reversion.

**Hypotheses.**
* **H2** — tighter constraints (smaller $k$) yield more negative $\beta$ and bounded paths.
* **H2a** — cross-path final-price dispersion shrinks as $k$ decreases.

**Headline outputs.**

| File | Content |
|---|---|
| `figures/exp2_multiple_paths_by_wealth_constraint.png` | Path bundles per $k$ |
| `figures/exp2_beta_distribution_by_wealth_multiplier.png` | $\beta$ distributions |
| `figures/exp2_mean_beta_ci_by_wealth_multiplier.png` | $\bar\beta$ with 95 % CI |
| `figures/exp2_log_deviation_variance_by_wealth_multiplier.png` | Dispersion vs $k$ |
| `tables/exp2_summary_by_wealth_multiplier.csv` | Mean $\beta$, CIs, dispersion |
| `tables/exp2_repeated_run_metrics.csv` | Per-run metrics across 5×100 runs |

---

## Experiment 3 — Market depth and volatility scaling

**Question.** Does increasing the number of buyers and sellers per day reduce
volatility and smooth the aggregate demand-/supply-curves?

**Setup.** Same lognormal order rule and clearing rule as Experiment 1.
Sweep market depth $N\in\{20,50,100,200,500,1000,5000\}$ (buyers and sellers
symmetric); 100 replications per depth, 20 representative paths saved.

**Volatility and extreme moves.**

$$ \sigma_r(N) = \mathrm{Std}(r_t \mid N), \qquad E(N) = \max_t |r_t|. $$

**Normalized order-curve roughness.**

$$
D_t^{\mathrm{norm}}(p)=\frac{D_t(p)}{\max_p D_t(p)},\qquad
S_t^{\mathrm{norm}}(p)=\frac{S_t(p)}{\max_p S_t(p)},
$$

$$
R_D=\overline{|\Delta D_t^{\mathrm{norm}}|},\quad
R_S=\overline{|\Delta S_t^{\mathrm{norm}}|},\quad
R=\tfrac{1}{2}(R_D+R_S).
$$

Normalising each side by its own maximum makes roughness comparable across
depths.

**Hypotheses.**
* **H3** — deeper markets reduce $\sigma_r$ and $R$, with the canonical scaling $\sigma_r(N)\propto 1/\sqrt{N}$.
* **H3a** — across depths $\mathrm{Corr}(\sigma_r, R) > 0$.

**Headline outputs.**

| File | Content |
|---|---|
| `figures/exp3_volatility_vs_depth.png` | $\sigma_r$ vs $N$ |
| `figures/exp3_curve_roughness_vs_depth.png` | $R$ vs $N$ |
| `figures/exp3_roughness_vs_volatility.png` | $R$ vs $\sigma_r$ scatter |
| `figures/exp3_extreme_move_vs_depth.png` | $E$ vs $N$ |
| `figures/exp3_depth_roughness_volatility_chain.png` | Combined scaling chain |
| `tables/exp3_metrics.csv` | Mean / CI per depth |
| `tables/exp3_roughness_volatility_summary.csv` | Roughness × volatility summary |

---

## Experiment 4 — Capital inflow and the random-walk channel

**Question.** Does a fixed daily capital inflow shift the **whole random-walk
channel** of auction-cleared prices upward while preserving its random-walk
shape?

**Setup.** A two-group design under the main `loose` cash regime
($W_0=20\,000$, $H_0=200$):

* **Baseline** — no inflow.
* **Treatment** — every agent receives $c=10$ extra cash at the start of every day, *before* orders are generated.

100 paths per group; 50 paths drawn in the spaghetti figure. The same contrast
is rerun under the `tight` regime ($W_0=200$) as a robustness check for the
constraint-mediation channel.

**Cash update**:

$$
W_{i,t} = W_{i,t-1} - \text{Spent}_{i,t-1} + \text{Received}_{i,t-1}
        + c\cdot \mathbf{1}(\text{treatment}).
$$

The bid equation is **identical in both groups** —
$\text{bid}_{i,t}=P_{t-1}\exp(\epsilon_{i,t})$, $\epsilon\sim\mathcal{N}(0,\sigma^2)$
— so any drift can only come from the auction filtering different *feasible*
order sets, not from a hand-coded trend term.

**Random-walk channel-shift metrics** (per path):
$\bar r$, $\mathrm{Std}(r_t)$, $\rho_1$, $VR(5)$, $\Pr(r_t>0)$, final price,
average volume, zero-trade days, average cash- and share-constrained ratios.

**Cross-group test.** Welch's two-sample $t$ on $\bar r$ plus Cohen's $d$:

$$
t=\frac{\bar r_{\text{trt}}-\bar r_{\text{base}}}{\sqrt{s_{\text{trt}}^2/n+s_{\text{base}}^2/n}},
\qquad d=\frac{\bar r_{\text{trt}}-\bar r_{\text{base}}}{s_{\text{pooled}}}.
$$

**Hypotheses.**
* **H4** — treatment shifts $\bar r$ above zero (channel shifts up).
* **H4a** — random-walk shape preserved ($\rho_1$, $VR(5)$ unchanged within tolerance).
* **H4b** — final-price distribution shifts upward.
* **H4c** — trading volume rises and zero-trade days fall.
* **H4d** — channel shift is *larger* under the `tight` regime (constraint mediation).

**Headline results from the committed full run** (`newoutput/tables/exp4_main_comparison.csv`, $n=100$ per group, $T=500$):

| metric | baseline (loose) | inflow=10 (loose) |
|---|---:|---:|
| `mean_log_return` | $-7.9\!\times\!10^{-7}$ | $3.57\!\times\!10^{-4}$ |
| `std_log_return` | $0.00323$ | $0.00320$ |
| `lag1_autocorr` | $0.0145$ | $0.0090$ |
| `variance_ratio_lag5` | $1.035$ | $1.027$ |
| `positive_return_pct` | $50.08\%$ | $54.58\%$ |
| `final_price_mean` | $100.01$ | $119.64$ |
| `final_price_std` | $3.04$ | $3.95$ |
| Welch $t$ on $\bar r$ | — | $\approx 39.96$ |
| Cohen's $d$ | — | $\approx 5.65$ |

Under the `tight` regime (`exp4_robustness_by_regime.csv`) the same inflow
moves the final-price mean from $1.03$ to $18.96$, i.e. the constraint regime
amplifies the same nominal inflow by an order of magnitude — direct evidence
for **H4d**.

**Headline outputs.**

| File | Content |
|---|---|
| `figures/exp4_paths_baseline_vs_inflow.png` | 50 baseline + 50 inflow paths with bold cross-path means |
| `figures/exp4_mean_paths_baseline_vs_inflow.png` | Cross-path means with 95 % CI bands |
| `figures/exp4_final_price_distribution.png` | Final-price histograms by group |
| `figures/exp4_random_walk_diagnostics_bars.png` | $\rho_1$, $VR(5)$, positive-return % bars |
| `tables/exp4_main_comparison.csv` | Main loose-regime two-group contrast |
| `tables/exp4_robustness_by_regime.csv` | Loose + tight robustness table |
| `tables/exp4_path_metrics.csv` | Per-path diagnostics for every run |
| `reports/exp4_findings.md` | Narrative findings |

---

## Supplemental analyses

| `--experiment` | Purpose | Key artefacts |
|---|---|---|
| `ablation` | Toggle buyer-cash and seller-inventory constraints to isolate each side | `tables/exp2_ablation_summary.csv`, `figures/exp2_ablation_*.png`, `reports/exp2_ablation_findings.md` |
| `roughness` | Joint scan of curve roughness and volatility across depths | `tables/exp3_roughness_volatility_summary.csv`, `figures/exp3_roughness_vs_volatility.png` |
| `benchmark` | Compare auction-cleared paths against pure Gaussian random walks calibrated to the same $\sigma_r$ | `tables/exp1_gaussian_benchmark_comparison.csv`, `figures/exp1_auction_vs_gaussian_*.png` |
| `volume` | Volume vs absolute return at different parameter settings | `tables/volume_analysis_summary.csv`, `figures/exp1_abs_return_vs_volume.png` |
| `extreme_cases` | Pull the largest up- and down-day windows from the Exp-1 paths and zoom into their auction curves | `tables/extreme_path_case_summary.csv`, `figures/extreme_case_*.png` |
| `sensitivity` | Sweep $\sigma\in\{0.01,0.03,0.05,0.08\}$, $q_{\max}\in\{50,100,200\}$, $T\in\{250,500,1000\}$ | `tables/sensitivity_*_summary.csv`, `reports/sensitivity_analysis_summary.md` |
| `cross` | One side-by-side comparison table across the four experiments | `tables/cross_experiment_summary.csv` + `.md`, `figures/cross_experiment_process_comparison.png` |
| `formal_model` | Regenerate formula and method figures (no simulation) | `formula_figures/*.png`, `reports/formal_model.md` |
| `hypotheses` | Map H1…H4d to the artefact that supports each one | `reports/hypotheses.md`, `tables/hypotheses_mapping.csv` |
| `diagnostics` | Re-run a small batch with the same seed and check byte-identical outputs | `diagnostics/reproducibility_check.{md,json}` |
| `index` | Build a master index of every file produced | `index.md` |
| `config` | Dump the resolved parameter grid | `tables/experiment_config_summary.{csv,md}`, `reports/parameter_documentation.md` |

---

## Output directory layout

After `python main.py --experiment full --output_dir newoutput` you get:

```
newoutput/
├── index.md                                # master index of every artefact
├── figures/                                # all PNG plots (~60 files)
├── formula_figures/                        # 4 generic + 4 per-experiment method figures
├── tables/                                 # 22 CSV + 2 MD tables
├── simulation_results/                     # representative raw paths (CSV)
├── reports/
│   ├── market_random_process_report.md     # main narrative report
│   ├── exp4_findings.md                    # Experiment-4-specific write-up
│   ├── exp2_ablation_findings.md           # constraint-ablation write-up
│   ├── formal_model.md                     # symbolic model
│   ├── hypotheses.md                       # H1…H4d → evidence map
│   ├── parameter_documentation.md          # parameter dictionary
│   └── sensitivity_analysis_summary.md     # sensitivity write-up
└── diagnostics/
    ├── reproducibility_check.md
    └── reproducibility_check.json
```

The committed `market_random_process_sim/newoutput/` directory contains the
exact full run for `seed=42, n_days=500, n_runs=100`.

---

## Hypotheses map

The full machine-readable mapping lives in
`<output_dir>/tables/hypotheses_mapping.csv`. The high-level summary:

| ID | Statement | Primary evidence |
|---|---|---|
| **H1** | Random-walk price emerges from random orders + auction clearing | Exp 1 diagnostics ($\rho_1\approx 0$, $VR\approx 1$) |
| **H2** | Wealth constraints induce weak mean reversion ($\beta<0$) | Exp 2 regression table |
| **H2a** | Cross-path dispersion shrinks as constraints tighten | Exp 2 dispersion-vs-$k$ figure |
| **H3** | Deeper markets reduce volatility and curve roughness | Exp 3 scaling figures |
| **H3a** | Roughness and volatility are positively correlated | `exp3_roughness_vs_volatility.png` |
| **H4** | A fixed inflow shifts the random-walk channel upward | Exp 4 main comparison ($t\approx 40$, $d\approx 5.7$) |
| **H4a** | Random-walk shape preserved under inflow | $\rho_1$, $VR(5)$ within tolerance |
| **H4b** | Final-price distribution shifts upward | `exp4_final_price_distribution.png` |
| **H4c** | Volume up, zero-trade days down | `exp4_path_metrics.csv` |
| **H4d** | Tight regime amplifies the inflow effect | `exp4_robustness_by_regime.csv` |

---

## Tests and reproducibility

Run the full pytest suite from the project subdirectory:

```bash
cd market_random_process_sim
pytest
```

The suite covers (`tests/`):

* `test_auction.py` — uniform-price clearing, tie-breaking, edge cases.
* `test_market.py` — day-by-day market loop, capital-inflow conservation laws, daily-diagnostics shape.
* `test_metrics.py` — autocorr, variance ratio, mean-return $t$-statistic, Gini.
* `test_reproducibility.py` — fixed-seed determinism for every public entry point.

**Reproducibility.** Every experiment derives its own RNG from the master
`--seed` plus a deterministic offset, so re-running with the same seed
(`--seed 42`) reproduces every CSV byte-for-byte and every figure pixel-for-pixel
on the same matplotlib version. The check is automated in
`diagnostics/reproducibility_check.md`.

---

## Future extensions

The current code intentionally avoids manipulation logic and focuses on the
first three stochastic-process layers (random walk → bounded process →
depth-scaled process), plus capital inflow as a fourth. The architecture
leaves room for:

1. **Sentiment shocks** — exogenous news / sentiment shifts in the order-distribution centre.
2. **Heterogeneous agents** — per-agent $\sigma$, wealth, risk preference, trading frequency.
3. **Adaptive agents** — agents that update quotes from past prices (momentum / contrarian rules).
4. **Continuous double auction** — replace the daily auction with continuous matching.
5. **Persistent limit-order book** — order-book state across days.
6. **Market impact** — large orders that move the clearing price within a day.
7. **Anomaly detection** — flag unusual order curves without optimising manipulation.
8. **Calibration** — fit $\sigma$, depth, and volume parameters to real data.
9. **Wealth-distribution dynamics** — Gini evolution, inequality and price feedback.
10. **Agent-based microstructure** — combinations of the above into one calibrated lab.

---

## License

No license file is included. The project is shared for educational and
research purposes; please contact the repository owner before reuse.

