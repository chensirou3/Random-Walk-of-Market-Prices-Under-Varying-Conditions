"""Formal model documentation and formula figure generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from analysis.plots import ensure_formula_figure_dir


def generate_formal_model(output_dir: str | Path = "outputs") -> dict[str, Path]:
    """Generate formal model markdown and formula figures."""
    output_path = Path(output_dir)
    report_dir = output_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    formula_dir = ensure_formula_figure_dir(output_path)

    markdown_path = report_dir / "formal_model.md"
    markdown_path.write_text(_formal_model_markdown(), encoding="utf-8")

    figures = {
        "order_submission": _formula_figure(
            formula_dir / "formal_model_order_submission.png",
            "Order Submission Rule",
            [
                r"$B_t=\{1,\ldots,N_b\},\quad S_t=\{1,\ldots,N_s\}$",
                r"Buy order: $(p_i^b,q_i^b)$,  Sell order: $(p_j^s,q_j^s)$",
                r"$p_i^b=P_{t-1}\exp(\epsilon_i^b),\quad p_j^s=P_{t-1}\exp(\epsilon_j^s)$",
                r"$\epsilon_i^b,\epsilon_j^s \sim \mathcal{N}(0,\sigma^2),\quad q\sim U(1,q_{\max})$",
            ],
        ),
        "auction_clearing": _formula_figure(
            formula_dir / "formal_model_auction_clearing.png",
            "Auction Clearing Rule",
            [
                r"$D_t(p)=\sum_i q_i^b\,\mathbf{1}(p_i^b\geq p)$",
                r"$S_t(p)=\sum_j q_j^s\,\mathbf{1}(p_j^s\leq p)$",
                r"$V_t(p)=\min(D_t(p),S_t(p)),\quad I_t(p)=|D_t(p)-S_t(p)|$",
                r"$P_t=p_t^*:\ \max V_t(p),\ \min I_t(p),\ \mathrm{median\ tie\ break}$",
            ],
        ),
        "price_dynamics": _formula_figure(
            formula_dir / "formal_model_price_dynamics_metrics.png",
            "Price Dynamics and Diagnostics",
            [
                r"$P_t=p_t^*$",
                r"$r_t=\log(P_t)-\log(P_{t-1})$",
                r"$x_t=\log(P_t/P_0)$",
                r"$r_{t+1}=\alpha+\beta x_t+\varepsilon_t,\quad \beta<0\Rightarrow\mathrm{mean\ reversion}$",
                r"$\sigma_r=\mathrm{Std}(r_t),\quad E=\max_t |r_t|$",
            ],
        ),
        "roughness": _formula_figure(
            formula_dir / "formal_model_roughness_metric.png",
            "Normalized Curve Roughness",
            [
                r"$D_t^{norm}(p)=D_t(p)/\max_p D_t(p)$",
                r"$S_t^{norm}(p)=S_t(p)/\max_p S_t(p)$",
                r"$R_D=\mathrm{mean}(|\Delta D_t^{norm}|),\quad R_S=\mathrm{mean}(|\Delta S_t^{norm}|)$",
                r"$R=(R_D+R_S)/2$",
            ],
        ),
    }
    method_figures = generate_experiment_method_figures(output_dir)
    return {"markdown": markdown_path, **figures, **method_figures}


def _formula_figure(path: Path, title: str, lines: list[str]) -> Path:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axis("off")
    ax.set_title(title, fontsize=15, pad=14)
    y = 0.82
    for line in lines:
        ax.text(0.04, y, line, fontsize=14, ha="left", va="center")
        y -= 0.17
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def _method_figure(
    path: Path,
    title: str,
    sections: list[tuple[str, list[tuple[str, str]]]],
) -> Path:
    """Render a per-experiment 'method and formulas' figure.

    Each section is a (heading, items) pair, and each item is a
    (kind, text) pair where ``kind`` is one of ``"text"`` (regular),
    ``"formula"`` (mathtext, larger), or ``"bullet"`` (bullet list).
    """
    fig, ax = plt.subplots(figsize=(12, 9.5))
    ax.axis("off")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14, loc="left")
    y = 0.97
    heading_step = 0.045
    text_step = 0.035
    formula_step = 0.050
    for heading, items in sections:
        ax.text(
            0.02, y, heading,
            fontsize=12.5, fontweight="bold", color="#1f4e79",
            ha="left", va="top",
        )
        y -= heading_step
        for kind, text in items:
            if kind == "formula":
                ax.text(0.05, y, text, fontsize=12.5, ha="left", va="top")
                y -= formula_step
            elif kind == "bullet":
                ax.text(0.05, y, "\u2022 " + text, fontsize=11, ha="left", va="top")
                y -= text_step
            else:
                ax.text(0.05, y, text, fontsize=11, ha="left", va="top")
                y -= text_step
        y -= 0.010
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


def generate_experiment_method_figures(output_dir: str | Path = "outputs") -> dict[str, Path]:
    """Generate one method-and-formula figure per experiment."""
    formula_dir = ensure_formula_figure_dir(Path(output_dir))
    return {
        "exp1_method": _method_figure(
            formula_dir / "exp1_method_formulas.png",
            "Experiment 1 — Random Walk Emergence (Unlimited-Wealth Auction Market)",
            _exp1_sections(),
        ),
        "exp2_method": _method_figure(
            formula_dir / "exp2_method_formulas.png",
            "Experiment 2 — Wealth Constraints and Modified Random Process",
            _exp2_sections(),
        ),
        "exp3_method": _method_figure(
            formula_dir / "exp3_method_formulas.png",
            "Experiment 3 — Market Depth and Volatility Scaling",
            _exp3_sections(),
        ),
        "exp4_method": _method_figure(
            formula_dir / "exp4_method_formulas.png",
            "Experiment 4 — Capital Inflow and Random-Walk Channel Shift",
            _exp4_sections(),
        ),
    }


def _exp1_sections() -> list[tuple[str, list[tuple[str, str]]]]:
    return [
        ("Setup", [
            ("bullet", "200 buyers and 200 sellers per day; no cash or inventory constraints."),
            ("bullet", "Initial price P0=100, sigma=0.03, q_max=100, n_days=500, seed=42."),
            ("bullet", "Single representative path plus 50 independent replication paths."),
        ]),
        ("Order submission (zero-drift lognormal bids)", [
            ("formula", r"$p_i^b=P_{t-1}\exp(\epsilon_i^b),\ \ p_j^s=P_{t-1}\exp(\epsilon_j^s)$"),
            ("formula", r"$\epsilon_i^b,\epsilon_j^s\sim\mathcal{N}(0,\sigma^2),\ \ q\sim U(1,q_{\max})$"),
        ]),
        ("Uniform-price auction clearing", [
            ("formula", r"$D_t(p)=\sum_i q_i^b\,\mathbf{1}(p_i^b\geq p),\ \ S_t(p)=\sum_j q_j^s\,\mathbf{1}(p_j^s\leq p)$"),
            ("formula", r"$P_t=\arg\max_p V_t(p),\ \min_p I_t(p),\ \mathrm{median\ tie\ break}$"),
        ]),
        ("Random-walk diagnostics", [
            ("formula", r"$r_t=\log P_t-\log P_{t-1},\ \ \rho_1=\mathrm{Corr}(r_t,r_{t-1})$"),
            ("formula", r"$VR(k)=\mathrm{Var}(r_t^{(k)})/(k\cdot\mathrm{Var}(r_t))$"),
        ]),
        ("Hypothesis H1", [
            ("text", "Auction clearing of zero-drift lognormal orders produces a random-walk-like price"),
            ("text", "process: rho_1 approx 0, VR(k) approx 1, returns approximately N(0, sigma_r^2)."),
        ]),
    ]


def _exp2_sections() -> list[tuple[str, list[tuple[str, str]]]]:
    return [
        ("Setup", [
            ("bullet", "500 agents endowed with cash W_i and shares H_j; participation rate 0.8."),
            ("bullet", "Wealth multipliers k in {0.5, 1, 2, 5, 10} scale both cash and shares."),
            ("bullet", "n_runs=100 paths per multiplier; baseline cash 20000, baseline shares 200."),
        ]),
        ("Constrained order generation", [
            ("formula", r"$W_{i}=k\cdot W_{0},\ \ H_{j}=k\cdot H_{0}$"),
            ("formula", r"$q_i^b\,p_i^b\leq W_i,\quad q_j^s\leq H_j$"),
        ]),
        ("Mean-reversion regression", [
            ("formula", r"$x_t=\log(P_t/P_0),\ \ r_{t+1}=\alpha+\beta\,x_t+\varepsilon_t$"),
            ("formula", r"$\beta<0\ \Rightarrow\ \mathrm{weak\ mean\ reversion}$"),
        ]),
        ("Cross-multiplier dispersion", [
            ("formula", r"$\mathrm{Disp}(k)=\mathrm{Std}_{\mathrm{paths}}(\log P_T/P_0)$"),
        ]),
        ("Hypotheses H2 / H2a", [
            ("text", "H2: tighter wealth constraints (smaller k) yield more negative beta and"),
            ("text", "         bounded price paths; H2a: cross-path dispersion shrinks as k decreases."),
        ]),
    ]


def _exp3_sections() -> list[tuple[str, list[tuple[str, str]]]]:
    return [
        ("Setup", [
            ("bullet", "Vary market depth N in {20, 50, 100, 200, 500, 1000, 5000} buyers = sellers."),
            ("bullet", "Same lognormal order rule and uniform-price auction clearing as Experiment 1."),
            ("bullet", "n_runs=100 per depth level; 20 representative paths saved per level."),
        ]),
        ("Volatility and extreme movement", [
            ("formula", r"$\sigma_r(N)=\mathrm{Std}(r_t\mid N),\quad E(N)=\max_t|r_t|$"),
        ]),
        ("Normalized order-curve roughness", [
            ("formula", r"$D_t^{\mathrm{norm}}(p)=D_t(p)/\max_p D_t(p),\ \ S_t^{\mathrm{norm}}(p)=S_t(p)/\max_p S_t(p)$"),
            ("formula", r"$R_D=\overline{|\Delta D_t^{\mathrm{norm}}|},\ \ R_S=\overline{|\Delta S_t^{\mathrm{norm}}|},\ \ R=(R_D+R_S)/2$"),
        ]),
        ("Scaling diagnostics", [
            ("formula", r"$\sigma_r(N)\propto 1/\sqrt{N},\quad \mathrm{Corr}(\sigma_r,R)>0$"),
        ]),
        ("Hypotheses H3 / H3a", [
            ("text", "H3: deeper markets reduce volatility and curve roughness; H3a: roughness and"),
            ("text", "         volatility are positively correlated across depth levels."),
        ]),
    ]


def _exp4_sections() -> list[tuple[str, list[tuple[str, str]]]]:
    return [
        ("Setup", [
            ("bullet", "Two-group design under main regime = loose (W0=20000, H0=200)."),
            ("bullet", "Baseline group: no inflow. Treatment group: c=10 cash per agent per day."),
            ("bullet", "n_runs=100 per group; 50 paths plotted; tight regime kept as robustness."),
        ]),
        ("Capital inflow update", [
            ("formula", r"$W_{i,t}=W_{i,t-1}-\mathrm{Spent}_{i,t-1}+\mathrm{Received}_{i,t-1}+c\cdot\mathbf{1}(\mathrm{treatment})$"),
            ("formula", r"$\mathrm{bid}_{i,t}=P_{t-1}\exp(\epsilon_{i,t}),\ \ \epsilon_{i,t}\sim\mathcal{N}(0,\sigma^2)$  (zero-drift, identical to baseline)"),
        ]),
        ("Random-walk channel-shift metrics", [
            ("formula", r"$\bar r=\overline{r_t},\ \ \rho_1,\ \ VR(5),\ \ \Pr(r_t>0),\ \ P_T,\ \ V_t,\ \ \mathbf{1}(V_t=0)$"),
        ]),
        ("Welch t-test and effect size", [
            ("formula", r"$t=(\bar r_{\mathrm{trt}}-\bar r_{\mathrm{base}})/\sqrt{s_{\mathrm{trt}}^2/n+s_{\mathrm{base}}^2/n}$"),
            ("formula", r"$d=(\bar r_{\mathrm{trt}}-\bar r_{\mathrm{base}})/s_{\mathrm{pooled}}$"),
        ]),
        ("Hypotheses H4 / H4a-H4d", [
            ("text", "H4: inflow shifts the random-walk channel upward (mean log return > baseline)."),
            ("text", "H4a: random-walk shape preserved (rho_1, VR(5) unchanged within tolerance)."),
            ("text", "H4b: final-price distribution shifts upward; H4c: trading volume rises, zero-trade days fall."),
            ("text", "H4d: channel shift is larger under tight regime (constraint mediation)."),
        ]),
    ]


def _formal_model_markdown() -> str:
    return """# Formal Model

## A. Agent Sets

At each day `t`, buyers and sellers are represented as finite sets:

```text
B_t = {1, ..., N_b}
S_t = {1, ..., N_s}
```

## B. Order Submission Rule

Each buyer submits `(p_i^b, q_i^b)` and each seller submits `(p_j^s, q_j^s)`.

```text
p_i^b = P_{t-1} exp(epsilon_i^b)
p_j^s = P_{t-1} exp(epsilon_j^s)
epsilon_i^b, epsilon_j^s ~ N(0, sigma^2)
q_i^b, q_j^s ~ Uniform(1, q_max)
```

## C. Budget and Inventory Constraints

Buyer cash constraint:

```text
q_i^b p_i^b <= W_i
```

Seller inventory constraint:

```text
q_j^s <= H_j
```

The ablation study switches these constraints on and off to identify their separate effects.

## D. Aggregate Demand and Supply

```text
D_t(p) = sum_i q_i^b 1(p_i^b >= p)
S_t(p) = sum_j q_j^s 1(p_j^s <= p)
```

## E. Matched Volume and Clearing Rule

```text
V_t(p) = min(D_t(p), S_t(p))
I_t(p) = |D_t(p) - S_t(p)|
```

The clearing price first maximizes `V_t(p)`, then minimizes `I_t(p)`, then uses the median candidate price if multiple prices remain.

## F. Price Evolution

```text
P_t = p_t*
r_t = log(P_t) - log(P_{t-1})
x_t = log(P_t / P_0)
```

## G. Mean Reversion Test

```text
r_{t+1} = alpha + beta x_t + epsilon_t
```

`beta < 0` indicates weak mean reversion.

## H. Volatility and Extreme Movement

```text
sigma_r = Std(r_t)
E = max_t |r_t|
```

## I. Curve Roughness

```text
D_norm(p) = D(p) / max(D(p))
S_norm(p) = S(p) / max(S(p))
R_D = mean(|Delta D_norm|)
R_S = mean(|Delta S_norm|)
R = (R_D + R_S) / 2
```

Lower roughness means smoother aggregate order curves.

## J. Random Process Interpretation

- Unlimited wealth: the order-auction mechanism produces a random-walk-like process.
- Wealth constraints: feasible order flow changes with price level, producing boundedness and weak mean reversion.
- Market depth: deeper markets produce smoother curves, lower volatility, and fewer extreme jumps.
"""
