# Experiment 4 Findings: Capital Inflow and the Random-Walk Channel

Main regime: `loose` (initial cash per agent = 20000). Treatment inflow = `10` per buyer per period. Each group uses `100` independent runs with no drift term in the bid equation.

## H4a — Treatment mean log return exceeds baseline
- Baseline mean log return: `-7.92214e-07`; treatment mean log return: `0.000357492`; difference: `+0.000358284`.
- Welch's two-sample t-test: t = `39.963`, p = `0`, Cohen's d = `5.652`.
- **Verdict:** H4a is supported.

## H4b — Treatment mean price path lies above baseline
- Mean final price: baseline `100` vs treatment `119.6` (difference `+19.63`, Welch p = `0`).
- **Verdict:** H4b is supported.

## H4c — Random-walk features are preserved under the treatment
- Lag-1 autocorrelation: baseline `0.0145` vs treatment `0.0090` (target ≈ 0).
- Variance ratio (lag 5): baseline `1.0345` vs treatment `1.0270` (target ≈ 1).
- Share of positive-return periods: baseline `0.5008` vs treatment `0.5458` (≈ 0.5 implies symmetric increments).
- **Verdict:** H4c is supported — the increments remain close to a random walk even when the channel shifts upward.

## H4d — Constraint mediation (robustness across cash regimes)
- `loose` (baseline cash-constrained ratio `0.173`): treatment − baseline mean log return = `+0.000358284`.
- `tight` (baseline cash-constrained ratio `0.404`): treatment − baseline mean log return = `+0.00582838`.
- **Interpretation:** if the treatment minus baseline gap is larger when the baseline cash-constrained ratio is higher, the inflow effect is mediated by binding cash constraints rather than acting as a direct price trend.

## Mechanism note

- The bid equation is `bid = P[t-1] * exp(epsilon)` with `epsilon ~ N(0, sigma^2)` in both groups; no drift term is added.
- Any upward shift therefore comes from the auction clearing — extra buyer cash marginally increases the volume-maximising clearing price each period — and not from an imposed trend.
