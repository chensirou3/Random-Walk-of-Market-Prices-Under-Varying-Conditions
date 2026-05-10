# Sensitivity Analysis Summary

The sensitivity checks vary `price_sigma`, `max_qty`, and `n_days` while tracking the core conclusions.

Most robust conclusions:
- Experiment 1 remains close to zero-drift and low-autocorrelation across tested settings.
- Experiment 3 preserves the ordering that high depth has lower volatility and lower curve roughness than low depth.
- Experiment 2 mean-reversion beta is generally negative under the baseline constrained mechanism.

More sensitive quantities:
- Absolute volatility scales with `price_sigma`.
- Extreme moves are sensitive to both `price_sigma` and simulation horizon.
- Wealth-constraint dispersion measures can vary with horizon and quantity scale, so later calibration should report robustness ranges.

Generated tables:
- `sensitivity_sigma_summary.csv`
- `sensitivity_max_qty_summary.csv`
- `sensitivity_n_days_summary.csv`
