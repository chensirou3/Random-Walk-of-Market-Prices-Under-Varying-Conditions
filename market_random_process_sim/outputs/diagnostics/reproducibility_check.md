# Reproducibility Check

Seed policy: All stochastic components use numpy.random.default_rng(seed) with deterministic offsets for paths, runs, depth levels, and constraint types.

Deterministic outputs:
- same seed -> same price path
- same seed -> same volume series
- same seed -> same selected auction curves
- same command and same dependency versions -> same generated tables

Statistically stable outputs:
- repeated-run mean beta
- market-depth average volatility
- curve roughness averages
- Gaussian benchmark summary statistics

Reproduce all outputs:

```bash
python main.py --experiment full --seed 42 --n_days 500 --n_runs 100 --output_dir outputs
```
