"""Command-line entry point for market random process experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import config
from analysis.formal_model import generate_formal_model
from analysis.index import (
    generate_experiment_config_summary,
    generate_hypotheses,
    generate_output_index,
    generate_reproducibility_diagnostics,
)
from analysis.report import generate_report
from simulator.experiments import (
    ensure_output_dirs,
    run_all_experiments,
    run_cross_experiment_summary,
    run_experiment_2_ablation,
    run_experiment_3_roughness_volatility,
    run_experiment_1,
    run_experiment_2,
    run_experiment_3,
    run_experiment_4,
    run_extreme_case_analysis,
    run_gaussian_benchmark,
    run_sensitivity_analysis,
    run_volume_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run market random process simulation experiments.")
    parser.add_argument(
        "--experiment",
        choices=[
            "all",
            "full",
            "exp1",
            "exp2",
            "exp3",
            "exp4",
            "ablation",
            "sensitivity",
            "benchmark",
            "volume",
            "extreme_cases",
            "formal_model",
            "hypotheses",
            "index",
            "diagnostics",
            "cross",
            "roughness",
            "config",
        ],
        default="all",
        help="Experiment or supplemental analysis to run. Defaults to all core experiments.",
    )
    parser.add_argument("--seed", type=int, default=config.DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--n_days", type=int, default=config.DEFAULT_N_DAYS, help="Number of simulated days.")
    parser.add_argument("--n_runs", type=int, default=config.EXP3_N_RUNS, help="Number of repeated runs for supplemental analyses.")
    parser.add_argument("--output_dir", type=str, default=config.OUTPUT_DIR, help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_output_dirs(output_dir)

    if args.experiment == "all":
        run_all_experiments(seed=args.seed, n_days=args.n_days, output_dir=output_dir, n_runs=args.n_runs)
    elif args.experiment == "full":
        run_all_experiments(seed=args.seed, n_days=args.n_days, output_dir=output_dir, n_runs=args.n_runs)
        run_experiment_2_ablation(seed=args.seed, n_days=args.n_days, output_dir=output_dir, n_runs=args.n_runs)
        run_experiment_3_roughness_volatility(seed=args.seed, n_days=args.n_days, output_dir=output_dir, n_runs=args.n_runs)
        run_experiment_4(seed=args.seed, n_days=args.n_days, output_dir=output_dir, n_runs=min(args.n_runs, config.EXP4_N_RUNS))
        run_gaussian_benchmark(seed=args.seed, n_days=args.n_days, output_dir=output_dir, n_runs=args.n_runs)
        run_volume_analysis(seed=args.seed, n_days=args.n_days, output_dir=output_dir, n_runs=args.n_runs)
        run_extreme_case_analysis(seed=args.seed, n_days=args.n_days, output_dir=output_dir, n_runs=max(10, min(args.n_runs, config.EXTREME_CASE_N_RUNS)))
        run_sensitivity_analysis(seed=args.seed, n_days=args.n_days, output_dir=output_dir, n_runs=max(5, min(args.n_runs, 30)))
        run_cross_experiment_summary(seed=args.seed, n_days=args.n_days, output_dir=output_dir)
        generate_experiment_config_summary(output_dir, n_days=args.n_days, n_runs=args.n_runs, seed=args.seed)
        generate_formal_model(output_dir)
        generate_hypotheses(output_dir)
        generate_reproducibility_diagnostics(output_dir)
        generate_report(output_dir)
        generate_output_index(output_dir)
    elif args.experiment == "exp1":
        run_experiment_1(seed=args.seed, n_days=args.n_days, output_dir=output_dir)
        generate_report(output_dir)
    elif args.experiment == "exp2":
        run_experiment_2(seed=args.seed, n_days=args.n_days, output_dir=output_dir, n_runs=args.n_runs)
        generate_report(output_dir)
    elif args.experiment == "exp3":
        run_experiment_3(seed=args.seed, n_days=args.n_days, output_dir=output_dir)
        generate_report(output_dir)
    elif args.experiment == "exp4":
        run_experiment_4(seed=args.seed, n_days=args.n_days, output_dir=output_dir, n_runs=min(args.n_runs, config.EXP4_N_RUNS))
        generate_report(output_dir)
    elif args.experiment == "ablation":
        run_experiment_2_ablation(seed=args.seed, n_days=args.n_days, output_dir=output_dir, n_runs=args.n_runs)
    elif args.experiment == "roughness":
        run_experiment_3_roughness_volatility(seed=args.seed, n_days=args.n_days, output_dir=output_dir, n_runs=args.n_runs)
    elif args.experiment == "sensitivity":
        run_sensitivity_analysis(seed=args.seed, n_days=args.n_days, output_dir=output_dir, n_runs=max(5, min(args.n_runs, 30)))
    elif args.experiment == "benchmark":
        run_gaussian_benchmark(seed=args.seed, n_days=args.n_days, output_dir=output_dir, n_runs=args.n_runs)
    elif args.experiment == "volume":
        run_volume_analysis(seed=args.seed, n_days=args.n_days, output_dir=output_dir, n_runs=args.n_runs)
    elif args.experiment == "extreme_cases":
        run_extreme_case_analysis(seed=args.seed, n_days=args.n_days, output_dir=output_dir, n_runs=max(10, min(args.n_runs, config.EXTREME_CASE_N_RUNS)))
    elif args.experiment == "formal_model":
        generate_formal_model(output_dir)
    elif args.experiment == "hypotheses":
        generate_hypotheses(output_dir)
    elif args.experiment == "index":
        generate_output_index(output_dir)
    elif args.experiment == "diagnostics":
        generate_reproducibility_diagnostics(output_dir)
    elif args.experiment == "cross":
        run_cross_experiment_summary(seed=args.seed, n_days=args.n_days, output_dir=output_dir)
    elif args.experiment == "config":
        generate_experiment_config_summary(output_dir, n_days=args.n_days, n_runs=args.n_runs, seed=args.seed)
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")

    print(f"Finished {args.experiment}. Outputs saved under {output_dir}.")


if __name__ == "__main__":
    main()
