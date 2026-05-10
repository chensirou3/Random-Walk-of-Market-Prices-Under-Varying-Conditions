# Market Random Process Simulation Output Index

Project: The Emergence of Random Market Processes from Random Orders and Auction Rules.

Core mechanism: random individual orders -> auction clearing rule -> emergent market-level random process.

## Recommended Figures for Final Report

- `figures/exp1_multiple_paths.png`: Shows random-walk-like diffusion from auction-generated prices.
- `figures/exp1_auction_vs_gaussian_return_distribution.png`: Compares endogenous auction returns with an exogenous Gaussian benchmark.
- `figures/exp2_mean_beta_ci_by_wealth_multiplier.png`: Shows weak mean reversion under wealth constraints.
- `figures/exp2_ablation_mean_beta_ci.png`: Decomposes cash and inventory constraint mechanisms.
- `figures/exp3_volatility_vs_depth.png`: Shows depth scaling of volatility.
- `figures/exp3_curve_roughness_vs_depth.png`: Shows that deeper markets produce smoother curves.
- `figures/exp3_roughness_vs_volatility.png`: Connects rough order curves to return volatility.
- `figures/cross_experiment_process_comparison.png`: Summarizes process differences across market rules.

## Recommended Appendix Figures

- Extreme case plots.
- Sensitivity figures.
- Volume analysis plots.
- Formula figures.

## File Purposes

- `diagnostics/reproducibility_check.json`: Supporting output file for the simulation study.
- `diagnostics/reproducibility_check.md`: Supporting output file for the simulation study.
- `figures/cross_experiment_process_comparison.png`: Supporting output file for the simulation study.
- `figures/exp1_abs_return_vs_volume.png`: Supporting output file for the simulation study.
- `figures/exp1_auction_vs_gaussian_diagnostics.png`: Supporting output file for the simulation study.
- `figures/exp1_auction_vs_gaussian_multiple_paths.png`: Supporting output file for the simulation study.
- `figures/exp1_auction_vs_gaussian_return_distribution.png`: Supporting output file for the simulation study.
- `figures/exp1_daily_volume_series.png`: Supporting output file for the simulation study.
- `figures/exp1_multiple_paths.png`: Supporting output file for the simulation study.
- `figures/exp1_price_path.png`: Supporting output file for the simulation study.
- `figures/exp1_random_walk_diagnostics_table.png`: Supporting output file for the simulation study.
- `figures/exp1_return_distribution.png`: Supporting output file for the simulation study.
- `figures/exp1_return_series.png`: Supporting output file for the simulation study.
- `figures/exp1_sample_auction_curve.png`: Supporting output file for the simulation study.
- `figures/exp1_volume_distribution.png`: Supporting output file for the simulation study.
- `figures/exp2_ablation_beta_distribution.png`: Supporting output file for the simulation study.
- `figures/exp2_ablation_log_deviation_variance.png`: Supporting output file for the simulation study.
- `figures/exp2_ablation_mean_beta_ci.png`: Compares mean-reversion beta across cash, inventory, and combined constraints.
- `figures/exp2_ablation_multiple_paths.png`: Supporting output file for the simulation study.
- `figures/exp2_beta_distribution_by_wealth_multiplier.png`: Supporting output file for the simulation study.
- `figures/exp2_log_deviation.png`: Supporting output file for the simulation study.
- `figures/exp2_log_deviation_variance_by_wealth_multiplier.png`: Supporting output file for the simulation study.
- `figures/exp2_mean_beta_ci_by_wealth_multiplier.png`: Shows negative mean-reversion coefficients across repeated wealth-constrained simulations.
- `figures/exp2_mean_reversion_scatter.png`: Supporting output file for the simulation study.
- `figures/exp2_multiple_paths_by_wealth_constraint.png`: Supporting output file for the simulation study.
- `figures/exp2_price_distribution_comparison.png`: Supporting output file for the simulation study.
- `figures/exp2_price_path_comparison.png`: Supporting output file for the simulation study.
- `figures/exp2_wealth_multiplier_mean_reversion.png`: Supporting output file for the simulation study.
- `figures/exp2_wealth_multiplier_paths.png`: Supporting output file for the simulation study.
- `figures/exp2_wealth_multiplier_volatility.png`: Supporting output file for the simulation study.
- `figures/exp2_wealth_multiplier_vs_volume.png`: Supporting output file for the simulation study.
- `figures/exp3_auction_curve_high_depth.png`: Supporting output file for the simulation study.
- `figures/exp3_auction_curve_low_depth.png`: Supporting output file for the simulation study.
- `figures/exp3_curve_roughness_vs_depth.png`: Shows that higher market depth produces smoother order curves.
- `figures/exp3_depth_roughness_volatility_chain.png`: Supporting output file for the simulation study.
- `figures/exp3_extreme_move_vs_depth.png`: Supporting output file for the simulation study.
- `figures/exp3_market_depth_vs_volume.png`: Supporting output file for the simulation study.
- `figures/exp3_multiple_paths_by_market_depth.png`: Supporting output file for the simulation study.
- `figures/exp3_price_paths_by_depth.png`: Supporting output file for the simulation study.
- `figures/exp3_return_distribution_by_depth.png`: Supporting output file for the simulation study.
- `figures/exp3_roughness_vs_volatility.png`: Shows the mechanism link from curve roughness to volatility.
- `figures/exp3_volatility_vs_depth.png`: Supporting output file for the simulation study.
- `figures/exp3_volume_vs_abs_return.png`: Supporting output file for the simulation study.
- `figures/extreme_case_exp1_extreme_down_auction_curve.png`: Supporting output file for the simulation study.
- `figures/extreme_case_exp1_extreme_down_event_window.png`: Supporting output file for the simulation study.
- `figures/extreme_case_exp1_extreme_down_price_path.png`: Supporting output file for the simulation study.
- `figures/extreme_case_exp1_extreme_up_auction_curve.png`: Supporting output file for the simulation study.
- `figures/extreme_case_exp1_extreme_up_event_window.png`: Supporting output file for the simulation study.
- `figures/extreme_case_exp1_extreme_up_price_path.png`: Supporting output file for the simulation study.
- `figures/extreme_case_exp1_typical_auction_curve.png`: Supporting output file for the simulation study.
- `figures/extreme_case_exp1_typical_event_window.png`: Supporting output file for the simulation study.
- `figures/extreme_case_exp1_typical_price_path.png`: Supporting output file for the simulation study.
- `figures/extreme_case_exp3_high_depth_stable_auction_curve.png`: Supporting output file for the simulation study.
- `figures/extreme_case_exp3_high_depth_stable_event_window.png`: Supporting output file for the simulation study.
- `figures/extreme_case_exp3_high_depth_stable_price_path.png`: Supporting output file for the simulation study.
- `figures/extreme_case_exp3_low_depth_down_auction_curve.png`: Supporting output file for the simulation study.
- `figures/extreme_case_exp3_low_depth_down_event_window.png`: Supporting output file for the simulation study.
- `figures/extreme_case_exp3_low_depth_down_price_path.png`: Supporting output file for the simulation study.
- `figures/extreme_case_exp3_low_depth_up_auction_curve.png`: Supporting output file for the simulation study.
- `figures/extreme_case_exp3_low_depth_up_event_window.png`: Supporting output file for the simulation study.
- `figures/extreme_case_exp3_low_depth_up_price_path.png`: Supporting output file for the simulation study.
- `figures/sensitivity_max_qty_core_metrics.png`: Supporting output file for the simulation study.
- `figures/sensitivity_n_days_core_metrics.png`: Supporting output file for the simulation study.
- `figures/sensitivity_sigma_core_metrics.png`: Supporting output file for the simulation study.
- `formula_figures/formal_model_auction_clearing.png`: Supporting output file for the simulation study.
- `formula_figures/formal_model_order_submission.png`: Supporting output file for the simulation study.
- `formula_figures/formal_model_price_dynamics_metrics.png`: Supporting output file for the simulation study.
- `formula_figures/formal_model_roughness_metric.png`: Supporting output file for the simulation study.
- `index.md`: Supporting output file for the simulation study.
- `reports/exp2_ablation_findings.md`: Supporting output file for the simulation study.
- `reports/formal_model.md`: Provides mathematical model definitions and diagnostics.
- `reports/hypotheses.md`: Defines hypotheses and maps them to metrics and outputs.
- `reports/market_random_process_report.md`: Supporting output file for the simulation study.
- `reports/parameter_documentation.md`: Supporting output file for the simulation study.
- `reports/sensitivity_analysis_summary.md`: Supporting output file for the simulation study.
- `simulation_results/exp1_multiple_paths.csv`: Supporting output file for the simulation study.
- `simulation_results/exp1_single_path.csv`: Supporting output file for the simulation study.
- `simulation_results/exp2_constrained_baseline.csv`: Supporting output file for the simulation study.
- `simulation_results/exp2_final_agent_state_multiplier_1.csv`: Supporting output file for the simulation study.
- `simulation_results/exp2_repeated_paths.csv`: Supporting output file for the simulation study.
- `simulation_results/exp2_unlimited_baseline.csv`: Supporting output file for the simulation study.
- `simulation_results/exp2_wealth_paths.csv`: Supporting output file for the simulation study.
- `simulation_results/exp3_representative_depth_100.csv`: Supporting output file for the simulation study.
- `simulation_results/exp3_representative_depth_1000.csv`: Supporting output file for the simulation study.
- `simulation_results/exp3_representative_depth_20.csv`: Supporting output file for the simulation study.
- `simulation_results/exp3_representative_depth_200.csv`: Supporting output file for the simulation study.
- `simulation_results/exp3_representative_depth_50.csv`: Supporting output file for the simulation study.
- `simulation_results/exp3_representative_depth_500.csv`: Supporting output file for the simulation study.
- `simulation_results/exp3_representative_depth_5000.csv`: Supporting output file for the simulation study.
- `tables/cross_experiment_summary.csv`: Compares process metrics across market mechanisms.
- `tables/cross_experiment_summary.md`: Supporting output file for the simulation study.
- `tables/exp1_gaussian_benchmark_comparison.csv`: Supporting output file for the simulation study.
- `tables/exp1_metrics.csv`: Supporting output file for the simulation study.
- `tables/exp1_random_walk_diagnostics.csv`: Supporting output file for the simulation study.
- `tables/exp2_ablation_run_metrics.csv`: Supporting output file for the simulation study.
- `tables/exp2_ablation_summary.csv`: Supporting output file for the simulation study.
- `tables/exp2_metrics.csv`: Supporting output file for the simulation study.
- `tables/exp2_repeated_run_metrics.csv`: Supporting output file for the simulation study.
- `tables/exp2_summary_by_wealth_multiplier.csv`: Supporting output file for the simulation study.
- `tables/exp3_metrics.csv`: Supporting output file for the simulation study.
- `tables/exp3_roughness_volatility_runs.csv`: Supporting output file for the simulation study.
- `tables/exp3_roughness_volatility_summary.csv`: Supporting output file for the simulation study.
- `tables/exp3_run_metrics.csv`: Supporting output file for the simulation study.
- `tables/experiment_config_summary.csv`: Records reproducible experiment parameters.
- `tables/experiment_config_summary.md`: Supporting output file for the simulation study.
- `tables/extreme_path_case_summary.csv`: Supporting output file for the simulation study.
- `tables/hypotheses_mapping.csv`: Supporting output file for the simulation study.
- `tables/sensitivity_max_qty_summary.csv`: Supporting output file for the simulation study.
- `tables/sensitivity_n_days_summary.csv`: Supporting output file for the simulation study.
- `tables/sensitivity_sigma_summary.csv`: Supporting output file for the simulation study.
- `tables/volume_analysis_summary.csv`: Supporting output file for the simulation study.

## How to Reproduce

```bash
python main.py --experiment full --seed 42 --n_days 500 --n_runs 100 --output_dir outputs
```