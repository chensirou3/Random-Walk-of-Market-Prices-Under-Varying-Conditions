import subprocess
import sys
from pathlib import Path

import numpy as np

from simulator.market import DailyAuctionMarket


def test_same_seed_reproduces_price_volume_and_curve():
    market_a = DailyAuctionMarket(seed=777)
    market_b = DailyAuctionMarket(seed=777)

    result_a = market_a.simulate_unlimited(30, 40, 40, sample_curve_days=[10])
    result_b = market_b.simulate_unlimited(30, 40, 40, sample_curve_days=[10])

    np.testing.assert_allclose(result_a.prices, result_b.prices)
    np.testing.assert_allclose(result_a.volumes, result_b.volumes)
    np.testing.assert_allclose(result_a.auction_curves[10].to_numpy(), result_b.auction_curves[10].to_numpy())


def test_different_seed_changes_path():
    result_a = DailyAuctionMarket(seed=1).simulate_unlimited(30, 40, 40)
    result_b = DailyAuctionMarket(seed=2).simulate_unlimited(30, 40, 40)

    assert not np.allclose(result_a.prices, result_b.prices)


def test_repeated_runs_aggregate_metrics_are_stable():
    means = []
    for seed in range(10, 20):
        result = DailyAuctionMarket(seed=seed).simulate_unlimited(120, 80, 80)
        means.append(float(np.mean(result.returns)))

    assert abs(np.mean(means)) < 0.002


def test_main_all_generates_core_outputs(tmp_path):
    cmd = [
        sys.executable,
        "main.py",
        "--experiment",
        "all",
        "--seed",
        "123",
        "--n_days",
        "5",
        "--n_runs",
        "3",
        "--output_dir",
        str(tmp_path),
    ]
    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)

    assert (tmp_path / "tables" / "exp1_random_walk_diagnostics.csv").exists()
    assert (tmp_path / "tables" / "exp2_summary_by_wealth_multiplier.csv").exists()
    assert (tmp_path / "tables" / "exp3_metrics.csv").exists()
    assert (tmp_path / "reports" / "market_random_process_report.md").exists()
