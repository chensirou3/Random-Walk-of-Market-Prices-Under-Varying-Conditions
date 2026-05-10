import numpy as np

from simulator.market import DailyAuctionMarket


def test_simulation_result_length_equals_n_days_plus_one():
    market = DailyAuctionMarket(seed=123)
    result = market.simulate_unlimited(n_days=20, n_buyers=30, n_sellers=30)

    assert len(result.prices) == 21
    assert len(result.volumes) == 21
    assert len(result.returns) == 20


def test_prices_are_always_positive():
    market = DailyAuctionMarket(seed=123)
    result = market.simulate_unlimited(n_days=50, n_buyers=20, n_sellers=20)

    assert np.all(result.prices > 0)


def test_same_seed_is_reproducible():
    market_a = DailyAuctionMarket(seed=42)
    market_b = DailyAuctionMarket(seed=42)

    result_a = market_a.simulate_unlimited(n_days=30, n_buyers=40, n_sellers=40)
    result_b = market_b.simulate_unlimited(n_days=30, n_buyers=40, n_sellers=40)

    np.testing.assert_allclose(result_a.prices, result_b.prices)
    np.testing.assert_allclose(result_a.volumes, result_b.volumes)
    np.testing.assert_allclose(result_a.returns, result_b.returns)


def test_capital_inflow_conserves_total_cash_and_shares():
    n_agents = 100
    n_days = 40
    initial_cash = 5_000.0
    initial_shares = 200.0
    inflow = 2.5
    market = DailyAuctionMarket(seed=2024)
    result = market.simulate_capital_inflow(
        n_days=n_days,
        n_agents=n_agents,
        initial_cash_per_agent=initial_cash,
        initial_shares_per_agent=initial_shares,
        inflow_per_period=inflow,
    )

    final_state = result.final_agent_state
    assert final_state is not None
    total_cash = float(final_state["cash"].sum())
    expected_cash = n_agents * initial_cash + n_agents * n_days * inflow
    assert np.isclose(total_cash, expected_cash, rtol=1e-9, atol=1e-6)

    total_shares = float(final_state["shares"].sum())
    expected_shares = n_agents * initial_shares
    assert np.isclose(total_shares, expected_shares, rtol=1e-9, atol=1e-6)


def test_capital_inflow_zero_drift_when_inflow_is_zero():
    market = DailyAuctionMarket(seed=7)
    result = market.simulate_capital_inflow(
        n_days=400,
        n_agents=200,
        initial_cash_per_agent=20_000.0,
        initial_shares_per_agent=200.0,
        inflow_per_period=0.0,
    )

    mean_log_return = float(np.mean(result.returns))
    std_log_return = float(np.std(result.returns, ddof=1))
    t_stat = mean_log_return / (std_log_return / np.sqrt(result.returns.size))
    assert abs(t_stat) < 3.0


def test_capital_inflow_daily_diagnostics_shape():
    market = DailyAuctionMarket(seed=99)
    result = market.simulate_capital_inflow(
        n_days=15,
        n_agents=50,
        initial_cash_per_agent=200.0,
        initial_shares_per_agent=200.0,
        inflow_per_period=1.0,
    )
    assert result.daily_diagnostics is not None
    diag = result.daily_diagnostics
    assert diag.shape[0] == 16
    assert {"cash_constrained_ratio", "share_constrained_ratio", "avg_cash", "zero_trade"}.issubset(
        diag.columns
    )
    assert np.all(diag["cash_constrained_ratio"].to_numpy() >= 0.0)
    assert np.all(diag["cash_constrained_ratio"].to_numpy() <= 1.0)
