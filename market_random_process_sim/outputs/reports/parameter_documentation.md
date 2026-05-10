# Parameter Documentation

- `initial_price`: Starting market price `P_0`.
- `n_days`: Number of daily auction periods.
- `n_runs`: Number of repeated stochastic simulations.
- `n_buyers`, `n_sellers`: Number of submitted buy and sell orders in the unlimited market.
- `n_agents`: Number of agents in constrained markets.
- `price_sigma`: Standard deviation of lognormal quote noise around yesterday's price.
- `max_qty`: Upper bound of uniform order quantity draws.
- `auction_rule`: Clearing rule that maximizes matched volume, minimizes imbalance, and median tie-breaks.
- `wealth_multiplier`: Scaling factor applied to initial cash and share inventory.
- `market_depth`: Buyers = sellers in the depth experiment.
- `seed_policy`: Deterministic seed offsets used to make repeated simulations reproducible.
- `order_price_distribution`: Lognormal quote generation mechanism.
- `order_quantity_distribution`: Uniform quantity generation mechanism, optionally capped by constraints.
- `constraints`: Active budget and inventory restrictions.
