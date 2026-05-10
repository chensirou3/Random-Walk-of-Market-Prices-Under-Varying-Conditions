"""Agent state and order generation helpers for constrained markets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulator.orders import Order


@dataclass
class Agent:
    """A minimal trading agent with cash and share inventory."""

    agent_id: int
    cash: float
    shares: float

    def affordable_qty(self, price: float) -> float:
        """Return the maximum number of shares the agent can buy at price."""
        if price <= 0:
            return 0.0
        return max(self.cash / price, 0.0)

    def sellable_qty(self) -> float:
        """Return the maximum number of shares the agent can sell."""
        return max(self.shares, 0.0)


def initialize_agents(
    n_agents: int,
    initial_cash_per_agent: float,
    initial_shares_per_agent: float,
) -> list[Agent]:
    """Create homogeneous agents for the first constrained-market version.

    The simulator keeps this helper intentionally small so future versions can
    replace it with heterogeneous beliefs, risk preferences, or adaptive rules.
    """
    return [
        Agent(
            agent_id=i,
            cash=float(initial_cash_per_agent),
            shares=float(initial_shares_per_agent),
        )
        for i in range(n_agents)
    ]


def generate_agent_order(
    agent: Agent,
    side: str,
    previous_price: float,
    price_sigma: float,
    max_qty: float,
    day: int,
    rng: np.random.Generator,
) -> Order | None:
    """Generate one constrained buy or sell order for an agent.

    This object-oriented helper is useful for tests and future extensions. The
    production constrained-market loop uses a vectorized equivalent for speed.
    """
    order_price = float(previous_price * np.exp(rng.normal(0.0, price_sigma)))
    desired_qty = float(rng.uniform(1.0, max_qty))

    if side == "buy":
        qty = min(desired_qty, agent.affordable_qty(order_price))
    elif side == "sell":
        qty = min(desired_qty, agent.sellable_qty())
    else:
        raise ValueError("side must be 'buy' or 'sell'.")

    if qty <= 0:
        return None

    return Order(side=side, price=order_price, qty=qty, agent_id=agent.agent_id, day=day)
