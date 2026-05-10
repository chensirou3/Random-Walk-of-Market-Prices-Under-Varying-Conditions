"""Daily auction market simulation loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from simulator.auction import clear_auction_arrays, clear_auction_price_volume


@dataclass
class MarketRunResult:
    """Time series and optional auction curves from a simulation run."""

    prices: np.ndarray
    volumes: np.ndarray
    returns: np.ndarray
    auction_curves: dict[int, pd.DataFrame] = field(default_factory=dict)
    metadata: dict[str, float | int | str] = field(default_factory=dict)
    final_agent_state: pd.DataFrame | None = None
    daily_diagnostics: pd.DataFrame | None = None

    def to_dataframe(self) -> pd.DataFrame:
        """Return a day-indexed DataFrame for saving or plotting."""
        returns_with_start = np.empty_like(self.prices, dtype=float)
        returns_with_start[0] = np.nan
        returns_with_start[1:] = self.returns
        return pd.DataFrame(
            {
                "day": np.arange(self.prices.size),
                "price": self.prices,
                "volume": self.volumes,
                "return": returns_with_start,
            }
        )

    def save_csv(self, path: str | Path) -> None:
        """Save price, volume, and return series to CSV."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)


class DailyAuctionMarket:
    """A daily one-price auction market.

    The class supports the first two research settings:
    unlimited random order flow and wealth/inventory constrained agents. Future
    versions can add sentiment shocks, adaptive agents, continuous double
    auction matching, or a full limit order book without changing experiment
    orchestration code.
    """

    def __init__(
        self,
        initial_price: float = 100.0,
        price_sigma: float = 0.03,
        max_qty: float = 100.0,
        seed: int | None = None,
    ) -> None:
        if initial_price <= 0:
            raise ValueError("initial_price must be positive.")
        if price_sigma < 0:
            raise ValueError("price_sigma cannot be negative.")
        if max_qty <= 0:
            raise ValueError("max_qty must be positive.")

        self.initial_price = float(initial_price)
        self.price_sigma = float(price_sigma)
        self.max_qty = float(max_qty)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def simulate_unlimited(
        self,
        n_days: int,
        n_buyers: int,
        n_sellers: int,
        sample_curve_days: Iterable[int] | None = None,
    ) -> MarketRunResult:
        """Simulate an unlimited-wealth random auction market."""
        if n_days < 1:
            raise ValueError("n_days must be at least 1.")
        sample_days = set(sample_curve_days or [])

        prices = np.empty(n_days + 1, dtype=float)
        volumes = np.zeros(n_days + 1, dtype=float)
        prices[0] = self.initial_price
        auction_curves: dict[int, pd.DataFrame] = {}

        for day in range(1, n_days + 1):
            previous_price = prices[day - 1]
            bid_prices = previous_price * np.exp(self.rng.normal(0.0, self.price_sigma, n_buyers))
            ask_prices = previous_price * np.exp(self.rng.normal(0.0, self.price_sigma, n_sellers))
            bid_qtys = self.rng.uniform(1.0, self.max_qty, n_buyers)
            ask_qtys = self.rng.uniform(1.0, self.max_qty, n_sellers)

            if day in sample_days:
                result = clear_auction_arrays(
                    bid_prices=bid_prices,
                    bid_qtys=bid_qtys,
                    ask_prices=ask_prices,
                    ask_qtys=ask_qtys,
                    price_hint=previous_price,
                    return_curve=True,
                )
                clearing_price = result.clearing_price
                clearing_volume = result.clearing_volume
                auction_curves[day] = result.auction_curve
            else:
                clearing_price, clearing_volume = clear_auction_price_volume(
                    bid_prices=bid_prices,
                    bid_qtys=bid_qtys,
                    ask_prices=ask_prices,
                    ask_qtys=ask_qtys,
                    price_hint=previous_price,
                )
            prices[day] = self._valid_next_price(clearing_price, previous_price)
            volumes[day] = clearing_volume

        returns = np.diff(np.log(prices))
        return MarketRunResult(
            prices=prices,
            volumes=volumes,
            returns=returns,
            auction_curves=auction_curves,
            metadata={
                "market_type": "unlimited",
                "seed": self.seed if self.seed is not None else -1,
                "n_days": n_days,
                "n_buyers": n_buyers,
                "n_sellers": n_sellers,
                "price_sigma": self.price_sigma,
                "max_qty": self.max_qty,
            },
        )

    def simulate_wealth_constrained(
        self,
        n_days: int,
        n_agents: int,
        initial_cash_per_agent: float,
        initial_shares_per_agent: float,
        participation_rate: float = 0.8,
        sample_curve_days: Iterable[int] | None = None,
        buyer_cash_constraint: bool = True,
        seller_inventory_constraint: bool = True,
    ) -> MarketRunResult:
        """Simulate a random auction market with optional cash/inventory constraints."""
        if n_days < 1:
            raise ValueError("n_days must be at least 1.")
        if n_agents < 2:
            raise ValueError("n_agents must be at least 2.")
        if not 0 < participation_rate <= 1:
            raise ValueError("participation_rate must be in (0, 1].")

        sample_days = set(sample_curve_days or [])
        prices = np.empty(n_days + 1, dtype=float)
        volumes = np.zeros(n_days + 1, dtype=float)
        prices[0] = self.initial_price
        auction_curves: dict[int, pd.DataFrame] = {}

        cash = np.full(n_agents, float(initial_cash_per_agent), dtype=float)
        shares = np.full(n_agents, float(initial_shares_per_agent), dtype=float)
        agent_ids = np.arange(n_agents)

        for day in range(1, n_days + 1):
            previous_price = prices[day - 1]
            active_mask = self.rng.random(n_agents) < participation_rate
            side_draw = self.rng.random(n_agents)
            buy_mask = active_mask & (side_draw < 0.5)
            sell_mask = active_mask & ~buy_mask

            buy_ids = agent_ids[buy_mask]
            sell_ids = agent_ids[sell_mask]

            bid_prices = previous_price * np.exp(self.rng.normal(0.0, self.price_sigma, buy_ids.size))
            ask_prices = previous_price * np.exp(self.rng.normal(0.0, self.price_sigma, sell_ids.size))
            desired_buy_qty = self.rng.uniform(1.0, self.max_qty, buy_ids.size)
            desired_sell_qty = self.rng.uniform(1.0, self.max_qty, sell_ids.size)

            if buyer_cash_constraint:
                max_affordable_qty = np.divide(
                    cash[buy_ids],
                    bid_prices,
                    out=np.zeros_like(bid_prices),
                    where=bid_prices > 0,
                )
                bid_qtys = np.minimum(desired_buy_qty, max_affordable_qty)
            else:
                bid_qtys = desired_buy_qty

            if seller_inventory_constraint:
                ask_qtys = np.minimum(desired_sell_qty, shares[sell_ids])
            else:
                ask_qtys = desired_sell_qty

            buy_valid = bid_qtys > 1e-12
            sell_valid = ask_qtys > 1e-12
            buy_ids = buy_ids[buy_valid]
            sell_ids = sell_ids[sell_valid]
            bid_prices = bid_prices[buy_valid]
            ask_prices = ask_prices[sell_valid]
            bid_qtys = bid_qtys[buy_valid]
            ask_qtys = ask_qtys[sell_valid]

            if day in sample_days:
                result = clear_auction_arrays(
                    bid_prices=bid_prices,
                    bid_qtys=bid_qtys,
                    ask_prices=ask_prices,
                    ask_qtys=ask_qtys,
                    price_hint=previous_price,
                    return_curve=True,
                )
                clearing_price_raw = result.clearing_price
                clearing_volume = result.clearing_volume
                auction_curves[day] = result.auction_curve
            else:
                clearing_price_raw, clearing_volume = clear_auction_price_volume(
                    bid_prices=bid_prices,
                    bid_qtys=bid_qtys,
                    ask_prices=ask_prices,
                    ask_qtys=ask_qtys,
                    price_hint=previous_price,
                )

            clearing_price = self._valid_next_price(clearing_price_raw, previous_price)
            prices[day] = clearing_price
            volumes[day] = clearing_volume

            if clearing_volume > 0 and bid_qtys.size and ask_qtys.size:
                self._allocate_pro_rata(
                    clearing_price=clearing_price,
                    clearing_volume=clearing_volume,
                    bid_prices=bid_prices,
                    bid_qtys=bid_qtys,
                    buy_ids=buy_ids,
                    ask_prices=ask_prices,
                    ask_qtys=ask_qtys,
                    sell_ids=sell_ids,
                    cash=cash,
                    shares=shares,
                    buyer_cash_constraint=buyer_cash_constraint,
                    seller_inventory_constraint=seller_inventory_constraint,
                )

        returns = np.diff(np.log(prices))
        final_state = pd.DataFrame({"agent_id": agent_ids, "cash": cash, "shares": shares})
        return MarketRunResult(
            prices=prices,
            volumes=volumes,
            returns=returns,
            auction_curves=auction_curves,
            metadata={
                "market_type": "wealth_constrained",
                "seed": self.seed if self.seed is not None else -1,
                "n_days": n_days,
                "n_agents": n_agents,
                "price_sigma": self.price_sigma,
                "max_qty": self.max_qty,
                "participation_rate": participation_rate,
                "buyer_cash_constraint": str(buyer_cash_constraint),
                "seller_inventory_constraint": str(seller_inventory_constraint),
            },
            final_agent_state=final_state,
        )

    def simulate_capital_inflow(
        self,
        n_days: int,
        n_agents: int,
        initial_cash_per_agent: float,
        initial_shares_per_agent: float,
        inflow_per_period: float,
        participation_rate: float = 0.8,
        sample_curve_days: Iterable[int] | None = None,
        buyer_cash_constraint: bool = True,
        seller_inventory_constraint: bool = True,
    ) -> MarketRunResult:
        """Simulate an auction market with daily per-agent cash inflow.

        The bid and ask price equations are unchanged from the wealth-constrained
        market. The only difference is that each agent receives
        ``inflow_per_period`` in cash before generating orders each day. Any
        observed price drift therefore comes from auction clearing on a feasible
        order set that has been expanded by the cash injection, not from a drift
        term in the bidding rule.
        """
        if n_days < 1:
            raise ValueError("n_days must be at least 1.")
        if n_agents < 2:
            raise ValueError("n_agents must be at least 2.")
        if not 0 < participation_rate <= 1:
            raise ValueError("participation_rate must be in (0, 1].")
        if inflow_per_period < 0:
            raise ValueError("inflow_per_period cannot be negative.")

        sample_days = set(sample_curve_days or [])
        prices = np.empty(n_days + 1, dtype=float)
        volumes = np.zeros(n_days + 1, dtype=float)
        prices[0] = self.initial_price
        auction_curves: dict[int, pd.DataFrame] = {}

        cash = np.full(n_agents, float(initial_cash_per_agent), dtype=float)
        shares = np.full(n_agents, float(initial_shares_per_agent), dtype=float)
        agent_ids = np.arange(n_agents)

        cash_constrained_ratio = np.zeros(n_days + 1, dtype=float)
        share_constrained_ratio = np.zeros(n_days + 1, dtype=float)
        avg_cash_per_day = np.zeros(n_days + 1, dtype=float)
        zero_trade_flags = np.zeros(n_days + 1, dtype=float)
        avg_cash_per_day[0] = float(np.mean(cash))

        for day in range(1, n_days + 1):
            if inflow_per_period > 0:
                cash += inflow_per_period

            previous_price = prices[day - 1]
            active_mask = self.rng.random(n_agents) < participation_rate
            side_draw = self.rng.random(n_agents)
            buy_mask = active_mask & (side_draw < 0.5)
            sell_mask = active_mask & ~buy_mask

            buy_ids = agent_ids[buy_mask]
            sell_ids = agent_ids[sell_mask]

            bid_prices = previous_price * np.exp(self.rng.normal(0.0, self.price_sigma, buy_ids.size))
            ask_prices = previous_price * np.exp(self.rng.normal(0.0, self.price_sigma, sell_ids.size))
            desired_buy_qty = self.rng.uniform(1.0, self.max_qty, buy_ids.size)
            desired_sell_qty = self.rng.uniform(1.0, self.max_qty, sell_ids.size)

            if buyer_cash_constraint:
                max_affordable_qty = np.divide(
                    cash[buy_ids],
                    bid_prices,
                    out=np.zeros_like(bid_prices),
                    where=bid_prices > 0,
                )
                bid_qtys = np.minimum(desired_buy_qty, max_affordable_qty)
                buy_constrained = desired_buy_qty > max_affordable_qty + 1e-12
            else:
                bid_qtys = desired_buy_qty
                buy_constrained = np.zeros(buy_ids.size, dtype=bool)

            if seller_inventory_constraint:
                ask_qtys = np.minimum(desired_sell_qty, shares[sell_ids])
                sell_constrained = desired_sell_qty > shares[sell_ids] + 1e-12
            else:
                ask_qtys = desired_sell_qty
                sell_constrained = np.zeros(sell_ids.size, dtype=bool)

            cash_constrained_ratio[day] = (
                float(np.mean(buy_constrained)) if buy_constrained.size else 0.0
            )
            share_constrained_ratio[day] = (
                float(np.mean(sell_constrained)) if sell_constrained.size else 0.0
            )

            buy_valid = bid_qtys > 1e-12
            sell_valid = ask_qtys > 1e-12
            buy_ids = buy_ids[buy_valid]
            sell_ids = sell_ids[sell_valid]
            bid_prices = bid_prices[buy_valid]
            ask_prices = ask_prices[sell_valid]
            bid_qtys = bid_qtys[buy_valid]
            ask_qtys = ask_qtys[sell_valid]

            if day in sample_days:
                result = clear_auction_arrays(
                    bid_prices=bid_prices,
                    bid_qtys=bid_qtys,
                    ask_prices=ask_prices,
                    ask_qtys=ask_qtys,
                    price_hint=previous_price,
                    return_curve=True,
                )
                clearing_price_raw = result.clearing_price
                clearing_volume = result.clearing_volume
                auction_curves[day] = result.auction_curve
            else:
                clearing_price_raw, clearing_volume = clear_auction_price_volume(
                    bid_prices=bid_prices,
                    bid_qtys=bid_qtys,
                    ask_prices=ask_prices,
                    ask_qtys=ask_qtys,
                    price_hint=previous_price,
                )

            clearing_price = self._valid_next_price(clearing_price_raw, previous_price)
            prices[day] = clearing_price
            volumes[day] = clearing_volume
            zero_trade_flags[day] = 1.0 if clearing_volume <= 1e-12 else 0.0

            if clearing_volume > 0 and bid_qtys.size and ask_qtys.size:
                self._allocate_pro_rata(
                    clearing_price=clearing_price,
                    clearing_volume=clearing_volume,
                    bid_prices=bid_prices,
                    bid_qtys=bid_qtys,
                    buy_ids=buy_ids,
                    ask_prices=ask_prices,
                    ask_qtys=ask_qtys,
                    sell_ids=sell_ids,
                    cash=cash,
                    shares=shares,
                    buyer_cash_constraint=buyer_cash_constraint,
                    seller_inventory_constraint=seller_inventory_constraint,
                )

            avg_cash_per_day[day] = float(np.mean(cash))

        returns = np.diff(np.log(prices))
        final_state = pd.DataFrame({"agent_id": agent_ids, "cash": cash, "shares": shares})
        daily_diagnostics = pd.DataFrame(
            {
                "day": np.arange(n_days + 1),
                "cash_constrained_ratio": cash_constrained_ratio,
                "share_constrained_ratio": share_constrained_ratio,
                "avg_cash": avg_cash_per_day,
                "zero_trade": zero_trade_flags,
            }
        )
        return MarketRunResult(
            prices=prices,
            volumes=volumes,
            returns=returns,
            auction_curves=auction_curves,
            metadata={
                "market_type": "capital_inflow",
                "seed": self.seed if self.seed is not None else -1,
                "n_days": n_days,
                "n_agents": n_agents,
                "price_sigma": self.price_sigma,
                "max_qty": self.max_qty,
                "participation_rate": participation_rate,
                "inflow_per_period": float(inflow_per_period),
                "initial_cash_per_agent": float(initial_cash_per_agent),
                "initial_shares_per_agent": float(initial_shares_per_agent),
                "buyer_cash_constraint": str(buyer_cash_constraint),
                "seller_inventory_constraint": str(seller_inventory_constraint),
            },
            final_agent_state=final_state,
            daily_diagnostics=daily_diagnostics,
        )

    @staticmethod
    def _valid_next_price(clearing_price: float, fallback_price: float) -> float:
        if np.isfinite(clearing_price) and clearing_price > 0:
            return float(clearing_price)
        return float(fallback_price)

    @staticmethod
    def _allocate_pro_rata(
        clearing_price: float,
        clearing_volume: float,
        bid_prices: np.ndarray,
        bid_qtys: np.ndarray,
        buy_ids: np.ndarray,
        ask_prices: np.ndarray,
        ask_qtys: np.ndarray,
        sell_ids: np.ndarray,
        cash: np.ndarray,
        shares: np.ndarray,
        buyer_cash_constraint: bool = True,
        seller_inventory_constraint: bool = True,
    ) -> None:
        """Allocate aggregate auction volume to executable orders pro rata."""
        executable_buys = bid_prices >= clearing_price
        executable_sells = ask_prices <= clearing_price
        total_demand = float(np.sum(bid_qtys[executable_buys]))
        total_supply = float(np.sum(ask_qtys[executable_sells]))

        if total_demand <= 0 or total_supply <= 0 or clearing_volume <= 0:
            return

        buy_fill_ratio = min(clearing_volume / total_demand, 1.0)
        sell_fill_ratio = min(clearing_volume / total_supply, 1.0)

        buy_exec_qty = bid_qtys[executable_buys] * buy_fill_ratio
        sell_exec_qty = ask_qtys[executable_sells] * sell_fill_ratio
        buy_exec_ids = buy_ids[executable_buys]
        sell_exec_ids = sell_ids[executable_sells]

        if buyer_cash_constraint:
            np.add.at(cash, buy_exec_ids, -buy_exec_qty * clearing_price)
        np.add.at(shares, buy_exec_ids, buy_exec_qty)
        np.add.at(cash, sell_exec_ids, sell_exec_qty * clearing_price)
        if seller_inventory_constraint:
            np.add.at(shares, sell_exec_ids, -sell_exec_qty)

        if buyer_cash_constraint:
            np.maximum(cash, 0.0, out=cash)
        if seller_inventory_constraint:
            np.maximum(shares, 0.0, out=shares)
