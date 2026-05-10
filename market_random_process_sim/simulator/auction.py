"""Daily auction clearing logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from simulator.orders import Order


@dataclass
class AuctionResult:
    """Result of one daily auction clearing calculation."""

    clearing_price: float
    clearing_volume: float
    demand_curve: pd.DataFrame
    supply_curve: pd.DataFrame
    auction_curve: pd.DataFrame


def _as_float_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Auction inputs must be one-dimensional.")
    return arr


def clear_auction(orders: Sequence[Order], price_hint: float | None = None) -> AuctionResult:
    """Clear an auction from a sequence of buy and sell orders.

    Candidate prices are all submitted order prices. The clearing rule is:
    maximize matched volume, then minimize imbalance, then take the median of
    remaining candidate prices.
    """
    buy_prices = [o.price for o in orders if o.side == "buy" and o.qty > 0]
    buy_qtys = [o.qty for o in orders if o.side == "buy" and o.qty > 0]
    sell_prices = [o.price for o in orders if o.side == "sell" and o.qty > 0]
    sell_qtys = [o.qty for o in orders if o.side == "sell" and o.qty > 0]

    return clear_auction_arrays(
        bid_prices=buy_prices,
        bid_qtys=buy_qtys,
        ask_prices=sell_prices,
        ask_qtys=sell_qtys,
        price_hint=price_hint,
        return_curve=True,
    )


def clear_auction_arrays(
    bid_prices: Sequence[float] | np.ndarray,
    bid_qtys: Sequence[float] | np.ndarray,
    ask_prices: Sequence[float] | np.ndarray,
    ask_qtys: Sequence[float] | np.ndarray,
    price_hint: float | None = None,
    return_curve: bool = True,
) -> AuctionResult:
    """Clear an auction from bid and ask price/quantity arrays.

    The implementation uses a sorted event pass so it can handle high-depth
    simulations without constructing one Python object per order. When
    ``return_curve`` is false, only the clearing result is materialized and the
    curve DataFrames are empty.
    """
    bid_prices_arr = _as_float_array(bid_prices)
    bid_qtys_arr = _as_float_array(bid_qtys)
    ask_prices_arr = _as_float_array(ask_prices)
    ask_qtys_arr = _as_float_array(ask_qtys)

    if bid_prices_arr.shape != bid_qtys_arr.shape:
        raise ValueError("bid_prices and bid_qtys must have the same shape.")
    if ask_prices_arr.shape != ask_qtys_arr.shape:
        raise ValueError("ask_prices and ask_qtys must have the same shape.")

    bid_mask = np.isfinite(bid_prices_arr) & np.isfinite(bid_qtys_arr) & (bid_prices_arr > 0) & (bid_qtys_arr > 0)
    ask_mask = np.isfinite(ask_prices_arr) & np.isfinite(ask_qtys_arr) & (ask_prices_arr > 0) & (ask_qtys_arr > 0)
    bid_prices_arr = bid_prices_arr[bid_mask]
    bid_qtys_arr = bid_qtys_arr[bid_mask]
    ask_prices_arr = ask_prices_arr[ask_mask]
    ask_qtys_arr = ask_qtys_arr[ask_mask]

    if bid_prices_arr.size + ask_prices_arr.size == 0:
        fallback = float(price_hint) if price_hint is not None and price_hint > 0 else float("nan")
        empty = _empty_curve()
        return AuctionResult(fallback, 0.0, empty.copy(), empty.copy(), empty)

    candidates, demand, supply = _compute_demand_supply_curves(
        bid_prices_arr,
        bid_qtys_arr,
        ask_prices_arr,
        ask_qtys_arr,
    )
    matched_volume = np.minimum(demand, supply)
    imbalance = np.abs(demand - supply)

    max_volume = float(np.max(matched_volume))
    volume_mask = np.isclose(matched_volume, max_volume, rtol=1e-12, atol=1e-12)
    min_imbalance = float(np.min(imbalance[volume_mask]))
    final_mask = volume_mask & np.isclose(imbalance, min_imbalance, rtol=1e-12, atol=1e-12)
    clearing_price = float(np.median(candidates[final_mask]))

    if return_curve:
        auction_curve = pd.DataFrame(
            {
                "price": candidates,
                "demand": demand,
                "supply": supply,
                "matched_volume": matched_volume,
                "imbalance": imbalance,
            }
        )
        demand_curve = auction_curve[["price", "demand"]].copy()
        supply_curve = auction_curve[["price", "supply"]].copy()
    else:
        auction_curve = _empty_curve()
        demand_curve = auction_curve[["price", "demand"]].copy()
        supply_curve = auction_curve[["price", "supply"]].copy()

    return AuctionResult(
        clearing_price=clearing_price,
        clearing_volume=max_volume,
        demand_curve=demand_curve,
        supply_curve=supply_curve,
        auction_curve=auction_curve,
    )


def clear_auction_price_volume(
    bid_prices: Sequence[float] | np.ndarray,
    bid_qtys: Sequence[float] | np.ndarray,
    ask_prices: Sequence[float] | np.ndarray,
    ask_qtys: Sequence[float] | np.ndarray,
    price_hint: float | None = None,
) -> tuple[float, float]:
    """Fast clearing path for simulation loops that do not need curve output."""
    bid_prices_arr = _as_float_array(bid_prices)
    bid_qtys_arr = _as_float_array(bid_qtys)
    ask_prices_arr = _as_float_array(ask_prices)
    ask_qtys_arr = _as_float_array(ask_qtys)

    if bid_prices_arr.shape != bid_qtys_arr.shape:
        raise ValueError("bid_prices and bid_qtys must have the same shape.")
    if ask_prices_arr.shape != ask_qtys_arr.shape:
        raise ValueError("ask_prices and ask_qtys must have the same shape.")

    bid_mask = np.isfinite(bid_prices_arr) & np.isfinite(bid_qtys_arr) & (bid_prices_arr > 0) & (bid_qtys_arr > 0)
    ask_mask = np.isfinite(ask_prices_arr) & np.isfinite(ask_qtys_arr) & (ask_prices_arr > 0) & (ask_qtys_arr > 0)
    bid_prices_arr = bid_prices_arr[bid_mask]
    bid_qtys_arr = bid_qtys_arr[bid_mask]
    ask_prices_arr = ask_prices_arr[ask_mask]
    ask_qtys_arr = ask_qtys_arr[ask_mask]

    if bid_prices_arr.size + ask_prices_arr.size == 0:
        fallback = float(price_hint) if price_hint is not None and price_hint > 0 else float("nan")
        return fallback, 0.0

    candidates, demand, supply = _compute_demand_supply_curves(
        bid_prices_arr,
        bid_qtys_arr,
        ask_prices_arr,
        ask_qtys_arr,
    )
    matched_volume = np.minimum(demand, supply)
    imbalance = np.abs(demand - supply)
    max_volume = float(np.max(matched_volume))
    volume_mask = np.isclose(matched_volume, max_volume, rtol=1e-12, atol=1e-12)
    min_imbalance = float(np.min(imbalance[volume_mask]))
    final_mask = volume_mask & np.isclose(imbalance, min_imbalance, rtol=1e-12, atol=1e-12)
    clearing_price = float(np.median(candidates[final_mask]))
    return clearing_price, max_volume


def clear_auction_price_volume_batch_continuous(
    bid_prices: np.ndarray,
    bid_qtys: np.ndarray,
    ask_prices: np.ndarray,
    ask_qtys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Clear many independent auctions with continuous random prices.

    Inputs are two-dimensional arrays with shape ``(n_runs, n_orders)``. The
    algorithm assumes price ties have probability zero, which is true for the
    lognormal order generation used in the market-depth experiment. It applies
    the same max-volume, min-imbalance, median-price rule while avoiding one
    Python function call per run.
    """
    if bid_prices.ndim != 2 or ask_prices.ndim != 2:
        raise ValueError("batch auction inputs must be two-dimensional.")
    if bid_prices.shape != bid_qtys.shape:
        raise ValueError("bid_prices and bid_qtys must have the same shape.")
    if ask_prices.shape != ask_qtys.shape:
        raise ValueError("ask_prices and ask_qtys must have the same shape.")
    if bid_prices.shape[0] != ask_prices.shape[0]:
        raise ValueError("bid and ask batches must have the same number of runs.")

    n_runs = bid_prices.shape[0]
    all_prices = np.concatenate([bid_prices, ask_prices], axis=1)
    buy_events = np.concatenate([bid_qtys, np.zeros_like(ask_qtys)], axis=1)
    sell_events = np.concatenate([np.zeros_like(bid_qtys), ask_qtys], axis=1)

    order = np.argsort(all_prices, axis=1)
    sorted_prices = np.take_along_axis(all_prices, order, axis=1)
    sorted_buy_events = np.take_along_axis(buy_events, order, axis=1)
    sorted_sell_events = np.take_along_axis(sell_events, order, axis=1)

    cumulative_buy = np.cumsum(sorted_buy_events, axis=1)
    cumulative_sell = np.cumsum(sorted_sell_events, axis=1)
    total_bid_qty = np.sum(bid_qtys, axis=1, keepdims=True)
    demand = total_bid_qty - cumulative_buy + sorted_buy_events
    supply = cumulative_sell
    matched_volume = np.minimum(demand, supply)
    imbalance = np.abs(demand - supply)

    max_volume = np.max(matched_volume, axis=1)
    volume_mask = np.isclose(matched_volume, max_volume[:, None], rtol=1e-12, atol=1e-12)
    masked_imbalance = np.where(volume_mask, imbalance, np.inf)
    min_imbalance = np.min(masked_imbalance, axis=1)
    final_mask = volume_mask & np.isclose(imbalance, min_imbalance[:, None], rtol=1e-12, atol=1e-12)

    clearing_prices = np.empty(n_runs, dtype=float)
    for row in range(n_runs):
        clearing_prices[row] = float(np.median(sorted_prices[row, final_mask[row]]))
    return clearing_prices, max_volume


def _compute_demand_supply_curves(
    bid_prices: np.ndarray,
    bid_qtys: np.ndarray,
    ask_prices: np.ndarray,
    ask_qtys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute demand and supply curves with one sorted event pass."""
    all_prices = np.concatenate([bid_prices, ask_prices])
    buy_events = np.concatenate([bid_qtys, np.zeros_like(ask_qtys)])
    sell_events = np.concatenate([np.zeros_like(bid_qtys), ask_qtys])

    order = np.argsort(all_prices)
    sorted_prices = all_prices[order]
    sorted_buy_events = buy_events[order]
    sorted_sell_events = sell_events[order]

    candidates, start_idx = np.unique(sorted_prices, return_index=True)
    buy_qty_at_price = np.add.reduceat(sorted_buy_events, start_idx)
    sell_qty_at_price = np.add.reduceat(sorted_sell_events, start_idx)

    total_bid_qty = float(np.sum(bid_qtys))
    cumulative_buy_qty = np.cumsum(buy_qty_at_price)
    demand = total_bid_qty - cumulative_buy_qty + buy_qty_at_price
    supply = np.cumsum(sell_qty_at_price)
    return candidates, demand, supply


def _empty_curve() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["price", "demand", "supply", "matched_volume", "imbalance"]
    )
