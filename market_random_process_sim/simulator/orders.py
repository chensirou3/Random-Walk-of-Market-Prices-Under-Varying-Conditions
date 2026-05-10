"""Order data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


OrderSide = Literal["buy", "sell"]


@dataclass(frozen=True)
class Order:
    """A simple limit order used by the daily auction mechanism."""

    side: OrderSide
    price: float
    qty: float
    agent_id: int | None = None
    day: int = 0

    def __post_init__(self) -> None:
        if self.side not in {"buy", "sell"}:
            raise ValueError("Order side must be 'buy' or 'sell'.")
        if self.price <= 0:
            raise ValueError("Order price must be positive.")
        if self.qty < 0:
            raise ValueError("Order quantity cannot be negative.")
