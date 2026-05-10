import numpy as np

from simulator.auction import clear_auction
from simulator.orders import Order


def test_simple_clearing_price_is_reasonable():
    orders = [
        Order(side="buy", price=100.0, qty=10.0),
        Order(side="sell", price=100.0, qty=5.0),
    ]

    result = clear_auction(orders)

    assert result.clearing_price == 100.0
    assert result.clearing_volume == 5.0
    assert set(["price", "demand", "supply", "matched_volume", "imbalance"]).issubset(result.auction_curve.columns)


def test_no_crossing_orders_are_handled_safely():
    orders = [
        Order(side="buy", price=90.0, qty=10.0),
        Order(side="sell", price=110.0, qty=10.0),
    ]

    result = clear_auction(orders)

    assert np.isfinite(result.clearing_price)
    assert result.clearing_price == 100.0
    assert result.clearing_volume == 0.0


def test_tie_break_uses_volume_then_imbalance_then_median():
    imbalance_orders = [
        Order(side="buy", price=100.0, qty=10.0),
        Order(side="buy", price=90.0, qty=100.0),
        Order(side="sell", price=90.0, qty=10.0),
    ]
    imbalance_result = clear_auction(imbalance_orders)

    assert imbalance_result.clearing_volume == 10.0
    assert imbalance_result.clearing_price == 100.0

    median_orders = [
        Order(side="buy", price=100.0, qty=10.0),
        Order(side="sell", price=90.0, qty=10.0),
    ]
    median_result = clear_auction(median_orders)

    assert median_result.clearing_volume == 10.0
    assert median_result.clearing_price == 95.0
