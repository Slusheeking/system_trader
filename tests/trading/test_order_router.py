import logging
import pytest
import time

from trading.execution.order_router import (
    OrderRouter,
    OrderRequest,
    Order,
    OrderStatus,
    OrderSide,
    TimeInForce,
)


class DummyBroker:
    """
    Dummy broker stub that returns a predetermined Order.
    """
    def __init__(self):
        self.name = 'DummyBroker'
        self.enabled = True

    def submit_order(self, order_request: OrderRequest) -> Order:
        """
        Return a fixed filled order based on the incoming request.
        """
        # Build an Order from the request
        order = Order(order_request)
        # Simulate a filled order
        order.update(
            status=OrderStatus.FILLED,
            filled_quantity=order_request.quantity,
            average_price=order_request.price,
            fees=0.1,
            order_id='dummy-id'
        )
        return order


test_signal = {
    'symbol': 'ABC:US',
    'side': 'sell',
    'quantity': 5.0,
    'price': 200.0,
    'confidence': 0.2,
}


@pytest.fixture
def router(monkeypatch):
    """
    Create an OrderRouter instance with a DummyBroker and bypass circuit breakers.
    """
    router = OrderRouter()
    # Override circuit breaker to always allow trading
    router.circuit_breaker.allow_trading = lambda: True
    router.circuit_breaker.allow_trading_for_symbol = lambda symbol: True
    # Patch broker selection to use DummyBroker
    dummy = DummyBroker()
    monkeypatch.setattr(router, '_select_broker', lambda order_request: dummy)
    # Disable execution monitoring side-effects
    router.execution_monitor.track_order = lambda order: None
    return router


def test_route_success(router):
    """
    Test that OrderRouter.route() returns correct payload for a limit sell order.
    """
    result = router.route(test_signal)
    order_payload = result['order']

    # Basic status checks
    assert result['status'] == 'success'
    assert isinstance(result['latency'], float)
    assert result['latency'] >= 0

    # Order content assertions
    assert order_payload['symbol'] == 'ABC:US'
    assert order_payload['quantity'] == 5.0
    assert order_payload['order_type'] == 'limit'
    # Calculate expected adjusted price for sell
    confidence = test_signal['confidence']
    price_buffer = 0.005 * (1 + confidence)
    expected_price = test_signal['price'] * (1 - price_buffer)
    assert order_payload['price'] == pytest.approx(expected_price, rel=1e-6)

    # Confirm broker response mapping
    assert order_payload['status'] == OrderStatus.FILLED.value
    assert order_payload['order_id'] == 'dummy-id'


def test_route_emits_metrics(router, caplog):
    """
    Ensure that routing logs include the metrics line.
    """
    caplog.set_level(logging.INFO)
    router.route(test_signal)
    # Check that a log message with metrics was emitted
    assert any('Order routing metrics' in rec.message for rec in caplog.records)
