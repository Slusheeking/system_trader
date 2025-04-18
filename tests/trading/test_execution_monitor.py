import pytest
import logging
from datetime import datetime, timedelta

from trading.execution.execution_monitor import ExecutionMonitor


class FakeReport:
    def __init__(self, order_id, symbol, send_time, fill_time,
                 requested_quantity, filled_quantity,
                 executed_price, expected_price):
        self.order_id = order_id
        self.symbol = symbol
        self.send_time = send_time
        self.fill_time = fill_time
        self.requested_quantity = requested_quantity
        self.filled_quantity = filled_quantity
        self.executed_price = executed_price
        self.expected_price = expected_price


@pytest.fixture
def monitor():
    # Create a fresh ExecutionMonitor instance with default settings
    return ExecutionMonitor(config_path=None)


def test_record_emits_metrics_and_logs_with_expected_price(monitor, caplog):
    caplog.set_level(logging.INFO)

    # Setup fake report with expected_price
    order_id = 'order-123'
    symbol = 'AAPL'
    send_time = datetime.now()
    fill_time = send_time + timedelta(milliseconds=150)
    requested_qty = 10
    filled_qty = 8
    expected_price = 100.0
    executed_price = 101.5

    report = FakeReport(
        order_id=order_id,
        symbol=symbol,
        send_time=send_time,
        fill_time=fill_time,
        requested_quantity=requested_qty,
        filled_quantity=filled_qty,
        executed_price=executed_price,
        expected_price=expected_price
    )

    # Call record and capture metrics
    metrics_data = monitor.record(signal=None, report=report)

    # Compute expected values
    expected_latency = (fill_time - send_time).total_seconds() * 1000
    expected_fill_ratio = filled_qty / requested_qty
    expected_slippage = (executed_price - expected_price) / expected_price * 10000

    # Assert returned metrics
    assert pytest.approx(metrics_data['fill_latency_ms'], rel=1e-3) == expected_latency
    assert pytest.approx(metrics_data['fill_ratio'], rel=1e-6) == expected_fill_ratio
    assert pytest.approx(metrics_data['slippage_bps'], rel=1e-6) == expected_slippage

    # Assert internal metrics updated
    assert monitor.metrics.fill_times.get(order_id) == fill_time
    assert monitor.metrics.executed_prices.get(order_id) == executed_price
    assert monitor.metrics.filled_quantities.get(order_id) == filled_qty

    # Assert log message
    log_msg = caplog.text
    assert f"Order executed: {order_id} for {symbol}" in log_msg
    assert f"Quantity: {filled_qty}/{requested_qty}" in log_msg
    assert f"Price: {executed_price}" in log_msg
    assert f"Latency: {expected_latency:.2f}ms" in log_msg


def test_record_emits_zero_slippage_when_no_expected_price(monitor, caplog):
    caplog.set_level(logging.INFO)

    # Setup fake report with expected_price=None
    order_id = 'order-456'
    symbol = 'GOOG'
    send_time = datetime.now()
    fill_time = send_time + timedelta(milliseconds=200)
    requested_qty = 5
    filled_qty = 5
    executed_price = 1500.0
    expected_price = None

    report = FakeReport(
        order_id=order_id,
        symbol=symbol,
        send_time=send_time,
        fill_time=fill_time,
        requested_quantity=requested_qty,
        filled_quantity=filled_qty,
        executed_price=executed_price,
        expected_price=expected_price
    )

    # Call record and capture metrics
    metrics_data = monitor.record(signal=None, report=report)

    # Compute expected values
    expected_latency = (fill_time - send_time).total_seconds() * 1000
    expected_fill_ratio = filled_qty / requested_qty

    # Assert returned metrics
    assert pytest.approx(metrics_data['fill_latency_ms'], rel=1e-3) == expected_latency
    assert pytest.approx(metrics_data['fill_ratio'], rel=1e-6) == expected_fill_ratio
    # Slippage should be zero when expected_price is None
    assert metrics_data['slippage_bps'] == 0

    # Assert internal metrics updated
    assert monitor.metrics.fill_times.get(order_id) == fill_time
    assert monitor.metrics.executed_prices.get(order_id) == executed_price
    assert monitor.metrics.filled_quantities.get(order_id) == filled_qty

    # Assert log message
    log_msg = caplog.text
    assert f"Order executed: {order_id} for {symbol}" in log_msg
