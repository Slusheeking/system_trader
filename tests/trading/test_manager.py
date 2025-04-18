import pandas as pd
import pytest

from trading.manager import TradingManager, TradingSessionState


def test_execute_trading_cycle_calls_components(monkeypatch):
    # Instantiate manager with no config path
    manager = TradingManager(config_path=None)

    # Ensure at least one symbol for processing
    manager.config['watchlist'] = ['SYM']
    manager.position_tracker.get_all_positions = lambda: []

    # Monkeypatch circuit breaker: allow trading and record check calls
    check_calls = []
    monkeypatch.setattr(manager.circuit_breaker, 'allow_trading', lambda: True)
    monkeypatch.setattr(manager.circuit_breaker, 'check', lambda metrics: check_calls.append(metrics))

    # Monkeypatch data client to capture symbols and return dummy data
    fetch_calls = []
    def fake_fetch(symbols):
        fetch_calls.append(list(symbols))
        return {'price': 100}
    monkeypatch.setattr(manager.data_client, 'fetch_latest', fake_fetch)

    # Prepare dummy signals DataFrame with one signal row
    now = pd.Timestamp.now()
    signals_df = pd.DataFrame({
        'timestamp': [now],
        'regime': ['up'],
        'regime_hmm': ['u_hmm'],
        'regime_xgb': ['u_xgb'],
        'prob_trending_up': [0.5],
        'prob_low_volatility': [0.5],
    })
    # Monkeypatch strategy to return our dummy signals
    monkeypatch.setattr(manager.strategy, 'generate_signals', lambda data: signals_df)

    # Monkeypatch order router to record route calls and return a client_order_id
    route_calls = []
    def fake_route(sig_dict):
        route_calls.append(sig_dict)
        return {'order': {'client_order_id': 'order123'}}
    monkeypatch.setattr(manager.order_router, 'route', fake_route)

    # Monkeypatch broker client (order_router) get_order_status
    status_calls = []
    def fake_get_status(order_id):
        status_calls.append(order_id)
        return {'status': 'filled'}
    monkeypatch.setattr(manager.broker, 'get_order_status', fake_get_status)

    # Monkeypatch execution monitor to record calls
    record_calls = []
    def fake_record(sig_dict, report):
        record_calls.append((sig_dict, report))
    monkeypatch.setattr(manager.execution_monitor, 'record', fake_record)

    # Monkeypatch strategy composer to avoid additional logic
    monkeypatch.setattr(manager.strategy_composer, 'process_signals', lambda s: {'entries': [], 'exits': []})
    monkeypatch.setattr(manager.strategy_composer, 'execute_decisions', lambda decisions: {'entries': {'orders': {}}, 'exits': {'orders': {}}})

    # Monkeypatch position update to no-op
    monkeypatch.setattr(manager, '_update_positions', lambda results: None)

    # Execute one trading cycle
    manager._execute_trading_cycle()

    # Assertions
    assert fetch_calls == [['SYM']], "data_client.fetch_latest should be called with the watchlist symbol"
    assert route_calls, "order_router.route should have been called"
    # Check that route was called with our signal as dict
    sig_dict = route_calls[0]
    assert sig_dict.get('regime') == 'up'

    assert status_calls == ['order123'], "broker.get_order_status should be called with the returned client_order_id"
    assert record_calls and record_calls[0][1] == {'status': 'filled'}, "execution_monitor.record should be called with signal and report"
    assert check_calls, "circuit_breaker.check should have been called with metrics"
