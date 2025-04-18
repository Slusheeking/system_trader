import pytest
import time
from datetime import datetime
from decimal import Decimal

from data.collectors.base_collector import BaseCollector, CollectorError
from data.collectors.schema import StandardRecord


class DummyConfig:
    """
    Dummy configuration object for testing BaseCollector behavior.
    """
    retry_attempts = 3
    backoff_factor = 0.1


class DummyCollector(BaseCollector):
    """
    Collector that simulates network failures and pagination.
    """
    def __init__(self, config):
        super().__init__(config)
        self.authenticated = False
        self.attempt_count = 0
        # Two-page dataset
        self._pages = [
            {"data": [{
                "symbol": "TEST", 
                "timestamp": datetime(2021, 1, 1),
                "record_type": "trade", 
                "source": "dummy", 
                "price": Decimal("1.0"), 
                "volume": 10
            }], "next": "token1"},
            {"data": [{
                "symbol": "TEST", 
                "timestamp": datetime(2021, 1, 2),
                "record_type": "trade", 
                "source": "dummy", 
                "price": Decimal("2.0"), 
                "volume": 20
            }], "next": None}
        ]

    def _authenticate(self) -> None:
        # Mark that authentication occurred
        self.authenticated = True

    def _request_page(self, page_token):
        # Simulate two network failures before success
        self.attempt_count += 1
        if self.attempt_count <= 2:
            raise Exception("Simulated network error")

        # Return page based on token
        if page_token is None:
            return self._pages[0]
        if page_token == "token1":
            return self._pages[1]
        return {"data": [], "next": None}

    def _parse(self, raw):
        # Convert raw dicts into StandardRecord instances
        records = []
        for entry in raw.get("data", []):
            records.append(StandardRecord(**entry))
        return records, raw.get("next")


@pytest.fixture(autouse=True)
def patch_sleep(monkeypatch):
    """
    Prevent actual sleeping during tests by patching time.sleep.
    """
    monkeypatch.setattr(time, "sleep", lambda _: None)


def test_retry_and_backoff_behavior():
    """
    Test that DummyCollector retries on failures and applies backoff.
    """
    config = DummyConfig()
    collector = DummyCollector(config)
    # Should collect two records across two pages after retries
    records = collector.collect(datetime(2021, 1, 1), datetime(2021, 1, 3))

    # Authentication must have occurred
    assert collector.authenticated is True
    # At least two retry attempts were made
    assert collector.attempt_count >= 3

    # Two records should be returned
    assert len(records) == 2
    assert records[0].price == Decimal("1.0")
    assert records[1].volume == 20


def test_pagination_without_failures():
    """
    Test that pagination works when no network errors occur.
    """
    class NoFailCollector(DummyCollector):
        def _request_page(self, page_token):
            # Always succeed on first try
            if page_token is None:
                return self._pages[0]
            return self._pages[1]

    collector = NoFailCollector(DummyConfig())
    records = collector.collect(datetime(2021, 1, 1), datetime(2021, 1, 3))

    assert len(records) == 2
    # Validate timestamps preserved correctly
    assert records[1].timestamp == datetime(2021, 1, 2)


def test_parse_output_must_match_schema():
    """
    Ensure that malformed parse output leads to validation errors.
    """
    class BadParseCollector(BaseCollector):
        def __init__(self, config):
            super().__init__(config)

        def _authenticate(self):
            pass

        def _request_page(self, page_token):
            return {"data": [{"symbol": None}], "next": None}

        def _parse(self, raw):
            # Return a record missing required fields
            # This will trigger a pydantic validation error
            return [StandardRecord(**{})], None

    collector = BadParseCollector(DummyConfig())
    with pytest.raises(Exception):
        collector.collect(datetime.now(), datetime.now())


def test_missing_hooks_raise_type_error():
    """
    Subclasses that do not implement required abstract methods cannot be instantiated.
    """
    # Attempting to instantiate without overrides should raise TypeError
    with pytest.raises(TypeError):
        class IncompleteCollector(BaseCollector):
            # No hooks implemented
            pass

        _ = IncompleteCollector(DummyConfig())
