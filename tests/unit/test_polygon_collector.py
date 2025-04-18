import pytest
import os
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from data.collectors.polygon_collector import PolygonCollector
from data.collectors.schema import RecordType
from config.collector_config import CollectorConfig


class TestPolygonCollector:
    """Tests for the PolygonCollector class."""

    @pytest.fixture
    def mock_polygon_client(self):
        """Create a mock Polygon client."""
        with patch('data.collectors.polygon_collector.RESTClient') as mock_client:
            # Mock the get_aggs method
            mock_instance = mock_client.return_value
            mock_instance.get_aggs.return_value = MagicMock(
                results=[
                    MagicMock(
                        open=100.0,
                        high=105.0,
                        low=99.0,
                        close=103.0,
                        volume=1000,
                        vwap=102.5,
                        timestamp=int(datetime.now().timestamp() * 1000),
                        transactions=500
                    ),
                    MagicMock(
                        open=103.0,
                        high=107.0,
                        low=102.0,
                        close=106.0,
                        volume=1200,
                        vwap=104.5,
                        timestamp=int((datetime.now() + timedelta(hours=1)).timestamp() * 1000),
                        transactions=600
                    )
                ]
            )
            
            # Mock the get_trades method
            mock_instance.get_trades.return_value = MagicMock(
                results=[
                    MagicMock(
                        price=103.5,
                        size=10,
                        exchange=1,
                        conditions=[1, 2],
                        timestamp=int(datetime.now().timestamp() * 1000),
                        trf_id="123456"
                    ),
                    MagicMock(
                        price=104.2,
                        size=15,
                        exchange=2,
                        conditions=[1],
                        timestamp=int((datetime.now() + timedelta(minutes=5)).timestamp() * 1000),
                        trf_id="123457"
                    )
                ]
            )
            
            # Mock the get_quotes method
            mock_instance.get_quotes.return_value = MagicMock(
                results=[
                    MagicMock(
                        bid_price=103.0,
                        bid_size=100,
                        ask_price=103.5,
                        ask_size=150,
                        timestamp=int(datetime.now().timestamp() * 1000)
                    ),
                    MagicMock(
                        bid_price=103.2,
                        bid_size=120,
                        ask_price=103.7,
                        ask_size=130,
                        timestamp=int((datetime.now() + timedelta(minutes=1)).timestamp() * 1000)
                    )
                ]
            )
            
            yield mock_instance

    @pytest.fixture
    def collector_config(self):
        """Create a collector configuration."""
        config_dict = {
            'api_key': 'test_api_key',
            'base_url': 'https://api.polygon.io/v2',
            'timeout_seconds': 10,
            'retry_attempts': 3,
            'retry_delay_seconds': 1,
            'data_types': {
                'bars': {'enabled': True},
                'trades': {'enabled': True},
                'quotes': {'enabled': True}
            }
        }
        return CollectorConfig(config_dict)

    @pytest.fixture
    def polygon_collector(self, collector_config, mock_polygon_client):
        """Create a PolygonCollector instance with mocked client."""
        collector = PolygonCollector(collector_config)
        collector.set_symbol('AAPL')
        return collector

    def test_collect_bars(self, polygon_collector, mock_polygon_client):
        """Test collecting bar data."""
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now()
        
        # Set data type to bars only
        polygon_collector.data_types = ['bars']
        
        # Collect data
        records = polygon_collector.collect(start_date, end_date)
        
        # Verify the results
        assert len(records) == 2
        assert all(r.record_type == RecordType.AGGREGATE for r in records)
        assert all(r.symbol == 'AAPL' for r in records)
        assert all(hasattr(r, 'open') for r in records)
        assert all(hasattr(r, 'high') for r in records)
