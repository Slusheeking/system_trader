# Real-time WebSocket Data Processing

This document explains the real-time WebSocket data processing system in System Trader, including how data flows from collectors through processing to final storage.

## Overview

The system uses a multi-layer architecture to handle real-time WebSocket market data:

1. **WebSocket Collectors** - Connect to provider APIs and stream real-time data
2. **Data Cache** - In-memory and Redis storage for low-latency access to recent data
3. **Data Cleaner** - Validates and cleans incoming data for consistency and reliability
4. **Database Storage** - TimescaleDB for long-term storage, analysis, and querying
5. **Realtime Data Provider** - Unified interface for accessing both WebSocket and REST API data

## Data Flow

```
WebSocket API → Collectors → Data Cache → ML Models
                     ↓
                Data Cleaner
                     ↓
                TimescaleDB
```

## Components

### WebSocket Collectors

* **PolygonWebsocketCollector** - Connects to Polygon.io WebSocket API
* **AlpacaWebsocketCollector** - Connects to Alpaca WebSocket API

Features:
- Automatic reconnection with exponential backoff
- Heartbeat monitoring
- API key authentication via secure credential management
- Performance monitoring and statistics
- Batch processing to reduce database load
- Data validation and cleaning integration

### Data Cache

The `DataCache` system uses Redis for optimized storage and retrieval of WebSocket data with:

- Time-series organization using sorted sets
- Automatic expiration of old data
- Symbol and data type indexing
- Range-based queries for time series analysis

### API Key Manager

The `APIKeyManager` handles secure API keys and credentials:

- Environment variable and config file support
- Automatic key rotation tracking
- Rate limit handling
- Validation and testing of credentials
- Secure storage of API keys

### Real-time Data Provider

The `RealtimeDataProvider` offers a unified interface for accessing market data:

- Combines WebSocket and REST API data sources
- Prioritizes real-time data when available
- Falls back to historical data when needed
- Provides formatted pandas DataFrames for analysis
- Supports multiple data sources with graceful fallback

## Configuration

API keys are configured in `config/api_credentials.yaml` (from template):

```yaml
polygon:
  api_key: "YOUR_POLYGON_API_KEY"

alpaca:
  api_key: "YOUR_ALPACA_API_KEY"
  api_secret: "YOUR_ALPACA_API_SECRET"
```

WebSocket-specific settings are in `config/collector_config.yaml`:

```yaml
polygon:
  websocket:
    enabled: true
    channels: ["T", "Q", "AM"]  # Trades, Quotes, Minute Aggregates
    reconnect_interval_seconds: 30
    heartbeat_interval_seconds: 30
    clean_data: true
    validate_data: true
    store_trades: true
    store_quotes: false  # High volume, default off
    batch_inserts: true
    db_batch_size: 100
```

## Database Schema

WebSocket data is stored in two main tables:

- `market_data` - For aggregate/bar data with OHLCV information
- `websocket_data` - Optimized for high-frequency trade and quote data

Both tables use TimescaleDB hypertables for efficient time-series operations with automatic roll-ups into continuous aggregates for faster queries.

## Usage Examples

### Basic Usage

```python
from data.collectors.polygon_websocket_collector import PolygonWebsocketCollector
from data.collectors.schema import RecordType

# Create collector
collector = PolygonWebsocketCollector()

# Add symbols
collector.add_symbol("AAPL")
collector.add_symbol("MSFT")

# Start collecting
collector.start()

# Get latest data
latest_trade = collector.get_latest_data("AAPL", RecordType.TRADE)
```

### Using the RealtimeDataProvider

```python
from data.processors.realtime_data_provider import RealtimeDataProvider, DataSource

# Get latest price
price = RealtimeDataProvider.get_latest_price("AAPL", DataSource.POLYGON)

# Get OHLCV data
ohlcv_df = RealtimeDataProvider.get_ohlcv_dataframe("AAPL", 100, DataSource.POLYGON)

# Get multi-symbol data
prices = RealtimeDataProvider.get_multi_symbol_prices(["AAPL", "MSFT", "GOOGL"])
```

## Troubleshooting

Common issues and solutions:

1. **WebSocket disconnections**: Check network connectivity and API key validity
2. **Missing data**: Ensure the symbol is correctly added to the subscription list
3. **High latency**: Check Redis performance and consider tuning batch sizes
4. **Database size growth**: Adjust retention policies in TimescaleDB

Run the test script to validate your setup:

```bash
python examples/test_polygon_websocket.py --symbols AAPL,MSFT --duration 60 --check-quality
```

## Service Management

Start the required services:

```bash
sudo systemctl start system-trader-redis
sudo systemctl start system-trader-timescaledb
```

Enable services to start at boot:

```bash
sudo systemctl enable system-trader-redis
sudo systemctl enable system-trader-timescaledb