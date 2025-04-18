# Real-time Data for ML Models

This document explains how to use the Polygon websocket collector to provide real-time market data to ML models in the system.

## Overview

The system now supports real-time market data through the Polygon.io websocket API. This enables ML models to receive and process market data in real-time, allowing for more timely trading signals and decisions.

The implementation consists of three main components:

1. **PolygonWebsocketCollector**: Connects to Polygon.io's websocket API and streams real-time market data
2. **RealtimeDataProvider**: Provides a simple interface for ML models to access the real-time data from Redis
3. **Example ML Integration**: Demonstrates how to use real-time data with the system's ML models

## Polygon Websocket Collector

The `PolygonWebsocketCollector` class connects to Polygon.io's websocket API and streams real-time market data. It handles:

- Authentication with Polygon.io
- Subscribing to data channels (trades, quotes, minute aggregates)
- Processing incoming messages
- Storing data in Redis for low-latency access
- Persisting data to TimescaleDB for historical analysis

### Configuration

The websocket collector is configured in `config/collector_config.yaml`:

```yaml
polygon:
  # ... other config ...
  websocket_url: "wss://socket.polygon.io/stocks"
  # ... other config ...
  websocket:
    enabled: true
    channels:
      - T  # Trades
      - Q  # Quotes
      - AM  # Minute aggregates
    reconnect_interval_seconds: 30
    heartbeat_interval_seconds: 30
```

### Usage

To use the websocket collector:

```python
from config.collector_config import CollectorConfig
from data.collectors.factory import CollectorFactory

# Create collector
config = CollectorConfig.load('polygon')
collector = CollectorFactory.create('polygon_websocket', config)

# Add symbols to subscribe to
collector.add_symbol('AAPL')
collector.add_symbol('MSFT')

# Start collector
collector.start()

# ... use the collector ...

# Stop collector when done
collector.stop()
```

## Real-time Data Provider

The `RealtimeDataProvider` class provides a simple interface for ML models to access real-time market data from Redis. It offers methods to:

- Get the latest data for a symbol and record type
- Get recent data as a list or DataFrame
- Get OHLCV data for technical analysis
- Get data for multiple symbols at once

### Usage

```python
from data.collectors.schema import RecordType
from data.processors.realtime_data_provider import RealtimeDataProvider

# Get the latest trade for a symbol
latest_trade = RealtimeDataProvider.get_latest_data('AAPL', RecordType.TRADE)

# Get the latest price for a symbol
latest_price = RealtimeDataProvider.get_latest_price('AAPL')

# Get recent OHLCV data as a DataFrame
ohlcv_df = RealtimeDataProvider.get_ohlcv_dataframe('AAPL', limit=30)

# Get data for multiple symbols
prices = RealtimeDataProvider.get_multi_symbol_prices(['AAPL', 'MSFT', 'GOOGL'])
```

## Integration with ML Models

The system's ML models can now use real-time data for more timely predictions:

### Peak Detection Model

The Peak Detection Model can use real-time data to identify potential price peaks for exit signals:

```python
from models.peak_detection.model import PeakDetectionModel
from data.processors.realtime_data_provider import RealtimeDataProvider

# Load model
model = PeakDetectionModel(config)
model.load('models/peak_detection/saved_model')

# Get real-time data
df = RealtimeDataProvider.get_ohlcv_dataframe('AAPL', limit=30)

# Make prediction
if not df.empty:
    prediction = model.predict(df)
    if prediction['peak_detected']:
        print(f"Peak detected for AAPL with probability {prediction['peak_probability']}")
```

### Entry Timing Model

The Entry Timing Model can use real-time data to identify optimal entry points:

```python
from models.entry_timing.model import EntryTimingModel
from data.processors.realtime_data_provider import RealtimeDataProvider

# Load model
model = EntryTimingModel(config)
model.load('models/entry_timing/saved_model')

# Get real-time data
df = RealtimeDataProvider.get_ohlcv_dataframe('AAPL', limit=30)

# Make prediction
if not df.empty:
    prediction = model.predict(df)
    if prediction['entry_signal']:
        print(f"Entry signal for AAPL with confidence {prediction['entry_confidence']}")
```

### Market Regime Model

The Market Regime Model can use real-time data to detect market regime changes:

```python
from models.market_regime.model import EnhancedMarketRegimeModel
from data.processors.realtime_data_provider import RealtimeDataProvider

# Load model
model = EnhancedMarketRegimeModel(config)
model.load('models/market_regime/saved_model')

# Get real-time data for multiple symbols
symbols = ['AAPL', 'MSFT', 'GOOGL']
data = {}
for symbol in symbols:
    df = RealtimeDataProvider.get_ohlcv_dataframe(symbol, limit=60)
    if not df.empty:
        data[symbol] = df

# Make prediction
if data:
    prediction = model.predict_regime(data)
    print(f"Current market regime: {prediction['regime']} with confidence {prediction['confidence']}")
```

## Example Script

An example script is provided to demonstrate how to use the websocket collector and real-time data provider with ML models:

```bash
python examples/realtime_ml_example.py --symbols AAPL,MSFT,GOOGL --interval 5
```

This script:
1. Starts the Polygon websocket collector for the specified symbols
2. Simulates the system's ML models using real-time data
3. Prints the results at the specified interval

## Requirements

To use the websocket functionality, you need:

1. A valid Polygon.io API key with websocket access
2. Redis server running for caching real-time data
3. The `websocket-client` Python package installed

## Troubleshooting

If you encounter issues with the websocket collector:

1. Check that your Polygon.io API key is valid and has websocket access
2. Ensure Redis is running and accessible
3. Check the logs for error messages
4. Verify that the symbols you're subscribing to are valid and supported by Polygon.io
5. Make sure you're not exceeding Polygon.io's rate limits or connection limits

## Performance Considerations

Real-time data processing can be resource-intensive. Consider the following:

1. Limit the number of symbols you subscribe to
2. Adjust the cache TTL based on your needs
3. Consider using a dedicated Redis instance for real-time data
4. Monitor memory usage, especially if subscribing to many symbols
5. For high-frequency data, consider disabling database persistence and only using Redis
