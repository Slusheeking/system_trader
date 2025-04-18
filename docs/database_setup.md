# Database Setup for System Trader

This document describes the PostgreSQL + TimescaleDB setup for the System Trader project.

## Database Configuration

The system uses PostgreSQL with the TimescaleDB extension for high-performance time-series data storage. The database configuration is stored in `config/system_config.yaml` under the `database.timescaledb` section.

Current configuration:
```yaml
timescaledb:
  dbname: timescaledb_test
  host: localhost
  max_connections: 10
  max_retries: 3
  max_retry_delay: 30
  min_connections: 1
  password: password
  port: 5432
  retry_delay: 1
  schema: public
  user: timescaleuser
```

## Database Schema

The database includes the following tables:

### 1. market_data

Stores OHLCV (Open, High, Low, Close, Volume) market data for various symbols.

**Schema:**
- `time` (TIMESTAMPTZ, NOT NULL): Timestamp of the data point
- `symbol` (VARCHAR(16), NOT NULL): Ticker symbol
- `open` (DOUBLE PRECISION): Opening price
- `high` (DOUBLE PRECISION): Highest price
- `low` (DOUBLE PRECISION): Lowest price
- `close` (DOUBLE PRECISION): Closing price
- `volume` (DOUBLE PRECISION): Trading volume
- `vwap` (DOUBLE PRECISION): Volume-weighted average price
- `num_trades` (INTEGER): Number of trades
- `source` (VARCHAR(32)): Data source identifier
- `metadata` (JSONB): Additional metadata
- `created_at` (TIMESTAMPTZ): Record creation timestamp

### 2. trade_data

Stores trade execution data.

**Schema:**
- `time` (TIMESTAMPTZ, NOT NULL): Timestamp of the trade
- `order_id` (VARCHAR(64), NOT NULL): Order identifier
- `trade_id` (VARCHAR(64)): Trade identifier
- `symbol` (VARCHAR(16), NOT NULL): Ticker symbol
- `side` (VARCHAR(8), NOT NULL): Buy or sell
- `quantity` (DOUBLE PRECISION, NOT NULL): Trade quantity
- `price` (DOUBLE PRECISION, NOT NULL): Trade price
- `order_type` (VARCHAR(16)): Order type (market, limit, etc.)
- `execution_venue` (VARCHAR(32)): Execution venue
- `strategy_id` (VARCHAR(64)): Strategy identifier
- `commission` (DOUBLE PRECISION): Commission paid
- `slippage` (DOUBLE PRECISION): Price slippage
- `metadata` (JSONB): Additional metadata
- `created_at` (TIMESTAMPTZ): Record creation timestamp

### 3. analytics_data

Stores analytics metrics.

**Schema:**
- `time` (TIMESTAMPTZ, NOT NULL): Timestamp of the metric
- `metric_name` (VARCHAR(64), NOT NULL): Name of the metric
- `metric_value` (DOUBLE PRECISION): Value of the metric
- `symbol` (VARCHAR(16)): Associated ticker symbol
- `strategy_id` (VARCHAR(64)): Associated strategy
- `dimension` (VARCHAR(32)): Metric dimension
- `metadata` (JSONB): Additional metadata
- `created_at` (TIMESTAMPTZ): Record creation timestamp

### 4. model_metrics

Stores machine learning model metrics.

**Schema:**
- `time` (TIMESTAMPTZ, NOT NULL): Timestamp of the metric
- `model_name` (VARCHAR(64), NOT NULL): Name of the model
- `version` (INTEGER, NOT NULL): Model version
- `metric_name` (VARCHAR(32), NOT NULL): Name of the metric
- `metric_value` (DOUBLE PRECISION, NOT NULL): Value of the metric
- `dataset_name` (VARCHAR(32)): Name of the dataset
- `is_training` (BOOLEAN): Whether the metric is from training
- `epoch` (INTEGER): Training epoch
- `batch` (INTEGER): Training batch
- `metadata` (JSONB): Additional metadata
- `created_at` (TIMESTAMPTZ): Record creation timestamp

### 5. positions

Stores trading positions.

**Schema:**
- `id` (SERIAL, PRIMARY KEY): Position identifier
- `symbol` (VARCHAR(16), NOT NULL): Ticker symbol
- `quantity` (INTEGER, NOT NULL): Position quantity
- `entry_price` (DOUBLE PRECISION, NOT NULL): Entry price
- `entry_time` (TIMESTAMPTZ, NOT NULL): Entry timestamp
- `current_price` (DOUBLE PRECISION): Current price
- `strategy_id` (VARCHAR(64)): Strategy identifier
- `status` (VARCHAR(16)): Position status (open, closed)
- `exit_price` (DOUBLE PRECISION): Exit price
- `exit_time` (TIMESTAMPTZ): Exit timestamp
- `pnl` (DOUBLE PRECISION): Profit and loss
- `pnl_percent` (DOUBLE PRECISION): Profit and loss percentage
- `metadata` (JSONB): Additional metadata
- `created_at` (TIMESTAMPTZ): Record creation timestamp
- `updated_at` (TIMESTAMPTZ): Record update timestamp

## Continuous Aggregates

The system uses TimescaleDB continuous aggregates for efficient querying of aggregated data:

1. **market_data_hourly**: Hourly OHLCV aggregation of market data
2. **market_data_daily**: Daily OHLCV aggregation of market data

## Database Maintenance

The system includes automatic data retention and compression policies:

- Market data is compressed after 7 days to save storage space
- Continuous aggregates are automatically refreshed (hourly for market_data_hourly, daily for market_data_daily)

## Connecting to the Database

To connect to the database from Python code, use the `TimeseriesDBClient` class:

```python
from data.database.timeseries_db import TimeseriesDBClient

# Create client using configuration from system_config.yaml
from utils.config_loader import ConfigLoader
config = ConfigLoader().get_config('timescaledb')
db_client = TimeseriesDBClient(config)

# Or create client directly
db_client = TimeseriesDBClient({
    'host': 'localhost',
    'port': 5432,
    'dbname': 'timescaledb_test',
    'user': 'timescaleuser',
    'password': 'password'
})

# Query data
market_data = db_client.query_market_data(
    symbol='AAPL',
    start_time=datetime(2023, 1, 1),
    end_time=datetime(2023, 1, 31)
)

# Insert data
db_client.insert_market_data([
    {
        'time': datetime.now(),
        'symbol': 'AAPL',
        'open': 175.50,
        'high': 178.25,
        'low': 174.80,
        'close': 177.30,
        'volume': 15000000,
        'source': 'test'
    }
])
```

## Testing API Data Collection

You can test the API data collectors and database integration using the `test_api_data_collector.py` script:

```bash
# Test all available APIs
python test_api_data_collector.py

# Test a specific API
python test_api_data_collector.py --api polygon

# Test with specific symbols
python test_api_data_collector.py --symbols AAPL,MSFT,GOOGL

# Test with a specific time range
python test_api_data_collector.py --days 30
```

The script will:
1. Check for available API keys in environment variables
2. Collect data from the specified APIs for the given symbols
3. Process the data using the data processors (cleaning, validation)
4. Store the processed data in the database

This is useful for testing the entire data pipeline from collection to storage.

### Setting API Keys

Before running the script, set the following environment variables:

```bash
# Polygon.io API
export POLYGON_API_KEY="your_polygon_api_key"

# Alpaca API
export ALPACA_API_KEY="your_alpaca_api_key"
export ALPACA_API_SECRET="your_alpaca_api_secret"

# Yahoo Finance API
export YAHOO_API_KEY="your_yahoo_api_key"

# Unusual Whales API
export UNUSUAL_WHALES_API_KEY="your_unusual_whales_api_key"
```

## Database Setup Script

The database tables can be recreated using the `db_setup.sql` script:

```bash
PGPASSWORD=password psql -h localhost -U timescaleuser -d timescaledb_test -f db_setup.sql
```

## Testing the Database Connection

You can test the database connection using the `simple_db_test.py` script:

```bash
python simple_db_test.py