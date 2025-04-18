-- PostgreSQL + TimescaleDB Setup Script for System Trader
-- This script creates the necessary tables, indexes, and hypertables
-- It includes defensive code to handle schema evolution over time

-- Enable PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For text search

-- Function to add a column if it doesn't exist
CREATE OR REPLACE FUNCTION add_column_if_not_exists(
    _table_name TEXT, 
    _column_name TEXT, 
    _column_type TEXT
) RETURNS VOID AS $$
BEGIN
    IF NOT EXISTS (
        SELECT FROM information_schema.columns 
        WHERE table_name = _table_name AND column_name = _column_name
    ) THEN
        EXECUTE FORMAT('ALTER TABLE %s ADD COLUMN %s %s', 
                       _table_name, _column_name, _column_type);
        RAISE NOTICE 'Added column % to table %', _column_name, _table_name;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function to create an index if it doesn't exist
CREATE OR REPLACE FUNCTION create_index_if_not_exists(
    _index_name TEXT, 
    _table_name TEXT, 
    _column_name TEXT
) RETURNS VOID AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes WHERE indexname = _index_name
    ) THEN
        EXECUTE FORMAT('CREATE INDEX %s ON %s (%s)', 
                       _index_name, _table_name, _column_name);
        RAISE NOTICE 'Created index % on table %', _index_name, _table_name;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function to create a table if it doesn't exist
CREATE OR REPLACE FUNCTION create_table_if_not_exists(
    _table_name TEXT,
    _table_definition TEXT
) RETURNS VOID AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables WHERE table_name = _table_name
    ) THEN
        EXECUTE FORMAT('CREATE TABLE %s (%s)', _table_name, _table_definition);
        RAISE NOTICE 'Created table %', _table_name;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Create a universal market data table that handles data from all sources
SELECT create_table_if_not_exists('market_data', '
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(16) NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    vwap DOUBLE PRECISION,
    num_trades INTEGER,
    source VARCHAR(32),
    data_type VARCHAR(32),
    record_type VARCHAR(32),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
');

-- Add source discrimination columns if they don't exist
SELECT add_column_if_not_exists('market_data', 'source', 'VARCHAR(32)');
SELECT add_column_if_not_exists('market_data', 'data_type', 'VARCHAR(32)');
SELECT add_column_if_not_exists('market_data', 'record_type', 'VARCHAR(32)');

-- Create indexes
SELECT create_index_if_not_exists('idx_market_data_symbol', 'market_data', 'symbol');
SELECT create_index_if_not_exists('idx_market_data_time_symbol', 'market_data', '(time, symbol)');
SELECT create_index_if_not_exists('idx_market_data_source', 'market_data', 'source');
SELECT create_index_if_not_exists('idx_market_data_data_type', 'market_data', 'data_type');

-- Convert to hypertable if not already
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT FROM timescaledb_information.hypertables 
        WHERE hypertable_name = 'market_data'
    ) THEN
        PERFORM create_hypertable('market_data', 'time', chunk_time_interval => INTERVAL '1 day');
        RAISE NOTICE 'Converted market_data to hypertable';
    END IF;
END $$;

-- Add compression policy if supported
DO $$
BEGIN
    IF EXISTS (
        SELECT FROM pg_proc WHERE proname = 'add_compression_policy'
    ) THEN
        BEGIN
            ALTER TABLE market_data SET (
                timescaledb.compress,
                timescaledb.compress_segmentby = 'symbol,source'
            );
            
            SELECT add_compression_policy('market_data', INTERVAL '7 days');
            RAISE NOTICE 'Added compression policy to market_data';
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'Could not add compression policy: %', SQLERRM;
        END;
    END IF;
END $$;

-- Create WebSocket-specific table optimized for real-time data
SELECT create_table_if_not_exists('websocket_data', '
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(16) NOT NULL,
    data_type VARCHAR(32) NOT NULL,
    source VARCHAR(32) NOT NULL,
    price DOUBLE PRECISION,
    size DOUBLE PRECISION,
    bid DOUBLE PRECISION,
    ask DOUBLE PRECISION,
    bid_size DOUBLE PRECISION,
    ask_size DOUBLE PRECISION,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
');

-- Create indexes for websocket_data
SELECT create_index_if_not_exists('idx_websocket_data_symbol', 'websocket_data', 'symbol');
SELECT create_index_if_not_exists('idx_websocket_data_time_symbol', 'websocket_data', '(time, symbol)');
SELECT create_index_if_not_exists('idx_websocket_data_type', 'websocket_data', 'data_type');
SELECT create_index_if_not_exists('idx_websocket_data_source', 'websocket_data', 'source');

-- Convert websocket_data to hypertable
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT FROM timescaledb_information.hypertables 
        WHERE hypertable_name = 'websocket_data'
    ) THEN
        PERFORM create_hypertable('websocket_data', 'time', chunk_time_interval => INTERVAL '1 hour');
        RAISE NOTICE 'Converted websocket_data to hypertable';
    END IF;
END $$;

-- News Data Table for storing news articles and sentiment
SELECT create_table_if_not_exists('news_data', '
    id SERIAL PRIMARY KEY,
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    news_id TEXT,
    title TEXT,
    author TEXT,
    source TEXT,
    url TEXT,
    sentiment_score FLOAT,
    sentiment_magnitude FLOAT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
');

-- Create hypertable for news_data
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT FROM timescaledb_information.hypertables 
        WHERE hypertable_name = 'news_data'
    ) THEN
        PERFORM create_hypertable('news_data', 'time', if_not_exists => TRUE);
        RAISE NOTICE 'Converted news_data to hypertable';
    END IF;
END $$;

-- Create index on symbol for faster queries
SELECT create_index_if_not_exists('idx_news_data_symbol', 'news_data', 'symbol');
SELECT create_index_if_not_exists('idx_news_data_time_symbol', 'news_data', '(time, symbol)');
SELECT create_index_if_not_exists('idx_news_data_sentiment', 'news_data', 'sentiment_score');

-- Social Data Table for storing Reddit and other social media data
SELECT create_table_if_not_exists('social_data', '
    id SERIAL PRIMARY KEY,
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    source TEXT NOT NULL,
    platform TEXT NOT NULL,
    subreddit TEXT,
    post_id TEXT,
    parent_id TEXT,
    author TEXT,
    content_type TEXT,
    sentiment_score FLOAT,
    sentiment_magnitude FLOAT,
    score INT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
');

-- Create hypertable for social_data
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT FROM timescaledb_information.hypertables 
        WHERE hypertable_name = 'social_data'
    ) THEN
        PERFORM create_hypertable('social_data', 'time', if_not_exists => TRUE);
        RAISE NOTICE 'Converted social_data to hypertable';
    END IF;
END $$;

-- Create indexes for social_data
SELECT create_index_if_not_exists('idx_social_data_symbol', 'social_data', 'symbol');
SELECT create_index_if_not_exists('idx_social_data_time_symbol', 'social_data', '(time, symbol)');
SELECT create_index_if_not_exists('idx_social_data_platform', 'social_data', 'platform');
SELECT create_index_if_not_exists('idx_social_data_sentiment', 'social_data', 'sentiment_score');

-- Unusual Whales options flow data
SELECT create_table_if_not_exists('options_flow', '
    id SERIAL PRIMARY KEY,
    time TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    strike FLOAT NOT NULL,
    expiry DATE NOT NULL,
    option_type VARCHAR(4) NOT NULL,
    size INT NOT NULL,
    premium FLOAT NOT NULL,
    spot FLOAT,
    implied_volatility FLOAT,
    open_interest INT,
    volume_oi_ratio FLOAT,
    sentiment_score FLOAT,
    source VARCHAR(32),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
');

-- Create hypertable for options_flow
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT FROM timescaledb_information.hypertables 
        WHERE hypertable_name = 'options_flow'
    ) THEN
        PERFORM create_hypertable('options_flow', 'time', if_not_exists => TRUE);
        RAISE NOTICE 'Converted options_flow to hypertable';
    END IF;
END $$;

-- Create indexes for options_flow
SELECT create_index_if_not_exists('idx_options_flow_symbol', 'options_flow', 'symbol');
SELECT create_index_if_not_exists('idx_options_flow_time_symbol', 'options_flow', '(time, symbol)');
SELECT create_index_if_not_exists('idx_options_flow_expiry', 'options_flow', 'expiry');
SELECT create_index_if_not_exists('idx_options_flow_sentiment', 'options_flow', 'sentiment_score');

-- Continuous Aggregates for Market Data
DO $$
BEGIN
    -- Hourly aggregate view
    IF NOT EXISTS (
        SELECT FROM timescaledb_information.continuous_aggregates 
        WHERE view_name = 'market_data_hourly'
    ) THEN
        CREATE MATERIALIZED VIEW market_data_hourly
        WITH (timescaledb.continuous) AS
        SELECT 
            time_bucket('1 hour', time) AS bucket,
            symbol,
            source,
            FIRST(open, time) AS open,
            MAX(high) AS high,
            MIN(low) AS low,
            LAST(close, time) AS close,
            SUM(volume) AS volume,
            AVG(vwap) AS vwap,
            SUM(num_trades) AS num_trades
        FROM market_data
        GROUP BY bucket, symbol, source;
        
        -- Add refresh policy (every hour)
        SELECT add_continuous_aggregate_policy('market_data_hourly',
            start_offset => INTERVAL '3 days',
            end_offset => INTERVAL '1 hour',
            schedule_interval => INTERVAL '1 hour');
            
        RAISE NOTICE 'Created market_data_hourly view';
    END IF;
    
    -- Daily aggregate view
    IF NOT EXISTS (
        SELECT FROM timescaledb_information.continuous_aggregates 
        WHERE view_name = 'market_data_daily'
    ) THEN
        CREATE MATERIALIZED VIEW market_data_daily
        WITH (timescaledb.continuous) AS
        SELECT 
            time_bucket('1 day', time) AS bucket,
            symbol,
            source,
            FIRST(open, time) AS open,
            MAX(high) AS high,
            MIN(low) AS low,
            LAST(close, time) AS close,
            SUM(volume) AS volume,
            AVG(vwap) AS vwap,
            SUM(num_trades) AS num_trades
        FROM market_data
        GROUP BY bucket, symbol, source;
        
        -- Add refresh policy (daily)
        SELECT add_continuous_aggregate_policy('market_data_daily',
            start_offset => INTERVAL '30 days',
            end_offset => INTERVAL '1 day',
            schedule_interval => INTERVAL '1 day');
            
        RAISE NOTICE 'Created market_data_daily view';
    END IF;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Error creating continuous aggregates: %', SQLERRM;
END $$;

-- Create a data retention policy for websocket data (keep for 7 days)
DO $$
BEGIN
    IF EXISTS (
        SELECT FROM pg_proc WHERE proname = 'add_retention_policy'
    ) THEN
        BEGIN
            SELECT add_retention_policy('websocket_data', INTERVAL '7 days');
            RAISE NOTICE 'Added retention policy to websocket_data';
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'Could not add retention policy: %', SQLERRM;
        END;
    END IF;
END $$;

-- Create a view for aggregated sentiment by day
CREATE OR REPLACE VIEW daily_sentiment AS
SELECT 
    date_trunc('day', time) AS day,
    symbol,
    'news' AS source,
    avg(sentiment_score) AS avg_sentiment,
    count(*) AS mention_count
FROM news_data
GROUP BY day, symbol
UNION ALL
SELECT 
    date_trunc('day', time) AS day,
    symbol,
    'social' AS source,
    avg(sentiment_score) AS avg_sentiment,
    count(*) AS mention_count
FROM social_data
GROUP BY day, symbol;

-- Create a function to check database health
CREATE OR REPLACE FUNCTION db_health_check()
RETURNS JSON AS $$
DECLARE
    result JSON;
BEGIN
    SELECT json_build_object(
        'status', 'healthy',
        'timestamp', NOW(),
        'tables', (
            SELECT json_object_agg(table_name, row_count)
            FROM (
                SELECT 
                    table_name,
                    (SELECT count(*) FROM information_schema.tables WHERE table_name = t.table_name) AS row_count
                FROM information_schema.tables t
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
            ) AS table_counts
        ),
        'hypertables', (
            SELECT json_object_agg(hypertable_name, chunks)
            FROM (
                SELECT 
                    hypertable_name,
                    (SELECT count(*) FROM timescaledb_information.chunks WHERE hypertable_name = h.hypertable_name) AS chunks
                FROM timescaledb_information.hypertables h
            ) AS hypertable_counts
        ),
        'continuous_aggregates', (
            SELECT json_agg(view_name)
            FROM timescaledb_information.continuous_aggregates
        ),
        'extensions', (
            SELECT json_agg(extname || ' ' || extversion)
            FROM pg_extension
        )
    ) INTO result;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- Create an upsert function for market data to handle duplicates
CREATE OR REPLACE FUNCTION upsert_market_data(
    p_time TIMESTAMPTZ,
    p_symbol VARCHAR,
    p_open DOUBLE PRECISION,
    p_high DOUBLE PRECISION,
    p_low DOUBLE PRECISION,
    p_close DOUBLE PRECISION,
    p_volume DOUBLE PRECISION,
    p_source VARCHAR,
    p_data_type VARCHAR
) RETURNS VOID AS $$
BEGIN
    INSERT INTO market_data (
        time, symbol, open, high, low, close, volume, source, data_type
    ) VALUES (
        p_time, p_symbol, p_open, p_high, p_low, p_close, p_volume, p_source, p_data_type
    )
    ON CONFLICT (time, symbol, source) 
    DO UPDATE SET
        open = p_open,
        high = p_high,
        low = p_low,
        close = p_close,
        volume = p_volume;
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'Error in upsert_market_data: %', SQLERRM;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO your_user;

-- Output success message
SELECT 'Database setup complete!' AS result;
