# Redis Setup

This document describes the Redis setup for the System Trader application. Redis is used for caching, data storage, and as a message broker for the application.

## Overview

Redis is an in-memory data structure store that is used as a database, cache, and message broker. In the System Trader application, Redis is used for:

- Caching model predictions
- Storing temporary data
- Sharing data between different components of the system
- Implementing pub/sub messaging patterns

## Components

### Redis Server

Redis server is the main component that stores data in memory and provides access to it.

- **Port**: 6379
- **Configuration**: `/home/ubuntu/system_trader/config/redis/redis.conf`
- **Data Directory**: `/home/ubuntu/system_trader/data/redis`
- **Service**: `system-trader-redis.service`

### Redis Client

The Redis client is a Python class that provides a high-level interface to interact with Redis. It is implemented in `/home/ubuntu/system_trader/data/database/redis_client.py`.

The client provides the following features:

- Connection pooling
- Automatic serialization and deserialization of data
- Namespacing of keys
- Caching with TTL (Time To Live)
- Error handling and retries
- Health checks

## Installation

Redis should be installed on the system. If it's not already installed, you can install it using:

```bash
sudo apt-get update
sudo apt-get install redis-server
```

## Configuration

The Redis configuration file is located at `/home/ubuntu/system_trader/config/redis/redis.conf`. This file defines the Redis server settings, including:

- Network settings
- Memory management
- Persistence
- Security
- Advanced options

## Starting and Stopping

### Starting Redis

To start Redis, run the following command:

```bash
./scripts/monitoring/start_redis.sh
```

This will start Redis with the configuration file specified above.

### Stopping Redis

To stop Redis, you can use the Redis CLI:

```bash
redis-cli shutdown
```

## Systemd Service

A systemd service file is provided to manage Redis as a system service. The service file is located at `/home/ubuntu/system_trader/config/redis/system-trader-redis.service`.

To install the service:

```bash
sudo cp /home/ubuntu/system_trader/config/redis/system-trader-redis.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable system-trader-redis.service
sudo systemctl start system-trader-redis.service
```

To check the status of the service:

```bash
sudo systemctl status system-trader-redis.service
```

## Using the Redis Client

The Redis client is implemented as a singleton, so you can get an instance of it using:

```python
from data.database.redis_client import get_redis_client

# Get the Redis client
redis_client = get_redis_client()

# Set a value
redis_client.set('key', 'value')

# Get a value
value = redis_client.get('key')

# Cache a model prediction
model_name = 'market_regime'
input_hash = redis_client.hash_input(input_data)
redis_client.cache_model_prediction(model_name, input_hash, prediction)

# Get a cached model prediction
prediction = redis_client.get_cached_model_prediction(model_name, input_hash)
```

## Monitoring

Redis can be monitored using the Redis CLI:

```bash
redis-cli info
```

This will show information about the Redis server, including memory usage, clients, and statistics.

## Troubleshooting

### Connection Issues

If you're having trouble connecting to Redis, check the following:

1. Make sure Redis is running:
   ```bash
   ps aux | grep redis-server
   ```

2. Check if the Redis port is open:
   ```bash
   lsof -i :6379
   ```

3. Check the Redis logs:
   ```bash
   tail -f /home/ubuntu/system_trader/logs/redis.log
   ```

### Memory Issues

If Redis is using too much memory, you can:

1. Check the memory usage:
   ```bash
   redis-cli info memory
   ```

2. Adjust the `maxmemory` setting in the Redis configuration file.

3. Change the `maxmemory-policy` to control how Redis removes keys when the memory limit is reached.

## Best Practices

1. Use namespaced keys to avoid key collisions.
2. Set appropriate TTLs for cached data.
3. Use the appropriate data structures for your data.
4. Monitor Redis memory usage.
5. Implement proper error handling in your code.
