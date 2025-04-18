#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Database Package
--------------
This package contains database-related modules for the system trader.
"""

from data.database.redis_client import RedisClient
from data.database.timeseries_db import TimeseriesDBClient, get_timescale_client

# alias for backward compat
TimeSeriesDB = TimeseriesDBClient
TimeSeriesDBFactory = get_timescale_client

__all__ = [
    'RedisClient',
    'TimeSeriesDB',
    'TimeSeriesDBFactory'
]
