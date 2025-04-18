"""
Schema definitions for data collectors.

This module defines the standard data structures used by all collectors
to ensure consistent data formats across different data sources.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class RecordType(str, Enum):
    """Enumeration of possible record types."""
    TRADE = "trade"
    QUOTE = "quote"
    BAR = "bar"
    TICK = "tick"
    AGGREGATE = "aggregate"
    SOCIAL = "social"


class StandardRecord(BaseModel):
    """
    Standard record format for all financial data collected.
    
    This model represents a unified schema for trades, quotes, and bars
    to ensure consistent data handling across different data sources.
    """
    # Common fields for all record types
    symbol: str = Field(..., description="Ticker symbol for the asset")
    timestamp: datetime = Field(..., description="Timestamp of the record")
    record_type: RecordType = Field(..., description="Type of record (trade, quote, bar, etc.)")
    source: str = Field(..., description="Data source identifier (e.g., 'polygon', 'yahoo', 'alpaca')")
    
    # Price and volume fields (used for trades and bars)
    price: Optional[Decimal] = Field(None, description="Trade price or last price")
    volume: Optional[int] = Field(None, description="Trade volume or bar volume")
    
    # Quote-specific fields
    bid_price: Optional[Decimal] = Field(None, description="Bid price")
    bid_size: Optional[int] = Field(None, description="Bid size")
    ask_price: Optional[Decimal] = Field(None, description="Ask price")
    ask_size: Optional[int] = Field(None, description="Ask size")
    
    # Bar-specific fields
    open: Optional[Decimal] = Field(None, description="Opening price for the bar period")
    high: Optional[Decimal] = Field(None, description="Highest price during the bar period")
    low: Optional[Decimal] = Field(None, description="Lowest price during the bar period")
    close: Optional[Decimal] = Field(None, description="Closing price for the bar period")
    vwap: Optional[Decimal] = Field(None, description="Volume-weighted average price")
    
    # Additional metadata fields
    exchange: Optional[str] = Field(None, description="Exchange or venue identifier")
    conditions: Optional[List[str]] = Field(None, description="Trade or quote conditions")
    trade_id: Optional[str] = Field(None, description="Unique trade identifier")
    tape: Optional[str] = Field(None, description="Tape identifier (A, B, C)")
    
    # Extended data storage for source-specific fields
    extended_data: Optional[Dict] = Field(None, description="Additional source-specific data")
    
    class Config:
        """Pydantic model configuration."""
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            Decimal: lambda d: str(d)
        }
        schema_extra = {
            "example": {
                # Trade example
                "symbol": "AAPL",
                "timestamp": "2023-04-17T09:30:00Z",
                "record_type": "trade",
                "source": "polygon",
                "price": "172.50",
                "volume": 100,
                "exchange": "NASDAQ",
                "conditions": ["@", "T"],
                "trade_id": "123456789",
                "tape": "C",
                
                # Quote fields would be null for a trade record
                "bid_price": None,
                "bid_size": None,
                "ask_price": None,
                "ask_size": None,
                
                # Bar fields would be null for a trade record
                "open": None,
                "high": None,
                "low": None,
                "close": None,
                "vwap": None,
                
                # Extended data example
                "extended_data": {
                    "polygon_specific_field": "some_value"
                }
            }
        }