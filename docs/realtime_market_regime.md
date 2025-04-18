# Real-time Market Regime Detection

This document explains how to use the real-time market regime detection capabilities in the system_trader platform.

## Overview

The Enhanced Market Regime Model is designed to identify different market states (regimes) in real-time using data from both Polygon websocket and Unusual Whales options flow. The model uses a hybrid approach combining:

1. **Hidden Markov Model (HMM)** - For unsupervised regime detection
2. **XGBoost Classifier** - For supervised regime prediction
3. **Options Flow Data** - To enhance regime detection with institutional activity

The model identifies four primary market regimes:
- **Trending Up** - Strong upward price movement with relatively low volatility
- **Trending Down** - Strong downward price movement with moderate volatility
- **High Volatility** - Choppy market conditions with large price swings
- **Low Volatility** - Range-bound market with small price movements

## Requirements

To use the real-time market regime detection, you need:

1. An active Polygon.io websocket connection (configured in `config/collector_config.yaml`)
2. Optional: Unusual Whales API access for options flow data
3. Redis cache for storing real-time data

## Usage

### Basic Usage

```python
from models.market_regime.model import EnhancedMarketRegimeModel
from data.collectors.factory import CollectorFactory
from config.collector_config import CollectorConfig

# Create and configure model
config = {
    'n_regimes': 4,
    'lookback_window': 60,  # Number of data points to use
    'smooth_window': 5,     # Window for smoothing regime transitions
    'feature_groups': {
        'returns': True,
        'volatility': True,
        'trend': True,
        'breadth': True,
        'sentiment': True,
        'options_flow': True  # Set to True to use options flow data
    }
}

# Create model
model = EnhancedMarketRegimeModel(config)

# Load pre-trained model (if available)
model.load('models/market_regime_model')

# Start data collectors
polygon_config = CollectorConfig.load('polygon')
collector = CollectorFactory.create('polygon_websocket', polygon_config)
collector.add_symbol('SPY')  # Add market index
collector.start()

# Get real-time prediction
prediction = model.predict_realtime('SPY')

# Access prediction results
current_regime = prediction['regime']
regime_confidence = prediction[f'prob_{current_regime}']
trading_bias = prediction['trading_bias']
suggested_strategy = prediction['suggested_strategy']

print(f"Current market regime: {current_regime} (confidence: {regime_confidence:.2f})")
print(f"Trading bias: {trading_bias}")
print(f"Suggested strategy: {suggested_strategy}")
```

### Example Script

The system includes an example script that demonstrates real-time market regime detection:

```bash
python examples/realtime_market_regime_example.py --symbols SPY,QQQ,IWM --interval 60 --visualize
```

Options:
- `--symbols`: Comma-separated list of symbols to monitor (default: SPY,QQQ,IWM)
- `--interval`: Prediction interval in seconds (default: 60)
- `--model`: Path to pre-trained model (optional)
- `--visualize`: Enable visualization of regime predictions
- `--duration`: Duration to run in seconds (default: 3600 - 1 hour)

## Prediction Output

The `predict_realtime()` method returns a dictionary with the following information:

```python
{
    'timestamp': datetime.datetime(2025, 4, 18, 2, 25, 0),
    'regime': 'trending_up',                  # Current regime
    'regime_hmm': 'trending_up',              # HMM model prediction
    'regime_xgb': 'trending_up',              # XGBoost model prediction
    'prob_trending_up': 0.85,                 # Probability for each regime
    'prob_trending_down': 0.05,
    'prob_high_volatility': 0.07,
    'prob_low_volatility': 0.03,
    'market_return': 0.0012,                  # Recent market return
    'market_volatility': 0.15,                # Current volatility
    'put_call_ratio': 0.75,                   # Options flow metrics (if available)
    'smart_money_direction': 0.65,
    'implied_volatility': 18.5,
    'unusual_activity_score': 2.3,
    'trading_bias': 'bullish',                # Trading implications
    'volatility_expectation': 'low to moderate',
    'suggested_strategy': 'trend-following'
}
```

## Trading Implications

The model provides trading implications based on the detected regime:

1. **Trending Up**
   - Trading Bias: Bullish
   - Volatility Expectation: Low to moderate
   - Suggested Strategy: Trend-following

2. **Trending Down**
   - Trading Bias: Bearish
   - Volatility Expectation: Moderate to high
   - Suggested Strategy: Trend-following with hedges

3. **High Volatility**
   - Trading Bias: Neutral
   - Volatility Expectation: High
   - Suggested Strategy: Mean-reversion with tight stops

4. **Low Volatility**
   - Trading Bias: Neutral
   - Volatility Expectation: Low
   - Suggested Strategy: Range-bound trading

## Integration with Other Models

The market regime detection can be used to adapt the behavior of other models:

```python
# Get current market regime
regime_prediction = market_regime_model.predict_realtime('SPY')
current_regime = regime_prediction['regime']

# Adjust peak detection threshold based on regime
if current_regime == 'high_volatility':
    # Use higher threshold in volatile markets to avoid false signals
    peak_threshold = 0.8
elif current_regime == 'trending_down':
    # Use lower threshold in downtrends to exit positions quickly
    peak_threshold = 0.6
else:
    # Use default threshold in other regimes
    peak_threshold = 0.7

# Make peak detection prediction with adjusted threshold
peak_prediction = peak_detection_model.predict_realtime('AAPL')
```

## Performance Considerations

- The real-time prediction requires sufficient historical data in Redis cache
- Prediction time increases with the lookback window size
- Options flow data significantly improves regime detection accuracy but requires Unusual Whales API access

## Troubleshooting

If you encounter issues with real-time market regime detection:

1. Ensure the Polygon websocket collector is running and collecting data
2. Check Redis cache for data availability
3. Verify model has been properly trained or loaded
4. For options flow integration, confirm Unusual Whales API access is configured

## Advanced Configuration

For advanced users, the model can be fine-tuned with additional parameters:

```python
config = {
    'n_regimes': 4,                # Number of regimes to detect
    'lookback_window': 60,         # Data points to use
    'smooth_window': 5,            # Smoothing window
    'hmm_n_iter': 100,             # HMM training iterations
    'xgb_n_estimators': 100,       # XGBoost trees
    'xgb_learning_rate': 0.1,      # XGBoost learning rate
    'xgb_max_depth': 5,            # XGBoost tree depth
    'feature_groups': {
        'returns': True,           # Price returns
        'volatility': True,        # Volatility metrics
        'trend': True,             # Trend indicators
        'breadth': True,           # Market breadth
        'sentiment': True,         # Sentiment indicators
        'options_flow': True       # Options flow data
    }
}
