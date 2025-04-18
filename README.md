# Streamlined Autonomous Day Trading System

A production-ready Python-based day trading system with advanced machine learning capabilities, optimized for GH200 ARM64 hardware.

## System Overview

This autonomous trading platform leverages multiple specialized machine learning models with comprehensive data collection pipelines, robust monitoring, and streamlined portfolio management for high-accuracy automated trading.

### Core Technologies

- **Machine Learning**: TensorFlow, XGBoost, scikit-learn
- **Model Management**: MLflow for experiment tracking and versioning
- **Model Optimization**: ONNX for hardware-accelerated inference
- **Data Storage**: PostgreSQL with TimescaleDB extension
- **Caching**: Redis for high-performance data access
- **Monitoring**: Prometheus and Grafana
- **Hardware**: Optimized for GH200 ARM64 processors

## Core Components

### 1. ML Model Suite
- **Stock Selection Model**: XGBoost-based classifier for identifying trading candidates
- **Entry Timing Model**: LSTM-Transformer hybrid for optimal entry points
- **Peak Detection Model**: CNN-LSTM hybrid for price exhaustion and exit signals
- **Risk Sizing Model**: XGBoost regressor for position sizing
- **Market Regime Detection Model**: HMM + XGBoost classifier for market condition analysis

### 2. Data Infrastructure
- **Data Collectors**: Modules for Polygon.io, Yahoo Finance, Alpaca Markets, and Unusual Whales
- **Data Validators**: Ensures data quality and completeness
- **Data Cache**: Redis-based caching for frequently accessed data
- **TimescaleDB**: Optimized time-series storage

### 3. Portfolio Management
- **Position Tracker**: Real-time position monitoring
- **Risk Calculator**: Exposure and risk assessment
- **Diversification Engine**: Portfolio diversification logic
- **Performance Monitor**: P&L and performance tracking

### 4. Trading Operations
- **Strategy Composition**: Framework for combining model outputs
- **Smart Order Router**: Optimized order execution
- **Circuit Breakers**: Trading safety mechanisms
- **Position Management**: Position tracking and management

### 5. Monitoring System
- **Prometheus Metrics**: Custom exporters for key metrics
- **Grafana Dashboards**: Real-time visualization
- **Alert Manager**: Intelligent alerting system

## Directory Structure

```
├── config/
│   ├── system_config.yaml              # System configuration
│   ├── collector_config.py             # Data collector configuration
│   ├── prometheus/                     # Prometheus configuration
│   └── grafana/                        # Grafana configuration
├── data/
│   ├── collectors/                     # Data collection modules
│   │   ├── base_collector.py           # Base collector class
│   │   ├── factory.py                  # Collector factory
│   │   ├── polygon_collector.py        # Polygon.io market data collector
│   │   ├── yahoo_collector.py          # Yahoo Finance data collector
│   │   ├── alpaca_collector.py         # Alpaca Markets data collector
│   │   ├── unusual_whales_collector.py # Options flow data collector
│   │   └── schema.py                   # Data schemas
│   ├── processors/
│   │   ├── data_validator.py           # Data quality validation
│   │   ├── data_cleaner.py             # Data cleaning and normalization
│   │   ├── feature_engineer.py         # Feature engineering framework
│   │   └── data_cache.py               # Redis-based caching
│   └── database/
│       ├── timeseries_db.py            # TimescaleDB integration
│       └── redis_client.py             # Redis client implementation
├── models/
│   ├── interfaces.py                   # Abstract base classes for models
│   ├── market_regime/
│   │   ├── model.py                    # HMM + XGBoost hybrid model
│   │   ├── features.py                 # Market regime features
│   │   └── trainer.py                  # Market regime model trainer
│   ├── stock_selection/
│   │   ├── model.py                    # XGBoost model
│   │   ├── features.py                 # Stock selection features
│   │   └── trainer.py                  # Stock selection model trainer
│   ├── entry_timing/
│   │   ├── model.py                    # LSTM-Transformer hybrid
│   │   ├── features.py                 # Entry timing features
│   │   └── trainer.py                  # Entry timing model trainer
│   ├── peak_detection/
│   │   ├── model.py                    # CNN-LSTM hybrid model
│   │   ├── features.py                 # Peak detection features
│   │   └── trainer.py                  # Peak detection model trainer
│   ├── risk_management/
│   │   ├── model.py                    # XGBoost Regressor
│   │   ├── features.py                 # Risk features
│   │   └── trainer.py                  # Risk model trainer
│   └── optimization/
│       ├── onnx_converter.py           # ONNX model conversion
│       └── gh200_optimizer.py          # GH200 ARM64 optimizations
├── portfolio/
│   ├── position_tracker.py             # Position monitoring
│   ├── risk_calculator.py              # Risk assessment
│   ├── diversification_engine.py       # Diversification logic
│   └── performance_monitor.py          # Performance tracking
├── orchestration/
│   ├── workflow_manager.py             # Model execution sequence
│   ├── decision_framework.py           # Signal reconciliation
│   ├── adaptive_thresholds.py          # Dynamic threshold adjustment
│   └── error_handler.py                # Error handling
├── scheduler/
│   ├── task_scheduler.py               # Scheduler engine
│   ├── tasks/                          # Task definitions
│   └── worker_pool.py                  # Worker management
├── trading/
│   ├── strategy.py                     # Strategy composition
│   ├── execution/
│   │   ├── order_router.py             # Order routing
│   │   ├── execution_monitor.py        # Execution tracking
│   │   └── circuit_breaker.py          # Safety mechanisms
│   └── manager.py                      # Trading manager
├── backtesting/
│   ├── engine.py                       # Backtesting engine
│   └── performance_analyzer.py         # Performance analysis
├── monitoring/
│   ├── prometheus/                     # Prometheus integration
│   │   └── exporters/                  # Metric exporters
│   ├── grafana/                        # Grafana integration
│   └── alerts/                         # Alerting system
├── mlflow/                             # MLflow integration
│   ├── tracking.py                     # Experiment tracking
│   └── registry.py                     # Model registry client
├── utils/
│   ├── logging.py                      # Logging configuration
│   ├── metrics.py                      # Performance metrics
│   └── config_loader.py                # Configuration loading
├── deployment/                         # Deployment configuration (empty)
├── logs/                               # Log files
│   └── yahoo_collector.log             # Yahoo collector logs
├── tests/                              # System tests
│   ├── unit/                           # Unit tests
│   ├── integration/                    # Integration tests
│   └── system/                         # End-to-end tests
├── docs/                               # Documentation
│   ├── architecture.md                 # System architecture
│   ├── models.md                       # Model documentation
│   ├── deployment.md                   # Deployment guide
│   └── monitoring.md                   # Monitoring guide
├── main.py                             # Application entry point
├── compare_ta_libraries.py             # Technical analysis library comparison
├── test_import.py                      # Import testing utility
├── requirements.txt                    # Project dependencies
├── setup.py                            # Package installation
├── .gitignore                          # Git ignore file
└── README.md                           # Project README
```

## Production Readiness

This system is designed for production operation with:

1. **Comprehensive Monitoring**: Prometheus metrics and Grafana dashboards for real-time visibility.

2. **Fault Tolerance**: Error handling, circuit breakers, and failover mechanisms.

3. **Performance Optimization**: ONNX model conversion and GH200 ARM64 optimizations.

4. **Data Management**: TimescaleDB for efficient time-series storage.

5. **Caching Strategy**: Redis caching for frequently accessed data.

6. **Logging System**: Structured logging with file rotation.

7. **Model Interfaces**: Abstract base classes ensuring consistent model implementations.

8. **Technical Analysis**: Comparison of libraries (pandas-ta, ta, ta-lib) for optimal performance.
