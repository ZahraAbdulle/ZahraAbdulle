# TFT Hybrid Forecaster

A Temporal Fusion Transformer (TFT) based hybrid forecasting system for stock price prediction using pytorch-forecasting.

## Overview

This repository implements a comprehensive stock price forecasting system using the Temporal Fusion Transformer architecture. The system includes:

- **Data Processing**: Handles ticker CSV files with OHLCV data, technical indicators, and sentiment features
- **Model Training**: TFT model training with early stopping and validation
- **Evaluation**: Expanding-origin evaluation with monthly refits and comprehensive metrics
- **Backtesting**: Strategy backtesting with transaction cost analysis

## Features

- **Train-only scaling** with sentiment zero-preservation
- **Fixed date splits**: Train (2021-02-03 → 2022-12-30), Val (2023-01-03 → 2023-05-31), Test (2023-06-01 → 2023-12-28)
- **Deterministic seeding** for reproducible results
- **Monthly refit** capability during test period
- **Comprehensive metrics**: RMSE, MAE, Theil U2, Directional Accuracy, Sharpe Ratio, Maximum Drawdown
- **Transaction cost analysis** at multiple cost levels

## Repository Structure

```
├── src/
│   ├── hybrid_tft.py      # Main training and inference CLI
│   ├── dataio.py          # Data loading and preprocessing
│   └── eval.py            # Evaluation and backtesting
├── outputs/hybrid/<TICKER>/  # Model outputs and results
├── requirements.txt       # Python dependencies
├── .devcontainer/        # VS Code development container
└── create_sample_data.py # Sample data generation
```

## Installation

### Option 1: Local Installation

```bash
pip install -r requirements.txt
```

### Option 2: GitHub Codespaces

This repository includes a devcontainer configuration for GitHub Codespaces with Python 3.11. Simply open in Codespaces and dependencies will be installed automatically.

## Usage

### Data Format

Each ticker requires a CSV file with the following schema:
- `date`: Date column (YYYY-MM-DD format)
- `ticker`: Ticker symbol
- `Open`, `High`, `Low`, `Close`, `Volume`: OHLCV data
- Technical indicators: Various engineered features
- Sentiment features: `Tw_*`, `Rd_*`, `Nw_SP500_*` (zeros preserved during scaling)
- `Target`: Next-day close price

### Training

Train a model for a single ticker:
```bash
python src/hybrid_tft.py --ticker AAPL --data-path ./data
```

Train models for all supported tickers:
```bash
python src/hybrid_tft.py --all-tickers --data-path ./data
```

### Evaluation

Evaluate a trained model:
```bash
python src/eval.py --ticker AAPL --data-path ./data
```

Evaluate all tickers:
```bash
python src/eval.py --all-tickers --data-path ./data
```

### Sample Data Generation

To test the system with synthetic data:
```bash
python create_sample_data.py
python src/hybrid_tft.py --ticker AAPL --data-path ./sample_data
```

## Output Files

For each ticker, the system generates:

- `run_config_hybrid_<TICKER>.json`: Training configuration
- `features_used_<TICKER>.json`: List of features used
- `scaler_summary_<TICKER>.json`: Scaling information
- `training_curves_<TICKER>.png`: Training progress plots
- `test_predictions_<TICKER>.csv`: Test set predictions
- `metrics_<TICKER>.json`: Evaluation metrics
- `backtest_0bps_<TICKER>.csv`: Backtest results (0 bps costs)
- `backtest_10bps_<TICKER>.csv`: Backtest results (10 bps costs)
- `cost_grid_<TICKER>.csv`: Performance across cost levels
- `provenance_<TICKER>.json`: Execution metadata
- `environment.txt`: System environment information

## Model Architecture

The TFT model includes:
- **Variable Selection Networks**: For relevant feature selection
- **Gated Residual Networks**: For non-linear processing
- **Multi-Head Attention**: For temporal relationships
- **Quantile Regression**: For uncertainty estimation

## Evaluation Methodology

- **Expanding Origin**: Uses all historical data up to prediction point
- **Monthly Refits**: Model parameters updated monthly during test period
- **One-Step-Ahead**: Single day forecast horizon
- **Comprehensive Metrics**: Statistical and financial performance measures

## Dependencies

- Python 3.11
- PyTorch >= 2.1.0
- PyTorch Lightning >= 2.1.0
- pytorch-forecasting >= 1.0.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0

## Contributing

This is a research and educational project. Feel free to fork and experiment with different configurations and improvements.

## About

Developed by Zahra - Data Science graduate specializing in financial and quantitative analysis, with interests in algorithmic trading and data-driven insights.