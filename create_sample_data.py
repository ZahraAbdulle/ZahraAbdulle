"""
Create sample data files for testing the TFT hybrid forecaster.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_sample_ticker_data(ticker: str, start_date: str = '2021-01-01', 
                            end_date: str = '2023-12-31') -> pd.DataFrame:
    """Create synthetic ticker data with required schema."""
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days only
    
    np.random.seed(42)  # For reproducible data
    n_days = len(date_range)
    
    # Base price simulation (random walk with drift)
    initial_price = 100.0
    returns = np.random.normal(0.0008, 0.02, n_days)  # Small positive drift
    prices = [initial_price]
    
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
    
    prices = np.array(prices)
    
    # OHLCV data
    highs = prices * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    opens = prices * (1 + np.random.normal(0, 0.005, n_days))
    volumes = np.random.lognormal(15, 0.5, n_days).astype(int)
    
    # Technical indicators (simplified)
    data = {
        'date': date_range,
        'ticker': ticker,
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': prices,
        'Volume': volumes
    }
    
    # Add some technical indicators
    df_temp = pd.DataFrame(data)
    
    # Simple moving averages
    df_temp['SMA_10'] = df_temp['Close'].rolling(10).mean()
    df_temp['SMA_20'] = df_temp['Close'].rolling(20).mean()
    df_temp['SMA_50'] = df_temp['Close'].rolling(50).mean()
    
    # RSI approximation
    delta = df_temp['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_temp['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df_temp['BB_middle'] = df_temp['Close'].rolling(20).mean()
    bb_std = df_temp['Close'].rolling(20).std()
    df_temp['BB_upper'] = df_temp['BB_middle'] + (bb_std * 2)
    df_temp['BB_lower'] = df_temp['BB_middle'] - (bb_std * 2)
    
    # Volume indicators
    df_temp['Volume_SMA'] = df_temp['Volume'].rolling(20).mean()
    df_temp['Volume_ratio'] = df_temp['Volume'] / df_temp['Volume_SMA']
    
    # Sentiment features (Twitter, Reddit, News with some zeros)
    sentiment_prob = 0.3  # 30% chance of non-zero sentiment
    
    # Twitter sentiment
    df_temp['Tw_sentiment_score'] = np.where(
        np.random.random(n_days) < sentiment_prob,
        np.random.normal(0, 0.5, n_days),
        0
    )
    df_temp['Tw_volume'] = np.where(
        df_temp['Tw_sentiment_score'] != 0,
        np.random.lognormal(8, 1, n_days),
        0
    )
    
    # Reddit sentiment
    df_temp['Rd_sentiment_score'] = np.where(
        np.random.random(n_days) < sentiment_prob,
        np.random.normal(0, 0.3, n_days),
        0
    )
    df_temp['Rd_mentions'] = np.where(
        df_temp['Rd_sentiment_score'] != 0,
        np.random.poisson(50, n_days),
        0
    )
    
    # News sentiment (S&P 500 related)
    df_temp['Nw_SP500_sentiment'] = np.where(
        np.random.random(n_days) < sentiment_prob,
        np.random.normal(0, 0.4, n_days),
        0
    )
    df_temp['Nw_SP500_volume'] = np.where(
        df_temp['Nw_SP500_sentiment'] != 0,
        np.random.lognormal(6, 0.8, n_days),
        0
    )
    
    # Market regime indicators
    df_temp['VIX_proxy'] = np.abs(np.random.normal(20, 8, n_days))
    df_temp['Market_regime'] = np.where(df_temp['VIX_proxy'] > 25, 1, 0)  # High volatility
    
    # Target (next day close)
    df_temp['Target'] = df_temp['Close'].shift(-1)
    
    # Remove rows with NaN targets (last row)
    df_temp = df_temp.dropna(subset=['Target'])
    
    # Fill NaN values for technical indicators
    df_temp = df_temp.fillna(method='bfill').fillna(method='ffill')
    
    return df_temp


def main():
    """Create sample data files for all tickers."""
    tickers = ['AAPL', 'AMZN', 'MSFT', 'TSLA', 'AMD']
    
    for ticker in tickers:
        print(f"Creating sample data for {ticker}...")
        df = create_sample_ticker_data(ticker)
        df.to_csv(f"sample_data/{ticker}_input.csv", index=False)
        print(f"Saved {len(df)} rows to sample_data/{ticker}_input.csv")
    
    print("Sample data creation completed!")


if __name__ == "__main__":
    main()