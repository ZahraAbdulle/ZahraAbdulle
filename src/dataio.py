"""
Data I/O module for TFT hybrid forecaster.
Handles loading ticker CSV files, enforces fixed Train/Val/Test splits,
and applies train-only scaling with sentiment zero-preservation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List, Any
import json
import os


class SentimentPreservingScaler:
    """Custom scaler that preserves sentiment zeros while scaling other features."""
    
    def __init__(self):
        self.scalers = {}
        self.sentiment_columns = []
        
    def fit(self, X: pd.DataFrame, sentiment_prefixes: List[str] = ['Tw_', 'Rd_', 'Nw_SP500_']):
        """Fit scalers on training data, identifying sentiment columns."""
        self.sentiment_columns = []
        for col in X.columns:
            for prefix in sentiment_prefixes:
                if col.startswith(prefix):
                    self.sentiment_columns.append(col)
                    break
        
        # Fit scalers for all columns
        for col in X.columns:
            if col in self.sentiment_columns:
                # For sentiment columns, only scale non-zero values
                non_zero_mask = X[col] != 0
                if non_zero_mask.sum() > 1:  # Need at least 2 non-zero values
                    scaler = StandardScaler()
                    scaler.fit(X.loc[non_zero_mask, [col]])
                    self.scalers[col] = scaler
            else:
                # Standard scaling for other columns
                scaler = StandardScaler()
                scaler.fit(X[[col]])
                self.scalers[col] = scaler
                
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data preserving sentiment zeros."""
        X_scaled = X.copy()
        
        for col in X.columns:
            if col in self.scalers:
                if col in self.sentiment_columns:
                    # Only scale non-zero values for sentiment columns
                    non_zero_mask = X[col] != 0
                    if non_zero_mask.sum() > 0:
                        X_scaled.loc[non_zero_mask, col] = self.scalers[col].transform(
                            X.loc[non_zero_mask, [col]]
                        ).flatten()
                else:
                    # Standard scaling for other columns
                    X_scaled[col] = self.scalers[col].transform(X[[col]]).flatten()
                    
        return X_scaled
    
    def fit_transform(self, X: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, **kwargs).transform(X)
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform scaled data."""
        X_orig = X.copy()
        
        for col in X.columns:
            if col in self.scalers:
                if col in self.sentiment_columns:
                    # Only inverse scale non-zero values for sentiment columns
                    non_zero_mask = X[col] != 0
                    if non_zero_mask.sum() > 0:
                        X_orig.loc[non_zero_mask, col] = self.scalers[col].inverse_transform(
                            X.loc[non_zero_mask, [col]]
                        ).flatten()
                else:
                    # Standard inverse scaling for other columns
                    X_orig[col] = self.scalers[col].inverse_transform(X[[col]]).flatten()
                    
        return X_orig


class DataLoader:
    """Handles data loading and preprocessing for TFT hybrid forecaster."""
    
    def __init__(self, lookback: int = 60, horizon: int = 1, random_seed: int = 42):
        self.lookback = lookback
        self.horizon = horizon
        self.random_seed = random_seed
        self.scaler = None
        
        # Fixed date splits
        self.train_start = '2021-02-03'
        self.train_end = '2022-12-30'
        self.val_start = '2023-01-03'
        self.val_end = '2023-05-31'
        self.test_start = '2023-06-01'
        self.test_end = '2023-12-28'
        
        np.random.seed(random_seed)
        
    def load_ticker_data(self, ticker: str, data_path: str = '.') -> pd.DataFrame:
        """Load ticker CSV file with expected schema."""
        filepath = os.path.join(data_path, f"{ticker}_input.csv")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
            
        df = pd.read_csv(filepath)
        
        # Ensure date column is datetime
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Validate expected columns exist
        required_cols = ['date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Target']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        return df
    
    def create_splits(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create fixed train/val/test splits."""
        train_df = df[(df['date'] >= self.train_start) & (df['date'] <= self.train_end)].copy()
        val_df = df[(df['date'] >= self.val_start) & (df['date'] <= self.val_end)].copy()
        test_df = df[(df['date'] >= self.test_start) & (df['date'] <= self.test_end)].copy()
        
        # Validate test set has 146 days as required
        if len(test_df) != 146:
            print(f"Warning: Test set has {len(test_df)} days, expected 146")
            
        return train_df, val_df, test_df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare feature columns for TFT."""
        # Exclude non-feature columns
        exclude_cols = ['date', 'ticker', 'Target']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Create time-based features
        df = df.copy()
        df['time_idx'] = range(len(df))
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        
        # Add time features to feature list
        time_features = ['day_of_week', 'month', 'quarter']
        feature_cols.extend(time_features)
        
        return df, feature_cols
    
    def fit_scaler(self, train_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
        """Fit scaler on training data only."""
        self.scaler = SentimentPreservingScaler()
        
        # Fit only on training features (exclude time_idx and categorical time features)
        scale_cols = [col for col in feature_cols if col not in ['time_idx', 'day_of_week', 'month', 'quarter']]
        self.scaler.fit(train_df[scale_cols])
        
        # Create scaler summary
        scaler_summary = {
            'sentiment_columns': self.scaler.sentiment_columns,
            'scaled_columns': scale_cols,
            'train_start': self.train_start,
            'train_end': self.train_end,
            'lookback': self.lookback,
            'horizon': self.horizon,
            'random_seed': self.random_seed
        }
        
        return scaler_summary
    
    def transform_data(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Transform data using fitted scaler."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
            
        df_scaled = df.copy()
        
        # Scale only the appropriate columns
        scale_cols = [col for col in feature_cols if col not in ['time_idx', 'day_of_week', 'month', 'quarter']]
        df_scaled[scale_cols] = self.scaler.transform(df[scale_cols])
        
        return df_scaled
    
    def create_sequences(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Create sequences for TFT with lookback and horizon."""
        sequences = []
        
        for i in range(self.lookback, len(df) - self.horizon + 1):
            seq = {
                'time_idx': df.iloc[i]['time_idx'],
                'date': df.iloc[i]['date'],
                'ticker': df.iloc[i]['ticker'],
                'target': df.iloc[i + self.horizon - 1]['Target']  # horizon=1 means next day
            }
            
            # Add features at current timestep
            for col in feature_cols:
                seq[col] = df.iloc[i][col]
                
            sequences.append(seq)
        
        return pd.DataFrame(sequences)
    
    def prepare_ticker_data(self, ticker: str, data_path: str = '.') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], List[str]]:
        """Complete data preparation pipeline for a ticker."""
        # Load raw data
        df = self.load_ticker_data(ticker, data_path)
        
        # Create features
        df, feature_cols = self.prepare_features(df)
        
        # Create splits
        train_df, val_df, test_df = self.create_splits(df)
        
        # Fit scaler on training data
        scaler_summary = self.fit_scaler(train_df, feature_cols)
        
        # Transform all splits
        train_df = self.transform_data(train_df, feature_cols)
        val_df = self.transform_data(val_df, feature_cols)
        test_df = self.transform_data(test_df, feature_cols)
        
        # Create sequences
        train_sequences = self.create_sequences(train_df, feature_cols)
        val_sequences = self.create_sequences(val_df, feature_cols)
        test_sequences = self.create_sequences(test_df, feature_cols)
        
        return train_sequences, val_sequences, test_sequences, scaler_summary, feature_cols


def save_scaler_summary(scaler_summary: Dict[str, Any], filepath: str):
    """Save scaler summary to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(scaler_summary, f, indent=2)


def save_features_used(feature_cols: List[str], filepath: str):
    """Save list of features used to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    features_dict = {
        'features_used': feature_cols,
        'num_features': len(feature_cols)
    }
    with open(filepath, 'w') as f:
        json.dump(features_dict, f, indent=2)