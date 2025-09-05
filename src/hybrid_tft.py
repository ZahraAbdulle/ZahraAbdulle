"""
Temporal Fusion Transformer (TFT) hybrid forecaster main module.
Handles training and inference CLI for stock price forecasting.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Tuple

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import RMSE, MAE, MAPE

from dataio import DataLoader, save_scaler_summary, save_features_used


class TFTHybridForecaster:
    """TFT-based hybrid forecaster for stock price prediction."""
    
    def __init__(self, max_epochs: int = 100, gpus: int = 0, random_seed: int = 42):
        self.max_epochs = max_epochs
        self.gpus = gpus
        self.random_seed = random_seed
        
        # Set seeds for reproducibility
        pl.seed_everything(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        self.model = None
        self.trainer = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.training_dataset = None
        self.validation_dataset = None
        
    def prepare_datasets(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                        feature_cols: List[str], max_encoder_length: int = 60,
                        max_prediction_length: int = 1) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
        """Prepare TFT datasets for training and validation."""
        
        # Identify categorical vs continuous features
        categorical_features = ['day_of_week', 'month', 'quarter']
        continuous_features = [col for col in feature_cols if col not in categorical_features + ['time_idx']]
        
        # Create training dataset
        self.training_dataset = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target="target",
            group_ids=["ticker"],
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=["ticker"],
            time_varying_known_categoricals=categorical_features,
            time_varying_known_reals=continuous_features,
            time_varying_unknown_reals=["target"],
            target_normalizer=GroupNormalizer(groups=["ticker"], transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=False,
        )
        
        # Create validation dataset
        self.validation_dataset = TimeSeriesDataSet.from_dataset(
            self.training_dataset, 
            val_df, 
            predict=True, 
            stop_randomization=True
        )
        
        return self.training_dataset, self.validation_dataset
    
    def create_dataloaders(self, batch_size: int = 64, num_workers: int = 0):
        """Create data loaders for training and validation."""
        self.train_dataloader = self.training_dataset.to_dataloader(
            train=True, batch_size=batch_size, num_workers=num_workers
        )
        self.val_dataloader = self.validation_dataset.to_dataloader(
            train=False, batch_size=batch_size * 10, num_workers=num_workers
        )
        
        return self.train_dataloader, self.val_dataloader
    
    def create_model(self, learning_rate: float = 0.03, hidden_size: int = 16, 
                    attention_head_size: int = 1, dropout: float = 0.1,
                    hidden_continuous_size: int = 8) -> TemporalFusionTransformer:
        """Create TFT model with specified hyperparameters."""
        
        self.model = TemporalFusionTransformer.from_dataset(
            self.training_dataset,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=hidden_continuous_size,
            output_size=7,  # 7 quantiles
            loss=RMSE(),
            reduce_on_plateau_patience=4,
        )
        
        return self.model
    
    def train_model(self, output_dir: str, early_stopping_patience: int = 10) -> Dict[str, Any]:
        """Train the TFT model with early stopping."""
        
        # Setup callbacks
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=early_stopping_patience,
            verbose=False,
            mode="min",
        )
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=output_dir,
            filename="best-checkpoint",
            save_top_k=1,
            verbose=False,
            monitor="val_loss",
            mode="min",
        )
        
        # Setup logger
        logger = CSVLogger(output_dir, name="tft_training")
        
        # Create trainer
        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator="auto",
            devices=1 if self.gpus > 0 else None,
            enable_model_summary=True,
            gradient_clip_val=0.1,
            callbacks=[early_stop_callback, checkpoint_callback],
            logger=logger,
            deterministic=True,
        )
        
        # Train model
        self.trainer.fit(
            self.model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader,
        )
        
        # Load best model
        best_model_path = checkpoint_callback.best_model_path
        self.model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        
        # Create training summary
        training_summary = {
            'best_val_loss': float(checkpoint_callback.best_model_score),
            'total_epochs': self.trainer.current_epoch + 1,
            'best_model_path': best_model_path,
            'early_stopped': early_stop_callback.stopped_epoch > 0,
            'stopped_epoch': early_stop_callback.stopped_epoch if early_stop_callback.stopped_epoch > 0 else None
        }
        
        return training_summary
    
    def generate_predictions(self, dataset: TimeSeriesDataSet) -> pd.DataFrame:
        """Generate predictions using trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
            
        # Create data loader for predictions
        dataloader = dataset.to_dataloader(train=False, batch_size=64, num_workers=0)
        
        # Generate predictions
        predictions = self.model.predict(dataloader, return_y=True)
        
        # Convert to dataframe
        pred_df = pd.DataFrame({
            'prediction': predictions.output.squeeze().detach().cpu().numpy(),
            'actual': predictions.y.squeeze().detach().cpu().numpy()
        })
        
        return pred_df
    
    def plot_training_curves(self, output_dir: str, save_path: str):
        """Plot and save training curves."""
        # Read training logs
        log_path = os.path.join(output_dir, "tft_training", "version_0", "metrics.csv")
        
        if not os.path.exists(log_path):
            print(f"Warning: Training log not found at {log_path}")
            return
            
        logs = pd.read_csv(log_path)
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot training and validation loss
        train_loss = logs.dropna(subset=['train_loss'])
        val_loss = logs.dropna(subset=['val_loss'])
        
        ax1.plot(train_loss['epoch'], train_loss['train_loss'], label='Training Loss', alpha=0.8)
        ax1.plot(val_loss['epoch'], val_loss['val_loss'], label='Validation Loss', alpha=0.8)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot learning rate (if available)
        if 'lr-AdamW' in logs.columns:
            lr_data = logs.dropna(subset=['lr-AdamW'])
            ax2.plot(lr_data['epoch'], lr_data['lr-AdamW'], alpha=0.8)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Learning Rate\nData Not Available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Learning Rate Schedule')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to: {save_path}")


def create_run_config(args: argparse.Namespace, ticker: str) -> Dict[str, Any]:
    """Create run configuration dictionary."""
    return {
        'ticker': ticker,
        'model_type': 'TemporalFusionTransformer',
        'max_epochs': args.max_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'hidden_size': args.hidden_size,
        'attention_head_size': args.attention_head_size,
        'dropout': args.dropout,
        'hidden_continuous_size': args.hidden_continuous_size,
        'early_stopping_patience': args.early_stopping_patience,
        'max_encoder_length': args.lookback,
        'max_prediction_length': args.horizon,
        'random_seed': args.random_seed,
        'gpus': args.gpus,
        'timestamp': datetime.now().isoformat()
    }


def create_environment_info() -> str:
    """Create environment information string."""
    import platform
    import torch
    import pytorch_lightning as pl
    
    env_info = f"""Environment Information:
Python Version: {platform.python_version()}
Platform: {platform.platform()}
PyTorch Version: {torch.__version__}
PyTorch Lightning Version: {pl.__version__}
CUDA Available: {torch.cuda.is_available()}
CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}
Device Count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}
"""
    return env_info


def train_ticker(args: argparse.Namespace, ticker: str):
    """Train TFT model for a specific ticker."""
    print(f"\n{'='*60}")
    print(f"Training TFT model for {ticker}")
    print(f"{'='*60}")
    
    # Create output directory
    output_dir = f"outputs/hybrid/{ticker}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data
    print("Loading and preparing data...")
    data_loader = DataLoader(
        lookback=args.lookback, 
        horizon=args.horizon, 
        random_seed=args.random_seed
    )
    
    try:
        train_df, val_df, test_df, scaler_summary, feature_cols = data_loader.prepare_ticker_data(
            ticker, args.data_path
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Skipping {ticker} - data file not found")
        return
    
    print(f"Data loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"Features: {len(feature_cols)} columns")
    
    # Save configuration and metadata
    run_config = create_run_config(args, ticker)
    
    with open(f"{output_dir}/run_config_hybrid_{ticker}.json", 'w') as f:
        json.dump(run_config, f, indent=2)
    
    save_features_used(feature_cols, f"{output_dir}/features_used_{ticker}.json")
    save_scaler_summary(scaler_summary, f"{output_dir}/scaler_summary_{ticker}.json")
    
    with open(f"{output_dir}/environment.txt", 'w') as f:
        f.write(create_environment_info())
    
    # Initialize forecaster
    forecaster = TFTHybridForecaster(
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        random_seed=args.random_seed
    )
    
    # Prepare datasets
    print("Preparing TFT datasets...")
    train_dataset, val_dataset = forecaster.prepare_datasets(
        train_df, val_df, feature_cols, 
        max_encoder_length=args.lookback,
        max_prediction_length=args.horizon
    )
    
    # Create data loaders
    train_loader, val_loader = forecaster.create_dataloaders(
        batch_size=args.batch_size,
        num_workers=0
    )
    
    # Create model
    print("Creating TFT model...")
    model = forecaster.create_model(
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        attention_head_size=args.attention_head_size,
        dropout=args.dropout,
        hidden_continuous_size=args.hidden_continuous_size
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train model
    print("Training model...")
    training_summary = forecaster.train_model(
        output_dir=output_dir,
        early_stopping_patience=args.early_stopping_patience
    )
    
    print(f"Training completed - Best val loss: {training_summary['best_val_loss']:.6f}")
    print(f"Total epochs: {training_summary['total_epochs']}")
    
    # Plot training curves
    forecaster.plot_training_curves(
        output_dir=output_dir,
        save_path=f"{output_dir}/training_curves_{ticker}.png"
    )
    
    # Save training summary
    with open(f"{output_dir}/training_summary_{ticker}.json", 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    print(f"Training artifacts saved to: {output_dir}")
    

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="TFT Hybrid Forecaster")
    
    # Data arguments
    parser.add_argument('--ticker', type=str, default='AAPL',
                       help='Ticker symbol to train (default: AAPL)')
    parser.add_argument('--data-path', type=str, default='.',
                       help='Path to data directory (default: current directory)')
    parser.add_argument('--lookback', type=int, default=60,
                       help='Lookback window length (default: 60)')
    parser.add_argument('--horizon', type=int, default=1,
                       help='Forecast horizon (default: 1)')
    
    # Model arguments
    parser.add_argument('--max-epochs', type=int, default=100,
                       help='Maximum training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--learning-rate', type=float, default=0.03,
                       help='Learning rate (default: 0.03)')
    parser.add_argument('--hidden-size', type=int, default=16,
                       help='Hidden size (default: 16)')
    parser.add_argument('--attention-head-size', type=int, default=1,
                       help='Attention head size (default: 1)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate (default: 0.1)')
    parser.add_argument('--hidden-continuous-size', type=int, default=8,
                       help='Hidden continuous size (default: 8)')
    parser.add_argument('--early-stopping-patience', type=int, default=10,
                       help='Early stopping patience (default: 10)')
    
    # System arguments
    parser.add_argument('--gpus', type=int, default=0,
                       help='Number of GPUs to use (default: 0)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--all-tickers', action='store_true',
                       help='Train models for all tickers (AAPL, AMZN, MSFT, TSLA, AMD)')
    
    args = parser.parse_args()
    
    # Determine tickers to process
    if args.all_tickers:
        tickers = ['AAPL', 'AMZN', 'MSFT', 'TSLA', 'AMD']
    else:
        tickers = [args.ticker]
    
    # Train models
    for ticker in tickers:
        try:
            train_ticker(args, ticker)
        except Exception as e:
            print(f"Error training {ticker}: {e}")
            continue
    
    print(f"\nTFT training completed for tickers: {', '.join(tickers)}")


if __name__ == "__main__":
    main()