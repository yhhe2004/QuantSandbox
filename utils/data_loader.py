"""Data loading and preprocessing utilities for QuantSandbox.

This module provides functions to load stock data, preprocess it, create features,
and split into training, validation, and testing sets.
"""

import os
import yaml
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from datetime import datetime

class DataLoader:
    """Class for loading and preprocessing stock market data.
    
    Attributes:
        config (dict): Configuration parameters
        stock_basic (pd.DataFrame): Stock metadata
        all_stock_data (Dict[str, pd.DataFrame]): Loaded stock data
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize DataLoader with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.stock_basic = self._load_stock_basic()
        self.all_stock_data = {}
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file.
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            dict: Configuration parameters
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_stock_basic(self) -> pd.DataFrame:
        """Load stock metadata from CSV file.
        
        Returns:
            pd.DataFrame: Stock metadata
        """
        path = self.config['data']['stock_basic_path']
        return pd.read_csv(path, encoding='utf-8')
    
    def load_stock_data(self, stock_codes: Optional[List[str]] = None) -> None:
        """Load stock data from CSV files.
        
        Args:
            stock_codes (Optional[List[str]]): List of stock codes to load. If None, load all available stocks.
        """
        data_dir = self.config['data']['stock_data_dir']
        
        if stock_codes is None:
            # Load all stock files in the directory
            stock_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            stock_codes = [os.path.splitext(f)[0] for f in stock_files]
        
        for stock_code in stock_codes:
            file_path = os.path.join(data_dir, f"{stock_code}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, encoding='utf-8')
                df = self._preprocess_single_stock(df, stock_code)
                self.all_stock_data[stock_code] = df
                print(f"Loaded data for {stock_code}: {len(df)} records")
            else:
                print(f"Warning: File not found for {stock_code}")
    
    def _preprocess_single_stock(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """Preprocess single stock data.
        
        Args:
            df (pd.DataFrame): Raw stock data
            stock_code (str): Stock code
            
        Returns:
            pd.DataFrame: Preprocessed stock data
        """
        # Convert date column to datetime
        date_col = self.config['data']['date_column']
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by date
        df = df.sort_values(by=date_col).reset_index(drop=True)
        
        # Calculate daily return (close / open - 1)
        open_col = self.config['data']['open_column']
        close_col = self.config['data']['close_column']
        target_col = self.config['data']['target_column']
        
        df[target_col] = df[close_col] / df[open_col] - 1
        
        # Add stock code column
        df['stock_code'] = stock_code
        
        # Handle missing values
        df = self._handle_missing_data(df)
        
        # Create technical features
        df = self._create_technical_features(df)
        
        return df
    
    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data in stock data.
        
        Args:
            df (pd.DataFrame): Stock data with potential missing values
            
        Returns:
            pd.DataFrame: Data with missing values handled
        """
        # Forward fill missing values first
        df = df.ffill()
        # Then backward fill any remaining missing values
        df = df.bfill()
        # Drop any rows that still have missing values
        df = df.dropna()
        
        return df
    
    def _create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional technical features from existing data.
        
        Args:
            df (pd.DataFrame): Stock data with basic features
            
        Returns:
            pd.DataFrame: Data with additional technical features
        """
        open_col = self.config['data']['open_column']
        close_col = self.config['data']['close_column']
        
        # Create rolling statistics
        window_sizes = self.config['features']['rolling_window_sizes']
        
        for window in window_sizes:
            # Rolling mean of returns
            df[f'return_mean_{window}'] = df[self.config['data']['target_column']].rolling(window=window).mean()
            
            # Rolling standard deviation of returns
            df[f'return_std_{window}'] = df[self.config['data']['target_column']].rolling(window=window).std()
            
            # Rolling mean of open prices
            df[f'open_mean_{window}'] = df[open_col].rolling(window=window).mean()
            
            # Rolling mean of close prices
            df[f'close_mean_{window}'] = df[close_col].rolling(window=window).mean()
            
            # Price momentum (current price / price N days ago - 1)
            df[f'momentum_{window}'] = df[close_col] / df[close_col].shift(window) - 1
        
        # Drop rows with NaN values created by rolling windows
        df = df.dropna()
        
        return df
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into training, validation, and testing sets by time.
        
        Args:
            df (pd.DataFrame): Data to split
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, test sets
        """
        train_ratio, val_ratio, test_ratio = self.config['data']['train_val_test_split']
        
        # Sort by date to ensure time-based split
        date_col = self.config['data']['date_column']
        df = df.sort_values(by=date_col).reset_index(drop=True)
        
        # Calculate split indices
        total_rows = len(df)
        train_end = int(total_rows * train_ratio)
        val_end = train_end + int(total_rows * val_ratio)
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        print(f"Data split complete:")
        print(f"  Training set: {len(train_df)} records ({train_ratio*100:.1f}%)")
        print(f"  Validation set: {len(val_df)} records ({val_ratio*100:.1f}%)")
        print(f"  Test set: {len(test_df)} records ({test_ratio*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def get_combined_data(self) -> pd.DataFrame:
        """Get combined data from all loaded stocks.
        
        Returns:
            pd.DataFrame: Combined data from all stocks
        """
        if not self.all_stock_data:
            raise ValueError("No stock data loaded. Call load_stock_data() first.")
        
        return pd.concat(self.all_stock_data.values(), ignore_index=True)
    
    def get_features_and_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract features and target variable from data.
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target
        """
        target_col = self.config['data']['target_column']
        
        # Get all technical indicators from config
        technical_indicators = self.config['features']['technical_indicators']
        
        # Get rolling feature columns
        rolling_features = [col for col in df.columns if col.startswith(('return_mean_', 'return_std_', 
                                                                         'open_mean_', 'close_mean_', 
                                                                         'momentum_'))]
        
        # Combine all feature columns
        feature_cols = technical_indicators + rolling_features
        
        # Ensure all feature columns exist
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        return X, y


def main():
    """Main function to test data loading and preprocessing."""
    # Create data loader
    loader = DataLoader()
    
    # Load a few stocks for testing
    test_stocks = ['000001.SZ', '000002.SZ', '000004.SZ']
    loader.load_stock_data(test_stocks)
    
    # Get combined data
    combined_df = loader.get_combined_data()
    print(f"\nCombined data shape: {combined_df.shape}")
    print(f"Columns: {list(combined_df.columns)}")
    
    # Split data
    train_df, val_df, test_df = loader.split_data(combined_df)
    
    # Get features and target
    X_train, y_train = loader.get_features_and_target(train_df)
    print(f"\nTraining features shape: {X_train.shape}")
    print(f"Training target shape: {y_train.shape}")
    
    # Save sample data for testing
    sample_dir = 'data/sample'
    os.makedirs(sample_dir, exist_ok=True)
    combined_df.head(100).to_csv(os.path.join(sample_dir, 'sample_data.csv'), index=False, encoding='utf-8')
    print(f"\nSample data saved to {sample_dir}/sample_data.csv")


if __name__ == "__main__":
    main()
