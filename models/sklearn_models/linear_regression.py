"""Linear Regression model implementation for QuantSandbox.

This module implements a Linear Regression model using scikit-learn.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.preprocessing import StandardScaler

from models.base.base_model import RegressionModel

class LinearRegression(RegressionModel):
    """Linear Regression model implementation.
    
    Attributes:
        model (SklearnLinearRegression): Scikit-learn LinearRegression model
        scaler (StandardScaler): Feature scaler
        params (dict): Model parameters
    """
    
    def __init__(self, config_path: str = "config.yaml", model_name: str = "linear_regression"):
        """Initialize LinearRegression model.
        
        Args:
            config_path (str): Path to configuration file
            model_name (str): Name of the model
        """
        super().__init__(config_path, model_name)
        self._init_params()
        self.model = None
        self.scaler = StandardScaler()
    
    def _init_params(self) -> None:
        """Initialize model parameters from config."""
        self.params = self.config.get('models', {}).get('sklearn', {}).get('linear_regression', {})
        self.random_state = self.config.get('models', {}).get('base', {}).get('random_state', 42)
        self.verbose = self.config.get('models', {}).get('base', {}).get('verbose', 1)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> None:
        """Train the Linear Regression model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (Optional[pd.DataFrame]): Validation features
            y_val (Optional[pd.Series]): Validation target
        """
        if self.verbose >= 1:
            print(f"Training Linear Regression model...")
            print(f"Training features shape: {X_train.shape}")
            print(f"Training target shape: {y_train.shape}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Initialize and train model
        self.model = SklearnLinearRegression(
            fit_intercept=self.params.get('fit_intercept', True),
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        if self.verbose >= 1:
            print("Linear Regression model training completed.")
        
        # Evaluate on training set
        train_metrics = self.evaluate(X_train, y_train)
        if self.verbose >= 1:
            print(f"\nTraining set metrics:")
            self._print_metrics(train_metrics)
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            if self.verbose >= 1:
                print(f"\nValidation set metrics:")
                self._print_metrics(val_metrics)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained Linear Regression model.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            np.ndarray: Predicted values
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        return self.model.predict(X_scaled)
    
    def _print_metrics(self, metrics: Dict[str, float]) -> None:
        """Print evaluation metrics in a readable format.
        
        Args:
            metrics (Dict[str, float]): Evaluation metrics
        """
        for metric_name, metric_value in metrics.items():
            if 'correlation' in metric_name or 'win_rate' in metric_name or 'r_squared' in metric_name:
                print(f"  {metric_name}: {metric_value:.4f}")
            elif 'return' in metric_name:
                print(f"  {metric_name}: {metric_value:.6f}")
            else:
                print(f"  {metric_name}: {metric_value:.6f}")
    
    def get_feature_importance(self, feature_names: Optional[list] = None) -> pd.DataFrame:
        """Get feature importance from the trained model.
        
        Args:
            feature_names (Optional[list]): List of feature names
            
        Returns:
            pd.DataFrame: Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.model.coef_))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': self.model.coef_,
            'abs_coefficient': np.abs(self.model.coef_)
        })
        
        # Sort by absolute coefficient descending
        importance_df = importance_df.sort_values(by='abs_coefficient', ascending=False).reset_index(drop=True)
        
        return importance_df


def main():
    """Main function to test Linear Regression model."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    from utils.data_loader import DataLoader
    
    # Create data loader
    loader = DataLoader()
    
    # Load a few stocks for testing
    test_stocks = ['000001.SZ', '000002.SZ', '000004.SZ']
    loader.load_stock_data(test_stocks)
    
    # Get combined data
    combined_df = loader.get_combined_data()
    
    # Split data
    train_df, val_df, test_df = loader.split_data(combined_df)
    
    # Get features and target
    X_train, y_train = loader.get_features_and_target(train_df)
    X_val, y_val = loader.get_features_and_target(val_df)
    
    # Initialize and train model
    model = LinearRegression(model_name='test_linear_regression')
    model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    X_test, y_test = loader.get_features_and_target(test_df)
    test_metrics = model.evaluate(X_test, y_test)
    print(f"\nTest set metrics:")
    model._print_metrics(test_metrics)
    
    # Evaluate top-N performance
    print(f"\nTop-N performance:")
    top_n_metrics = model.evaluate_top_n(X_test, y_test)
    model._print_metrics(top_n_metrics)
    
    # Get feature importance
    feature_importance = model.get_feature_importance(X_train.columns)
    print(f"\nTop 10 feature importance:")
    print(feature_importance.head(10))
    
    # Save model
    model.save_model()


if __name__ == "__main__":
    main()
