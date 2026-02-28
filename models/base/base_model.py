"""Abstract base classes for all models in QuantSandbox.

This module defines the abstract base classes that all models must implement.
"""

import abc
import pickle
import os
import yaml
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

class BaseModel(abc.ABC):
    """Abstract base class for all models.
    
    Attributes:
        model (Any): The underlying model object
        config (dict): Configuration parameters
        model_name (str): Name of the model
        model_dir (str): Directory to save models
        metrics (dict): Dictionary to store evaluation metrics
    """
    
    def __init__(self, config_path: str = "config.yaml", model_name: str = "base_model"):
        """Initialize BaseModel.
        
        Args:
            config_path (str): Path to configuration file
            model_name (str): Name of the model
        """
        self.config = self._load_config(config_path)
        self.model_name = model_name
        self.model_dir = self.config.get('logging', {}).get('model_save_dir', 'saved_models/')
        self.model = None
        self.metrics = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file.
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            dict: Configuration parameters
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    @abc.abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> None:
        """Train the model on the provided data.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (Optional[pd.DataFrame]): Validation features
            y_val (Optional[pd.Series]): Validation target
        """
        pass
    
    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            np.ndarray: Predicted values
        """
        pass
    
    @abc.abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate the model on the provided data.
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series): True target values
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        pass
    
    def save_model(self, filename: Optional[str] = None) -> str:
        """Save the trained model to file.
        
        Args:
            filename (Optional[str]): Custom filename for the model
            
        Returns:
            str: Path to the saved model file
        """
        if filename is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model_name}_{timestamp}.pkl"
        
        filepath = os.path.join(self.model_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"Model saved to: {filepath}")
        return filepath
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from file.
        
        Args:
            filepath (str): Path to the saved model file
        """
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"Model loaded from: {filepath}")
    
    def get_metrics(self) -> Dict[str, float]:
        """Get the stored evaluation metrics.
        
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        return self.metrics.copy()
    
    def log_metric(self, metric_name: str, metric_value: float) -> None:
        """Log an evaluation metric.
        
        Args:
            metric_name (str): Name of the metric
            metric_value (float): Value of the metric
        """
        self.metrics[metric_name] = metric_value
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the model.
        
        Returns:
            Dict[str, any]: Model information
        """
        return {
            'model_name': self.model_name,
            'metrics': self.metrics,
            'model_dir': self.model_dir
        }


class RegressionModel(BaseModel):
    """Abstract base class for regression models.
    
    This class extends BaseModel with regression-specific functionality.
    """
    
    def __init__(self, config_path: str = "config.yaml", model_name: str = "regression_model"):
        """Initialize RegressionModel.
        
        Args:
            config_path (str): Path to configuration file
            model_name (str): Name of the model
        """
        super().__init__(config_path, model_name)
        self.regression_metrics = [
            'mean_absolute_error',
            'mean_squared_error',
            'root_mean_squared_error',
            'r_squared',
            'pearson_correlation',
            'spearman_correlation'
        ]
    
    @abc.abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> None:
        """Train the regression model on the provided data.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (Optional[pd.DataFrame]): Validation features
            y_val (Optional[pd.Series]): Validation target
        """
        pass
    
    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make regression predictions using the trained model.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            np.ndarray: Predicted values
        """
        pass
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate the regression model on the provided data.
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series): True target values
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        y_pred = self.predict(X)
        
        # Handle cases where predictions contain NaN values
        if np.isnan(y_pred).any():
            # Remove NaN predictions and corresponding actual values
            mask = ~np.isnan(y_pred)
            y_pred_clean = y_pred[mask]
            y_clean = y.iloc[np.where(mask)[0]].values if len(y) == len(y_pred) else y.values
            
            # If no valid predictions left, return empty metrics
            if len(y_pred_clean) == 0:
                metrics = {}
                for metric in self.regression_metrics:
                    metrics[metric] = np.nan
                return metrics
            
            metrics = self._calculate_regression_metrics(y_clean, y_pred_clean)
        else:
            metrics = self._calculate_regression_metrics(y, y_pred)
        
        # Update internal metrics
        self.metrics.update(metrics)
        
        return metrics
    
    def _calculate_regression_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate regression metrics.
        
        Args:
            y_true (pd.Series): True target values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            Dict[str, float]: Dictionary of regression metrics
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from scipy.stats import pearsonr, spearmanr
        
        metrics = {}
        
        # Basic regression metrics
        metrics['mean_absolute_error'] = mean_absolute_error(y_true, y_pred)
        metrics['mean_squared_error'] = mean_squared_error(y_true, y_pred)
        metrics['root_mean_squared_error'] = np.sqrt(metrics['mean_squared_error'])
        metrics['r_squared'] = r2_score(y_true, y_pred)
        
        # Correlation metrics
        pearson_corr, _ = pearsonr(y_true, y_pred)
        spearman_corr, _ = spearmanr(y_true, y_pred)
        
        metrics['pearson_correlation'] = pearson_corr
        metrics['spearman_correlation'] = spearman_corr
        
        return metrics
    
    def evaluate_top_n(self, X: pd.DataFrame, y: pd.Series, n_values: List[int] = [5, 10, 50, 100]) -> Dict[str, float]:
        """Evaluate top-N performance of the regression model.
        
        Args:
            X (pd.DataFrame): Input features
            y (pd.Series): True target values
            n_values (List[int]): List of N values to evaluate
            
        Returns:
            Dict[str, float]: Top-N evaluation metrics
        """
        y_pred = self.predict(X)
        
        # Create dataframe with predictions and actual values
        eval_df = pd.DataFrame({
            'y_true': y,
            'y_pred': y_pred,
            'actual_return': y
        })
        
        metrics = {}
        
        # Evaluate top-N performance
        for n in n_values:
            # Sort by predicted returns descending
            sorted_df = eval_df.sort_values(by='y_pred', ascending=False).head(n)
            
            # Calculate average actual return for top-N
            avg_return = sorted_df['actual_return'].mean()
            metrics[f'top_{n}_avg_return'] = avg_return
            
            # Calculate median actual return for top-N
            median_return = sorted_df['actual_return'].median()
            metrics[f'top_{n}_median_return'] = median_return
            
            # Calculate win rate (percentage of positive returns in top-N)
            win_rate = (sorted_df['actual_return'] > 0).mean()
            metrics[f'top_{n}_win_rate'] = win_rate
            
            # Calculate average return vs market
            market_avg_return = eval_df['actual_return'].mean()
            metrics[f'top_{n}_vs_market'] = avg_return - market_avg_return
        
        # Update internal metrics
        self.metrics.update(metrics)
        
        return metrics
