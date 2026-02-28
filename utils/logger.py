"""Logging utilities for QuantSandbox.

This module provides comprehensive logging functionality for model training,
evaluation, and experiment tracking.
"""

import os
import json
import time
import pandas as pd
from typing import Dict, Optional, Any
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler

class Logger:
    """Comprehensive logger for tracking model training and evaluation.
    
    Attributes:
        logger (logging.Logger): The underlying logger object
        log_dir (str): Directory to save log files
        experiment_name (str): Name of the current experiment
        metrics (Dict[str, Any]): Collected metrics
        hyperparameters (Dict[str, Any]): Experiment hyperparameters
    """
    
    def __init__(self, experiment_name: Optional[str] = None, log_dir: str = "logs"):
        """Initialize Logger.
        
        Args:
            experiment_name (Optional[str]): Name of the experiment
            log_dir (str): Directory to save log files
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.metrics = {}
        self.hyperparameters = {}
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.DEBUG)
        
        # Avoid duplicate handlers
        if self.logger.handlers:
            return
        
        # File handler with rotation
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, f"{self.experiment_name}.log"),
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        file_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        """Log experiment hyperparameters.
        
        Args:
            hyperparameters (Dict[str, Any]): Dictionary of hyperparameters
        """
        self.hyperparameters.update(hyperparameters)
        
        self.logger.info("Experiment Hyperparameters:")
        for key, value in hyperparameters.items():
            self.logger.info(f"  {key}: {value}")
        
        # Save hyperparameters to JSON file
        with open(os.path.join(self.log_dir, f"{self.experiment_name}_hyperparams.json"), 'w', encoding='utf-8') as f:
            json.dump(self.hyperparameters, f, indent=2, default=str)
    
    def log_metrics(self, metrics: Dict[str, float], phase: str = "training") -> None:
        """Log evaluation metrics.
        
        Args:
            metrics (Dict[str, float]): Dictionary of metrics
            phase (str): Phase name (training, validation, test, etc.)
        """
        if phase not in self.metrics:
            self.metrics[phase] = []
        
        # Add timestamp to metrics
        timestamped_metrics = {
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        self.metrics[phase].append(timestamped_metrics)
        
        self.logger.info(f"{phase.capitalize()} Metrics:")
        for key, value in metrics.items():
            if 'correlation' in key or 'win_rate' in key or 'r_squared' in key:
                self.logger.info(f"  {key}: {value:.4f}")
            elif 'return' in key:
                self.logger.info(f"  {key}: {value:.6f}")
            else:
                self.logger.info(f"  {key}: {value:.6f}")
        
        # Save metrics to JSON file
        with open(os.path.join(self.log_dir, f"{self.experiment_name}_metrics.json"), 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        # Save metrics to CSV file for easy analysis
        for phase_name, phase_metrics in self.metrics.items():
            df = pd.DataFrame(phase_metrics)
            df.to_csv(os.path.join(self.log_dir, f"{self.experiment_name}_{phase_name}_metrics.csv"), 
                      index=False, encoding='utf-8')
    
    def log_message(self, message: str, level: str = "info") -> None:
        """Log a general message.
        
        Args:
            message (str): Message to log
            level (str): Logging level (debug, info, warning, error, critical)
        """
        if level.lower() == "debug":
            self.logger.debug(message)
        elif level.lower() == "info":
            self.logger.info(message)
        elif level.lower() == "warning":
            self.logger.warning(message)
        elif level.lower() == "error":
            self.logger.error(message)
        elif level.lower() == "critical":
            self.logger.critical(message)
        else:
            self.logger.info(message)
    
    def log_training_start(self, model_name: str, features_shape: tuple) -> None:
        """Log the start of training.
        
        Args:
            model_name (str): Name of the model being trained
            features_shape (tuple): Shape of the input features
        """
        self.logger.info("=" * 80)
        self.logger.info(f"Starting training: {model_name}")
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Features shape: {features_shape}")
        self.logger.info("=" * 80)
    
    def log_training_end(self, training_time: float = None) -> None:
        """Log the end of training.
        
        Args:
            training_time (float): Total training time in seconds
        """
        self.logger.info("=" * 80)
        self.logger.info("Training completed!")
        if training_time is not None:
            minutes, seconds = divmod(training_time, 60)
            self.logger.info(f"Total training time: {int(minutes)}m {seconds:.2f}s")
        self.logger.info("=" * 80)
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float] = None,
                  metrics: Optional[Dict[str, float]] = None) -> None:
        """Log epoch results.
        
        Args:
            epoch (int): Epoch number
            train_loss (float): Training loss
            val_loss (Optional[float]): Validation loss
            metrics (Optional[Dict[str, float]]): Additional metrics
        """
        message = f"Epoch [{epoch}]: Train Loss = {train_loss:.6f}"
        if val_loss is not None:
            message += f", Val Loss = {val_loss:.6f}"
        
        self.logger.info(message)
        
        if metrics:
            self.logger.info("  Additional metrics:")
            for key, value in metrics.items():
                self.logger.info(f"    {key}: {value:.4f}")
    
    def save_experiment_summary(self) -> str:
        """Save a comprehensive experiment summary.
        
        Returns:
            str: Path to the saved summary file
        """
        summary = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'hyperparameters': self.hyperparameters,
            'metrics': self.metrics,
            'log_files': {
                'main_log': os.path.join(self.log_dir, f"{self.experiment_name}.log"),
                'hyperparameters': os.path.join(self.log_dir, f"{self.experiment_name}_hyperparams.json"),
                'metrics': os.path.join(self.log_dir, f"{self.experiment_name}_metrics.json")
            }
        }
        
        summary_path = os.path.join(self.log_dir, f"{self.experiment_name}_summary.json")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Experiment summary saved to: {summary_path}")
        
        return summary_path
    
    def get_latest_experiments(self, num_experiments: int = 10) -> pd.DataFrame:
        """Get information about recent experiments.
        
        Args:
            num_experiments (int): Number of recent experiments to retrieve
            
        Returns:
            pd.DataFrame: DataFrame with experiment information
        """
        experiment_files = []
        
        for filename in os.listdir(self.log_dir):
            if filename.endswith('_summary.json'):
                filepath = os.path.join(self.log_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                        experiment_files.append(summary)
                except:
                    continue
        
        # Sort by timestamp descending
        experiment_files.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Create DataFrame
        df = pd.DataFrame(experiment_files)
        
        if not df.empty:
            # Extract key information
            df['model_name'] = df['hyperparameters'].apply(lambda x: x.get('model_name', 'unknown'))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            df['time'] = df['timestamp'].dt.time
            
            # Get best metrics
            df['best_val_r2'] = df['metrics'].apply(
                lambda x: max([m.get('r_squared', -float('inf')) for m in x.get('validation', [])]) if x.get('validation') else None
            )
            df['best_test_r2'] = df['metrics'].apply(
                lambda x: max([m.get('r_squared', -float('inf')) for m in x.get('test', [])]) if x.get('test') else None
            )
        
        return df.head(num_experiments)


class ModelLogger:
    """Specialized logger for model training and evaluation.
    
    This class provides a simplified interface for logging during model training.
    """
    
    def __init__(self, model_name: str, experiment_name: Optional[str] = None):
        """Initialize ModelLogger.
        
        Args:
            model_name (str): Name of the model
            experiment_name (Optional[str]): Name of the experiment
        """
        self.logger = Logger(
            experiment_name or f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.model_name = model_name
        self.start_time = None
    
    def start_training(self, features_shape: tuple, hyperparameters: Optional[Dict[str, Any]] = None) -> None:
        """Start training logging.
        
        Args:
            features_shape (tuple): Shape of the input features
            hyperparameters (Optional[Dict[str, Any]]): Model hyperparameters
        """
        self.start_time = time.time()
        self.logger.log_training_start(self.model_name, features_shape)
        
        if hyperparameters:
            hyperparameters['model_name'] = self.model_name
            self.logger.log_hyperparameters(hyperparameters)
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float] = None,
                  metrics: Optional[Dict[str, float]] = None) -> None:
        """Log epoch results.
        
        Args:
            epoch (int): Epoch number
            train_loss (float): Training loss
            val_loss (Optional[float]): Validation loss
            metrics (Optional[Dict[str, float]]): Additional metrics
        """
        self.logger.log_epoch(epoch, train_loss, val_loss, metrics)
    
    def log_phase(self, phase: str, metrics: Dict[str, float]) -> None:
        """Log results for a specific phase.
        
        Args:
            phase (str): Phase name (training, validation, test)
            metrics (Dict[str, float]): Metrics for this phase
        """
        self.logger.log_metrics(metrics, phase)
    
    def end_training(self) -> None:
        """End training logging."""
        training_time = time.time() - self.start_time if self.start_time else None
        self.logger.log_training_end(training_time)
        self.logger.save_experiment_summary()
    
    def log_message(self, message: str, level: str = "info") -> None:
        """Log a general message.
        
        Args:
            message (str): Message to log
            level (str): Logging level
        """
        self.logger.log_message(message, level)


def get_logger(model_name: str, experiment_name: Optional[str] = None) -> ModelLogger:
    """Factory function to create a ModelLogger.
    
    Args:
        model_name (str): Name of the model
        experiment_name (Optional[str]): Name of the experiment
        
    Returns:
        ModelLogger: A specialized logger for model training
    """
    return ModelLogger(model_name, experiment_name)


# Initialize root logger for general use
root_logger = logging.getLogger('QuantSandbox')
root_logger.setLevel(logging.INFO)

# Add console handler if not already present
if not root_logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
