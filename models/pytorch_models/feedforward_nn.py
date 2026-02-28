"""Feedforward Neural Network model implementation for QuantSandbox.

This module implements a feedforward neural network using PyTorch for regression tasks.
"""

import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Optional, Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

from models.base.base_model import RegressionModel

class FeedforwardNN(RegressionModel):
    """Feedforward Neural Network regression model implementation.
    
    Attributes:
        model (nn.Module): The PyTorch neural network model
        config (dict): Configuration parameters
        model_name (str): Name of the model
        model_dir (str): Directory to save models
        metrics (dict): Dictionary to store evaluation metrics
        scaler (StandardScaler): Feature scaler
        device (torch.device): Device to use for training (CPU/GPU)
    """
    
    def __init__(self, config_path: str = "config.yaml", model_name: str = "feedforward_nn"):
        """Initialize FeedforwardNN model.
        
        Args:
            config_path (str): Path to configuration file
            model_name (str): Name of the model
        """
        super().__init__(config_path, model_name)
        self._init_params()
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        if self.verbose >= 1:
            print(f"Using device: {self.device}")
    
    def _init_params(self) -> None:
        """Initialize model parameters from config."""
        self.params = self.config.get('models', {}).get('pytorch', {}).get('feedforward', {})
        self.random_state = self.config.get('models', {}).get('base', {}).get('random_state', 42)
        self.verbose = self.config.get('models', {}).get('base', {}).get('verbose', 1)
        
        # Set random seeds for reproducibility
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)
    
    def _build_model(self, input_dim: int) -> nn.Module:
        """Build the feedforward neural network architecture.
        
        Args:
            input_dim (int): Number of input features
            
        Returns:
            nn.Module: PyTorch neural network model
        """
        hidden_layers = self.params.get('hidden_layers', [256, 128, 64])
        dropout_rate = self.params.get('dropout_rate', 0.2)
        
        # Create layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        return nn.Sequential(*layers)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> None:
        """Train the feedforward neural network model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (Optional[pd.DataFrame]): Validation features
            y_val (Optional[pd.Series]): Validation target
        """
        if self.verbose >= 1:
            print(f"Training Feedforward Neural Network model...")
            print(f"Training features shape: {X_train.shape}")
            print(f"Training target shape: {y_train.shape}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # Create DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.params.get('batch_size', 64),
            shuffle=True,
            drop_last=True
        )
        
        # Build model
        self.model = self._build_model(X_train.shape[1]).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.params.get('learning_rate', 0.001)
        )
        
        # Early stopping configuration
        early_stopping_patience = self.params.get('early_stopping_patience', 10)
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        # Training loop
        num_epochs = self.params.get('epochs', 50)
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            self.model.train()
            total_train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * batch_X.size(0)
            
            avg_train_loss = total_train_loss / len(train_loader.dataset)
            train_losses.append(avg_train_loss)
            
            # Validation
            val_loss = None
            if X_val is not None and y_val is not None:
                val_loss = self._validate(X_val, y_val, criterion)
                val_losses.append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    # Save best model
                    self.save_model(f"{self.model_name}_best.pkl")
                else:
                    epochs_without_improvement += 1
                    
                    if epochs_without_improvement >= early_stopping_patience:
                        if self.verbose >= 1:
                            print(f"Early stopping at epoch {epoch+1}")
                        # Load best model
                        self.load_model(os.path.join(self.model_dir, f"{self.model_name}_best.pkl"))
                        break
            
            if self.verbose >= 1 and (epoch + 1) % 10 == 0:
                if val_loss is not None:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}")
                else:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}")
        
        if self.verbose >= 1:
            print("Feedforward Neural Network model training completed.")
        
        # Plot training history
        if self.verbose >= 1 and len(val_losses) > 0:
            self._plot_training_history(train_losses, val_losses)
        
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
    
    def _validate(self, X_val: pd.DataFrame, y_val: pd.Series, criterion: nn.Module) -> float:
        """Validate the model on the validation set.
        
        Args:
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation target
            criterion (nn.Module): Loss function
            
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        
        X_val_scaled = self.scaler.transform(X_val)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_val_tensor)
            loss = criterion(outputs, y_val_tensor)
        
        return loss.item()
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained feedforward neural network.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            np.ndarray: Predicted values
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        self.model.eval()
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Convert to PyTorch tensor
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = outputs.cpu().numpy().flatten()
        
        return predictions
    
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
    
    def _plot_training_history(self, train_losses: List[float], val_losses: List[float]) -> None:
        """Plot training and validation loss history.
        
        Args:
            train_losses (List[float]): Training losses per epoch
            val_losses (List[float]): Validation losses per epoch
        """
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training and Validation Loss History')
        plt.legend()
        plt.grid(True)
        
        # Save plot to file
        plot_dir = 'plots'
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{self.model_name}_training_history.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose >= 1:
            print(f"Training history plot saved to: {plot_path}")


def main():
    """Main function to test Feedforward Neural Network model."""
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
    model = FeedforwardNN(model_name='test_feedforward_nn')
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
    
    # Save model
    model.save_model()


if __name__ == "__main__":
    main()
