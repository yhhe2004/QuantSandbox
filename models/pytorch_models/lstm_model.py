"""LSTM Model for return prediction in QuantSandbox.

This module implements an LSTM model using PyTorch for sequential time series
prediction of stock returns.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Optional, Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import os

from models.base.base_model import BaseModel, RegressionModel
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Optional, Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import os
from models.base.base_model import BaseModel, RegressionModel
from utils.logger import get_logger


class LSTMModel(RegressionModel):
    """LSTM (Long Short-Term Memory) model for return prediction.
    
    This model implements an LSTM network for sequential time series prediction
    of stock returns based on historical data and technical indicators.
    """
    
    def __init__(self, model_name: str = "lstm_model", params: Optional[Dict] = None, config_path: str = "config.yaml"):
        """Initialize LSTMModel.
        
        Args:
            model_name (str): Name of the model
            params (Optional[Dict]): Model parameters
                - 'input_size': Number of input features (default: 33)
                - 'hidden_size': Number of LSTM hidden units (default: 128)
                - 'num_layers': Number of LSTM layers (default: 2)
                - 'output_size': Number of output units (default: 1)
                - 'dropout_rate': Dropout rate (default: 0.2)
                - 'bidirectional': Whether to use bidirectional LSTM (default: False)
                - 'learning_rate': Learning rate (default: 0.001)
                - 'batch_size': Batch size for training (default: 32)
                - 'num_epochs': Number of training epochs (default: 50)
                - 'early_stopping_patience': Patience for early stopping (default: 10)
                - 'sequence_length': Length of input sequences (default: 20)
                - 'device': Device to use for training (default: 'auto' - uses GPU if available)
            config_path (str): Path to configuration file (default: "config.yaml")
        """
        super().__init__(config_path, model_name)
        self.params = params or {}
        
        # LSTM-specific parameters
        self.input_size = self.params.get('input_size', 33)
        self.hidden_size = self.params.get('hidden_size', 128)
        self.num_layers = self.params.get('num_layers', 2)
        self.output_size = self.params.get('output_size', 1)
        self.dropout_rate = self.params.get('dropout_rate', 0.2)
        self.bidirectional = self.params.get('bidirectional', False)
        self.sequence_length = self.params.get('sequence_length', 20)
        
        # Training parameters
        self.learning_rate = self.params.get('learning_rate', 0.001)
        self.batch_size = self.params.get('batch_size', 32)
        self.num_epochs = self.params.get('num_epochs', 50)
        self.early_stopping_patience = self.params.get('early_stopping_patience', 10)
        
        # Device configuration
        if self.params.get('device') == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.params.get('device', 'cpu'))
        
        # Initialize logger
        self.logger = get_logger(model_name=model_name, experiment_name=f"{model_name}_training")
        
        # Scaler for input features
        self.scaler = StandardScaler()
        
        # Build the model
        self.model = self._build_model()
        
        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training history
        self.train_loss_history = []
        self.val_loss_history = []
        
        # Plot directory
        self.plot_dir = self.config.get('logging', {}).get('plot_save_dir', 'plots/')
        os.makedirs(self.plot_dir, exist_ok=True)
    
    def _build_model(self) -> nn.Module:
        """Build the LSTM model architecture.
        
        Returns:
            nn.Module: LSTM model
        """
        model = nn.Sequential()
        
        # LSTM layers
        lstm_layer = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_rate if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        model.add_module('lstm', lstm_layer)
        
        # Dropout layer if needed
        if self.dropout_rate > 0 and self.num_layers == 1:
            model.add_module('dropout', nn.Dropout(self.dropout_rate))
        
        # Fully connected layer for output
        output_dim = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        model.add_module('fc', nn.Linear(output_dim, self.output_size))
        
        return model.to(self.device)
    
    def _create_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Create input sequences for LSTM.
        
        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features)
            y (Optional[np.ndarray]): Target values of shape (n_samples,)
            
        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: Sequences and corresponding targets
        """
        n_samples = len(X)
        sequences = []
        targets = []
        
        # Handle cases where sample size is less than sequence length
        if n_samples <= self.sequence_length:
            # Return empty arrays with appropriate shape
            sequences = np.array([]).reshape(0, self.sequence_length, X.shape[1])
            targets = np.array([]).reshape(0, 1) if y is not None else None
            return sequences, targets
        
        for i in range(self.sequence_length, n_samples):
            sequences.append(X[i-self.sequence_length:i])
            if y is not None:
                # Convert to numpy array before indexing if it's a pandas Series
                y_np = y.to_numpy() if hasattr(y, 'to_numpy') else y
                targets.append(y_np[i])
        
        sequences = np.array(sequences)
        targets = np.array(targets) if y is not None else None
        
        return sequences, targets
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Train the LSTM model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            X_val (Optional[np.ndarray]): Validation features
            y_val (Optional[np.ndarray]): Validation targets
        """
        # Log training start
        self.logger.start_training(
            features_shape=X_train.shape,
            hyperparameters=self.params
        )
        
        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        
        # Create sequences
        X_train_seq, y_train_seq = self._create_sequences(X_train_scaled, y_train)
        
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32).to(self.device)
            y_val_tensor = torch.tensor(y_val_seq, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # Create DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.logger.log_message(f"Training LSTM model on {self.device}")
        self.logger.log_message(f"Training sequences: {len(X_train_seq)}, Validation sequences: {len(X_val_seq) if X_val is not None else 0}")
        
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0
            
            # Iterate over batches
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs, _ = self.model[0](batch_X)  # Get LSTM output
                outputs = outputs[:, -1, :]  # Take the last time step output
                if len(self.model) > 1 and isinstance(self.model[1], nn.Dropout):
                    outputs = self.model[1](outputs)  # Apply dropout if present
                outputs = self.model[-1](outputs)  # Final dense layer
                
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
            
            # Calculate average training loss
            train_loss /= len(train_loader.dataset)
            self.train_loss_history.append(train_loss)
            
            # Calculate validation loss
            val_loss = None
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    outputs, _ = self.model[0](X_val_tensor)
                    outputs = outputs[:, -1, :]
                    if len(self.model) > 1 and isinstance(self.model[1], nn.Dropout):
                        outputs = self.model[1](outputs)
                    outputs = self.model[-1](outputs)
                    val_loss = self.criterion(outputs, y_val_tensor).item()
                
                self.val_loss_history.append(val_loss)
            
            # Log epoch results
            self.logger.log_epoch(epoch + 1, train_loss, val_loss)
            
            # Early stopping
            if X_val is not None and y_val is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_model()
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.early_stopping_patience:
                    self.logger.log_message(f"Early stopping at epoch {epoch + 1}", "info")
                    break
            else:
                # Save every epoch if no validation set
                if epoch % 5 == 0:
                    self.save_model()
        
        # Load best model
        if X_val is not None and y_val is not None:
            self.load_model()
        
        # Plot training history
        self._plot_training_history()
        
        # Log training metrics
        train_metrics = self.evaluate(X_train, y_train)
        self.logger.log_phase(phase="training", metrics=train_metrics)
        
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            self.logger.log_phase(phase="validation", metrics=val_metrics)
        
        self.logger.end_training()
        
        self.logger.log_message("LSTM model training completed.", "info")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            np.ndarray: Predicted values
        """
        if self.model is None:
            raise ValueError("Model not trained. Please call train() first.")
        
        # Convert to numpy array
        X_np = X.values
        
        # Scale features
        X_scaled = self.scaler.transform(X_np)
        
        # Create sequences
        X_seq, _ = self._create_sequences(X_scaled)
        
        # If no sequences can be created (not enough samples), return empty predictions
        if len(X_seq) == 0:
            return np.zeros((len(X),)) * np.nan
        
        # Convert to PyTorch tensor
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs, _ = self.model[0](X_tensor)
            outputs = outputs[:, -1, :]
            if len(self.model) > 1 and isinstance(self.model[1], nn.Dropout):
                outputs = self.model[1](outputs)
            outputs = self.model[-1](outputs)
            predictions = outputs.cpu().numpy().flatten()
        
        # The predictions correspond to the last time step of each sequence
        # We need to pad the predictions to match the original input length
        padded_predictions = np.full((len(X),), np.nan)
        padded_predictions[self.sequence_length:] = predictions
        
        return padded_predictions
    
    def save_model(self, filename: Optional[str] = None) -> str:
        """Save the trained model to file.
        
        Args:
            filename (Optional[str]): Custom filename for the model
            
        Returns:
            str: Path to the saved model file
        """
        if filename is None:
            filename = f"{self.model_name}_best.pkl"
        
        filepath = os.path.join(self.model_dir, filename)
        
        # Save model state
        save_dict = {
            'params': self.params,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_params': self.scaler.get_params()
        }
        
        # Save scaler attributes if they exist
        try:
            if hasattr(self.scaler, 'mean_'):
                save_dict['scaler_mean'] = self.scaler.mean_
            if hasattr(self.scaler, 'scale_'):
                save_dict['scaler_scale'] = self.scaler.scale_
            if hasattr(self.scaler, 'data_min_'):
                save_dict['scaler_data_min'] = self.scaler.data_min_
            if hasattr(self.scaler, 'data_max_'):
                save_dict['scaler_data_max'] = self.scaler.data_max_
            if hasattr(self.scaler, 'data_range_'):
                save_dict['scaler_data_range'] = self.scaler.data_range_
        except:
            pass
        
        torch.save(save_dict, filepath)
        self.logger.log_message(f"Model saved to: {filepath}", "info")
        
        return filepath
    
    def load_model(self, filepath: Optional[str] = None) -> None:
        """Load a trained model from file.
        
        Args:
            filepath (Optional[str]): Path to the saved model file
        """
        if filepath is None:
            filepath = os.path.join(self.model_dir, f"{self.model_name}_best.pkl")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load model with weights_only=False due to numpy objects in scaler
        device = torch.device('cpu') if not torch.cuda.is_available() else self.device
        save_dict = torch.load(filepath, map_location=device, weights_only=False)
        
        # Update parameters
        self.params.update(save_dict['params'])
        
        # Rebuild model
        self.model = self._build_model()
        self.model.load_state_dict(save_dict['model_state_dict'])
        
        # Initialize optimizer with saved state
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if 'optimizer_state_dict' in save_dict:
            self.optimizer.load_state_dict(save_dict['optimizer_state_dict'])
        
        # Restore scaler
        self.scaler = StandardScaler()
        self.scaler.set_params(**save_dict['scaler_params'])
        self.scaler.mean_ = save_dict.get('scaler_mean', save_dict.get('scaler_data_min', None))
        self.scaler.scale_ = save_dict.get('scaler_scale', None)
        if hasattr(self.scaler, 'data_min_'):
            self.scaler.data_min_ = save_dict.get('scaler_data_min', None)
            self.scaler.data_max_ = save_dict.get('scaler_data_max', None)
        
        self.logger.log_message(f"Model loaded from: {filepath}", "info")
    
    def _plot_training_history(self) -> None:
        """Plot training history and save to file."""
        if not self.train_loss_history:
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_loss_history, label='Train Loss')
        if self.val_loss_history:
            plt.plot(self.val_loss_history, label='Validation Loss')
        
        plt.title('LSTM Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Create plots directory if needed
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Save plot
        plot_path = os.path.join(self.plot_dir, f"{self.model_name}_training_history.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.log_message(f"Training history plot saved to: {plot_path}", "info")


def main():
    """Example usage of LSTMModel."""
    from utils.data_loader import DataLoader
    
    # Initialize data loader
    loader = DataLoader()
    
    # Load test data
    test_stocks = ['000001.SZ', '000002.SZ', '000004.SZ']
    loader.load_stock_data(test_stocks)
    
    # Get combined data
    combined_df = loader.get_combined_data()
    
    # Split data
    train_df, val_df, test_df = loader.split_data(combined_df)
    
    # Get features and target
    X_train, y_train = loader.get_features_and_target(train_df)
    X_val, y_val = loader.get_features_and_target(val_df)
    X_test, y_test = loader.get_features_and_target(test_df)
    
    # Initialize LSTM model with custom parameters
    params = {
        'input_size': X_train.shape[1],
        'hidden_size': 64,
        'num_layers': 2,
        'dropout_rate': 0.2,
        'sequence_length': 20,
        'learning_rate': 0.001,
        'batch_size': 32,
        'num_epochs': 50,
        'early_stopping_patience': 10
    }
    
    model = LSTMModel(model_name='test_lstm', params=params)
    
    # Train model
    print(f"Using device: {model.device}")
    model.train(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    test_metrics = model.evaluate(X_test, y_test)
    print("\nTest set metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.6f}")
    
    # Evaluate top-N performance
    top_n_metrics = model.evaluate_top_n(X_test, y_test)
    print("\nTop-N performance:")
    for key, value in top_n_metrics.items():
        print(f"  {key}: {value:.6f}")
    
    # Save model
    model.save_model()


if __name__ == "__main__":
    main()
