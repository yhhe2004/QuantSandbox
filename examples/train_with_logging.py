"""Example script showing how to train models with logging.

This demonstrates the complete workflow of loading data, training a model,
and logging all results with the QuantSandbox logging system.
"""

import os
import sys
from utils.data_loader import DataLoader
from utils.logger import get_logger
from models.sklearn_models.random_forest import RandomForest
from models.pytorch_models.feedforward_nn import FeedforwardNN

def train_random_forest_with_logging():
    """Train Random Forest model with comprehensive logging."""
    # Create logger
    logger = get_logger(model_name="random_forest", experiment_name="random_forest_baseline")
    
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
    X_test, y_test = loader.get_features_and_target(test_df)
    
    # Initialize model
    model = RandomForest(model_name='random_forest_baseline')
    
    # Log training start
    logger.start_training(
        features_shape=X_train.shape,
        hyperparameters={
            'model_name': 'random_forest',
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'train_val_test_split': [0.8, 0.1, 0.1],
            'num_features': X_train.shape[1],
            'num_train_samples': len(X_train),
            'num_val_samples': len(X_val),
            'num_test_samples': len(X_test)
        }
    )
    
    try:
        # Train model
        model.train(X_train, y_train, X_val, y_val)
        
        # Log training metrics
        train_metrics = model.evaluate(X_train, y_train)
        logger.log_phase(phase="training", metrics=train_metrics)
        
        # Log validation metrics
        val_metrics = model.evaluate(X_val, y_val)
        logger.log_phase(phase="validation", metrics=val_metrics)
        
        # Log test metrics
        test_metrics = model.evaluate(X_test, y_test)
        logger.log_phase(phase="test", metrics=test_metrics)
        
        # Log top-N performance
        top_n_metrics = model.evaluate_top_n(X_test, y_test)
        logger.log_phase(phase="test_top_n", metrics=top_n_metrics)
        
        # Log feature importance
        feature_importance = model.get_feature_importance(X_train.columns)
        logger.log_message("Top 10 feature importance:")
        for _, row in feature_importance.head(10).iterrows():
            logger.log_message(f"  {row['feature']}: {row['importance']:.6f}")
            
    finally:
        # End training logging
        logger.end_training()
    
    print(f"\nExperiment completed! Logs saved to: logs/{logger.logger.experiment_name}")
    
    return model, logger

def train_feedforward_nn_with_logging():
    """Train Feedforward Neural Network model with comprehensive logging."""
    # Create logger
    logger = get_logger(model_name="feedforward_nn", experiment_name="feedforward_nn_baseline")
    
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
    X_test, y_test = loader.get_features_and_target(test_df)
    
    # Initialize model
    model = FeedforwardNN(model_name='feedforward_nn_baseline')
    
    # Log training start
    logger.start_training(
        features_shape=X_train.shape,
        hyperparameters={
            'model_name': 'feedforward_nn',
            'hidden_layers': [256, 128, 64],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 50,
            'early_stopping_patience': 10,
            'train_val_test_split': [0.8, 0.1, 0.1],
            'num_features': X_train.shape[1],
            'num_train_samples': len(X_train),
            'num_val_samples': len(X_val),
            'num_test_samples': len(X_test)
        }
    )
    
    try:
        # Train model
        model.train(X_train, y_train, X_val, y_val)
        
        # Log training metrics
        train_metrics = model.evaluate(X_train, y_train)
        logger.log_phase(phase="training", metrics=train_metrics)
        
        # Log validation metrics
        val_metrics = model.evaluate(X_val, y_val)
        logger.log_phase(phase="validation", metrics=val_metrics)
        
        # Log test metrics
        test_metrics = model.evaluate(X_test, y_test)
        logger.log_phase(phase="test", metrics=test_metrics)
        
        # Log top-N performance
        top_n_metrics = model.evaluate_top_n(X_test, y_test)
        logger.log_phase(phase="test_top_n", metrics=top_n_metrics)
        
    finally:
        # End training logging
        logger.end_training()
    
    print(f"\nExperiment completed! Logs saved to: logs/{logger.logger.experiment_name}")
    
    return model, logger

def compare_models():
    """Compare model performances using logged results."""
    from utils.logger import Logger
    
    logger = Logger()
    experiments_df = logger.get_latest_experiments(num_experiments=10)
    
    print("\nRecent Experiments Summary:")
    print("-" * 120)
    print(experiments_df[['experiment_name', 'model_name', 'date', 'time', 'best_val_r2', 'best_test_r2']].to_string(index=False))
    print("-" * 120)
    
    return experiments_df

if __name__ == "__main__":
    print("=" * 80)
    print("QuantSandbox - Model Training with Logging Example")
    print("=" * 80)
    
    # Train Random Forest with logging
    print("\n1. Training Random Forest with logging...")
    rf_model, rf_logger = train_random_forest_with_logging()
    
    # Train Feedforward NN with logging
    print("\n2. Training Feedforward Neural Network with logging...")
    nn_model, nn_logger = train_feedforward_nn_with_logging()
    
    # Compare models
    print("\n3. Comparing model performances...")
    experiments_df = compare_models()
    
    print("\nAll experiments completed! Check the logs/ directory for detailed results.")
