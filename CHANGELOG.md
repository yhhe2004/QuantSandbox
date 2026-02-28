# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-28

### Added
- Initial project structure setup
- Created todo.json for task tracking with comprehensive task list
- Added AGENTS.md with project workflow and guidelines
- Established CHANGELOG.md for tracking all significant changes
- Set up core directory structure for data, models, backtesting, utils, tests, and docs
- Implemented comprehensive data loading and preprocessing module
- Added support for technical indicator features and rolling statistics
- Implemented time-based train/val/test split (80/10/10)
- Added missing data handling functionality
- Created abstract base model classes (BaseModel, RegressionModel)
- Implemented Linear Regression model with scikit-learn
- Implemented Random Forest model with scikit-learn
- Implemented Feedforward Neural Network model with PyTorch
- Added comprehensive regression metrics (MAE, MSE, RMSE, R², Pearson/Spearman correlation)
- Added top-N performance evaluation metrics (top 5, 10, 50, 100) with average return, win rate, and market comparison
- Added model saving/loading functionality and checkpointing
- Implemented comprehensive logging system with experiment tracking
- Added early stopping for deep learning models
- Created example scripts demonstrating complete workflow with logging integration

### Notes
- Core framework is complete with data loading, preprocessing, multiple model implementations, evaluation metrics, and logging system
- Random Forest shows best overall performance with test set R²=0.159 and top-5 average return of 0.039 (80% win rate)
- Feedforward Neural Network shows competitive performance (R²=0.139) with proper regularization and early stopping
- Logging system tracks all experiments, hyperparameters, and metrics for reproducibility
- Ready to start developing backtesting framework and advanced models (LSTMs, etc.)
