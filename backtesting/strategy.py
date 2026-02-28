"""Base classes for trading strategies in QuantSandbox.

This module defines the abstract base classes for all trading strategies.
"""

import abc
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

class BaseStrategy(abc.ABC):
    """Abstract base class for all trading strategies.
    
    Attributes:
        name (str): Name of the strategy
        params (Dict[str, any]): Strategy parameters
        data (pd.DataFrame): Strategy data
        positions (pd.DataFrame): Trading positions
        returns (pd.Series): Strategy returns
    """
    
    def __init__(self, name: str = "base_strategy", params: Optional[Dict[str, any]] = None):
        """Initialize BaseStrategy.
        
        Args:
            name (str): Name of the strategy
            params (Optional[Dict[str, any]]): Strategy parameters
        """
        self.name = name
        self.params = params or {}
        self.data = None
        self.positions = None
        self.returns = None
    
    @abc.abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on the provided data.
        
        Args:
            data (pd.DataFrame): Input data with predictions and market data
            
        Returns:
            pd.DataFrame: DataFrame containing trading signals
        """
        pass
    
    @abc.abstractmethod
    def execute_trades(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Execute trades based on generated signals.
        
        Args:
            signals (pd.DataFrame): Trading signals
            
        Returns:
            pd.DataFrame: Positions after executing trades
        """
        pass
    
    @abc.abstractmethod
    def calculate_returns(self, positions: pd.DataFrame, data: pd.DataFrame) -> pd.Series:
        """Calculate strategy returns based on positions and market data.
        
        Args:
            positions (pd.DataFrame): Trading positions
            data (pd.DataFrame): Market data
            
        Returns:
            pd.Series: Strategy returns
        """
        pass
    
    def run(self, data: pd.DataFrame) -> pd.Series:
        """Run the complete strategy pipeline.
        
        Args:
            data (pd.DataFrame): Input data with predictions and market data
            
        Returns:
            pd.Series: Strategy returns
        """
        self.data = data.copy()
        
        # Generate signals
        signals = self.generate_signals(data)
        
        # Execute trades
        self.positions = self.execute_trades(signals)
        
        # Calculate returns
        self.returns = self.calculate_returns(self.positions, data)
        
        return self.returns
    
    def get_positions(self) -> pd.DataFrame:
        """Get the trading positions.
        
        Returns:
            pd.DataFrame: Trading positions
        """
        return self.positions.copy() if self.positions is not None else None
    
    def get_returns(self) -> pd.Series:
        """Get the strategy returns.
        
        Returns:
            pd.Series: Strategy returns
        """
        return self.returns.copy() if self.returns is not None else None
    
    def get_strategy_info(self) -> Dict[str, any]:
        """Get information about the strategy.
        
        Returns:
            Dict[str, any]: Strategy information
        """
        return {
            'name': self.name,
            'params': self.params,
            'has_returns': self.returns is not None,
            'total_return': self.returns.sum() if self.returns is not None else None
        }


class LongOnlyStrategy(BaseStrategy):
    """Long-only trading strategy based on model predictions.
    
    This strategy goes long on stocks with the highest predicted returns.
    """
    
    def __init__(self, name: str = "long_only_strategy", params: Optional[Dict[str, any]] = None):
        """Initialize LongOnlyStrategy.
        
        Args:
            name (str): Name of the strategy
            params (Optional[Dict[str, any]]): Strategy parameters
                - 'n_stocks': Number of stocks to hold (default: 10)
                - 'min_prediction': Minimum predicted return to consider (default: 0.0)
                - 'rebalance_frequency': Rebalance frequency (default: 'daily')
        """
        super().__init__(name, params)
        self.n_stocks = self.params.get('n_stocks', 10)
        self.min_prediction = self.params.get('min_prediction', 0.0)
        self.rebalance_frequency = self.params.get('rebalance_frequency', 'daily')
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on predicted returns.
        
        Args:
            data (pd.DataFrame): Input data with 'predicted_return' column
            
        Returns:
            pd.DataFrame: DataFrame containing 'signal' column (1 for buy, 0 for hold)
        """
        data = data.copy()
        
        # Sort by predicted return descending
        data = data.sort_values(by='predicted_return', ascending=False)
        
        # Initialize signals
        data['signal'] = 0
        
        # Select top N stocks with positive predicted returns
        top_stocks = data[data['predicted_return'] > self.min_prediction].head(self.n_stocks)
        data.loc[top_stocks.index, 'signal'] = 1
        
        return data
    
    def execute_trades(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Execute trades based on generated signals.
        
        Args:
            signals (pd.DataFrame): DataFrame with 'signal' column
            
        Returns:
            pd.DataFrame: Positions with 'position' column
        """
        positions = signals.copy()
        
        # For long-only strategy, position size is equal-weighted
        positions['position'] = positions['signal'] / positions['signal'].sum() if positions['signal'].sum() > 0 else 0
        
        return positions
    
    def calculate_returns(self, positions: pd.DataFrame, data: pd.DataFrame) -> pd.Series:
        """Calculate strategy returns based on positions and actual returns.
        
        Args:
            positions (pd.DataFrame): DataFrame with 'position' column
            data (pd.DataFrame): DataFrame with 'actual_return' column
            
        Returns:
            pd.Series: Strategy returns
        """
        # Merge positions with actual returns
        combined = pd.merge(positions, data['actual_return'], left_index=True, right_index=True)
        
        # Calculate daily strategy return
        combined['strategy_return'] = combined['position'] * combined['actual_return']
        
        # Group by date and sum returns
        daily_returns = combined.groupby(combined.index.get_level_values('交易日期'))['strategy_return'].sum()
        
        return daily_returns


class TopNLongStrategy(BaseStrategy):
    """Top-N long-only strategy with position sizing options.
    
    This strategy allows different position sizing methods for the top N stocks.
    """
    
    def __init__(self, name: str = "top_n_long_strategy", params: Optional[Dict[str, any]] = None):
        """Initialize TopNLongStrategy.
        
        Args:
            name (str): Name of the strategy
            params (Optional[Dict[str, any]]): Strategy parameters
                - 'n_stocks': Number of stocks to hold (default: 10)
                - 'position_sizing': Position sizing method ('equal', 'rank_weighted', 'prediction_weighted')
                - 'min_prediction': Minimum predicted return to consider (default: 0.0)
        """
        super().__init__(name, params)
        self.n_stocks = self.params.get('n_stocks', 10)
        self.position_sizing = self.params.get('position_sizing', 'equal')
        self.min_prediction = self.params.get('min_prediction', 0.0)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on predicted returns.
        
        Args:
            data (pd.DataFrame): Input data with 'predicted_return' column
            
        Returns:
            pd.DataFrame: DataFrame containing 'signal' column
        """
        data = data.copy()
        
        # Sort by predicted return descending
        data = data.sort_values(by='predicted_return', ascending=False)
        
        # Initialize signals
        data['signal'] = 0
        
        # Select top N stocks with positive predicted returns
        top_stocks = data[data['predicted_return'] > self.min_prediction].head(self.n_stocks)
        data.loc[top_stocks.index, 'signal'] = 1
        
        return data
    
    def execute_trades(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Execute trades with different position sizing methods.
        
        Args:
            signals (pd.DataFrame): DataFrame with 'signal' and 'predicted_return' columns
            
        Returns:
            pd.DataFrame: Positions with 'position' column
        """
        positions = signals.copy()
        positions['position'] = 0.0
        
        # Filter to only stocks with buy signal
        buy_signals = positions[positions['signal'] == 1]
        
        if len(buy_signals) == 0:
            return positions
        
        if self.position_sizing == 'equal':
            # Equal-weighted positions
            positions.loc[buy_signals.index, 'position'] = 1.0 / len(buy_signals)
        
        elif self.position_sizing == 'rank_weighted':
            # Weight positions by rank (higher rank = higher weight)
            buy_signals = buy_signals.sort_values(by='predicted_return', ascending=False)
            ranks = np.arange(1, len(buy_signals) + 1)
            weights = ranks / ranks.sum()
            positions.loc[buy_signals.index, 'position'] = weights
        
        elif self.position_sizing == 'prediction_weighted':
            # Weight positions by predicted returns
            total_pred = buy_signals['predicted_return'].sum()
            if total_pred > 0:
                positions.loc[buy_signals.index, 'position'] = buy_signals['predicted_return'] / total_pred
            else:
                positions.loc[buy_signals.index, 'position'] = 1.0 / len(buy_signals)
        
        else:
            # Default to equal-weighted
            positions.loc[buy_signals.index, 'position'] = 1.0 / len(buy_signals)
        
        return positions
    
    def calculate_returns(self, positions: pd.DataFrame, data: pd.DataFrame) -> pd.Series:
        """Calculate strategy returns based on positions and actual returns.
        
        Args:
            positions (pd.DataFrame): DataFrame with 'position' column
            data (pd.DataFrame): DataFrame with 'actual_return' column
            
        Returns:
            pd.Series: Strategy returns
        """
        combined = pd.merge(positions, data['actual_return'], left_index=True, right_index=True)
        combined['strategy_return'] = combined['position'] * combined['actual_return']
        
        # Group by date to get daily returns
        daily_returns = combined.groupby(combined.index.get_level_values('交易日期'))['strategy_return'].sum()
        
        return daily_returns


def create_strategy(strategy_type: str, **kwargs) -> BaseStrategy:
    """Factory function to create strategy instances.
    
    Args:
        strategy_type (str): Type of strategy to create ('long_only', 'top_n_long')
        **kwargs: Strategy parameters
        
    Returns:
        BaseStrategy: Strategy instance
    
    Raises:
        ValueError: If strategy_type is not recognized
    """
    if strategy_type.lower() == 'long_only':
        return LongOnlyStrategy(**kwargs)
    elif strategy_type.lower() == 'top_n_long':
        return TopNLongStrategy(**kwargs)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
