"""
Simplified Backtesting module for Kriterion Quant Trading System
Version without vectorbt dependency for better compatibility
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import json

from config import Config

class Backtester:
    """Simplified backtester without external dependencies"""
    
    def __init__(self, initial_capital: float = None, fees: float = None):
        """
        Initialize the backtester
        
        Parameters
        ----------
        initial_capital : float, optional
            Starting capital. Defaults to Config.INITIAL_CAPITAL
        fees : float, optional
            Trading fees as percentage. Defaults to Config.TRADING_FEES
        """
        self.initial_capital = initial_capital or Config.INITIAL_CAPITAL
        self.fees = fees or Config.TRADING_FEES
    
    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """
        Run a simple vectorized backtest
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with signals and prices
        
        Returns
        -------
        Dict
            Backtest results and metrics
        """
        print("📊 Running backtest...")
        
        # Ensure required columns exist
        if 'signal' not in df.columns or 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'signal' and 'close' columns")
        
        # Initialize results
        results = df.copy()
        
        # Calculate returns
        results['returns'] = results['close'].pct_change().fillna(0)
        
        # Create position array
        results['position'] = 0.0
        
        # Track positions based on signals
        current_position = 0
        positions = []
        
        for i, row in results.iterrows():
            if row['signal'] == 'BUY':
                current_position = 1
            elif row['signal'] == 'SELL':
                current_position = 0
            positions.append(current_position)
        
        results['position'] = positions
        
        # Calculate strategy returns
        results['strategy_returns'] = results['position'].shift(1).fillna(0) * results['returns']
        
        # Apply transaction costs
        results['trade'] = results['position'].diff().abs()
        results['costs'] = results['trade'] * self.fees
        results['strategy_returns_net'] = results['strategy_returns'] - results['costs']
        
        # Calculate cumulative performance
        results['cum_returns'] = (1 + results['returns']).cumprod()
        results['cum_strategy'] = (1 + results['strategy_returns_net']).cumprod()
        
        # Calculate equity curve
        results['equity'] = self.initial_capital * results['cum_strategy']
        results['benchmark_equity'] = self.initial_capital * results['cum_returns']
        
        # Calculate metrics
        metrics = self._calculate_metrics(results)
        
        return {
            'results': results,
            'metrics': metrics,
            'final_equity': float(results['equity'].iloc[-1]),
            'total_return': float((results['equity'].iloc[-1] / self.initial_capital - 1) * 100)
        }
    
    def run_walk_forward_analysis(
        self, 
        df: pd.DataFrame,
        in_sample_ratio: float = None,
        window_size: int = None,
        step_size: int = None
    ) -> Dict:
        """
        Perform walk-forward analysis for robust validation
        
        Parameters
        ----------
        df : pd.DataFrame
            Complete dataset with prices
        in_sample_ratio : float, optional
            Ratio for in-sample period. Defaults to Config.IN_SAMPLE_RATIO
        window_size : int, optional
            Size of rolling window in days. If None, uses simple IS/OOS split
        step_size : int, optional
            Step size for rolling window. If None, uses window_size // 4
        
        Returns
        -------
        Dict
            Walk-forward analysis results
        """
        print("🔄 Running walk-forward analysis...")
        
        in_sample_ratio = in_sample_ratio or Config.IN_SAMPLE_RATIO
        
        # Simple in-sample/out-of-sample split
        split_idx = int(len(df) * in_sample_ratio)
        
        is_data = df.iloc[:split_idx]
        oos_data = df.iloc[split_idx:]
        
        print(f"  In-Sample: {is_data.index[0].date()} to {is_data.index[-1].date()} ({len(is_data)} days)")
        print(f"  Out-of-Sample: {oos_data.index[0].date()} to {oos_data.index[-1].date()} ({len(oos_data)} days)")
        
        # Run backtests
        is_results = self.run_backtest(is_data) if len(is_data) > 10 else None
        oos_results = self.run_backtest(oos_data) if len(oos_data) > 10 else None
        
        result = {}
        
        if is_results:
            result['in_sample'] = is_results
            result['is_period'] = (is_data.index[0].date(), is_data.index[-1].date())
        
        if oos_results:
            result['out_of_sample'] = oos_results
            result['oos_period'] = (oos_data.index[0].date(), oos_data.index[-1].date())
        
        return result
    
    def _calculate_metrics(self, results: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive backtest metrics
        
        Parameters
        ----------
        results : pd.DataFrame
            Backtest results DataFrame
        
        Returns
        -------
        Dict
            Performance metrics
        """
        # Basic returns metrics
        total_return = float((results['equity'].iloc[-1] / self.initial_capital - 1) * 100)
        
        # Calculate drawdown
        running_max = results['equity'].expanding().max()
        drawdown = (results['equity'] - running_max) / running_max
        max_drawdown = float(drawdown.min() * 100)
        
        # Risk metrics
        strategy_returns = results['strategy_returns_net']
        
        # Remove any NaN or infinite values
        strategy_returns = strategy_returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(strategy_returns) > 0 and strategy_returns.std() > 0:
            daily_mean = strategy_returns.mean()
            daily_std = strategy_returns.std()
            
            # Sharpe ratio (annualized)
            sharpe_ratio = float((daily_mean / daily_std) * np.sqrt(252)) if daily_std != 0 else 0
            
            # Sortino ratio (downside deviation)
            negative_returns = strategy_returns[strategy_returns < 0]
            if len(negative_returns) > 0:
                downside_std = negative_returns.std()
                sortino_ratio = float((daily_mean / downside_std) * np.sqrt(252)) if downside_std != 0 else 0
            else:
                sortino_ratio = float(sharpe_ratio)  # If no negative returns, use Sharpe
            
            # Calmar ratio
            annual_return = daily_mean * 252 * 100
            calmar_ratio = float(annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0
        else:
            sharpe_ratio = sortino_ratio = calmar_ratio = 0.0
        
        # Trading statistics
        trades = results['trade'].sum() / 2  # Divide by 2 for round trips
        
        # Calculate win rate from trade-by-trade analysis
        trade_returns = []
        entry_price = None
        
        for i in range(len(results)):
            if results.iloc[i]['signal'] == 'BUY' and entry_price is None:
                entry_price = results.iloc[i]['close']
            elif results.iloc[i]['signal'] == 'SELL' and entry_price is not None:
                exit_price = results.iloc[i]['close']
                trade_return = (exit_price - entry_price) / entry_price
                trade_returns.append(trade_return)
                entry_price = None
        
        if trade_returns:
            winning_trades = sum(1 for r in trade_returns if r > 0)
            win_rate = float((winning_trades / len(trade_returns)) * 100)
            
            # Profit factor
            gross_profits = sum(r for r in trade_returns if r > 0)
            gross_losses = abs(sum(r for r in trade_returns if r < 0))
            profit_factor = float(gross_profits / gross_losses) if gross_losses != 0 else 0.0
        else:
            win_rate = 0.0
            profit_factor = 0.0
        
        return {
            'total_return_%': total_return,
            'max_drawdown_%': abs(max_drawdown),
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'total_trades': int(trades),
            'win_rate_%': win_rate,
            'profit_factor': profit_factor
        }
    
    def save_backtest_results(self, results: Dict, filename: str = None) -> str:
        """
        Save backtest results to JSON file
        
        Parameters
        ----------
        results : Dict
            Backtest results
        filename : str, optional
            Output filename. Defaults to Config.BACKTEST_RESULTS_FILE
        
        Returns
        -------
        str
            Path to saved file
        """
        filename = filename or Config.BACKTEST_RESULTS_FILE
        
        # Extract only serializable data
        save_dict = {}
        
        if 'metrics' in results:
            save_dict['metrics'] = results['metrics']
        
        if 'in_sample' in results and results['in_sample']:
            save_dict['in_sample_metrics'] = results['in_sample']['metrics']
            save_dict['in_sample_return'] = results['in_sample'].get('total_return', 0)
        
        if 'out_of_sample' in results and results['out_of_sample']:
            save_dict['out_of_sample_metrics'] = results['out_of_sample']['metrics']
            save_dict['out_of_sample_return'] = results['out_of_sample'].get('total_return', 0)
        
        # Ensure all values are JSON serializable
        def make_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            else:
                return obj
        
        save_dict = make_serializable(save_dict)
        
        with open(filename, 'w') as f:
            json.dump(save_dict, f, indent=2, default=str)
        
        print(f"💾 Backtest results saved to {filename}")
        return filename
