"""
Backtesting module for Kriterion Quant Trading System
Implements walk-forward analysis for strategy validation
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import json

from config import Config

class Backtester:
    """Class to perform backtesting with walk-forward analysis"""
    
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
        results['returns'] = results['close'].pct_change()
        
        # Determine positions (1 for long, 0 for flat)
        results['position'] = 0
        position = 0
        
        for i in range(len(results)):
            if results.iloc[i]['signal'] == 'BUY':
                position = 1
            elif results.iloc[i]['signal'] == 'SELL':
                position = 0
            results.iloc[i, results.columns.get_loc('position')] = position
        
        # Calculate strategy returns (position from previous day)
        results['strategy_returns'] = results['position'].shift(1) * results['returns']
        
        # Apply transaction costs
        results['trades'] = results['position'].diff().abs()
        results['costs'] = results['trades'] * self.fees
        results['strategy_returns'] = results['strategy_returns'] - results['costs']
        
        # Calculate cumulative returns
        results['cumulative_returns'] = (1 + results['returns']).cumprod()
        results['cumulative_strategy'] = (1 + results['strategy_returns']).cumprod()
        
        # Calculate equity curve
        results['equity'] = self.initial_capital * results['cumulative_strategy']
        results['benchmark_equity'] = self.initial_capital * results['cumulative_returns']
        
        # Calculate metrics
        metrics = self._calculate_metrics(results)
        
        return {
            'results': results,
            'metrics': metrics,
            'final_equity': results['equity'].iloc[-1],
            'total_return': (results['equity'].iloc[-1] / self.initial_capital - 1) * 100
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
        
        if window_size is None:
            # Simple in-sample/out-of-sample split
            split_idx = int(len(df) * in_sample_ratio)
            
            is_data = df.iloc[:split_idx]
            oos_data = df.iloc[split_idx:]
            
            print(f"  In-Sample: {is_data.index[0].date()} to {is_data.index[-1].date()} ({len(is_data)} days)")
            print(f"  Out-of-Sample: {oos_data.index[0].date()} to {oos_data.index[-1].date()} ({len(oos_data)} days)")
            
            # Run backtests
            is_results = self.run_backtest(is_data)
            oos_results = self.run_backtest(oos_data)
            
            return {
                'in_sample': is_results,
                'out_of_sample': oos_results,
                'is_period': (is_data.index[0].date(), is_data.index[-1].date()),
                'oos_period': (oos_data.index[0].date(), oos_data.index[-1].date())
            }
        
        else:
            # Rolling walk-forward analysis
            step_size = step_size or window_size // 4
            is_window = int(window_size * in_sample_ratio)
            oos_window = window_size - is_window
            
            walk_forward_results = []
            
            for start_idx in range(0, len(df) - window_size, step_size):
                end_idx = start_idx + window_size
                
                # Split window into IS and OOS
                is_end = start_idx + is_window
                
                is_data = df.iloc[start_idx:is_end]
                oos_data = df.iloc[is_end:end_idx]
                
                # Run backtests
                is_results = self.run_backtest(is_data)
                oos_results = self.run_backtest(oos_data)
                
                walk_forward_results.append({
                    'window': (df.index[start_idx].date(), df.index[end_idx-1].date()),
                    'is_metrics': is_results['metrics'],
                    'oos_metrics': oos_results['metrics']
                })
            
            # Aggregate results
            aggregated = self._aggregate_walk_forward_results(walk_forward_results)
            
            return {
                'walk_forward_windows': walk_forward_results,
                'aggregated_metrics': aggregated,
                'num_windows': len(walk_forward_results)
            }
    
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
        total_return = (results['equity'].iloc[-1] / self.initial_capital - 1) * 100
        
        # Calculate drawdown
        running_max = results['equity'].expanding().max()
        drawdown = (results['equity'] - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Risk metrics
        strategy_returns = results['strategy_returns'].dropna()
        
        if len(strategy_returns) > 0:
            sharpe_ratio = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252) if strategy_returns.std() != 0 else 0
            
            # Sortino ratio (downside deviation)
            negative_returns = strategy_returns[strategy_returns < 0]
            sortino_std = negative_returns.std() if len(negative_returns) > 0 else 0
            sortino_ratio = (strategy_returns.mean() / sortino_std) * np.sqrt(252) if sortino_std != 0 else 0
            
            # Calmar ratio
            annual_return = strategy_returns.mean() * 252 * 100
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        else:
            sharpe_ratio = sortino_ratio = calmar_ratio = 0
        
        # Trading statistics
        trades = results['trades'].sum() / 2  # Divide by 2 because each trade has entry and exit
        
        # Win rate
        trade_returns = []
        in_position = False
        entry_price = 0
        
        for i in range(len(results)):
            if results.iloc[i]['signal'] == 'BUY' and not in_position:
                entry_price = results.iloc[i]['close']
                in_position = True
            elif results.iloc[i]['signal'] == 'SELL' and in_position:
                exit_price = results.iloc[i]['close']
                trade_return = (exit_price - entry_price) / entry_price
                trade_returns.append(trade_return)
                in_position = False
        
        if trade_returns:
            winning_trades = sum(1 for r in trade_returns if r > 0)
            win_rate = (winning_trades / len(trade_returns)) * 100
            
            # Profit factor
            gross_profits = sum(r for r in trade_returns if r > 0)
            gross_losses = abs(sum(r for r in trade_returns if r < 0))
            profit_factor = gross_profits / gross_losses if gross_losses != 0 else np.inf
        else:
            win_rate = 0
            profit_factor = 0
        
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
    
    def _aggregate_walk_forward_results(self, results: list) -> Dict:
        """
        Aggregate results from multiple walk-forward windows
        
        Parameters
        ----------
        results : list
            List of walk-forward window results
        
        Returns
        -------
        Dict
            Aggregated metrics
        """
        # Extract all OOS metrics
        oos_metrics = [w['oos_metrics'] for w in results]
        
        # Calculate averages
        aggregated = {}
        metric_keys = oos_metrics[0].keys()
        
        for key in metric_keys:
            values = [m[key] for m in oos_metrics]
            aggregated[f'avg_{key}'] = np.mean(values)
            aggregated[f'std_{key}'] = np.std(values)
            aggregated[f'min_{key}'] = np.min(values)
            aggregated[f'max_{key}'] = np.max(values)
        
        # Calculate stability metrics
        is_returns = [w['is_metrics']['total_return_%'] for w in results]
        oos_returns = [w['oos_metrics']['total_return_%'] for w in results]
        
        # Correlation between IS and OOS performance
        if len(is_returns) > 1:
            correlation = np.corrcoef(is_returns, oos_returns)[0, 1]
            aggregated['is_oos_correlation'] = correlation
        
        return aggregated
    
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
        
        # Convert non-serializable objects
        save_dict = {
            'metrics': results.get('metrics', {}),
            'final_equity': float(results.get('final_equity', 0)),
            'total_return': float(results.get('total_return', 0))
        }
        
        if 'in_sample' in results:
            save_dict['in_sample_metrics'] = results['in_sample']['metrics']
            save_dict['out_of_sample_metrics'] = results['out_of_sample']['metrics']
        
        with open(filename, 'w') as f:
            json.dump(save_dict, f, indent=2, default=str)
        
        print(f"💾 Backtest results saved to {filename}")
        return filename
