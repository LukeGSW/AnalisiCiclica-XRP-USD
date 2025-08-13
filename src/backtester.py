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
    
    # In src/backtester.py

    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """
        Esegue un backtest realistico basato su eventi, simulando un conto di trading.
        """
        print("📊 Running realistic event-based backtest...")
        
        if 'signal' not in df.columns or 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'signal' and 'close' columns")
    
        results = df.copy()
        trade_size = self.initial_capital  # Usiamo initial_capital come dimensione fissa del trade
        
        # Inizializza le colonne per il backtest
        cash = self.initial_capital
        shares = 0
        equity = [cash]
        positions = [0]
        trades_log = []
    
        for i in range(1, len(results)):
            current_price = results.iloc[i]['close']
            signal = results.iloc[i]['signal']
            
            # Gestione dei segnali
            if signal == 'BUY' and cash >= trade_size:
                # Entra in posizione se non siamo già dentro
                if shares == 0:
                    shares_to_buy = (trade_size / current_price)
                    shares += shares_to_buy
                    cash -= shares_to_buy * current_price * (1 + self.fees)
                    trades_log.append({'entry_date': results.index[i], 'entry_price': current_price})
    
            elif signal == 'SELL' and shares > 0:
                # Esce dalla posizione
                cash += shares * current_price * (1 - self.fees)
                
                # Logga il rendimento del trade
                last_trade = trades_log[-1]
                last_trade['exit_date'] = results.index[i]
                last_trade['exit_price'] = current_price
                last_trade['return'] = (current_price - last_trade['entry_price']) / last_trade['entry_price']
                
                shares = 0
                
            # Aggiorna l'equity giornaliera e la posizione
            current_equity = cash + (shares * current_price)
            equity.append(current_equity)
            positions.append(1 if shares > 0 else 0)
    
        # Aggiungi i risultati al DataFrame
        results['equity'] = equity
        results['position'] = positions
        results['returns'] = results['close'].pct_change().fillna(0)
        results['benchmark_equity'] = self.initial_capital * (1 + results['returns']).cumprod()
        
        # Le metriche ora possono usare il log dei trade
        metrics = self._calculate_metrics(results, trades_log)
        
        return {
            'results': results,
            'metrics': metrics,
            'final_equity': float(results['equity'].iloc[-1]),
            'total_return': float((results['equity'].iloc[-1] / self.initial_capital - 1) * 100),
            'trades_log': trades_log
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
    
    # In src/backtester.py

    def _calculate_metrics(self, results: pd.DataFrame, trades_log: list = []) -> Dict:
        """
        Calcola le metriche complete del backtest.
        """
        # ... (tutta la parte iniziale di calcolo drawdown, sharpe, etc. rimane uguale)
        
        # Calcolo drawdown
        running_max = results['equity'].expanding().max()
        drawdown = (results['equity'] - running_max) / running_max
        max_drawdown = float(drawdown.min() * 100)
        
        # ... (calcolo sharpe, sortino, calmar... rimane uguale)
        # ... usa results['equity'].pct_change() al posto di strategy_returns_net
        daily_strategy_returns = results['equity'].pct_change().fillna(0)
    
        # ... calcola sharpe, sortino, calmar usando daily_strategy_returns ...
        if len(daily_strategy_returns) > 0 and daily_strategy_returns.std() > 0:
            daily_mean = daily_strategy_returns.mean()
            daily_std = daily_strategy_returns.std()
            sharpe_ratio = float((daily_mean / daily_std) * np.sqrt(252)) if daily_std != 0 else 0
            
            negative_returns = daily_strategy_returns[daily_strategy_returns < 0]
            if len(negative_returns) > 0:
                downside_std = negative_returns.std()
                sortino_ratio = float((daily_mean / downside_std) * np.sqrt(252)) if downside_std != 0 else 0
            else:
                sortino_ratio = float('inf')
            
            annual_return = daily_mean * 252 * 100
            calmar_ratio = float(annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0
        else:
            sharpe_ratio = sortino_ratio = calmar_ratio = 0.0
    
        # Usa il trade_log per calcolare le statistiche dei trade (molto più preciso)
        trade_returns = [t['return'] for t in trades_log if 'return' in t]
        
        if trade_returns:
            winning_trades = sum(1 for r in trade_returns if r > self.fees * 2) # Considera i costi
            win_rate = float((winning_trades / len(trade_returns)) * 100) if trade_returns else 0.0
            
            gross_profits = sum(r for r in trade_returns if r > 0)
            gross_losses = abs(sum(r for r in trade_returns if r < 0))
            profit_factor = float(gross_profits / gross_losses) if gross_losses != 0 else float('inf')
        else:
            win_rate = 0.0
            profit_factor = 0.0
            
        return {
            'total_return_%': float((results['equity'].iloc[-1] / self.initial_capital - 1) * 100),
            'max_drawdown_%': abs(max_drawdown),
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'total_trades': len(trade_returns),
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
