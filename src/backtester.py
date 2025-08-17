"""
Backtesting module for Kriterion Quant Trading System
Implements a realistic event-driven backtester for fixed-size trades.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import json
import os
from config import Config

class Backtester:
    """Class to perform realistic, event-driven backtesting."""
    
    # ================================================================= #
    #                     <<< SEZIONE MODIFICATA 1 >>>                    #
    # ================================================================= #
    def __init__(self, data_path: str, initial_capital: float = None, fees: float = None):
        """
        Initialize the backtester.
        
        Parameters
        ----------
        data_path : str
            The path to the directory where backtest results will be saved.
        initial_capital : float, optional
            Starting capital. Defaults to Config.INITIAL_CAPITAL.
        fees : float, optional
            Trading fees as a percentage. Defaults to Config.TRADING_FEES.
        """
        self.data_path = data_path
        self.initial_capital = initial_capital or Config.INITIAL_CAPITAL
        self.fees = fees or Config.TRADING_FEES
    # ================================================================= #

    def run_backtest(self, df: pd.DataFrame) -> Dict:
        """
        Runs a realistic event-driven backtest simulating a trading account.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with signals and prices.
            
        Returns
        -------
        Dict
            A dictionary containing the results DataFrame, performance metrics, and trade log.
        """
        print("📊 Running realistic event-driven backtest...")
        
        if 'signal' not in df.columns or 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'signal' and 'close' columns")

        results = df.copy()
        
        # --- Simulazione del Conto di Trading ---
        cash = self.initial_capital
        shares = 0.0
        trade_size_dollars = self.initial_capital # Usa il capitale iniziale come dimensione fissa per ogni trade

        equity_over_time = []
        positions_over_time = []
        trades_log = []
        
        entry_details = {}

        for i in range(len(results)):
            current_price = results.iloc[i]['close']
            signal = results.iloc[i]['signal']
            current_date = results.index[i]

            # Gestione dei segnali
            if signal == 'BUY' and shares == 0:  # Entra solo se siamo FLAT
                shares_to_buy = trade_size_dollars / current_price
                cost = shares_to_buy * current_price * (1 + self.fees)
                
                # Non usiamo il cash totale, ma simuliamo un'operazione fissa
                # In un sistema reale, qui si controllerebbe il margine, non il cash totale.
                shares = shares_to_buy
                entry_details = {'entry_date': current_date, 'entry_price': current_price}

            elif signal == 'SELL' and shares > 0: # Esce solo se siamo LONG
                revenue = shares * current_price * (1 - self.fees)
                
                if entry_details:
                    # Calcola il rendimento del trade
                    trade_return = (current_price - entry_details['entry_price']) / entry_details['entry_price']
                    
                    # Logga il trade completato
                    trades_log.append({
                        'entry_date': entry_details['entry_date'],
                        'entry_price': entry_details['entry_price'],
                        'exit_date': current_date,
                        'exit_price': current_price,
                        'return': trade_return
                    })
                    
                    # Aggiorna il cash con il profitto/perdita del trade
                    # (Prezzo di vendita - Prezzo di acquisto) * numero di azioni - costi
                    profit_loss = (current_price - entry_details['entry_price']) * shares - (trade_size_dollars * self.fees * 2)
                    cash += profit_loss

                shares = 0
                entry_details = {}

            # Calcola l'equity giornaliera
            # L'equity è il cash + il valore delle azioni se in posizione.
            # Se siamo flat, l'equity è solo il cash accumulato.
            # Se siamo long, l'equity è il cash meno il costo dell'operazione + il valore corrente delle azioni.
            # Per semplicità, calcoliamo l'equity come il capitale iniziale + i profitti/perdite accumulati.
            current_equity = cash
            equity_over_time.append(current_equity)
            positions_over_time.append(1 if shares > 0 else 0)

        # Aggiungi i risultati al DataFrame
        results['equity'] = equity_over_time
        results['position'] = positions_over_time
        results['returns'] = results['close'].pct_change().fillna(0)
        results['benchmark_equity'] = self.initial_capital * (1 + results['returns']).cumprod()
        
        metrics = self._calculate_metrics(results, trades_log)
        
        return {
            'results': results,
            'metrics': metrics,
            'final_equity': float(results['equity'].iloc[-1]),
            'total_return': float((results['equity'].iloc[-1] / self.initial_capital - 1) * 100),
            'trades_log': trades_log
        }

    def run_walk_forward_analysis(self, df: pd.DataFrame, in_sample_ratio: float = None) -> Dict:
        """
        Perform walk-forward analysis for robust validation.
        This function now correctly uses the realistic backtester.
        """
        print("🔄 Running walk-forward analysis...")
        
        in_sample_ratio = in_sample_ratio or Config.IN_SAMPLE_RATIO
        
        split_idx = int(len(df) * in_sample_ratio)
        
        is_data = df.iloc[:split_idx]
        oos_data = df.iloc[split_idx:]
        
        print(f"  In-Sample: {is_data.index[0].date()} to {is_data.index[-1].date()} ({len(is_data)} days)")
        print(f"  Out-of-Sample: {oos_data.index[0].date()} to {oos_data.index[-1].date()} ({len(oos_data)} days)")
        
        # Run backtests on each segment
        is_results = self.run_backtest(is_data) if len(is_data) > 10 else None
        oos_results = self.run_backtest(oos_data) if len(oos_data) > 10 else None
        
        result = {}
        if is_results:
            result['in_sample'] = is_results
            result['in_sample_metrics'] = is_results['metrics']
        
        if oos_results:
            result['out_of_sample'] = oos_results
            result['out_of_sample_metrics'] = oos_results['metrics']
            
        return result

    def _calculate_metrics(self, results: pd.DataFrame, trades_log: List[Dict]) -> Dict:
        """
        Calculate comprehensive backtest metrics from realistic simulation results.
        """
        # --- Metriche basate sull'Equity Curve ---
        total_return = (results['equity'].iloc[-1] / self.initial_capital - 1) * 100
        
        running_max = results['equity'].expanding().max()
        drawdown = (results['equity'] - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        daily_returns = results['equity'].pct_change().fillna(0)
        
        if daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            
            downside_returns = daily_returns[daily_returns < 0]
            downside_std = downside_returns.std()
            sortino_ratio = (daily_returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else float('inf')
            
            annual_return = daily_returns.mean() * 252 * 100
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else float('inf')
        else:
            sharpe_ratio = sortino_ratio = calmar_ratio = 0.0

        # --- Metriche basate sul Trade Log (più precise) ---
        trade_returns = [t['return'] for t in trades_log]
        
        if trade_returns:
            # Un trade è vincente se il suo rendimento supera i costi di transazione (andata e ritorno)
            winning_trades = sum(1 for r in trade_returns if r > (self.fees * 2))
            win_rate = (winning_trades / len(trade_returns)) * 100 if trade_returns else 0.0
            
            gross_profits = sum(r for r in trade_returns if r > 0)
            gross_losses = abs(sum(r for r in trade_returns if r < 0))
            profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        else:
            win_rate = 0.0
            profit_factor = 0.0
            
        return {
            'total_return_%': float(total_return),
            'max_drawdown_%': abs(float(max_drawdown)),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'calmar_ratio': float(calmar_ratio),
            'total_trades': len(trade_returns),
            'win_rate_%': float(win_rate),
            'profit_factor': float(profit_factor)
        }
        
    # ================================================================= #
    #                     <<< SEZIONE MODIFICATA 2 >>>                    #
    # ================================================================= #
    def save_backtest_results(self, results: Dict) -> str:
        """
        Save backtest results to JSON file inside the data_path directory.
        """
        # Costruiamo il percorso completo partendo da self.data_path
        filepath = os.path.join(self.data_path, 'backtest_results.json')
        
        serializable_results = {}
        if 'in_sample_metrics' in results:
            serializable_results['in_sample'] = results['in_sample_metrics']
        if 'out_of_sample_metrics' in results:
            serializable_results['out_of_sample'] = results['out_of_sample_metrics']
            
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        print(f"💾 Backtest results saved to {filepath}")
        return filepath
    # ================================================================= #
