"""
Main analysis script for Kriterion Quant Trading System
Orchestrates data fetching, cycle analysis, signal generation, and notifications
"""

import sys
import os
import traceback
from datetime import datetime
import pandas as pd
import json

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import Config
from data_fetcher import DataFetcher
from cycle_analyzer import CycleAnalyzer
from signal_generator import SignalGenerator
from backtester import Backtester
from notifier import TelegramNotifier

def main():
    """Main execution function"""
    
    print("=" * 50)
    print(f"KRITERION QUANT TRADING SYSTEM")
    print(f"Ticker: {Config.TICKER}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    # Definiamo e creiamo il percorso dati specifico per il ticker
    base_data_dir = 'data'
    ticker_data_path = os.path.join(base_data_dir, Config.TICKER)
    os.makedirs(ticker_data_path, exist_ok=True)
    print(f"📂 I dati di output verranno salvati in: {ticker_data_path}")

    # ================================================================= #
    #    INIZIALIZZAZIONE UNICA DI TUTTI I COMPONENTI ALL'INIZIO          #
    # ================================================================= #
    notifier = TelegramNotifier()
    fetcher = DataFetcher(data_path=ticker_data_path)
    analyzer = CycleAnalyzer()
    generator = SignalGenerator(data_path=ticker_data_path)
    backtester = Backtester(data_path=ticker_data_path)
    # ================================================================= #
    
    try:
        # Validate configuration
        Config.validate()
        print("✅ Configuration validated")
        
        # Step 1: Fetch or update data
        print("\n" + "=" * 30)
        print("STEP 1: DATA ACQUISITION")
        print("=" * 30)
        
        # --- RIGA RIMOSSA: fetcher = DataFetcher() ---
        df = fetcher.update_latest_data(Config.TICKER)
        print(f"📊 Data range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"📊 Total data points: {len(df)}")
        
        # Step 2: Perform cycle analysis
        print("\n" + "=" * 30)
        print("STEP 2: CYCLE ANALYSIS")
        print("=" * 30)
        
        # --- RIGA RIMOSSA: analyzer = CycleAnalyzer() ---
        df_analyzed = analyzer.analyze_cycle(df)
        
        spectral_results = analyzer.run_spectral_analysis(df_analyzed['oscillator'])
        print(f"🔍 Dominant cycle period: {spectral_results['dominant_period']:.1f} days")
        
        monte_carlo_results = analyzer.run_monte_carlo_significance_test(df_analyzed['oscillator'])
        print(f"📊 Cycle significance p-value: {monte_carlo_results['p_value']:.4f}")
        
        if monte_carlo_results['significant']:
            print("✅ Cycle is statistically significant")
        else:
            print("⚠️ Cycle may not be statistically significant")
        
        # Step 3: Generate trading signals
        print("\n" + "=" * 30)
        print("STEP 3: SIGNAL GENERATION")
        print("=" * 30)
        
        # --- RIGA RIMOSSA: generator = SignalGenerator() ---
        df_signals = generator.generate_signals(df_analyzed)
        
        latest_signal = generator.get_latest_signal(df_signals)
        print(f"📍 Latest Signal: {latest_signal['signal']}")
        print(f"📍 Current Position: {latest_signal['position']}")
        print(f"📍 Signal Strength: {latest_signal['signal_strength']:.1f}/100")
        print(f"📍 Confidence: {latest_signal['confidence']}")
        
        generator.save_signals(df_signals)
        
        # Step 4: Run backtest
        print("\n" + "=" * 30)
        print("STEP 4: BACKTESTING")
        print("=" * 30)
        
        # --- RIGA RIMOSSA: backtester = Backtester() ---
        wf_results = backtester.run_walk_forward_analysis(df_signals)
        
        if 'in_sample' in wf_results:
            print("\n📊 In-Sample Performance:")
            is_metrics = wf_results['in_sample']['metrics']
            for key, value in is_metrics.items():
                print(f"  {key}: {value:.2f}")
            
            print("\n📊 Out-of-Sample Performance:")
            oos_metrics = wf_results['out_of_sample']['metrics']
            for key, value in oos_metrics.items():
                print(f"  {key}: {value:.2f}")
            
            backtester.save_backtest_results(wf_results)
            backtest_metrics = oos_metrics
        else:
            backtest_metrics = wf_results['aggregated_metrics']
            print("\n📊 Aggregated Out-of-Sample Performance:")
            for key, value in backtest_metrics.items():
                if 'avg_' in key:
                    print(f"  {key}: {value:.2f}")
        
        # ================================================================= #
        #                <<< INSERISCI QUESTO BLOCCO QUI >>>                  #
        # ================================================================= #
        # Step 5: Send notifications
        print("\n" + "=" * 30)
        print("STEP 5: NOTIFICATIONS")
        print("=" * 30)
        
        if Config.SEND_TELEGRAM_NOTIFICATIONS:
            # Invia l'alert del segnale solo se non è 'HOLD'
            if latest_signal['signal'] != 'HOLD':
                notifier.send_signal_alert(latest_signal)
        
            # Invia sempre il riepilogo giornaliero
            notifier.send_daily_summary(latest_signal, backtest_metrics)
            print("✅ Telegram notifications sent.")
        else:
            print("📵 Telegram notifications disabled")
        # ================================================================= #

        # Step 6: Create summary report
        print("\n" + "=" * 30)
        print("STEP 6: SUMMARY REPORT")
        print("=" * 30)
        
        summary = {
            'timestamp': datetime.now().isoformat(), 'ticker': Config.TICKER,
            'data_points': len(df_signals), 'latest_signal': latest_signal,
            'backtest_metrics': backtest_metrics,
            'cycle_analysis': {
                'dominant_period': spectral_results['dominant_period'],
                'p_value': monte_carlo_results['p_value'],
                'significant': monte_carlo_results['significant']
            }
        }
        
        summary_filename = 'analysis_summary.json'
        summary_file = os.path.join(ticker_data_path, summary_filename)
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"📄 Summary saved to {summary_file}")
        
        print("\n" + "=" * 50)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        error_msg = f"Error in main analysis: {str(e)}\n{traceback.format_exc()}"
        print(f"\n❌ {error_msg}")
        if Config.SEND_TELEGRAM_NOTIFICATIONS:
            notifier.send_error_alert(str(e))
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
