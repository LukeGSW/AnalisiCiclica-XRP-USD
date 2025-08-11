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
    
    # Initialize components
    notifier = TelegramNotifier()
    
    try:
        # Validate configuration
        Config.validate()
        print("✅ Configuration validated")
        
        # Step 1: Fetch or update data
        print("\n" + "=" * 30)
        print("STEP 1: DATA ACQUISITION")
        print("=" * 30)
        
        fetcher = DataFetcher()
        df = fetcher.update_latest_data(Config.TICKER)
        print(f"📊 Data range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"📊 Total data points: {len(df)}")
        
        # Step 2: Perform cycle analysis
        print("\n" + "=" * 30)
        print("STEP 2: CYCLE ANALYSIS")
        print("=" * 30)
        
        analyzer = CycleAnalyzer()
        df_analyzed = analyzer.analyze_cycle(df)
        
        # Run spectral analysis for validation
        spectral_results = analyzer.run_spectral_analysis(df_analyzed['oscillator'])
        print(f"🔍 Dominant cycle period: {spectral_results['dominant_period']:.1f} days")
        
        # Test statistical significance
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
        
        generator = SignalGenerator()
        df_signals = generator.generate_signals(df_analyzed)
        
        # Get latest signal
        latest_signal = generator.get_latest_signal(df_signals)
        print(f"📍 Latest Signal: {latest_signal['signal']}")
        print(f"📍 Current Position: {latest_signal['position']}")
        print(f"📍 Signal Strength: {latest_signal['signal_strength']:.1f}/100")
        print(f"📍 Confidence: {latest_signal['confidence']}")
        
        # Save signals
        generator.save_signals(df_signals)
        
        # Step 4: Run backtest
        print("\n" + "=" * 30)
        print("STEP 4: BACKTESTING")
        print("=" * 30)
        
        backtester = Backtester()
        
        # Run walk-forward analysis
        wf_results = backtester.run_walk_forward_analysis(df_signals)
        
        # Display results
        if 'in_sample' in wf_results:
            print("\n📊 In-Sample Performance:")
            is_metrics = wf_results['in_sample']['metrics']
            for key, value in is_metrics.items():
                print(f"  {key}: {value:.2f}")
            
            print("\n📊 Out-of-Sample Performance:")
            oos_metrics = wf_results['out_of_sample']['metrics']
            for key, value in oos_metrics.items():
                print(f"  {key}: {value:.2f}")
            
            # Save backtest results
            backtester.save_backtest_results(wf_results)
            
            # Use OOS metrics for notifications
            backtest_metrics = oos_metrics
        else:
            # Use aggregated metrics from rolling walk-forward
            backtest_metrics = wf_results['aggregated_metrics']
            print("\n📊 Aggregated Out-of-Sample Performance:")
            for key, value in backtest_metrics.items():
                if 'avg_' in key:
                    print(f"  {key}: {value:.2f}")
        
        # Step 5: Send notifications
        print("\n" + "=" * 30)
        print("STEP 5: NOTIFICATIONS")
        print("=" * 30)
        
        if Config.SEND_TELEGRAM_NOTIFICATIONS:
            # Send signal alert if there's a new signal
            if latest_signal['signal'] != 'HOLD':
                notifier.send_signal_alert(latest_signal)
            
            # Send daily summary
            notifier.send_daily_summary(latest_signal, backtest_metrics)
        else:
            print("📵 Telegram notifications disabled")
        
        # Step 6: Create summary report
        print("\n" + "=" * 30)
        print("STEP 6: SUMMARY REPORT")
        print("=" * 30)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'ticker': Config.TICKER,
            'data_points': len(df_signals),
            'latest_signal': latest_signal,
            'backtest_metrics': backtest_metrics,
            'cycle_analysis': {
                'dominant_period': spectral_results['dominant_period'],
                'p_value': monte_carlo_results['p_value'],
                'significant': monte_carlo_results['significant']
            }
        }
        
        # Save summary
        summary_file = os.path.join(Config.DATA_DIR, 'analysis_summary.json')
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
        
        # Send error notification
        if Config.SEND_TELEGRAM_NOTIFICATIONS:
            notifier.send_error_alert(str(e))
        
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
