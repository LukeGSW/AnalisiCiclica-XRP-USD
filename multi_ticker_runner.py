"""
Multi-Ticker Analysis Runner
Allows running analysis for multiple tickers with custom parameters
"""

import sys
import os
import argparse
import json
from datetime import datetime, timedelta
import time

# Add src to path
sys.path.insert(0, 'src')

from config import Config
from data_fetcher import DataFetcher
from cycle_analyzer import CycleAnalyzer
from signal_generator import SignalGenerator
from backtester import Backtester
from notifier import TelegramNotifier

def analyze_ticker(ticker, start_date=None, end_date=None, lookback_years=5):
    """
    Analyze a single ticker
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    start_date : str, optional
        Start date in YYYY-MM-DD format
    end_date : str, optional
        End date in YYYY-MM-DD format
    lookback_years : int, optional
        Years to look back if start_date not provided
    """
    print(f"\n{'='*50}")
    print(f"ANALYZING: {ticker}")
    print(f"{'='*50}")
    
    # Set dates
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if not start_date:
        start_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=lookback_years*365)
        start_date = start_dt.strftime('%Y-%m-%d')
    
    print(f"Period: {start_date} to {end_date}")
    
    try:
        # 1. Fetch data
        print(f"\n📡 Fetching data for {ticker}...")
        fetcher = DataFetcher()
        df = fetcher.fetch_historical_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date
        )
        
        # Save with ticker-specific name
        data_file = f'data/{ticker}_historical.csv'
        df.to_csv(data_file)
        print(f"💾 Data saved to {data_file}")
        
        # 2. Cycle analysis
        print(f"\n🔄 Performing cycle analysis...")
        analyzer = CycleAnalyzer()
        df_analyzed = analyzer.analyze_cycle(df)
        
        # Get spectral analysis
        spectral = analyzer.run_spectral_analysis(df_analyzed['oscillator'])
        monte_carlo = analyzer.run_monte_carlo_significance_test(df_analyzed['oscillator'])
        
        print(f"  Dominant period: {spectral['dominant_period']:.1f} days")
        print(f"  P-value: {monte_carlo['p_value']:.4f}")
        print(f"  Significant: {'✅ Yes' if monte_carlo['significant'] else '⚠️ No'}")
        
        # 3. Generate signals
        print(f"\n🎯 Generating signals...")
        generator = SignalGenerator()
        df_signals = generator.generate_signals(df_analyzed)
        
        # Save signals with ticker name
        signals_file = f'data/{ticker}_signals.csv'
        df_signals.to_csv(signals_file)
        print(f"💾 Signals saved to {signals_file}")
        
        # Get latest signal
        latest_signal = generator.get_latest_signal(df_signals)
        print(f"\n📍 Latest Signal: {latest_signal['signal']}")
        print(f"📍 Position: {latest_signal['position']}")
        print(f"📍 Price: ${latest_signal['price']:.2f}")
        print(f"📍 Strength: {latest_signal['signal_strength']:.1f}/100")
        
        # Save latest signal
        latest_file = f'data/{ticker}_latest.json'
        with open(latest_file, 'w') as f:
            json.dump(latest_signal, f, indent=2, default=str)
        
        # 4. Run backtest
        print(f"\n📊 Running backtest...")
        backtester = Backtester()
        wf_results = backtester.run_walk_forward_analysis(df_signals)
        
        # Save backtest results
        backtest_file = f'data/{ticker}_backtest.json'
        backtester.save_backtest_results(wf_results, backtest_file)
        
        # Display key metrics
        if 'out_of_sample' in wf_results:
            metrics = wf_results['out_of_sample']['metrics']
            print(f"\n📈 Out-of-Sample Performance:")
            print(f"  Total Return: {metrics['total_return_%']:.2f}%")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {metrics['max_drawdown_%']:.2f}%")
            print(f"  Win Rate: {metrics['win_rate_%']:.1f}%")
        
        # 5. Send notification
        notifier = TelegramNotifier()
        if notifier.enabled:
            # Format message for this ticker
            message = f"""
🎯 **{ticker} Analysis Complete**
{'='*30}

**Signal:** {latest_signal['signal']} {
    '🟢' if latest_signal['signal'] == 'BUY' else 
    '🔴' if latest_signal['signal'] == 'SELL' else '⏸️'
}
**Position:** {latest_signal['position']}
**Price:** ${latest_signal['price']:.2f}
**Strength:** {latest_signal['signal_strength']:.1f}/100

**Performance (OOS):**
Return: {metrics.get('total_return_%', 0):.2f}%
Sharpe: {metrics.get('sharpe_ratio', 0):.2f}
Win Rate: {metrics.get('win_rate_%', 0):.1f}%

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
            notifier.send_message(message)
        
        print(f"\n✅ Analysis complete for {ticker}")
        return True
        
    except Exception as e:
        print(f"\n❌ Error analyzing {ticker}: {str(e)}")
        
        # Send error notification
        notifier = TelegramNotifier()
        if notifier.enabled:
            notifier.send_error_alert(f"Failed to analyze {ticker}: {str(e)}")
        
        return False

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Multi-Ticker Analysis Runner')
    
    parser.add_argument(
        '--ticker',
        type=str,
        help='Single ticker to analyze'
    )
    
    parser.add_argument(
        '--tickers',
        type=str,
        help='Comma-separated list of tickers (e.g., SPY,QQQ,GLD)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--lookback-years',
        type=int,
        default=5,
        help='Years to look back (default: 5)'
    )
    
    parser.add_argument(
        '--delay',
        type=int,
        default=5,
        help='Delay in seconds between tickers to avoid API limits (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Determine which tickers to analyze
    tickers_to_analyze = []
    
    if args.ticker:
        tickers_to_analyze = [args.ticker.upper()]
    elif args.tickers:
        tickers_to_analyze = [t.strip().upper() for t in args.tickers.split(',')]
    else:
        # Default to environment variable or config
        ticker_env = os.getenv('TICKER', Config.TICKER)
        tickers_to_analyze = [ticker_env]
    
    print(f"📊 Tickers to analyze: {', '.join(tickers_to_analyze)}")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Analyze each ticker
    results = {}
    for i, ticker in enumerate(tickers_to_analyze):
        if i > 0:
            print(f"\n⏳ Waiting {args.delay} seconds before next ticker...")
            time.sleep(args.delay)
        
        success = analyze_ticker(
            ticker,
            start_date=args.start_date,
            end_date=args.end_date,
            lookback_years=args.lookback_years
        )
        results[ticker] = success
    
    # Summary
    print(f"\n{'='*50}")
    print("ANALYSIS SUMMARY")
    print(f"{'='*50}")
    
    successful = [t for t, s in results.items() if s]
    failed = [t for t, s in results.items() if not s]
    
    if successful:
        print(f"✅ Successful: {', '.join(successful)}")
    if failed:
        print(f"❌ Failed: {', '.join(failed)}")
    
    # Send summary notification
    if len(tickers_to_analyze) > 1:
        notifier = TelegramNotifier()
        if notifier.enabled:
            summary_msg = f"""
📊 **Multi-Ticker Analysis Complete**
{'='*30}

✅ Successful: {len(successful)}/{len(tickers_to_analyze)}
{', '.join(successful) if successful else 'None'}

{'❌ Failed: ' + ', '.join(failed) if failed else ''}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
            notifier.send_message(summary_msg)
    
    # Exit with appropriate code
    sys.exit(0 if all(results.values()) else 1)

if __name__ == "__main__":
    main()
