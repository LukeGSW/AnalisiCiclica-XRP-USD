"""
Streamlit Dashboard for Kriterion Quant Trading System
Enhanced version with multi-ticker support and date selection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
import sys

# Add src directory to path
sys.path.insert(0, 'src')

from config import Config
from data_fetcher import DataFetcher
from cycle_analyzer import CycleAnalyzer
from signal_generator import SignalGenerator
from backtester import Backtester

# Page configuration
st.set_page_config(
    page_title="Kriterion Quant Multi-Ticker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for ticker management
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = Config.TICKER
if 'analysis_cache' not in st.session_state:
    st.session_state.analysis_cache = {}

# Popular tickers for quick selection
POPULAR_TICKERS = {
    'ETFs': ['SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'SLV', 'TLT', 'XLE', 'XLF', 'VXX'],
    'Stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'V', 'JNJ'],
    'Commodities': ['GLD', 'SLV', 'USO', 'UNG', 'DBA', 'CORN', 'WEAT', 'SOYB'],
    'Bonds': ['TLT', 'IEF', 'SHY', 'AGG', 'BND', 'HYG', 'LQD', 'EMB'],
    'International': ['EWJ', 'EWG', 'EWU', 'FXI', 'EEM', 'EFA', 'INDA', 'EWZ']
}

def run_analysis_for_ticker(ticker, start_date=None, end_date=None):
    """Run analysis for a specific ticker with custom dates"""
    try:
        with st.spinner(f'🔄 Analyzing {ticker}... This may take 1-2 minutes'):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Override config temporarily
            original_ticker = Config.TICKER
            original_start = Config.START_DATE
            original_end = Config.END_DATE
            
            Config.TICKER = ticker
            if start_date:
                Config.START_DATE = start_date
            if end_date:
                Config.END_DATE = end_date
            
            # Step 1: Fetch data
            status_text.text(f'📡 Fetching data for {ticker}...')
            progress_bar.progress(20)
            
            fetcher = DataFetcher()
            df = fetcher.fetch_historical_data(
                ticker=ticker,
                start_date=Config.START_DATE,
                end_date=Config.END_DATE
            )
            
            # Save data with ticker-specific filename
            ticker_data_file = f'data/{ticker}_historical.csv'
            df.to_csv(ticker_data_file)
            
            # Step 2: Cycle analysis
            status_text.text('🔄 Performing cycle analysis...')
            progress_bar.progress(40)
            
            analyzer = CycleAnalyzer()
            df_analyzed = analyzer.analyze_cycle(df)
            
            # Step 3: Generate signals
            status_text.text('🎯 Generating trading signals...')
            progress_bar.progress(60)
            
            generator = SignalGenerator()
            df_signals = generator.generate_signals(df_analyzed)
            
            # Save with ticker-specific filename
            signals_file = f'data/{ticker}_signals.csv'
            df_signals.to_csv(signals_file)
            
            # Get latest signal
            latest_signal = generator.get_latest_signal(df_signals)
            
            # Step 4: Run backtest
            status_text.text('📊 Running backtest...')
            progress_bar.progress(80)
            
            backtester = Backtester()
            wf_results = backtester.run_walk_forward_analysis(df_signals)
            
            # Save backtest results
            backtest_file = f'data/{ticker}_backtest.json'
            backtester.save_backtest_results(wf_results, backtest_file)
            
            # Step 5: Complete
            status_text.text(f'✅ Analysis complete for {ticker}!')
            progress_bar.progress(100)
            
            # Cache results
            st.session_state.analysis_cache[ticker] = {
                'signals': df_signals,
                'latest_signal': latest_signal,
                'backtest': wf_results,
                'timestamp': datetime.now()
            }
            
            # Restore original config
            Config.TICKER = original_ticker
            Config.START_DATE = original_start
            Config.END_DATE = original_end
            
            return True, f"Analysis completed for {ticker}"
            
    except Exception as e:
        # Restore original config on error
        Config.TICKER = original_ticker
        Config.START_DATE = original_start
        Config.END_DATE = original_end
        return False, f"Error analyzing {ticker}: {str(e)}"

def load_ticker_data(ticker):
    """Load data for a specific ticker"""
    data = {}
    
    # Check cache first
    if ticker in st.session_state.analysis_cache:
        cache_data = st.session_state.analysis_cache[ticker]
        # Check if cache is recent (less than 24 hours old)
        if (datetime.now() - cache_data['timestamp']).total_seconds() < 86400:
            return cache_data
    
    # Try to load from files
    signals_file = f'data/{ticker}_signals.csv'
    if os.path.exists(signals_file):
        data['signals'] = pd.read_csv(signals_file, index_col='date', parse_dates=True)
        
        # Load backtest results
        backtest_file = f'data/{ticker}_backtest.json'
        if os.path.exists(backtest_file):
            with open(backtest_file, 'r') as f:
                data['backtest'] = json.load(f)
        
        # Extract latest signal
        if 'signals' in data and not data['signals'].empty:
            latest_row = data['signals'].iloc[-1]
            data['latest_signal'] = {
                'date': data['signals'].index[-1].strftime('%Y-%m-%d'),
                'signal': latest_row.get('signal', 'HOLD'),
                'position': 'LONG' if latest_row.get('position', 0) == 1 else 'FLAT',
                'phase_quadrant': latest_row.get('phase_quadrant', 'Unknown'),
                'phase_value': float(latest_row.get('phase', 0)),
                'oscillator_value': float(latest_row.get('oscillator', 0)),
                'signal_strength': float(latest_row.get('signal_strength', 50)),
                'confidence': latest_row.get('confidence', 'MEDIUM'),
                'price': float(latest_row['close'])
            }
        
        data['timestamp'] = datetime.now()
        return data
    
    return None

def create_comparison_table(tickers_list):
    """Create comparison table for multiple tickers"""
    comparison_data = []
    
    for ticker in tickers_list:
        data = load_ticker_data(ticker)
        if data and 'latest_signal' in data:
            signal_info = data['latest_signal']
            
            # Get backtest metrics if available
            metrics = {}
            if 'backtest' in data:
                if 'out_of_sample_metrics' in data['backtest']:
                    metrics = data['backtest']['out_of_sample_metrics']
                elif 'metrics' in data['backtest']:
                    metrics = data['backtest']['metrics']
            
            comparison_data.append({
                'Ticker': ticker,
                'Signal': signal_info['signal'],
                'Position': signal_info['position'],
                'Price': f"${signal_info['price']:.2f}",
                'Strength': f"{signal_info['signal_strength']:.1f}",
                'Return %': f"{metrics.get('total_return_%', 0):.2f}%",
                'Sharpe': f"{metrics.get('sharpe_ratio', 0):.2f}",
                'Win Rate': f"{metrics.get('win_rate_%', 0):.1f}%"
            })
    
    if comparison_data:
        return pd.DataFrame(comparison_data)
    return None

def main():
    """Main dashboard function with multi-ticker support"""
    
    # Header
    st.title("🎯 Kriterion Quant Multi-Ticker System")
    st.markdown("### Cycle-Based Trading Strategy Analysis")
    
    # Sidebar for ticker selection and configuration
    with st.sidebar:
        st.header("📊 Ticker Selection")
        
        # Quick selection from categories
        category = st.selectbox(
            "Select Category",
            ['Custom'] + list(POPULAR_TICKERS.keys())
        )
        
        if category != 'Custom':
            ticker_list = POPULAR_TICKERS[category]
            selected_ticker = st.selectbox(
                "Select Ticker",
                ticker_list,
                index=ticker_list.index(st.session_state.selected_ticker) 
                if st.session_state.selected_ticker in ticker_list else 0
            )
        else:
            selected_ticker = st.text_input(
                "Enter Custom Ticker",
                value=st.session_state.selected_ticker
            ).upper()
        
        st.session_state.selected_ticker = selected_ticker
        
        # Date range configuration
        st.header("📅 Analysis Period")
        
        col1, col2 = st.columns(2)
        with col1:
            lookback_years = st.number_input(
                "Lookback Years",
                min_value=1,
                max_value=20,
                value=5,
                step=1
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                max_value=datetime.now()
            )
        
        start_date = end_date - timedelta(days=lookback_years * 365)
        st.info(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Analysis parameters
        st.header("⚙️ Parameters")
        
        with st.expander("Analysis Settings", expanded=False):
            fast_ma = st.slider("Fast MA Window", 5, 20, Config.FAST_MA_WINDOW)
            slow_ma = st.slider("Slow MA Window", 20, 60, Config.SLOW_MA_WINDOW)
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000,
                max_value=1000000,
                value=Config.INITIAL_CAPITAL,
                step=1000
            )
        
        # Run analysis button
        st.markdown("---")
        if st.button(f"🚀 Analyze {selected_ticker}", use_container_width=True, type="primary"):
            # Update config with new parameters
            Config.FAST_MA_WINDOW = fast_ma
            Config.SLOW_MA_WINDOW = slow_ma
            Config.INITIAL_CAPITAL = initial_capital
            
            success, message = run_analysis_for_ticker(
                selected_ticker,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
        
        # Multi-ticker comparison
        st.markdown("---")
        st.header("🔍 Compare Tickers")
        
        compare_tickers = st.multiselect(
            "Select tickers to compare",
            options=['SPY', 'QQQ', 'IWM', 'GLD', 'SLV', 'TLT', 'XLE', 'XLF'],
            default=[]
        )
        
        if st.button("📊 Run Comparison", use_container_width=True):
            for ticker in compare_tickers:
                if ticker not in st.session_state.analysis_cache:
                    success, _ = run_analysis_for_ticker(
                        ticker,
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d')
                    )
    
    # Main content area
    # Load data for selected ticker
    ticker_data = load_ticker_data(selected_ticker)
    
    if ticker_data is None:
        st.warning(f"⚠️ No data found for {selected_ticker}. Click 'Analyze {selected_ticker}' to generate.")
        
        # Show comparison table if available
        if compare_tickers:
            st.header("📊 Ticker Comparison")
            comparison_df = create_comparison_table(compare_tickers)
            if comparison_df is not None:
                st.dataframe(comparison_df, use_container_width=True)
        return
    
    # Display analysis results
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Current Status", 
        "📈 Analysis", 
        "🎯 Backtest", 
        "📋 History",
        "🔍 Comparison"
    ])
    
    with tab1:
        st.header(f"Current Status - {selected_ticker}")
        
        if 'latest_signal' in ticker_data:
            latest = ticker_data['latest_signal']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                signal_color = "🟢" if latest['signal'] == 'BUY' else "🔴" if latest['signal'] == 'SELL' else "⏸️"
                st.metric(
                    "Last Signal",
                    f"{signal_color} {latest['signal']}",
                    f"on {latest['date']}"
                )
            
            with col2:
                position_emoji = "💰" if latest['position'] == 'LONG' else "💤"
                st.metric(
                    "Position",
                    f"{position_emoji} {latest['position']}",
                    f"${latest['price']:.2f}"
                )
            
            with col3:
                st.metric(
                    "Signal Strength",
                    f"{latest['signal_strength']:.1f}/100",
                    f"{latest['confidence']} confidence"
                )
            
            with col4:
                st.metric(
                    "Cycle Phase",
                    f"{latest['phase_value']:.2f} rad",
                    latest['phase_quadrant']
                )
    
    with tab2:
        st.header(f"Cycle Analysis - {selected_ticker}")
        
        if 'signals' in ticker_data:
            df_signals = ticker_data['signals']
            
            # Price chart with signals
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.5, 0.25, 0.25],
                subplot_titles=(f'{selected_ticker} Price & Signals', 'Oscillator', 'Phase')
            )
            
            # Add price
            fig.add_trace(
                go.Scatter(x=df_signals.index, y=df_signals['close'], name='Price'),
                row=1, col=1
            )
            
            # Add signals
            buy_signals = df_signals[df_signals['signal'] == 'BUY']
            sell_signals = df_signals[df_signals['signal'] == 'SELL']
            
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index, y=buy_signals['close'],
                    mode='markers', name='Buy',
                    marker=dict(symbol='triangle-up', size=10, color='green')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=sell_signals.index, y=sell_signals['close'],
                    mode='markers', name='Sell',
                    marker=dict(symbol='triangle-down', size=10, color='red')
                ),
                row=1, col=1
            )
            
            # Add oscillator
            if 'oscillator' in df_signals.columns:
                fig.add_trace(
                    go.Scatter(x=df_signals.index, y=df_signals['oscillator'], name='Oscillator'),
                    row=2, col=1
                )
            
            # Add phase
            if 'phase' in df_signals.columns:
                fig.add_trace(
                    go.Scatter(x=df_signals.index, y=df_signals['phase'], name='Phase'),
                    row=3, col=1
                )
            
            fig.update_layout(height=800, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header(f"Backtest Results - {selected_ticker}")
        
        if 'backtest' in ticker_data and 'signals' in ticker_data:
            # Display metrics
            col1, col2 = st.columns(2)
            
            if 'out_of_sample_metrics' in ticker_data['backtest']:
                metrics = ticker_data['backtest']['out_of_sample_metrics']
            else:
                metrics = ticker_data['backtest'].get('metrics', {})
            
            with col1:
                st.metric("Total Return", f"{metrics.get('total_return_%', 0):.2f}%")
                st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                st.metric("Max Drawdown", f"{metrics.get('max_drawdown_%', 0):.2f}%")
            
            with col2:
                st.metric("Win Rate", f"{metrics.get('win_rate_%', 0):.1f}%")
                st.metric("Total Trades", f"{metrics.get('total_trades', 0):.0f}")
                st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
    
    with tab4:
        st.header(f"Trading History - {selected_ticker}")
        
        if 'signals' in ticker_data:
            df_signals = ticker_data['signals']
            recent_signals = df_signals[df_signals['signal'] != 'HOLD'].tail(20)
            
            if not recent_signals.empty:
                st.dataframe(recent_signals[['close', 'signal', 'phase_quadrant']], use_container_width=True)
    
    with tab5:
        st.header("Multi-Ticker Comparison")
        
        # Add current ticker to comparison if not already there
        all_tickers = list(set([selected_ticker] + compare_tickers))
        
        if all_tickers:
            comparison_df = create_comparison_table(all_tickers)
            if comparison_df is not None:
                st.dataframe(
                    comparison_df.style.highlight_max(subset=['Strength', 'Sharpe'], color='lightgreen')
                                      .highlight_min(subset=['Strength', 'Sharpe'], color='lightcoral'),
                    use_container_width=True
                )
                
                # Download comparison
                csv = comparison_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Comparison",
                    data=csv,
                    file_name=f"ticker_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    main()
