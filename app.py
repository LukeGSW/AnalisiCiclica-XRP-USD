"""
Streamlit Dashboard for Kriterion Quant Trading System
Version with simple ticker selection
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
    page_title="Kriterion Quant Trading",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Fix per metrics visibility */
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    
    [data-testid="metric-container"] > div:nth-child(1) {
        color: #666666 !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }
    
    [data-testid="metric-container"] > div:nth-child(2) {
        color: #1f1f1f !important;
        font-size: 24px !important;
        font-weight: bold !important;
    }
    
    [data-testid="metric-container"] > div:nth-child(3) {
        color: #666666 !important;
        font-size: 12px !important;
    }
    
    .stButton > button {
        background-color: #4a90e2;
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 5px;
        font-weight: 600;
    }
    
    .stButton > button:hover {
        background-color: #357abd;
    }
</style>
""", unsafe_allow_html=True)

# Popular tickers for quick selection
POPULAR_TICKERS = ['GLD', 'SPY', 'QQQ', 'IWM', 'SLV', 'TLT', 'XLE', 'XLF', 'AAPL', 'MSFT']

def run_analysis_for_ticker(ticker_symbol):
    """Run analysis for a specific ticker"""
    try:
        with st.spinner(f'🔄 Analyzing {ticker_symbol}... This may take 1-2 minutes'):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Update Config
            original_ticker = Config.TICKER
            Config.TICKER = ticker_symbol
            
            # Step 1: Fetch data
            status_text.text(f'📡 Fetching data for {ticker_symbol}...')
            progress_bar.progress(20)
            
            fetcher = DataFetcher()
            df = fetcher.update_latest_data(ticker_symbol)
            
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
            
            # Save with ticker-specific name
            signals_file = f'data/{ticker_symbol}_signals.csv'
            df_signals.to_csv(signals_file)
            
            # Save latest signal
            latest_signal = generator.get_latest_signal(df_signals)
            latest_file = f'data/{ticker_symbol}_latest.json'
            with open(latest_file, 'w') as f:
                json.dump(latest_signal, f, indent=2, default=str)
            
            # Step 4: Run backtest
            status_text.text('📊 Running backtest...')
            progress_bar.progress(80)
            
            backtester = Backtester()
            wf_results = backtester.run_walk_forward_analysis(df_signals)
            
            # Save backtest
            backtest_file = f'data/{ticker_symbol}_backtest.json'
            backtester.save_backtest_results(wf_results, backtest_file)
            
            # Complete
            status_text.text(f'✅ Analysis complete for {ticker_symbol}!')
            progress_bar.progress(100)
            
            # Restore original ticker
            Config.TICKER = original_ticker
            
            return True, f"Analysis completed for {ticker_symbol}"
            
    except Exception as e:
        Config.TICKER = original_ticker
        return False, f"Error: {str(e)}"

def load_ticker_data(ticker_symbol):
    """Load data for a specific ticker"""
    data = {}
    
    # Try ticker-specific files first
    signals_file = f'data/{ticker_symbol}_signals.csv'
    if not os.path.exists(signals_file):
        # Fall back to default
        signals_file = Config.SIGNALS_FILE
    
    if os.path.exists(signals_file):
        data['signals'] = pd.read_csv(signals_file, index_col='date', parse_dates=True)
        
        # Load latest signal
        latest_file = f'data/{ticker_symbol}_latest.json'
        if not os.path.exists(latest_file):
            latest_file = signals_file.replace('.csv', '_latest.json')
        
        if os.path.exists(latest_file):
            with open(latest_file, 'r') as f:
                data['latest_signal'] = json.load(f)
        
        # Load backtest
        backtest_file = f'data/{ticker_symbol}_backtest.json'
        if not os.path.exists(backtest_file):
            backtest_file = Config.BACKTEST_RESULTS_FILE
        
        if os.path.exists(backtest_file):
            with open(backtest_file, 'r') as f:
                data['backtest'] = json.load(f)
        
        return data
    
    return None

def create_price_chart(df):
    """Create interactive price chart with signals"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=('Price & Signals', 'Cycle Oscillator', 'Phase')
    )
    
    # Price chart
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df['close'],
            name='Close Price',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Add buy signals
    buy_signals = df[df['signal'] == 'BUY']
    if not buy_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=buy_signals.index,
                y=buy_signals['close'],
                mode='markers',
                name='Buy Signal',
                marker=dict(symbol='triangle-up', size=12, color='green')
            ),
            row=1, col=1
        )
    
    # Add sell signals
    sell_signals = df[df['signal'] == 'SELL']
    if not sell_signals.empty:
        fig.add_trace(
            go.Scatter(
                x=sell_signals.index,
                y=sell_signals['close'],
                mode='markers',
                name='Sell Signal',
                marker=dict(symbol='triangle-down', size=12, color='red')
            ),
            row=1, col=1
        )
    
    # Oscillator
    if 'oscillator' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['oscillator'],
                name='Oscillator',
                line=dict(color='orange', width=1)
            ),
            row=2, col=1
        )
        fig.add_hline(y=0, row=2, col=1, line_dash="dash", line_color="gray")
    
    # Phase
    if 'phase' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['phase'],
                name='Phase',
                line=dict(color='green', width=1)
            ),
            row=3, col=1
        )
        fig.add_hline(y=np.pi/2, row=3, col=1, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_hline(y=0, row=3, col=1, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_hline(y=-np.pi/2, row=3, col=1, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        height=800,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white'
    )
    
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Oscillator", row=2, col=1)
    fig.update_yaxes(title_text="Phase (rad)", row=3, col=1)
    
    return fig

def create_equity_chart(df):
    """Create equity curve chart"""
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['position'].shift(1) * df['returns']
    df['cumulative_strategy'] = (1 + df['strategy_returns']).cumprod()
    df['cumulative_buy_hold'] = (1 + df['returns']).cumprod()
    
    initial_capital = float(Config.INITIAL_CAPITAL)
    df['equity'] = df['cumulative_strategy'] * initial_capital
    df['buy_hold_equity'] = df['cumulative_buy_hold'] * initial_capital
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['equity'],
            name='Strategy',
            line=dict(color='blue', width=2)
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['buy_hold_equity'],
            name='Buy & Hold',
            line=dict(color='gray', width=1, dash='dash')
        )
    )
    
    fig.update_layout(
        title='Equity Curve Comparison',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        height=400,
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def main():
    """Main dashboard function"""
    
    # Header
    st.title("🎯 Kriterion Quant Trading System")
    
    # Initialize session state for ticker
    if 'selected_ticker' not in st.session_state:
        st.session_state.selected_ticker = Config.TICKER
    
    # Ticker selection in header
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"### Cycle-Based Trading Strategy")
    
    with col2:
        # Ticker selection dropdown
        selected_ticker = st.selectbox(
            "Select Ticker",
            options=POPULAR_TICKERS,
            index=POPULAR_TICKERS.index(st.session_state.selected_ticker) 
            if st.session_state.selected_ticker in POPULAR_TICKERS else 0,
            key="ticker_selector"
        )
        st.session_state.selected_ticker = selected_ticker
    
    with col3:
        # Custom ticker input
        custom_ticker = st.text_input("Or enter custom", placeholder="e.g., NVDA")
        if custom_ticker:
            st.session_state.selected_ticker = custom_ticker.upper()
            selected_ticker = custom_ticker.upper()
    
    # Load data for selected ticker
    data = load_ticker_data(selected_ticker)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Show current ticker info
        st.info(f"""
        **Ticker:** {selected_ticker}  
        **Fast MA:** {Config.FAST_MA_WINDOW}  
        **Slow MA:** {Config.SLOW_MA_WINDOW}  
        **Initial Capital:** ${float(Config.INITIAL_CAPITAL):,.0f}  
        **Trading Fees:** {float(Config.TRADING_FEES)*100:.1f}%
        """)
        
        # Analysis button for selected ticker
        st.markdown("---")
        st.markdown(f"### 🔄 Analysis for {selected_ticker}")
        
        if st.button(f"Run Analysis for {selected_ticker}", use_container_width=True, type="primary"):
            success, message = run_analysis_for_ticker(selected_ticker)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
        
        # Quick analysis for multiple tickers
        st.markdown("---")
        st.markdown("### 🔍 Quick Analysis")
        
        multi_tickers = st.multiselect(
            "Select multiple tickers",
            options=['SPY', 'QQQ', 'IWM', 'SLV', 'TLT', 'XLE', 'XLF'],
            default=[]
        )
        
        if st.button("Analyze Selected", use_container_width=True):
            for ticker in multi_tickers:
                with st.spinner(f'Analyzing {ticker}...'):
                    success, msg = run_analysis_for_ticker(ticker)
                    if success:
                        st.success(f"✅ {ticker}")
                    else:
                        st.error(f"❌ {ticker}: {msg}")
        
        # Date range filter (if data exists)
        if data and 'signals' in data:
            st.markdown("---")
            st.header("📅 Date Range")
            df_signals = data['signals']
            
            date_range = st.date_input(
                "Select date range",
                value=(df_signals.index[0], df_signals.index[-1]),
                min_value=df_signals.index[0],
                max_value=df_signals.index[-1]
            )
            
            if len(date_range) == 2:
                mask = (df_signals.index >= pd.Timestamp(date_range[0])) & (df_signals.index <= pd.Timestamp(date_range[1]))
                df_filtered = df_signals.loc[mask]
            else:
                df_filtered = df_signals
        else:
            df_filtered = None
    
    # Main content
    if data is None:
        st.warning(f"⚠️ No data found for {selected_ticker}.")
        st.info(f"Click 'Run Analysis for {selected_ticker}' in the sidebar to generate data.")
        return
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Current Status", "📈 Analysis", "🎯 Backtest", "📋 History"])
    
    with tab1:
        st.header(f"Current Trading Status - {selected_ticker}")
        
        if 'latest_signal' in data:
            latest = data['latest_signal']
            
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
                    f"${float(latest['price']):.2f}"
                )
            
            with col3:
                st.metric(
                    "Signal Strength",
                    f"{float(latest['signal_strength']):.1f}/100",
                    f"{latest['confidence']} confidence"
                )
            
            with col4:
                st.metric(
                    "Cycle Phase",
                    f"{float(latest['phase_value']):.2f} rad",
                    latest['phase_quadrant']
                )
            
            st.subheader("📍 Signal Details")
            details_col1, details_col2 = st.columns(2)
            
            with details_col1:
                st.info(f"""
                **Oscillator Value:** {float(latest['oscillator_value']):.4f}  
                **Phase Quadrant:** {latest['phase_quadrant']}  
                **Generated:** {latest.get('timestamp', 'N/A')}
                """)
            
            with details_col2:
                if latest['signal'] == 'BUY':
                    st.success("🎯 **Action:** Enter Long Position")
                elif latest['signal'] == 'SELL':
                    st.warning("🎯 **Action:** Exit Long Position")
                else:
                    st.info("🎯 **Action:** Maintain Current Position")
    
    with tab2:
        st.header(f"Cycle Analysis - {selected_ticker}")
        
        if df_filtered is not None:
            st.subheader("📈 Price Chart with Signals")
            fig_price = create_price_chart(df_filtered)
            st.plotly_chart(fig_price, use_container_width=True)
    
    with tab3:
        st.header(f"Backtest Results - {selected_ticker}")
        
        if df_filtered is not None:
            st.subheader("💰 Equity Curve")
            fig_equity = create_equity_chart(df_filtered)
            st.plotly_chart(fig_equity, use_container_width=True)
        
        if 'backtest' in data:
            st.subheader("📊 Performance Metrics")
            
            col1, col2 = st.columns(2)
            
            if 'out_of_sample_metrics' in data['backtest']:
                metrics = data['backtest']['out_of_sample_metrics']
            elif 'metrics' in data['backtest']:
                metrics = data['backtest']['metrics']
            else:
                metrics = {}
            
            with col1:
                st.metric("Total Return", f"{float(metrics.get('total_return_%', 0)):.2f}%")
                st.metric("Max Drawdown", f"{float(metrics.get('max_drawdown_%', 0)):.2f}%")
                st.metric("Sharpe Ratio", f"{float(metrics.get('sharpe_ratio', 0)):.2f}")
            
            with col2:
                st.metric("Win Rate", f"{float(metrics.get('win_rate_%', 0)):.1f}%")
                st.metric("Total Trades", f"{int(metrics.get('total_trades', 0))}")
                st.metric("Profit Factor", f"{float(metrics.get('profit_factor', 0)):.2f}")
    
    with tab4:
        st.header(f"Trading History - {selected_ticker}")
        
        if df_filtered is not None:
            recent_signals = df_filtered[df_filtered['signal'] != 'HOLD'].tail(20)
            
            if not recent_signals.empty:
                st.dataframe(recent_signals[['close', 'signal', 'phase_quadrant']], use_container_width=True)
            else:
                st.info("No signals in the selected date range")
    
    # Footer
    st.markdown("---")
    st.caption(f"Last updated: {data.get('latest_signal', {}).get('timestamp', 'Unknown')}")
    st.caption("Kriterion Quant Trading System - Cycle-Based Strategy")

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    main()
