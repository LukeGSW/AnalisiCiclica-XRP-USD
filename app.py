"""
Streamlit Dashboard for Kriterion Quant Trading System
Interactive visualization of cycle analysis and trading signals
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
import subprocess

# Add src directory to path
sys.path.insert(0, 'src')

from config import Config
from data_fetcher import DataFetcher
from cycle_analyzer import CycleAnalyzer
from signal_generator import SignalGenerator
from backtester import Backtester

# Page configuration
st.set_page_config(
    page_title=f"Kriterion Quant - {Config.TICKER}",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
# Sostituisci la sezione CSS all'inizio di app.py con questa versione migliorata:

# Custom CSS for better styling with dark mode support
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
    
    /* Metric value styling */
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
    
    /* Info boxes styling */
    .stAlert > div {
        background-color: #ffffff !important;
        color: #1f1f1f !important;
        border: 1px solid #4a90e2 !important;
    }
    
    /* Success boxes */
    div[data-baseweb="notification"][kind="success"] {
        background-color: #d4edda !important;
        color: #155724 !important;
        border: 1px solid #c3e6cb !important;
    }
    
    /* Warning boxes */
    div[data-baseweb="notification"][kind="warning"] {
        background-color: #fff3cd !important;
        color: #856404 !important;
        border: 1px solid #ffeaa7 !important;
    }
    
    /* Error boxes */
    div[data-baseweb="notification"][kind="error"] {
        background-color: #f8d7da !important;
        color: #721c24 !important;
        border: 1px solid #f5c6cb !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
    }
    
    section[data-testid="stSidebar"] .stAlert > div {
        background-color: #e7f3ff !important;
        color: #004085 !important;
        border: 1px solid #b8daff !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #4a90e2;
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 5px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #357abd;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Primary button */
    .stButton > button[kind="primary"] {
        background-color: #28a745;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #218838;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1f1f1f !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #666666 !important;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #4a90e2 !important;
        border-radius: 5px;
        font-weight: 600;
    }
    
    /* DataFrame styling */
    .dataframe {
        font-size: 14px !important;
    }
    
    .dataframe th {
        background-color: #4a90e2 !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    .dataframe td {
        background-color: white !important;
        color: #1f1f1f !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #4a90e2;
    }
    
    /* Spinner text */
    .stSpinner > div {
        color: #4a90e2 !important;
    }
</style>
""", unsafe_allow_html=True)

def run_analysis():
    """Run the complete analysis pipeline"""
    try:
        with st.spinner('🔄 Running analysis... This may take 1-2 minutes'):
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Fetch data
            status_text.text('📡 Fetching market data...')
            progress_bar.progress(20)
            
            fetcher = DataFetcher()
            df = fetcher.update_latest_data(Config.TICKER)
            
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
            generator.save_signals(df_signals)
            
            # Step 4: Run backtest
            status_text.text('📊 Running backtest...')
            progress_bar.progress(80)
            
            backtester = Backtester()
            wf_results = backtester.run_walk_forward_analysis(df_signals)
            backtester.save_backtest_results(wf_results)
            
            # Step 5: Complete
            status_text.text('✅ Analysis complete!')
            progress_bar.progress(100)
            
            # Save summary
            latest_signal = generator.get_latest_signal(df_signals)
            summary = {
                'timestamp': datetime.now().isoformat(),
                'ticker': Config.TICKER,
                'data_points': len(df_signals),
                'latest_signal': latest_signal
            }
            
            summary_file = os.path.join(Config.DATA_DIR, 'analysis_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            return True, "Analysis completed successfully!"
            
    except Exception as e:
        return False, f"Error: {str(e)}"

def load_data():
    """Load all necessary data files"""
    data = {}
    
    # Load signals data
    signals_file = Config.SIGNALS_FILE
    if os.path.exists(signals_file):
        data['signals'] = pd.read_csv(signals_file, index_col='date', parse_dates=True)
    else:
        return None
    
    # Load latest signal
    latest_signal_file = signals_file.replace('.csv', '_latest.json')
    if os.path.exists(latest_signal_file):
        with open(latest_signal_file, 'r') as f:
            data['latest_signal'] = json.load(f)
    
    # Load backtest results
    if os.path.exists(Config.BACKTEST_RESULTS_FILE):
        with open(Config.BACKTEST_RESULTS_FILE, 'r') as f:
            data['backtest'] = json.load(f)
    
    # Load analysis summary
    summary_file = os.path.join(Config.DATA_DIR, 'analysis_summary.json')
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            data['summary'] = json.load(f)
    
    return data

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
    fig.add_trace(
        go.Scatter(
            x=buy_signals.index,
            y=buy_signals['close'],
            mode='markers',
            name='Buy Signal',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='green'
            )
        ),
        row=1, col=1
    )
    
    # Add sell signals
    sell_signals = df[df['signal'] == 'SELL']
    fig.add_trace(
        go.Scatter(
            x=sell_signals.index,
            y=sell_signals['close'],
            mode='markers',
            name='Sell Signal',
            marker=dict(
                symbol='triangle-down',
                size=12,
                color='red'
            )
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
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['position'].shift(1) * df['returns']
    df['cumulative_strategy'] = (1 + df['strategy_returns']).cumprod()
    df['cumulative_buy_hold'] = (1 + df['returns']).cumprod()
    
    initial_capital = Config.INITIAL_CAPITAL
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
    st.title(f"🎯 Kriterion Quant Trading System")
    st.markdown(f"### Cycle-Based Trading Strategy for {Config.TICKER}")
    
    # Check if data exists
    data = load_data()
    
    # If no data, show setup page
    if data is None:
        st.warning("⚠️ No analysis data found. Please run the analysis first.")
        
        st.markdown("---")
        
        # Big centered container for the run button
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### 🚀 Initial Setup")
            st.info("""
            This appears to be your first time using the dashboard. 
            Click the button below to run the initial analysis and generate trading signals.
            
            This process will:
            1. Fetch historical market data
            2. Perform cycle analysis
            3. Generate trading signals
            4. Run backtesting
            5. Create all necessary data files
            """)
            
            # Check API key
            if not Config.EODHD_API_KEY:
                st.error("❌ EODHD API Key not configured!")
                st.markdown("""
                Please configure your API key:
                1. Get an API key from [EODHD](https://eodhistoricaldata.com/)
                2. Add it to Streamlit Secrets (Settings → Secrets)
                3. Format: `EODHD_API_KEY = "your_key_here"`
                """)
                return
            
            # Run analysis button
            if st.button("🎯 Run Initial Analysis", use_container_width=True, type="primary"):
                success, message = run_analysis()
                
                if success:
                    st.success(f"✅ {message}")
                    st.balloons()
                    st.markdown("### 🎉 Analysis Complete!")
                    st.markdown("Click the button below to view the results.")
                    if st.button("📊 View Dashboard", use_container_width=True):
                        st.rerun()
                else:
                    st.error(f"❌ {message}")
                    st.markdown("""
                    **Troubleshooting tips:**
                    - Check your API key is valid
                    - Ensure you have internet connection
                    - Check the error message above
                    """)
        
        return
    
    # Regular dashboard view (when data exists)
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info(f"""
        **Ticker:** {Config.TICKER}  
        **Fast MA:** {Config.FAST_MA_WINDOW}  
        **Slow MA:** {Config.SLOW_MA_WINDOW}  
        **Initial Capital:** ${Config.INITIAL_CAPITAL:,.0f}  
        **Trading Fees:** {Config.TRADING_FEES*100:.1f}%
        """)
        
        # Update analysis button
        st.markdown("---")
        st.markdown("### 🔄 Update Analysis")
        if st.button("Run New Analysis", use_container_width=True):
            success, message = run_analysis()
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
        
        # Date range filter
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
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Current Status", "📈 Analysis", "🎯 Backtest", "📋 History"])
    
    with tab1:
        st.header("Current Trading Status")
        
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
            
            st.subheader("📍 Signal Details")
            details_col1, details_col2 = st.columns(2)
            
            with details_col1:
                st.info(f"""
                **Oscillator Value:** {latest['oscillator_value']:.4f}  
                **Phase Quadrant:** {latest['phase_quadrant']}  
                **Generated:** {latest['timestamp']}
                """)
            
            with details_col2:
                if latest['signal'] == 'BUY':
                    st.success("🎯 **Action:** Enter Long Position")
                elif latest['signal'] == 'SELL':
                    st.warning("🎯 **Action:** Exit Long Position")
                else:
                    st.info("🎯 **Action:** Maintain Current Position")
    
    with tab2:
        st.header("Cycle Analysis")
        st.subheader("📈 Price Chart with Signals")
        fig_price = create_price_chart(df_filtered)
        st.plotly_chart(fig_price, use_container_width=True)
        
        if 'phase_quadrant' in df_filtered.columns:
            st.subheader("🔄 Cycle Phase Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                phase_counts = df_filtered['phase_quadrant'].value_counts()
                fig_pie = go.Figure(data=[
                    go.Pie(
                        labels=phase_counts.index,
                        values=phase_counts.values,
                        hole=0.3
                    )
                ])
                fig_pie.update_layout(
                    title="Time Spent in Each Phase",
                    height=300
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                if 'summary' in data and 'cycle_analysis' in data['summary']:
                    cycle_info = data['summary']['cycle_analysis']
                    st.metric(
                        "Dominant Cycle Period",
                        f"{cycle_info.get('dominant_period', 0):.1f} days"
                    )
                    st.metric(
                        "Statistical Significance",
                        f"p-value: {cycle_info.get('p_value', 1):.4f}",
                        "✅ Significant" if cycle_info.get('significant', False) else "⚠️ Not Significant"
                    )
    
    with tab3:
        st.header("Backtest Results")
        st.subheader("💰 Equity Curve")
        fig_equity = create_equity_chart(df_filtered)
        st.plotly_chart(fig_equity, use_container_width=True)
        
        if 'backtest' in data:
            st.subheader("📊 Performance Metrics")
            
            col1, col2 = st.columns(2)
            
            if 'in_sample_metrics' in data['backtest']:
                with col1:
                    st.markdown("**In-Sample Performance**")
                    is_metrics = data['backtest']['in_sample_metrics']
                    for key, value in is_metrics.items():
                        if isinstance(value, (int, float)):
                            st.metric(key.replace('_', ' ').title(), f"{value:.2f}")
                
                with col2:
                    st.markdown("**Out-of-Sample Performance**")
                    oos_metrics = data['backtest']['out_of_sample_metrics']
                    for key, value in oos_metrics.items():
                        if isinstance(value, (int, float)):
                            st.metric(key.replace('_', ' ').title(), f"{value:.2f}")
            else:
                metrics = data['backtest'].get('metrics', {})
                
                with col1:
                    st.metric("Total Return", f"{metrics.get('total_return_%', 0):.2f}%")
                    st.metric("Max Drawdown", f"{metrics.get('max_drawdown_%', 0):.2f}%")
                    st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                    st.metric("Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.2f}")
                
                with col2:
                    st.metric("Total Trades", f"{metrics.get('total_trades', 0):.0f}")
                    st.metric("Win Rate", f"{metrics.get('win_rate_%', 0):.1f}%")
                    st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
                    st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}")
    
    with tab4:
        st.header("Trading History")
        st.subheader("📝 Recent Signals")
        recent_signals = df_filtered[df_filtered['signal'] != 'HOLD'].tail(20)
        
        if not recent_signals.empty:
            display_df = recent_signals[['close', 'signal', 'phase_quadrant', 'signal_strength', 'confidence']]
            display_df.index = display_df.index.date
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No recent signals in the selected date range")
        
        st.subheader("💾 Download Data")
        csv = df_filtered.to_csv()
        st.download_button(
            label="Download Signals Data (CSV)",
            data=csv,
            file_name=f"kriterion_signals_{Config.TICKER}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    st.caption(f"Last updated: {data.get('summary', {}).get('timestamp', 'Unknown')}")
    st.caption("Kriterion Quant Trading System - Cycle-Based Strategy")

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    main()
