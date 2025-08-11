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

# Add src directory to path
sys.path.insert(0, 'src')

from config import Config

# Page configuration
st.set_page_config(
    page_title=f"Kriterion Quant - {Config.TICKER}",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .positive-metric {
        color: #00cc44;
    }
    .negative-metric {
        color: #ff4444;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load all necessary data files"""
    data = {}
    
    # Load signals data
    signals_file = Config.SIGNALS_FILE
    if os.path.exists(signals_file):
        data['signals'] = pd.read_csv(signals_file, index_col='date', parse_dates=True)
    else:
        st.error(f"Signals file not found: {signals_file}")
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
        
        # Add zero line
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
        
        # Add phase quadrant lines
        fig.add_hline(y=np.pi/2, row=3, col=1, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_hline(y=0, row=3, col=1, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_hline(y=-np.pi/2, row=3, col=1, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Update layout
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
    # Calculate cumulative returns
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['position'].shift(1) * df['returns']
    df['cumulative_strategy'] = (1 + df['strategy_returns']).cumprod()
    df['cumulative_buy_hold'] = (1 + df['returns']).cumprod()
    
    # Normalize to starting capital
    initial_capital = Config.INITIAL_CAPITAL
    df['equity'] = df['cumulative_strategy'] * initial_capital
    df['buy_hold_equity'] = df['cumulative_buy_hold'] * initial_capital
    
    # Create figure
    fig = go.Figure()
    
    # Strategy equity
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['equity'],
            name='Strategy',
            line=dict(color='blue', width=2)
        )
    )
    
    # Buy & Hold equity
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
    
    # Load data
    data = load_data()
    
    if data is None:
        st.error("Failed to load data. Please run the analysis script first.")
        return
    
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
        
        # Date range filter
        st.header("📅 Date Range")
        df_signals = data['signals']
        
        date_range = st.date_input(
            "Select date range",
            value=(df_signals.index[0], df_signals.index[-1]),
            min_value=df_signals.index[0],
            max_value=df_signals.index[-1]
        )
        
        # Filter data
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
            
            # Display metrics in columns
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
            
            # Signal details
            st.subheader("📍 Signal Details")
            details_col1, details_col2 = st.columns(2)
            
            with details_col1:
                st.info(f"""
                **Oscillator Value:** {latest['oscillator_value']:.4f}  
                **Phase Quadrant:** {latest['phase_quadrant']}  
                **Generated:** {latest['timestamp']}
                """)
            
            with details_col2:
                # Recommendation based on signal
                if latest['signal'] == 'BUY':
                    st.success("🎯 **Action:** Enter Long Position")
                elif latest['signal'] == 'SELL':
                    st.warning("🎯 **Action:** Exit Long Position")
                else:
                    st.info("🎯 **Action:** Maintain Current Position")
    
    with tab2:
        st.header("Cycle Analysis")
        
        # Interactive price chart
        st.subheader("📈 Price Chart with Signals")
        fig_price = create_price_chart(df_filtered)
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Phase distribution
        if 'phase_quadrant' in df_filtered.columns:
            st.subheader("🔄 Cycle Phase Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Phase distribution pie chart
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
                # Performance by phase
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
        
        # Equity curve
        st.subheader("💰 Equity Curve")
        fig_equity = create_equity_chart(df_filtered)
        st.plotly_chart(fig_equity, use_container_width=True)
        
        # Performance metrics
        if 'backtest' in data:
            st.subheader("📊 Performance Metrics")
            
            col1, col2 = st.columns(2)
            
            # Check if we have IS/OOS results
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
                # Single backtest results
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
        
        # Recent signals
        st.subheader("📝 Recent Signals")
        recent_signals = df_filtered[df_filtered['signal'] != 'HOLD'].tail(20)
        
        if not recent_signals.empty:
            display_df = recent_signals[['close', 'signal', 'phase_quadrant', 'signal_strength', 'confidence']]
            display_df.index = display_df.index.date
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No recent signals in the selected date range")
        
        # Download data
        st.subheader("💾 Download Data")
        
        csv = df_filtered.to_csv()
        st.download_button(
            label="Download Signals Data (CSV)",
            data=csv,
            file_name=f"kriterion_signals_{Config.TICKER}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.caption(f"Last updated: {data.get('summary', {}).get('timestamp', 'Unknown')}")
    st.caption("Kriterion Quant Trading System - Cycle-Based Strategy")

if __name__ == "__main__":
    main()
