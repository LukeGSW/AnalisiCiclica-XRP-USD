"""
Streamlit Dashboard for Kriterion Quant Trading System
Multi-ticker version with dynamic analysis and selection
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
import re

# Add src directory to path
sys.path.insert(0, 'src')

# Import delle classi necessarie
from data_fetcher import DataFetcher
from cycle_analyzer import CycleAnalyzer
from signal_generator import SignalGenerator
from backtester import Backtester

# --- INIZIO FUNZIONI HELPER ---

@st.cache_data(ttl=300) # Cache per 5 minuti per non scansionare i file a ogni interazione
def get_available_tickers():
    """Scans the data directory to find available analysis summaries and extract tickers."""
    data_dir = 'data'
    tickers = []
    if os.path.exists(data_dir):
        pattern = re.compile(r"analysis_summary_(.+)\.json")
        for filename in os.listdir(data_dir):
            match = pattern.match(filename)
            if match:
                tickers.append(match.group(1))
    return sorted(tickers)

@st.cache_data(ttl=60) # Cache per 1 minuto
def load_data(ticker):
    """Load all necessary data files for a specific ticker."""
    if not ticker:
        return None
        
    data_dir = 'data'
    data = {}
    
    signals_file = os.path.join(data_dir, f'signals_{ticker}.csv')
    backtest_file = os.path.join(data_dir, f'backtest_results_{ticker}.json')
    summary_file = os.path.join(data_dir, f'analysis_summary_{ticker}.json')

    if not os.path.exists(signals_file):
        return None
        
    data['signals'] = pd.read_csv(signals_file, index_col='date', parse_dates=True)
            
    if os.path.exists(backtest_file):
        with open(backtest_file, 'r') as f:
            data['backtest'] = json.load(f)
            
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            data['summary'] = json.load(f)
            if 'latest_signal' in data['summary']:
                 data['latest_signal'] = data['summary']['latest_signal']

    return data

def run_analysis(ticker, lookback_years):
    """Run the complete analysis pipeline for a specific ticker."""
    try:
        with st.spinner(f'🔄 Running analysis for {ticker} with {lookback_years} years lookback...'):
            progress_bar = st.progress(0, text="Initializing...")
            
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=lookback_years*365)).strftime('%Y-%m-%d')
            
            progress_bar.progress(10, text=f'📡 Fetching data for {ticker}...')
            fetcher = DataFetcher()
            df = fetcher.fetch_historical_data(ticker=ticker, start_date=start_date, end_date=end_date)
            
            progress_bar.progress(30, text='🔄 Performing cycle analysis...')
            analyzer = CycleAnalyzer()
            df_analyzed = analyzer.analyze_cycle(df)
            
            progress_bar.progress(50, text='🎯 Generating trading signals...')
            generator = SignalGenerator()
            df_signals = generator.generate_signals(df_analyzed)
            # NOTA: Assicurati che i metodi save_* accettino il ticker per salvare file con nomi dinamici
            generator.save_signals(df_signals, ticker)
            
            progress_bar.progress(70, text='📊 Running backtest...')
            backtester = Backtester()
            wf_results = backtester.run_walk_forward_analysis(df_signals)
            backtester.save_backtest_results(wf_results, ticker)
            
            progress_bar.progress(90, text='📝 Generating summary...')
            latest_signal = generator.get_latest_signal(df_signals)
            spectral_results = analyzer.run_spectral_analysis(df_analyzed['oscillator'])
            monte_carlo_results = analyzer.run_monte_carlo_significance_test(df_analyzed['oscillator'])
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'ticker': ticker,
                'lookback_years': lookback_years,
                'data_points': len(df_signals),
                'date_range': {'start': start_date, 'end': end_date},
                'latest_signal': latest_signal,
                'cycle_analysis': {
                    'dominant_period': float(spectral_results.get('dominant_period', 0)),
                    'p_value': float(monte_carlo_results.get('p_value', 1.0)),
                    'significant': bool(monte_carlo_results.get('significant', False))
                }
            }
            
            summary_file = os.path.join('data', f'analysis_summary_{ticker}.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            progress_bar.progress(100, text='✅ Analysis complete!')
            return True, f"Analysis for {ticker} completed successfully!"
            
    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
        return False, f"Error: {str(e)}"


def create_price_chart(df, ticker):
    """Create interactive price chart with signals."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f'Price & Signals for {ticker}', 'Cycle Oscillator', 'Phase')
    )
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close Price', line=dict(color='blue', width=2)), row=1, col=1)
    buy_signals = df[df['signal'] == 'BUY']
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['close'], mode='markers', name='Buy Signal', marker=dict(symbol='triangle-up', size=12, color='green')), row=1, col=1)
    sell_signals = df[df['signal'] == 'SELL']
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['close'], mode='markers', name='Sell Signal', marker=dict(symbol='triangle-down', size=12, color='red')), row=1, col=1)
    if 'oscillator' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['oscillator'], name='Oscillator', line=dict(color='orange', width=1)), row=2, col=1)
        fig.add_hline(y=0, row=2, col=1, line_dash="dash", line_color="gray")
    if 'phase' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['phase'], name='Phase', line=dict(color='green', width=1)), row=3, col=1)
        fig.add_hline(y=np.pi/2, row=3, col=1, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_hline(y=0, row=3, col=1, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_hline(y=-np.pi/2, row=3, col=1, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(height=800, showlegend=True, hovermode='x unified', template='plotly_white', margin=dict(t=40))
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    return fig

def create_equity_chart(df, ticker, initial_capital=10000):
    """Create equity curve chart."""
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['position'].shift(1) * df['returns']
    df['cumulative_strategy'] = (1 + df['strategy_returns']).cumprod()
    df['cumulative_buy_hold'] = (1 + df['returns']).cumprod()
    
    df['equity'] = df['cumulative_strategy'] * initial_capital
    df['buy_hold_equity'] = df['cumulative_buy_hold'] * initial_capital
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['equity'], name='Strategy', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['buy_hold_equity'], name=f'Buy & Hold {ticker}', line=dict(color='gray', width=1, dash='dash')))
    fig.update_layout(title=f'Equity Curve Comparison for {ticker}', xaxis_title='Date', yaxis_title='Portfolio Value ($)', height=400, hovermode='x unified', template='plotly_white')
    return fig

# --- FINE FUNZIONI HELPER ---

def main():
    """Main dashboard function"""
    
    st.set_page_config(page_title="Kriterion Quant", page_icon="📊", layout="wide")
    st.markdown("""<style> ... (il tuo CSS qui) ... </style>""", unsafe_allow_html=True) # CSS omesso per brevità

    st.title("🎯 Kriterion Quant Trading System")
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        available_tickers = get_available_tickers()
        
        selected_ticker = st.selectbox(
            "Select Ticker to View", options=available_tickers,
            index=0 if available_tickers else None, placeholder="No analyses found"
        )
        
        data = load_data(selected_ticker)
        
        if data and 'summary' in data:
            summary = data['summary']
            st.info(f"""
            **Ticker:** {summary.get('ticker', 'N/A')}
            **Lookback:** {summary.get('lookback_years', 'N/A')} years
            **Data Range:** {summary.get('date_range', {}).get('start', 'N/A')} to {summary.get('date_range', {}).get('end', 'N/A')}
            """)
            
            st.markdown("### 📊 Cycle Analysis")
            cycle_info = summary.get('cycle_analysis', {})
            st.metric("Dominant Cycle", f"{float(cycle_info.get('dominant_period', 0)):.1f} days")
            st.metric("Significance", f"p-value: {float(cycle_info.get('p_value', 1)):.4f}", "✅ Significant" if cycle_info.get('significant') else "⚠️ Not Significant")
        
        st.markdown("---")
        with st.expander("🔄 Run New/Update Analysis", expanded=True):
            ticker_input = st.text_input("Enter Ticker to Analyze", placeholder="SPY.US, AAPL.US, QQQ.US").upper()
            lookback_input = st.slider("Lookback period (years)", 1, 20, 10, 1)

            if st.button("Run Analysis", use_container_width=True, type="primary"):
                if ticker_input:
                    with st.spinner(f"Running analysis for {ticker_input}..."):
                        success, message = run_analysis(ticker_input, lookback_input)
                    if success:
                        st.success(message)
                        st.cache_data.clear() # Pulisce la cache per ricaricare i dati
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Please enter a ticker symbol.")

    # --- MAIN CONTENT AREA ---
    
    if not selected_ticker or not data:
        st.info("👈 Welcome to Kriterion Quant! Use the sidebar to run your first analysis.")
        return

    st.markdown(f"## Cycle-Based Trading Strategy for **{selected_ticker}**")
    
    df_signals = data['signals']
    
    with st.sidebar:
        st.markdown("---")
        st.header("📅 Date Filter")
        date_range = st.date_input(
            "Select date range", value=(df_signals.index.min().date(), df_signals.index.max().date()),
            min_value=df_signals.index.min().date(), max_value=df_signals.index.max().date()
        )
        if len(date_range) == 2:
            df_filtered = df_signals.loc[str(date_range[0]):str(date_range[1])]
        else:
            df_filtered = df_signals
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Current Status", "📈 Analysis", "🎯 Backtest", "📋 History"])
    
    with tab1:
        st.header(f"Current Trading Status")
        if 'latest_signal' in data:
            latest = data['latest_signal']
            col1, col2, col3, col4 = st.columns(4)
            # ... (Tutto il codice per le metriche è invariato, usa la variabile `latest`)
            with col1:
                signal_color = "🟢" if latest['signal'] == 'BUY' else "🔴" if latest['signal'] == 'SELL' else "⏸️"
                st.metric("Last Signal", f"{signal_color} {latest['signal']}", f"on {latest['date']}")
            with col2:
                position_emoji = "💰" if latest['position'] == 'LONG' else "💤"
                st.metric("Position", f"{position_emoji} {latest['position']}", f"${float(latest['price']):.2f}")
            with col3:
                st.metric("Signal Strength", f"{float(latest['signal_strength']):.1f}/100", f"{latest['confidence']} confidence")
            with col4:
                st.metric("Cycle Phase", f"{float(latest['phase_value']):.2f} rad", latest['phase_quadrant'])
            
            st.subheader("📍 Signal Details")
            # ... (il codice per i dettagli è invariato)

    with tab2:
        st.header("Cycle Analysis")
        st.subheader(f"📈 Price Chart with Signals for {selected_ticker}")
        fig_price = create_price_chart(df_filtered, selected_ticker)
        st.plotly_chart(fig_price, use_container_width=True)
        # ... (Il resto del tab, come la pie chart, è invariato)

    with tab3:
        st.header("Backtest Results")
        st.subheader("💰 Equity Curve")
        fig_equity = create_equity_chart(df_filtered, selected_ticker)
        st.plotly_chart(fig_equity, use_container_width=True)
        # ... (Il resto del tab con le metriche è invariato, usa `data['backtest']`)

    with tab4:
        st.header("Trading History")
        st.subheader("📝 Recent Signals")
        # ... (La logica per mostrare i segnali recenti è invariata)
        st.subheader("💾 Download Data")
        csv = df_filtered.to_csv(index=True)
        st.download_button(
            label=f"Download Signals Data for {selected_ticker} (CSV)",
            data=csv,
            file_name=f"kriterion_signals_{selected_ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
    st.markdown("---")
    st.caption(f"Dashboard last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. Data timestamp: {data.get('summary', {}).get('timestamp', 'N/A')}")
    st.caption("Kriterion Quant Trading System - Cycle-Based Strategy")

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    main()
