"""
Streamlit Dashboard for Kriterion Quant Trading System
Single ticker version with adjustable lookback period
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
    page_title=f"Kriterion Quant - {Config.TICKER}",
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
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #357abd;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    h1, h2, h3 {
        color: #1f1f1f !important;
    }
</style>
""", unsafe_allow_html=True)
def get_available_tickers(path='data'):
    """Scansiona la directory dati e restituisce un elenco di ticker disponibili."""
    if not os.path.exists(path):
        return []
    tickers = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    return sorted(tickers)

@st.cache_data(ttl=600) # Cache dei dati per 10 minuti
def load_data_from_path(ticker_path): # <<< Nome e parametro cambiati
    """Carica tutti i dati necessari da un percorso specifico."""
    if not os.path.exists(ticker_path):
        return None

    data = {}
    try:
        # Carica i file usando il percorso fornito
        signals_file = os.path.join(ticker_path, 'signals.csv')
        data['signals'] = pd.read_csv(signals_file, index_col='date', parse_dates=True)

        latest_signal_file = os.path.join(ticker_path, 'signals_latest.json')
        with open(latest_signal_file, 'r') as f:
            data['latest_signal'] = json.load(f)

        backtest_file = os.path.join(ticker_path, 'backtest_results.json')
        with open(backtest_file, 'r') as f:
            data['backtest'] = json.load(f)

        summary_file = os.path.join(ticker_path, 'analysis_summary.json')
        with open(summary_file, 'r') as f:
            data['summary'] = json.load(f)

    except Exception as e:
        st.error(f"Errore nel caricamento dei dati da {ticker_path}: {e}")
        return None

    return data


def run_analysis_for_ticker(ticker, lookback_years=10):
    """Run the complete analysis pipeline for a specific ticker."""
    try:
        with st.spinner(f'🔄 Running analysis for {ticker} with {lookback_years} years lookback...'):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Definiamo il percorso dati per questo ticker
            ticker_data_path = os.path.join('data', ticker)
            os.makedirs(ticker_data_path, exist_ok=True)
            
            # Inizializziamo le classi con il percorso corretto
            fetcher = DataFetcher(data_path=ticker_data_path)
            analyzer = CycleAnalyzer()
            generator = SignalGenerator(data_path=ticker_data_path)
            backtester = Backtester(data_path=ticker_data_path)
            
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=lookback_years*365.25)).strftime('%Y-%m-%d')
            
            # Step 1: Fetch data
            status_text.text(f'📡 Fetching data for {ticker}...')
            progress_bar.progress(20)
            df = fetcher.fetch_historical_data(ticker=ticker, start_date=start_date, end_date=end_date)
            fetcher.save_data(df)
            
            # Step 2: Cycle analysis
            status_text.text('🔄 Performing cycle analysis...')
            progress_bar.progress(40)
            df_analyzed = analyzer.analyze_cycle(df)
            
            # Step 3: Generate signals
            status_text.text('🎯 Generating trading signals...')
            progress_bar.progress(60)
            df_signals = generator.generate_signals(df_analyzed)
            generator.save_signals(df_signals)
            
            # Step 4: Run backtest
            status_text.text('📊 Running backtest...')
            progress_bar.progress(80)
            wf_results = backtester.run_walk_forward_analysis(df_signals)
            backtester.save_backtest_results(wf_results)
            
            # Step 5: Save summary
            status_text.text('📄 Creating summary...')
            latest_signal = generator.get_latest_signal(df_signals)
            spectral_results = analyzer.run_spectral_analysis(df_analyzed['oscillator'])
            monte_carlo_results = analyzer.run_monte_carlo_significance_test(df_analyzed['oscillator'])
            
            summary = {
                'timestamp': datetime.now().isoformat(), 'ticker': ticker,
                'lookback_years': lookback_years, 'data_points': len(df_signals),
                'date_range': {'start': df_signals.index[0].strftime('%Y-%m-%d'), 'end': df_signals.index[-1].strftime('%Y-%m-%d')},
                'latest_signal': latest_signal,
                'cycle_analysis': {
                    'dominant_period': float(spectral_results['dominant_period']) if spectral_results['dominant_period'] else None,
                    'p_value': float(monte_carlo_results['p_value']),
                    'significant': bool(monte_carlo_results['significant'])
                }
            }
            summary_file = os.path.join(ticker_data_path, 'analysis_summary.json')
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
                
            progress_bar.progress(100)
            status_text.text(f'✅ Analysis for {ticker} complete!')
            return True, f"Analysis for {ticker} completed successfully!"
            
    except Exception as e:
        return False, f"Error during analysis for {ticker}: {e}"
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
    if not buy_signals.empty:
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
    if not sell_signals.empty:
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

def create_equity_chart(df_results: pd.DataFrame):
    """
    Crea il grafico della curva di equity dai risultati del backtest.
    Questa funzione ora si occupa solo della VISUALIZZAZIONE.
    df_results: Il DataFrame 'results' restituito dal Backtester.
    """
    fig = go.Figure()
    
    # Grafico dell'equity della strategia (calcolata dal backtester)
    if 'equity' in df_results.columns:
        fig.add_trace(
            go.Scatter(
                x=df_results.index,
                y=df_results['equity'],
                name='Strategy',
                line=dict(color='blue', width=2)
            )
        )
    
    # Grafico dell'equity del benchmark (Buy & Hold)
    if 'benchmark_equity' in df_results.columns:
        fig.add_trace(
            go.Scatter(
                x=df_results.index,
                y=df_results['benchmark_equity'],
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
    
    st.title(f"🎯 Kriterion Quant Trading System")

    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        available_tickers = get_available_tickers()
        
        if not available_tickers:
            st.warning("No data found. Run an analysis via GitHub Actions first.")
            st.stop()
            
        selected_ticker = st.selectbox("Select Ticker", available_tickers)
        
    # ================================================================= #
    #                         <<< CORREZIONE QUI >>>                      #
    # ================================================================= #
    # Definiamo il percorso UNA SOLA VOLTA qui
    ticker_data_path = os.path.join('data', selected_ticker)

    # Carichiamo i dati usando il percorso completo
    data = load_data_from_path(ticker_data_path)
    # ================================================================= #

    # --- Main Page ---
    st.markdown(f"### Cycle-Based Trading Strategy for {selected_ticker}")

    if data is None:
        st.warning(f"⚠️ Data for {selected_ticker} is missing or corrupted. Please re-run the analysis.")
        # Qui potresti aggiungere un pulsante per rieseguire l'analisi per questo specifico ticker se lo desideri
        return
    
    # Regular dashboard view (when data exists)
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Display current configuration
        summary = data.get('summary', {})
        current_lookback = summary.get('lookback_years', 10)
        date_range_info = summary.get('date_range', {})
        
        st.info(f"""
        **Ticker:** {Config.TICKER}  
        **Current Lookback:** {current_lookback} years  
        **Data Range:** {date_range_info.get('start', 'N/A')} to {date_range_info.get('end', 'N/A')}  
        **Fast MA:** {Config.FAST_MA_WINDOW}  
        **Slow MA:** {Config.SLOW_MA_WINDOW}  
        **Initial Capital:** ${float(Config.INITIAL_CAPITAL):,.0f}  
        **Trading Fees:** {float(Config.TRADING_FEES)*100:.1f}%
        """)
        
        # Update analysis section
        st.markdown("---")
        st.markdown("### 🔄 Update Analysis")
        
        # Lookback period selector
        new_lookback = st.slider(
            "Lookback period (years)",
            min_value=1,
            max_value=20,
            value=current_lookback,
            step=1,
            help="Number of years of historical data to analyze"
        )
        
        # Show what date range this would be
        if new_lookback != current_lookback:
            new_start = (datetime.now() - timedelta(days=new_lookback*365.25)).strftime('%Y-%m-%d')
            new_end = datetime.now().strftime('%Y-%m-%d')
            st.caption(f"This will analyze data from {new_start} to {new_end}")
        
        # Run analysis button
# All'interno della sidebar...
        if st.button("Run New Analysis", use_container_width=True):
            # Usiamo il nuovo nome e passiamo anche il ticker selezionato
            success, message = run_analysis_for_ticker(selected_ticker, new_lookback)
            if success:
                st.success(message)
                # Rimuovi la cache per forzare il ricaricamento dei nuovi dati
                st.cache_data.clear()
                st.rerun()
            else:
                st.error(message)     
        # Cycle analysis info
        if 'cycle_analysis' in summary:
            st.markdown("---")
            st.markdown("### 📊 Cycle Analysis")
            cycle_info = summary['cycle_analysis']
            
            if cycle_info.get('dominant_period'):
                st.metric(
                    "Dominant Cycle",
                    f"{float(cycle_info['dominant_period']):.1f} days"
                )
            
            st.metric(
                "Statistical Significance",
                f"p-value: {float(cycle_info.get('p_value', 1)):.4f}",
                "✅ Significant" if cycle_info.get('significant', False) else "⚠️ Not Significant"
            )
        
        # Date range filter
        st.markdown("---")
        st.header("📅 Date Filter")
        df_signals = data['signals']
        
        date_range = st.date_input(
            "Select date range",
            value=(df_signals.index[0], df_signals.index[-1]),
            min_value=df_signals.index[0],
            max_value=df_signals.index[-1],
            key="date_filter_widget"
        )
            
        # Filtra i dati in base all'intervallo selezionato nel widget
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
            
            # Signal details
            st.subheader("📍 Signal Details")
            details_col1, details_col2 = st.columns(2)
            
            with details_col1:
                st.info(f"""
                **Oscillator Value:** {float(latest['oscillator_value']):.4f}  
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
                        labels=phase_counts.index.tolist(),
                        values=phase_counts.values.tolist(),
                        hole=0.3
                    )
                ])
                fig_pie.update_layout(
                    title="Time Spent in Each Phase",
                    height=300
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Signal statistics
                total_signals = len(df_filtered[df_filtered['signal'] != 'HOLD'])
                buy_signals = len(df_filtered[df_filtered['signal'] == 'BUY'])
                sell_signals = len(df_filtered[df_filtered['signal'] == 'SELL'])
                
                st.metric("Total Signals", total_signals)
                st.metric("Buy Signals", buy_signals)
                st.metric("Sell Signals", sell_signals)
    
    with tab3:
            st.header("Backtest Results")
    
            # --- Equity Curve per l'intervallo selezionato ---
            st.subheader("💰 Equity Curve")
            
            # ================================================================= #
            #                         <<< CORREZIONE QUI >>>                      #
            # ================================================================= #
            # Passiamo il percorso dati del ticker selezionato al Backtester
            backtester = Backtester(data_path=ticker_data_path)
            # ================================================================= #
            
            # Eseguiamo un backtest semplice solo per la visualizzazione sull'intervallo scelto
            backtest_visual_output = backtester.run_backtest(df_filtered)
            results_visual_df = backtest_visual_output.get('results')
            
            if results_visual_df is not None:
                fig_equity = create_equity_chart(results_visual_df)
                st.plotly_chart(fig_equity, use_container_width=True)
            else:
                st.warning("Could not generate equity curve for the selected range.")
    
            # --- Metriche dal Walk-Forward Analysis (più robuste) ---
            st.subheader("📊 Performance Metrics (Walk-Forward Analysis)")
            backtest_data = data.get('backtest', {})
            
            # Controlla se abbiamo i risultati del Walk-Forward
            if 'in_sample' in backtest_data and 'out_of_sample' in backtest_data:
                is_metrics = backtest_data['in_sample']
                oos_metrics = backtest_data['out_of_sample']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**In-Sample Performance**")
                    for key, value in is_metrics.items():
                        st.metric(
                            label=key.replace('_', ' ').title().replace('%', ''),
                            value=f"{float(value):.2f}{'%' if '%' in key else ''}"
                        )
                
                with col2:
                    st.markdown("**Out-of-Sample Performance**")
                    for key, value in oos_metrics.items():
                        st.metric(
                            label=key.replace('_', ' ').title().replace('%', ''),
                            value=f"{float(value):.2f}{'%' if '%' in key else ''}"
                        )
            else:
                # Fallback
                st.markdown("*(Displaying simple backtest metrics as Walk-Forward results are not available)*")
                metrics = backtest_visual_output.get('metrics', {})
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Return", f"{metrics.get('total_return_%', 0):.2f}%")
                    st.metric("Max Drawdown", f"{metrics.get('max_drawdown_%', 0):.2f}%")
                    st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                with col2:
                    st.metric("Total Trades", f"{int(metrics.get('total_trades', 0)):.0f}")
                    st.metric("Win Rate", f"{metrics.get('win_rate_%', 0):.1f}%")
                    st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
    with tab4:
        st.header("Trading History")
        
        # Recent signals
        st.subheader("📝 Recent Signals")
        recent_signals = df_filtered[df_filtered['signal'] != 'HOLD'].tail(20)
        
        if not recent_signals.empty:
            display_df = recent_signals[['close', 'signal', 'phase_quadrant', 'signal_strength', 'confidence']].copy()
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
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    main()
