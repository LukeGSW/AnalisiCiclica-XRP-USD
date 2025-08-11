# 🎯 Kriterion Quant Trading System

[![Daily Analysis](https://github.com/yourusername/kriterion-quant/actions/workflows/daily_analysis.yml/badge.svg)](https://github.com/yourusername/kriterion-quant/actions/workflows/daily_analysis.yml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kriterion-quant.streamlit.app)

## 📊 Overview

Kriterion Quant is an automated trading system that uses cycle analysis based on the Hilbert Transform to generate trading signals. The system implements a causal oscillator strategy that avoids look-ahead bias and validates signals through walk-forward analysis.

### Key Features

- 🔄 **Cycle Analysis**: Identifies dominant market cycles using spectral analysis
- 📈 **Causal Signals**: Generates signals without look-ahead bias
- 🎯 **Statistical Validation**: Monte Carlo testing for cycle significance
- 📊 **Walk-Forward Analysis**: Robust backtesting with IS/OOS validation
- 🔔 **Telegram Alerts**: Real-time signal notifications
- 📱 **Interactive Dashboard**: Streamlit web interface for monitoring
- 🤖 **Full Automation**: GitHub Actions for daily analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- EODHD API key
- Telegram Bot (optional)
- GitHub account

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/kriterion-quant.git
cd kriterion-quant
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Run the analysis**
```bash
python main_analysis.py
```

5. **Launch the dashboard**
```bash
streamlit run app.py
```

## 📁 Project Structure

```
kriterion-quant/
├── .github/
│   └── workflows/
│       └── daily_analysis.yml    # GitHub Actions workflow
├── src/
│   ├── __init__.py
│   ├── config.py                 # Configuration settings
│   ├── data_fetcher.py          # EODHD data acquisition
│   ├── cycle_analyzer.py        # Hilbert Transform analysis
│   ├── signal_generator.py      # Trading signal generation
│   ├── backtester.py            # Walk-forward backtesting
│   └── notifier.py              # Telegram notifications
├── data/
│   ├── signals.csv              # Generated signals
│   ├── historical_data.csv      # Price data
│   └── backtest_results.json    # Performance metrics
├── app.py                       # Streamlit dashboard
├── main_analysis.py             # Main execution script
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
└── README.md                    # Documentation
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Required
EODHD_API_KEY=your_api_key_here

# Optional (for notifications)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Optional (custom parameters)
TICKER=GLD
INITIAL_CAPITAL=10000
```

### GitHub Secrets

For GitHub Actions automation, add these secrets to your repository:

1. Go to Settings → Secrets → Actions
2. Add the following secrets:
   - `EODHD_API_KEY`
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`

## 📈 Trading Strategy

### Cycle Analysis Method

1. **Oscillator Creation**: Difference between fast (10-day) and slow (40-day) moving averages
2. **Hilbert Transform**: Extracts instantaneous phase and amplitude
3. **Phase Classification**: Divides cycle into 4 quadrants
4. **Signal Generation**: 
   - BUY: Enter bullish phase (Quadrants 1 & 2)
   - SELL: Exit bullish phase (Quadrants 3 & 4)

### Validation Process

1. **Spectral Analysis**: Welch periodogram to identify dominant cycles
2. **Monte Carlo Testing**: Statistical significance validation
3. **Walk-Forward Analysis**: 70% in-sample, 30% out-of-sample testing

## 🔔 Telegram Notifications

### Setup Telegram Bot

1. Message [@BotFather](https://t.me/botfather) on Telegram
2. Create new bot with `/newbot`
3. Save the bot token
4. Start conversation with your bot
5. Get your chat ID from `https://api.telegram.org/bot<TOKEN>/getUpdates`

### Notification Types

- 🟢 **Buy Signals**: Entry notifications with confidence levels
- 🔴 **Sell Signals**: Exit notifications
- 📊 **Daily Summary**: Complete analysis report
- 🚨 **Error Alerts**: System issues

## 📱 Streamlit Dashboard

### Local Deployment

```bash
streamlit run app.py
```

### Cloud Deployment

1. Push code to GitHub
2. Connect repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy app with environment variables
4. Access at `https://your-app.streamlit.app`

### Dashboard Features

- **Current Status**: Latest signal and position
- **Cycle Analysis**: Interactive price charts with signals
- **Backtest Results**: Performance metrics and equity curves
- **Trading History**: Signal log and data export

## 🤖 Automation with GitHub Actions

The system runs automatically every weekday at market close:

### Workflow Schedule

- **Time**: 4:30 PM EST (after market close)
- **Frequency**: Monday-Friday
- **Actions**:
  1. Fetch latest market data
  2. Run cycle analysis
  3. Generate signals
  4. Send notifications
  5. Update dashboard

### Manual Trigger

Run analysis manually from GitHub:

1. Go to Actions tab
2. Select "Daily Trading Analysis"
3. Click "Run workflow"
4. Optional: specify different ticker

## 📊 Performance Metrics

### Key Metrics Tracked

- **Total Return**: Overall strategy performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Maximum peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profits / Gross losses

### Validation Metrics

- **IS/OOS Correlation**: Consistency between periods
- **Monte Carlo p-value**: Statistical significance
- **Walk-Forward Stability**: Performance across windows

## 🛠️ Troubleshooting

### Common Issues

1. **No data found**
   - Check EODHD API key
   - Verify ticker symbol
   - Check date range

2. **Telegram not working**
   -
