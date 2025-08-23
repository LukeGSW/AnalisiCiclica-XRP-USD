"""
Configuration module for Kriterion Quant Trading System
Central configuration for all system parameters
"""

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Helper function to determine ticker (NOT a class method)
def get_ticker():
    """Get ticker with priority to environment variables"""
    # First priority: Environment variable
    env_ticker = os.getenv('TICKER')
    if env_ticker:
        return env_ticker.upper()
    
    # Default - CHANGE THIS TO YOUR DESIRED TICKER
    return 'XRP-USD'  # ← CAMBIA QUESTO CON IL TUO TICKER

class Config:
    """Central configuration class for the trading system"""
    
    # Set the ticker using the helper function
    TICKER = get_ticker()
    
    # API Keys and Tokens (from environment variables)
    EODHD_API_KEY = os.getenv('EODHD_API_KEY')
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')  # Optional
    
    # Trading Parameters
    EXCHANGE = 'CC'  # Default exchange
    
    # Analysis Parameters (matching the notebook)
    FAST_MA_WINDOW = 10  # Fast moving average for causal oscillator
    SLOW_MA_WINDOW = 40  # Slow moving average for causal oscillator
    
    # Spectral Analysis Parameters
    NPERSEG = 252  # Segment length for Welch periodogram (1 trading year)
    CWT_SCALES = list(range(2, 127))  # Scales for CWT analysis
    MONTE_CARLO_SIMULATIONS = 500  # Number of Monte Carlo simulations
    
    # Backtest Parameters
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', 10000.0))
    TRADING_FEES = float(os.getenv('TRADING_FEES', 0.001))  # 0.1% per trade
    IN_SAMPLE_RATIO = 0.7  # 70% for in-sample, 30% for out-of-sample
    
    # ======================================================================
    # Data Parameters (CORRETTO)
    
    # Definisci il lookback desiderato in anni.
    LOOKBACK_YEARS = 10
    
    # Calcola correttamente la data di inizio sottraendo anni di calendario.
    # Moltiplichiamo per 365.25 per tenere conto degli anni bisestili.
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    START_DATE = (datetime.now() - timedelta(days=LOOKBACK_YEARS * 365.25)).strftime('%Y-%m-%d')
    
    # Variabile deprecata, la lasciamo per compatibilità ma non è più il driver principale
    LOOKBACK_DAYS = LOOKBACK_YEARS * 252
    
    # ======================================================================
    
    # File Paths
    DATA_DIR = 'data'
    SIGNALS_FILE = os.path.join(DATA_DIR, 'signals.csv')
    HISTORICAL_DATA_FILE = os.path.join(DATA_DIR, 'historical_data.csv')
    BACKTEST_RESULTS_FILE = os.path.join(DATA_DIR, 'backtest_results.json')
    
    # Trading Rules (based on cycle phases)
    BULLISH_QUADRANTS = [
        "Quadrante 1 (Minimo -> Salita)",
        "Quadrante 2 (Salita -> Picco)"
    ]
    BEARISH_QUADRANTS = [
        "Quadrante 3 (Picco -> Discesa)",
        "Quadrante 4 (Discesa -> Minimo)"
    ]
    
    # Notification Settings
    SEND_TELEGRAM_NOTIFICATIONS = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
    SAVE_TO_GITHUB = bool(GITHUB_TOKEN)
    
    @classmethod
    def validate(cls):
        """Validate that all required configuration is present"""
        errors = []
        
        # EODHD API Key is always required
        if not cls.EODHD_API_KEY:
            errors.append("EODHD_API_KEY is missing")
        
        # Telegram is optional - only validate if tokens are partially configured
        if cls.TELEGRAM_BOT_TOKEN and not cls.TELEGRAM_CHAT_ID:
            errors.append("TELEGRAM_CHAT_ID is missing (BOT_TOKEN is set)")
        elif cls.TELEGRAM_CHAT_ID and not cls.TELEGRAM_BOT_TOKEN:
            errors.append("TELEGRAM_BOT_TOKEN is missing (CHAT_ID is set)")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        # Print configuration status
        print("✅ Configuration validated successfully")
        print(f"  - Ticker: {cls.TICKER}")
        print(f"  - EODHD API: Configured")
        print(f"  - Telegram: {'Configured' if cls.SEND_TELEGRAM_NOTIFICATIONS else 'Not configured (optional)'}")
        print(f"  - GitHub: {'Available' if cls.SAVE_TO_GITHUB else 'Not needed'}")
        
        return True
    
    @classmethod
    def get_phase_labels(cls):
        """Get phase quadrant labels for cycle analysis"""
        return cls.BULLISH_QUADRANTS + cls.BEARISH_QUADRANTS
