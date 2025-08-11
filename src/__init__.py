"""
Kriterion Quant Trading System
A cycle-based trading strategy using Hilbert Transform analysis
"""

__version__ = "1.0.0"
__author__ = "Kriterion Quant"

from .config import Config
from .data_fetcher import DataFetcher
from .cycle_analyzer import CycleAnalyzer
from .signal_generator import SignalGenerator
from .backtester import Backtester
from .notifier import TelegramNotifier

__all__ = [
    'Config',
    'DataFetcher',
    'CycleAnalyzer',
    'SignalGenerator',
    'Backtester',
    'TelegramNotifier'
]
