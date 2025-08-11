"""
Cycle analysis module for Kriterion Quant Trading System
Implements causal cycle analysis using Hilbert Transform
"""

import pandas as pd
import numpy as np
from scipy.signal import hilbert, welch, find_peaks
from scipy.stats import percentileofscore
import pywt
from typing import Tuple, Dict, Optional

from config import Config

class CycleAnalyzer:
    """Class to perform cycle analysis on price data"""
    
    def __init__(self, fast_window: int = None, slow_window: int = None):
        """
        Initialize the cycle analyzer
        
        Parameters
        ----------
        fast_window : int, optional
            Fast MA window. Defaults to Config.FAST_MA_WINDOW
        slow_window : int, optional
            Slow MA window. Defaults to Config.SLOW_MA_WINDOW
        """
        self.fast_window = fast_window or Config.FAST_MA_WINDOW
        self.slow_window = slow_window or Config.SLOW_MA_WINDOW
    
    def create_causal_oscillator(self, price_series: pd.Series) -> pd.Series:
        """
        Create a causal oscillator using dual moving average difference
        This avoids look-ahead bias
        
        Parameters
        ----------
        price_series : pd.Series
            Price series (typically close prices)
        
        Returns
        -------
        pd.Series
            Causal oscillator values
        """
        # Calculate moving averages
        fast_ma = price_series.rolling(window=self.fast_window).mean()
        slow_ma = price_series.rolling(window=self.slow_window).mean()
        
        # Oscillator is the difference
        oscillator = fast_ma - slow_ma
        
        return oscillator
    
    def apply_hilbert_transform(self, oscillator: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Apply Hilbert Transform to extract phase and amplitude
        
        Parameters
        ----------
        oscillator : pd.Series
            Oscillator series
        
        Returns
        -------
        Tuple[pd.Series, pd.Series]
            Phase and amplitude series
        """
        # Remove NaN values for Hilbert transform
        clean_oscillator = oscillator.dropna()
        
        if len(clean_oscillator) < 10:
            raise ValueError("Not enough data points for Hilbert transform")
        
        # Calculate analytic signal
        analytic_signal = hilbert(clean_oscillator.values)
        
        # Extract phase and amplitude
        phase = np.angle(analytic_signal)
        amplitude = np.abs(analytic_signal)
        
        # Convert back to series with original index
        phase_series = pd.Series(phase, index=clean_oscillator.index, name='phase')
        amplitude_series = pd.Series(amplitude, index=clean_oscillator.index, name='amplitude')
        
        return phase_series, amplitude_series
    
    def classify_phase_quadrant(self, phase: pd.Series) -> pd.Series:
        """
        Classify phase into quadrants for trading signals
        
        Parameters
        ----------
        phase : pd.Series
            Phase series in radians (-π to π)
        
        Returns
        -------
        pd.Series
            Quadrant classification
        """
        bins = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
        labels = Config.get_phase_labels()
        
        quadrants = pd.cut(phase, bins=bins, labels=labels, include_lowest=True)
        
        return quadrants
    
    def analyze_cycle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform complete cycle analysis on price data
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with OHLC data
        
        Returns
        -------
        pd.DataFrame
            DataFrame with added cycle analysis columns
        """
        print("⚙️ Performing cycle analysis...")
        
        # Create copy to avoid modifying original
        result_df = df.copy()
        
        # Step 1: Create causal oscillator
        oscillator = self.create_causal_oscillator(result_df['close'])
        result_df['oscillator'] = oscillator
        
        # Step 2: Apply Hilbert Transform
        phase, amplitude = self.apply_hilbert_transform(oscillator)
        result_df['phase'] = phase
        result_df['amplitude'] = amplitude
        
        # Step 3: Classify phase quadrants
        result_df['phase_quadrant'] = self.classify_phase_quadrant(result_df['phase'])
        
        # Step 4: Determine if in bullish regime
        result_df['bullish_regime'] = result_df['phase_quadrant'].isin(Config.BULLISH_QUADRANTS)
        
        # Remove rows with NaN values from moving averages
        result_df.dropna(inplace=True)
        
        print(f"✅ Cycle analysis complete. Analyzed {len(result_df)} data points")
        
        return result_df
    
    def run_spectral_analysis(self, oscillator: pd.Series) -> Dict:
        """
        Run spectral analysis using Welch periodogram
        
        Parameters
        ----------
        oscillator : pd.Series
            Oscillator series to analyze
        
        Returns
        -------
        Dict
            Spectral analysis results
        """
        clean_oscillator = oscillator.dropna().values
        
        # Welch periodogram
        frequencies, power = welch(clean_oscillator, fs=1, nperseg=min(Config.NPERSEG, len(clean_oscillator)//2))
        
        # Convert to periods
        periods = 1 / frequencies[1:] if len(frequencies) > 1 else np.array([])
        power = power[1:] if len(power) > 1 else np.array([])
        
        # Find peaks
        peaks, properties = find_peaks(power, prominence=np.max(power) * 0.1)
        
        dominant_period = periods[np.argmax(power)] if len(power) > 0 else None
        dominant_power = np.max(power) if len(power) > 0 else None
        
        return {
            'periods': periods,
            'power': power,
            'peaks': peaks,
            'dominant_period': dominant_period,
            'dominant_power': dominant_power
        }
    
    def run_monte_carlo_significance_test(
        self, 
        oscillator: pd.Series, 
        n_simulations: int = None
    ) -> Dict:
        """
        Test statistical significance of cycles using Monte Carlo
        
        Parameters
        ----------
        oscillator : pd.Series
            Oscillator series to test
        n_simulations : int, optional
            Number of simulations. Defaults to Config.MONTE_CARLO_SIMULATIONS
        
        Returns
        -------
        Dict
            Monte Carlo test results including p-value
        """
        n_simulations = n_simulations or Config.MONTE_CARLO_SIMULATIONS
        clean_oscillator = oscillator.dropna().values
        
        # Get observed max power
        spectral_results = self.run_spectral_analysis(oscillator)
        observed_max_power = spectral_results['dominant_power']
        
        if observed_max_power is None:
            return {'p_value': 1.0, 'significant': False}
        
        # Run simulations
        simulated_max_powers = []
        
        for _ in range(n_simulations):
            # Shuffle data to destroy temporal patterns
            shuffled = np.random.permutation(clean_oscillator)
            
            # Calculate periodogram for shuffled data
            _, sim_power = welch(shuffled, fs=1, nperseg=min(Config.NPERSEG, len(shuffled)//2))
            
            if len(sim_power) > 0:
                simulated_max_powers.append(np.max(sim_power))
        
        # Calculate p-value
        if simulated_max_powers:
            p_value = np.sum(np.array(simulated_max_powers) >= observed_max_power) / len(simulated_max_powers)
        else:
            p_value = 1.0
        
        return {
            'p_value': p_value,
            'significant': p_value < 0.05,
            'observed_power': observed_max_power,
            'simulated_powers': simulated_max_powers
        }
    
    def calculate_forward_returns(
        self, 
        df: pd.DataFrame, 
        horizons: list = [1, 5, 10, 21]
    ) -> pd.DataFrame:
        """
        Calculate forward returns for diagnostic analysis
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with price and phase data
        horizons : list, optional
            Forward return horizons in days
        
        Returns
        -------
        pd.DataFrame
            DataFrame with forward returns added
        """
        result_df = df.copy()
        
        for horizon in horizons:
            result_df[f'fwd_return_{horizon}d'] = (
                result_df['close'].pct_change(periods=horizon).shift(-horizon) * 100
            )
        
        return result_df
    
    def get_phase_performance_map(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create performance map by phase quadrant
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with phase and forward returns
        
        Returns
        -------
        pd.DataFrame
            Performance statistics by phase quadrant
        """
        # Ensure forward returns are calculated
        if 'fwd_return_1d' not in df.columns:
            df = self.calculate_forward_returns(df)
        
        # Group by phase quadrant and calculate mean returns
        performance_map = df.groupby('phase_quadrant')[
            [f'fwd_return_{h}d' for h in [1, 5, 10, 21]]
        ].mean()
        
        return performance_map
