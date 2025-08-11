"""
Data fetching module for Kriterion Quant Trading System
Handles all data acquisition from EODHD API
"""

import pandas as pd
import requests
import time
from typing import Optional, Dict, List
from datetime import datetime
import os

from config import Config

class DataFetcher:
    """Class to handle data fetching from EODHD API"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the data fetcher
        
        Parameters
        ----------
        api_key : str, optional
            EODHD API key. If not provided, will use from Config
        """
        self.api_key = api_key or Config.EODHD_API_KEY
        if not self.api_key:
            raise ValueError("EODHD API key is required")
        
        # Create data directory if it doesn't exist
        os.makedirs(Config.DATA_DIR, exist_ok=True)
    
    def fetch_historical_data(
        self,
        ticker: str = None,
        start_date: str = None,
        end_date: str = None,
        max_retries: int = 5
    ) -> pd.DataFrame:
        """
        Fetch historical OHLC data from EODHD API
        
        Parameters
        ----------
        ticker : str, optional
            Stock ticker symbol. Defaults to Config.TICKER
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format. Defaults to Config.START_DATE
        end_date : str, optional
            End date in 'YYYY-MM-DD' format. Defaults to Config.END_DATE
        max_retries : int, optional
            Maximum number of retry attempts for rate limiting
        
        Returns
        -------
        pd.DataFrame
            DataFrame with OHLC data and volume
        """
        ticker = ticker or Config.TICKER
        start_date = start_date or Config.START_DATE
        end_date = end_date or Config.END_DATE
        
        print(f"📡 Fetching data for {ticker} from {start_date} to {end_date}...")
        
        endpoint = f"https://eodhistoricaldata.com/api/eod/{ticker.upper()}.{Config.EXCHANGE}"
        params = {
            'api_token': self.api_key,
            'from': start_date,
            'to': end_date,
            'period': 'd',
            'fmt': 'json'
        }
        
        retries = 0
        backoff_factor = 1
        
        while retries < max_retries:
            try:
                response = requests.get(endpoint, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if not data:
                        raise ValueError(f"No data returned for {ticker}")
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(data)
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    
                    # Ensure we have all required columns
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    if 'adjusted_close' in df.columns:
                        df['close'] = df['adjusted_close']
                    
                    # Select and validate columns
                    df = df[required_cols]
                    
                    print(f"✅ Successfully fetched {len(df)} days of data")
                    return df
                
                elif response.status_code == 429:
                    print(f"⚠️ Rate limited. Waiting {backoff_factor} seconds...")
                    time.sleep(backoff_factor)
                    retries += 1
                    backoff_factor *= 2
                
                elif response.status_code == 404:
                    raise ValueError(f"Ticker '{ticker}' not found")
                
                else:
                    raise Exception(f"API error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ Network error: {e}. Retrying...")
                time.sleep(backoff_factor)
                retries += 1
                backoff_factor *= 2
        
        raise Exception(f"Failed to fetch data after {max_retries} retries")
    
    def save_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Save DataFrame to CSV file
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to save
        filename : str, optional
            Filename to save to. Defaults to Config.HISTORICAL_DATA_FILE
        
        Returns
        -------
        str
            Path to saved file
        """
        filename = filename or Config.HISTORICAL_DATA_FILE
        df.to_csv(filename)
        print(f"💾 Data saved to {filename}")
        return filename
    
    def load_data(self, filename: str = None) -> pd.DataFrame:
        """
        Load DataFrame from CSV file
        
        Parameters
        ----------
        filename : str, optional
            Filename to load from. Defaults to Config.HISTORICAL_DATA_FILE
        
        Returns
        -------
        pd.DataFrame
            Loaded DataFrame
        """
        filename = filename or Config.HISTORICAL_DATA_FILE
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Data file not found: {filename}")
        
        df = pd.read_csv(filename, index_col='date', parse_dates=True)
        print(f"📂 Loaded {len(df)} days of data from {filename}")
        return df
    
    def update_latest_data(self, ticker: str = None) -> pd.DataFrame:
        """
        Update data with the latest available information
        
        Parameters
        ----------
        ticker : str, optional
            Stock ticker symbol. Defaults to Config.TICKER
        
        Returns
        -------
        pd.DataFrame
            Updated DataFrame with latest data
        """
        ticker = ticker or Config.TICKER
        
        # Try to load existing data
        try:
            existing_df = self.load_data()
            last_date = existing_df.index[-1].strftime('%Y-%m-%d')
            print(f"📅 Last data point: {last_date}")
            
            # Fetch only new data
            new_df = self.fetch_historical_data(
                ticker=ticker,
                start_date=last_date,
                end_date=Config.END_DATE
            )
            
            # Combine and remove duplicates
            combined_df = pd.concat([existing_df, new_df])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df.sort_index(inplace=True)
            
        except FileNotFoundError:
            print("📥 No existing data found. Fetching full history...")
            combined_df = self.fetch_historical_data(ticker=ticker)
        
        # Save updated data
        self.save_data(combined_df)
        return combined_df
