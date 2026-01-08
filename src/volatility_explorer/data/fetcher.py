"""
Options Data Fetcher
Retrieves real-time and historical options data from market sources.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
import time

warnings.filterwarnings('ignore')


class OptionDataFetcher:
    """
    Fetches and processes options chain data for volatility surface construction.
    
    Attributes:
        ticker (str): Underlying asset ticker symbol
        data (pd.DataFrame): Cached options data
        spot_price (float): Current underlying price
        risk_free_rate (float): Risk-free interest rate
    """
    
    def __init__(self, ticker: str = 'SPY', risk_free_rate: float = 0.05):
        """
        Initialize the option data fetcher.
        
        Args:
            ticker: Stock ticker symbol
            risk_free_rate: Annual risk-free rate (default: 5%)
        """
        self.ticker = ticker
        self.risk_free_rate = risk_free_rate
        self.data = None
        self.spot_price = None
        self._underlying = None
        self._last_request_time = 0
        self._request_count = 0
        
    def _rate_limit(self):
        """
        Enforce rate limiting to avoid Yahoo Finance 429 errors.
        Waits 2 seconds between requests and tracks request count.
        """
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < 2:
            time.sleep(2 - time_since_last)
        
        self._last_request_time = time.time()
        self._request_count += 1
        
        if self._request_count % 10 == 0:
            time.sleep(5)
        
    def fetch_underlying_data(self) -> Dict:
        """
        Fetch current underlying asset data.
        
        Returns:
            Dictionary with current price, volume, and other metrics
        """
        self._rate_limit()
        
        try:
            ticker_obj = yf.Ticker(self.ticker)
            info = ticker_obj.info
            
            self.spot_price = info.get('currentPrice', info.get('regularMarketPrice', 100))
            
            return {
                'ticker': self.ticker,
                'price': self.spot_price,
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'beta': info.get('beta', 1.0),
                'dividend_yield': info.get('dividendYield', 0)
            }
        except Exception as e:
            print(f"Error fetching underlying data: {e}")
            return {'ticker': self.ticker, 'price': 100, 'volume': 0}
    
    def get_option_chain(self, min_dte: int = 7, max_dte: int = 365) -> pd.DataFrame:
        """
        Fetch options chain for all available expirations within date range.
        
        Args:
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration
            
        Returns:
            DataFrame containing options data with columns for strike, expiry, 
            bid, ask, volume, open interest, implied volatility, etc.
        """
        self._rate_limit()
        
        try:
            ticker_obj = yf.Ticker(self.ticker)
            
            if self.spot_price is None:
                self.fetch_underlying_data()
            
            expirations = ticker_obj.options
            
            if len(expirations) == 0:
                print(f"No options available for {self.ticker}")
                return pd.DataFrame()
            
            all_options = []
            
            for exp_date in expirations:
                exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
                days_to_expiry = (exp_datetime - datetime.now()).days
                
                if days_to_expiry < min_dte or days_to_expiry > max_dte:
                    continue
                
                time.sleep(0.5)
                
                try:
                    opt = ticker_obj.option_chain(exp_date)
                    
                    calls = opt.calls.copy()
                    calls['option_type'] = 'call'
                    calls['expiration'] = exp_date
                    calls['days_to_expiry'] = days_to_expiry
                    calls['time_to_expiry'] = days_to_expiry / 365.0
                    
                    puts = opt.puts.copy()
                    puts['option_type'] = 'put'
                    puts['expiration'] = exp_date
                    puts['days_to_expiry'] = days_to_expiry
                    puts['time_to_expiry'] = days_to_expiry / 365.0
                    
                    all_options.extend([calls, puts])
                    
                except Exception as e:
                    print(f"Error fetching options for {exp_date}: {e}")
                    continue
            
            if len(all_options) == 0:
                return pd.DataFrame()
            
            options_df = pd.concat(all_options, ignore_index=True)
            
            options_df['spot_price'] = self.spot_price
            options_df['moneyness'] = options_df['strike'] / self.spot_price
            options_df['mid_price'] = (options_df['bid'] + options_df['ask']) / 2
            
            options_df = self._clean_data(options_df)
            
            self.data = options_df
            
            return options_df
            
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                print("Rate limit reached. Please wait and try again.")
                time.sleep(60)
            else:
                print(f"Error fetching option chain: {e}")
            return pd.DataFrame()
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and filter options data.
        
        Args:
            df: Raw options DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        if len(df) == 0:
            return df
        
        df = df[(df['volume'] > 0) | (df['openInterest'] > 0)].copy()
        
        df = df[
            (df['moneyness'] >= 0.7) & (df['moneyness'] <= 1.3)
        ].copy()
        
        if 'impliedVolatility' in df.columns:
            df = df[
                (df['impliedVolatility'] > 0) & (df['impliedVolatility'] < 3.0)
            ].copy()
        
        df['volume'].fillna(0, inplace=True)
        df['openInterest'].fillna(0, inplace=True)
        
        return df
    
    def calculate_time_to_expiry(self, expiry_date: str) -> float:
        """
        Calculate time to expiry in years.
        
        Args:
            expiry_date: Expiration date string (YYYY-MM-DD)
            
        Returns:
            Time to expiry in years
        """
        exp_dt = datetime.strptime(expiry_date, '%Y-%m-%d')
        days_to_expiry = (exp_dt - datetime.now()).days
        return max(days_to_expiry / 365.0, 1/365.0)
    
    def get_atm_options(self, tolerance: float = 0.05) -> pd.DataFrame:
        """
        Filter for at-the-money options.
        
        Args:
            tolerance: Moneyness tolerance (default 5%)
            
        Returns:
            DataFrame with ATM options
        """
        if self.data is None or len(self.data) == 0:
            self.get_option_chain()
        
        if self.data is None or len(self.data) == 0:
            return pd.DataFrame()
        
        atm_mask = (
            (self.data['moneyness'] >= 1 - tolerance) &
            (self.data['moneyness'] <= 1 + tolerance)
        )
        
        return self.data[atm_mask].copy()
    
    def get_options_by_expiry(self, expiry_date: str) -> pd.DataFrame:
        """
        Get all options for a specific expiration date.
        
        Args:
            expiry_date: Expiration date (YYYY-MM-DD)
            
        Returns:
            DataFrame with options for that expiry
        """
        if self.data is None or len(self.data) == 0:
            self.get_option_chain()
        
        if self.data is None or len(self.data) == 0:
            return pd.DataFrame()
        
        return self.data[self.data['expiration'] == expiry_date].copy()
    
    def get_unique_expiries(self) -> List[str]:
        """
        Get list of unique expiration dates in the data.
        
        Returns:
            Sorted list of expiration dates
        """
        if self.data is None or len(self.data) == 0:
            self.get_option_chain()
        
        if self.data is None or len(self.data) == 0:
            return []
        
        return sorted(self.data['expiration'].unique().tolist())
    
    def get_strike_range(self, moneyness_range: Tuple[float, float] = (0.8, 1.2)) -> List[float]:
        """
        Get strikes within a moneyness range.
        
        Args:
            moneyness_range: Tuple of (min_moneyness, max_moneyness)
            
        Returns:
            Sorted list of strikes
        """
        if self.data is None or len(self.data) == 0:
            self.get_option_chain()
        
        if self.data is None or len(self.data) == 0:
            return []
        
        min_m, max_m = moneyness_range
        
        strikes = self.data[
            (self.data['moneyness'] >= min_m) &
            (self.data['moneyness'] <= max_m)
        ]['strike'].unique()
        
        return sorted(strikes.tolist())
    
    def refresh_data(self):
        """
        Clear cached data and force refresh on next fetch.
        """
        self.data = None
        self.spot_price = None
        self._underlying = None
