"""
Options Data Fetcher - Polygon.io
Fetches real-time options data from Polygon.io API
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests
import streamlit as st
import time
import warnings

warnings.filterwarnings('ignore')


class OptionDataFetcher:
    """
    Fetches options data from Polygon.io API.
    Free tier: 5 API calls per minute
    
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
        self.ticker = ticker.upper()
        self.risk_free_rate = risk_free_rate
        self.data = None
        self.spot_price = None
        self._underlying = None
        self.base_url = "https://api.polygon.io"
        
        # Get API key from Streamlit secrets
        try:
            self.api_key = st.secrets["polygon"]["api_key"]
        except:
            st.error("""
            **Polygon.io API Key Not Configured**
            
            Please add your API key to Streamlit secrets:
            1. Go to Settings > Secrets
            2. Add:
```
            [polygon]
            api_key = "YOUR_KEY_HERE"
```
            
            Get free API key at: https://polygon.io/dashboard/signup
            """)
            self.api_key = None
    
    def _make_request(self, endpoint: str, params: dict = None) -> dict:
        """
        Make API request with rate limiting.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            JSON response
        """
        if not self.api_key:
            raise Exception("API key not configured")
        
        if params is None:
            params = {}
        
        params['apiKey'] = self.api_key
        
        url = f"{self.base_url}{endpoint}"
        
        # Rate limiting - free tier allows 5 calls/minute
        time.sleep(0.5)
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {e}")
            raise
    
    def fetch_underlying_data(self) -> Dict:
        """
        Fetch current underlying asset data.
        
        Returns:
            Dictionary with current price, volume, and other metrics
        """
        if not self.api_key:
            return {'ticker': self.ticker, 'price': 100, 'volume': 0}
        
        try:
            # Get previous day's data
            endpoint = f"/v2/aggs/ticker/{self.ticker}/prev"
            data = self._make_request(endpoint)
            
            if 'results' not in data or len(data['results']) == 0:
                st.warning(f"No data found for {self.ticker}")
                return {'ticker': self.ticker, 'price': 100, 'volume': 0}
            
            result = data['results'][0]
            
            self.spot_price = result['c']  # Close price
            
            return {
                'ticker': self.ticker,
                'price': self.spot_price,
                'volume': result.get('v', 0),
                'open': result.get('o', 0),
                'high': result.get('h', 0),
                'low': result.get('l', 0),
                'vwap': result.get('vw', 0)
            }
            
        except Exception as e:
            st.warning(f"Error fetching underlying data: {e}")
            return {'ticker': self.ticker, 'price': 100, 'volume': 0}
    
    def get_option_chain(self, min_dte: int = 7, max_dte: int = 365) -> pd.DataFrame:
        """
        Fetch options chain for all available expirations within date range.
        
        Args:
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration
            
        Returns:
            DataFrame containing options data
        """
        if not self.api_key:
            st.error("API key required to fetch options data")
            return pd.DataFrame()
        
        try:
            # Ensure we have spot price
            if self.spot_price is None:
                self.fetch_underlying_data()
            
            # Calculate date range
            today = datetime.now()
            exp_min = (today + timedelta(days=min_dte)).strftime('%Y-%m-%d')
            exp_max = (today + timedelta(days=max_dte)).strftime('%Y-%m-%d')
            
            st.info(f"Fetching option contracts for {self.ticker}...")
            
            # Get option contracts
            endpoint = "/v3/reference/options/contracts"
            params = {
                'underlying_ticker': self.ticker,
                'contract_type': '',  # Both calls and puts
                'expiration_date.gte': exp_min,
                'expiration_date.lte': exp_max,
                'limit': 1000
            }
            
            contracts_data = self._make_request(endpoint, params)
            
            if 'results' not in contracts_data or len(contracts_data['results']) == 0:
                st.warning(f"No option contracts found for {self.ticker}")
                return pd.DataFrame()
            
            contracts = contracts_data['results']
            st.info(f"Found {len(contracts)} option contracts. Fetching quotes (this may take a moment)...")
            
            # Process contracts
            all_options = []
            processed = 0
            
            for contract in contracts[:100]:  # Limit to 100 to avoid rate limits
                try:
                    ticker_symbol = contract['ticker']
                    
                    # Get quote for this contract
                    quote_endpoint = f"/v3/quotes/{ticker_symbol}"
                    quote_params = {'limit': 1, 'order': 'desc'}
                    
                    quote_data = self._make_request(quote_endpoint, quote_params)
                    
                    if 'results' not in quote_data or len(quote_data['results']) == 0:
                        continue
                    
                    quote = quote_data['results'][0]
                    
                    # Calculate days to expiry
                    exp_date = contract['expiration_date']
                    exp_dt = datetime.strptime(exp_date, '%Y-%m-%d')
                    days_to_expiry = (exp_dt - today).days
                    
                    # Calculate moneyness
                    strike = contract['strike_price']
                    moneyness = strike / self.spot_price
                    
                    # Only include options in reasonable moneyness range
                    if moneyness < 0.7 or moneyness > 1.3:
                        continue
                    
                    option_data = {
                        'strike': strike,
                        'expiration': exp_date,
                        'option_type': contract['contract_type'].lower(),
                        'bid': quote.get('bid_price', 0),
                        'ask': quote.get('ask_price', 0),
                        'mid_price': (quote.get('bid_price', 0) + quote.get('ask_price', 0)) / 2,
                        'lastPrice': quote.get('last_price', 0),
                        'volume': 0,  # Polygon free tier doesn't include volume
                        'openInterest': 0,  # Not available in free tier
                        'days_to_expiry': days_to_expiry,
                        'time_to_expiry': days_to_expiry / 365.0,
                        'spot_price': self.spot_price,
                        'moneyness': moneyness
                    }
                    
                    # Estimate implied volatility if mid_price available
                    if option_data['mid_price'] > 0:
                        option_data['impliedVolatility'] = self._estimate_iv(
                            option_data['mid_price'],
                            self.spot_price,
                            strike,
                            option_data['time_to_expiry'],
                            option_data['option_type']
                        )
                    else:
                        option_data['impliedVolatility'] = 0.20  # Default
                    
                    all_options.append(option_data)
                    processed += 1
                    
                    # Update progress
                    if processed % 10 == 0:
                        st.info(f"Processed {processed} options...")
                    
                    # Rate limiting
                    time.sleep(0.2)
                    
                except Exception as e:
                    continue
            
            if len(all_options) == 0:
                st.warning("No valid option data retrieved")
                return pd.DataFrame()
            
            df = pd.DataFrame(all_options)
            self.data = df
            
            st.success(f"Successfully loaded {len(df)} options contracts")
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching option chain: {e}")
            return pd.DataFrame()
    
    def _estimate_iv(self, price: float, S: float, K: float, T: float, 
                     option_type: str) -> float:
        """
        Rough IV estimate using simplified approach.
        
        Args:
            price: Option price
            S: Spot price
            K: Strike price
            T: Time to expiry
            option_type: 'call' or 'put'
            
        Returns:
            Estimated implied volatility
        """
        if T <= 0 or S <= 0:
            return 0.20
        
        # Simple approximation
        intrinsic = max(0, S - K) if option_type == 'call' else max(0, K - S)
        time_value = max(0, price - intrinsic)
        
        if time_value <= 0:
            return 0.10
        
        # Rough approximation: IV ≈ time_value / (S * sqrt(T) * 0.4)
        iv_estimate = time_value / (S * np.sqrt(T) * 0.4)
        
        return max(0.05, min(1.0, iv_estimate))
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and filter options data"""
        if len(df) == 0:
            return df
        
        df = df[df['mid_price'] > 0].copy()
        df = df[(df['moneyness'] >= 0.7) & (df['moneyness'] <= 1.3)].copy()
        
        if 'impliedVolatility' in df.columns:
            df = df[(df['impliedVolatility'] > 0) & (df['impliedVolatility'] < 3.0)].copy()
        
        return df
    
    def calculate_time_to_expiry(self, expiry_date: str) -> float:
        """Calculate time to expiry in years"""
        exp_dt = datetime.strptime(expiry_date, '%Y-%m-%d')
        days_to_expiry = (exp_dt - datetime.now()).days
        return max(days_to_expiry / 365.0, 1/365.0)
    
    def get_atm_options(self, tolerance: float = 0.05) -> pd.DataFrame:
        """Filter for at-the-money options"""
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
        """Get all options for a specific expiration date"""
        if self.data is None or len(self.data) == 0:
            self.get_option_chain()
        
        if self.data is None or len(self.data) == 0:
            return pd.DataFrame()
        
        return self.data[self.data['expiration'] == expiry_date].copy()
    
    def get_unique_expiries(self) -> List[str]:
        """Get list of unique expiration dates"""
        if self.data is None or len(self.data) == 0:
            self.get_option_chain()
        
        if self.data is None or len(self.data) == 0:
            return []
        
        return sorted(self.data['expiration'].unique().tolist())
    
    def get_strike_range(self, moneyness_range: Tuple[float, float] = (0.8, 1.2)) -> List[float]:
        """Get strikes within a moneyness range"""
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
    
    def get_surface_grid(self, option_type: str = 'call',
                        min_maturity_days: int = 7,
                        max_maturity_days: int = 365) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get gridded volatility surface data"""
        if self.data is None:
            self.get_option_chain(min_maturity_days, max_maturity_days)
        
        df = self.data[self.data['option_type'] == option_type].copy()
        
        strikes = df['strike'].values
        maturities = df['time_to_expiry'].values
        ivs = df['impliedVolatility'].values
        
        return strikes, maturities, ivs
    
    def get_term_structure(self, option_type: str = 'call') -> pd.DataFrame:
        """Get ATM volatility term structure"""
        atm_data = self.get_atm_options(tolerance=0.02)
        
        if len(atm_data) == 0:
            return pd.DataFrame()
        
        atm_data = atm_data[atm_data['option_type'] == option_type]
        
        term_struct = atm_data.groupby('expiration').agg({
            'impliedVolatility': 'mean',
            'time_to_expiry': 'first'
        }).reset_index()
        
        return term_struct.sort_values('time_to_expiry')
    
    def compute_summary_statistics(self) -> Dict:
        """Compute summary statistics"""
        if self.data is None or len(self.data) == 0:
            return {
                'total_options': 0,
                'num_calls': 0,
                'num_puts': 0,
                'num_expiries': 0
            }
        
        return {
            'total_options': len(self.data),
            'num_calls': len(self.data[self.data['option_type'] == 'call']),
            'num_puts': len(self.data[self.data['option_type'] == 'put']),
            'num_expiries': self.data['expiration'].nunique()
        }
    
    def refresh_data(self):
        """Clear cached data"""
        self.data = None
        self.spot_price = None
        self._underlying = None
