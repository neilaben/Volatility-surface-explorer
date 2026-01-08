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
        
    def fetch_underlying_data(self) -> Dict:
        """
        Fetch current underlying asset data.
        
        Returns:
            Dictionary with current price, volume, and other metrics
        """
        ticker_obj = yf.Ticker(self.ticker)
        info = ticker_obj.info
        
        self.spot_price = info.get('currentPrice', info.get('regularMarketPrice'))
        
        return {
            'price': self.spot_price,
            'volume': info.get('volume'),
            'bid': info.get('bid'),
            'ask': info.get('ask'),
            'previous_close': info.get('previousClose')
        }
    
    def get_option_chain(self, expiry_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch complete options chain for the underlying.
        
        Args:
            expiry_date: Specific expiration date (YYYY-MM-DD). If None, fetches all.
            
        Returns:
            DataFrame with options data including strikes, prices, IVs, volumes
        """
        ticker_obj = yf.Ticker(self.ticker)
        
        # Get current spot price
        if self.spot_price is None:
            self.fetch_underlying_data()
        
        # Get all available expiration dates
        expiry_dates = ticker_obj.options
        
        if not expiry_dates:
            raise ValueError(f"No options data available for {self.ticker}")
        
        # If specific expiry requested, use it; otherwise get all
        dates_to_fetch = [expiry_date] if expiry_date else expiry_dates
        
        all_options = []
        
        for exp_date in dates_to_fetch:
            try:
                # Fetch options chain for this expiry
                opt_chain = ticker_obj.option_chain(exp_date)
                
                # Process calls
                calls = opt_chain.calls.copy()
                calls['option_type'] = 'call'
                calls['expiration'] = exp_date
                
                # Process puts
                puts = opt_chain.puts.copy()
                puts['option_type'] = 'put'
                puts['expiration'] = exp_date
                
                # Combine
                combined = pd.concat([calls, puts], ignore_index=True)
                all_options.append(combined)
                
            except Exception as e:
                print(f"Error fetching data for expiry {exp_date}: {e}")
                continue
        
        if not all_options:
            raise ValueError("Failed to fetch any options data")
        
        # Combine all expiries
        self.data = pd.concat(all_options, ignore_index=True)
        
        # Add calculated fields
        self._process_options_data()
        
        return self.data
    
    def _process_options_data(self):
        """
        Process and enrich options data with additional fields.
        """
        if self.data is None:
            return
        
        # Convert expiration to datetime
        self.data['expiration_date'] = pd.to_datetime(self.data['expiration'])
        
        # Calculate time to expiration in years
        today = datetime.now()
        self.data['time_to_expiry'] = (
            (self.data['expiration_date'] - today).dt.days / 365.25
        )
        
        # Calculate moneyness (S/K)
        self.data['moneyness'] = self.spot_price / self.data['strike']
        
        # Log moneyness
        self.data['log_moneyness'] = np.log(self.data['moneyness'])
        
        # Mid price
        self.data['mid_price'] = (self.data['bid'] + self.data['ask']) / 2
        
        # Bid-ask spread
        self.data['spread'] = self.data['ask'] - self.data['bid']
        self.data['spread_pct'] = (
            self.data['spread'] / self.data['mid_price'] * 100
        )
        
        # Filter out options with no volume or open interest (stale/illiquid)
        self.data = self.data[
            (self.data['volume'].notna() & (self.data['volume'] > 0)) |
            (self.data['openInterest'].notna() & (self.data['openInterest'] > 0))
        ].copy()
        
        # Filter out deep ITM/OTM options (typically unreliable)
        self.data = self.data[
            (self.data['moneyness'] > 0.7) & 
            (self.data['moneyness'] < 1.3)
        ].copy()
        
        # Filter out very short-dated options (< 7 days)
        self.data = self.data[self.data['time_to_expiry'] > 7/365].copy()
        
    def get_atm_options(self, window: float = 0.02) -> pd.DataFrame:
        """
        Get at-the-money options (within a specified window).
        
        Args:
            window: Percentage window around ATM (default: 2%)
            
        Returns:
            DataFrame with ATM options only
        """
        if self.data is None:
            self.get_option_chain()
        
        atm_filter = (
            (self.data['moneyness'] >= 1 - window) &
            (self.data['moneyness'] <= 1 + window)
        )
        
        return self.data[atm_filter].copy()
    
    def get_options_by_expiry(self, expiry_date: str) -> pd.DataFrame:
        """
        Get all options for a specific expiration date.
        
        Args:
            expiry_date: Expiration date in YYYY-MM-DD format
            
        Returns:
            DataFrame filtered to specific expiry
        """
        if self.data is None:
            self.get_option_chain()
        
        return self.data[self.data['expiration'] == expiry_date].copy()
    
    def get_surface_grid(
        self, 
        option_type: str = 'call',
        min_maturity_days: int = 7,
        max_maturity_days: int = 365
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare grid data for volatility surface construction.
        
        Args:
            option_type: 'call' or 'put'
            min_maturity_days: Minimum time to expiry in days
            max_maturity_days: Maximum time to expiry in days
            
        Returns:
            Tuple of (strikes, maturities, implied_vols) as 2D arrays
        """
        if self.data is None:
            self.get_option_chain()
        
        # Filter by option type
        filtered = self.data[self.data['option_type'] == option_type].copy()
        
        # Filter by maturity range
        filtered = filtered[
            (filtered['time_to_expiry'] >= min_maturity_days/365) &
            (filtered['time_to_expiry'] <= max_maturity_days/365)
        ].copy()
        
        # Remove options with invalid implied volatility
        filtered = filtered[
            filtered['impliedVolatility'].notna() &
            (filtered['impliedVolatility'] > 0) &
            (filtered['impliedVolatility'] < 3)  # Remove outliers > 300%
        ].copy()
        
        if filtered.empty:
            raise ValueError("No valid options data for surface construction")
        
        # Create grid
        strikes = filtered['strike'].values
        maturities = filtered['time_to_expiry'].values
        ivs = filtered['impliedVolatility'].values
        
        return strikes, maturities, ivs
    
    def get_term_structure(self, option_type: str = 'call') -> pd.DataFrame:
        """
        Get ATM volatility term structure.
        
        Args:
            option_type: 'call' or 'put'
            
        Returns:
            DataFrame with maturity vs ATM implied vol
        """
        atm_data = self.get_atm_options()
        term_structure = atm_data[
            atm_data['option_type'] == option_type
        ].groupby('expiration').agg({
            'impliedVolatility': 'mean',
            'time_to_expiry': 'first'
        }).reset_index()
        
        term_structure = term_structure.sort_values('time_to_expiry')
        
        return term_structure
    
    def compute_summary_statistics(self) -> Dict:
        """
        Compute summary statistics for the options chain.
        
        Returns:
            Dictionary with key metrics
        """
        if self.data is None:
            self.get_option_chain()
        
        calls = self.data[self.data['option_type'] == 'call']
        puts = self.data[self.data['option_type'] == 'put']
        
        return {
            'total_options': len(self.data),
            'num_calls': len(calls),
            'num_puts': len(puts),
            'num_expiries': self.data['expiration'].nunique(),
            'avg_iv_calls': calls['impliedVolatility'].mean(),
            'avg_iv_puts': puts['impliedVolatility'].mean(),
            'total_volume': self.data['volume'].sum(),
            'total_open_interest': self.data['openInterest'].sum(),
            'avg_spread_pct': self.data['spread_pct'].mean(),
            'spot_price': self.spot_price
        }


# Example usage
if __name__ == "__main__":
    # Initialize fetcher for SPY
    fetcher = OptionDataFetcher(ticker='SPY')
    
    # Get underlying data
    underlying = fetcher.fetch_underlying_data()
    print(f"Spot Price: ${underlying['price']:.2f}")
    
    # Fetch complete options chain
    options_data = fetcher.get_option_chain()
    print(f"\nFetched {len(options_data)} options contracts")
    
    # Get summary statistics
    stats = fetcher.compute_summary_statistics()
    print(f"\nSummary Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Get ATM volatility term structure
    term_struct = fetcher.get_term_structure(option_type='call')
    print(f"\nATM Term Structure:")
    print(term_struct)
