"""
Black-Scholes Option Pricing Model
Implements pricing, Greeks, and implied volatility calculation.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, minimize
from typing import Union, Dict, Tuple
import pandas as pd


class BlackScholes:
    """
    Black-Scholes option pricing model with Greeks calculation.
    
    Supports both European calls and puts.
    """
    
    def __init__(self, risk_free_rate: float = 0.05, dividend_yield: float = 0.0):
        """
        Initialize Black-Scholes pricer.
        
        Args:
            risk_free_rate: Annual risk-free interest rate
            dividend_yield: Annual dividend yield of underlying
        """
        self.r = risk_free_rate
        self.q = dividend_yield
    
    def _d1(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate d1 parameter."""
        return (np.log(S/K) + (self.r - self.q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    
    def _d2(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate d2 parameter."""
        return self._d1(S, K, T, sigma) - sigma*np.sqrt(T)
    
    def price(
        self, 
        S: Union[float, np.ndarray], 
        K: Union[float, np.ndarray], 
        T: Union[float, np.ndarray], 
        sigma: Union[float, np.ndarray],
        option_type: str = 'call'
    ) -> Union[float, np.ndarray]:
        """
        Calculate Black-Scholes option price.
        
        Args:
            S: Spot price of underlying
            K: Strike price
            T: Time to expiration (years)
            sigma: Volatility (annual)
            option_type: 'call' or 'put'
            
        Returns:
            Option price
        """
        # Handle arrays
        S = np.asarray(S)
        K = np.asarray(K)
        T = np.asarray(T)
        sigma = np.asarray(sigma)
        
        # Avoid division by zero
        T = np.maximum(T, 1e-10)
        sigma = np.maximum(sigma, 1e-10)
        
        d1 = self._d1(S, K, T, sigma)
        d2 = self._d2(S, K, T, sigma)
        
        if option_type.lower() == 'call':
            price = (S * np.exp(-self.q*T) * norm.cdf(d1) - 
                    K * np.exp(-self.r*T) * norm.cdf(d2))
        elif option_type.lower() == 'put':
            price = (K * np.exp(-self.r*T) * norm.cdf(-d2) - 
                    S * np.exp(-self.q*T) * norm.cdf(-d1))
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        return price
    
    def delta(
        self, 
        S: float, 
        K: float, 
        T: float, 
        sigma: float,
        option_type: str = 'call'
    ) -> float:
        """
        Calculate Delta (∂V/∂S).
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Delta value
        """
        d1 = self._d1(S, K, T, sigma)
        
        if option_type.lower() == 'call':
            return np.exp(-self.q*T) * norm.cdf(d1)
        elif option_type.lower() == 'put':
            return np.exp(-self.q*T) * (norm.cdf(d1) - 1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
    
    def gamma(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        Calculate Gamma (∂²V/∂S²).
        
        Gamma is the same for calls and puts.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration
            sigma: Volatility
            
        Returns:
            Gamma value
        """
        d1 = self._d1(S, K, T, sigma)
        
        return (np.exp(-self.q*T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))
    
    def vega(self, S: float, K: float, T: float, sigma: float) -> float:
        """
        Calculate Vega (∂V/∂σ).
        
        Vega is the same for calls and puts.
        Note: Returns Vega per 1% change in volatility.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration
            sigma: Volatility
            
        Returns:
            Vega value (per 1% vol change)
        """
        d1 = self._d1(S, K, T, sigma)
        
        vega = S * np.exp(-self.q*T) * norm.pdf(d1) * np.sqrt(T)
        
        # Return per 1% change in volatility
        return vega / 100
    
    def theta(
        self, 
        S: float, 
        K: float, 
        T: float, 
        sigma: float,
        option_type: str = 'call'
    ) -> float:
        """
        Calculate Theta (∂V/∂T).
        
        Note: Returns Theta per day (negative for long positions).
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Theta value (per day)
        """
        d1 = self._d1(S, K, T, sigma)
        d2 = self._d2(S, K, T, sigma)
        
        term1 = -(S * np.exp(-self.q*T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        
        if option_type.lower() == 'call':
            term2 = -self.r * K * np.exp(-self.r*T) * norm.cdf(d2)
            term3 = self.q * S * np.exp(-self.q*T) * norm.cdf(d1)
            theta = term1 + term2 + term3
        elif option_type.lower() == 'put':
            term2 = self.r * K * np.exp(-self.r*T) * norm.cdf(-d2)
            term3 = -self.q * S * np.exp(-self.q*T) * norm.cdf(-d1)
            theta = term1 + term2 + term3
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        # Return per day
        return theta / 365
    
    def rho(
        self, 
        S: float, 
        K: float, 
        T: float, 
        sigma: float,
        option_type: str = 'call'
    ) -> float:
        """
        Calculate Rho (∂V/∂r).
        
        Note: Returns Rho per 1% change in interest rate.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Rho value (per 1% rate change)
        """
        d2 = self._d2(S, K, T, sigma)
        
        if option_type.lower() == 'call':
            rho = K * T * np.exp(-self.r*T) * norm.cdf(d2)
        elif option_type.lower() == 'put':
            rho = -K * T * np.exp(-self.r*T) * norm.cdf(-d2)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        # Return per 1% change in rate
        return rho / 100
    
    def all_greeks(
        self, 
        S: float, 
        K: float, 
        T: float, 
        sigma: float,
        option_type: str = 'call'
    ) -> Dict[str, float]:
        """
        Calculate all Greeks at once.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary with all Greeks
        """
        return {
            'price': self.price(S, K, T, sigma, option_type),
            'delta': self.delta(S, K, T, sigma, option_type),
            'gamma': self.gamma(S, K, T, sigma),
            'vega': self.vega(S, K, T, sigma),
            'theta': self.theta(S, K, T, sigma, option_type),
            'rho': self.rho(S, K, T, sigma, option_type)
        }
    
    def implied_volatility(
        self, 
        market_price: float,
        S: float, 
        K: float, 
        T: float,
        option_type: str = 'call',
        max_iterations: int = 100,
        precision: float = 1e-5
    ) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Args:
            market_price: Observed market price
            S: Spot price
            K: Strike price
            T: Time to expiration
            option_type: 'call' or 'put'
            max_iterations: Maximum iterations for convergence
            precision: Convergence precision
            
        Returns:
            Implied volatility
        """
        # Objective function: price difference
        def objective(sigma):
            return self.price(S, K, T, sigma, option_type) - market_price
        
        # Use Brent's method for robustness
        try:
            # Search in range [0.001, 5.0] (0.1% to 500% vol)
            iv = brentq(objective, 0.001, 5.0, maxiter=max_iterations, xtol=precision)
            return iv
        except ValueError:
            # If no solution found, try Newton-Raphson with initial guess
            sigma = 0.3  # Initial guess: 30% vol
            
            for _ in range(max_iterations):
                price_diff = objective(sigma)
                
                if abs(price_diff) < precision:
                    return sigma
                
                # Vega (derivative of price w.r.t. sigma)
                vega_val = self.vega(S, K, T, sigma) * 100  # Convert back to absolute
                
                if vega_val == 0:
                    break
                
                # Newton-Raphson update
                sigma = sigma - price_diff / vega_val
                
                # Keep sigma positive
                sigma = max(sigma, 0.001)
            
            # Return NaN if no convergence
            return np.nan
    
    def build_implied_vol_surface(
        self, 
        options_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Build implied volatility surface from options data.
        
        Args:
            options_data: DataFrame with option chain data
            
        Returns:
            DataFrame with calculated IVs
        """
        result = options_data.copy()
        
        # Calculate implied vol for each option
        ivs = []
        for idx, row in result.iterrows():
            # Use mid price if available, else lastPrice
            if pd.notna(row['mid_price']) and row['mid_price'] > 0:
                market_price = row['mid_price']
            else:
                market_price = row['lastPrice']
            
            # Skip if no valid price
            if pd.isna(market_price) or market_price <= 0:
                ivs.append(np.nan)
                continue
            
            # Calculate IV
            try:
                iv = self.implied_volatility(
                    market_price=market_price,
                    S=row.get('spot_price', 100),  # Assuming spot is stored
                    K=row['strike'],
                    T=row['time_to_expiry'],
                    option_type=row['option_type']
                )
                ivs.append(iv)
            except:
                ivs.append(np.nan)
        
        result['calculated_iv'] = ivs
        
        return result


# Example usage
if __name__ == "__main__":
    # Initialize Black-Scholes model
    bs = BlackScholes(risk_free_rate=0.05)
    
    # Example parameters
    S = 100    # Spot price
    K = 105    # Strike price
    T = 0.5    # 6 months to expiration
    sigma = 0.25  # 25% volatility
    
    # Price a call option
    call_price = bs.price(S, K, T, sigma, 'call')
    print(f"Call Price: ${call_price:.4f}")
    
    # Calculate all Greeks
    greeks = bs.all_greeks(S, K, T, sigma, 'call')
    print("\nGreeks:")
    for greek, value in greeks.items():
        print(f"  {greek.capitalize()}: {value:.6f}")
    
    # Calculate implied volatility
    market_price = 5.5
    iv = bs.implied_volatility(market_price, S, K, T, 'call')
    print(f"\nImplied Volatility for market price ${market_price}: {iv:.4f} ({iv*100:.2f}%)")
