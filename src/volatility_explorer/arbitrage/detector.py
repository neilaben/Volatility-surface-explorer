"""
Arbitrage Detection for Options
Checks for violations of no-arbitrage conditions in options pricing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.interpolate import interp1d


class ArbitrageDetector:
    """
    Detect arbitrage opportunities and no-arbitrage violations in options data.
    
    Checks for:
    1. Monotonicity: Call prices decrease with strike
    2. Convexity: Butterfly spread constraints
    3. Calendar spreads: Later expiries >= earlier expiries
    4. Put-Call Parity violations
    5. Price bounds: C >= max(S - K*exp(-rT), 0)
    """
    
    def __init__(self, tolerance: float = 0.01):
        """
        Initialize arbitrage detector.
        
        Args:
            tolerance: Tolerance for numerical violations (default: 1%)
        """
        self.tolerance = tolerance
        self.violations = []
        
    def check_monotonicity(
        self,
        strikes: np.ndarray,
        prices: np.ndarray,
        option_type: str = 'call'
    ) -> List[Dict]:
        """
        Check if option prices are monotonic in strike.
        
        For calls: C(K1) >= C(K2) for K1 < K2
        For puts: P(K1) <= P(K2) for K1 < K2
        
        Args:
            strikes: Array of strike prices (must be sorted)
            prices: Array of option prices
            option_type: 'call' or 'put'
            
        Returns:
            List of violation dictionaries
        """
        violations = []
        
        # Sort by strike
        sorted_idx = np.argsort(strikes)
        strikes_sorted = strikes[sorted_idx]
        prices_sorted = prices[sorted_idx]
        
        for i in range(len(strikes_sorted) - 1):
            if option_type.lower() == 'call':
                # Calls should decrease with strike
                if prices_sorted[i] < prices_sorted[i+1] - self.tolerance:
                    violations.append({
                        'type': 'monotonicity',
                        'option_type': 'call',
                        'strike1': strikes_sorted[i],
                        'strike2': strikes_sorted[i+1],
                        'price1': prices_sorted[i],
                        'price2': prices_sorted[i+1],
                        'severity': abs(prices_sorted[i+1] - prices_sorted[i])
                    })
            else:  # put
                # Puts should increase with strike
                if prices_sorted[i] > prices_sorted[i+1] + self.tolerance:
                    violations.append({
                        'type': 'monotonicity',
                        'option_type': 'put',
                        'strike1': strikes_sorted[i],
                        'strike2': strikes_sorted[i+1],
                        'price1': prices_sorted[i],
                        'price2': prices_sorted[i+1],
                        'severity': abs(prices_sorted[i] - prices_sorted[i+1])
                    })
        
        return violations
    
    def check_convexity(
        self,
        strikes: np.ndarray,
        prices: np.ndarray,
        option_type: str = 'call'
    ) -> List[Dict]:
        """
        Check butterfly spread no-arbitrage condition.
        
        For any three strikes K1 < K2 < K3:
        C(K2) <= [(K3-K2)*C(K1) + (K2-K1)*C(K3)] / (K3-K1)
        
        Args:
            strikes: Array of strike prices
            prices: Array of option prices
            option_type: 'call' or 'put'
            
        Returns:
            List of violation dictionaries
        """
        violations = []
        
        # Sort by strike
        sorted_idx = np.argsort(strikes)
        strikes_sorted = strikes[sorted_idx]
        prices_sorted = prices[sorted_idx]
        
        # Check all triplets
        for i in range(len(strikes_sorted) - 2):
            K1, K2, K3 = strikes_sorted[i:i+3]
            C1, C2, C3 = prices_sorted[i:i+3]
            
            # Butterfly spread constraint
            lambda_val = (K3 - K2) / (K3 - K1)
            upper_bound = lambda_val * C1 + (1 - lambda_val) * C3
            
            if C2 > upper_bound + self.tolerance:
                violations.append({
                    'type': 'convexity',
                    'option_type': option_type,
                    'strikes': (K1, K2, K3),
                    'prices': (C1, C2, C3),
                    'max_price': upper_bound,
                    'actual_price': C2,
                    'severity': C2 - upper_bound
                })
        
        return violations
    
    def check_calendar_spreads(
        self,
        options_data: pd.DataFrame,
        strike: float,
        option_type: str = 'call'
    ) -> List[Dict]:
        """
        Check calendar spread no-arbitrage condition.
        
        For same strike, longer-dated options should be worth more.
        
        Args:
            options_data: DataFrame with options data
            strike: Strike price to check
            option_type: 'call' or 'put'
            
        Returns:
            List of violation dictionaries
        """
        violations = []
        
        # Filter for specific strike and option type
        subset = options_data[
            (np.abs(options_data['strike'] - strike) < 0.01) &
            (options_data['option_type'] == option_type)
        ].copy()
        
        if len(subset) < 2:
            return violations
        
        # Sort by time to expiry
        subset = subset.sort_values('time_to_expiry')
        
        # Check calendar spread condition
        for i in range(len(subset) - 1):
            T1 = subset.iloc[i]['time_to_expiry']
            T2 = subset.iloc[i+1]['time_to_expiry']
            P1 = subset.iloc[i]['mid_price']
            P2 = subset.iloc[i+1]['mid_price']
            
            if P2 < P1 - self.tolerance:
                violations.append({
                    'type': 'calendar_spread',
                    'option_type': option_type,
                    'strike': strike,
                    'maturity1': T1,
                    'maturity2': T2,
                    'price1': P1,
                    'price2': P2,
                    'severity': P1 - P2
                })
        
        return violations
    
    def check_put_call_parity(
        self,
        calls: pd.DataFrame,
        puts: pd.DataFrame,
        spot_price: float,
        risk_free_rate: float = 0.05
    ) -> List[Dict]:
        """
        Check put-call parity violations.
        
        Put-Call Parity: C - P = S - K*exp(-rT)
        
        Args:
            calls: DataFrame with call options
            puts: DataFrame with put options
            spot_price: Current spot price
            risk_free_rate: Risk-free interest rate
            
        Returns:
            List of violation dictionaries
        """
        violations = []
        
        # Merge calls and puts on strike and expiration
        merged = pd.merge(
            calls,
            puts,
            on=['strike', 'expiration', 'time_to_expiry'],
            suffixes=('_call', '_put')
        )
        
        for _, row in merged.iterrows():
            K = row['strike']
            T = row['time_to_expiry']
            C = row['mid_price_call']
            P = row['mid_price_put']
            
            # Put-call parity
            pcp_lhs = C - P
            pcp_rhs = spot_price - K * np.exp(-risk_free_rate * T)
            
            violation_amount = abs(pcp_lhs - pcp_rhs)
            
            # Check if violation exceeds tolerance
            if violation_amount > self.tolerance:
                violations.append({
                    'type': 'put_call_parity',
                    'strike': K,
                    'maturity': T,
                    'call_price': C,
                    'put_price': P,
                    'expected_diff': pcp_rhs,
                    'actual_diff': pcp_lhs,
                    'severity': violation_amount
                })
        
        return violations
    
    def check_price_bounds(
        self,
        options_data: pd.DataFrame,
        spot_price: float,
        risk_free_rate: float = 0.05
    ) -> List[Dict]:
        """
        Check if option prices respect theoretical bounds.
        
        For calls: C >= max(S - K*exp(-rT), 0)
        For puts: P >= max(K*exp(-rT) - S, 0)
        
        Args:
            options_data: DataFrame with options data
            spot_price: Current spot price
            risk_free_rate: Risk-free interest rate
            
        Returns:
            List of violation dictionaries
        """
        violations = []
        
        for _, row in options_data.iterrows():
            K = row['strike']
            T = row['time_to_expiry']
            price = row['mid_price']
            option_type = row['option_type']
            
            if option_type == 'call':
                lower_bound = max(spot_price - K * np.exp(-risk_free_rate * T), 0)
                if price < lower_bound - self.tolerance:
                    violations.append({
                        'type': 'price_bounds',
                        'option_type': 'call',
                        'strike': K,
                        'maturity': T,
                        'price': price,
                        'lower_bound': lower_bound,
                        'severity': lower_bound - price
                    })
            else:  # put
                lower_bound = max(K * np.exp(-risk_free_rate * T) - spot_price, 0)
                if price < lower_bound - self.tolerance:
                    violations.append({
                        'type': 'price_bounds',
                        'option_type': 'put',
                        'strike': K,
                        'maturity': T,
                        'price': price,
                        'lower_bound': lower_bound,
                        'severity': lower_bound - price
                    })
        
        return violations
    
    def run_all_checks(
        self,
        options_data: pd.DataFrame,
        spot_price: float,
        risk_free_rate: float = 0.05
    ) -> Dict[str, List[Dict]]:
        """
        Run all arbitrage checks on options data.
        
        Args:
            options_data: DataFrame with complete options chain
            spot_price: Current spot price
            risk_free_rate: Risk-free interest rate
            
        Returns:
            Dictionary mapping check type to list of violations
        """
        all_violations = {}
        
        # Separate calls and puts
        calls = options_data[options_data['option_type'] == 'call']
        puts = options_data[options_data['option_type'] == 'put']
        
        # Check monotonicity
        if len(calls) > 1:
            call_mono = self.check_monotonicity(
                calls['strike'].values,
                calls['mid_price'].values,
                'call'
            )
            all_violations['call_monotonicity'] = call_mono
        
        if len(puts) > 1:
            put_mono = self.check_monotonicity(
                puts['strike'].values,
                puts['mid_price'].values,
                'put'
            )
            all_violations['put_monotonicity'] = put_mono
        
        # Check convexity
        if len(calls) > 2:
            call_conv = self.check_convexity(
                calls['strike'].values,
                calls['mid_price'].values,
                'call'
            )
            all_violations['call_convexity'] = call_conv
        
        if len(puts) > 2:
            put_conv = self.check_convexity(
                puts['strike'].values,
                puts['mid_price'].values,
                'put'
            )
            all_violations['put_convexity'] = put_conv
        
        # Check put-call parity
        pcp_violations = self.check_put_call_parity(calls, puts, spot_price, risk_free_rate)
        all_violations['put_call_parity'] = pcp_violations
        
        # Check price bounds
        bound_violations = self.check_price_bounds(options_data, spot_price, risk_free_rate)
        all_violations['price_bounds'] = bound_violations
        
        return all_violations
    
    def summarize_violations(self, violations: Dict[str, List[Dict]]) -> pd.DataFrame:
        """
        Create summary DataFrame of all violations.
        
        Args:
            violations: Dictionary of violations from run_all_checks
            
        Returns:
            DataFrame summarizing violations
        """
        summary_data = []
        
        for check_type, violation_list in violations.items():
            if violation_list:
                summary_data.append({
                    'check_type': check_type,
                    'num_violations': len(violation_list),
                    'avg_severity': np.mean([v['severity'] for v in violation_list]),
                    'max_severity': max([v['severity'] for v in violation_list])
                })
        
        if not summary_data:
            return pd.DataFrame(columns=['check_type', 'num_violations', 'avg_severity', 'max_severity'])
        
        return pd.DataFrame(summary_data)


# Example usage
if __name__ == "__main__":
    # Create sample options data
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'strike': [95, 100, 105, 95, 100, 105],
        'mid_price': [8.0, 5.0, 3.0, 2.5, 4.0, 6.5],
        'option_type': ['call', 'call', 'call', 'put', 'put', 'put'],
        'time_to_expiry': [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        'expiration': ['2025-06-01'] * 6
    })
    
    # Initialize detector
    detector = ArbitrageDetector(tolerance=0.01)
    
    # Run all checks
    violations = detector.run_all_checks(
        options_data=sample_data,
        spot_price=100,
        risk_free_rate=0.05
    )
    
    # Print summary
    summary = detector.summarize_violations(violations)
    print("Arbitrage Violations Summary:")
    print(summary)
    
    # Print detailed violations
    for check_type, violation_list in violations.items():
        if violation_list:
            print(f"\n{check_type.upper()}:")
            for v in violation_list:
                print(f"  {v}")
