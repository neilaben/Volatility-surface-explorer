"""
Conformal Prediction for Uncertainty Quantification
Implements distribution-free prediction intervals for implied volatility.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split


class ConformalPredictor:
    """
    Conformal prediction for implied volatility uncertainty quantification.
    
    Provides distribution-free, finite-sample valid prediction intervals.
    
    References:
        - Romano et al. (2019): "Conformalized Quantile Regression"
        - Vovk et al. (2005): "Algorithmic Learning in a Random World"
    """
    
    def __init__(self, alpha: float = 0.1, method: str = 'absolute'):
        """
        Initialize conformal predictor.
        
        Args:
            alpha: Miscoverage rate (default: 0.1 for 90% coverage)
            method: Conformity score method ('absolute', 'normalized', 'cqr')
        """
        self.alpha = alpha
        self.method = method
        self.calibration_scores = None
        self.quantile = None
        
    def calibrate(
        self, 
        predictions: np.ndarray,
        actuals: np.ndarray,
        lower_predictions: Optional[np.ndarray] = None,
        upper_predictions: Optional[np.ndarray] = None
    ):
        """
        Calibrate conformal predictor on calibration set.
        
        Args:
            predictions: Point predictions on calibration set
            actuals: Actual values on calibration set
            lower_predictions: Lower quantile predictions (for CQR)
            upper_predictions: Upper quantile predictions (for CQR)
        """
        if self.method == 'absolute':
            # Absolute conformity score
            self.calibration_scores = np.abs(predictions - actuals)
            
        elif self.method == 'normalized':
            # Normalized by prediction magnitude
            self.calibration_scores = np.abs(
                (predictions - actuals) / (predictions + 1e-10)
            )
            
        elif self.method == 'cqr':
            # Conformalized Quantile Regression
            if lower_predictions is None or upper_predictions is None:
                raise ValueError("CQR requires lower and upper predictions")
            
            # CQR conformity score
            self.calibration_scores = np.maximum(
                lower_predictions - actuals,
                actuals - upper_predictions
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Compute quantile for prediction intervals
        n = len(self.calibration_scores)
        q = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile = np.quantile(self.calibration_scores, q)
    
    def predict(
        self,
        predictions: np.ndarray,
        lower_predictions: Optional[np.ndarray] = None,
        upper_predictions: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate prediction intervals for new predictions.
        
        Args:
            predictions: Point predictions for test set
            lower_predictions: Lower quantile predictions (for CQR)
            upper_predictions: Upper quantile predictions (for CQR)
            
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        if self.quantile is None:
            raise ValueError("Must calibrate before predicting")
        
        if self.method == 'absolute':
            lower = predictions - self.quantile
            upper = predictions + self.quantile
            
        elif self.method == 'normalized':
            lower = predictions * (1 - self.quantile)
            upper = predictions * (1 + self.quantile)
            
        elif self.method == 'cqr':
            if lower_predictions is None or upper_predictions is None:
                raise ValueError("CQR requires lower and upper predictions")
            
            lower = lower_predictions - self.quantile
            upper = upper_predictions + self.quantile
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return lower, upper
    
    def get_coverage(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        lower_predictions: Optional[np.ndarray] = None,
        upper_predictions: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate empirical coverage on a test set.
        
        Args:
            predictions: Point predictions
            actuals: Actual values
            lower_predictions: Lower quantile predictions (for CQR)
            upper_predictions: Upper quantile predictions (for CQR)
            
        Returns:
            Empirical coverage rate
        """
        lower, upper = self.predict(predictions, lower_predictions, upper_predictions)
        coverage = np.mean((actuals >= lower) & (actuals <= upper))
        return coverage


class VolatilitySurfaceUncertainty:
    """
    Uncertainty quantification for volatility surfaces using conformal prediction.
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        Initialize volatility surface uncertainty quantifier.
        
        Args:
            alpha: Miscoverage rate (1 - coverage level)
        """
        self.alpha = alpha
        self.cp = ConformalPredictor(alpha=alpha, method='absolute')
        
    def fit_predict(
        self,
        options_data: pd.DataFrame,
        calibration_fraction: float = 0.3
    ) -> pd.DataFrame:
        """
        Fit conformal predictor and generate prediction intervals.
        
        Args:
            options_data: DataFrame with options and implied volatilities
            calibration_fraction: Fraction of data for calibration
            
        Returns:
            DataFrame with prediction intervals
        """
        # Filter valid data
        valid_data = options_data[
            options_data['impliedVolatility'].notna() &
            (options_data['impliedVolatility'] > 0)
        ].copy()
        
        if len(valid_data) < 50:
            raise ValueError("Insufficient data for uncertainty quantification")
        
        # Split into calibration and test sets
        calib_data, test_data = train_test_split(
            valid_data, 
            test_size=1-calibration_fraction,
            random_state=42
        )
        
        # Use market IV as "prediction" and create synthetic residuals
        # In practice, you'd use a fitted model's predictions
        calib_predictions = calib_data['impliedVolatility'].values
        
        # Add small random noise to create calibration actuals
        # (simulating model residuals)
        noise_scale = 0.02  # 2% noise
        calib_actuals = calib_predictions * (1 + np.random.randn(len(calib_predictions)) * noise_scale)
        
        # Calibrate conformal predictor
        self.cp.calibrate(calib_predictions, calib_actuals)
        
        # Generate prediction intervals for test set
        test_predictions = test_data['impliedVolatility'].values
        lower, upper = self.cp.predict(test_predictions)
        
        # Add to test data
        test_data = test_data.copy()
        test_data['iv_lower'] = lower
        test_data['iv_upper'] = upper
        test_data['iv_width'] = upper - lower
        
        return test_data
    
    def compute_surface_uncertainty(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        ivs: np.ndarray,
        bootstrap_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute uncertainty bands for volatility surface using bootstrap.
        
        Args:
            strikes: Array of strike prices
            maturities: Array of times to maturity
            ivs: Array of implied volatilities
            bootstrap_samples: Number of bootstrap samples
            
        Returns:
            Tuple of (lower_surface, upper_surface)
        """
        n = len(strikes)
        
        # Store bootstrap surfaces
        bootstrap_ivs = np.zeros((bootstrap_samples, n))
        
        for i in range(bootstrap_samples):
            # Resample with replacement
            indices = np.random.choice(n, size=n, replace=True)
            
            bootstrap_ivs[i] = ivs[indices]
        
        # Compute quantiles
        lower_quantile = self.alpha / 2
        upper_quantile = 1 - self.alpha / 2
        
        lower_surface = np.quantile(bootstrap_ivs, lower_quantile, axis=0)
        upper_surface = np.quantile(bootstrap_ivs, upper_quantile, axis=0)
        
        return lower_surface, upper_surface


class BootstrapGreeks:
    """
    Bootstrap-based uncertainty quantification for Greeks.
    """
    
    def __init__(self, n_bootstrap: int = 1000, confidence_level: float = 0.90):
        """
        Initialize bootstrap Greeks calculator.
        
        Args:
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        
    def compute_greek_intervals(
        self,
        greek_calculator,
        S: float,
        K: float,
        T: float,
        sigma: float,
        greek_name: str,
        option_type: str = 'call',
        param_uncertainty: Dict[str, float] = None
    ) -> Tuple[float, float, float]:
        """
        Compute confidence intervals for a Greek using bootstrap.
        
        Args:
            greek_calculator: Object with Greek calculation methods (e.g., BlackScholes)
            S: Spot price
            K: Strike price
            T: Time to maturity
            sigma: Volatility
            greek_name: Name of Greek ('delta', 'gamma', etc.)
            option_type: 'call' or 'put'
            param_uncertainty: Dict with parameter uncertainties (std devs)
            
        Returns:
            Tuple of (point_estimate, lower_bound, upper_bound)
        """
        if param_uncertainty is None:
            # Default parameter uncertainties
            param_uncertainty = {
                'S': S * 0.001,      # 0.1% price uncertainty
                'sigma': sigma * 0.05  # 5% vol uncertainty
            }
        
        # Get Greek calculation method
        greek_method = getattr(greek_calculator, greek_name)
        
        # Point estimate
        if greek_name in ['gamma', 'vega']:
            # These don't depend on option_type
            point_estimate = greek_method(S, K, T, sigma)
        else:
            point_estimate = greek_method(S, K, T, sigma, option_type)
        
        # Bootstrap samples
        bootstrap_greeks = []
        
        for _ in range(self.n_bootstrap):
            # Perturb parameters
            S_boot = S + np.random.randn() * param_uncertainty.get('S', 0)
            sigma_boot = sigma + np.random.randn() * param_uncertainty.get('sigma', 0)
            
            # Ensure positive values
            S_boot = max(S_boot, 0.01)
            sigma_boot = max(sigma_boot, 0.001)
            
            # Calculate Greek with perturbed parameters
            if greek_name in ['gamma', 'vega']:
                greek_val = greek_method(S_boot, K, T, sigma_boot)
            else:
                greek_val = greek_method(S_boot, K, T, sigma_boot, option_type)
            
            bootstrap_greeks.append(greek_val)
        
        # Compute confidence intervals
        alpha = 1 - self.confidence_level
        lower = np.percentile(bootstrap_greeks, 100 * alpha / 2)
        upper = np.percentile(bootstrap_greeks, 100 * (1 - alpha / 2))
        
        return point_estimate, lower, upper


# Example usage
if __name__ == "__main__":
    # Example: Conformal prediction for IV
    np.random.seed(42)
    
    # Simulate some IV data
    n_samples = 200
    true_iv = 0.25
    predictions = np.random.normal(true_iv, 0.02, n_samples)
    actuals = predictions + np.random.normal(0, 0.01, n_samples)
    
    # Split into calibration and test
    n_calib = 100
    calib_pred = predictions[:n_calib]
    calib_actual = actuals[:n_calib]
    test_pred = predictions[n_calib:]
    test_actual = actuals[n_calib:]
    
    # Initialize and calibrate
    cp = ConformalPredictor(alpha=0.1)
    cp.calibrate(calib_pred, calib_actual)
    
    # Generate prediction intervals
    lower, upper = cp.predict(test_pred)
    
    # Check coverage
    coverage = cp.get_coverage(test_pred, test_actual)
    
    print(f"Expected coverage: {1-cp.alpha:.1%}")
    print(f"Empirical coverage: {coverage:.1%}")
    print(f"Average interval width: {np.mean(upper - lower):.4f}")
    
    # Example: Bootstrap Greeks
    from models.black_scholes import BlackScholes
    
    bs = BlackScholes(risk_free_rate=0.05)
    bootstrap = BootstrapGreeks(n_bootstrap=1000)
    
    point, lower, upper = bootstrap.compute_greek_intervals(
        greek_calculator=bs,
        S=100,
        K=100,
        T=0.5,
        sigma=0.25,
        greek_name='delta',
        option_type='call'
    )
    
    print(f"\nDelta: {point:.4f} [{lower:.4f}, {upper:.4f}]")
