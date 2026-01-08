"""
Volatility Surface Visualization
Creates interactive 3D plots and heatmaps for volatility analysis.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from scipy.interpolate import griddata, RBFInterpolator
from typing import Optional, Tuple, List


class VolatilitySurfacePlotter:
    """
    Interactive 3D visualization for volatility surfaces.
    """
    
    def __init__(self, theme: str = 'plotly_dark'):
        """
        Initialize surface plotter.
        
        Args:
            theme: Plotly theme ('plotly', 'plotly_dark', 'plotly_white')
        """
        self.theme = theme
        
    def create_3d_surface(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        ivs: np.ndarray,
        lower_surface: Optional[np.ndarray] = None,
        upper_surface: Optional[np.ndarray] = None,
        title: str = "Implied Volatility Surface",
        colorscale: str = 'Viridis'
    ) -> go.Figure:
        """
        Create interactive 3D volatility surface.
        
        Args:
            strikes: Strike prices
            maturities: Times to maturity (years)
            ivs: Implied volatilities
            lower_surface: Lower uncertainty band (optional)
            upper_surface: Upper uncertainty band (optional)
            title: Plot title
            colorscale: Plotly colorscale name
            
        Returns:
            Plotly Figure object
        """
        # Create meshgrid for surface
        strike_grid = np.linspace(strikes.min(), strikes.max(), 50)
        maturity_grid = np.linspace(maturities.min(), maturities.max(), 50)
        strike_mesh, maturity_mesh = np.meshgrid(strike_grid, maturity_grid)
        
        # Interpolate IV on grid
        points = np.column_stack([strikes, maturities])
        iv_mesh = griddata(
            points, 
            ivs, 
            (strike_mesh, maturity_mesh), 
            method='cubic'
        )
        
        # Create figure
        fig = go.Figure()
        
        # Main surface
        fig.add_trace(go.Surface(
            x=strike_mesh,
            y=maturity_mesh,
            z=iv_mesh * 100,  # Convert to percentage
            colorscale=colorscale,
            name='Implied Volatility',
            colorbar=dict(title="IV (%)", len=0.75),
            hovertemplate='Strike: %{x:.2f}<br>Maturity: %{y:.2f}yr<br>IV: %{z:.2f}%<extra></extra>'
        ))
        
        # Add scatter points for actual data
        fig.add_trace(go.Scatter3d(
            x=strikes,
            y=maturities,
            z=ivs * 100,
            mode='markers',
            marker=dict(
                size=3,
                color='red',
                symbol='circle'
            ),
            name='Market Data',
            hovertemplate='Strike: %{x:.2f}<br>Maturity: %{y:.2f}yr<br>IV: %{z:.2f}%<extra></extra>'
        ))
        
        # Add uncertainty bands if provided
        if lower_surface is not None and upper_surface is not None:
            lower_mesh = griddata(points, lower_surface, (strike_mesh, maturity_mesh), method='cubic')
            upper_mesh = griddata(points, upper_surface, (strike_mesh, maturity_mesh), method='cubic')
            
            fig.add_trace(go.Surface(
                x=strike_mesh,
                y=maturity_mesh,
                z=lower_mesh * 100,
                opacity=0.3,
                colorscale='Greys',
                showscale=False,
                name='Lower Band',
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Surface(
                x=strike_mesh,
                y=maturity_mesh,
                z=upper_mesh * 100,
                opacity=0.3,
                colorscale='Greys',
                showscale=False,
                name='Upper Band',
                hoverinfo='skip'
            ))
        
        # Update layout
        fig.update_layout(
            title=title,
            template=self.theme,
            scene=dict(
                xaxis_title='Strike Price ($)',
                yaxis_title='Time to Maturity (years)',
                zaxis_title='Implied Volatility (%)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            height=700,
            showlegend=True
        )
        
        return fig
    
    def create_volatility_smile(
        self,
        strikes: np.ndarray,
        ivs: np.ndarray,
        maturity: float,
        spot_price: float,
        lower_bound: Optional[np.ndarray] = None,
        upper_bound: Optional[np.ndarray] = None
    ) -> go.Figure:
        """
        Create volatility smile plot for a specific maturity.
        
        Args:
            strikes: Strike prices
            ivs: Implied volatilities
            maturity: Time to maturity
            spot_price: Current spot price
            lower_bound: Lower uncertainty bound
            upper_bound: Upper uncertainty bound
            
        Returns:
            Plotly Figure
        """
        # Calculate moneyness
        moneyness = strikes / spot_price
        
        fig = go.Figure()
        
        # Main smile
        fig.add_trace(go.Scatter(
            x=moneyness,
            y=ivs * 100,
            mode='lines+markers',
            name='IV Smile',
            line=dict(width=2, color='blue'),
            marker=dict(size=8)
        ))
        
        # Uncertainty bands
        if lower_bound is not None and upper_bound is not None:
            fig.add_trace(go.Scatter(
                x=np.concatenate([moneyness, moneyness[::-1]]),
                y=np.concatenate([upper_bound * 100, lower_bound[::-1] * 100]),
                fill='toself',
                fillcolor='rgba(0,100,250,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='90% Confidence'
            ))
        
        # ATM line
        fig.add_vline(
            x=1.0, 
            line_dash="dash", 
            line_color="red",
            annotation_text="ATM"
        )
        
        fig.update_layout(
            title=f'Volatility Smile (T = {maturity:.2f} years)',
            xaxis_title='Moneyness (K/S)',
            yaxis_title='Implied Volatility (%)',
            template=self.theme,
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def create_term_structure(
        self,
        maturities: np.ndarray,
        atm_vols: np.ndarray,
        lower_bound: Optional[np.ndarray] = None,
        upper_bound: Optional[np.ndarray] = None
    ) -> go.Figure:
        """
        Create ATM volatility term structure plot.
        
        Args:
            maturities: Times to maturity
            atm_vols: ATM implied volatilities
            lower_bound: Lower uncertainty bound
            upper_bound: Upper uncertainty bound
            
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        # Main term structure
        fig.add_trace(go.Scatter(
            x=maturities,
            y=atm_vols * 100,
            mode='lines+markers',
            name='ATM IV',
            line=dict(width=2, color='green'),
            marker=dict(size=8)
        ))
        
        # Uncertainty bands
        if lower_bound is not None and upper_bound is not None:
            fig.add_trace(go.Scatter(
                x=np.concatenate([maturities, maturities[::-1]]),
                y=np.concatenate([upper_bound * 100, lower_bound[::-1] * 100]),
                fill='toself',
                fillcolor='rgba(0,250,100,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='90% Confidence'
            ))
        
        fig.update_layout(
            title='ATM Volatility Term Structure',
            xaxis_title='Time to Maturity (years)',
            yaxis_title='Implied Volatility (%)',
            template=self.theme,
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def create_greeks_heatmap(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        greek_values: np.ndarray,
        greek_name: str
    ) -> go.Figure:
        """
        Create heatmap for Greeks across strikes and maturities.
        
        Args:
            strikes: Strike prices
            maturities: Times to maturity
            greek_values: Greek values
            greek_name: Name of the Greek
            
        Returns:
            Plotly Figure
        """
        # Create meshgrid
        strike_grid = np.linspace(strikes.min(), strikes.max(), 50)
        maturity_grid = np.linspace(maturities.min(), maturities.max(), 50)
        strike_mesh, maturity_mesh = np.meshgrid(strike_grid, maturity_grid)
        
        # Interpolate
        points = np.column_stack([strikes, maturities])
        greek_mesh = griddata(
            points,
            greek_values,
            (strike_mesh, maturity_mesh),
            method='cubic'
        )
        
        fig = go.Figure(data=go.Heatmap(
            x=strike_grid,
            y=maturity_grid,
            z=greek_mesh,
            colorscale='RdBu',
            colorbar=dict(title=greek_name.capitalize()),
            hovertemplate='Strike: %{x:.2f}<br>Maturity: %{y:.2f}yr<br>' + 
                         f'{greek_name.capitalize()}: %{{z:.4f}}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'{greek_name.capitalize()} Heatmap',
            xaxis_title='Strike Price ($)',
            yaxis_title='Time to Maturity (years)',
            template=self.theme,
            height=600
        )
        
        return fig
    
    def create_dashboard_layout(
        self,
        main_surface_fig: go.Figure,
        smile_fig: go.Figure,
        term_structure_fig: go.Figure,
        greeks_figs: List[go.Figure]
    ) -> go.Figure:
        """
        Create comprehensive dashboard layout with multiple subplots.
        
        Args:
            main_surface_fig: 3D surface figure
            smile_fig: Volatility smile figure
            term_structure_fig: Term structure figure
            greeks_figs: List of Greeks figures
            
        Returns:
            Combined dashboard figure
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Volatility Surface', 'Volatility Smile',
                'Term Structure', 'Delta',
                'Gamma', 'Vega'
            ),
            specs=[
                [{'type': 'surface', 'rowspan': 2}, {'type': 'scatter'}],
                [None, {'type': 'scatter'}],
                [{'type': 'heatmap'}, {'type': 'heatmap'}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # This would require restructuring the individual plots
        # For now, return the main surface as the primary visualization
        # In a real dashboard, you'd use Streamlit's column layout instead
        
        return main_surface_fig


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    
    # Create sample volatility surface data
    n_points = 100
    strikes = np.random.uniform(80, 120, n_points)
    maturities = np.random.uniform(0.1, 2, n_points)
    
    # Simulate volatility smile effect
    moneyness = strikes / 100
    ivs = 0.2 + 0.1 * (moneyness - 1)**2 + 0.05 * np.sqrt(maturities)
    ivs += np.random.normal(0, 0.01, n_points)
    
    # Create plotter
    plotter = VolatilitySurfacePlotter()
    
    # Create 3D surface
    fig = plotter.create_3d_surface(strikes, maturities, ivs)
    fig.show()
    
    # Create smile for T=0.5
    mask = (maturities > 0.45) & (maturities < 0.55)
    smile_fig = plotter.create_volatility_smile(
        strikes[mask],
        ivs[mask],
        maturity=0.5,
        spot_price=100
    )
    smile_fig.show()
