"""
Volatility Surface Explorer Dashboard
Main Streamlit application for interactive volatility surface analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from data.fetcher import OptionDataFetcher
from models.black_scholes import BlackScholes
from uncertainty.conformal import ConformalPredictor, VolatilitySurfaceUncertainty, BootstrapGreeks
from visualization.surface_plot import VolatilitySurfacePlotter
from arbitrage.detector import ArbitrageDetector

# Force reload modules on every run
if 'modules_reloaded' not in st.session_state:
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.modules_reloaded = True

# Page configuration
st.set_page_config(
    page_title="Volatility Surface Explorer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">📈 Volatility Surface Explorer with Uncertainty Quantification</div>', 
            unsafe_allow_html=True)
st.markdown("### *Interactive Options Analytics Dashboard*")
st.markdown("---")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Ticker selection
ticker = st.sidebar.text_input("Ticker Symbol", value="SPY", help="Enter stock ticker (e.g., SPY, QQQ, AAPL)")

# Risk-free rate
risk_free_rate = st.sidebar.slider(
    "Risk-Free Rate (%)",
    min_value=0.0,
    max_value=10.0,
    value=5.0,
    step=0.1,
    help="Annual risk-free interest rate"
) / 100

# Confidence level for uncertainty quantification
confidence_level = st.sidebar.slider(
    "Confidence Level (%)",
    min_value=80,
    max_value=99,
    value=90,
    step=1,
    help="Confidence level for prediction intervals"
) / 100

# Option type
option_type = st.sidebar.selectbox(
    "Option Type",
    options=["call", "put"],
    help="Select call or put options"
)

# Maturity filters
st.sidebar.subheader("Maturity Filters")
min_days = st.sidebar.number_input("Min Days to Expiry", value=7, min_value=1, step=1)
max_days = st.sidebar.number_input("Max Days to Expiry", value=365, min_value=1, step=1)

# Fetch data button
fetch_button = st.sidebar.button("🔄 Fetch Data", use_container_width=True)

# Initialize session state
if 'data_fetched' not in st.session_state:
    st.session_state.data_fetched = False
    st.session_state.options_data = None
    st.session_state.spot_price = None

# Main content
if fetch_button or st.session_state.data_fetched:
    try:
        with st.spinner(f'Fetching options data for {ticker}...'):
            # Initialize fetcher
            fetcher = OptionDataFetcher(ticker=ticker, risk_free_rate=risk_free_rate)
            
            # Fetch underlying data
            underlying_data = fetcher.fetch_underlying_data()
            spot_price = underlying_data['price']
            
            # Fetch options chain
            options_data = fetcher.get_option_chain()
            
            # Store in session state
            st.session_state.data_fetched = True
            st.session_state.options_data = options_data
            st.session_state.spot_price = spot_price
            st.session_state.underlying_data = underlying_data
        
        st.success(f"✅ Successfully fetched {len(options_data)} options contracts for {ticker}")
        
        # Display underlying info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Spot Price", f"${spot_price:.2f}")
        with col2:
            st.metric("Bid", f"${underlying_data.get('bid', 0):.2f}")
        with col3:
            st.metric("Ask", f"${underlying_data.get('ask', 0):.2f}")
        with col4:
            st.metric("Volume", f"{underlying_data.get('volume', 0):,}")
        
        st.markdown("---")
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Volatility Surface", 
            "🎯 Greeks Analysis", 
            "⚠️ Arbitrage Detection",
            "📈 Term Structure",
            "📋 Data Explorer"
        ])
        
        # Tab 1: Volatility Surface
        with tab1:
            st.markdown('<div class="sub-header">3D Implied Volatility Surface</div>', 
                       unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                try:
                    # Get surface grid data
                    strikes, maturities, ivs = fetcher.get_surface_grid(
                        option_type=option_type,
                        min_maturity_days=min_days,
                        max_maturity_days=max_days
                    )
                    
                    # Add uncertainty quantification
                    use_uncertainty = st.checkbox("Show Uncertainty Bands", value=True)
                    
                    if use_uncertainty:
                        with st.spinner("Computing uncertainty bands..."):
                            uncertainty_calc = VolatilitySurfaceUncertainty(
                                alpha=1-confidence_level
                            )
                            lower_surface, upper_surface = uncertainty_calc.compute_surface_uncertainty(
                                strikes, maturities, ivs, bootstrap_samples=100
                            )
                    else:
                        lower_surface = None
                        upper_surface = None
                    
                    # Create visualization
                    plotter = VolatilitySurfacePlotter(theme='plotly_dark')
                    surface_fig = plotter.create_3d_surface(
                        strikes=strikes,
                        maturities=maturities,
                        ivs=ivs,
                        lower_surface=lower_surface,
                        upper_surface=upper_surface,
                        title=f"{ticker} {option_type.capitalize()} Implied Volatility Surface"
                    )
                    
                    st.plotly_chart(surface_fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error creating volatility surface: {str(e)}")
                    st.info("Try adjusting maturity filters or selecting a different ticker.")
            
            with col2:
                st.markdown("#### Surface Statistics")
                if len(strikes) > 0:
                    st.metric("Data Points", len(strikes))
                    st.metric("Avg IV", f"{np.mean(ivs)*100:.2f}%")
                    st.metric("IV Range", f"{np.min(ivs)*100:.2f}% - {np.max(ivs)*100:.2f}%")
                    st.metric("Std Dev", f"{np.std(ivs)*100:.2f}%")
                    
                    if use_uncertainty:
                        st.markdown("#### Uncertainty Stats")
                        avg_width = np.mean(upper_surface - lower_surface) * 100
                        st.metric("Avg Band Width", f"{avg_width:.2f}%")
            
            # Volatility smile
            st.markdown("#### Volatility Smile Analysis")
            
            # Select maturity for smile
            available_maturities = sorted(options_data['time_to_expiry'].unique())
            selected_maturity_idx = st.select_slider(
                "Select Maturity",
                options=range(len(available_maturities)),
                format_func=lambda x: f"{available_maturities[x]:.2f} years"
            )
            selected_maturity = available_maturities[selected_maturity_idx]
            
            # Filter data for selected maturity
            maturity_tolerance = 0.05
            maturity_data = options_data[
                (np.abs(options_data['time_to_expiry'] - selected_maturity) < maturity_tolerance) &
                (options_data['option_type'] == option_type)
            ].copy()
            
            if len(maturity_data) > 3:
                smile_fig = plotter.create_volatility_smile(
                    strikes=maturity_data['strike'].values,
                    ivs=maturity_data['impliedVolatility'].values,
                    maturity=selected_maturity,
                    spot_price=spot_price
                )
                st.plotly_chart(smile_fig, use_container_width=True)
            else:
                st.warning("Insufficient data for volatility smile at selected maturity")
        
        # Tab 2: Greeks Analysis
        with tab2:
            st.markdown('<div class="sub-header">Greeks with Uncertainty Quantification</div>', 
                       unsafe_allow_html=True)
            
            # Initialize Black-Scholes model
            bs = BlackScholes(risk_free_rate=risk_free_rate)
            bootstrap_greeks = BootstrapGreeks(n_bootstrap=500, confidence_level=confidence_level)
            
            # Select strike and maturity for Greeks analysis
            col1, col2 = st.columns(2)
            with col1:
                strike_for_greeks = st.number_input(
                    "Strike Price",
                    min_value=float(spot_price * 0.5),
                    max_value=float(spot_price * 1.5),
                    value=float(spot_price),
                    step=1.0
                )
            with col2:
                maturity_for_greeks = st.number_input(
                    "Time to Maturity (years)",
                    min_value=0.01,
                    max_value=2.0,
                    value=0.5,
                    step=0.05
                )
            
            sigma_for_greeks = st.slider(
                "Volatility (for Greeks)",
                min_value=0.05,
                max_value=1.0,
                value=0.25,
                step=0.01
            )
            
            # Calculate all Greeks with uncertainty
            greeks_results = {}
            greek_names = ['delta', 'gamma', 'vega', 'theta', 'rho']
            
            with st.spinner("Calculating Greeks with uncertainty bands..."):
                for greek_name in greek_names:
                    point, lower, upper = bootstrap_greeks.compute_greek_intervals(
                        greek_calculator=bs,
                        S=spot_price,
                        K=strike_for_greeks,
                        T=maturity_for_greeks,
                        sigma=sigma_for_greeks,
                        greek_name=greek_name,
                        option_type=option_type
                    )
                    greeks_results[greek_name] = {
                        'point': point,
                        'lower': lower,
                        'upper': upper
                    }
            
            # Display Greeks
            col1, col2, col3, col4, col5 = st.columns(5)
            
            for idx, (greek_name, col) in enumerate(zip(greek_names, [col1, col2, col3, col4, col5])):
                with col:
                    result = greeks_results[greek_name]
                    st.metric(
                        greek_name.capitalize(),
                        f"{result['point']:.4f}",
                        delta=None
                    )
                    st.caption(f"[{result['lower']:.4f}, {result['upper']:.4f}]")
            
            st.markdown("---")
            
            # Greeks surface visualization
            st.markdown("#### Greeks Heatmaps")
            
            selected_greek = st.selectbox(
                "Select Greek to Visualize",
                options=['delta', 'gamma', 'vega']
            )
            
            with st.spinner(f"Computing {selected_greek} surface..."):
                # Compute Greek across surface
                greek_values = []
                
                for strike, maturity, iv in zip(strikes, maturities, ivs):
                    if selected_greek == 'delta':
                        val = bs.delta(spot_price, strike, maturity, iv, option_type)
                    elif selected_greek == 'gamma':
                        val = bs.gamma(spot_price, strike, maturity, iv)
                    else:  # vega
                        val = bs.vega(spot_price, strike, maturity, iv)
                    greek_values.append(val)
                
                greek_values = np.array(greek_values)
                
                # Create heatmap
                heatmap_fig = plotter.create_greeks_heatmap(
                    strikes, maturities, greek_values, selected_greek
                )
                st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # Tab 3: Arbitrage Detection
        with tab3:
            st.markdown('<div class="sub-header">No-Arbitrage Condition Checks</div>', 
                       unsafe_allow_html=True)
            
            tolerance = st.slider(
                "Violation Tolerance",
                min_value=0.00,
                max_value=0.10,
                value=0.01,
                step=0.01,
                help="Minimum threshold for flagging violations"
            )
            
            detector = ArbitrageDetector(tolerance=tolerance)
            
            with st.spinner("Running arbitrage checks..."):
                violations = detector.run_all_checks(
                    options_data=options_data,
                    spot_price=spot_price,
                    risk_free_rate=risk_free_rate
                )
                
                summary = detector.summarize_violations(violations)
            
            if len(summary) == 0:
                st.success("✅ No arbitrage violations detected!")
            else:
                st.warning(f"⚠️ Found {summary['num_violations'].sum()} total violations")
                
                # Display summary
                st.dataframe(summary, use_container_width=True)
                
                # Display detailed violations
                for check_type, violation_list in violations.items():
                    if violation_list:
                        with st.expander(f"{check_type.replace('_', ' ').title()} ({len(violation_list)} violations)"):
                            st.json(violation_list[:5])  # Show first 5
        
        # Tab 4: Term Structure
        with tab4:
            st.markdown('<div class="sub-header">ATM Volatility Term Structure</div>', 
                       unsafe_allow_html=True)
            
            try:
                term_struct = fetcher.get_term_structure(option_type=option_type)
                
                if len(term_struct) > 0:
                    term_fig = plotter.create_term_structure(
                        maturities=term_struct['time_to_expiry'].values,
                        atm_vols=term_struct['impliedVolatility'].values
                    )
                    st.plotly_chart(term_fig, use_container_width=True)
                    
                    # Display term structure data
                    st.dataframe(
                        term_struct[['expiration', 'time_to_expiry', 'impliedVolatility']]
                        .rename(columns={
                            'expiration': 'Expiration',
                            'time_to_expiry': 'Years to Maturity',
                            'impliedVolatility': 'ATM IV'
                        }),
                        use_container_width=True
                    )
                else:
                    st.warning("Insufficient ATM data for term structure")
            except Exception as e:
                st.error(f"Error creating term structure: {str(e)}")
        
        # Tab 5: Data Explorer
        with tab5:
            st.markdown('<div class="sub-header">Raw Options Data</div>', 
                       unsafe_allow_html=True)
            
            # Summary statistics
            stats = fetcher.compute_summary_statistics()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Options", stats['total_options'])
            with col2:
                st.metric("Calls", stats['num_calls'])
            with col3:
                st.metric("Puts", stats['num_puts'])
            with col4:
                st.metric("Expiries", stats['num_expiries'])
            
            # Filter options
            st.markdown("#### Filter Options")
            
            col1, col2 = st.columns(2)
            with col1:
                filter_option_type = st.multiselect(
                    "Option Type",
                    options=['call', 'put'],
                    default=['call', 'put']
                )
            with col2:
                min_volume = st.number_input("Min Volume", value=0, min_value=0, step=10)
            
            filtered_data = options_data[
                (options_data['option_type'].isin(filter_option_type)) &
                (options_data['volume'] >= min_volume)
            ].copy()
            
            # Display data
            display_columns = [
                'strike', 'expiration', 'option_type', 'mid_price', 
                'impliedVolatility', 'volume', 'openInterest', 'delta'
            ]
            
            # Only show columns that exist
            display_columns = [col for col in display_columns if col in filtered_data.columns]
            
            st.dataframe(
                filtered_data[display_columns].sort_values('expiration'),
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=f"{ticker}_options_data.csv",
                mime="text/csv"
            )
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Please check the ticker symbol and try again.")
        st.session_state.data_fetched = False

else:
    # Welcome screen
    st.info("👈 Configure parameters in the sidebar and click 'Fetch Data' to begin analysis")
    
    st.markdown("""
    ### Features:
    
    - **3D Volatility Surface**: Interactive visualization with uncertainty bands
    - **Greeks Analysis**: Calculate Delta, Gamma, Vega, Theta, Rho with confidence intervals
    - **Arbitrage Detection**: Check for violations of no-arbitrage conditions
    - **Term Structure**: Analyze ATM volatility across maturities
    - **Data Explorer**: Browse and download raw options data
    
    ### Methods:
    
    - **Conformal Prediction**: Distribution-free prediction intervals
    - **Bootstrap Resampling**: Uncertainty quantification for Greeks
    - **Black-Scholes Model**: Standard options pricing framework
    
    ### Getting Started:
    
    1. Enter a ticker symbol (e.g., SPY, QQQ, AAPL)
    2. Set risk-free rate and confidence level
    3. Click "Fetch Data" to load options chain
    4. Explore different tabs for various analyses
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Volatility Surface Explorer | Built with Python, Streamlit, and Plotly</p>
    <p>Data Source: Yahoo Finance | Educational and Research Purposes Only</p>
</div>
""", unsafe_allow_html=True)
