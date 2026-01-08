# 📁 Project Structure

Complete file organization for Volatility Surface Explorer.

```
volatility-surface-explorer/
│
├── .streamlit/
│   ├── config.toml                    # Streamlit theme and settings
│   └── secrets.toml.example           # Template for API keys
│
├── .gitignore                         # Git ignore rules
├── README.md                          # Main documentation
├── requirements.txt                   # Python dependencies
├── setup.sh                           # Setup script
├── LICENSE                            # MIT License
│
├── docs/
│   ├── DEPLOYMENT.md                  # Streamlit Cloud deployment guide
│   ├── MATH_CONCEPTS.md              # Mathematical documentation
│   ├── INTERVIEW_GUIDE.md            # Study guide for interviews
│   └── API_REFERENCE.md              # Code API documentation
│
├── data/
│   └── universes/                     # Saved ticker configurations
│       ├── .gitkeep
│       └── default_broad.json         # Preset universe
│
├── src/
│   └── volatility_explorer/
│       │
│       ├── __init__.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   └── fetcher.py            # Options data fetching (yfinance)
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   └── black_scholes.py      # Black-Scholes pricing & Greeks
│       │
│       ├── uncertainty/
│       │   ├── __init__.py
│       │   └── conformal.py          # Conformal prediction (PhD-level UQ)
│       │
│       ├── arbitrage/
│       │   ├── __init__.py
│       │   └── detector.py           # Arbitrage detection (5 checks)
│       │
│       ├── visualization/
│       │   ├── __init__.py
│       │   └── surface_plot.py       # 3D volatility surface plots
│       │
│       ├── strategies/
│       │   ├── __init__.py
│       │   ├── ticker_discovery.py           # Auto-discover liquid tickers
│       │   ├── ticker_validator.py           # Validate user input
│       │   ├── sector_classifier.py          # Auto/manual sector grouping
│       │   ├── universe_manager.py           # Save/load ticker configs
│       │   ├── multi_market_portfolio.py     # Portfolio construction
│       │   ├── anti_hft_multi_factor.py      # 6-factor scoring
│       │   └── complete_portfolio_system.py  # Orchestrator
│       │
│       └── dashboard/
│           ├── __init__.py
│           ├── app.py                        # Main Streamlit app
│           │
│           ├── components/
│           │   ├── __init__.py
│           │   ├── portfolio_tab.py          # Portfolio Builder UI
│           │   └── multifactor_tab.py        # Multi-Factor Analysis UI
│           │
│           └── utils/
│               ├── __init__.py
│               └── session_state.py          # Streamlit state management
│
└── tests/
    ├── __init__.py
    ├── test_black_scholes.py
    ├── test_conformal.py
    ├── test_portfolio.py
    └── test_integration.py
```

---

## 📝 File Descriptions

### Configuration Files

- **`.streamlit/config.toml`**: Theme colors, server settings, browser preferences
- **`.streamlit/secrets.toml.example`**: Template for API keys (copy to secrets.toml)
- **`requirements.txt`**: All Python dependencies with pinned versions
- **`.gitignore`**: Prevents committing secrets, cache, etc.

### Documentation

- **`README.md`**: Main project documentation, features, usage
- **`docs/DEPLOYMENT.md`**: Step-by-step Streamlit Cloud deployment
- **`docs/MATH_CONCEPTS.md`**: Complete mathematical reference
- **`docs/INTERVIEW_GUIDE.md`**: Study guide for quant interviews

### Core Modules

#### Data Layer
- **`fetcher.py`**: Fetches options data from Yahoo Finance, handles caching

#### Models
- **`black_scholes.py`**: Options pricing, Greeks calculation, implied volatility

#### Uncertainty Quantification
- **`conformal.py`**: Conformal prediction, bootstrap methods, confidence intervals

#### Arbitrage
- **`detector.py`**: Put-call parity, monotonicity, convexity, calendar spreads, bounds

#### Visualization
- **`surface_plot.py`**: 3D Plotly surfaces, heatmaps, interactive charts

#### Strategies (NEW)
- **`ticker_discovery.py`**: Auto-scan S&P 500, filter by liquidity
- **`ticker_validator.py`**: Validate user input, check options availability
- **`sector_classifier.py`**: Auto-detect or manually assign sectors
- **`universe_manager.py`**: Save/load custom ticker lists (JSON)
- **`multi_market_portfolio.py`**: Correlation analysis, diversified selection
- **`anti_hft_multi_factor.py`**: 6-factor scoring system
- **`complete_portfolio_system.py`**: Ties everything together

#### Dashboard
- **`app.py`**: Main Streamlit application
- **`components/portfolio_tab.py`**: Portfolio Builder UI
- **`components/multifactor_tab.py`**: Multi-Factor Analysis UI
- **`utils/session_state.py`**: Manages Streamlit session state

### Tests
- Unit tests for each module
- Integration tests for end-to-end workflows

---

## 🔄 Data Flow

```
User Input (Dashboard)
        ↓
Ticker Discovery/Validation
        ↓
Options Data Fetching (yfinance)
        ↓
Black-Scholes Pricing
        ↓
Uncertainty Quantification
        ↓
Arbitrage Detection
        ↓
Multi-Factor Scoring
        ↓
Portfolio Construction
        ↓
Visualization
        ↓
Results Display
```

---

## 🚀 Quick Navigation

**To add new features:**
- Add Python modules to `src/volatility_explorer/`
- Create new dashboard tabs in `dashboard/components/`
- Add dependencies to `requirements.txt`

**To deploy:**
1. Push to GitHub
2. Connect to Streamlit Cloud
3. Point to `src/volatility_explorer/dashboard/app.py`

**To test:**
```bash
pytest tests/
```

---

**Next Steps**: See [DEPLOYMENT.md](../docs/DEPLOYMENT.md) for deployment guide.
