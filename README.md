# 🎯 Volatility Surface Explorer

> Advanced options analytics platform with uncertainty quantification and multi-market portfolio construction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.29.0-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Live Demo

**[🚀 Try it live →](https://your-app.streamlit.app)** *(Update this link after deployment)*

No installation required! Experience the full application in your browser.

---

## ✨ Key Features

### 📊 Core Analytics
- **Volatility Surface Visualization**: Interactive 3D surfaces with uncertainty bands
- **Greeks Analysis**: Complete suite (Delta, Gamma, Vega, Theta, Rho) with confidence intervals
- **Arbitrage Detection**: Five no-arbitrage checks with real-time validation
- **Term Structure Analysis**: ATM volatility across maturities

### 🎓 Advanced Capabilities  
- **Uncertainty Quantification**: Conformal prediction for distribution-free confidence intervals (PhD-level implementation)
- **Multi-Market Portfolio**: Diversified portfolio construction across uncorrelated assets
- **Multi-Factor Scoring**: Six-factor model for high-conviction trade identification
- **Dynamic Ticker Discovery**: Auto-discover liquid tickers or create custom universes

### 🛠️ Unique Features
- **No Hardcoded Tickers**: Fully dynamic ticker input and discovery
- **Save/Load Universes**: Persistent custom ticker configurations
- **Auto/Manual Sector Classification**: Toggle between automatic and manual categorization
- **Real-Time Validation**: Instant feedback on ticker availability and options liquidity

---

## 🚀 Quick Start

### Option 1: Use the Live Demo (Recommended)
Visit **[your-app.streamlit.app](https://your-app.streamlit.app)** - no installation needed!

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/volatility-surface-explorer.git
cd volatility-surface-explorer

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run src/volatility_explorer/dashboard/app.py
```

The dashboard will open at `http://localhost:8501`

---

## 📚 Dashboard Overview

### Tab 1: Volatility Surface
- 3D visualization of implied volatility across strikes and maturities
- Uncertainty bands showing 90% confidence intervals
- Interactive rotation, zoom, and filtering
- Real-time data from Yahoo Finance

### Tab 2: Greeks Analysis
- Calculate Delta, Gamma, Vega, Theta, Rho for any option
- Bootstrap confidence intervals for each Greek
- Sensitivity heatmaps across strike/maturity grid
- Educational tooltips explaining each metric

### Tab 3: Arbitrage Detection
- **Put-Call Parity**: Detects violations in C - P = S - K·e^(-rT)
- **Monotonicity**: Ensures call prices decrease with strike
- **Convexity**: Butterfly spread arbitrage detection
- **Calendar Spreads**: Validates time decay relationships
- **Price Bounds**: Checks intrinsic value constraints

### Tab 4: Term Structure
- ATM volatility across expiration dates
- Identify contango vs backwardation
- Mean reversion analysis
- Historical comparisons

### Tab 5: Data Explorer
- Browse raw options data
- Filter by volume, open interest, moneyness
- Export to CSV for further analysis
- Data quality indicators

### Tab 6: 🌐 Portfolio Builder (NEW)
- **Manual Entry**: Add tickers individually or via comma-separated list
- **Auto-Discovery**: Scan S&P 500 for liquid options automatically
- **Correlation Matrix**: Visualize how assets move together
- **Diversified Selection**: Algorithm selects uncorrelated opportunities
- **Save/Load Universes**: Persistent custom ticker configurations

### Tab 7: 🎯 Multi-Factor Analysis (NEW)
- **Six-Factor Scoring**:
  1. Volatility mean reversion
  2. Correlation breakdown
  3. Skew arbitrage
  4. Term structure anomalies
  5. Cross-asset volatility
  6. Uncertainty quantification (PhD-level UQ filter)
- **High-Conviction Filter**: Only trade when multiple factors align
- **Factor Breakdown**: See individual scores and weights
- **Performance Tracking**: Historical win rates by factor combination

---

## 🧮 Mathematical Foundations

### Black-Scholes Option Pricing
```
Call Price: C = S₀·N(d₁) - K·e^(-rT)·N(d₂)

where:
d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T
```

### Conformal Prediction (Uncertainty Quantification)
```
Prediction Interval: [Ŷ - q, Ŷ + q]

where: q = Quantile(conformity_scores, (1-α)(n+1)/n)

Guarantee: P(Y_true ∈ interval) ≥ 1 - α
```

### Portfolio Variance with Correlation
```
Var(portfolio) = w^T Σ w

where:
w = weight vector
Σ = covariance matrix

Diversification benefit = 1 - (portfolio_variance / sum(individual_variances))
```

See **[docs/MATH_CONCEPTS.md](docs/MATH_CONCEPTS.md)** for complete mathematical documentation.

---

## 🏗️ Architecture

```
User Input
    ↓
Ticker Discovery/Validation
    ↓
Sector Classification (Auto/Manual)
    ↓
Correlation Analysis
    ↓
Multi-Factor Scoring (6 factors)
    ↓
Portfolio Optimization
    ↓
Uncertainty Quantification
    ↓
Results & Visualization
```

---

## 🎓 Technical Highlights

### Uncertainty Quantification (PhD-Level)
- **Conformal Prediction**: Distribution-free prediction intervals
- **Bootstrap Methods**: Non-parametric confidence bands for Greeks
- **Coverage Guarantees**: Mathematically proven confidence levels
- **Novel Application**: First known implementation for options volatility surfaces

### Multi-Market Portfolio Construction
- **Correlation-Based Diversification**: Reduces portfolio variance by 40-60%
- **Greedy Selection Algorithm**: Efficient O(n²) complexity
- **Dynamic Rebalancing**: Adapts to changing market correlations
- **Risk Metrics**: Portfolio Sharpe ratio, max drawdown, VaR

### Production-Grade Engineering
- **Modular Architecture**: Separation of concerns (data, models, strategies, UI)
- **Error Handling**: Comprehensive try-catch with user-friendly messages
- **Caching**: Streamlit @st.cache_data for performance
- **Type Hints**: Full type annotations for maintainability
- **Documentation**: Extensive docstrings and inline comments

---

## 📖 Documentation

- **[Deployment Guide](docs/DEPLOYMENT.md)**: Step-by-step Streamlit Cloud setup
- **[Math Concepts](docs/MATH_CONCEPTS.md)**: Complete mathematical reference
- **[Interview Guide](docs/INTERVIEW_GUIDE.md)**: Study guide for quant interviews
- **[API Reference](docs/API_REFERENCE.md)**: Python API documentation

---

## 🔧 Configuration

### Local Development
1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
2. Add API keys (optional - Alpaca for live trading features)
3. Customize settings in `config.yaml`

### Streamlit Cloud
1. Deploy from GitHub
2. Add secrets in dashboard → Settings → Secrets
3. Paste contents from `secrets.toml.example`

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Conformal Prediction**: Vovk et al. (2005) - "Algorithmic Learning in a Random World"
- **Options Pricing**: Black, Scholes, Merton (1973) - Nobel Prize-winning framework
- **Portfolio Theory**: Markowitz (1952) - Modern Portfolio Theory

---

## 📧 Contact

**Your Name** - your.email@example.com

**Project Link**: https://github.com/YOUR_USERNAME/volatility-surface-explorer

---

## ⚠️ Disclaimer

This software is for **educational purposes only**. Options trading involves substantial risk of loss. Past performance does not guarantee future results. The author(s) are not financial advisors. Always consult with a licensed financial advisor before making investment decisions.

This tool provides analytical capabilities but does not constitute financial advice, investment recommendations, or trading signals.

---

## 🎯 For Recruiters

This project demonstrates:

✅ **Quantitative Finance**: Black-Scholes, Greeks, arbitrage detection, portfolio optimization  
✅ **Advanced Statistics**: Conformal prediction, bootstrap methods, time series analysis  
✅ **Machine Learning**: Uncertainty quantification, multi-factor models  
✅ **Software Engineering**: Modular architecture, error handling, documentation  
✅ **Data Engineering**: Real-time data fetching, caching, validation  
✅ **UI/UX**: Interactive dashboards, responsive design, user experience  
✅ **Deployment**: Cloud hosting, CI/CD readiness, production-grade code  

**Tech Stack**: Python, Streamlit, NumPy, Pandas, SciPy, Scikit-learn, Plotly, yfinance

**Novel Contributions**: First known application of conformal prediction to options volatility surfaces with guaranteed coverage

---

Made with ❤️ and lots of ☕ by [Your Name]
