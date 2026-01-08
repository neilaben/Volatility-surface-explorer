#!/bin/bash
# Setup script for Volatility Surface Explorer
# This creates the complete project structure

echo "🚀 Setting up Volatility Surface Explorer..."

# Create directory structure
mkdir -p src/volatility_explorer/{data,models,uncertainty,arbitrage,strategies,visualization,dashboard}
mkdir -p src/volatility_explorer/dashboard/{components,utils}
mkdir -p data/universes
mkdir -p docs
mkdir -p tests

# Create __init__.py files
touch src/volatility_explorer/__init__.py
touch src/volatility_explorer/data/__init__.py
touch src/volatility_explorer/models/__init__.py
touch src/volatility_explorer/uncertainty/__init__.py
touch src/volatility_explorer/arbitrage/__init__.py
touch src/volatility_explorer/strategies/__init__.py
touch src/volatility_explorer/visualization/__init__.py
touch src/volatility_explorer/dashboard/__init__.py
touch src/volatility_explorer/dashboard/components/__init__.py
touch src/volatility_explorer/dashboard/utils/__init__.py
touch tests/__init__.py

# Create placeholder files
touch data/universes/.gitkeep
touch LICENSE

echo "✅ Directory structure created!"
echo ""
echo "📋 Next steps:"
echo "1. Add your existing Python files to the appropriate directories"
echo "2. Run: git init"
echo "3. Run: git add ."
echo "4. Run: git commit -m 'Initial commit'"
echo "5. Create repo on GitHub and push"
echo ""
echo "🌐 To deploy on Streamlit Cloud:"
echo "1. Visit: https://share.streamlit.io"
echo "2. Connect your GitHub repo"
echo "3. Set main file: src/volatility_explorer/dashboard/app.py"
echo "4. Deploy!"
