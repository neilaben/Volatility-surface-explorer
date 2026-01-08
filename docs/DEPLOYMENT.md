# 🚀 Deployment Guide - Streamlit Cloud

Complete guide to deploying Volatility Surface Explorer on Streamlit Cloud.

---

## ✅ Prerequisites

- GitHub account
- Your repository pushed to GitHub (public or private)
- Streamlit Community Cloud account (free - sign up with GitHub)

---

## 📋 Step-by-Step Deployment

### Step 1: Prepare Your Repository

Ensure these files exist in your repo root:

```
✅ requirements.txt          (all Python dependencies)
✅ .streamlit/config.toml    (theme and server settings)
✅ .gitignore                (excludes secrets.toml)
✅ src/volatility_explorer/dashboard/app.py (main entry point)
```

### Step 2: Push to GitHub

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Volatility Surface Explorer"

# Create repo on GitHub (via web interface or CLI)
# Then push:
git remote add origin https://github.com/YOUR_USERNAME/volatility-surface-explorer.git
git branch -M main
git push -u origin main
```

### Step 3: Sign Up for Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Click "Sign up with GitHub"
3. Authorize Streamlit to access your GitHub

### Step 4: Deploy Your App

1. Click "**New app**" button
2. Fill in the form:
   ```
   Repository: YOUR_USERNAME/volatility-surface-explorer
   Branch: main
   Main file path: src/volatility_explorer/dashboard/app.py
   ```
3. (Optional) Add custom subdomain:
   ```
   App URL: your-custom-name.streamlit.app
   ```
4. Click "**Deploy!**"

### Step 5: Wait for Deployment

- ⏳ Initial deployment: 2-5 minutes
- 📦 Streamlit installs all requirements.txt dependencies
- 🚀 App launches automatically

### Step 6: Access Your Live App

Your app will be available at:
```
https://YOUR_USERNAME-volatility-surface-explorer.streamlit.app
```

or your custom URL:
```
https://your-custom-name.streamlit.app
```

---

## 🔐 Adding Secrets (Optional)

If using Alpaca API or other secrets:

### Via Streamlit Cloud Dashboard

1. Go to your app dashboard
2. Click "**Settings**" → "**Secrets**"
3. Paste contents from `.streamlit/secrets.toml.example`:
   ```toml
   [alpaca]
   api_key = "YOUR_ACTUAL_KEY"
   secret_key = "YOUR_ACTUAL_SECRET"
   paper = true
   
   [settings]
   cache_ttl = 300
   ```
4. Click "**Save**"
5. App will automatically restart with secrets loaded

### Accessing Secrets in Code

```python
import streamlit as st

# Access secrets
if 'alpaca' in st.secrets:
    api_key = st.secrets["alpaca"]["api_key"]
    secret_key = st.secrets["alpaca"]["secret_key"]
```

---

## 🔄 Updating Your Deployed App

### Automatic Updates (Recommended)

Every time you push to GitHub, Streamlit Cloud automatically redeploys:

```bash
# Make changes locally
git add .
git commit -m "Added new feature"
git push

# Streamlit Cloud detects push and redeploys (1-2 minutes)
```

### Manual Reboot

If needed, reboot from dashboard:
1. Go to app dashboard
2. Click "**⋮**" menu → "**Reboot app**"

---

## 📊 Monitoring Your App

### View Logs

1. App dashboard → "**Manage app**"
2. Click "**Logs**" tab
3. See real-time application logs

### Check Status

- 🟢 **Running**: App is live
- 🟡 **Starting**: App is deploying
- 🔴 **Error**: Check logs for issues

### Common Issues

**Import Error**
```
Fix: Add missing package to requirements.txt
Then: git push (auto-redeploys)
```

**Memory Limit Exceeded**
```
Fix: Optimize data loading, add caching
Streamlit Cloud free tier: 1GB RAM limit
```

**Slow Performance**
```
Fix: Add @st.cache_data decorators
Example:
@st.cache_data(ttl=300)
def fetch_data(ticker):
    ...
```

---

## 🎨 Customization

### Change Theme

Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#YOUR_COLOR"
backgroundColor = "#YOUR_BG_COLOR"
```

Then push to GitHub.

### Update App Settings

Edit `.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 200  # MB
enableXsrfProtection = true
```

---

## 💰 Pricing & Limits

### Free Tier (Community Cloud)

✅ **Free forever** for public repos  
✅ 1GB RAM  
✅ 1 CPU core  
✅ Unlimited apps  
✅ Auto-sleep after inactivity (wakes on visit)  

### Paid Tier

- **Private repos**: $7/month per developer
- **Teams**: $20/month per developer
- **Enterprise**: Custom pricing

For most portfolio projects, **free tier is perfect!**

---

## 🔒 Security Best Practices

### Never Commit Secrets

```bash
# .gitignore should include:
.streamlit/secrets.toml
.env
config.yaml
*.key
```

### Use Environment Variables

For sensitive data, use Streamlit secrets:
```python
# ❌ Don't hardcode
API_KEY = "pk_live_123abc"

# ✅ Use secrets
API_KEY = st.secrets["api_key"]
```

### Public vs Private Repos

- **Public**: Anyone can see code (but not secrets)
- **Private**: Code hidden (requires paid tier)

**For portfolio**: Public is fine! Secrets stay in Streamlit dashboard.

---

## 🐛 Troubleshooting

### App Won't Start

**Check requirements.txt**
```bash
# Ensure exact versions:
streamlit==1.29.0
pandas==2.1.3
# (not just 'streamlit' or 'pandas')
```

**Check main file path**
```
Must be: src/volatility_explorer/dashboard/app.py
Not: app.py or src/app.py
```

### Module Not Found Error

**Add to requirements.txt**
```bash
# If error: "ModuleNotFoundError: No module named 'xyz'"
# Add to requirements.txt:
xyz==1.2.3
```

### App Crashes After Deploy

**Check logs:**
1. Dashboard → Logs
2. Find error traceback
3. Common issues:
   - Missing dependency
   - File path error
   - Memory limit exceeded

---

## 📈 Performance Optimization

### Caching Data

```python
import streamlit as st

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_options_data(ticker):
    # Expensive operation
    return data

# Called once per 5 minutes, not on every interaction
data = fetch_options_data("SPY")
```

### Lazy Loading

```python
# Don't load all data at startup
if st.button("Fetch Data"):
    data = fetch_large_dataset()
    st.session_state['data'] = data
```

### Reduce Memory

```python
# Use generators instead of lists
def process_large_data():
    for chunk in data_chunks:
        yield process(chunk)

# Delete unused data
del large_dataframe
```

---

## 🎯 Going Live Checklist

Before sharing your app:

- [ ] Test all features locally
- [ ] Update README with live URL
- [ ] Add descriptive app title/favicon
- [ ] Include disclaimer text
- [ ] Test on mobile (responsive design)
- [ ] Check logs for errors
- [ ] Add analytics (optional - Google Analytics)
- [ ] Create demo video/screenshots
- [ ] Share on LinkedIn/portfolio

---

## 🌐 Sharing Your App

### Update README

```markdown
## 🚀 Live Demo

**[Try it now →](https://your-app.streamlit.app)**

No installation required!
```

### Social Media

**LinkedIn Post:**
```
🚀 Excited to share my latest project: Volatility Surface Explorer

Advanced options analytics with:
📊 3D volatility surfaces
🎯 Uncertainty quantification (PhD-level)
🌐 Multi-market portfolio construction

Try it live: https://your-app.streamlit.app

Built with Python, Streamlit, and cutting-edge quant finance techniques.

#QuantFinance #Python #MachineLearning #DataScience
```

### In Interviews

"I built a production-grade options analytics platform deployed on Streamlit Cloud. You can try it live at [URL]. It demonstrates my skills in quantitative finance, statistical modeling, and full-stack development."

---

## 🔗 Useful Links

- **Streamlit Cloud**: https://share.streamlit.io
- **Streamlit Docs**: https://docs.streamlit.io
- **Community Forum**: https://discuss.streamlit.io
- **GitHub**: https://github.com/streamlit/streamlit

---

## 📧 Need Help?

**Streamlit Community Forum**: https://discuss.streamlit.io  
**GitHub Issues**: Open issue in your repo  
**Email Support**: support@streamlit.io (for cloud issues)

---

**Ready to deploy?** Follow steps 1-6 above and you'll be live in 10 minutes! 🚀
