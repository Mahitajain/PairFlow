# PairFlow — Dynamic Statistical Arbitrage

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pairflow-8mltvonck9hbe4ufxrxhnm.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

> ML-enhanced pairs trading system that dynamically identifies 
> cointegrated stock pairs and ranks them using XGBoost — 
> achieving **46× improvement in top-10 pair selection performance** in pair selection quality 
> over statistical baselines.

## Live Demo
**[Launch Dashboard →](https://pairflow-8mltvonck9hbe4ufxrxhnm.streamlit.app/)**

![Dashboard Preview](assets/dashboard_preview.gif)

---

## Results

| Ranking Method | Top-10 Avg Sharpe | vs Baseline |
|---------------|-------------------|-------------|
| P-value Baseline | 0.015 | — |
| Hurst Exponent | 0.185 | +0.170 |
| Random Forest | 0.674 | +0.659 |
| **XGBoost** | **0.693** | **+0.678** |

**Best pair identified:** CRM/META (Predicted Sharpe: 1.657, Actual Sharpe: 1.683)

---

## Project Overview

PairFlow is an end-to-end quantitative trading system built 
from scratch. It combines classical statistical arbitrage 
methodology with machine learning to dynamically identify 
and rank tradeable stock pairs.

### The Problem
Traditional pair trading uses fixed pairs and ranks them 
purely by cointegration p-value. This ignores multiple 
dimensions of pair quality and fails when market 
relationships change.

### The Solution
PairFlow dynamically re-selects pairs using a rolling 
12-month window and ranks them using XGBoost trained on 
11 engineered features — achieving 46x better pair 
selection than p-value ranking.

---

## Pipeline Architecture

```
Raw Prices (110 S&P 500 stocks, 6 years, 6 sectors)
         ↓
Sector-based pair filtering
(978 candidates → 91 fast filter → 23 cointegrated)
         ↓
Feature Engineering (11 features per pair)
- Statistical: cointegration p-value, correlation, ADF
- Spread: volatility, skewness, kurtosis, mean crossings
- Temporal: half-life, Hurst exponent, beta stability
         ↓
XGBoost Regressor (predicts out-of-sample Sharpe)
         ↓
Z-score signal generation (entry ±2σ, exit ±0.5σ)
         ↓
Backtesting with transaction costs
(Sharpe, drawdown, win rate, P&L)
```

---

## Key Features

- **Dynamic pair selection** — rolling 12-month formation 
  windows, pairs updated monthly
- **Sector-based filtering** — 978 → 91 candidates using 
  correlation + log-ratio ADF pre-filter
- **Parallel cointegration testing** — multiprocessing.Pool 
  across all CPU cores, 91 pairs in 6.3 seconds
- **11 engineered features** — statistical + spread + 
  temporal characteristics per pair
- **XGBoost pair ranking** — 46x improvement over p-value 
  baseline on out-of-sample data
- **SHAP explainability** — feature attribution for every 
  prediction
- **Walk-forward validation** — no look-ahead bias, 
  train 2018–2021, test 2022–2023
- **Live Streamlit dashboard** — interactive pair analysis, 
  z-score charts, cumulative P&L

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Programming Language | Python |
| Data Collection | yfinance |
| Data Storage | Parquet |
| Data Processing | pandas, NumPy |
| Statistical Analysis | statsmodels (ADF, Engle-Granger, OLS) |
| Machine Learning | scikit-learn, XGBoost |
| Model Explainability | SHAP |
| Visualization | Plotly, Matplotlib |
| Dashboard | Streamlit |
| Deployment | Streamlit Cloud |
| Parallel Computing | multiprocessing |
| Version Control | Git, GitHub |

---

## Installation

```bash
git clone https://github.com/Mahitajain/PairFlow.git
cd pairflow
pip install -r requirements.txt
```

**Download data and run pipeline:**
```bash
pip install -r requirements.txt

streamlit run app/dashboard.py
```

---

## Project Structure

```
pairflow/
├── app/
│   └── dashboard.py       # Streamlit dashboard
├── data/                  # Generated data files
├── notebooks/             # Day-by-day development
│   ├── day1_data.ipynb
│   ├── ...
│   └── june11_scaling.ipynb
├── src/
│   └── data_loader.py     # Core pipeline functions
├── requirements.txt
└── README.md
```

---

## Concepts Used

**Statistics:** Cointegration, ADF stationarity test, 
OLS regression, Engle-Granger two-step method, 
z-score, rolling windows

**Finance:** Pairs trading, mean reversion, hedge ratio, 
spread construction, P&L calculation, Sharpe ratio, 
maximum drawdown

**ML:** Feature engineering, Random Forest, XGBoost, 
SHAP values, walk-forward validation, 
cross-validation

---

## Author

**Mahita Jain** — Computer Science Engineering Student | Machine Learning & Quantitative Finance Enthusiast 
[LinkedIn](https://www.linkedin.com/in/mahita-jain-b276392b1/) | [GitHub](https://github.com/Mahitajain)

---

*Built to demonstrate end-to-end quantitative finance, statistical arbitrage, machine learning, and interactive dashboard development.*