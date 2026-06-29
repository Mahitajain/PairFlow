# %%
# at top of dashboard.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pickle
import sys
import os
sys.path.append("../src")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SRC_DIR  = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# then use DATA_DIR for all file paths
@st.cache_data
def load_data():
    prices = pd.read_parquet(
        os.path.join(DATA_DIR, "prices_large.parquet"))
    ml_ranked = pd.read_csv(
        os.path.join(DATA_DIR, "ml_large_ranked.csv"))
    return prices, ml_ranked

@st.cache_resource
def load_models():
    with open(os.path.join(DATA_DIR, "xgb_large.pkl"), "rb") as f:
        xgb = pickle.load(f)
    with open(os.path.join(DATA_DIR, "scaler_large.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return xgb, scaler




from data_loader import (load_prices, compute_hedge_ratio,
                          compute_spread, compute_zscore,
                          generate_signals, compute_pnl_with_costs,
                          sharpe_ratio, max_drawdown)

# ── page config ──────────────────────────────────────────
st.set_page_config(
    page_title="PairFlow",
    page_icon="📈",
    layout="wide"
)

# ── load data ────────────────────────────────────────────
@st.cache_data
def load_data():
    prices = pd.read_parquet("../data/prices_large.parquet")
    ml_ranked = pd.read_csv("../data/ml_large_ranked.csv")
    return prices, ml_ranked

@st.cache_resource
def load_models():
    with open("../data/xgb_large.pkl", "rb") as f:
        xgb = pickle.load(f)
    with open("../data/scaler_large.pkl", "rb") as f:
        scaler = pickle.load(f)
    return xgb, scaler

prices, ml_ranked = load_data()
xgb_model, scaler = load_models()

# ── sidebar ───────────────────────────────────────────────
st.sidebar.title("PairFlow")
st.sidebar.caption("Dynamic Statistical Arbitrage")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Pair Analysis", "ML Rankings", "Metrics"]
)

# ═════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════════════════
if page == "Overview":
    st.title("PairFlow — Dynamic Statistical Arbitrage")
    st.caption("Pair selection using cointegration + XGBoost ranking across 110 S&P 500 stocks")

    # top metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Universe", "110 stocks", "6 sectors")
    col2.metric("Valid Pairs", f"{len(ml_ranked)}", "cointegrated")
    col3.metric("Best XGB Sharpe", 
                f"{ml_ranked['xgb_score'].max():.2f}",
                "out-of-sample")
    col4.metric("XGB vs Baseline", 
                f"+{(0.693 - 0.015):.3f}",
                "avg top-10 Sharpe improvement")

    st.divider()

    # ranking comparison bar chart
    st.subheader("Pair Ranking Method Comparison")
    st.caption("Average out-of-sample Sharpe ratio — Top 10 pairs per method")

    comparison_data = pd.DataFrame({
        "Method": ["P-value Baseline", "Hurst Baseline",
                   "RF Regressor", "XGBoost"],
        "Avg Sharpe": [0.015, 0.185, 0.674, 0.693]
    })

    fig = px.bar(
        comparison_data,
        x="Method", y="Avg Sharpe",
        color="Avg Sharpe",
        color_continuous_scale="RdYlGn",
        title="XGBoost Ranking vs Statistical Baselines"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # top pairs table
    st.subheader("Top 10 XGBoost-Ranked Pairs")
    display_cols = ["ticker_1", "ticker_2",
                    "xgb_score", "test_sharpe",
                    "hurst", "half_life", "mean_crossings"]
    
    top10 = ml_ranked.nlargest(10, "xgb_score")[display_cols].copy()
    top10.columns = ["Stock 1", "Stock 2", "XGB Score",
                     "Actual Sharpe", "Hurst", 
                     "Half-Life (days)", "Mean Crossings"]
    top10 = top10.reset_index(drop=True)
    top10.index += 1

    st.dataframe(
        top10.style.background_gradient(
            subset=["XGB Score", "Actual Sharpe"],
            cmap="RdYlGn"),
        use_container_width=True
    )

# ═════════════════════════════════════════════════════════
# PAGE 2 — PAIR ANALYSIS
# ═════════════════════════════════════════════════════════
elif page == "Pair Analysis":
    st.title("Pair Analysis")

    # pair selector
    pair_options = [
        f"{row['ticker_1']} / {row['ticker_2']}"
        for _, row in ml_ranked.nlargest(
            len(ml_ranked), "xgb_score").iterrows()
    ]

    selected = st.selectbox(
        "Select a pair to analyse:", pair_options)
    t1, t2 = selected.split(" / ")

    # date range
    col1, col2 = st.columns(2)
    start_date = col1.date_input(
        "Start date", 
        value=pd.to_datetime("2018-01-01"))
    end_date = col2.date_input(
        "End date",
        value=pd.to_datetime("2023-12-31"))

    period_prices = prices[
        (prices.index >= pd.Timestamp(start_date)) &
        (prices.index <= pd.Timestamp(end_date))
    ]

    if t1 in period_prices.columns and t2 in period_prices.columns:
        try:
            beta, intercept = compute_hedge_ratio(
                period_prices, t1, t2)
            spread = compute_spread(
                period_prices, t1, t2, beta, intercept)
            zscore = compute_zscore(spread)
            signals = generate_signals(zscore)
            pnl = compute_pnl_with_costs(
                period_prices, t1, t2, beta, signals)

            # metrics row
            sharpe = sharpe_ratio(pnl)
            mdd = max_drawdown(pnl)
            total_ret = round((pnl.sum()/10000)*100, 2)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sharpe Ratio", f"{sharpe:.3f}")
            c2.metric("Max Drawdown", f"{mdd:.1f}%")
            c3.metric("Total Return", f"{total_ret:.1f}%")
            c4.metric("Hedge Ratio β", f"{beta:.4f}")

            st.divider()

            # price chart
            st.subheader(f"{t1} vs {t2} — Prices")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=period_prices.index,
                y=period_prices[t1],
                name=t1, line=dict(width=1.5)))
            fig.add_trace(go.Scatter(
                x=period_prices.index,
                y=period_prices[t2],
                name=t2, line=dict(width=1.5)))
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

            # spread chart
            st.subheader("Spread with ±2σ Bands")
            roll_mean = spread.rolling(60).mean()
            roll_std  = spread.rolling(60).std()

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=spread.index, y=spread,
                name="Spread",
                line=dict(color="purple", width=1)))
            fig2.add_trace(go.Scatter(
                x=spread.index, y=roll_mean,
                name="Rolling Mean",
                line=dict(color="red", width=1,
                          dash="dash")))
            fig2.add_trace(go.Scatter(
                x=spread.index,
                y=roll_mean + 2*roll_std,
                name="+2σ",
                line=dict(color="green", width=0.8,
                          dash="dot")))
            fig2.add_trace(go.Scatter(
                x=spread.index,
                y=roll_mean - 2*roll_std,
                name="-2σ",
                line=dict(color="green", width=0.8,
                          dash="dot")))
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)

            # z-score chart
            st.subheader("Z-Score & Trading Signals")
            colors = signals.map(
                {1: "green", -1: "red", 0: "gray"})

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=zscore.index, y=zscore,
                name="Z-Score",
                line=dict(color="purple", width=1)))
            fig3.add_hline(y=2, line_dash="dash",
                           line_color="red",
                           annotation_text="Entry +2σ")
            fig3.add_hline(y=-2, line_dash="dash",
                           line_color="green",
                           annotation_text="Entry -2σ")
            fig3.add_hline(y=0, line_dash="dot",
                           line_color="gray")
            fig3.update_layout(height=350)
            st.plotly_chart(fig3, use_container_width=True)

            # cumulative P&L
            st.subheader("Cumulative P&L")
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(
                x=pnl.index,
                y=pnl.cumsum(),
                name="Cumulative P&L",
                fill="tozeroy",
                line=dict(color="steelblue", width=1.5)))
            fig4.add_hline(y=0, line_dash="dash",
                           line_color="black")
            fig4.update_layout(
                height=350,
                yaxis_title="Cumulative P&L ($)")
            st.plotly_chart(fig4, use_container_width=True)

        except Exception as e:
            st.error(f"Error computing pair: {e}")
    else:
        st.warning(
            f"One or both tickers not in dataset.")

# ═════════════════════════════════════════════════════════
# PAGE 3 — ML RANKINGS
# ═════════════════════════════════════════════════════════
elif page == "ML Rankings":
    st.title("ML Pair Rankings")

    st.subheader("XGBoost Predicted vs Actual Sharpe")
    fig = px.scatter(
        ml_ranked,
        x="xgb_score", y="test_sharpe",
        text="ticker_1",
        color="test_sharpe",
        color_continuous_scale="RdYlGn",
        labels={
            "xgb_score": "XGB Predicted Sharpe",
            "test_sharpe": "Actual Test Sharpe"
        },
        title="Predicted vs Actual — XGBoost Regressor"
    )
    fig.add_hline(y=0, line_dash="dash",
                  line_color="gray")
    fig.add_vline(x=0, line_dash="dash",
                  line_color="gray")
    fig.update_traces(textposition="top center",
                      marker_size=10)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Importance — XGBoost")
    feature_cols = [
        "coint_pval", "correlation", "spread_std",
        "half_life", "hurst", "spread_skew",
        "spread_kurtosis", "mean_crossings",
        "beta_stability"
    ]

    xgb_imp = pd.Series(
        xgb_model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=True)

    fig2 = px.bar(
        x=xgb_imp.values,
        y=xgb_imp.index,
        orientation="h",
        color=xgb_imp.values,
        color_continuous_scale="Blues",
        title="XGBoost Feature Importance"
    )
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("All Pairs Ranked by XGBoost Score")
    all_pairs = ml_ranked[[
        "ticker_1", "ticker_2",
        "xgb_score", "rf_score",
        "test_sharpe", "hurst",
        "half_life", "mean_crossings"
    ]].sort_values(
        "xgb_score", ascending=False
    ).reset_index(drop=True)
    all_pairs.index += 1

    st.dataframe(
        all_pairs.style.background_gradient(
            subset=["xgb_score", "test_sharpe"],
            cmap="RdYlGn"),
        use_container_width=True
    )

# ═════════════════════════════════════════════════════════
# PAGE 4 — METRICS
# ═════════════════════════════════════════════════════════
elif page == "Metrics":
    st.title("Strategy Metrics")

    st.subheader("Ranking Method Comparison")
    comparison = pd.DataFrame({
        "Method": [
            "P-value Baseline",
            "Hurst Baseline",
            "Random Forest",
            "XGBoost"
        ],
        "Top-10 Avg Sharpe": [0.015, 0.185, 0.674, 0.693],
        "Improvement vs P-value": [
            "Baseline",
            f"+{0.185-0.015:.3f}",
            f"+{0.674-0.015:.3f}",
            f"+{0.693-0.015:.3f}"
        ]
    })
    st.dataframe(comparison, use_container_width=True)

    st.divider()

    st.subheader("Project Architecture")
    st.markdown("""
    **Pipeline:**
                Raw Prices (110 stocks, 6 years)
     ↓
Sector-based pair filtering (978 → 91 candidates)
     ↓
Cointegration testing (91 → 23 valid pairs)
     ↓
Feature engineering (11 features per pair)
     ↓
XGBoost pair ranking (Top 10 avg Sharpe: 0.693)
     ↓
Z-score signal generation (entry ±2σ, exit ±0.5σ)
     ↓
Backtesting with transaction costs
                """)

    st.subheader("Key Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("XGBoost Top-10 Sharpe", "0.693",
                "+0.678 vs baseline")
    col2.metric("Best Pair Sharpe", "1.683",
                "CRM/META out-of-sample")
    col3.metric("Universe Scaled", "110 stocks",
                "from 41 (+168%)")

    st.subheader("Tech Stack")
    st.markdown("""
    | Component | Technology |
    |-----------|-----------|
    | Data | yfinance, pandas, parquet |
    | Statistics | statsmodels (ADF, cointegration, OLS) |
    | ML | scikit-learn, XGBoost |
    | Explainability | SHAP |
    | Visualisation | matplotlib, Plotly |
    | Dashboard | Streamlit |
    | Parallelisation | multiprocessing |
    """)

# %%



