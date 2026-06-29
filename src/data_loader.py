import yfinance as yf
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller

TICKERS = [
    "XOM", "CVX", "COP", "EOG", "SLB", "HAL",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP",
    "KO", "PEP", "PG", "CL", "KMB", "GIS",
    "V", "MA", "PYPL",
    "MSFT", "GOOGL", "META", "AAPL", "AMZN",
    "JNJ", "PFE", "MRK", "ABT", "UNH",
    "CAT", "DE", "HON", "MMM", "GE",
    "WMT", "TGT", "COST", "KR",
]

def download_prices(tickers =TICKERS , start = "2018-01-01", end = "2024-01-01"):
    raw = yf.download(tickers, start = start , end = end , auto_adjust = True)
    prices = raw["Close"]
    prices = prices.ffil(limit = 3 , inplace = True)    
    return prices
"""
ffill means:Forward Fill
It copies the previous valid value forward into missing places (NaN).
Now What Does limit=3 Mean?
prices.ffill(limit=3)
means:
"Only fill at most 3 consecutive missing values."

Real Finance Example

Suppose a stock stopped trading for 20 days.

Without limit:
ffill() would repeat the same old price for 20 days.
That is dangerous because:
volatility becomes wrong
returns become distorted
backtesting becomes unrealistic
"""

def load_prices (path = "../data/prices.parquet"):
    return pd.read_parquet(path)

def compute_returns(prices): 
    log_returns = np.log(prices/prices.shift(1))
    log_returns = log_returns.dropna()
    return log_returns

def load_returns(path = "../data/log_returns.parquet"):
    return pd.read_parquet(path)

def compute_correlation(log_returns, threashold = 0.8):
    corr_matrix = log_returns.corr()
    pairs = []
    tickers = corr_matrix.columns.tolist()

    for i in range(len(tickers)):
        for j in range(i+1,len(tickers)):
            r =corr_matrix.iloc[i,j]
            if r > threashold:
                pairs.append((tickers[i],tickers[j] , round(r,4)))

    pairs_df = pd.DataFrame(pairs, columns = ["tickers_1", "tickers_2","correlation"])
    pairs_df = pairs_df.sort_values("correlation",ascending=False).reset_index(drop=True)
    return corr_matrix,pairs_df

def adf_test(series,name=""):
    result = adfuller(series)
    p_value = result[1]
    adf_stat = result[0]
    critical_5 = result[4]["5%"]
    stationary = p_value < 0.05

    print(f"{name}")
    print(f"  ADF Statistic : {adf_stat:.4f}")
    print(f"  p-value       : {p_value:.6f}")
    print(f"  Critical (5%) : {critical_5:.4f}")
    print(f"  Stationary    : {stationary}")
    print()
    
    return stationary

def find_cointegrated_pairs(prices,pairs_df):
    from statsmodels.tsa.stattools import coint

    coint_results = []
    for _,row in pairs_df.iterrows():
        t1 = row["tickers_1"]
        t2 = row["tickers_2"]

        score, p_value, _ = coint(prices[t1],prices[t2])
        
        coint_results.append({
            "ticker_1" : t1,
            "ticker_2" : t2,
            "correlation" : row["correlation"],
            "coint_pvalue" : round(p_value ,6),
            "cointegrated" : p_value < 0.05 
        })

    coint_df = pd.DataFrame(coint_results)
    coint_df = coint_df.sort_values("coint_pvalue").reset_index(drop = True)
    return coint_df

def compute_hedge_ratio(prices,ticker1 ,ticker2):
    from statsmodels.regression.linear_model import OLS 
    from statsmodels.tools import add_constant

    y= prices[ticker1]
    x = add_constant(prices[ticker2])
    model = OLS(y,x).fit()
    beta = model.params[ticker2]
    intercept = model.params["const"]
    return beta , intercept

def compute_spread (prices, ticker1 , ticker2, beta , intercept):
    spread = prices[ticker1] - beta * prices[ticker2] - intercept
    return spread

def compute_zscore(spread,window = 60):
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    zscore = (spread - rolling_mean) / rolling_std
    return zscore



def generate_signals(zscore , entry = 2.0 , exit = 0.5) :
    signals = pd.Series(0, index = zscore.index)
    position = 0

    for i in range(len(zscore)):
        z = zscore.iloc[i]

        if pd.isna(z) : 
            signals.iloc[i] = 0
            continue

        # entry logic 
        if position == 0:
            if z > entry:
                position = -1   # short spread
            elif z < -entry:
                position = 1    # long spread

        # exit logic
        elif position == 1:
            if z > -exit:
                position = 0    # close long
            
        elif position == -1:
            if z < exit:
                position = 0    # close short

        signals.iloc[i] = position

    return signals


def compute_pnl_with_costs(prices , ticker1 , ticker2 , beta , signals , capital = 10000 , cost_pct = 0.0015):
    ret_1 = prices[ticker1].pct_change()
    ret_2 = prices[ticker2].pct_change()

    half_capital = capital / 2
    pnl_leg1 = ret_1 * half_capital
    pnl_leg2 = ret_2 * half_capital

    signal_shifted = signals.shift(1).fillna(0)

    daily_pnl = signal_shifted * (pnl_leg1 - pnl_leg2)

    trade_entry = (signal_shifted != 0) & (signal_shifted.shift(1) == 0)
    trade_exit= (signal_shifted == 0) & (signal_shifted.shift(1) != 0)
    transaction_cost = (trade_entry | trade_exit) * capital * cost_pct

    net_pnl = daily_pnl - transaction_cost

    return net_pnl

def sharpe_ratio(pnl, capital=10000, periods_per_year=252):
    daily_returns = pnl / capital
    mean_return = daily_returns.mean()
    std_return = daily_returns.std()
    if std_return == 0:
        return 0
    return round((mean_return / std_return) * np.sqrt(periods_per_year), 4)

def max_drawdown(pnl):
    cumulative = pnl.cumsum()
    rolling_peak = cumulative.cummax()
    drawdown = (cumulative - rolling_peak) / (rolling_peak + 10000)
    return round(drawdown.min() * 100, 2)

def sharpe_ratio(pnl, capital=10000, periods_per_year=252):
    daily_returns = pnl / capital
    mean_return = daily_returns.mean()
    std_return = daily_returns.std()
    if std_return == 0:
        return 0
    return round((mean_return / std_return) * np.sqrt(periods_per_year), 4)

def run_rolling_selection(prices, windows, corr_threshold=0.8, 
                           coint_threshold=0.05, top_n=5):
    from stat_arb.src.data_loader import compute_correlation, find_cointegrated_pairs
    
    all_results = []
    
    for i, w in enumerate(windows):
        formation_prices = prices[
            (prices.index >= w["formation_start"]) &
            (prices.index < w["formation_end"])
        ]
        
        if len(formation_prices) < 200:
            continue
        
        try:
            corr_matrix, pairs_df = compute_correlation(
                formation_prices, threshold=corr_threshold)
            
            if len(pairs_df) == 0:
                continue
            
            coint_df = find_cointegrated_pairs(formation_prices, pairs_df)
            valid = coint_df[coint_df["cointegrated"] == True].head(top_n)
            
            for _, row in valid.iterrows():
                all_results.append({
                    "window": i,
                    "formation_start": w["formation_start"],
                    "formation_end": w["formation_end"],
                    "trade_start": w["trade_start"],
                    "ticker_1": row["ticker_1"],
                    "ticker_2": row["ticker_2"],
                    "coint_pvalue": row["coint_pvalue"],
                    "correlation": row["correlation"]
                })
        except:
            continue
    
    return pd.DataFrame(all_results)


def run_strategy_on_period(prices_period, ticker_1, ticker_2):
    from stat_arb.src.data_loader import (compute_hedge_ratio, compute_spread, compute_zscore, generate_signals, compute_pnl_with_costs, sharpe_ratio, max_drawdown)
    try:
        beta, intercept = compute_hedge_ratio(prices_period, ticker_1, ticker_2)
        spread = compute_spread(prices_period, ticker_1, ticker_2, beta, intercept)
        zscore = compute_zscore(spread)
        signals = generate_signals(zscore)
        pnl = compute_pnl_with_costs(prices_period, ticker_1, ticker_2, beta, signals)
        return {
            "ticker_1": ticker_1,
            "ticker_2": ticker_2,
            "total_return_pct": round((pnl.sum() / 10000) * 100, 2),
            "sharpe": sharpe_ratio(pnl),
            "max_drawdown_pct": max_drawdown(pnl),
            "trades": int((signals.diff().abs() > 0).sum() / 2)
        }
    except:
        return None