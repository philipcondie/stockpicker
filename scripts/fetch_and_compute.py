#!/usr/bin/env python3
"""
Fetch S&P 500 historical data and compute daily best/worst stock performance.

Compares three strategies starting with $10,000:
1. Buy & hold the S&P 500 index
2. "Perfect picker" — buy the best-performing stock each day
3. "Worst picker" — buy the worst-performing stock each day

Outputs pre-computed JSON for the static GitHub Pages visualization.
"""

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
START_DATE = "2000-01-01"
INITIAL_INVESTMENT = 10_000


def get_sp500_tickers():
    """Scrape current S&P 500 constituents from Wikipedia."""
    print("Fetching S&P 500 ticker list from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df["Symbol"].tolist()
    # Yahoo Finance uses hyphens instead of dots
    tickers = [t.replace(".", "-") for t in tickers]
    print(f"  Found {len(tickers)} tickers")
    return tickers


def download_data(tickers, start_date=START_DATE):
    """Download historical daily adjusted close data for all tickers + S&P 500 index."""
    all_tickers = tickers + ["^GSPC"]
    print(f"Downloading data for {len(all_tickers)} tickers from {start_date}...")

    batch_size = 50
    frames = []

    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(all_tickers) - 1) // batch_size + 1
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} tickers)...")

        try:
            data = yf.download(
                tickers=batch,
                start=start_date,
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            if data is not None and not data.empty:
                # yf.download returns multi-level columns: (Price, Ticker)
                # We want the Close prices
                if len(batch) == 1:
                    # Single ticker returns flat columns
                    close = data[["Close"]].copy()
                    close.columns = batch
                else:
                    close = data["Close"]
                frames.append(close)
        except Exception as e:
            print(f"    Warning: batch failed: {e}")

        if i + batch_size < len(all_tickers):
            time.sleep(2)

    if not frames:
        print("ERROR: No data downloaded!")
        sys.exit(1)

    df = pd.concat(frames, axis=1)
    # Remove duplicate columns if any
    df = df.loc[:, ~df.columns.duplicated()]
    print(f"  Downloaded data: {df.shape[0]} trading days, {df.shape[1]} tickers")
    return df


def compute_strategies(prices_df):
    """Compute cumulative portfolio values for all three strategies."""
    # Separate index from stocks
    sp500 = prices_df["^GSPC"].dropna()
    stocks = prices_df.drop(columns=["^GSPC"], errors="ignore")

    # Compute daily returns
    sp500_returns = sp500.pct_change()
    stock_returns = stocks.pct_change()

    # For each day, find best and worst returning stock (among those with data)
    # Need at least some stocks with valid data on each day
    daily_best = stock_returns.max(axis=1)
    daily_worst = stock_returns.min(axis=1)
    daily_best_ticker = stock_returns.idxmax(axis=1)
    daily_worst_ticker = stock_returns.idxmin(axis=1)

    # Align all series to the same date index (only days where we have SP500 + stocks)
    common_idx = sp500_returns.dropna().index.intersection(daily_best.dropna().index)
    sp500_returns = sp500_returns.loc[common_idx]
    daily_best = daily_best.loc[common_idx]
    daily_worst = daily_worst.loc[common_idx]
    daily_best_ticker = daily_best_ticker.loc[common_idx]
    daily_worst_ticker = daily_worst_ticker.loc[common_idx]

    # Compute cumulative portfolio values
    # S&P 500 stays in normal dollar values
    sp500_cumulative = INITIAL_INVESTMENT * (1 + sp500_returns).cumprod()

    # Best/worst use log10 space to handle extreme compounding
    best_log = np.log10(INITIAL_INVESTMENT) + np.log10(1 + daily_best).cumsum()
    worst_log = np.log10(INITIAL_INVESTMENT) + np.log10(1 + daily_worst).cumsum()

    print(f"\nResults over {len(common_idx)} trading days:")
    print(f"  S&P 500:      ${sp500_cumulative.iloc[-1]:>20,.2f}")
    print(f"  Best picker:  $10^{best_log.iloc[-1]:.1f}")
    print(f"  Worst picker: $10^{worst_log.iloc[-1]:.1f}")

    return {
        "dates": common_idx,
        "sp500": sp500_cumulative,
        "best_log": best_log,
        "worst_log": worst_log,
        "best_tickers": daily_best_ticker,
        "worst_tickers": daily_worst_ticker,
        "best_returns": daily_best,
        "worst_returns": daily_worst,
        "sp500_returns": sp500_returns,
    }


def build_output(results):
    """Build the JSON output for the frontend visualization."""
    dates = results["dates"]

    records = []
    for i, date in enumerate(dates):
        d = date.strftime("%Y-%m-%d")
        records.append({
            "d": d,
            "sp": round(float(results["sp500"].iloc[i]), 2),
            "bl": round(float(results["best_log"].iloc[i]), 4),
            "wl": round(float(results["worst_log"].iloc[i]), 4),
            "bt": str(results["best_tickers"].iloc[i]) if pd.notna(results["best_tickers"].iloc[i]) else "",
            "wt": str(results["worst_tickers"].iloc[i]) if pd.notna(results["worst_tickers"].iloc[i]) else "",
            "br": round(float(results["best_returns"].iloc[i]) * 100, 2),
            "wr": round(float(results["worst_returns"].iloc[i]) * 100, 2),
        })

    output = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "initial_investment": INITIAL_INVESTMENT,
        "start_date": records[0]["d"] if records else START_DATE,
        "end_date": records[-1]["d"] if records else "",
        "total_trading_days": len(records),
        "final_values": {
            "sp500": records[-1]["sp"] if records else 0,
            "best_log10": records[-1]["bl"] if records else 0,
            "worst_log10": records[-1]["wl"] if records else 0,
        },
        "data": records,
    }
    return output


def main():
    DATA_DIR.mkdir(exist_ok=True)

    tickers = get_sp500_tickers()
    prices = download_data(tickers)
    results = compute_strategies(prices)
    output = build_output(results)

    out_path = DATA_DIR / "sp500_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f)

    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nOutput written to {out_path} ({file_size_mb:.1f} MB)")
    print(f"Date range: {output['start_date']} to {output['end_date']}")
    print(f"Trading days: {output['total_trading_days']}")


if __name__ == "__main__":
    main()
