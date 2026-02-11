#!/usr/bin/env python3
"""
Generate realistic sample data for the S&P 500 best/worst stock picker visualization.

Uses statistical models calibrated to actual market characteristics:
- S&P 500 daily returns: modeled with regime-switching volatility
- Best/worst daily stock returns: modeled from empirical distributions

Run the fetch_and_compute.py script locally (with internet access) to get real data.
This script produces realistic sample data for development and demonstration.
"""

import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INITIAL_INVESTMENT = 10_000

# S&P 500 historical characteristics by era
# (start_year, end_year, annual_return, annual_vol, description)
REGIMES = [
    (2000, 2002, -0.15, 0.25, "dot-com crash"),
    (2003, 2007, 0.12, 0.12, "recovery & housing boom"),
    (2008, 2008, -0.38, 0.45, "financial crisis"),
    (2009, 2009, 0.23, 0.30, "recovery"),
    (2010, 2014, 0.14, 0.16, "slow recovery"),
    (2015, 2017, 0.10, 0.12, "steady growth"),
    (2018, 2018, -0.06, 0.20, "Q4 selloff"),
    (2019, 2019, 0.29, 0.13, "strong year"),
    (2020, 2020, 0.16, 0.35, "COVID crash & recovery"),
    (2021, 2021, 0.27, 0.14, "post-COVID boom"),
    (2022, 2022, -0.19, 0.25, "rate hike selloff"),
    (2023, 2023, 0.24, 0.13, "AI rally"),
    (2024, 2024, 0.23, 0.13, "continued growth"),
    (2025, 2026, 0.10, 0.15, "moderate growth"),
]

# S&P 500 tickers (a subset for realistic best/worst labels)
SP500_SAMPLE = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B", "UNH", "JNJ",
    "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP", "KO",
    "AVGO", "COST", "LLY", "WMT", "MCD", "CSCO", "TMO", "ACN", "ABT", "DHR",
    "NEE", "LIN", "TXN", "BMY", "UNP", "PM", "RTX", "HON", "AMGN", "LOW",
    "COP", "QCOM", "IBM", "CAT", "GE", "SBUX", "INTU", "BA", "DE", "AMAT",
    "MDLZ", "ADI", "ISRG", "GILD", "ADP", "SYK", "VRTX", "MMC", "CB", "PLD",
    "REGN", "BDX", "CI", "NOW", "ZTS", "SO", "DUK", "CL", "SLB", "EOG",
    "CME", "PYPL", "APD", "AON", "HUM", "ICE", "ABNB", "SNAP", "NFLX", "DIS",
    "CRM", "AMD", "INTC", "MU", "MRVL", "PANW", "CRWD", "SNOW", "DDOG", "NET",
    "ENPH", "SEDG", "FSLR", "PLUG", "RIVN", "LCID", "GME", "AMC", "COIN", "HOOD",
]


def get_trading_days(start_year=2000, end_year=2025):
    """Generate realistic trading day dates (weekdays, skip major holidays)."""
    holidays_md = {
        (1, 1), (1, 15), (1, 20), (2, 17), (2, 19), (4, 10), (4, 14),
        (5, 25), (5, 27), (5, 29), (6, 19), (7, 4), (9, 1), (9, 4), (9, 7),
        (11, 24), (11, 25), (11, 26), (11, 27), (11, 28), (12, 25),
    }

    days = []
    current = datetime(start_year, 1, 3)
    end = datetime(end_year, 12, 31)

    while current <= end:
        if current.weekday() < 5 and (current.month, current.day) not in holidays_md:
            days.append(current)
        current += timedelta(days=1)

    return days


def get_regime(year):
    """Get market regime parameters for a given year."""
    for start, end, ret, vol, desc in REGIMES:
        if start <= year <= end:
            return ret, vol
    return 0.08, 0.15  # default


def generate_data():
    """Generate realistic daily market data."""
    np.random.seed(42)
    random.seed(42)

    days = get_trading_days()
    n = len(days)

    # Generate S&P 500 daily returns with regime-based parameters
    sp500_returns = np.zeros(n)
    for i, day in enumerate(days):
        annual_ret, annual_vol = get_regime(day.year)
        daily_ret = annual_ret / 252
        daily_vol = annual_vol / math.sqrt(252)

        # Fat tails via t-distribution (df=5 gives realistic kurtosis)
        z = np.random.standard_t(df=5)
        sp500_returns[i] = daily_ret + daily_vol * z * 0.7

    # Generate best stock daily returns
    # Empirically, the best S&P 500 stock on any day returns ~2-15%,
    # with occasional 20-60%+ moves (earnings, M&A, etc.)
    best_returns = np.zeros(n)
    best_tickers = []
    for i, day in enumerate(days):
        _, annual_vol = get_regime(day.year)
        vol_mult = annual_vol / 0.15  # scale with market vol

        # Base: lognormal with fat right tail
        base = np.abs(np.random.lognormal(mean=np.log(0.04 * vol_mult), sigma=0.6))
        # Occasional extreme moves
        if random.random() < 0.02:  # ~2% chance of extreme day
            base = random.uniform(0.15, 0.60)
        best_returns[i] = min(base, 0.80)  # cap at 80%
        best_tickers.append(random.choice(SP500_SAMPLE))

    # Generate worst stock daily returns
    # Mirror of best but negative, slightly less extreme
    worst_returns = np.zeros(n)
    worst_tickers = []
    for i, day in enumerate(days):
        _, annual_vol = get_regime(day.year)
        vol_mult = annual_vol / 0.15

        base = -np.abs(np.random.lognormal(mean=np.log(0.04 * vol_mult), sigma=0.6))
        if random.random() < 0.02:
            base = -random.uniform(0.15, 0.50)
        worst_returns[i] = max(base, -0.80)
        worst_tickers.append(random.choice(SP500_SAMPLE))

    # Compute cumulative portfolio values using Decimal for precision
    # The best picker compounds to astronomical values, so we use log-space
    # to avoid floating-point overflow and store as log10 values
    sp500_log = np.zeros(n)
    best_log = np.zeros(n)
    worst_log = np.zeros(n)

    sp500_log[0] = math.log10(INITIAL_INVESTMENT) + math.log10(1 + sp500_returns[0])
    best_log[0] = math.log10(INITIAL_INVESTMENT) + math.log10(1 + best_returns[0])
    worst_log[0] = math.log10(INITIAL_INVESTMENT) + math.log10(1 + worst_returns[0])

    for i in range(1, n):
        sp500_log[i] = sp500_log[i - 1] + math.log10(1 + sp500_returns[i])
        best_log[i] = best_log[i - 1] + math.log10(1 + best_returns[i])
        worst_log[i] = worst_log[i - 1] + math.log10(1 + worst_returns[i])

    print(f"Generated {n} trading days from {days[0].date()} to {days[-1].date()}")
    print(f"  S&P 500:      $10^{sp500_log[-1]:.1f}  (${10**sp500_log[-1]:,.2f})")
    print(f"  Best picker:  $10^{best_log[-1]:.1f}")
    print(f"  Worst picker: $10^{worst_log[-1]:.1f}")

    # Build output JSON — store log10 values for best/worst to handle extreme ranges
    # The frontend will use these directly on a log scale
    records = []
    for i in range(n):
        # S&P 500 stays in normal dollar values (reasonable range)
        sp_val = 10 ** sp500_log[i]

        records.append({
            "d": days[i].strftime("%Y-%m-%d"),
            "sp": round(sp_val, 2),
            "bl": round(best_log[i], 4),   # log10 of best portfolio value
            "wl": round(worst_log[i], 4),  # log10 of worst portfolio value
            "bt": best_tickers[i],
            "wt": worst_tickers[i],
            "br": round(float(best_returns[i]) * 100, 2),
            "wr": round(float(worst_returns[i]) * 100, 2),
        })

    output = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "initial_investment": INITIAL_INVESTMENT,
        "start_date": records[0]["d"],
        "end_date": records[-1]["d"],
        "total_trading_days": len(records),
        "final_values": {
            "sp500": records[-1]["sp"],
            "best_log10": records[-1]["bl"],
            "worst_log10": records[-1]["wl"],
        },
        "data": records,
    }
    return output


def main():
    DATA_DIR.mkdir(exist_ok=True)
    output = generate_data()

    out_path = DATA_DIR / "sp500_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f)

    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nOutput written to {out_path} ({file_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
