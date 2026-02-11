# S&P 500 Stock Picker

What if you had a crystal ball and always picked the best-performing S&P 500 stock each trading day? This visualization compares three strategies starting with $10,000:

- **S&P 500 Index** — simple buy-and-hold indexing
- **Best Stock Each Day** — the hypothetical "perfect picker"
- **Worst Stock Each Day** — always picking the single worst performer

## Live Demo

Visit the [GitHub Pages site](https://philipcondie.github.io/stockpicker/) to explore the interactive chart.

## How It Works

1. **Data pipeline** (`scripts/fetch_and_compute.py`) — fetches historical daily prices for all S&P 500 stocks via [yfinance](https://github.com/ranaroussi/yfinance), computes daily best/worst returns, and outputs pre-computed JSON
2. **Visualization** (`index.html`) — interactive Plotly.js chart loaded from the static JSON data

The chart uses a log scale because the "perfect picker" compounds to truly astronomical values — that's the whole point. It makes the scale of indexing vs. stock-picking viscerally clear.

## Running the Data Pipeline

To refresh with real market data:

```bash
pip install -r requirements.txt
python scripts/fetch_and_compute.py
```

This downloads data for ~500 tickers from Yahoo Finance (takes a few minutes) and writes `data/sp500_analysis.json`.

## Features

- Interactive Plotly.js chart with hover details (daily pick, return %)
- Adjustable start year to rebase from any year (2000-2025)
- Toggle individual series on/off
- Log-scale y-axis with readable dollar tick labels
- Dark theme

## Caveats

- Uses current S&P 500 constituents applied historically (survivorship bias)
- No transaction costs or taxes
- Dividends included via adjusted prices
- This is purely hypothetical — no one can predict the best stock each day
