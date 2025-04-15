# Stock Market Dashboard - User Guide

## Overview

The Stock Market Dashboard is a Python-based web application that allows you to:

- View real-time and historical stock data
- Analyze stock performance with interactive charts
- Track multiple stocks in a watchlist
- Compare different stocks
- Visualize technical indicators like RSI, MACD, and Bollinger Bands
- Customize time frames and data display options

## Getting Started

### Running the Dashboard

1. Navigate to the stock_dashboard directory
2. Activate the virtual environment:
   ```
   source venv/bin/activate
   ```
3. Run the dashboard:
   ```
   python app.py
   ```
4. Open your browser and go to: http://localhost:8050

## Dashboard Features

### Main Components

1. **Stock Selector**: Choose from popular stocks or enter a custom stock symbol
2. **Timeframe Selector**: Select different time ranges (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, YTD, Max)
3. **Interval Selector**: Choose data granularity (1m, 2m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo)
4. **Chart Type Selector**: Switch between candlestick and line charts
5. **Technical Indicators**: Toggle different indicators (Moving Averages, Bollinger Bands, RSI, MACD)
6. **Watchlist**: Track multiple stocks and see their current prices and changes
7. **Comparison Tool**: Compare performance of multiple stocks on the same chart

### Price Chart

The main price chart displays stock price data in your chosen format (candlestick or line). You can:

- Zoom in/out using the chart controls
- Pan across different time periods
- Hover over data points to see detailed information
- Add technical indicators as overlays
- Download the chart as an image

### Technical Indicators

The dashboard includes several technical indicators:

- **Moving Averages (20, 50, 200-day)**: Show average price trends
- **Bollinger Bands**: Display price volatility with upper and lower bands
- **RSI (Relative Strength Index)**: Measure overbought or oversold conditions
- **MACD (Moving Average Convergence Divergence)**: Identify trend changes and momentum
- **Stochastic Oscillator**: Another momentum indicator for overbought/oversold conditions

### Volume Chart

The volume chart shows trading volume over time, with colors indicating price movement:
- Green bars: Price increased during that period
- Red bars: Price decreased during that period

### Watchlist

The watchlist allows you to track multiple stocks at once:

- Add stocks by entering their symbol and clicking "Add"
- Remove stocks by clicking the "×" button
- View current price and daily change for each stock
- Refresh data manually with the "Refresh" button

### Stock Comparison

To compare multiple stocks:

1. Select your primary stock in the Stock Selector
2. Add comparison stocks in the "Compare With" dropdown
3. The chart will display normalized performance to show relative changes

## Tips for Effective Use

1. **For day trading**: Use shorter timeframes (1d) with smaller intervals (1m, 5m)
2. **For long-term analysis**: Use longer timeframes (1y, 5y) with daily or weekly intervals
3. **For technical analysis**: Enable relevant indicators like RSI, MACD, and Bollinger Bands
4. **For performance comparison**: Add benchmark indices like ^GSPC (S&P 500) or ^DJI (Dow Jones)
5. **For portfolio tracking**: Add all your holdings to the watchlist for quick monitoring

## Troubleshooting

- If a stock symbol doesn't load, verify it's a valid symbol on Yahoo Finance
- For real-time data, ensure you have a stable internet connection
- If charts appear empty, try changing the timeframe or interval
- If the dashboard becomes slow, reduce the number of comparison stocks or indicators

## Data Sources

All stock data is provided by Yahoo Finance API, with data refreshing automatically every minute.
