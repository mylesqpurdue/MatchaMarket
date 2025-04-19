# Stock Market Dashboard

A Python-based stock market dashboard that fetches real-time data, displays interactive charts, and uses XGBoost for price predictions.

## Features

- Real-time stock data fetching from Yahoo Finance API
- Interactive price charts with candlestick and line chart options
- Volume visualization with color-coded bars
- Technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
- Stock watchlist to track multiple stocks
- Stock comparison functionality
- XGBoost-based price prediction with confidence intervals
- Customizable timeframes and data display options
- Responsive design for desktop and mobile

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/stock-dashboard.git
cd stock-dashboard
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate or \venv\Scripts\Activate.ps1
```

3. Install dependencies:
```bash
pip install -r requirements.txt
and another other depednacies need to run
```

## Usage

1. Start the dashboard:
```bash
python app.py
```

2. Open your browser and go to:
```
http://localhost:8050 
```

## Dashboard Components

- **Stock Selector**: Choose from popular stocks or enter a custom stock symbol
- **Timeframe Selector**: Select different time ranges (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, YTD, Max)
- **Chart Type**: Switch between candlestick and line charts
- **Technical Indicators**: Toggle different indicators
- **Watchlist**: Track multiple stocks and see their current prices and changes
- **Prediction**: Train XGBoost models and predict future stock prices

## Prediction Feature

The dashboard includes an XGBoost-based prediction feature that:
- Trains models on historical stock data
- Creates features from technical indicators
- Predicts future stock prices with confidence intervals
- Displays model performance metrics

## Project Structure

- `app.py`: Main application file
- `data/`: Data fetching and processing modules
  - `fetcher.py`: Stock data fetching with error handling
  - `predictor.py`: XGBoost prediction model
  - `test_metrics`: Tests RSME 
- `components/`: UI components
  - `price_chart.py`: Price chart visualizations
  - `volume_chart.py`: Volume chart visualizations
  - `indicators.py`: Technical indicators
  - `stock_selector.py`: Stock selection UI
  - `watchlist.py`: Watchlist component
  - `prediction.py`: Prediction UI component
- `layouts/`: Layout components
  - `main_layout.py`: Main dashboard layout
  - `responsive.py`: Responsive design utilities
- `assets/`: Static assets
  - `styles.css`: CSS styles
- `models/`: Directory for storing trained models

## Dependencies

- dash
- dash-bootstrap-components
- plotly
- pandas
- numpy
- xgboost
- scikit-learn
- requests

## License

MIT

## Acknowledgments

- Yahoo Finance API for stock data
- Dash and Plotly for interactive visualizations
- XGBoost for machine learning predictions
- All of my friends
