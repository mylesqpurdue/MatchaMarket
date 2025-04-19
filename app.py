import os
import sys
import json
import pandas as pd
from datetime import datetime


import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

# Import dashboard components
from layouts.main_layout import create_dashboard_layout
from components.price_chart import create_candlestick_chart, create_line_chart, add_moving_averages, add_bollinger_bands
from components.volume_chart import create_volume_chart
from components.indicators import create_rsi_chart, create_macd_chart, create_stochastic_oscillator, calculate_rsi, calculate_macd
from components.prediction import register_prediction_callbacks
from data.fetcher import StockDataFetcher

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

app.title = "Stock Market Dashboard"
server = app.server

# Set the layout
app.layout = create_dashboard_layout()

# Initialize data fetcher
data_fetcher = StockDataFetcher()

# Register prediction callbacks
register_prediction_callbacks(app)

# Callback to update stock data based on user selections
@app.callback(
    Output("stock-data-store", "data"),
    [
        Input("stock-selector", "value"),
        Input("timeframe-selector", "value"),
        Input("interval-selector", "value"),
        Input("interval-component", "n_intervals")
    ]
)
def update_stock_data(symbol, timeframe, interval, n_intervals):
    if not symbol:
        symbol = "AAPL"  # Default symbol
    
    if not timeframe:
        timeframe = "1mo"  # Default timeframe
    
    if not interval:
        interval = "1d"  # Default interval
    
    # Fetch stock data
    stock_data = data_fetcher.get_stock_chart(
        symbol=symbol,
        interval=interval,
        range=timeframe
    )
    
    # Convert DataFrame to JSON-serializable format
    if stock_data and 'data' in stock_data:
        raw = stock_data['data']

        if isinstance(raw, pd.DataFrame):
            # it’s a DataFrame → reset the index and serialize
            df = raw.reset_index()
            stock_data['data'] = df.to_dict('records')
        elif isinstance(raw, list):
            # already a list of dicts → leave it as-is
            stock_data['data'] = raw
        else:
            # unexpected type: try coercing to DataFrame
            df = pd.DataFrame(raw).reset_index()
            stock_data['data'] = df.to_dict('records')

    return stock_data

# Callback to update comparison data
@app.callback(
    Output("comparison-data-store", "data"),
    [
        Input("comparison-selector", "value"),
        Input("timeframe-selector", "value"),
        Input("interval-selector", "value"),
        Input("interval-component", "n_intervals")
    ]
)

def update_comparison_data(symbols, timeframe, interval, n_intervals):
    if not symbols or not isinstance(symbols, list):
        return {}

    if not timeframe:
        timeframe = "1mo"
    if not interval:
        interval = "1d"

    comparison_data = {}
    for symbol in symbols:
        stock_data = data_fetcher.get_stock_chart(
            symbol=symbol,
            interval=interval,
            range=timeframe
        )

        if stock_data and 'data' in stock_data:
            raw = stock_data['data']

            if isinstance(raw, pd.DataFrame):
                # Reset the index and convert to list of dicts
                df = raw.reset_index()
                stock_data['data'] = df.to_dict('records')

            elif isinstance(raw, list):
                # Already JSON-serializable
                stock_data['data'] = raw

            else:
                # Fallback: coerce into a DataFrame then serialize
                df = pd.DataFrame(raw)
                stock_data['data'] = df.reset_index().to_dict('records')

        comparison_data[symbol] = stock_data

    return comparison_data


# Callback to update price chart
@app.callback(
    Output("price-chart", "figure"),
    [
        Input("stock-data-store", "data"),
        Input("comparison-data-store", "data"),
        Input("chart-type-selector", "value"),
        Input("indicator-selector", "value")
    ]
)
def update_price_chart(stock_data, comparison_data, chart_type, indicators):
    if not stock_data:
        # Return empty figure if no data
        return go.Figure()
    
    # Convert data back to DataFrame format
    df = pd.DataFrame(stock_data['data'])
    if 'datetime' in df.columns:
        df.set_index('datetime', inplace=True)
    elif 'Datetime' in df.columns:
        df.set_index('Datetime', inplace=True)
    
    # Reconstruct stock_data with DataFrame
    stock_data_with_df = stock_data.copy()
    stock_data_with_df['data'] = df
    
    # Create chart based on selected type
    if chart_type == 'line':
        fig = create_line_chart(stock_data_with_df)
    else:  # Default to candlestick
        fig = create_candlestick_chart(stock_data_with_df, show_volume=False)
    
    # Add selected indicators
    if indicators:
        if 'ma20' in indicators:
            fig = add_moving_averages(fig, stock_data_with_df, periods=[20])
        if 'ma50' in indicators:
            fig = add_moving_averages(fig, stock_data_with_df, periods=[50])
        if 'ma200' in indicators:
            fig = add_moving_averages(fig, stock_data_with_df, periods=[200])
        if 'bollinger' in indicators:
            fig = add_bollinger_bands(fig, stock_data_with_df)
    
    # Add comparison data if available
    if comparison_data and isinstance(comparison_data, dict):
        for symbol, comp_data in comparison_data.items():
            if not comp_data:
                continue
            
            # Convert data back to DataFrame format
            comp_df = pd.DataFrame(comp_data['data'])
            if 'datetime' in comp_df.columns:
                comp_df.set_index('datetime', inplace=True)
            elif 'Datetime' in comp_df.columns:
                comp_df.set_index('Datetime', inplace=True)
            
            # Add comparison line
            fig.add_trace(go.Scatter(
                x=comp_df.index,
                y=comp_df['close'],
                mode='lines',
                name=symbol,
                line=dict(width=1.5)
            ))
    
    return fig

# Callback to update volume chart
@app.callback(
    Output("volume-chart", "figure"),
    [
        Input("stock-data-store", "data"),
        Input("indicator-selector", "value")
    ]
)
def update_volume_chart(stock_data, indicators):
    if not stock_data or not indicators or 'volume' not in indicators:
        # Return empty figure if no data or volume not selected
        return go.Figure()
    
    # Convert data back to DataFrame format
    df = pd.DataFrame(stock_data['data'])
    if 'datetime' in df.columns:
        df.set_index('datetime', inplace=True)
    elif 'Datetime' in df.columns:
        df.set_index('Datetime', inplace=True)
    
    # Reconstruct stock_data with DataFrame
    stock_data_with_df = stock_data.copy()
    stock_data_with_df['data'] = df
    
    # Create volume chart
    fig = create_volume_chart(stock_data_with_df, standalone=True)
    
    return fig

# Callback to update RSI chart
@app.callback(
    Output("rsi-chart", "figure"),
    [
        Input("stock-data-store", "data"),
        Input("indicator-selector", "value")
    ]
)
def update_rsi_chart(stock_data, indicators):
    if not stock_data or not indicators or 'rsi' not in indicators:
        # Return empty figure if no data or RSI not selected
        return go.Figure()
    
    # Convert data back to DataFrame format
    df = pd.DataFrame(stock_data['data'])
    if 'datetime' in df.columns:
        df.set_index('datetime', inplace=True)
    elif 'Datetime' in df.columns:
        df.set_index('Datetime', inplace=True)
    
    # Reconstruct stock_data with DataFrame
    stock_data_with_df = stock_data.copy()
    stock_data_with_df['data'] = df
    
    # Create RSI chart
    fig = create_rsi_chart(stock_data_with_df)
    
    return fig

# Callback to update MACD chart
@app.callback(
    Output("macd-chart", "figure"),
    [
        Input("stock-data-store", "data"),
        Input("indicator-selector", "value")
    ]
)
def update_macd_chart(stock_data, indicators):
    if not stock_data or not indicators or 'macd' not in indicators:
        # Return empty figure if no data or MACD not selected
        return go.Figure()
    
    # Convert data back to DataFrame format
    df = pd.DataFrame(stock_data['data'])
    if 'datetime' in df.columns:
        df.set_index('datetime', inplace=True)
    elif 'Datetime' in df.columns:
        df.set_index('Datetime', inplace=True)
    
    # Reconstruct stock_data with DataFrame
    stock_data_with_df = stock_data.copy()
    stock_data_with_df['data'] = df
    
    # Create MACD chart
    fig = create_macd_chart(stock_data_with_df)
    
    return fig

# Callback to update stochastic chart
@app.callback(
    Output("stochastic-chart", "figure"),
    [
        Input("stock-data-store", "data"),
        Input("indicator-selector", "value")
    ]
)
def update_stochastic_chart(stock_data, indicators):
    if not stock_data:
        # Return empty figure if no data
        return go.Figure()
    
    # Convert data back to DataFrame format
    df = pd.DataFrame(stock_data['data'])
    if 'datetime' in df.columns:
        df.set_index('datetime', inplace=True)
    elif 'Datetime' in df.columns:
        df.set_index('Datetime', inplace=True)
    
    # Reconstruct stock_data with DataFrame
    stock_data_with_df = stock_data.copy()
    stock_data_with_df['data'] = df
    
    # Create stochastic chart
    fig = create_stochastic_oscillator(stock_data_with_df)
    
    return fig

# Callback to update stock info
@app.callback(
    [
        Output("stock-info-current", "children"),
        Output("stock-info-open", "children"),
        Output("stock-info-high", "children"),
        Output("stock-info-low", "children"),
        Output("stock-info-volume", "children"),
        Output("stock-info-change", "children"),
        Output("stock-info-change", "className")
    ],
    [
        Input("stock-data-store", "data")
    ]
)
def update_stock_info(stock_data):
    if not stock_data or 'data' not in stock_data or not stock_data['data']:
        return "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "info-value"
    
    # Get the latest data point
    df = pd.DataFrame(stock_data['data'])
    if df.empty:
        return "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "info-value"
    
    latest = df.iloc[-1]
    
    # Calculate change
    if 'open' in latest and 'close' in latest:
        change = latest['close'] - latest['open']
        change_pct = (change / latest['open']) * 100 if latest['open'] != 0 else 0
        change_str = f"{change:.2f} ({change_pct:.2f}%)"
        change_class = "info-value positive-change" if change >= 0 else "info-value negative-change"
    else:
        change_str = "N/A"
        change_class = "info-value"
    
    # Format volume with commas
    volume_str = f"{int(latest['volume']):,}" if 'volume' in latest else "N/A"
    
    # Get currency from stock data
    currency = stock_data.get('currency', '$')
    
    # Format values with currency
    current = f"{currency}{latest['close']:.2f}" if 'close' in latest else "N/A"
    open_val = f"{currency}{latest['open']:.2f}" if 'open' in latest else "N/A"
    high = f"{currency}{latest['high']:.2f}" if 'high' in latest else "N/A"
    low = f"{currency}{latest['low']:.2f}" if 'low' in latest else "N/A"
    
    return current, open_val, high, low, volume_str, change_str, change_class

# Callback to update last updated time
@app.callback(
    Output("last-updated-time", "children"),
    [
        Input("interval-component", "n_intervals")
    ]
)
def update_last_updated_time(n_intervals):
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

# Callback to add stock to watchlist
@app.callback(
    Output("stock-selector", "options"),
    [
        Input("add-stock-button", "n_clicks")
    ],
    [
        State("stock-input", "value"),
        State("stock-selector", "options")
    ]
)
def add_stock_to_watchlist(n_clicks, stock_symbol, current_options):
    if n_clicks is None or not stock_symbol:
        return current_options
    
    # Check if stock already exists in options
    if any(opt['value'] == stock_symbol.upper() for opt in current_options):
        return current_options
    
    # Add new stock to options
    new_options = current_options.copy()
    new_options.append({'label': stock_symbol.upper(), 'value': stock_symbol.upper()})
    
    return new_options

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
