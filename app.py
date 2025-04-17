"""
Main application file for the Stock Market Dashboard.
This module integrates all components and handles the application logic.
"""

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
    """
    Fetch and store stock data based on user selections.
    
    Args:
        symbol (str): Stock symbol
        timeframe (str): Selected timeframe
        interval (str): Selected interval
        n_intervals (int): Number of refresh intervals
        
    Returns:
        dict: JSON-serialized stock data
    """
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
        # Convert DataFrame index to string for JSON serialization
        df_dict = stock_data['data'].reset_index().to_dict('records')
        stock_data['data'] = df_dict
        raw = stock_data['data']
        if isinstance(raw, list):
            # Already a list of dicts: assume it’s JSON-serializable already
            df_dict = raw
        else:
            # It’s still a DataFrame: reset_index and serialize
            df = raw.reset_index()
            df_dict = df.to_dict('records')
        stock_data['data'] = df_dict
    
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
    """
    Fetch and store comparison stock data.
    
    Args:
        symbols (list): List of stock symbols for comparison
        timeframe (str): Selected timeframe
        interval (str): Selected interval
        n_intervals (int): Number of refresh intervals
        
    Returns:
        dict: JSON-serialized comparison data
    """
    if not symbols or not isinstance(symbols, list):
        return {}
    
    if not timeframe:
        timeframe = "1mo"  # Default timeframe
    
    if not interval:
        interval = "1d"  # Default interval
    
    # Fetch data for each comparison symbol
    comparison_data = {}
    for symbol in symbols:
        stock_data = data_fetcher.get_stock_chart(
            symbol=symbol,
            interval=interval,
            range=timeframe
        )
        
        # Convert DataFrame to JSON-serializable format
        if stock_data and 'data' in stock_data:
            # Convert DataFrame index to string for JSON serialization
            df_dict = stock_data['data'].reset_index().to_dict('records')
            stock_data['data'] = df_dict
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
    """
    Update the price chart based on stock data and user selections.
    
    Args:
        stock_data (dict): Stock data from data store
        comparison_data (dict): Comparison data from data store
        chart_type (str): Selected chart type
        indicators (list): Selected technical indicators
        
    Returns:
        plotly.graph_objects.Figure: Updated price chart
    """
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
    """
    Update the volume chart based on stock data.
    
    Args:
        stock_data (dict): Stock data from data store
        indicators (list): Selected technical indicators
        
    Returns:
        plotly.graph_objects.Figure: Updated volume chart
    """
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
    """
    Update the RSI chart based on stock data.
    
    Args:
        stock_data (dict): Stock data from data store
        indicators (list): Selected technical indicators
        
    Returns:
        plotly.graph_objects.Figure: Updated RSI chart
    """
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
    """
    Update the MACD chart based on stock data.
    
    Args:
        stock_data (dict): Stock data from data store
        indicators (list): Selected technical indicators
        
    Returns:
        plotly.graph_objects.Figure: Updated MACD chart
    """
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
    """
    Update the stochastic oscillator chart based on stock data.
    
    Args:
        stock_data (dict): Stock data from data store
        indicators (list): Selected technical indicators
        
    Returns:
        plotly.graph_objects.Figure: Updated stochastic chart
    """
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
    """
    Update the stock information card.
    
    Args:
        stock_data (dict): Stock data from data store
        
    Returns:
        tuple: Updated stock information values and styles
    """
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
    """
    Update the last updated time display.
    
    Args:
        n_intervals (int): Number of refresh intervals
        
    Returns:
        str: Formatted current time
    """
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
    """
    Add a stock to the watchlist dropdown.
    
    Args:
        n_clicks (int): Number of button clicks
        stock_symbol (str): Stock symbol to add
        current_options (list): Current dropdown options
        
    Returns:
        list: Updated dropdown options
    """
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
