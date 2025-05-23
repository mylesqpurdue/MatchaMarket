import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_candlestick_chart(stock_data, title=None, height=600, show_volume=True):
    # Extract data
    df = stock_data['data']
    symbol = stock_data['symbol']
    currency = stock_data['currency']

    if 'date' in df.columns:
        df.index = pd.to_datetime(df['date'])
    else:
        df.index = pd.to_datetime(df.index)
    
    # Set up figure with secondary y-axis for volume if needed
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.8, 0.2]
        )
    else:
        fig = go.Figure()
    
    # Add candlestick trace
    candlestick = go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price',
        increasing_line_color='#26a69a', 
        decreasing_line_color='#ef5350'
    )
    
    if show_volume:
        fig.add_trace(candlestick, row=1, col=1)
    else:
        fig.add_trace(candlestick)
    
    # Add volume trace if requested
    if show_volume and 'volume' in df.columns:
        # Calculate colors for volume bars based on price movement
        colors = ['#26a69a' if row['close'] >= row['open'] else '#ef5350' for _, row in df.iterrows()]
        
        volume = go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        )
        fig.add_trace(volume, row=2, col=1)
    
    # Set chart title
    if title:
        chart_title = title
    else:
        chart_title = f"{symbol} Stock Price"
        if currency:
            chart_title += f" ({currency})"
    
    # Update layout
    fig.update_layout(
        title=chart_title,
        height=height,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    if show_volume:
        fig.update_yaxes(title_text="Price",  row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    else:
        fig.update_yaxes(title_text="Price")
    
     # … everything up to just before the updatemenus block …

    # Compute safe look‐back indices
    n = len(df)
    start_1m  = df.index[max(n - 30,  0)]
    start_3m  = df.index[max(n - 90,  0)]
    start_6m  = df.index[max(n - 180, 0)]
    # YTD is always Jan 1 of the first year
    ytd_year  = df.index[0].year
    start_ytd = df.index[0].replace(year=ytd_year, month=1, day=1)
    start_1y  = df.index[max(n - 365, 0)]

    # Now build the buttons using those safe values
    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "direction": "right",
            "active": 0,
            "x": 0.1,
            "y": 1.1,
            "buttons": [
                {
                    "label": "1M",
                    "method": "relayout",
                    "args": [{"xaxis.range": [start_1m,   df.index[-1]]}]
                },
                {
                    "label": "3M",
                    "method": "relayout",
                    "args": [{"xaxis.range": [start_3m,   df.index[-1]]}]
                },
                {
                    "label": "6M",
                    "method": "relayout",
                    "args": [{"xaxis.range": [start_6m,   df.index[-1]]}]
                },
                {
                    "label": "YTD",
                    "method": "relayout",
                    "args": [{"xaxis.range": [start_ytd,  df.index[-1]]}]
                },
                {
                    "label": "1Y",
                    "method": "relayout",
                    "args": [{"xaxis.range": [start_1y,   df.index[-1]]}]
                },
                {
                    "label": "All",
                    "method": "relayout",
                    "args": [{"xaxis.range": [df.index[0], df.index[-1]]}]
                }
            ]
        }]
    )

    return fig

def create_line_chart(stock_data, title=None, height=500, use_adjclose=True):

    # Extract data
    df = stock_data['data']
    symbol = stock_data['symbol']
    currency = stock_data['currency']
    
    # Create figure
    fig = go.Figure()
    
    # Determine which price column to use
    price_col = 'adjclose' if use_adjclose and 'adjclose' in df.columns else 'close'
    
    # Add line trace
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[price_col],
        mode='lines',
        name=f'{symbol} {price_col}',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Set chart title
    if title:
        chart_title = title
    else:
        chart_title = f"{symbol} Stock Price"
        if currency:
            chart_title += f" ({currency})"
    
    # Update layout
    fig.update_layout(
        title=chart_title,
        height=height,
        margin=dict(l=40, r=40, t=40, b=40),
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title=f"Price ({currency})" if currency else "Price"
    )
    
    return fig

def create_comparison_chart(stock_data_dict, title="Stock Price Comparison", height=500, use_adjclose=True, normalize=True):
    # Create figure
    fig = go.Figure()
    
    # Color palette for multiple lines
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Process each stock
    for i, (symbol, stock_data) in enumerate(stock_data_dict.items()):
        # Extract data
        df = stock_data['data']
        
        # Determine which price column to use
        price_col = 'adjclose' if use_adjclose and 'adjclose' in df.columns else 'close'
        
        # Normalize data if requested
        if normalize:
            # Calculate percentage change from first value
            first_value = df[price_col].iloc[0]
            y_values = (df[price_col] / first_value - 1) * 100
            y_axis_title = "Percentage Change (%)"
        else:
            y_values = df[price_col]
            currency = stock_data.get('currency', '')
            y_axis_title = f"Price ({currency})" if currency else "Price"
        
        # Add line trace
        fig.add_trace(go.Scatter(
            x=df.index,
            y=y_values,
            mode='lines',
            name=symbol,
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=40, r=40, t=40, b=40),
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title=y_axis_title
    )
    
    return fig

def add_moving_averages(fig, stock_data, periods=[20, 50, 200], row=1, col=1):
    # Extract data
    df = stock_data['data']
    
    # Add moving average traces
    for period in periods:
        # Calculate moving average
        ma_col = f'MA{period}'
        df[ma_col] = df['close'].rolling(window=period).mean()
        
        # Add trace
        ma_trace = go.Scatter(
            x=df.index,
            y=df[ma_col],
            mode='lines',
            name=f'{period}-day MA',
            line=dict(width=1.5)
        )
        
        # Add to figure
        if hasattr(fig, 'rows') and hasattr(fig, 'cols'):
            fig.add_trace(ma_trace, row=row, col=col)
        else:
            fig.add_trace(ma_trace)
    
    return fig

def add_bollinger_bands(fig, stock_data, period=20, std_dev=2, row=1, col=1):
    # Extract data
    df = stock_data['data']
    
    # Calculate Bollinger Bands
    df['MA'] = df['close'].rolling(window=period).mean()
    df['STD'] = df['close'].rolling(window=period).std()
    df['Upper'] = df['MA'] + (df['STD'] * std_dev)
    df['Lower'] = df['MA'] - (df['STD'] * std_dev)
    
    # Add upper band
    upper_trace = go.Scatter(
        x=df.index,
        y=df['Upper'],
        mode='lines',
        name='Upper Band',
        line=dict(width=1, color='rgba(250, 128, 114, 0.7)'),
        fill=None
    )
    
    # Add moving average
    ma_trace = go.Scatter(
        x=df.index,
        y=df['MA'],
        mode='lines',
        name=f'{period}-day MA',
        line=dict(width=1.5, color='rgba(184, 134, 11, 0.7)')
    )
    
    # Add lower band
    lower_trace = go.Scatter(
        x=df.index,
        y=df['Lower'],
        mode='lines',
        name='Lower Band',
        line=dict(width=1, color='rgba(173, 216, 230, 0.7)'),
        fill='tonexty'  # Fill area between lower and upper bands
    )
    
    # Add traces to figure
    if hasattr(fig, 'rows') and hasattr(fig, 'cols'):
        fig.add_trace(upper_trace, row=row, col=col)
        fig.add_trace(ma_trace, row=row, col=col)
        fig.add_trace(lower_trace, row=row, col=col)
    else:
        fig.add_trace(upper_trace)
        fig.add_trace(ma_trace)
        fig.add_trace(lower_trace)
    
    return fig
