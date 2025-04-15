"""
Volume chart component for the Stock Market Dashboard.
This module provides functions to create interactive volume charts using Plotly.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_volume_chart(stock_data, title=None, height=300, standalone=True):
    """
    Create an interactive volume chart for stock data.
    
    Args:
        stock_data (dict): Processed stock data from the fetcher module
        title (str, optional): Chart title
        height (int, optional): Chart height in pixels
        standalone (bool, optional): Whether this is a standalone chart or part of another chart
        
    Returns:
        plotly.graph_objects.Figure: Interactive volume chart
    """
    # Extract data
    df = stock_data['data']
    symbol = stock_data['symbol']
    
    # Create figure
    if standalone:
        fig = go.Figure()
    else:
        fig = make_subplots(rows=1, cols=1)
    
    # Calculate colors for volume bars based on price movement
    colors = ['#26a69a' if row['close'] >= row['open'] else '#ef5350' for _, row in df.iterrows()]
    
    # Add volume trace
    volume = go.Bar(
        x=df.index,
        y=df['volume'],
        name='Volume',
        marker_color=colors,
        opacity=0.7
    )
    
    if standalone:
        fig.add_trace(volume)
    else:
        fig.add_trace(volume, row=1, col=1)
    
    # Set chart title
    if standalone and title:
        chart_title = title
    elif standalone:
        chart_title = f"{symbol} Trading Volume"
    else:
        chart_title = None
    
    # Update layout
    if standalone:
        fig.update_layout(
            title=chart_title,
            height=height,
            margin=dict(l=40, r=40, t=40, b=40),
            template='plotly_white',
            hovermode="x unified",
            xaxis_title="Date",
            yaxis_title="Volume"
        )
    
    return fig

def create_volume_profile(stock_data, title=None, height=500, bins=50):
    """
    Create a volume profile chart showing volume distribution by price level.
    
    Args:
        stock_data (dict): Processed stock data from the fetcher module
        title (str, optional): Chart title
        height (int, optional): Chart height in pixels
        bins (int, optional): Number of price bins for volume distribution
        
    Returns:
        plotly.graph_objects.Figure: Volume profile chart
    """
    # Extract data
    df = stock_data['data']
    symbol = stock_data['symbol']
    currency = stock_data['currency']
    
    # Create figure with secondary y-axis
    fig = make_subplots(rows=1, cols=2, column_widths=[0.8, 0.2], shared_yaxes=True)
    
    # Add price line to first subplot
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['close'],
            mode='lines',
            name='Price',
            line=dict(color='#1f77b4', width=2)
        ),
        row=1, col=1
    )
    
    # Calculate volume profile
    price_min = df['low'].min()
    price_max = df['high'].max()
    price_range = price_max - price_min
    bin_size = price_range / bins
    
    # Create price bins
    price_bins = [price_min + i * bin_size for i in range(bins + 1)]
    volume_by_price = np.zeros(bins)
    
    # Distribute volume into price bins
    for _, row in df.iterrows():
        # Determine which bins this candle spans
        low_bin = max(0, int((row['low'] - price_min) / bin_size))
        high_bin = min(bins - 1, int((row['high'] - price_min) / bin_size))
        
        # Distribute volume proportionally across bins
        if high_bin == low_bin:
            volume_by_price[low_bin] += row['volume']
        else:
            span = high_bin - low_bin + 1
            for i in range(low_bin, high_bin + 1):
                volume_by_price[i] += row['volume'] / span
    
    # Calculate bin centers for plotting
    bin_centers = [price_min + (i + 0.5) * bin_size for i in range(bins)]
    
    # Add volume profile to second subplot
    fig.add_trace(
        go.Bar(
            x=volume_by_price,
            y=bin_centers,
            orientation='h',
            name='Volume Profile',
            marker_color='rgba(158, 202, 225, 0.6)',
            opacity=0.8
        ),
        row=1, col=2
    )
    
    # Set chart title
    if title:
        chart_title = title
    else:
        chart_title = f"{symbol} Price and Volume Profile"
        if currency:
            chart_title += f" ({currency})"
    
    # Update layout
    fig.update_layout(
        title=chart_title,
        height=height,
        margin=dict(l=40, r=40, t=40, b=40),
        template='plotly_white',
        hovermode="x unified",
        showlegend=False
    )
    
    # Update axes
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Volume", row=1, col=2)
    fig.update_yaxes(title_text=f"Price ({currency})" if currency else "Price", row=1, col=1)
    
    return fig

def create_volume_by_price_chart(stock_data, title=None, height=400, bins=20):
    """
    Create a horizontal bar chart showing volume by price level.
    
    Args:
        stock_data (dict): Processed stock data from the fetcher module
        title (str, optional): Chart title
        height (int, optional): Chart height in pixels
        bins (int, optional): Number of price bins for volume distribution
        
    Returns:
        plotly.graph_objects.Figure: Volume by price chart
    """
    # Extract data
    df = stock_data['data']
    symbol = stock_data['symbol']
    currency = stock_data['currency']
    
    # Create figure
    fig = go.Figure()
    
    # Calculate volume by price
    price_min = df['low'].min()
    price_max = df['high'].max()
    price_range = price_max - price_min
    bin_size = price_range / bins
    
    # Create price bins
    price_bins = [price_min + i * bin_size for i in range(bins + 1)]
    volume_by_price = np.zeros(bins)
    
    # Distribute volume into price bins
    for _, row in df.iterrows():
        # Determine which bins this candle spans
        low_bin = max(0, int((row['low'] - price_min) / bin_size))
        high_bin = min(bins - 1, int((row['high'] - price_min) / bin_size))
        
        # Distribute volume proportionally across bins
        if high_bin == low_bin:
            volume_by_price[low_bin] += row['volume']
        else:
            span = high_bin - low_bin + 1
            for i in range(low_bin, high_bin + 1):
                volume_by_price[i] += row['volume'] / span
    
    # Calculate bin centers for plotting
    bin_centers = [price_min + (i + 0.5) * bin_size for i in range(bins)]
    
    # Format price labels
    price_labels = [f"{price:.2f}" for price in bin_centers]
    
    # Add volume by price trace
    fig.add_trace(
        go.Bar(
            x=volume_by_price,
            y=price_labels,
            orientation='h',
            name='Volume by Price',
            marker_color='rgba(58, 71, 80, 0.6)',
            opacity=0.8
        )
    )
    
    # Add current price line
    current_price = df['close'].iloc[-1]
    fig.add_shape(
        type="line",
        x0=0,
        y0=current_price,
        x1=max(volume_by_price) * 1.05,
        y1=current_price,
        line=dict(
            color="red",
            width=2,
            dash="dash",
        )
    )
    
    # Set chart title
    if title:
        chart_title = title
    else:
        chart_title = f"{symbol} Volume by Price"
        if currency:
            chart_title += f" ({currency})"
    
    # Update layout
    fig.update_layout(
        title=chart_title,
        height=height,
        margin=dict(l=40, r=40, t=40, b=40),
        template='plotly_white',
        xaxis_title="Volume",
        yaxis_title=f"Price ({currency})" if currency else "Price"
    )
    
    return fig

def create_volume_comparison_chart(stock_data_dict, title="Volume Comparison", height=400, normalize=True):
    """
    Create a chart comparing trading volumes of multiple stocks.
    
    Args:
        stock_data_dict (dict): Dictionary of processed stock data keyed by symbol
        title (str, optional): Chart title
        height (int, optional): Chart height in pixels
        normalize (bool, optional): Whether to normalize volumes to percentage of average
        
    Returns:
        plotly.graph_objects.Figure: Volume comparison chart
    """
    # Create figure
    fig = go.Figure()
    
    # Color palette for multiple lines
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Process each stock
    for i, (symbol, stock_data) in enumerate(stock_data_dict.items()):
        # Extract data
        df = stock_data['data']
        
        # Normalize data if requested
        if normalize:
            # Calculate percentage of average volume
            avg_volume = df['volume'].mean()
            y_values = (df['volume'] / avg_volume) * 100
            y_axis_title = "Percentage of Average Volume (%)"
        else:
            y_values = df['volume']
            y_axis_title = "Volume"
        
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
