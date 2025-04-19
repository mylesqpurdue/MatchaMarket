import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def calculate_moving_average(df, period=20, price_col='close'):
    return df[price_col].rolling(window=period).mean()

def calculate_exponential_moving_average(df, period=20, price_col='close'):
    return df[price_col].ewm(span=period, adjust=False).mean()

def calculate_bollinger_bands(df, period=20, std_dev=2, price_col='close'):
    middle_band = calculate_moving_average(df, period, price_col)
    std = df[price_col].rolling(window=period).std()
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return middle_band, upper_band, lower_band

def calculate_rsi(df, period=14, price_col='close'):
    # Calculate price changes
    delta = df[price_col].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9, price_col='close'):
    # Calculate fast and slow EMAs
    fast_ema = calculate_exponential_moving_average(df, fast_period, price_col)
    slow_ema = calculate_exponential_moving_average(df, slow_period, price_col)
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def create_rsi_chart(stock_data, period=14, height=300, overbought=70, oversold=30):
    # Extract data
    df = stock_data['data']

    if isinstance(df, list):
        df = pd.DataFrame(df)

        for timecol in ('datetime', 'Datetime', 'Date', 'date', 'index'):
            if timecol in df.columns:
                df[timecol] = pd.to_datetime(df[timecol])
                df.set_index(timecol, inplace = True)
                break

    symbol = stock_data['symbol']
    
    # Calculate RSI
    rsi = calculate_rsi(df, period)
    
    # Create figure
    fig = go.Figure()
    
    # Add RSI trace
    fig.add_trace(go.Scatter(
        x=df.index,
        y=rsi,
        mode='lines',
        name=f'RSI ({period})',
        line=dict(color='#8e44ad', width=2)
    ))
    
    # Add overbought and oversold lines
    fig.add_shape(
        type="line",
        x0=df.index[0],
        y0=overbought,
        x1=df.index[-1],
        y1=overbought,
        line=dict(
            color="red",
            width=1,
            dash="dash",
        )
    )
    
    fig.add_shape(
        type="line",
        x0=df.index[0],
        y0=oversold,
        x1=df.index[-1],
        y1=oversold,
        line=dict(
            color="green",
            width=1,
            dash="dash",
        )
    )
    
    # Add middle line
    fig.add_shape(
        type="line",
        x0=df.index[0],
        y0=50,
        x1=df.index[-1],
        y1=50,
        line=dict(
            color="gray",
            width=1,
            dash="dash",
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} RSI ({period})",
        height=height,
        margin=dict(l=40, r=40, t=40, b=40),
        template='plotly_white',
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="RSI",
        yaxis=dict(
            range=[0, 100]
        )
    )
    
    return fig

def create_macd_chart(stock_data, fast_period=12, slow_period=26, signal_period=9, height=300):
    # Extract data
    df = stock_data['data']
    
    if isinstance(df, list):
        df = pd.DataFrame(df)

        for timecol in ('datetime', 'Datetime', 'Date', 'date', 'index'):
            if timecol in df.columns:
                df[timecol] = pd.to_datetime(df[timecol])
                df.set_index(timecol, inplace = True)
                break
    symbol = stock_data['symbol']
    
    # Calculate MACD
    macd_line, signal_line, histogram = calculate_macd(df, fast_period, slow_period, signal_period)
    
    # Create figure
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Bar(
        x=df.index,
        y=histogram,
        name='Histogram',
        marker_color=['#26a69a' if val >= 0 else '#ef5350' for val in histogram]
    ))
    
    # Add MACD line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=macd_line,
        mode='lines',
        name=f'MACD ({fast_period},{slow_period})',
        line=dict(color='#2c3e50', width=2)
    ))
    
    # Add signal line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=signal_line,
        mode='lines',
        name=f'Signal ({signal_period})',
        line=dict(color='#e74c3c', width=1.5)
    ))
    
    # Add zero line
    fig.add_shape(
        type="line",
        x0=df.index[0],
        y0=0,
        x1=df.index[-1],
        y1=0,
        line=dict(
            color="gray",
            width=1,
            dash="dash",
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} MACD ({fast_period},{slow_period},{signal_period})",
        height=height,
        margin=dict(l=40, r=40, t=40, b=40),
        template='plotly_white',
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="MACD"
    )
    
    return fig

def create_stochastic_oscillator(stock_data, k_period=14, d_period=3, height=300):
    # Extract data
    df = stock_data['data']
    symbol = stock_data['symbol']
    
    # Calculate Stochastic Oscillator
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    
    # Calculate %K
    k = 100 * ((df['close'] - low_min) / (high_max - low_min))
    
    # Calculate %D
    d = k.rolling(window=d_period).mean()
    
    # Create figure
    fig = go.Figure()
    
    # Add %K line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=k,
        mode='lines',
        name='%K',
        line=dict(color='#3498db', width=2)
    ))
    
    # Add %D line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=d,
        mode='lines',
        name='%D',
        line=dict(color='#e67e22', width=1.5)
    ))
    
    # Add overbought and oversold lines
    fig.add_shape(
        type="line",
        x0=df.index[0],
        y0=80,
        x1=df.index[-1],
        y1=80,
        line=dict(
            color="red",
            width=1,
            dash="dash",
        )
    )
    
    fig.add_shape(
        type="line",
        x0=df.index[0],
        y0=20,
        x1=df.index[-1],
        y1=20,
        line=dict(
            color="green",
            width=1,
            dash="dash",
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} Stochastic Oscillator ({k_period},{d_period})",
        height=height,
        margin=dict(l=40, r=40, t=40, b=40),
        template='plotly_white',
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Stochastic",
        yaxis=dict(
            range=[0, 100]
        )
    )
    
    return fig

def create_technical_dashboard(stock_data, height=800):
    # Extract data
    df = stock_data['data']
    symbol = stock_data['symbol']
    currency = stock_data['currency']
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.2, 0.15, 0.15],
        subplot_titles=(
            f"{symbol} Price",
            "Volume",
            "RSI (14)",
            "MACD (12,26,9)"
        )
    )
    
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
    fig.add_trace(candlestick, row=1, col=1)
    
    # Add Bollinger Bands
    middle_band, upper_band, lower_band = calculate_bollinger_bands(df)
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=upper_band,
            mode='lines',
            name='Upper Band',
            line=dict(width=1, color='rgba(250, 128, 114, 0.7)'),
            fill=None
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=middle_band,
            mode='lines',
            name='20-day MA',
            line=dict(width=1.5, color='rgba(184, 134, 11, 0.7)')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=lower_band,
            mode='lines',
            name='Lower Band',
            line=dict(width=1, color='rgba(173, 216, 230, 0.7)'),
            fill='tonexty'
        ),
        row=1, col=1
    )
    
    # Add volume trace
    colors = ['#26a69a' if row['close'] >= row['open'] else '#ef5350' for _, row in df.iterrows()]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Add RSI
    rsi = calculate_rsi(df)
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=rsi,
            mode='lines',
            name='RSI (14)',
            line=dict(color='#8e44ad', width=2)
        ),
        row=3, col=1
    )
    
    # Add RSI reference lines
    fig.add_shape(
        type="line",
        x0=df.index[0],
        y0=70,
        x1=df.index[-1],
        y1=70,
        line=dict(
            color="red",
            width=1,
            dash="dash",
        ),
        row=3, col=1
    )
    
    fig.add_shape(
        type="line",
        x0=df.index[0],
        y0=30,
        x1=df.index[-1],
        y1=30,
        line=dict(
            color="green",
            width=1,
            dash="dash",
        ),
        row=3, col=1
    )
    
    # Add MACD
    macd_line, signal_line, histogram = calculate_macd(df)
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=histogram,
            name='Histogram',
            marker_color=['#26a69a' if val >= 0 else '#ef5350' for val in histogram]
        ),
        row=4, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=macd_line,
            mode='lines',
            name='MACD (12,26)',
            line=dict(color='#2c3e50', width=2)
        ),
        row=4, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=signal_line,
            mode='lines',
            name='Signal (9)',
            line=dict(color='#e74c3c', width=1.5)
        ),
        row=4, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} Technical Analysis Dashboard",
        height=height,
        margin=dict(l=40, r=40, t=40, b=40),
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text=f"Price ({currency})" if currency else "Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    
    # Update x-axis labels
    fig.update_xaxes(title_text="Date", row=4, col=1)
    
    # Remove rangeslider
    fig.update_layout(xaxis_rangeslider_visible=False)
    
    return fig
