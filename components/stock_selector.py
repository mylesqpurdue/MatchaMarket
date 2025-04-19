import plotly.graph_objects as go
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

def create_stock_selector(id='stock-selector', default_value='AAPL', popular_stocks=None):
    if popular_stocks is None:
        popular_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'JNJ']
    
    return html.Div([
        html.Label('Select Stock:'),
        dcc.Dropdown(
            id=id,
            options=[{'label': symbol, 'value': symbol} for symbol in popular_stocks],
            value=default_value,
            clearable=False,
            className='stock-dropdown'
        ),
        html.Div([
            dbc.Input(
                id='stock-input',
                type='text',
                placeholder='Enter stock symbol...',
                className='stock-input'
            ),
            dbc.Button(
                'Add',
                id='add-stock-button',
                color='primary',
                className='add-button'
            )
        ], className='stock-input-container')
    ], className='stock-selector-container')

def create_timeframe_selector(id='timeframe-selector', default_value='1mo'):
    timeframes = [
        {'label': '1 Day', 'value': '1d'},
        {'label': '5 Days', 'value': '5d'},
        {'label': '1 Month', 'value': '1mo'},
        {'label': '3 Months', 'value': '3mo'},
        {'label': '6 Months', 'value': '6mo'},
        {'label': '1 Year', 'value': '1y'},
        {'label': '2 Years', 'value': '2y'},
        {'label': '5 Years', 'value': '5y'},
        {'label': 'Year to Date', 'value': 'ytd'},
        {'label': 'Max', 'value': 'max'}
    ]
    
    return html.Div([
        html.Label('Select Timeframe:'),
        dcc.Dropdown(
            id=id,
            options=timeframes,
            value=default_value,
            clearable=False,
            className='timeframe-dropdown'
        )
    ], className='timeframe-selector-container')

def create_interval_selector(id='interval-selector', default_value='1d'):
    intervals = [
        {'label': '1 Minute', 'value': '1m'},
        {'label': '2 Minutes', 'value': '2m'},
        {'label': '5 Minutes', 'value': '5m'},
        {'label': '15 Minutes', 'value': '15m'},
        {'label': '30 Minutes', 'value': '30m'},
        {'label': '60 Minutes', 'value': '60m'},
        {'label': '1 Day', 'value': '1d'},
        {'label': '1 Week', 'value': '1wk'},
        {'label': '1 Month', 'value': '1mo'}
    ]
    
    return html.Div([
        html.Label('Select Interval:'),
        dcc.Dropdown(
            id=id,
            options=intervals,
            value=default_value,
            clearable=False,
            className='interval-dropdown'
        )
    ], className='interval-selector-container')

def create_chart_type_selector(id='chart-type-selector', default_value='candlestick'):
    chart_types = [
        {'label': 'Candlestick', 'value': 'candlestick'},
        {'label': 'Line', 'value': 'line'},
        {'label': 'OHLC', 'value': 'ohlc'}
    ]
    
    return html.Div([
        html.Label('Chart Type:'),
        dcc.Dropdown(
            id=id,
            options=chart_types,
            value=default_value,
            clearable=False,
            className='chart-type-dropdown'
        )
    ], className='chart-type-selector-container')

def create_indicator_selector(id='indicator-selector'):
    indicators = [
        {'label': 'Moving Average (20)', 'value': 'ma20'},
        {'label': 'Moving Average (50)', 'value': 'ma50'},
        {'label': 'Moving Average (200)', 'value': 'ma200'},
        {'label': 'Bollinger Bands', 'value': 'bollinger'},
        {'label': 'RSI', 'value': 'rsi'},
        {'label': 'MACD', 'value': 'macd'},
        {'label': 'Volume', 'value': 'volume'}
    ]
    
    return html.Div([
        html.Label('Technical Indicators:'),
        dbc.Checklist(
            id=id,
            options=indicators,
            value=['volume'],
            inline=False,
            className='indicator-checklist'
        )
    ], className='indicator-selector-container')

def create_comparison_selector(id='comparison-selector', popular_stocks=None):
    if popular_stocks is None:
        popular_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'JNJ', '^GSPC', '^DJI', '^IXIC']
    
    return html.Div([
        html.Label('Compare With:'),
        dcc.Dropdown(
            id=id,
            options=[{'label': symbol, 'value': symbol} for symbol in popular_stocks],
            value=[],
            multi=True,
            className='comparison-dropdown'
        )
    ], className='comparison-selector-container')

def create_control_panel():
    return dbc.Card([
        dbc.CardHeader("Dashboard Controls"),
        dbc.CardBody([
            create_stock_selector(),
            create_timeframe_selector(),
            create_interval_selector(),
            create_chart_type_selector(),
            create_indicator_selector(),
            create_comparison_selector()
        ])
    ], className='control-panel-card')

def create_watchlist(stocks=None):
    if stocks is None:
        stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    return dbc.Card([
        dbc.CardHeader("Watchlist"),
        dbc.CardBody([
            html.Div(id='watchlist-content', className='watchlist-content'),
            dbc.Button(
                'Refresh',
                id='refresh-watchlist-button',
                color='secondary',
                className='refresh-button'
            )
        ])
    ], className='watchlist-card')
