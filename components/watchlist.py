import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from data.fetcher import StockDataFetcher

class StockWatchlist:
    """Class to manage a stock watchlist."""
    
    def __init__(self):
        """Initialize the watchlist with default stocks."""
        self.default_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        self.stocks = self.default_stocks.copy()
        self.data_fetcher = StockDataFetcher()
    
    def add_stock(self, symbol):
        if not symbol or symbol.upper() in self.stocks:
            return False
        
        # Try to fetch data to validate the symbol
        try:
            data = self.data_fetcher.get_stock_chart(symbol=symbol.upper(), interval='1d', range='1d')
            if data:
                self.stocks.append(symbol.upper())
                return True
        except Exception:
            return False
        
        return False
    
    def remove_stock(self, symbol):
        if symbol in self.stocks:
            self.stocks.remove(symbol)
            return True
        return False
    
    def get_stocks(self):
        return self.stocks
    
    def get_watchlist_data(self):
        data = {}
        for symbol in self.stocks:
            try:
                stock_data = self.data_fetcher.get_stock_chart(symbol=symbol, interval='1d', range='1d')
                if stock_data:
                    data[symbol] = stock_data
            except Exception:
                continue
        
        return data

def create_watchlist_item(symbol, price, change, change_pct):
    # Determine color based on change
    color = "success" if change >= 0 else "danger"
    change_icon = "▲" if change >= 0 else "▼"
    
    return dbc.ListGroupItem([
        dbc.Row([
            # Symbol
            dbc.Col(html.H5(symbol), width=3),
            # Price
            dbc.Col(html.Div(f"${price:.2f}"), width=3),
            # Change
            dbc.Col(html.Div([
                html.Span(f"{change_icon} {abs(change):.2f} "),
                html.Span(f"({abs(change_pct):.2f}%)"),
            ], className=f"text-{color}"), width=4),
            # Remove button
            dbc.Col(
                dbc.Button("×", id=f"remove-{symbol}", color="link", size="sm", className="p-0"),
                width=2,
                className="text-right"
            )
        ])
    ], className="watchlist-item")

def create_watchlist_content(watchlist_data):
    if not watchlist_data:
        return html.Div("No stocks in watchlist", className="text-muted")
    
    items = []
    for symbol, data in watchlist_data.items():
        if not data or 'data' not in data or not data['data'].shape[0]:
            continue
        
        # Get latest data
        latest = data['data'].iloc[-1]
        
        # Calculate change
        if 'open' in latest and 'close' in latest:
            change = latest['close'] - latest['open']
            change_pct = (change / latest['open']) * 100 if latest['open'] != 0 else 0
        else:
            change = 0
            change_pct = 0
        
        # Create watchlist item
        items.append(create_watchlist_item(symbol, latest['close'], change, change_pct))
    
    return dbc.ListGroup(items, className="watchlist-items")

def create_watchlist_card():
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Watchlist", className="d-inline"),
            dbc.Button(
                "Refresh",
                id="refresh-watchlist-button",
                color="link",
                size="sm",
                className="float-right"
            )
        ]),
        dbc.CardBody([
            # Watchlist content will be updated by callback
            html.Div(id="watchlist-content"),
            
            # Add stock form
            html.Hr(),
            dbc.InputGroup([
                dbc.Input(
                    id="watchlist-input",
                    placeholder="Add symbol...",
                    type="text"
                ),
                dbc.InputGroupAddon(
                    dbc.Button("Add", id="add-to-watchlist-button", color="primary"),
                    addon_type="append"
                )
            ], className="mt-3")
        ])
    ], className="watchlist-card")

def register_watchlist_callbacks(app):
    # Initialize watchlist
    watchlist = StockWatchlist()
    
    # Callback to update watchlist content
    @app.callback(
        Output("watchlist-content", "children"),
        [
            Input("refresh-watchlist-button", "n_clicks"),
            Input("add-to-watchlist-button", "n_clicks"),
            Input("interval-component", "n_intervals")
        ],
        [
            State("watchlist-input", "value")
        ]
    )
    def update_watchlist(refresh_clicks, add_clicks, n_intervals, new_symbol):

        ctx = dash.callback_context
        
        # Check which input triggered the callback
        if ctx.triggered:
            input_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            # Add new stock if add button was clicked
            if input_id == "add-to-watchlist-button" and new_symbol:
                watchlist.add_stock(new_symbol)
        
        # Get watchlist data
        watchlist_data = watchlist.get_watchlist_data()
        
        # Create watchlist content
        return create_watchlist_content(watchlist_data)
    
    # Callback to clear input after adding
    @app.callback(
        Output("watchlist-input", "value"),
        [
            Input("add-to-watchlist-button", "n_clicks")
        ],
        [
            State("watchlist-input", "value")
        ]
    )
    def clear_input(n_clicks, value):
        if n_clicks and value:
            return ""
        return value
    
    # Callbacks for remove buttons
    for symbol in watchlist.default_stocks:
        @app.callback(
            Output("watchlist-content", "children", allow_duplicate=True),
            [
                Input(f"remove-{symbol}", "n_clicks")
            ],
            prevent_initial_call=True
        )
        def remove_stock(n_clicks, symbol=symbol):
            if n_clicks:
                watchlist.remove_stock(symbol)
                
            # Get watchlist data
            watchlist_data = watchlist.get_watchlist_data()
            
            # Create watchlist content
            return create_watchlist_content(watchlist_data)
