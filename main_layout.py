"""
Main layout module for the Stock Market Dashboard.
This module defines the main dashboard layout and UI components.
"""

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from datetime import datetime

def create_header():
    """
    Create the dashboard header.
    
    Returns:
        dash component: Header component
    """
    return html.Div([
        html.H1("Stock Market Dashboard", className="dashboard-title"),
        html.P("Real-time stock data visualization and analysis", className="dashboard-subtitle"),
        html.Hr()
    ], className="dashboard-header")

def create_info_card(title, id_prefix):
    """
    Create an information card for displaying stock metrics.
    
    Args:
        title (str): Card title
        id_prefix (str): Prefix for component IDs
        
    Returns:
        dash component: Info card
    """
    return dbc.Card([
        dbc.CardHeader(title),
        dbc.CardBody([
            html.Div([
                html.Span("Current: ", className="info-label"),
                html.Span(id=f"{id_prefix}-current", className="info-value")
            ]),
            html.Div([
                html.Span("Open: ", className="info-label"),
                html.Span(id=f"{id_prefix}-open", className="info-value")
            ]),
            html.Div([
                html.Span("High: ", className="info-label"),
                html.Span(id=f"{id_prefix}-high", className="info-value")
            ]),
            html.Div([
                html.Span("Low: ", className="info-label"),
                html.Span(id=f"{id_prefix}-low", className="info-value")
            ]),
            html.Div([
                html.Span("Volume: ", className="info-label"),
                html.Span(id=f"{id_prefix}-volume", className="info-value")
            ]),
            html.Div([
                html.Span("Change: ", className="info-label"),
                html.Span(id=f"{id_prefix}-change", className="info-value")
            ])
        ])
    ], className="info-card")

def create_chart_card(title, chart_id, height=500):
    """
    Create a card containing a chart.
    
    Args:
        title (str): Card title
        chart_id (str): ID for the chart component
        height (int, optional): Chart height
        
    Returns:
        dash component: Chart card
    """
    return dbc.Card([
        dbc.CardHeader(title),
        dbc.CardBody([
            dcc.Graph(
                id=chart_id,
                figure=go.Figure(),
                style={'height': f'{height}px'},
                config={
                    'displayModeBar': True,
                    'scrollZoom': True,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                }
            )
        ])
    ], className="chart-card")

def create_indicators_card():
    """
    Create a card containing technical indicators.
    
    Returns:
        dash component: Indicators card
    """
    return dbc.Card([
        dbc.CardHeader("Technical Indicators"),
        dbc.CardBody([
            dcc.Tabs([
                dcc.Tab(label="RSI", children=[
                    dcc.Graph(
                        id="rsi-chart",
                        figure=go.Figure(),
                        style={'height': '300px'},
                        config={'displayModeBar': False}
                    )
                ]),
                dcc.Tab(label="MACD", children=[
                    dcc.Graph(
                        id="macd-chart",
                        figure=go.Figure(),
                        style={'height': '300px'},
                        config={'displayModeBar': False}
                    )
                ]),
                dcc.Tab(label="Stochastic", children=[
                    dcc.Graph(
                        id="stochastic-chart",
                        figure=go.Figure(),
                        style={'height': '300px'},
                        config={'displayModeBar': False}
                    )
                ])
            ])
        ])
    ], className="indicators-card")

def create_dashboard_layout():
    """
    Create the main dashboard layout.
    
    Returns:
        dash component: Main dashboard layout
    """
    # Import components here to avoid circular imports
    from components.stock_selector import create_control_panel, create_watchlist
    from components.prediction import create_prediction_card
    
    return html.Div([
        # Store components for holding data
        dcc.Store(id="stock-data-store"),
        dcc.Store(id="comparison-data-store"),
        dcc.Interval(
            id='interval-component',
            interval=60*1000,  # in milliseconds (1 minute)
            n_intervals=0
        ),
        
        # Header
        create_header(),
        
        # Main content
        dbc.Row([
            # Left sidebar with controls
            dbc.Col([
                create_control_panel(),
                html.Div(style={"height": "20px"}),  # Spacer
                create_watchlist(),
                html.Div(style={"height": "20px"}),  # Spacer
                create_info_card("Stock Info", "stock-info")
            ], width=3, className="sidebar-column"),
            
            # Main content area
            dbc.Col([
                # Main price chart
                create_chart_card("Price Chart", "price-chart", height=500),
                html.Div(style={"height": "20px"}),  # Spacer
                
                # Prediction card
                create_prediction_card(),
                html.Div(style={"height": "20px"}),  # Spacer
                
                # Technical indicators
                create_indicators_card(),
                html.Div(style={"height": "20px"}),  # Spacer
                
                # Volume chart
                create_chart_card("Volume", "volume-chart", height=250)
            ], width=9, className="content-column")
        ]),
        
        # Footer
        html.Footer([
            html.P([
                "Data provided by Yahoo Finance API • Last updated: ",
                html.Span(id="last-updated-time")
            ]),
            html.P("© 2025 Stock Market Dashboard")
        ], className="dashboard-footer")
    ], className="dashboard-container")
