import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

from data.predictor import StockPredictor, create_prediction_chart

def create_prediction_card():
    return dbc.Card([
        dbc.CardHeader([
            html.H5("Price Prediction (XGBoost)", className="d-inline"),
            dbc.Button(
                "Train Model",
                id="train-model-button",
                color="primary",
                size="sm",
                className="float-right ml-2"
            ),
            dbc.Button(
                "Predict",
                id="predict-button",
                color="success",
                size="sm",
                className="float-right"
            )
        ]),
        dbc.CardBody([
            # Prediction settings
            dbc.Form([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Forecast Days:"),
                        dbc.Input(
                            id="forecast-days-input",
                            type="number",
                            min=1,
                            max=30,
                            step=1,
                            value=5
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Confidence Level:"),
                        dbc.Select(
                            id="confidence-level-select",
                            options=[
                                {"label": "95%", "value": "95"},
                                {"label": "90%", "value": "90"},
                                {"label": "80%", "value": "80"}
                            ],
                            value="95"
                        )
                    ], width=6)
                ]),
                html.Div(style={"height": "10px"}),  # Spacer
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Model Status:"),
                        html.Div(id="model-status", className="text-info")
                    ], width=12)
                ])
            ]),
            
            html.Div(style={"height": "20px"}),  # Spacer
            
            # Prediction chart
            dcc.Loading(
                id="prediction-loading",
                type="circle",
                children=[
                    dcc.Graph(
                        id="prediction-chart",
                        figure=go.Figure(),
                        style={'height': '300px'},
                        config={'displayModeBar': False}
                    )
                ]
            ),
            
            html.Div(style={"height": "100px"}),  # Spacer
            
            # Prediction metrics
            html.Div(id="prediction-metrics", className="prediction-metrics")
        ])
    ], className="prediction-card")

def register_prediction_callbacks(app):
    # Initialize predictor
    predictor = StockPredictor()
    
    # Callback to update prediction chart
    @app.callback(
        [
            Output("prediction-chart", "figure"),
            Output("prediction-metrics", "children"),
            Output("model-status", "children")
        ],
        [
            Input("predict-button", "n_clicks"),
            Input("train-model-button", "n_clicks")
        ],
        [
            State("stock-data-store", "data"),
            State("forecast-days-input", "value"),
            State("confidence-level-select", "value")
        ]
    )
    def update_prediction(predict_clicks, train_clicks, stock_data, forecast_days, confidence_level):
        if not stock_data:
            return go.Figure(), html.Div("No data available for prediction"), "No model trained"
        
        ctx = dash.callback_context
        button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
        
        # Convert data back to DataFrame format
        df = pd.DataFrame(stock_data['data'])
        if 'datetime' in df.columns:
            df.set_index('datetime', inplace=True)
        
        # Reconstruct stock_data with DataFrame
        stock_data_with_df = stock_data.copy()
        stock_data_with_df['data'] = df
        
        symbol = stock_data.get('symbol', 'Unknown')
        
        # Train model if requested
        if button_id == "train-model-button":
            try:
                results = predictor.train_model(stock_data_with_df, symbol, forecast_days)
                
                # Create prediction chart
                fig = create_prediction_chart(results, stock_data_with_df)
                
                # Create metrics display
                metrics = results.get('metrics', {})
                metrics_html = html.Div([
                    html.H6("Model Metrics:"),
                    html.Div([
                        html.Span("RMSE: ", className="metric-label"),
                        html.Span(f"{metrics.get('rmse', 0):.4f}", className="metric-value")
                    ]),
                    html.Div([
                        html.Span("MAE: ", className="metric-label"),
                        html.Span(f"{metrics.get('mae', 0):.4f}", className="metric-value")
                    ]),
                    html.Div([
                        html.Span("R²: ", className="metric-label"),
                        html.Span(f"{metrics.get('r2', 0):.4f}", className="metric-value")
                    ])
                ])
                
                return fig, metrics_html, f"Model trained for {symbol} at {datetime.now().strftime('%H:%M:%S')}"
                
            except Exception as e:
                print(f"Error training model: {e}")
                return go.Figure(), html.Div(f"Error training model: {str(e)}"), "Training failed"
        
        # Make prediction if requested
        elif button_id == "predict-button":
            try:
                results = predictor.predict(stock_data_with_df, symbol, forecast_days)
                
                # Create prediction chart
                fig = create_prediction_chart(results, stock_data_with_df)
                
                # Create prediction display
                forecast = results.get('forecast', pd.DataFrame())
                if not forecast.empty:
                    last_price = results.get('last_price', 0)
                    next_price = forecast['predicted_close'].iloc[0]
                    change = next_price - last_price
                    change_pct = (change / last_price) * 100 if last_price != 0 else 0
                    
                    # Determine color based on predicted change
                    color = "success" if change >= 0 else "danger"
                    change_icon = "▲" if change >= 0 else "▼"
                    
                    metrics_html = html.Div([
                        html.H6("Prediction Summary:"),
                        html.Div([
                            html.Span("Current Price: ", className="metric-label"),
                            html.Span(f"${last_price:.2f}", className="metric-value")
                        ]),
                        html.Div([
                            html.Span("Next Day Prediction: ", className="metric-label"),
                            html.Span(f"${next_price:.2f}", className="metric-value")
                        ]),
                        html.Div([
                            html.Span("Predicted Change: ", className="metric-label"),
                            html.Span([
                                f"{change_icon} ${abs(change):.2f} ",
                                f"({abs(change_pct):.2f}%)"
                            ], className=f"text-{color}")
                        ])
                    ])
                    
                    return fig, metrics_html, f"Using trained model for {symbol}"
                else:
                    return go.Figure(), html.Div("No forecast data available"), "Prediction failed"
                
            except Exception as e:
                print(f"Error making prediction: {e}")
                return go.Figure(), html.Div(f"Error making prediction: {str(e)}"), "Prediction failed"
        
        # Default empty state
        return go.Figure(), html.Div("Click 'Train Model' to train a new model or 'Predict' to use existing model"), "No model trained"