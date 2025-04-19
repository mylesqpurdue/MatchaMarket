import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime, timedelta



class StockPredictor:
    
    def __init__(self, model_dir='/home/ubuntu/stock_dashboard/models'):
        self.model_dir = model_dir
        self.models = {}  # Cache for loaded models
        self.scalers = {}  # Cache for loaded scalers
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def _create_features(self, df):
        # Make a copy of the dataframe to avoid modifying the original
        data = df.copy()
        
        # Technical indicators
        # Moving averages
        data['ma5'] = data['close'].rolling(window=5).mean()
        data['ma10'] = data['close'].rolling(window=10).mean()
        data['ma20'] = data['close'].rolling(window=20).mean()
        
        # Price momentum
        data['price_momentum'] = data['close'].pct_change(periods=5)
        
        # Volatility
        data['volatility'] = data['close'].rolling(window=10).std()
        
        # Price difference
        data['price_diff'] = data['close'].diff()
        
        # Relative Strength Index (RSI)
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        data['ema12'] = data['close'].ewm(span=12, adjust=False).mean()
        data['ema26'] = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = data['ema12'] - data['ema26']
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=20).mean()
        data['bb_std'] = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + (data['bb_std'] * 2)
        data['bb_lower'] = data['bb_middle'] - (data['bb_std'] * 2)
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        
        # Volume features
        data['volume_change'] = data['volume'].pct_change()
        data['volume_ma5'] = data['volume'].rolling(window=5).mean()
        
        # Target variable (next day's closing price)
        data['target'] = data['close'].shift(-1)
        #data['target'] = data['close'].pct_change().shift(-1)
        #FOR TESTING RSME
        
        # Drop NaN values
        data = data.dropna()
        
        return data
    

    def train_model(self, stock_data, symbol, forecast_days=5):
        # 1) Pull out raw data
        df = stock_data['data']
        if isinstance(df, list):
            df = pd.DataFrame(df)

        # 2) Ensure a DateTimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            for col in ("datetime","Datetime","Date","date","index"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
                    df.set_index(col, inplace=True)
                    break
            else:
                raise ValueError(
                    "Stock data must have a 'datetime','Datetime','Date','date', or 'index' column."
                )

        # 3) Build features
        data = self._create_features(df)

        # 4) Pre‑flight: do we have enough rows?
        if data.shape[0] < 2:
            raise ValueError(
                f"Not enough data after feature‑engineering for {symbol}: "
                f"{data.shape[0]} rows."
            )

        # 5) Define features + target
        features = [
            'lag1',    # yesterday's close
            'open','high','low','close','volume',
            'ma5','ma10','ma20','price_momentum','volatility',
            'price_diff','rsi','macd','macd_signal',
            'bb_width','volume_change','volume_ma5'
        ]
        # inject lagged close
        data['lag1'] = data['close'].shift(1)
        data = data.dropna(subset=['lag1'])

        X = data[features]
        y = data['target']

        # 6) Split, scale, train
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100, learning_rate=0.1,
            max_depth=5, subsample=0.8, colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train_s, y_train)

        # 7) Evaluate
        y_pred = model.predict(X_test_s)
        mse  = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred) if len(y_test) >= 2 else float('nan')

        # 8) Persist model + scaler
        os.makedirs(self.model_dir, exist_ok=True)
        with open(os.path.join(self.model_dir, f"{symbol}_model.pkl"), 'wb') as f:
            pickle.dump(model, f)
        with open(os.path.join(self.model_dir, f"{symbol}_scaler.pkl"), 'wb') as f:
            pickle.dump(scaler, f)

        # 9) Multi‑step forecast
        forecast = self._generate_forecast(
            raw_df=df,         # pass the *raw* price/volume DF
            model=model,
            scaler=scaler,
            features=features,
            days=forecast_days
        )

        return {
            'symbol'            : symbol,
            'metrics'           : {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2},
            'forecast'          : forecast,
            'feature_importance': {
                'features': features,
                'importance': model.feature_importances_.tolist()
            }
        }


    def _generate_forecast(self, raw_df, model, scaler, features, days=5):
        """
        raw_df: a DataFrame with a proper DatetimeIndex and columns
                ['open','high','low','close','volume',...]
        """
        # 0) Pre‑flight
        if raw_df.empty:
            raise ValueError("Cannot forecast: raw_df is empty")

        # 1) Copy raw history
        history = raw_df.copy()

        # 2) Prepare forecast DataFrame
        last_date = history.index[-1]
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(days)]
        forecast_df = pd.DataFrame(
            index=forecast_dates,
            columns=['predicted_close','lower_bound','upper_bound']
        )

        for date in forecast_dates:
            # 3a) Recompute all features on history
            feat_df = self._create_features(history)

            # 3b) Inject lag1
            feat_df['lag1'] = feat_df['close'].shift(1)
            feat_df = feat_df.dropna(subset=['lag1'])

            # 3c) Grab last row of features
            row = feat_df.iloc[[-1]][features]
            if row.empty:
                raise ValueError(f"No valid feature row for forecasting on {date}")

            # 4) Scale & predict
            x_scaled = scaler.transform(row)
            pred = model.predict(x_scaled)[0]

            # 5) Record
            forecast_df.at[date, 'predicted_close'] = pred
            forecast_df.at[date, 'lower_bound']     = pred * 0.95
            forecast_df.at[date, 'upper_bound']     = pred * 1.05

            # 6) Append to raw history for next step
            new = history.iloc[[-1]].copy()
            new['close'] = pred
            new['open']  = pred
            new['high']  = pred
            new['low']   = pred
            # keep last volume (or you could set volume=np.nan/0)
            history = pd.concat([history, new])

        return forecast_df
    
    def predict(self, stock_data, symbol, days=5):
        # 1) If there’s no saved model/scaler on disk, train a new one:
        model_path  = os.path.join(self.model_dir, f"{symbol}_model.pkl")
        scaler_path = os.path.join(self.model_dir, f"{symbol}_scaler.pkl")

        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            # this will also cache self.models[symbol] & self.scalers[symbol]
            return self.train_model(stock_data, symbol, days)

        # 2) Load into memory if not already cached
        if symbol not in self.models:
            with open(model_path, 'rb')   as f: self.models[symbol]  = pickle.load(f)
            with open(scaler_path, 'rb')  as f: self.scalers[symbol] = pickle.load(f)

        model  = self.models[symbol]
        scaler = self.scalers[symbol]

        # 3) Rehydrate the DataFrame
        df = stock_data['data']
        if isinstance(df, list):
            df = pd.DataFrame(df)

        # 4) Ensure a datetime index (your extended version with 'Date', 'index', etc.)
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
            elif 'Datetime' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Datetime'])
                df.set_index('Datetime', inplace=True)
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            elif 'index' in df.columns:
                df['index'] = pd.to_datetime(df['index'])
                df.set_index('index', inplace=True)
            else:
                raise ValueError(
                    "Stock data must have one of: 'datetime', 'Datetime', 'Date', 'date', or 'index' columns."
                )

        # 5) Feature engineering
        data = self._create_features(df)

        # 6) Forecast
        features = [
            'open','high','low','close','volume',
            'ma5','ma10','ma20','price_momentum','volatility',
            'price_diff','rsi','macd','macd_signal',
            'bb_width','volume_change','volume_ma5'
        ]
        forecast = self._generate_forecast(data, model, scaler, features, days)

        return {
            'symbol'         : symbol,
            'forecast'       : forecast,
            'last_price'     : df['close'].iloc[-1],
            'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


def create_prediction_chart(prediction_data, stock_data, title=None, height=400):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Extract data
    df = stock_data['data']
    symbol = stock_data['symbol']
    forecast = prediction_data['forecast']
    
    # Create figure
    fig = go.Figure()
    
    # Add historical price
    fig.add_trace(go.Scatter(
        x=df.index[-30:],  # Show last 30 days
        y=df['close'][-30:],
        mode='lines',
        name='Historical Price',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Add prediction
    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast['predicted_close'],
        mode='lines+markers',
        name='Predicted Price',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast['upper_bound'],
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast['lower_bound'],
        mode='lines',
        name='Lower Bound',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255, 127, 14, 0.2)',
        showlegend=False
    ))
    
    # Set chart title
    if title:
        chart_title = title
    else:
        chart_title = f"{symbol} Price Prediction"
    
    # Update layout
    fig.update_layout(
        title=chart_title,
        height=height,
        margin=dict(l=40, r=40, t=40, b=40),
        template='plotly_white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Price"
    )
    
    return fig

def compute_rmse(y_true, y_pred):

    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)