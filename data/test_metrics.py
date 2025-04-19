import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb

from data.fetcher     import StockDataFetcher
from data.predictor   import StockPredictor

def compute_rmse(y_true, y_pred):
    if len(y_true) == 0:
        raise ValueError("No samples to compare")
    return np.sqrt(mean_squared_error(y_true, y_pred))

# 1) Fetch 6 months of VOO data
fetcher = StockDataFetcher()
stock_data = fetcher.get_stock_chart(
    symbol   = "VOO",    # ← changed here
    interval = "1d",
    range    = "6mo"
)

# 2) Rehydrate into DataFrame
df = stock_data['data']
if isinstance(df, list):
    df = pd.DataFrame(df)

# 3) Only reset index if not already DatetimeIndex
if not isinstance(df.index, pd.DatetimeIndex):
    for col in ("datetime","Datetime","Date","date","index"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df.set_index(col, inplace=True)
            break
    else:
        raise RuntimeError("Couldn't find a date column in VOO data")

# 4) Feature engineering
predictor = StockPredictor(model_dir="models")
data      = predictor._create_features(df)

features = [
    'open','high','low','close','volume',
    'ma5','ma10','ma20','price_momentum','volatility',
    'price_diff','rsi','macd','macd_signal',
    'bb_width','volume_change','volume_ma5'
]
X = data[features]
y = data['target']

# 5) Train/test split (last 20% as test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# 6) Scale & train
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

model = xgb.XGBRegressor(
    objective        = 'reg:squarederror',
    n_estimators     = 100,
    learning_rate    = 0.1,
    max_depth        = 5,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    random_state     = 42
)
model.fit(X_train_s, y_train)

# 7) Predict & compute RMSE
y_pred = model.predict(X_test_s)
rmse   = compute_rmse(y_test.values, y_pred)

# 8) Baseline (yesterday = today)
baseline_yesterday = y_test.shift(1).dropna()
baseline_actual    = y_test.loc[baseline_yesterday.index]
baseline_rmse      = compute_rmse(baseline_actual.values,
                                  baseline_yesterday.values)

print(f"VOO test RMSE over last {len(y_test)} days: {rmse:.4f}")
print(f"Naïve baseline RMSE: {baseline_rmse:.4f}")
# baseline: predict returnₜ₊₁ = returnₜ
returns = df['close'].pct_change().dropna()
y_test_ret = returns.iloc[-len(y_test):]
baseline_ret = returns.shift(1).iloc[-len(y_test):]
rmse_ret_baseline = compute_rmse(y_test_ret.values, baseline_ret.values)
print(f"Baseline RMSE on returns: {rmse_ret_baseline:.4f}")



