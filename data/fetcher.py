"""
Stock data fetcher module for the Stock Market Dashboard.
This module provides functions to fetch and process stock data from Yahoo Finance.
"""

import sys
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import random

class StockDataFetcher:
    """Class to fetch and process stock data."""
    
    def __init__(self):
        """Initialize the StockDataFetcher."""
        self.cache = {}  # Simple in-memory cache
        self.cache_expiry = {}  # Track when cache entries expire
    
    def get_stock_chart(self, symbol, interval='1d', range='1mo', region='US', 
                        include_pre_post=False, include_adjusted_close=True):
        
        # Create cache key
        cache_key = f"chart_{symbol}_{interval}_{range}_{region}_{include_pre_post}_{include_adjusted_close}"
        
        # Check cache first
        if cache_key in self.cache and datetime.now() < self.cache_expiry.get(cache_key, datetime.now()):
            return self.cache[cache_key]
        
        # Convert range to period for yfinance
        period = range
        
        # Call yfinance API
        try:
            # For intraday data with specific intervals
            if interval in ['1m', '2m', '5m', '15m', '30m', '60m']:
                # yfinance has limitations on intraday data
                # For intervals less than 1d, we can only get limited history
                if range in ['1d', '5d', '1wk']:
                    ticker_data = yf.Ticker(symbol)
                    df = ticker_data.history(period=range, interval=interval, prepost=include_pre_post)
                else:
                    # For longer ranges with intraday data, we need to limit to 60 days
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=60)
                    ticker_data = yf.Ticker(symbol)
                    df = ticker_data.history(start=start_date, end=end_date, interval=interval, prepost=include_pre_post)
            else:
                # For daily or longer intervals
                ticker_data = yf.Ticker(symbol)
                df = ticker_data.history(period=period, interval=interval, prepost=include_pre_post)
            
            # Get ticker info
            try:
                info = ticker_data.info
                currency = info.get('currency', 'USD')
                exchange = info.get('exchange', 'Unknown')
                regular_market_price = info.get('regularMarketPrice', df['Close'].iloc[-1] if not df.empty else 0)
                regular_market_time = info.get('regularMarketTime', int(datetime.now().timestamp()))
            except Exception as e:
                print(f"Error getting ticker info: {e}")
                currency = 'USD'
                exchange = 'Unknown'
                regular_market_price = df['Close'].iloc[-1] if not df.empty else 0
                regular_market_time = int(datetime.now().timestamp())
            
            # Process the data
            if df.empty:
                print(f"Empty data received for {symbol}. Using default data.")
                return self._create_default_data(symbol)
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Add timestamp column
            df['timestamp'] = [int(dt.timestamp()) for dt in df.index]
            
            # Add adjusted close if needed
            if include_adjusted_close and 'Adj Close' in df.columns:
                df = df.rename(columns={'Adj Close': 'adjclose'})
            
            # Create result dictionary
            result = {
                'meta': {
                    'symbol': symbol,
                    'currency': currency,
                    'exchangeName': exchange,
                    'instrumentType': 'EQUITY',
                    'regularMarketPrice': regular_market_price,
                    'regularMarketTime': regular_market_time
                },
                'data': df,
                'symbol': symbol,
                'currency': currency,
                'exchange': exchange
            }
            
            # Cache the result with appropriate expiry time
            self._cache_result(cache_key, result, interval)
            
            return result
            
        except Exception as e:
            print(f"Error fetching stock chart data for {symbol}: {e}")
            # Return default data in case of error
            return self._create_default_data(symbol)
            
    def _create_default_data(self, symbol):

        # Create a date range for the last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create a default DataFrame with sample data
        df = pd.DataFrame({
            'timestamp': [int(dt.timestamp()) for dt in date_range],
            'open': [100.0 + i * 0.1 for i in range(len(date_range))],
            'high': [101.0 + i * 0.1 for i in range(len(date_range))],
            'low': [99.0 + i * 0.1 for i in range(len(date_range))],
            'close': [100.5 + i * 0.1 for i in range(len(date_range))],
            'volume': [1000000 for _ in range(len(date_range))],
            'adjclose': [100.5 + i * 0.1 for i in range(len(date_range))]
        }, index=date_range)
        
        # Create default result
        result = {
            'meta': {
                'symbol': symbol,
                'currency': 'USD',
                'exchangeName': 'Default Exchange',
                'instrumentType': 'EQUITY',
                'regularMarketPrice': df['close'].iloc[-1],
                'regularMarketTime': int(datetime.now().timestamp())
            },
            'data': df,
            'symbol': symbol,
            'currency': 'USD',
            'exchange': 'Default Exchange'
        }
        
        return result
    
    def get_stock_insights(self, symbol):
        # Create cache key
        cache_key = f"insights_{symbol}"
        
        # Check cache first
        if cache_key in self.cache and datetime.now() < self.cache_expiry.get(cache_key, datetime.now()):
            return self.cache[cache_key]
        
        try:
            # Get ticker data
            ticker = yf.Ticker(symbol)
            
            # Get recommendations
            try:
                recommendations = ticker.recommendations
                if recommendations is not None and not recommendations.empty:
                    latest_rec = recommendations.iloc[-1]
                    recommendation = {
                        'firm': latest_rec.name,
                        'toGrade': latest_rec['To Grade'] if 'To Grade' in latest_rec else 'N/A',
                        'fromGrade': latest_rec['From Grade'] if 'From Grade' in latest_rec else 'N/A',
                        'action': latest_rec['Action'] if 'Action' in latest_rec else 'N/A'
                    }
                else:
                    recommendation = {'firm': 'N/A', 'toGrade': 'N/A', 'fromGrade': 'N/A', 'action': 'N/A'}
            except Exception as e:
                print(f"Error getting recommendations: {e}")
                recommendation = {'firm': 'N/A', 'toGrade': 'N/A', 'fromGrade': 'N/A', 'action': 'N/A'}
            
            # Get info
            try:
                info = ticker.info
            except Exception as e:
                print(f"Error getting info: {e}")
                info = {}
            
            # Create insights data
            insights = {
                'symbol': symbol,
                'recommendation': recommendation,
                'targetPrice': info.get('targetMeanPrice', 0),
                'targetHighPrice': info.get('targetHighPrice', 0),
                'targetLowPrice': info.get('targetLowPrice', 0),
                'numberOfAnalystOpinions': info.get('numberOfAnalystOpinions', 0),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'beta': info.get('beta', 0),
                'forwardPE': info.get('forwardPE', 0),
                'dividendYield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                'marketCap': info.get('marketCap', 0)
            }
            
            # Cache the result
            self._cache_result(cache_key, insights, '1d')
            
            return insights
            
        except Exception as e:
            print(f"Error fetching stock insights: {e}")
            return {
                'symbol': symbol,
                'recommendation': {'firm': 'N/A', 'toGrade': 'N/A', 'fromGrade': 'N/A', 'action': 'N/A'},
                'targetPrice': 0,
                'targetHighPrice': 0,
                'targetLowPrice': 0,
                'numberOfAnalystOpinions': 0,
                'sector': 'N/A',
                'industry': 'N/A',
                'beta': 0,
                'forwardPE': 0,
                'dividendYield': 0,
                'marketCap': 0
            }
    
    def _cache_result(self, key, data, interval):
        self.cache[key] = data
        
        # Set expiry time based on interval
        if interval in ['1m', '2m', '5m']:
            expiry_time = datetime.now() + timedelta(minutes=1)
        elif interval in ['15m', '30m', '60m']:
            expiry_time = datetime.now() + timedelta(minutes=5)
        elif interval == '1d':
            expiry_time = datetime.now() + timedelta(hours=1)
        else:
            expiry_time = datetime.now() + timedelta(days=1)
        
        self.cache_expiry[key] = expiry_time
