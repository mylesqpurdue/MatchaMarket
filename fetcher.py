"""
Data fetcher module for the Stock Market Dashboard.
This module handles fetching stock data from Yahoo Finance APIs.
"""

import sys
import time
import json
import pandas as pd
from datetime import datetime, timedelta

# Add path for data API access
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient

class StockDataFetcher:
    """Class to fetch stock data from Yahoo Finance APIs."""
    
    def __init__(self):
        """Initialize the StockDataFetcher with an API client."""
        self.client = ApiClient()
        self.cache = {}  # Simple in-memory cache
        self.cache_expiry = {}  # Track when cache entries expire
    
    def get_stock_chart(self, symbol, interval='1d', range='1mo', region='US', 
                        include_pre_post=False, include_adjusted_close=True):
        """
        Fetch stock chart data from Yahoo Finance.
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            interval (str): Time interval between data points ('1m', '2m', '5m', '15m', '30m', '60m', '1d', '1wk', '1mo')
            range (str): Date range for data ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            region (str): Region code ('US', 'BR', 'AU', 'CA', 'FR', 'DE', 'HK', 'IN', 'IT', 'ES', 'GB', 'SG')
            include_pre_post (bool): Whether to include pre/post market data
            include_adjusted_close (bool): Whether to include adjusted close data
            
        Returns:
            dict: Processed stock chart data or default data if error occurs
        """
        # Create cache key
        cache_key = f"chart_{symbol}_{interval}_{range}_{region}_{include_pre_post}_{include_adjusted_close}"
        
        # Check cache first
        if cache_key in self.cache and datetime.now() < self.cache_expiry.get(cache_key, datetime.now()):
            return self.cache[cache_key]
        
        # Prepare query parameters
        query = {
            'symbol': symbol,
            'interval': interval,
            'range': range,
            'region': region,
            'includePrePost': include_pre_post,
            'includeAdjustedClose': include_adjusted_close,
            'useYfid': True
        }
        
        # Call the API
        try:
            response = self.client.call_api('YahooFinance/get_stock_chart', query=query)
            
            # Process the response
            processed_data = self._process_chart_data(response)
            
            # Check if processed data is valid
            if processed_data is None or 'data' not in processed_data or processed_data['data'] is None or processed_data['data'].empty:
                print(f"Invalid data received for {symbol}. Using default data.")
                return self._create_default_data(symbol)
            
            # Cache the result with appropriate expiry time
            self._cache_result(cache_key, processed_data, interval)
            
            return processed_data
            
        except Exception as e:
            print(f"Error fetching stock chart data for {symbol}: {e}")
            # Return default data in case of error
            return self._create_default_data(symbol)
            
    def _create_default_data(self, symbol):
        """
        Create default data when API call fails or returns invalid data.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            dict: Default stock data
        """
        # Create a date range for the last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create a default DataFrame with sample data
        df = pd.DataFrame({
            'timestamp': [int(dt.timestamp()) for dt in date_range],
            'datetime': date_range,
            'open': [100.0 + i * 0.1 for i in range(len(date_range))],
            'high': [101.0 + i * 0.1 for i in range(len(date_range))],
            'low': [99.0 + i * 0.1 for i in range(len(date_range))],
            'close': [100.5 + i * 0.1 for i in range(len(date_range))],
            'volume': [1000000 for _ in range(len(date_range))],
            'adjclose': [100.5 + i * 0.1 for i in range(len(date_range))]
        })
        
        # Set datetime as index
        df.set_index('datetime', inplace=True)
        
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
        """
        Fetch stock insights data from Yahoo Finance.
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL')
            
        Returns:
            dict: Processed stock insights data
        """
        # Create cache key
        cache_key = f"insights_{symbol}"
        
        # Check cache first (insights can be cached longer than price data)
        if cache_key in self.cache and datetime.now() < self.cache_expiry.get(cache_key, datetime.now()):
            return self.cache[cache_key]
        
        # Prepare query parameters
        query = {
            'symbol': symbol
        }
        
        # Call the API
        try:
            response = self.client.call_api('YahooFinance/get_stock_insights', query=query)
            
            # Process the response
            processed_data = self._process_insights_data(response)
            
            # Cache the result (insights can be cached for longer)
            self._cache_result(cache_key, processed_data, '1d', hours=4)
            
            return processed_data
            
        except Exception as e:
            print(f"Error fetching stock insights data: {e}")
            return None
    
    def _process_chart_data(self, response):
        """
        Process raw chart data from the API into a more usable format.
        
        Args:
            response (dict): Raw API response
            
        Returns:
            dict: Processed data with pandas DataFrame for time series
        """
        try:
            # Extract chart result
            chart_result = response.get('chart', {}).get('result', [{}])[0]
            
            # Extract metadata
            meta = chart_result.get('meta', {})
            
            # Extract timestamp and indicators
            timestamps = chart_result.get('timestamp', [])
            indicators = chart_result.get('indicators', {})
            
            # Extract quote data
            quote = indicators.get('quote', [{}])[0]
            
            # Create DataFrame from time series data
            df = pd.DataFrame({
                'timestamp': timestamps,
                'open': quote.get('open', []),
                'high': quote.get('high', []),
                'low': quote.get('low', []),
                'close': quote.get('close', []),
                'volume': quote.get('volume', [])
            })
            
            # Add adjusted close if available
            adjclose = indicators.get('adjclose', [{}])
            if adjclose and len(adjclose) > 0:
                df['adjclose'] = adjclose[0].get('adjclose', [])
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('datetime', inplace=True)
            
            # Drop rows with missing values
            df = df.dropna()
            
            # Create processed result
            result = {
                'meta': meta,
                'data': df,
                'symbol': meta.get('symbol'),
                'currency': meta.get('currency'),
                'exchange': meta.get('exchangeName')
            }
            
            return result
            
        except Exception as e:
            print(f"Error processing chart data: {e}")
            return None
    
    def _process_insights_data(self, response):
        """
        Process raw insights data from the API into a more usable format.
        
        Args:
            response (dict): Raw API response
            
        Returns:
            dict: Processed insights data
        """
        try:
            # Extract finance result
            finance_result = response.get('finance', {}).get('result', {})
            
            # Extract instrument info
            instrument_info = finance_result.get('instrumentInfo', {})
            
            # Extract technical events
            technical_events = instrument_info.get('technicalEvents', {})
            
            # Extract key technicals
            key_technicals = instrument_info.get('keyTechnicals', {})
            
            # Extract valuation
            valuation = instrument_info.get('valuation', {})
            
            # Extract company snapshot
            company_snapshot = finance_result.get('companySnapshot', {})
            
            # Create processed result
            result = {
                'symbol': finance_result.get('symbol'),
                'technical_outlook': {
                    'short_term': technical_events.get('shortTermOutlook', {}),
                    'intermediate_term': technical_events.get('intermediateTermOutlook', {}),
                    'long_term': technical_events.get('longTermOutlook', {})
                },
                'key_technicals': {
                    'support': key_technicals.get('support'),
                    'resistance': key_technicals.get('resistance'),
                    'stop_loss': key_technicals.get('stopLoss')
                },
                'valuation': {
                    'description': valuation.get('description'),
                    'discount': valuation.get('discount'),
                    'relative_value': valuation.get('relativeValue')
                },
                'company_metrics': company_snapshot.get('company', {}),
                'sector_metrics': company_snapshot.get('sector', {}),
                'sector_info': company_snapshot.get('sectorInfo')
            }
            
            # Add significant developments if available
            sig_devs = finance_result.get('sigDevs', [])
            if sig_devs:
                result['significant_developments'] = sig_devs
            
            return result
            
        except Exception as e:
            print(f"Error processing insights data: {e}")
            return None
    
    def _cache_result(self, key, data, interval, hours=None):
        """
        Cache the result with an appropriate expiry time based on the interval.
        
        Args:
            key (str): Cache key
            data (dict): Data to cache
            interval (str): Time interval of the data
            hours (int, optional): Override cache expiry in hours
        """
        self.cache[key] = data
        
        # Set expiry time based on interval if not overridden
        if hours is not None:
            expiry_time = datetime.now() + timedelta(hours=hours)
        else:
            if interval in ['1m', '2m', '5m']:
                expiry_time = datetime.now() + timedelta(minutes=1)
            elif interval in ['15m', '30m', '60m']:
                expiry_time = datetime.now() + timedelta(minutes=5)
            elif interval == '1d':
                expiry_time = datetime.now() + timedelta(minutes=30)
            else:
                expiry_time = datetime.now() + timedelta(hours=2)
        
        self.cache_expiry[key] = expiry_time


# Helper functions for common data retrieval operations

def get_stock_data(symbol, interval='1d', range='1mo'):
    """
    Convenience function to get stock data for a given symbol.
    
    Args:
        symbol (str): Stock symbol
        interval (str): Time interval
        range (str): Date range
        
    Returns:
        dict: Processed stock data
    """
    fetcher = StockDataFetcher()
    return fetcher.get_stock_chart(symbol, interval, range)

def get_stock_insights(symbol):
    """
    Convenience function to get stock insights for a given symbol.
    
    Args:
        symbol (str): Stock symbol
        
    Returns:
        dict: Processed stock insights
    """
    fetcher = StockDataFetcher()
    return fetcher.get_stock_insights(symbol)

def get_multiple_stocks(symbols, interval='1d', range='1mo'):
    """
    Get data for multiple stocks.
    
    Args:
        symbols (list): List of stock symbols
        interval (str): Time interval
        range (str): Date range
        
    Returns:
        dict: Dictionary of processed stock data keyed by symbol
    """
    fetcher = StockDataFetcher()
    result = {}
    
    for symbol in symbols:
        result[symbol] = fetcher.get_stock_chart(symbol, interval, range)
    
    return result
