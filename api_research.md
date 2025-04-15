# Yahoo Finance API Research

## Available APIs

### 1. YahooFinance/get_stock_chart
This API provides comprehensive stock market data including:
- Meta information (currency, symbol, exchange details)
- Trading periods
- Time-series data for price indicators (open, close, high, low, volume)
- Adjusted close prices

#### Parameters:
- `symbol` (Required): The stock symbol (e.g., AAPL)
- `region` (Optional): Region code (US, BR, AU, CA, FR, DE, HK, IN, IT, ES, GB, SG)
- `interval` (Required): Time interval (1m, 2m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo)
- `range` (Required): Date range (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
- `period1` (Optional): Start timestamp in seconds
- `period2` (Optional): End timestamp in seconds
- `comparisons` (Optional): Symbols for comparison
- `events` (Optional): Events to include (capitalGain, div, split, earn, history)
- `includePrePost` (Optional): Whether to include pre/post market data
- `includeAdjustedClose` (Optional): Whether to include adjusted close data
- `useYfid` (Optional): Whether to use yfId instead of symbol

### 2. YahooFinance/get_stock_insights
This API provides comprehensive financial analysis data including:
- Technical indicators (short/intermediate/long-term outlooks)
- Company metrics (innovativeness, sustainability, hiring)
- Valuation details
- Research reports
- Significant developments
- SEC filings

#### Parameters:
- `symbol` (Required): The stock symbol (e.g., AAPL)

## Data Points Available for Dashboard

### Price Data
- Open, high, low, close prices
- Volume
- Adjusted close prices
- Pre/post market data (optional)

### Time Frames
- Minute intervals: 1m, 2m, 5m, 15m, 30m, 60m
- Daily, weekly, monthly intervals: 1d, 1wk, 1mo
- Historical ranges: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max

### Technical Analysis
- Short-term outlook
- Intermediate-term outlook
- Long-term outlook
- Support and resistance levels
- Stop loss recommendations

### Company Information
- Currency
- Exchange information
- Full company name
- 52-week high/low
- Regular market price, day high/low, volume
- Innovativeness, hiring, sustainability metrics
- Insider sentiments
- Earnings reports
- Dividends

### Comparison Data
- Ability to compare with other stocks or indices

## Implementation Strategy

1. **Core Data Fetching**:
   - Use `get_stock_chart` API for price and volume data
   - Support different time intervals and ranges
   - Implement caching to reduce API calls

2. **Enhanced Analysis**:
   - Use `get_stock_insights` API for technical analysis and company metrics
   - Display technical indicators alongside price charts
   - Show company metrics in a separate panel

3. **User Customization**:
   - Allow users to select different stocks
   - Provide options for different time frames
   - Enable toggling of different data points and indicators

4. **Visualization Approach**:
   - Use Plotly for interactive charts
   - Implement candlestick charts for price data
   - Create volume bars below price charts
   - Add technical indicator overlays
   - Include comparison charts when multiple symbols are selected
