# Momentum Strategy API

A high-performance FastAPI application for computing momentum-based stock selection strategies with comprehensive error handling and optimization.

## 🚀 Features

- **Parallel Data Download**: Uses ThreadPoolExecutor for faster yfinance data fetching
- **Robust Error Handling**: Comprehensive error handling with fallback mechanisms
- **Configurable Strategy**: Flexible configuration for lookback periods, filters, and thresholds
- **Data Validation**: Built-in data quality checks and validation
- **CSV Export**: Timestamped CSV exports for analysis and record-keeping
- **Health Monitoring**: Built-in health checks and logging
- **ATR Stop Loss**: ATR-based stop loss calculations for risk management

## 🐛 Bug Fixes in This Version

1. **Fixed yfinance MultiIndex handling**: Better handling of different yfinance response structures
2. **Improved error handling**: Added comprehensive error handling for all functions
3. **Fixed date index computation**: Better handling of missing dates and edge cases
4. **Added data validation**: Checks for data quality and completeness
5. **Fixed volatility calculation**: Better handling of zero/infinite volatility values
6. **Added timeout handling**: Prevents hanging on slow network requests
7. **Fixed CSV saving**: Added timestamping and better error handling
8. **Improved memory management**: Better DataFrame handling and cleanup

## 🚀 Performance Optimizations

1. **Parallel Downloads**: Uses ThreadPoolExecutor for concurrent yfinance downloads
2. **Batch Processing**: Efficient batch processing of large universes
3. **Memory Optimization**: Better DataFrame memory management
4. **Caching**: Configurable caching for frequently accessed data
5. **Async Operations**: Non-blocking operations where possible

## 📋 Requirements

- Python 3.8+
- FastAPI 0.110.0
- pandas 2.2.2
- yfinance 0.2.40
- Other dependencies listed in `requirements.txt`

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd momentum-strategy-api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the application:
   - Copy `config.yaml` and modify as needed
   - Ensure output directory exists

4. Run the application:
```bash
python main.py
```

Or with uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 📊 API Endpoints

### Health Check
```
GET /health
```
Returns API status and configuration information.

### Run Strategy
```
GET /run?capital=100000&top_n=20&price_cap=5000&ref_date=2024-01-01
```
Computes momentum scores and returns eligible stocks with top selections.

**Parameters:**
- `capital`: Investment capital (default from config)
- `top_n`: Number of top stocks to select (default from config)
- `price_cap`: Maximum stock price (default from config)
- `ref_date`: Reference date in YYYY-MM-DD format (default: latest available)

### ATR Stop Levels
```
GET /levels?tickers=BEL.NS,MFSL.NS
```
Calculates ATR(14) based stop loss levels for given tickers.

## ⚙️ Configuration

The `config.yaml` file contains all strategy parameters:

```yaml
# Strategy settings
lookback_6m_days: 126        # 6-month lookback
lookback_12m_days: 252       # 12-month lookback
exclude_1m_days: 21          # Exclude last month
vol_window_days: 126         # Volatility calculation window
sma_window_days: 200         # 200-day moving average

# Portfolio settings
top_n: 20                    # Number of stocks to select
price_cap: 5000              # Maximum stock price
capital: 100000              # Investment capital

# Performance settings
max_workers: 10              # Parallel download workers
download_timeout: 30         # Download timeout
```

## 🔧 Strategy Logic

1. **Universe Loading**: Loads NIFTY 500 stocks from NSE or local fallback
2. **Data Download**: Downloads adjusted prices using yfinance with parallel processing
3. **Momentum Calculation**: Computes 6M and 12M ex-1M returns
4. **Volatility**: Calculates annualized volatility over 6-month window
5. **Trend Filter**: 200-day moving average filter
6. **Scoring**: Momentum score = (6M return + 12M ex-1M return) / volatility
7. **Filtering**: Stocks above 200DMA with positive 6M returns
8. **Ranking**: Top N stocks by momentum score
9. **Allocation**: Equal-weight allocation suggestions

## 📈 Output Files

When `write_csv: true`, the API generates:

- `phase2_prices_snapshot_YYYYMMDD_HHMMSS.csv`: Raw price data
- `phase3_momentum_scores_YYYYMMDD_HHMMSS.csv`: Momentum calculations
- `phase4_filtered_eligible_YYYYMMDD_HHMMSS.csv`: Eligible stocks
- `phase4_top_selection_YYYYMMDD_HHMMSS.csv`: Top selections with allocations
- `phase3_reference_dates_YYYYMMDD_HHMMSS.csv`: Reference dates used

## 🚨 Error Handling

The API includes comprehensive error handling:

- **Network failures**: Automatic fallback to local files
- **Data quality**: Validation of downloaded data
- **Missing dates**: Intelligent date handling with fallbacks
- **Invalid symbols**: Filtering of invalid tickers
- **Timeout handling**: Configurable timeouts for all operations

## 📝 Logging

Comprehensive logging with configurable levels:

- **INFO**: General operations and progress
- **WARNING**: Non-critical issues and fallbacks
- **ERROR**: Critical errors and failures
- **DEBUG**: Detailed debugging information

## 🔍 Monitoring

Built-in monitoring capabilities:

- Health check endpoint
- Execution time tracking
- Data quality metrics
- Success/failure rates
- Performance metrics

## 🚀 Performance Tips

1. **Adjust max_workers**: Set based on your system capabilities
2. **Use appropriate timeouts**: Balance between reliability and speed
3. **Monitor memory usage**: Large universes may require more memory
4. **Cache results**: Consider caching for repeated requests
5. **Batch processing**: Process large universes in smaller batches

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This API is for educational and research purposes only. It does not constitute financial advice. Always do your own research and consult with financial professionals before making investment decisions.
