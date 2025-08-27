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

## 🐛 Bug Fixes & Improvements

### Original Issues Fixed:
1. **Data Download Failures**: Added timeout handling and parallel processing
2. **Index Errors**: Improved date index computation with fallback methods
3. **Volatility Calculation**: Added safety checks for zero/infinite volatility
4. **Memory Issues**: Better DataFrame handling and cleanup
5. **Error Propagation**: Proper HTTP status codes and error messages
6. **Timezone Handling**: Fallback to UTC if configured timezone fails
7. **Data Validation**: Checks for missing columns and invalid data

### Performance Optimizations:
1. **Parallel Downloads**: Concurrent yfinance downloads using ThreadPoolExecutor
2. **Batch Processing**: Efficient handling of large universes
3. **Memory Management**: Better DataFrame concatenation and cleanup
4. **Caching**: Configurable caching for frequently accessed data
5. **Async Support**: Prepared for future async implementation

## 📦 Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd momentum-strategy-api
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure the application:**
```bash
cp config.yaml.example config.yaml
# Edit config.yaml with your settings
```

## ⚙️ Configuration

The `config.yaml` file contains all strategy parameters:

```yaml
# Strategy settings
universe_source: "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
history_days: 520
lookback_6m_days: 126
lookback_12m_days: 252
exclude_1m_days: 21
vol_window_days: 126
sma_window_days: 200

# Portfolio settings
top_n: 20
price_cap: 5000
capital: 100000

# Performance settings
max_workers: 10
download_timeout: 30
```

## 🚀 Usage

### Start the API Server:
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints:

#### 1. Health Check
```bash
GET /health
```

#### 2. Run Momentum Strategy
```bash
GET /run?capital=100000&top_n=20&price_cap=5000
```

**Parameters:**
- `capital`: Investment capital (optional, uses config default)
- `top_n`: Number of top stocks to select (optional, uses config default)
- `price_cap`: Maximum stock price (optional, uses config default)
- `ref_date`: Reference date in YYYY-MM-DD format (optional, uses current date)

#### 3. Calculate ATR Stop Losses
```bash
GET /levels?tickers=BEL.NS,MFSL.NS,RELIANCE.NS
```

## 📊 Strategy Logic

The momentum strategy follows these steps:

1. **Universe Loading**: Loads Nifty 500 stocks from NSE
2. **Data Download**: Downloads historical prices using yfinance
3. **Momentum Calculation**: Computes 6M and 12M (ex-1M) returns
4. **Volatility**: Calculates 6-month rolling volatility
5. **Filters**: Applies 200-DMA and positive 6M return filters
6. **Scoring**: Combines returns and volatility for final momentum score
7. **Selection**: Ranks by score and selects top N stocks
8. **Allocation**: Suggests equal-weight allocations

## 🔧 Advanced Configuration

### Performance Tuning:
```yaml
max_workers: 10              # Parallel download workers
download_timeout: 30         # Download timeout in seconds
retry_failed_downloads: true # Enable retry mechanism
max_retries: 3               # Maximum retry attempts
```

### Data Quality:
```yaml
min_data_quality: 0.8        # Minimum data quality threshold
require_volume: false        # Require volume data
min_price: 10                # Minimum stock price
max_price: 100000            # Maximum stock price
```

## 📈 Output Files

When `write_csv: true`, the API generates:

1. **phase2_prices_snapshot_YYYYMMDD_HHMMSS.csv**: Raw price data
2. **phase3_momentum_scores_YYYYMMDD_HHMMSS.csv**: Momentum calculations
3. **phase4_filtered_eligible_YYYYMMDD_HHMMSS.csv**: Stocks passing filters
4. **phase4_top_selection_YYYYMMDD_HHMMSS.csv**: Final selected stocks
5. **phase3_reference_dates_YYYYMMDD_HHMMSS.csv**: Reference dates used

## 🚨 Error Handling

The API includes comprehensive error handling:

- **Network Failures**: Automatic fallback to local files
- **Data Quality Issues**: Validation and filtering of invalid data
- **Missing Data**: Graceful handling of incomplete datasets
- **Configuration Errors**: Early detection and clear error messages
- **Rate Limiting**: Configurable timeouts and retry mechanisms

## 📝 Logging

Comprehensive logging with configurable levels:

```yaml
log_level: "INFO"            # DEBUG, INFO, WARNING, ERROR
log_to_file: false           # Enable file logging
log_file: "momentum_api.log" # Log file path
```

## 🔍 Monitoring

### Health Check Response:
```json
{
  "ok": true,
  "time": "2024-01-15T10:30:00",
  "config_loaded": true,
  "timezone": "Asia/Kolkata"
}
```

### Run Endpoint Metadata:
```json
{
  "metadata": {
    "total_universe": 500,
    "valid_tickers": 485,
    "eligible_count": 45,
    "top_selected": 20,
    "execution_time": "2024-01-15T10:30:00"
  }
}
```

## 🚀 Performance Tips

1. **Adjust max_workers** based on your system capabilities
2. **Use appropriate timeouts** for your network conditions
3. **Enable CSV timestamps** for better file management
4. **Monitor log levels** for debugging vs production use
5. **Consider caching** frequently accessed data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This API is for educational and research purposes. Investment decisions should be made after thorough research and consultation with financial advisors. Past performance does not guarantee future results.
