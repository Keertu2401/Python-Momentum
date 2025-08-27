"""
Enhanced utility functions for momentum strategy optimization
"""
import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta
import os

try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

try:
    import diskcache as dc
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

logger = logging.getLogger(__name__)

# Initialize cache if available
if CACHE_AVAILABLE:
    cache = dc.Cache('cache_dir')
else:
    cache = None

class DataCache:
    """Simple caching mechanism for market data"""
    
    def __init__(self, cache_hours: int = 6):
        self.cache_hours = cache_hours
        self.cache_dir = "cache_data"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_cache_key(self, tickers: List[str], start_date: str, end_date: str) -> str:
        """Generate cache key for data request"""
        ticker_hash = hash(tuple(sorted(tickers)))
        return f"data_{ticker_hash}_{start_date}_{end_date}"
    
    def is_cache_valid(self, cache_file: str) -> bool:
        """Check if cache file is still valid"""
        if not os.path.exists(cache_file):
            return False
        
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        expiry_time = datetime.now() - timedelta(hours=self.cache_hours)
        
        return file_time > expiry_time
    
    def get_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Retrieve cached data if valid"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if self.is_cache_valid(cache_file):
            try:
                return pd.read_pickle(cache_file)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        return None
    
    def save_cached_data(self, cache_key: str, data: pd.DataFrame) -> None:
        """Save data to cache"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        try:
            data.to_pickle(cache_file)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

# Optimized numerical functions using Numba if available
if NUMBA_AVAILABLE:
    @numba.jit(nopython=True)
    def fast_momentum_score(returns_6m, returns_12m, volatility, weight_6m=0.4, weight_12m=0.6):
        """Fast momentum score calculation using Numba JIT"""
        return (weight_6m * returns_6m + weight_12m * returns_12m) / (volatility + 0.01)
    
    @numba.jit(nopython=True)
    def fast_volatility(returns):
        """Fast volatility calculation using Numba JIT"""
        return np.std(returns) * np.sqrt(252) * 100
    
    @numba.jit(nopython=True)
    def fast_sma(prices, window):
        """Fast simple moving average using Numba JIT"""
        result = np.empty(len(prices))
        result[:window-1] = np.nan
        
        for i in range(window-1, len(prices)):
            result[i] = np.mean(prices[i-window+1:i+1])
        
        return result

else:
    def fast_momentum_score(returns_6m, returns_12m, volatility, weight_6m=0.4, weight_12m=0.6):
        """Fallback momentum score calculation"""
        return (weight_6m * returns_6m + weight_12m * returns_12m) / (volatility + 0.01)
    
    def fast_volatility(returns):
        """Fallback volatility calculation"""
        return returns.std() * np.sqrt(252) * 100
    
    def fast_sma(prices, window):
        """Fallback simple moving average"""
        return prices.rolling(window=window).mean()

class EnhancedDataDownloader:
    """Enhanced data downloader with caching and optimization"""
    
    def __init__(self, cache_hours: int = 6, batch_size: int = 100):
        self.cache = DataCache(cache_hours)
        self.batch_size = batch_size
    
    def download_with_cache(self, tickers: List[str], start_date: pd.Timestamp, 
                          end_date: pd.Timestamp) -> pd.DataFrame:
        """Download data with caching support"""
        cache_key = self.cache.get_cache_key(
            tickers, 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d')
        )
        
        # Try to get from cache first
        cached_data = self.cache.get_cached_data(cache_key)
        if cached_data is not None:
            logger.info(f"Using cached data for {len(tickers)} tickers")
            return cached_data
        
        # Download fresh data
        logger.info(f"Downloading fresh data for {len(tickers)} tickers")
        data = self._download_data(tickers, start_date, end_date)
        
        # Save to cache if successful
        if not data.empty:
            self.cache.save_cached_data(cache_key, data)
        
        return data
    
    def _download_data(self, tickers: List[str], start_date: pd.Timestamp, 
                      end_date: pd.Timestamp) -> pd.DataFrame:
        """Internal method for actual data download"""
        frames = []
        failed_tickers = []
        
        for i in range(0, len(tickers), self.batch_size):
            chunk = tickers[i:i+self.batch_size]
            
            try:
                raw = yf.download(
                    chunk,
                    start=start_date,
                    end=end_date + pd.Timedelta(days=1),
                    auto_adjust=True,
                    progress=False,
                    threads=True
                )
                
                if raw.empty:
                    failed_tickers.extend(chunk)
                    continue
                
                # Process the downloaded data
                if len(chunk) == 1:
                    if 'Close' in raw.columns:
                        df = raw[['Close']].copy()
                        df.columns = chunk
                    else:
                        df = raw.iloc[:, [-1]].copy()
                        df.columns = chunk
                else:
                    if isinstance(raw.columns, pd.MultiIndex):
                        close_data = {}
                        for ticker in chunk:
                            try:
                                if (ticker, 'Close') in raw.columns:
                                    close_data[ticker] = raw[(ticker, 'Close')]
                                elif ('Close', ticker) in raw.columns:
                                    close_data[ticker] = raw[('Close', ticker)]
                            except KeyError:
                                failed_tickers.append(ticker)
                        
                        if close_data:
                            df = pd.DataFrame(close_data)
                        else:
                            continue
                    else:
                        df = raw.copy()
                
                if not df.empty:
                    frames.append(df)
                    
            except Exception as e:
                logger.warning(f"Failed to download batch: {e}")
                failed_tickers.extend(chunk)
        
        if not frames:
            return pd.DataFrame()
        
        result = pd.concat(frames, axis=1, sort=True)
        result = result.fillna(method='ffill', limit=5)
        
        # Remove low-quality data
        min_data_points = int(0.7 * len(result))
        result = result.loc[:, result.count() >= min_data_points]
        
        return result

class TechnicalIndicators:
    """Enhanced technical indicators for better analysis"""
    
    @staticmethod
    def relative_strength_index(prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, window: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line, signal_line

class PortfolioOptimizer:
    """Advanced portfolio optimization methods"""
    
    @staticmethod
    def risk_parity_weights(returns: pd.DataFrame, target_vol: float = 0.15) -> pd.Series:
        """Calculate risk parity weights"""
        # Simple equal volatility weighting approach
        volatilities = returns.std() * np.sqrt(252)
        inv_vol = 1 / volatilities
        weights = inv_vol / inv_vol.sum()
        
        # Scale to target volatility
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(returns.cov() * 252, weights)))
        scale_factor = target_vol / portfolio_vol
        
        return weights * scale_factor
    
    @staticmethod
    def max_diversification_weights(returns: pd.DataFrame) -> pd.Series:
        """Calculate maximum diversification weights"""
        corr_matrix = returns.corr()
        volatilities = returns.std()
        
        # Simple heuristic: inverse correlation weighting
        avg_corr = corr_matrix.mean()
        weights = (1 - avg_corr) / volatilities
        return weights / weights.sum()

class PerformanceMetrics:
    """Enhanced performance analysis"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.06) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        return excess_returns / volatility if volatility > 0 else 0
    
    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        annual_return = returns.mean() * 252
        max_dd = PerformanceMetrics.calculate_max_drawdown(returns)
        return annual_return / abs(max_dd) if max_dd != 0 else 0

def validate_stock_data(data: pd.DataFrame, min_history_days: int = 252) -> Tuple[pd.DataFrame, List[str]]:
    """Validate and clean stock data"""
    original_count = data.shape[1]
    issues = []
    
    # Remove stocks with insufficient history
    min_data_points = int(0.8 * min_history_days)
    valid_stocks = data.loc[:, data.count() >= min_data_points]
    
    if valid_stocks.shape[1] < original_count:
        removed = original_count - valid_stocks.shape[1]
        issues.append(f"Removed {removed} stocks with insufficient history")
    
    # Remove stocks with extreme price movements (potential data errors)
    daily_returns = valid_stocks.pct_change()
    extreme_moves = (daily_returns.abs() > 0.5).any()  # More than 50% daily move
    clean_stocks = valid_stocks.loc[:, ~extreme_moves]
    
    if clean_stocks.shape[1] < valid_stocks.shape[1]:
        removed = valid_stocks.shape[1] - clean_stocks.shape[1]
        issues.append(f"Removed {removed} stocks with extreme price movements")
    
    # Remove stocks with zero or negative prices
    positive_prices = (clean_stocks > 0).all()
    final_stocks = clean_stocks.loc[:, positive_prices]
    
    if final_stocks.shape[1] < clean_stocks.shape[1]:
        removed = clean_stocks.shape[1] - final_stocks.shape[1]
        issues.append(f"Removed {removed} stocks with non-positive prices")
    
    return final_stocks, issues