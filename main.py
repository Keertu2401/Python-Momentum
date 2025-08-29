from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import yfinance as yf
import yaml
import os
import logging
from datetime import datetime, timedelta
from dateutil.tz import gettz
from typing import Optional, List, Dict, Any
import warnings
import requests
from pandas.tseries.offsets import BDay

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Config loader ----------------
CFG_PATH = "config.yaml"

def load_config():
    """Load configuration with error handling"""
    try:
        with open(CFG_PATH, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {CFG_PATH}")
        return config
    except FileNotFoundError:
        logger.error(f"Config file {CFG_PATH} not found")
        raise HTTPException(status_code=500, detail=f"Configuration file {CFG_PATH} not found")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {e}")
        raise HTTPException(status_code=500, detail=f"Error parsing configuration: {e}")

cfg = load_config()
os.makedirs(cfg.get("output_dir", "output"), exist_ok=True)

app = FastAPI(
    title="Momentum Strategy API", 
    version="2.1.0",
    description="Enhanced momentum trading strategy with T+1 settlement, trading-day alignment, and liquidity filters"
)

# ---------------- NSE Trading Calendar ----------------
NSE_HOLIDAYS_2025 = [
    "2025-01-26",  # Republic Day
    "2025-03-13",  # Holi
    "2025-03-31",  # Ram Navami
    "2025-04-14",  # Mahavir Jayanti
    "2025-04-18",  # Good Friday
    "2025-05-01",  # May Day
    "2025-08-15",  # Independence Day
    "2025-08-16",  # Parsi New Year
    "2025-09-02",  # Ganesh Chaturthi
    "2025-10-02",  # Gandhi Jayanti
    "2025-10-20",  # Dussehra
    "2025-11-01",  # Diwali Laxmi Puja
    "2025-11-04",  # Govardhan Puja
    "2025-11-05",  # Bhai Dooj
    "2025-12-25"   # Christmas
]

def get_nse_trading_days():
    """Get NSE trading days excluding weekends and holidays"""
    holidays = pd.to_datetime(NSE_HOLIDAYS_2025)
    
    def is_trading_day(date):
        if date.weekday() >= 5:  # Weekend
            return False
        if date.normalize() in holidays:  # Holiday
            return False
        return True
    
    return is_trading_day

# ---------------- Helpers ----------------
def get_last_trading_day():
    """Get the last NSE trading day with proper holiday handling"""
    local_tz = gettz(cfg.get("tz", "Asia/Kolkata"))
    now = pd.Timestamp.now(tz=local_tz)
    is_trading_day = get_nse_trading_days()
    
    # Go back until we find a trading day
    check_date = now.normalize()
    while not is_trading_day(check_date):
        check_date -= pd.Timedelta(days=1)
    
    # If current time is before market close (3:30 PM), use previous trading day
    if now.hour < 15 or (now.hour == 15 and now.minute < 30):
        check_date -= pd.Timedelta(days=1)
        while not is_trading_day(check_date):
            check_date -= pd.Timedelta(days=1)
    
    return check_date.tz_localize(None)

def count_trading_days_back(end_date, num_days):
    """Count backwards by actual trading days, not calendar days"""
    is_trading_day = get_nse_trading_days()
    
    trading_days_found = 0
    current_date = end_date
    
    while trading_days_found < num_days:
        current_date -= pd.Timedelta(days=1)
        if is_trading_day(current_date):
            trading_days_found += 1
    
    return current_date

def load_universe():
    """Load stock universe with fallback handling and liquidity filter"""
    url = cfg["universe_source"]
    try:
        logger.info(f"Downloading universe from {url}")
        df = pd.read_csv(url)
        logger.info(f"Successfully downloaded {len(df)} stocks from NSE")
    except Exception as e:
        logger.warning(f"Failed to download from {url}: {e}")
        # Try alternative sources or fallback
        fallback_files = ["ind_nifty500list.csv", "nifty500.csv", "universe.csv"]
        df = None
        for file in fallback_files:
            if os.path.exists(file):
                try:
                    df = pd.read_csv(file)
                    logger.info(f"Using fallback file: {file}")
                    break
                except Exception:
                    continue
        
        if df is None:
            raise HTTPException(
                status_code=500, 
                detail="Failed to load stock universe from URL and no fallback file found"
            )
    
    # Handle different column name variations
    symbol_columns = ["Symbol", "SYMBOL", "symbol", "ticker", "Ticker"]
    col = None
    for c in symbol_columns:
        if c in df.columns:
            col = c
            break
    
    if col is None:
        raise HTTPException(
            status_code=500, 
            detail=f"No symbol column found in universe data. Available columns: {df.columns.tolist()}"
        )
    
    # Clean and format symbols
    syms = df[col].astype(str).str.strip().str.upper().tolist()
    # Remove any invalid symbols
    syms = [s for s in syms if s and s != 'NAN' and len(s) > 0]
    # Add .NS suffix for NSE stocks
    universe = [s + ".NS" for s in syms]
    
    logger.info(f"Loaded {len(universe)} symbols from universe")
    return universe

def yf_download_with_volume(tickers, start, end, batch=100):
    """Enhanced download with volume data for liquidity filtering"""
    if not tickers:
        return pd.DataFrame(), pd.DataFrame()
    
    logger.info(f"Downloading OHLCV data for {len(tickers)} tickers from {start.date()} to {end.date()}")
    
    price_frames = []
    volume_frames = []
    failed_tickers = []
    
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i+batch]
        logger.info(f"Downloading batch {i//batch + 1}/{(len(tickers)-1)//batch + 1} ({len(chunk)} tickers)")
        
        try:
            raw = yf.download(
                chunk, 
                start=start, 
                end=end + pd.Timedelta(days=1), 
                auto_adjust=True, 
                progress=False,
                threads=True,
                group_by='ticker'
            )
            
            if raw.empty:
                failed_tickers.extend(chunk)
                continue
            
            # Extract Close and Volume data
            if len(chunk) == 1:
                # Single ticker case
                ticker = chunk[0]
                if all(col in raw.columns for col in ['Close', 'Volume']):
                    price_data = {ticker: raw['Close']}
                    volume_data = {ticker: raw['Volume']}
                else:
                    failed_tickers.append(ticker)
                    continue
            else:
                # Multi-ticker case
                price_data = {}
                volume_data = {}
                
                for ticker in chunk:
                    try:
                        # Try different column arrangements
                        if (ticker, 'Close') in raw.columns and (ticker, 'Volume') in raw.columns:
                            price_data[ticker] = raw[(ticker, 'Close')]
                            volume_data[ticker] = raw[(ticker, 'Volume')]
                        elif ('Close', ticker) in raw.columns and ('Volume', ticker) in raw.columns:
                            price_data[ticker] = raw[('Close', ticker)]
                            volume_data[ticker] = raw[('Volume', ticker)]
                        else:
                            failed_tickers.append(ticker)
                    except KeyError:
                        failed_tickers.append(ticker)
            
            # Create dataframes for this batch
            if price_data:
                price_df = pd.DataFrame(price_data)
                volume_df = pd.DataFrame(volume_data)
                
                # Clean the data
                price_df = price_df.dropna(how='all')
                volume_df = volume_df.dropna(how='all')
                
                if not price_df.empty and not volume_df.empty:
                    price_frames.append(price_df)
                    volume_frames.append(volume_df)
                    
        except Exception as e:
            logger.warning(f"Failed to download batch {i//batch + 1}: {e}")
            failed_tickers.extend(chunk)
            continue
    
    if failed_tickers:
        logger.warning(f"Failed to download {len(failed_tickers)} tickers")
    
    if not price_frames:
        logger.error("No price data downloaded successfully")
        return pd.DataFrame(), pd.DataFrame()
    
    # Combine all data
    prices = pd.concat(price_frames, axis=1, sort=True)
    volumes = pd.concat(volume_frames, axis=1, sort=True)
    
    # Forward fill missing values (up to 3 days)
    prices = prices.fillna(method='ffill', limit=3)
    volumes = volumes.fillna(method='ffill', limit=3)
    
    # Remove columns with insufficient data
    min_data_points = int(0.6 * len(prices))
    valid_cols = prices.columns[prices.count() >= min_data_points]
    prices = prices[valid_cols]
    volumes = volumes[valid_cols]
    
    logger.info(f"Downloaded data for {prices.shape[1]} tickers with {prices.shape[0]} days")
    return prices, volumes

def compute_dates_idx_trading_days(prices: pd.DataFrame, ref_date: pd.Timestamp):
    """Compute date indices using actual trading days, not calendar days"""
    if prices.empty:
        raise ValueError("Price data is empty")

    # Make timezone-naive for comparison
    available_dates = prices.index
    if isinstance(available_dates, pd.DatetimeIndex) and available_dates.tz is not None:
        available_dates = available_dates.tz_localize(None)
    if isinstance(ref_date, pd.Timestamp) and ref_date.tzinfo is not None:
        ref_date = ref_date.tz_localize(None)

    # Find reference date in actual data
    if ref_date not in available_dates:
        earlier_dates = available_dates[available_dates <= ref_date]
        if len(earlier_dates) == 0:
            raise ValueError(f"No data available before reference date {ref_date.date()}")
        ref_date = earlier_dates[-1]
        logger.info(f"Adjusted reference date to last available: {ref_date.date()}")

    # Use actual trading days for lookbacks
    trading_dates = available_dates[available_dates <= ref_date]
    pos = len(trading_dates) - 1  # Position of ref_date
    
    # Calculate lookback positions using actual trading days
    lookback_6m = cfg.get("lookback_6m_trading_days", 126)  # ~6 months of trading
    lookback_12m = cfg.get("lookback_12m_trading_days", 252)  # ~12 months of trading
    lookback_1m = cfg.get("lookback_1m_trading_days", 21)   # ~1 month of trading
    sma_window = cfg.get("sma_window_trading_days", 200)    # 200 trading days
    vol_window = cfg.get("vol_window_trading_days", 60)     # 60 trading days
    
    min_needed = max(lookback_12m, sma_window, vol_window)
    if pos < min_needed:
        raise ValueError(
            f"Insufficient trading history at {ref_date.date()}. "
            f"Need {min_needed} trading days, but only have {pos}"
        )

    idx_today = pos
    idx_6m = max(0, pos - lookback_6m)
    idx_12m = max(0, pos - lookback_12m)
    idx_1m_back = max(0, pos - lookback_1m)
    
    return idx_today, idx_6m, idx_12m, idx_1m_back, trading_dates

def apply_liquidity_filter(prices: pd.DataFrame, volumes: pd.DataFrame, min_turnover_cr=10):
    """Filter out illiquid stocks based on average daily turnover"""
    if volumes.empty:
        logger.warning("No volume data available for liquidity filtering")
        return prices.columns.tolist()
    
    # Calculate average daily turnover for last 30 trading days
    recent_prices = prices.tail(30)
    recent_volumes = volumes.tail(30)
    
    # Calculate turnover = price * volume (in crores)
    avg_turnover = {}
    for ticker in prices.columns:
        if ticker in recent_prices.columns and ticker in recent_volumes.columns:
            price_series = recent_prices[ticker].dropna()
            volume_series = recent_volumes[ticker].dropna()
            
            if len(price_series) >= 15 and len(volume_series) >= 15:  # At least 15 days data
                # Align data
                common_dates = price_series.index.intersection(volume_series.index)
                if len(common_dates) >= 15:
                    turnover = (price_series[common_dates] * volume_series[common_dates]).mean()
                    avg_turnover[ticker] = turnover / 1e7  # Convert to crores
    
    # Filter tickers with sufficient liquidity
    liquid_tickers = [
        ticker for ticker, turnover in avg_turnover.items() 
        if turnover >= min_turnover_cr
    ]
    
    logger.info(f"Liquidity filter: {len(liquid_tickers)}/{len(prices.columns)} stocks passed (min ₹{min_turnover_cr}Cr turnover)")
    return liquid_tickers

def momentum_frame_fixed(prices: pd.DataFrame, volumes: pd.DataFrame, ref_date: pd.Timestamp):
    """Enhanced momentum calculation with T+1 settlement awareness and trading-day alignment"""
    
    # FIX 1: T+1 Settlement - Use previous day's close for signal generation
    # This ensures we're using prices that were available when making the decision
    signal_date = ref_date - pd.Timedelta(days=1)
    
    # Find the actual trading day for signal
    trading_dates = prices.index[prices.index <= signal_date]
    if len(trading_dates) == 0:
        raise ValueError(f"No trading data available before {signal_date.date()}")
    
    actual_signal_date = trading_dates[-1]
    logger.info(f"Using T-1 signal date: {actual_signal_date.date()} (T+1 settlement fix)")
    
    # Get indices using trading days
    idx_today, idx_6m, idx_12m, idx_1m_back, trading_dates = compute_dates_idx_trading_days(prices, actual_signal_date)
    
    # FIX 4: Apply liquidity filter first
    min_turnover = cfg.get("min_daily_turnover_cr", 10)
    liquid_tickers = apply_liquidity_filter(prices, volumes, min_turnover)
    
    if not liquid_tickers:
        raise ValueError("No liquid stocks found after applying turnover filter")
    
    # Filter prices to liquid stocks only
    liquid_prices = prices[liquid_tickers]
    
    # Get price points for momentum calculation
    today = liquid_prices.iloc[idx_today]
    sixm = liquid_prices.iloc[idx_6m] 
    twlv = liquid_prices.iloc[idx_12m]
    one_m = liquid_prices.iloc[idx_1m_back]
    
    # Find tickers with complete data across all time points
    valid_mask = (
        today.notna() & 
        sixm.notna() & 
        twlv.notna() & 
        one_m.notna() &
        (today > 0) &
        (sixm > 0) &
        (twlv > 0) &
        (one_m > 0)
    )
    valid = today[valid_mask].index
    
    if len(valid) == 0:
        raise ValueError("No valid liquid tickers with complete price history")
    
    logger.info(f"Computing momentum for {len(valid)} valid liquid tickers")
    
    # Calculate volatility using trading days
    vol_window_days = cfg.get("vol_window_trading_days", 60)
    vol_start = max(0, idx_today - vol_window_days + 1)
    vol_window = liquid_prices[valid].iloc[vol_start:idx_today + 1]
    
    daily_returns = vol_window.pct_change().dropna()
    vol = daily_returns.std() * np.sqrt(252) * 100  # Annualized volatility
    
    # Calculate returns
    ret_6m = ((today[valid] / sixm[valid]) - 1.0) * 100
    ret_12m_ex1 = ((one_m[valid] / twlv[valid]) - 1.0) * 100
    
    # Calculate 200-day moving average using trading days
    sma_window_days = cfg.get("sma_window_trading_days", 200)
    sma_start = max(0, idx_today - sma_window_days + 1)
    sma_window = liquid_prices[valid].iloc[sma_start:idx_today + 1]
    sma_200 = sma_window.mean()
    above_200dma = today[valid] > sma_200
    
    # Calculate average daily turnover for final validation
    turnover_window = min(30, idx_today + 1)
    recent_prices = liquid_prices[valid].iloc[-turnover_window:]
    recent_volumes = volumes[valid].iloc[-turnover_window:] if not volumes.empty else pd.DataFrame()
    
    avg_turnover = pd.Series(index=valid, dtype=float)
    if not recent_volumes.empty:
        for ticker in valid:
            if ticker in recent_volumes.columns:
                price_vol = recent_prices[ticker] * recent_volumes[ticker]
                avg_turnover[ticker] = price_vol.mean() / 1e7  # In crores
            else:
                avg_turnover[ticker] = 0
    else:
        avg_turnover = avg_turnover.fillna(0)
    
    # Enhanced momentum score
    risk_adjusted_vol = vol + 0.01
    momentum_score = (0.4 * ret_6m + 0.6 * ret_12m_ex1) / risk_adjusted_vol
    
    # Create results DataFrame
    df = pd.DataFrame({
        "Price": today[valid],
        "6M_Return(%)": ret_6m,
        "12M_ex1_Return(%)": ret_12m_ex1,
        "Volatility(%)": vol,
        "MomentumScore": momentum_score,
        "Above_200DMA": above_200dma,
        "Positive_6M": ret_6m > 0,
        "Positive_12M_ex1": ret_12m_ex1 > 0,
        "SMA_200": sma_200,
        "AvgTurnover_Cr": avg_turnover
    })
    
    # Remove any remaining NaN or infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Reference information with trading day awareness
    ref_info = {
        "SignalDate": actual_signal_date,  # T-1 date used for signal
        "ExecutionDate": ref_date,         # T date for execution
        "Date_6M": trading_dates[idx_6m],
        "Date_12M": trading_dates[idx_12m], 
        "Date_1MBack": trading_dates[idx_1m_back],
        "UniverseCount": len(liquid_tickers),
        "ValidCount": len(valid),
        "EligibleCount": int(((df["Above_200DMA"]) & (df["Positive_6M"]) & (df["Positive_12M_ex1"])).sum()),
        "TradingDaysUsed": {
            "6M_Lookback": lookback_6m := cfg.get("lookback_6m_trading_days", 126),
            "12M_Lookback": cfg.get("lookback_12m_trading_days", 252),
            "SMA_Window": cfg.get("sma_window_trading_days", 200),
            "Vol_Window": cfg.get("vol_window_trading_days", 60)
        }
    }
    
    logger.info(f"Momentum calculation complete. Eligible stocks: {ref_info['EligibleCount']}")
    return df, ref_info

def suggest_allocations_t1(top_df: pd.DataFrame, capital: float, top_n: int):
    """Enhanced allocation with T+1 settlement consideration"""
    if top_df.empty:
        return top_df
    
    top_df = top_df.copy()
    
    # Equal weight allocation
    alloc_per = capital / min(len(top_df), top_n)
    
    # FIX: Use a conservative price estimate for T+1 execution
    # Add 0.5% buffer for potential gap-up on strong momentum stocks
    execution_price_estimate = top_df["Price"] * 1.005
    
    # Calculate quantities (whole shares only)
    qty = np.floor(alloc_per / execution_price_estimate).astype(int)
    amt_estimated = qty * execution_price_estimate
    amt_signal_price = qty * top_df["Price"]  # For reference
    
    # Calculate allocation percentages based on estimated execution
    total_invested = amt_estimated.sum()
    allocation_pct = (amt_estimated / total_invested * 100) if total_invested > 0 else 0
    
    top_df["Qty"] = qty
    top_df["Amount_Signal"] = amt_signal_price  # Based on signal price
    top_df["Amount_Estimated"] = amt_estimated  # Expected execution amount
    top_df["Price_Gap_Buffer"] = execution_price_estimate - top_df["Price"]
    top_df["Allocation(%)"] = allocation_pct
    top_df["Unutilized"] = alloc_per - amt_estimated
    
    return top_df

# ---------------- Pydantic Models ----------------
class StockData(BaseModel):
    Ticker: str
    Rank: Optional[int] = None
    Price: float
    Qty: Optional[int] = None
    Amount_Signal: Optional[float] = None
    Amount_Estimated: Optional[float] = None
    Price_Gap_Buffer: Optional[float] = None
    Allocation: Optional[float] = Field(None, alias="Allocation(%)")
    MomentumScore: float
    Return_6M: float = Field(alias="6M_Return(%)")
    Return_12M_ex1: float = Field(alias="12M_ex1_Return(%)")
    Volatility: float = Field(alias="Volatility(%)")
    Above_200DMA: bool
    Positive_6M: bool
    AvgTurnover_Cr: float

class RunResponse(BaseModel):
    reference_dates: Dict[str, Any]
    eligible: List[Dict[str, Any]]
    top: List[Dict[str, Any]]
    csv_paths: Dict[str, str]
    summary: Dict[str, Any]

class LevelData(BaseModel):
    Ticker: str
    Close: float
    ATR14: float
    Stop_2xATR: float
    Stop_1_5xATR: float
    Target_2xATR: float

class LevelsResponse(BaseModel):
    levels: List[LevelData]

# ---------------- API Endpoints ----------------
@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Momentum Strategy API v2.1.0",
        "description": "Enhanced momentum trading strategy with T+1 settlement and trading-day alignment",
        "endpoints": {
            "health": "/health",
            "run_strategy": "/run",
            "get_levels": "/levels", 
            "market_status": "/market-status",
            "trading_calendar": "/trading-calendar",
            "webhook_rebalance": "/webhook/rebalance",
            "portfolio_compare": "/webhook/portfolio-compare",
            "validate_orders": "/webhook/validate-orders"
        },
        "documentation": "/docs",
        "features": [
            "T+1 settlement awareness",
            "Trading-day aligned lookbacks", 
            "NSE holiday calendar integration",
            "Liquidity filtering",
            "Enhanced risk management"
        ],
        "status": "live",
        "timestamp": datetime.now(gettz(cfg.get("tz", "Asia/Kolkata"))).isoformat()
    }

@app.get("/health")
def health():
    """Health check endpoint"""
    is_trading_day = get_nse_trading_days()
    current_time = datetime.now(gettz(cfg.get("tz", "Asia/Kolkata")))
    
    return {
        "status": "healthy",
        "timestamp": current_time.isoformat(),
        "version": "2.1.0",
        "market_status": {
            "is_trading_day": is_trading_day(current_time),
            "market_open": (
                current_time.weekday() < 5 and 
                is_trading_day(current_time) and
                ((current_time.hour == 9 and current_time.minute >= 15) or 
                 (9 < current_time.hour < 15) or 
                 (current_time.hour == 15 and current_time.minute <= 30))
            )
        }
    }

@app.get("/run", response_model=RunResponse)
def run(
    capital: Optional[float] = Query(None, description="Capital amount (default from config)", gt=0),
    top_n: Optional[int] = Query(None, description="Number of top stocks (default from config)", gt=0, le=50),
    price_cap: Optional[float] = Query(None, description="Maximum price per share in INR"),
    ref_date: Optional[str] = Query(None, description="Execution date YYYY-MM-DD (default: next trading day)"),
    min_momentum_score: Optional[float] = Query(None, description="Minimum momentum score filter"),
    min_turnover: Optional[float] = Query(None, description="Minimum daily turnover in crores", gt=0),
):
    """
    Run enhanced momentum strategy with T+1 settlement and trading-day fixes
    """
    try:
        # Validate and set parameters
        _capital = capital if capital is not None else cfg["capital"]
        _top_n = top_n if top_n is not None else cfg["top_n"]
        _cap = price_cap if price_cap is not None else cfg.get("price_cap")
        _min_turnover = min_turnover if min_turnover is not None else cfg.get("min_daily_turnover_cr", 10)
        out_dir = cfg["output_dir"]
        
        # Parse reference date (execution date)
        if ref_date:
            try:
                end_dt = pd.Timestamp(ref_date)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        else:
            end_dt = get_last_trading_day() + pd.Timedelta(days=1)  # Next trading day
        
        # FIX 5: Validate it's a trading day
        is_trading_day = get_nse_trading_days()
        if not is_trading_day(end_dt):
            raise HTTPException(
                status_code=400, 
                detail=f"Execution date {end_dt.date()} is not a trading day (weekend/holiday)"
            )
        
        # Load universe and download data with volume
        universe = load_universe()
        
        # Use sufficient history for trading-day aligned calculations
        history_days = cfg.get("history_days", 400)  # Increased for trading-day buffer
        start_dt = end_dt - pd.Timedelta(days=history_days)
        
        prices, volumes = yf_download_with_volume(universe, start_dt, end_dt)
        if prices.empty:
            raise HTTPException(status_code=500, detail="No price data downloaded")
        
        # Calculate momentum metrics with all fixes applied
        mom_df, ref_info = momentum_frame_fixed(prices, volumes, end_dt)
        
        if mom_df.empty:
            raise HTTPException(status_code=500, detail="No valid momentum data calculated")
        
        # Enhanced filtering with liquidity
        eligible = mom_df[
            (mom_df["Above_200DMA"]) & 
            (mom_df["Positive_6M"]) &
            (mom_df["Positive_12M_ex1"]) &
            (mom_df["AvgTurnover_Cr"] >= _min_turnover)  # Liquidity filter
        ].copy()
        
        # Apply price cap filter
        if _cap is not None:
            eligible = eligible[eligible["Price"] <= _cap]
        
        # Apply minimum momentum score filter
        if min_momentum_score is not None:
            eligible = eligible[eligible["MomentumScore"] >= min_momentum_score]
        
        # Rank by momentum score
        ranked = eligible.sort_values("MomentumScore", ascending=False).copy()
        top = ranked.head(_top_n).copy()
        
        # Calculate T+1 aware allocations
        if not top.empty:
            top = suggest_allocations_t1(top, _capital, _top_n)
            top.insert(0, "Rank", range(1, len(top) + 1))
        
        # Enhanced summary with T+1 awareness
        summary = {
            "total_universe": len(universe),
            "liquid_stocks": len(liquid_tickers) if 'liquid_tickers' in locals() else len(universe),
            "data_available": prices.shape[1],
            "momentum_calculated": len(mom_df),
            "eligible_stocks": len(eligible),
            "selected_stocks": len(top),
            "total_capital": _capital,
            "invested_amount_signal": float(top["Amount_Signal"].sum()) if not top.empty else 0,
            "invested_amount_estimated": float(top["Amount_Estimated"].sum()) if not top.empty else 0,
            "utilization_pct": float(top["Amount_Estimated"].sum() / _capital * 100) if not top.empty else 0,
            "avg_momentum_score": float(top["MomentumScore"].mean()) if not top.empty else 0,
            "execution_buffer": float(top["Price_Gap_Buffer"].sum()) if not top.empty else 0,
            "filters_applied": {
                "above_200dma": True,
                "positive_6m": True,
                "positive_12m_ex1": True,
                "price_cap": _cap,
                "min_momentum_score": min_momentum_score,
                "min_turnover_cr": _min_turnover,
                "t_plus_1_settlement": True,
                "trading_days_aligned": True
            },
            "settlement_info": {
                "signal_generated_on": str(ref_info["SignalDate"]),
                "execution_planned_for": str(ref_info["ExecutionDate"]),
                "settlement_cycle": "T+1"
            }
        }
        
        # Save enhanced CSV files
        csv_paths = {}
        if cfg.get("write_csv", True):
            timestamp = end_dt.strftime("%Y%m%d")
            
            # Save with enhanced metadata
            prices.to_csv(os.path.join(out_dir, f"prices_snapshot_{timestamp}.csv"))
            mom_df.to_csv(os.path.join(out_dir, f"momentum_scores_{timestamp}.csv"))
            eligible.to_csv(os.path.join(out_dir, f"eligible_stocks_{timestamp}.csv"))
            
            if not top.empty:
                top.to_csv(os.path.join(out_dir, f"top_selection_{timestamp}.csv"), index=True)
            
            # Enhanced reference info
            ref_df = pd.DataFrame([ref_info])
            ref_df.to_csv(os.path.join(out_dir, f"reference_dates_{timestamp}.csv"), index=False)
            
            csv_paths = {
                "prices": f"prices_snapshot_{timestamp}.csv",
                "scores": f"momentum_scores_{timestamp}.csv", 
                "eligible": f"eligible_stocks_{timestamp}.csv",
                "top": f"top_selection_{timestamp}.csv",
                "reference_dates": f"reference_dates_{timestamp}.csv"
            }
        
        # Prepare response data
        eligible_records = eligible.reset_index().rename(columns={"index": "Ticker"}).round(4).to_dict(orient="records")
        top_records = top.reset_index().rename(columns={"index": "Ticker"}).round(4).to_dict(orient="records") if not top.empty else []
        
        return {
            "reference_dates": {k: str(v) for k, v in ref_info.items()},
            "eligible": eligible_records,
            "top": top_records,
            "csv_paths": csv_paths,
            "summary": summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in run endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/levels", response_model=LevelsResponse)
def levels(tickers: str = Query(..., description="Comma-separated tickers (e.g., BEL.NS,MFSL.NS)")):
    """
    Enhanced stop-loss and target calculations using ATR with T+1 awareness
    """
    try:
        # Parse and validate tickers
        tick_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if not tick_list:
            raise HTTPException(status_code=400, detail="No valid tickers provided")
        
        # Ensure .NS suffix for NSE stocks
        tick_list = [t if t.endswith('.NS') else t + '.NS' for t in tick_list]
        
        # Use last trading day for levels calculation
        end = get_last_trading_day()
        start = end - pd.Timedelta(days=365)  # 1 year of data for better ATR
        
        results = []
        
        for ticker in tick_list:
            try:
                # Download OHLCV data
                raw = yf.download(
                    ticker, 
                    start=start, 
                    end=end + pd.Timedelta(days=1), 
                    auto_adjust=True, 
                    progress=False
                )
                
                if raw.empty or len(raw) < 20:
                    logger.warning(f"Insufficient data for {ticker}")
                    continue
                
                # Ensure required columns exist
                required_cols = ["High", "Low", "Close"]
                if not all(col in raw.columns for col in required_cols):
                    logger.warning(f"Missing OHLC data for {ticker}")
                    continue
                
                # Calculate True Range
                high_low = raw["High"] - raw["Low"]
                high_close_prev = (raw["High"] - raw["Close"].shift(1)).abs()
                low_close_prev = (raw["Low"] - raw["Close"].shift(1)).abs()
                
                tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
                
                # Calculate ATR (14-period)
                atr = tr.rolling(window=14, min_periods=10).mean()
                
                # Get latest values (T-1 for signal generation)
                last_close = float(raw["Close"].iloc[-1])
                last_atr = float(atr.iloc[-1])
                
                # Enhanced stop-loss and target calculations
                # Conservative stops for T+1 settlement
                stop_2x = last_close - (2.0 * last_atr)
                stop_1_5x = last_close - (1.5 * last_atr) 
                target_2x = last_close + (2.0 * last_atr)
                target_3x = last_close + (3.0 * last_atr)  # Additional target
                
                results.append({
                    "Ticker": ticker,
                    "Close": round(last_close, 2),
                    "ATR14": round(last_atr, 2),
                    "Stop_2xATR": round(stop_2x, 2),
                    "Stop_1_5xATR": round(stop_1_5x, 2),
                    "Target_2xATR": round(target_2x, 2),
                    "Target_3xATR": round(target_3x, 2),
                    "Risk_Reward_2x": round((target_2x - last_close) / (last_close - stop_2x), 2),
                    "ATR_Percent": round((last_atr / last_close) * 100, 2)
                })
                
            except Exception as e:
                logger.warning(f"Error processing {ticker}: {e}")
                continue
        
        if not results:
            raise HTTPException(status_code=404, detail="No valid data found for any ticker")
        
        return {"levels": results}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in levels endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ---------------- Enhanced N8N Integration ----------------
from fastapi import BackgroundTasks

class WebhookData(BaseModel):
    source: str = "n8n"
    execution_id: Optional[str] = None
    capital: Optional[float] = None
    top_n: Optional[int] = None
    price_cap: Optional[float] = None
    ref_date: Optional[str] = None
    callback_url: Optional[str] = None
    min_turnover: Optional[float] = None

@app.post("/webhook/rebalance")
async def webhook_rebalance(webhook_data: WebhookData):
    """
    Enhanced webhook endpoint with T+1 settlement awareness
    """
    try:
        logger.info(f"Webhook triggered by {webhook_data.source}, execution_id: {webhook_data.execution_id}")
        
        # Validate execution date is a trading day
        if webhook_data.ref_date:
            exec_date = pd.Timestamp(webhook_data.ref_date)
            is_trading_day = get_nse_trading_days()
            if not is_trading_day(exec_date):
                return {
                    "error": "Invalid execution date - not a trading day",
                    "suggested_date": str(get_last_trading_day() + pd.Timedelta(days=1)),
                    "webhook_id": webhook_data.execution_id
                }
        
        # Extract parameters
        _capital = webhook_data.capital or cfg['capital']
        _top_n = webhook_data.top_n or cfg['top_n']
        _cap = webhook_data.price_cap or cfg.get('price_cap')
        _min_turnover = webhook_data.min_turnover or cfg.get('min_daily_turnover_cr', 10)
        
        # Run enhanced strategy analysis
        result = run(
            capital=_capital, 
            top_n=_top_n, 
            price_cap=_cap, 
            ref_date=webhook_data.ref_date,
            min_turnover=_min_turnover
        )
        
        # Add webhook metadata with T+1 awareness
        result['webhook'] = {
            'triggered_at': datetime.now().isoformat(),
            'source': webhook_data.source,
            'execution_id': webhook_data.execution_id,
            'callback_url': webhook_data.callback_url,
            'settlement_cycle': 'T+1',
            'trading_day_validated': True
        }
        
        # Enhanced callback notification
        if webhook_data.callback_url:
            logger.info(f"Would send T+1 aware callback to: {webhook_data.callback_url}")
        
        return result
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/webhook/status/{execution_id}")
async def webhook_status(execution_id: str):
    """
    Enhanced status check with market timing
    """
    current_time = datetime.now(gettz(cfg.get("tz", "Asia/Kolkata")))
    is_trading_day_fn = get_nse_trading_days()
    
    return {
        "execution_id": execution_id, 
        "status": "completed",
        "timestamp": current_time.isoformat(),
        "market_info": {
            "is_trading_day": is_trading_day_fn(current_time),
            "next_trading_day": str(get_last_trading_day() + pd.Timedelta(days=1)),
            "settlement_cycle": "T+1"
        }
    }

@app.post("/webhook/portfolio-compare")
async def webhook_portfolio_compare(
    target_portfolio: List[Dict[str, Any]],
    current_holdings: List[Dict[str, Any]]
):
    """
    Enhanced portfolio comparison with T+1 settlement and execution timing
    """
    try:
        rebalance_orders = []
        current_map = {holding['ticker']: holding for holding in current_holdings}
        
        # Check market timing for execution
        current_time = datetime.now(gettz(cfg.get("tz", "Asia/Kolkata")))
        is_trading_day_fn = get_nse_trading_days()
        is_market_open = (
            is_trading_day_fn(current_time) and
            ((current_time.hour == 9 and current_time.minute >= 15) or 
             (9 < current_time.hour < 15) or 
             (current_time.hour == 15 and current_time.minute <= 30))
        )
        
        # Generate rebalancing orders with T+1 awareness
        for target in target_portfolio:
            ticker = target.get('Ticker', target.get('ticker'))
            target_qty = target.get('Qty', target.get('quantity', 0))
            current = current_map.get(ticker, {'quantity': 0})
            current_qty = current.get('quantity', 0)
            
            qty_diff = target_qty - current_qty
            
            if abs(qty_diff) > 0:
                order = {
                    'ticker': ticker,
                    'action': 'BUY' if qty_diff > 0 else 'SELL',
                    'quantity': abs(qty_diff),
                    'target_qty': target_qty,
                    'current_qty': current_qty,
                    'signal_price': target.get('Price', target.get('price', 0)),
                    'estimated_execution_price': target.get('Amount_Estimated', 0) / target_qty if target_qty > 0 else 0,
                    'momentum_score': target.get('MomentumScore', 0),
                    'priority': target.get('Rank', 999),
                    'avg_turnover_cr': target.get('AvgTurnover_Cr', 0),
                    'settlement': 'T+1'
                }
                rebalance_orders.append(order)
        
        # Handle positions to exit (with liquidity check)
        for ticker, holding in current_map.items():
            if not any(t.get('Ticker', t.get('ticker')) == ticker for t in target_portfolio):
                if holding.get('quantity', 0) > 0:
                    rebalance_orders.append({
                        'ticker': ticker,
                        'action': 'SELL',
                        'quantity': holding['quantity'],
                        'target_qty': 0,
                        'current_qty': holding['quantity'],
                        'reason': 'EXIT_POSITION',
                        'priority': 1,
                        'settlement': 'T+1'
                    })
        
        # Sort orders: Sells first (for liquidity), then buys by priority
        rebalance_orders.sort(key=lambda x: (x['action'] == 'BUY', x['priority']))
        
        return {
            'timestamp': current_time.isoformat(),
            'market_status': {
                'is_open': is_market_open,
                'is_trading_day': is_trading_day_fn(current_time),
                'next_trading_day': str(get_last_trading_day() + pd.Timedelta(days=1))
            },
            'total_orders': len(rebalance_orders),
            'buy_orders': len([o for o in rebalance_orders if o['action'] == 'BUY']),
            'sell_orders': len([o for o in rebalance_orders if o['action'] == 'SELL']),
            'orders': rebalance_orders,
            'settlement_info': {
                'cycle': 'T+1',
                'execution_timing': 'Next trading session if market closed'
            }
        }
        
    except Exception as e:
        logger.error(f"Portfolio compare error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook/validate-orders")
async def webhook_validate_orders(
    orders: List[Dict[str, Any]], 
    max_position_size: float = 0.15,
    min_liquidity_cr: float = 5.0
):
    """
    Enhanced order validation with liquidity and T+1 settlement checks
    """
    try:
        current_time = datetime.now(gettz(cfg.get("tz", "Asia/Kolkata")))
        is_trading_day_fn = get_nse_trading_days()
        
        is_market_open = (
            is_trading_day_fn(current_time) and
            ((current_time.hour == 9 and current_time.minute >= 15) or 
             (9 < current_time.hour < 15) or 
             (current_time.hour == 15 and current_time.minute <= 30))
        )
        
        validated_orders = []
        rejected_orders = []
        warnings = []
        
        if not is_market_open:
            warnings.append("Market is closed. Orders will execute in next trading session (T+1 settlement).")
        
        # Enhanced validation for each order
        for order in orders:
            rejection_reasons = []
            
            # Position size check
            estimated_value = order.get('estimated_value', order.get('quantity', 0) * order.get('signal_price', 0))
            total_capital = order.get('total_capital', cfg['capital'])
            position_size = estimated_value / total_capital if total_capital > 0 else 0
            
            if position_size > max_position_size:
                rejection_reasons.append(f"Position size {position_size*100:.2f}% exceeds limit {max_position_size*100:.2f}%")
            
            # Liquidity check
            avg_turnover = order.get('avg_turnover_cr', 0)
            if avg_turnover < min_liquidity_cr:
                rejection_reasons.append(f"Insufficient liquidity: ₹{avg_turnover:.1f}Cr < ₹{min_liquidity_cr}Cr minimum")
            
            # T+1 settlement awareness
            if order.get('action') == 'BUY':
                signal_price = order.get('signal_price', 0)
                estimated_exec_price = order.get('estimated_execution_price', signal_price * 1.005)
                gap_risk = ((estimated_exec_price - signal_price) / signal_price) * 100
                
                if gap_risk > 2.0:  # More than 2% gap risk
                    warnings.append(f"{order.get('ticker')}: High gap risk {gap_risk:.2f}% - consider smaller position")
            
            # Final validation
            if rejection_reasons:
                rejected_orders.append({
                    **order,
                    'rejection_reasons': rejection_reasons
                })
            else:
                validated_orders.append({
                    **order,
                    'validated_at': current_time.isoformat(),
                    'settlement_date': str(get_last_trading_day() + pd.Timedelta(days=2))  # T+1 settlement
                })
        
        return {
            'timestamp': current_time.isoformat(),
            'market_open': is_market_open,
            'validated_orders': validated_orders,
            'rejected_orders': rejected_orders,
            'warnings': warnings,
            'validation_summary': {
                'total_orders': len(orders),
                'validated': len(validated_orders),
                'rejected': len(rejected_orders),
                'settlement_cycle': 'T+1'
            },
            'execution_info': {
                'recommended_execution': 'Market open next trading day',
                'settlement_completion': str(get_last_trading_day() + pd.Timedelta(days=2))
            }
        }
        
    except Exception as e:
        logger.error(f"Order validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market-status")
def market_status():
    """
    Get current NSE market status with holiday calendar
    """
    try:
        current_time = datetime.now(gettz(cfg.get("tz", "Asia/Kolkata")))
        is_trading_day_fn = get_nse_trading_days()
        
        is_trading_day = is_trading_day_fn(current_time)
        is_market_hours = (
            ((current_time.hour == 9 and current_time.minute >= 15) or 
             (9 < current_time.hour < 15) or 
             (current_time.hour == 15 and current_time.minute <= 30))
        )
        
        is_market_open = is_trading_day and is_market_hours
        
        # Find next trading day
        next_trading = current_time + pd.Timedelta(days=1)
        while not is_trading_day_fn(next_trading):
            next_trading += pd.Timedelta(days=1)
        
        return {
            "current_time": current_time.isoformat(),
            "is_trading_day": is_trading_day,
            "is_market_hours": is_market_hours,
            "is_market_open": is_market_open,
            "next_trading_day": str(next_trading.date()),
            "last_trading_day": str(get_last_trading_day().date()),
            "market_hours": "09:15 - 15:30 IST",
            "settlement_cycle": "T+1",
            "upcoming_holidays": [h for h in NSE_HOLIDAYS_2025 if pd.Timestamp(h) > current_time][:5]
        }
        
    except Exception as e:
        logger.error(f"Market status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trading-calendar")
def trading_calendar(year: int = Query(2025, description="Year for trading calendar")):
    """
    Get NSE trading calendar for given year
    """
    if year != 2025:
        return {
            "error": f"Trading calendar for {year} not available",
            "available_year": 2025,
            "holidays_2025": NSE_HOLIDAYS_2025
        }
    
    return {
        "year": 2025,
        "holidays": NSE_HOLIDAYS_2025,
        "total_holidays": len(NSE_HOLIDAYS_2025),
        "trading_days_approx": 365 - len(NSE_HOLIDAYS_2025) - (52 * 2),  # Exclude weekends
        "settlement_cycle": "T+1"
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Enhanced global exception handler"""
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error occurred",
            "timestamp": datetime.now().isoformat(),
            "settlement_info": "Check T+1 settlement timing for trade execution"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
