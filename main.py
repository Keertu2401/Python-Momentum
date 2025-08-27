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
    version="2.0.0",
    description="Advanced momentum trading strategy with optimized performance and bug fixes"
)

# ---------------- Helpers ----------------
def now_local():
    """Get current local time (timezone-naive date) and find the last trading day"""
    local_tz = gettz(cfg.get("tz", "Asia/Kolkata"))
    now = pd.Timestamp.now(tz=local_tz)
    
    # If it's weekend or before market hours, go to previous trading day
    if now.weekday() >= 5:  # Saturday or Sunday
        days_back = now.weekday() - 4  # Go to Friday
        candidate = (now - pd.Timedelta(days=days_back)).normalize()
    elif now.hour < 9:  # Before market open
        candidate = (now - pd.Timedelta(days=1)).normalize()
    else:
        candidate = now.normalize()
    
    # Return timezone-naive timestamp to avoid tz-aware vs tz-naive comparisons later
    return candidate.tz_localize(None)

def load_universe():
    """Load stock universe with fallback handling"""
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

def yf_download(tickers, start, end, batch=100):
    """Optimized batch download with better error handling"""
    if not tickers:
        return pd.DataFrame()
    
    logger.info(f"Downloading data for {len(tickers)} tickers from {start.date()} to {end.date()}")
    
    frames = []
    failed_tickers = []
    
    # Larger batch size for better performance
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
                threads=True,  # Enable multi-threading
                group_by='ticker'
            )
            
            if raw.empty:
                failed_tickers.extend(chunk)
                continue
                
            # Handle multi-ticker vs single ticker response
            if len(chunk) == 1:
                # Single ticker case
                if 'Close' in raw.columns:
                    df = raw[['Close']].copy()
                    df.columns = chunk
                else:
                    # Fallback to last column if Close not found
                    df = raw.iloc[:, [-1]].copy()
                    df.columns = chunk
            else:
                # Multi-ticker case
                if isinstance(raw.columns, pd.MultiIndex):
                    # Extract Close prices
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
                        failed_tickers.extend(chunk)
                        continue
                else:
                    # Single level columns
                    df = raw.copy()
            
            # Clean the data
            df = df.dropna(how='all')  # Remove completely empty rows
            if not df.empty:
                frames.append(df)
                
        except Exception as e:
            logger.warning(f"Failed to download batch {i//batch + 1}: {e}")
            failed_tickers.extend(chunk)
            continue
    
    if failed_tickers:
        logger.warning(f"Failed to download {len(failed_tickers)} tickers: {failed_tickers[:10]}...")
    
    if not frames:
        logger.error("No data downloaded successfully")
        return pd.DataFrame()
    
    # Combine all data
    result = pd.concat(frames, axis=1, sort=True)

    # Ensure datetime index is timezone-naive to avoid comparison issues
    result.index = pd.to_datetime(result.index)
    if getattr(result.index, 'tz', None) is not None:
        result.index = result.index.tz_localize(None)

    # Forward fill missing values (up to 5 days)
    result = result.fillna(method='ffill', limit=5)
    
    # Remove columns with insufficient data (less than 60% of total days)
    min_data_points = int(0.6 * len(result))
    result = result.loc[:, result.count() >= min_data_points]
    
    logger.info(f"Successfully downloaded data for {result.shape[1]} tickers with {result.shape[0]} days")
    return result

def compute_dates_idx(prices: pd.DataFrame, ref_date: pd.Timestamp):
    """Compute date indices with better error handling"""
    if prices.empty:
        raise ValueError("Price data is empty")
    
    # Normalize reference date to timezone-naive date to match prices index
    ref_date = pd.Timestamp(ref_date)
    if ref_date.tz is not None:
        ref_date = ref_date.tz_localize(None)
    ref_date = ref_date.normalize()

    # Find the actual reference date in the data
    available_dates = pd.DatetimeIndex(pd.to_datetime(prices.index))
    # Ensure available_dates is timezone-naive
    if getattr(available_dates, 'tz', None) is not None:
        available_dates = available_dates.tz_localize(None)
    if ref_date not in available_dates:
        # Find the closest available date before ref_date
        earlier_dates = available_dates[available_dates <= ref_date]
        if earlier_dates.empty:
            raise ValueError(f"No data available before reference date {ref_date.date()}")
        ref_date = earlier_dates[-1]
        logger.info(f"Adjusted reference date to last available: {ref_date.date()}")
    
    pos = available_dates.get_loc(ref_date)
    
    # Check if we have enough history
    min_needed = max(cfg["lookback_12m_days"], cfg["sma_window_days"], cfg["vol_window_days"])
    if pos < min_needed:
        raise ValueError(
            f"Insufficient history at {ref_date.date()}. "
            f"Need {min_needed} trading days, but only have {pos}"
        )
    
    idx_today = pos
    idx_6m = max(0, pos - cfg["lookback_6m_days"])
    idx_12m = max(0, pos - cfg["lookback_12m_days"])
    idx_1m_back = max(0, pos - cfg["exclude_1m_days"])
    
    return idx_today, idx_6m, idx_12m, idx_1m_back

def momentum_frame(prices: pd.DataFrame, ref_date: pd.Timestamp):
    """Enhanced momentum calculation with better logic"""
    idx_today, idx_6m, idx_12m, idx_1m_back = compute_dates_idx(prices, ref_date)
    
    # Get price points
    today = prices.iloc[idx_today]
    sixm = prices.iloc[idx_6m]
    twlv = prices.iloc[idx_12m]
    one_m = prices.iloc[idx_1m_back]
    
    # Find tickers with complete data across all time points
    valid_mask = (
        today.notna() & 
        sixm.notna() & 
        twlv.notna() & 
        one_m.notna() &
        (today > 0) &  # Ensure positive prices
        (sixm > 0) &
        (twlv > 0) &
        (one_m > 0)
    )
    valid = today[valid_mask].index
    
    if len(valid) == 0:
        raise ValueError("No valid tickers with complete price history")
    
    logger.info(f"Computing momentum for {len(valid)} valid tickers")
    
    # Calculate volatility (annualized standard deviation of returns)
    vol_start = max(0, idx_today - cfg["vol_window_days"])
    vol_window = prices[valid].iloc[vol_start:idx_today + 1]
    
    # Calculate daily returns (avoiding deprecated fill_method)
    daily_returns = vol_window.pct_change().dropna()
    vol = daily_returns.std() * np.sqrt(252) * 100  # Annualized volatility in %
    
    # Calculate returns
    ret_6m = ((today[valid] / sixm[valid]) - 1.0) * 100
    ret_12m_ex1 = ((one_m[valid] / twlv[valid]) - 1.0) * 100
    
    # Calculate 200-day moving average
    sma_start = max(0, idx_today - cfg["sma_window_days"] + 1)
    sma_window = prices[valid].iloc[sma_start:idx_today + 1]
    sma_200 = sma_window.mean()
    above_200dma = today[valid] > sma_200
    
    # Enhanced momentum score with risk adjustment
    # Add small constant to avoid division by zero
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
        "SMA_200": sma_200
    })
    
    # Remove any remaining NaN or infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Reference information
    ref_info = {
        "ReferenceDate": prices.index[idx_today],
        "TodayDate": prices.index[idx_today],
        "Date_6M": prices.index[idx_6m],
        "Date_12M": prices.index[idx_12m],
        "Date_1MBack": prices.index[idx_1m_back],
        "UniverseCount": prices.shape[1],
        "ValidCount": len(valid),
        "EligibleCount": int(((df["Above_200DMA"]) & (df["Positive_6M"])).sum())
    }
    
    logger.info(f"Momentum calculation complete. Eligible stocks: {ref_info['EligibleCount']}")
    return df, ref_info

def suggest_allocations(top_df: pd.DataFrame, capital: float, top_n: int):
    """Enhanced allocation calculation with better logic"""
    if top_df.empty:
        return top_df
    
    top_df = top_df.copy()
    
    # Equal weight allocation
    alloc_per = capital / min(len(top_df), top_n)
    
    # Calculate quantities (whole shares only)
    qty = np.floor(alloc_per / top_df["Price"]).astype(int)
    amt = qty * top_df["Price"]
    
    # Calculate allocation percentages
    total_invested = amt.sum()
    allocation_pct = (amt / total_invested * 100) if total_invested > 0 else 0
    
    top_df["Qty"] = qty
    top_df["Amount"] = amt
    top_df["Allocation(%)"] = allocation_pct
    top_df["Unutilized"] = alloc_per - amt
    
    return top_df

# ---------------- Pydantic Models ----------------
class StockData(BaseModel):
    Ticker: str
    Rank: Optional[int] = None
    Price: float
    Qty: Optional[int] = None
    Amount: Optional[float] = None
    Allocation: Optional[float] = Field(None, alias="Allocation(%)")
    MomentumScore: float
    Return_6M: float = Field(alias="6M_Return(%)")
    Return_12M_ex1: float = Field(alias="12M_ex1_Return(%)")
    Volatility: float = Field(alias="Volatility(%)")
    Above_200DMA: bool
    Positive_6M: bool

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
@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

@app.get("/run", response_model=RunResponse)
def run(
    capital: Optional[float] = Query(None, description="Capital amount (default from config)", gt=0),
    top_n: Optional[int] = Query(None, description="Number of top stocks (default from config)", gt=0, le=50),
    price_cap: Optional[float] = Query(None, description="Maximum price per share in INR"),
    ref_date: Optional[str] = Query(None, description="Reference date YYYY-MM-DD (default: last trading day)"),
    min_momentum_score: Optional[float] = Query(None, description="Minimum momentum score filter"),
):
    """
    Run momentum strategy analysis with enhanced features
    """
    try:
        # Validate and set parameters
        _capital = capital if capital is not None else cfg["capital"]
        _top_n = top_n if top_n is not None else cfg["top_n"]
        _cap = price_cap if price_cap is not None else cfg.get("price_cap")
        out_dir = cfg["output_dir"]
        
        # Parse reference date
        if ref_date:
            try:
                end_dt = pd.Timestamp(ref_date)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        else:
            end_dt = now_local()

        # Ensure end_dt is timezone-naive and normalized to match price index
        end_dt = pd.Timestamp(end_dt)
        if getattr(end_dt, 'tz', None) is not None:
            end_dt = end_dt.tz_localize(None)
        end_dt = end_dt.normalize()
        
        # Load universe and download data
        universe = load_universe()
        start_dt = end_dt - pd.Timedelta(days=cfg["history_days"])
        
        prices = yf_download(universe, start_dt, end_dt)
        if prices.empty:
            raise HTTPException(status_code=500, detail="No price data downloaded")
        
        # Calculate momentum metrics
        mom_df, ref_info = momentum_frame(prices, end_dt)
        
        if mom_df.empty:
            raise HTTPException(status_code=500, detail="No valid momentum data calculated")
        
        # Apply filters
        eligible = mom_df[
            (mom_df["Above_200DMA"]) & 
            (mom_df["Positive_6M"]) &
            (mom_df["Positive_12M_ex1"])  # Additional filter for better quality
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
        
        # Calculate allocations
        if not top.empty:
            top = suggest_allocations(top, _capital, _top_n)
            top.insert(0, "Rank", range(1, len(top) + 1))
        
        # Summary statistics
        summary = {
            "total_universe": len(universe),
            "data_available": prices.shape[1],
            "momentum_calculated": len(mom_df),
            "eligible_stocks": len(eligible),
            "selected_stocks": len(top),
            "total_capital": _capital,
            "invested_amount": float(top["Amount"].sum()) if not top.empty else 0,
            "utilization_pct": float(top["Amount"].sum() / _capital * 100) if not top.empty else 0,
            "avg_momentum_score": float(top["MomentumScore"].mean()) if not top.empty else 0,
            "filters_applied": {
                "above_200dma": True,
                "positive_6m": True,
                "positive_12m_ex1": True,
                "price_cap": _cap,
                "min_momentum_score": min_momentum_score
            }
        }
        
        # Save CSV files
        csv_paths = {}
        if cfg.get("write_csv", True):
            timestamp = end_dt.strftime("%Y%m%d")
            
            prices.to_csv(os.path.join(out_dir, f"prices_snapshot_{timestamp}.csv"))
            mom_df.to_csv(os.path.join(out_dir, f"momentum_scores_{timestamp}.csv"))
            eligible.to_csv(os.path.join(out_dir, f"eligible_stocks_{timestamp}.csv"))
            
            if not top.empty:
                top.to_csv(os.path.join(out_dir, f"top_selection_{timestamp}.csv"), index=True)
            
            # Save reference info
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
    Enhanced stop-loss and target calculations using ATR
    """
    try:
        # Parse and validate tickers
        tick_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if not tick_list:
            raise HTTPException(status_code=400, detail="No valid tickers provided")
        
        # Ensure .NS suffix for NSE stocks
        tick_list = [t if t.endswith('.NS') else t + '.NS' for t in tick_list]
        
        start = now_local() - pd.Timedelta(days=365)  # 1 year of data for better ATR
        end = now_local()
        
        results = []
        
        for ticker in tick_list:
            try:
                # Download data
                raw = yf.download(
                    ticker, 
                    start=start, 
                    end=end + pd.Timedelta(days=1), 
                    auto_adjust=True, 
                    progress=False
                )
                
                if raw.empty or len(raw) < 20:  # Need at least 20 days for ATR
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
                
                # Get latest values
                last_close = float(raw["Close"].iloc[-1])
                last_atr = float(atr.iloc[-1])
                
                # Calculate stops and targets
                stop_2x = last_close - (2 * last_atr)
                stop_1_5x = last_close - (1.5 * last_atr)
                target_2x = last_close + (2 * last_atr)
                
                results.append({
                    "Ticker": ticker,
                    "Close": round(last_close, 2),
                    "ATR14": round(last_atr, 2),
                    "Stop_2xATR": round(stop_2x, 2),
                    "Stop_1_5xATR": round(stop_1_5x, 2),
                    "Target_2xATR": round(target_2x, 2)
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

# ---------------- N8N Integration Endpoints ----------------
from fastapi import BackgroundTasks

class WebhookData(BaseModel):
    source: str = "n8n"
    execution_id: Optional[str] = None
    capital: Optional[float] = None
    top_n: Optional[int] = None
    price_cap: Optional[float] = None
    ref_date: Optional[str] = None
    callback_url: Optional[str] = None

@app.post("/webhook/rebalance")
async def webhook_rebalance(webhook_data: WebhookData):
    """
    Webhook endpoint for n8n rebalancing automation
    """
    try:
        logger.info(f"Webhook triggered by {webhook_data.source}, execution_id: {webhook_data.execution_id}")
        
        # Extract parameters
        _capital = webhook_data.capital or cfg['capital']
        _top_n = webhook_data.top_n or cfg['top_n']
        _cap = webhook_data.price_cap or cfg.get('price_cap')
        
        # Run strategy analysis
        result = run(
            capital=_capital, 
            top_n=_top_n, 
            price_cap=_cap, 
            ref_date=webhook_data.ref_date
        )
        
        # Add webhook metadata
        result['webhook'] = {
            'triggered_at': datetime.now().isoformat(),
            'source': webhook_data.source,
            'execution_id': webhook_data.execution_id,
            'callback_url': webhook_data.callback_url
        }
        
        # If callback URL provided, send async notification
        if webhook_data.callback_url:
            # This would normally send an async HTTP request to the callback URL
            logger.info(f"Would send callback to: {webhook_data.callback_url}")
        
        return result
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/webhook/status/{execution_id}")
async def webhook_status(execution_id: str):
    """
    Check status of webhook execution
    """
    # In a real implementation, this would check a database or cache
    return {
        "execution_id": execution_id, 
        "status": "completed",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/webhook/portfolio-compare")
async def webhook_portfolio_compare(
    target_portfolio: List[Dict[str, Any]],
    current_holdings: List[Dict[str, Any]]
):
    """
    Compare target portfolio with current holdings and generate rebalance orders
    """
    try:
        rebalance_orders = []
        current_map = {holding['ticker']: holding for holding in current_holdings}
        
        # Generate rebalancing logic
        for target in target_portfolio:
            ticker = target.get('Ticker', target.get('ticker'))
            target_qty = target.get('Qty', target.get('quantity', 0))
            current = current_map.get(ticker, {'quantity': 0})
            current_qty = current.get('quantity', 0)
            
            qty_diff = target_qty - current_qty
            
            if abs(qty_diff) > 0:
                rebalance_orders.append({
                    'ticker': ticker,
                    'action': 'BUY' if qty_diff > 0 else 'SELL',
                    'quantity': abs(qty_diff),
                    'target_qty': target_qty,
                    'current_qty': current_qty,
                    'price': target.get('Price', target.get('price', 0)),
                    'momentum_score': target.get('MomentumScore', 0),
                    'priority': target.get('Rank', 999)
                })
        
        # Handle positions to exit
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
                        'priority': 1
                    })
        
        # Sort orders
        rebalance_orders.sort(key=lambda x: (x['action'] == 'BUY', x['priority']))
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_orders': len(rebalance_orders),
            'buy_orders': len([o for o in rebalance_orders if o['action'] == 'BUY']),
            'sell_orders': len([o for o in rebalance_orders if o['action'] == 'SELL']),
            'orders': rebalance_orders
        }
        
    except Exception as e:
        logger.error(f"Portfolio compare error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook/validate-orders")
async def webhook_validate_orders(orders: List[Dict[str, Any]], max_position_size: float = 0.15):
    """
    Validate orders for risk management and market hours
    """
    try:
        # Check market hours
        now = datetime.now(gettz(cfg.get("tz", "Asia/Kolkata")))
        hour = now.hour
        minute = now.minute
        day = now.weekday()
        
        is_market_open = (
            day < 5 and  # Monday to Friday (0-4)
            ((hour == 9 and minute >= 15) or (hour > 9 and hour < 15) or (hour == 15 and minute <= 30))
        )
        
        validated_orders = []
        rejected_orders = []
        warnings = []
        
        if not is_market_open:
            warnings.append("Market is closed. Orders will be queued for next session.")
        
        # Validate each order
        for order in orders:
            estimated_value = order.get('estimated_value', 0)
            total_capital = order.get('total_capital', cfg['capital'])
            position_size = estimated_value / total_capital if total_capital > 0 else 0
            
            if position_size > max_position_size:
                rejected_orders.append({
                    **order,
                    'rejection_reason': f"Position size {position_size*100:.2f}% exceeds limit {max_position_size*100:.2f}%"
                })
                warnings.append(f"Rejected {order.get('ticker')}: Position size too large")
            else:
                validated_orders.append(order)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'market_open': is_market_open,
            'validated_orders': validated_orders,
            'rejected_orders': rejected_orders,
            'warnings': warnings,
            'validation_summary': {
                'total_orders': len(orders),
                'validated': len(validated_orders),
                'rejected': len(rejected_orders)
            }
        }
        
    except Exception as e:
        logger.error(f"Order validation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)