from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import yfinance as yf
import yaml
import os
from datetime import datetime, timedelta
from dateutil.tz import gettz
from typing import Optional, List, Dict, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ---------------- Config loader ----------------
CFG_PATH = "config.yaml"

def load_config():
    """Load configuration with error handling."""
    try:
        with open(CFG_PATH, "r") as f:
            config = yaml.safe_load(f)
        
        # Validate required config keys
        required_keys = [
            "universe_source", "history_days", "lookback_6m_days", 
            "lookback_12m_days", "exclude_1m_days", "vol_window_days", 
            "sma_window_days", "top_n", "capital"
        ]
        
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")
        
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {CFG_PATH} not found")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {e}")

# Load config early to catch errors
try:
    cfg = load_config()
    os.makedirs(cfg.get("output_dir", "output"), exist_ok=True)
except Exception as e:
    print(f"Failed to load config: {e}")
    cfg = {}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Momentum Strategy API", version="1.0.0")

# ---------------- Helpers ----------------
def now_local():
    """Get current local time based on config timezone."""
    try:
        return pd.Timestamp.now(tz=gettz(cfg.get("tz", "Asia/Kolkata"))).normalize()
    except Exception:
        # Fallback to UTC if timezone fails
        return pd.Timestamp.now(tz='UTC').normalize()

def load_universe():
    """Load universe with better error handling and fallback."""
    url = cfg.get("universe_source")
    if not url:
        raise ValueError("Universe source not configured")
    
    try:
        # Try online source first
        df = pd.read_csv(url, timeout=30)
        logger.info(f"Successfully loaded universe from {url}")
    except Exception as e:
        logger.warning(f"Failed to load from {url}: {e}")
        # Fallback to local file
        local_file = "ind_nifty500list.csv"
        if os.path.exists(local_file):
            df = pd.read_csv(local_file)
            logger.info(f"Loaded universe from local file {local_file}")
        else:
            raise FileNotFoundError(f"Neither online source nor local file {local_file} available")
    
    # Handle different column names
    symbol_cols = ["Symbol", "SYMBOL", "symbol", "Ticker", "TICKER"]
    col = None
    for col_name in symbol_cols:
        if col_name in df.columns:
            col = col_name
            break
    
    if col is None:
        raise ValueError(f"No symbol column found. Available columns: {list(df.columns)}")
    
    # Clean and validate symbols
    syms = df[col].astype(str).str.strip().str.upper()
    # Filter out invalid symbols
    valid_syms = [s for s in syms if s and s != 'NAN' and len(s) > 0]
    
    if not valid_syms:
        raise ValueError("No valid symbols found in universe")
    
    # Add .NS suffix for NSE
    nse_syms = [s + ".NS" for s in valid_syms]
    logger.info(f"Loaded {len(nse_syms)} symbols from universe")
    return nse_syms

def download_single_ticker(ticker, start, end):
    """Download data for a single ticker with error handling."""
    try:
        raw = yf.download(
            ticker, 
            start=start, 
            end=end + pd.Timedelta(days=1), 
            auto_adjust=True, 
            progress=False,
            timeout=30
        )
        if raw.empty:
            return None
        
        # Handle single ticker case
        if "Close" in raw.columns:
            df = raw[["Close"]].copy()
            df.columns = [ticker]
        else:
            # Single column case
            df = raw.copy()
            df.columns = [ticker]
        
        return df
    except Exception as e:
        logger.warning(f"Failed to download {ticker}: {e}")
        return None

def yf_download(tickers, start, end, batch=50):
    """Download adjusted prices in batches with parallel processing."""
    if not tickers:
        return pd.DataFrame()
    
    frames = []
    
    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=min(10, len(tickers))) as executor:
        # Submit all download tasks
        future_to_ticker = {
            executor.submit(download_single_ticker, ticker, start, end): ticker 
            for ticker in tickers
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                if result is not None:
                    frames.append(result)
            except Exception as e:
                logger.warning(f"Exception for {ticker}: {e}")
    
    if not frames:
        logger.warning("No data downloaded for any ticker")
        return pd.DataFrame()
    
    # Combine all frames
    out = pd.concat(frames, axis=1)
    out = out.sort_index().ffill()
    
    # Drop all-empty columns and rows
    out = out.loc[:, out.notna().any()]
    out = out.loc[out.notna().any(axis=1)]
    
    logger.info(f"Successfully downloaded data for {out.shape[1]} tickers over {out.shape[0]} days")
    return out

def compute_dates_idx(prices: pd.DataFrame, ref_date: pd.Timestamp):
    """Return indices for various lookback periods with validation."""
    if prices.empty:
        raise ValueError("Price data is empty")
    
    # Find the closest available date
    if ref_date not in prices.index:
        pos = prices.index.get_indexer([ref_date], method="pad")[0]
        if pos == -1:
            # If pad fails, try forward fill
            pos = prices.index.get_indexer([ref_date], method="ffill")[0]
            if pos == -1:
                raise ValueError(f"Reference date {ref_date.date()} not found in price data")
    else:
        pos = prices.index.get_loc(ref_date)
    
    # Calculate required lookback
    need = max(cfg["lookback_12m_days"], cfg["sma_window_days"])
    if pos < need:
        raise ValueError(
            f"Not enough history at {ref_date.date()} "
            f"(need {need} trading days, have {pos})"
        )
    
    idx_today = pos
    idx_6m = pos - cfg["lookback_6m_days"]
    idx_12m = pos - cfg["lookback_12m_days"]
    idx_1m_back = pos - cfg["exclude_1m_days"]
    
    return idx_today, idx_6m, idx_12m, idx_1m_back

def momentum_frame(prices: pd.DataFrame, ref_date: pd.Timestamp):
    """Compute momentum metrics with improved error handling."""
    try:
        idx_today, idx_6m, idx_12m, idx_1m_back = compute_dates_idx(prices, ref_date)
    except ValueError as e:
        logger.error(f"Date index computation failed: {e}")
        raise
    
    # Extract price data
    today = prices.iloc[idx_today]
    sixm = prices.iloc[idx_6m]
    twlv = prices.iloc[idx_12m]
    one_m = prices.iloc[idx_1m_back]
    
    # Find valid tickers with complete data
    valid = today.dropna().index.intersection(sixm.dropna().index)
    valid = valid.intersection(twlv.dropna().index).intersection(one_m.dropna().index)
    
    if len(valid) == 0:
        raise ValueError("No tickers have complete data for all required dates")
    
    logger.info(f"Processing {len(valid)} valid tickers")
    
    # Calculate volatility with error handling
    try:
        vol_start_idx = max(0, idx_today - cfg["vol_window_days"])
        vol_window = prices[valid].iloc[vol_start_idx:idx_today + 1]
        pct = vol_window.pct_change(fill_method=None).dropna(how="all")
        vol = pct.std() * np.sqrt(252) * 100
        # Handle zero volatility
        vol = vol.replace([0, np.inf, -np.inf], np.nan)
    except Exception as e:
        logger.error(f"Volatility calculation failed: {e}")
        vol = pd.Series([np.nan] * len(valid), index=valid)
    
    # Calculate returns
    ret_6m = ((today[valid] / sixm[valid]) - 1.0) * 100
    ret_12m_ex1 = ((one_m[valid] / twlv[valid]) - 1.0) * 100
    
    # Calculate 200-DMA
    try:
        sma_start_idx = max(0, idx_today - cfg["sma_window_days"])
        sma_slice = prices[valid].iloc[sma_start_idx:idx_today + 1]
        sma_200 = sma_slice.mean()
        above_200dma = today[valid] > sma_200
    except Exception as e:
        logger.error(f"SMA calculation failed: {e}")
        above_200dma = pd.Series([False] * len(valid), index=valid)
    
    # Calculate momentum score with safety checks
    score = pd.Series(index=valid, dtype=float)
    for ticker in valid:
        if pd.notna(vol[ticker]) and vol[ticker] > 0:
            score[ticker] = (ret_6m[ticker] + ret_12m_ex1[ticker]) / vol[ticker]
        else:
            score[ticker] = np.nan
    
    # Create results DataFrame
    df = pd.DataFrame({
        "Price": today[valid],
        "6M_Return(%)": ret_6m,
        "12M_ex1_Return(%)": ret_12m_ex1,
        "Volatility(%)": vol,
        "MomentumScore": score,
        "Above_200DMA": above_200dma,
        "Positive_6M": ret_6m > 0
    }).dropna()
    
    # Reference dates snapshot
    ref = pd.DataFrame([{
        "ReferenceDate": prices.index[idx_today],
        "TodayDate": prices.index[idx_today],
        "Date_6M": prices.index[idx_6m],
        "Date_12M": prices.index[idx_12m],
        "Date_1MBack": prices.index[idx_1m_back],
        "UniverseCount": prices.shape[1],
        "EligibleCount": int(((df["Above_200DMA"]) & (df["Positive_6M"])).sum())
    }])
    
    return df, ref

def suggest_allocations(top_df: pd.DataFrame, capital: float, top_n: int):
    """Suggest allocations with improved logic."""
    if top_df.empty:
        return top_df
    
    top_df = top_df.copy()
    
    # Equal weight allocation
    alloc_per = capital / max(1, top_n)
    
    # Calculate quantities and amounts
    qty = (alloc_per // top_df["Price"]).astype(int)
    amt = qty * top_df["Price"]
    
    # Add allocation columns
    top_df["Qty"] = qty
    top_df["Amount"] = amt
    top_df["Allocation(%)"] = (amt / capital * 100).round(2)
    
    return top_df

# ----------------- Models for docs -----------------
class RunResponse(BaseModel):
    reference_dates: Dict[str, Any]
    eligible: List[Dict[str, Any]]
    top: List[Dict[str, Any]]
    csv_paths: Dict[str, str]
    metadata: Dict[str, Any]

class LevelsResponse(BaseModel):
    levels: List[Dict[str, Any]]

# ----------------- API endpoints ------------------
@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "ok": True, 
        "time": datetime.now().isoformat(),
        "config_loaded": bool(cfg),
        "timezone": cfg.get("tz", "Not set")
    }

@app.get("/run", response_model=RunResponse)
def run(
    capital: Optional[float] = Query(default=None, description="Override capital; default from config"),
    top_n: Optional[int] = Query(default=None, description="Override top_n; default from config"),
    price_cap: Optional[float] = Query(default=None, description="Override price cap in INR; null to disable"),
    ref_date: str = Query(default="", description="YYYY-MM-DD; default = last available trading day"),
):
    """
    Compute momentum scores and return eligible + top list.
    Saves CSVs if write_csv=true in config.
    """
    try:
        # Load settings with validation
        _capital = capital if capital is not None else cfg.get("capital")
        _top_n = top_n if top_n is not None else cfg.get("top_n")
        _cap = price_cap if price_cap is not None else cfg.get("price_cap")
        out_dir = cfg.get("output_dir", "output")
        
        if _capital is None or _top_n is None:
            raise HTTPException(status_code=400, detail="Missing required config values")
        
        # Load universe
        universe = load_universe()
        if not universe:
            raise HTTPException(status_code=500, detail="Failed to load universe")
        
        # Parse reference date
        try:
            end_dt = pd.Timestamp(ref_date) if ref_date else now_local()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        start_dt = end_dt - pd.Timedelta(days=cfg["history_days"])
        
        # Download prices
        logger.info(f"Downloading prices for {len(universe)} tickers from {start_dt.date()} to {end_dt.date()}")
        prices = yf_download(universe, start_dt, end_dt)
        
        if prices.empty:
            raise HTTPException(status_code=500, detail="No price data downloaded")
        
        min_required = max(cfg["lookback_12m_days"], cfg["sma_window_days"])
        if prices.shape[0] < min_required:
            raise HTTPException(
                status_code=500, 
                detail=f"Insufficient price history. Need {min_required} days, got {prices.shape[0]}"
            )
        
        # Compute momentum
        logger.info("Computing momentum scores")
        mom_df, ref_df = momentum_frame(prices, end_dt)
        
        # Apply filters
        eligible = mom_df[
            (mom_df["Above_200DMA"]) & 
            (mom_df["Positive_6M"])
        ].copy()
        
        if _cap is not None:
            eligible = eligible[eligible["Price"] <= _cap]
        
        if eligible.empty:
            logger.warning("No stocks passed filters")
            return {
                "reference_dates": ref_df.iloc[0].astype(str).to_dict(),
                "eligible": [],
                "top": [],
                "csv_paths": {},
                "metadata": {"warning": "No stocks passed filters"}
            }
        
        # Rank and select top
        ranked = eligible.sort_values("MomentumScore", ascending=False).copy()
        top = ranked.head(_top_n).copy()
        top = suggest_allocations(top, _capital, _top_n)
        
        # Add rank column
        top.insert(0, "Rank", range(1, len(top) + 1))
        
        # Save CSVs if requested
        csv_paths = {}
        if cfg.get("write_csv", False):
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                prices.to_csv(os.path.join(out_dir, f"phase2_prices_snapshot_{timestamp}.csv"))
                mom_df.to_csv(os.path.join(out_dir, f"phase3_momentum_scores_{timestamp}.csv"))
                eligible.to_csv(os.path.join(out_dir, f"phase4_filtered_eligible_{timestamp}.csv"))
                top.to_csv(os.path.join(out_dir, f"phase4_top_selection_{timestamp}.csv"), index=True)
                ref_df.to_csv(os.path.join(out_dir, f"phase3_reference_dates_{timestamp}.csv"), index=False)
                
                csv_paths = {
                    "prices": f"phase2_prices_snapshot_{timestamp}.csv",
                    "scores": f"phase3_momentum_scores_{timestamp}.csv",
                    "eligible": f"phase4_filtered_eligible_{timestamp}.csv",
                    "top": f"phase4_top_selection_{timestamp}.csv",
                    "reference_dates": f"phase3_reference_dates_{timestamp}.csv"
                }
                logger.info("CSV files saved successfully")
            except Exception as e:
                logger.error(f"Failed to save CSV files: {e}")
        
        # Prepare response
        metadata = {
            "total_universe": len(universe),
            "valid_tickers": len(mom_df),
            "eligible_count": len(eligible),
            "top_selected": len(top),
            "execution_time": datetime.now().isoformat()
        }
        
        return {
            "reference_dates": ref_df.iloc[0].astype(str).to_dict(),
            "eligible": eligible.reset_index().rename(columns={"index": "Ticker"}).round(6).to_dict(orient="records"),
            "top": top.reset_index().rename(columns={"index": "Ticker"}).round(6).to_dict(orient="records"),
            "csv_paths": csv_paths,
            "metadata": metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in run endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/levels", response_model=LevelsResponse)
def levels(tickers: str = Query(..., description="Comma-separated tickers, e.g., BEL.NS,MFSL.NS")):
    """
    Calculate ATR(14) stop suggestions for given tickers.
    """
    try:
        tick_list = [t.strip() for t in tickers.split(",") if t.strip()]
        if not tick_list:
            raise HTTPException(status_code=400, detail="No valid tickers provided")
        
        start = now_local() - pd.Timedelta(days=180)
        end = now_local()
        
        rows = []
        for ticker in tick_list:
            try:
                raw = yf.download(
                    ticker, 
                    start=start, 
                    end=end + pd.Timedelta(days=1), 
                    auto_adjust=True, 
                    progress=False,
                    timeout=30
                )
                
                if raw.empty:
                    logger.warning(f"No data for {ticker}")
                    continue
                
                # Ensure required columns exist
                required_cols = ["High", "Low", "Close"]
                if not all(col in raw.columns for col in required_cols):
                    logger.warning(f"Missing required columns for {ticker}")
                    continue
                
                # Calculate True Range and ATR
                high_low = raw["High"] - raw["Low"]
                high_close = np.abs(raw["High"] - raw["Close"].shift())
                low_close = np.abs(raw["Low"] - raw["Close"].shift())
                
                tr = np.maximum(high_low, np.maximum(high_close, low_close))
                atr = tr.rolling(14).mean()
                
                last_close = float(raw.iloc[-1]["Close"])
                last_atr = float(atr.iloc[-1])
                
                if pd.isna(last_atr) or last_atr <= 0:
                    logger.warning(f"Invalid ATR for {ticker}")
                    continue
                
                stop = float(last_close - 2 * last_atr)
                
                rows.append({
                    "Ticker": ticker,
                    "Close": last_close,
                    "ATR14": round(last_atr, 2),
                    "Stop_2xATR": round(stop, 2),
                    "Stop_Distance(%)": round(((last_close - stop) / last_close * 100), 2)
                })
                
            except Exception as e:
                logger.warning(f"Failed to process {ticker}: {e}")
                continue
        
        if not rows:
            raise HTTPException(status_code=404, detail="No valid data found for any ticker")
        
        return {"levels": rows}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in levels endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)