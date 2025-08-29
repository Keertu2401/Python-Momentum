# main.py
# FastAPI momentum service — India-market safe version
# - Signals at t-1 (T+1 entry)
# - Trading-day aligned lookbacks
# - True 12-1 return
# - Corporate-action anomaly guard
# - Optional liquidity / turnover filter (₹ Cr)

import os
import math
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf
import yaml
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# ------------ Logging ------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("momentum-india")

# ------------ Config ------------
CFG_PATH = "config.yaml"
DEFAULT_CFG = {
    "capital": 1000000.0,
    "top_n": 10,
    "price_cap": None,
    "universe_source": "ind_nifty500list.csv",  # fallback local csv ok
    "history_days": 900,  # ensure enough for 252+200 windows
    "lookback_6m_days": 126,
    "lookback_12m_days": 252,
    "lookback_1m_days": 21,
    "sma_window_days": 200,
    "vol_window_days": 90,
    "output_dir": "outputs",
    "write_csv": True,
    "tz": "Asia/Kolkata",
    "min_avg_turnover_cr": None,  # e.g., 10.0 for ₹10 Cr liquidity filter
}

def load_config() -> dict:
    if not os.path.exists(CFG_PATH):
        log.warning("config.yaml not found — using defaults.")
        return DEFAULT_CFG.copy()
    try:
        with open(CFG_PATH, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = DEFAULT_CFG.copy()
        cfg.update(user_cfg)
        os.makedirs(cfg["output_dir"], exist_ok=True)
        return cfg
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read config.yaml: {e}")

cfg = load_config()

# ------------ FastAPI app ------------
app = FastAPI(
    title="Momentum Strategy (India-safe)",
    description="Momentum selection with T+1 safety, trading-day windows, 12-1 return, and liquidity filtering.",
    version="1.1.0",
)

# ------------ Time helpers ------------
def now_local() -> datetime:
    tz = ZoneInfo(cfg.get("tz", "Asia/Kolkata"))
    return datetime.now(tz)

# ------------ Universe loader ------------
POSSIBLE_TICKER_COLS = [
    "Symbol","SYMBOL","symbol","Ticker","TICKER","ticker","NSE Symbol","NSE_SYMBOL","nse_symbol"
]

def load_universe() -> List[str]:
    src = cfg["universe_source"]
    if src and os.path.exists(src):
        df = pd.read_csv(src)
    else:
        # Try a few sane local fallbacks
        for fn in ["ind_nifty500list.csv", "nse_universe.csv", "universe.csv"]:
            if os.path.exists(fn):
                df = pd.read_csv(fn)
                break
        else:
            raise HTTPException(status_code=500, detail="Universe source not found (check config or provide CSV).")

    col = None
    for c in POSSIBLE_TICKER_COLS:
        if c in df.columns:
            col = c
            break
    if col is None:
        # try first column as last resort
        col = df.columns[0]

    tickers = []
    for raw in df[col].astype(str).tolist():
        s = raw.strip().upper()
        if not s:
            continue
        if not s.endswith(".NS") and not s.endswith(".BSE"):
            s = s + ".NS"
        tickers.append(s)
    tickers = sorted(set(tickers))
    if not tickers:
        raise HTTPException(status_code=500, detail="No tickers found in universe file.")
    return tickers

# ------------ Data download & guards (India-safe) ------------
TRADING_DAYS_PER_YEAR = 246  # ~244–248 typical for NSE/BSE

def _chunked(xs: List[str], n: int) -> List[List[str]]:
    return [xs[i:i+n] for i in range(0, len(xs), n)]

def yf_download_full(
    tickers: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    batch: int = 80,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.Series], Dict[str, pd.Series]]:
    """
    Robust downloader: returns (Adj Close, Close, Volume) aligned frames,
    plus per-ticker Stock Splits & Dividends series.
    Uses auto_adjust=False so we have both Close and Adj Close for checks.
    """
    adj_frames, close_frames, vol_frames = [], [], []
    splits_map: Dict[str, pd.Series] = {}
    divs_map: Dict[str, pd.Series] = {}

    for chunk in _chunked(tickers, batch):
        df = yf.download(
            chunk,
            start=start,
            end=end,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            actions=True,
            progress=False,
        )
        if df is None or df.empty:
            continue

        if isinstance(df.columns, pd.MultiIndex):
            for t in chunk:
                if (t, "Adj Close") in df.columns and (t, "Close") in df.columns and (t, "Volume") in df.columns:
                    adj_frames.append(df[(t, "Adj Close")].rename(t))
                    close_frames.append(df[(t, "Close")].rename(t))
                    vol_frames.append(df[(t, "Volume")].rename(t))
                if (t, "Stock Splits") in df.columns:
                    s = df[(t, "Stock Splits")]
                    splits_map[t] = s[s != 0.0]
                if (t, "Dividends") in df.columns:
                    d = df[(t, "Dividends")]
                    divs_map[t] = d[d != 0.0]
        else:
            t = chunk[0]
            must = {"Adj Close", "Close", "Volume"}
            if must.issubset(set(df.columns)):
                adj_frames.append(df["Adj Close"].rename(t))
                close_frames.append(df["Close"].rename(t))
                vol_frames.append(df["Volume"].rename(t))
            if "Stock Splits" in df.columns:
                s = df["Stock Splits"]
                splits_map[t] = s[s != 0.0]
            if "Dividends" in df.columns:
                d = df["Dividends"]
                divs_map[t] = d[d != 0.0]

    if not adj_frames:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {}

    adj_close = pd.concat(adj_frames, axis=1).sort_index()
    close = pd.concat(close_frames, axis=1).reindex(adj_close.index).sort_index()
    volume = pd.concat(vol_frames, axis=1).reindex(adj_close.index).sort_index()

    # Fill short gaps, filter low-coverage tickers
    adj_close = adj_close.ffill(limit=5)
    close = close.ffill(limit=5)
    volume = volume.fillna(0)

    coverage = adj_close.notna().mean(axis=0)
    keep_cols = coverage.index[coverage >= 0.60]
    adj_close = adj_close.loc[:, keep_cols]
    close = close.loc[:, keep_cols]
    volume = volume.loc[:, keep_cols]

    return adj_close, close, volume, splits_map, divs_map

def detect_corporate_action_anomalies(
    adj_close: pd.DataFrame,
    close: pd.DataFrame,
    splits_map: Dict[str, pd.Series],
    max_abs_move_without_split: float = 0.40,
    ratio_jump_threshold: float = 0.10,
) -> List[str]:
    """
    Flag tickers where adjusted series looks inconsistent with recorded actions.
    """
    if adj_close.empty:
        return []
    suspects: set = set()
    daily_ret = adj_close.pct_change()

    for t in adj_close.columns:
        if t not in close.columns:
            suspects.add(t)
            continue

        # Big adjusted move with no split nearby
        big_dates = daily_ret.index[(daily_ret[t].abs() > max_abs_move_without_split)].tolist()
        for d in big_dates:
            split_near = False
            if t in splits_map and not splits_map[t].empty:
                split_dates = pd.to_datetime(splits_map[t].index)
                if any(abs((pd.Timestamp(d) - sd).days) <= 2 for sd in split_dates):
                    split_near = True
            if not split_near:
                suspects.add(t)
                break

        # Adj/Close ratio sanity jumps
        ratio = (adj_close[t] / close[t]).replace([np.inf, -np.inf], np.nan)
        if ratio.dropna().pct_change().abs().gt(ratio_jump_threshold).any():
            suspects.add(t)

    return sorted(suspects)

def _loc_at_or_before(idx: pd.DatetimeIndex, when: pd.Timestamp) -> int:
    pos = np.searchsorted(idx.values, np.datetime64(when), side="right") - 1
    if pos < 0:
        raise ValueError("Reference date earlier than first available trading day.")
    return pos

def resolve_ref_date_from_prices(adj_close: pd.DataFrame, requested_ref_date: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    """Pick the last trading day at/ before requested date (or last index if None)."""
    if adj_close.empty:
        raise ValueError("No price data.")
    idx = adj_close.index
    if requested_ref_date is None:
        return pd.Timestamp(idx[-1])
    pos = _loc_at_or_before(idx, pd.Timestamp(requested_ref_date))
    return pd.Timestamp(idx[pos])

def compute_momentum_frame_india(
    adj_close: pd.DataFrame,
    close: pd.DataFrame,
    volume: pd.DataFrame,
    ref_date: pd.Timestamp,
    lookback_6m_days: int = 126,
    lookback_12m_days: int = 252,
    lookback_1m_days: int = 21,
    sma_window_days: int = 200,
    vol_window_days: int = 90,
    min_avg_turnover_cr: Optional[float] = None,
    price_cap: Optional[float] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Momentum table with India-safe logic:
    - Signal at t-1 (entry on t)
    - 6M = P(t-1)/P(t-126)-1
    - 12-1 = P(t-21)/P(t-252)-1
    - Vol over last 'vol_window_days' up to t-1, annualized by sqrt(246)
    - SMA200 at t-1
    - Optional liquidity filter by avg turnover (₹ Cr) over last 60 sessions up to t-1
    """
    if adj_close.empty:
        raise ValueError("No price data.")

    idx = adj_close.index
    ref_pos = _loc_at_or_before(idx, pd.Timestamp(ref_date))
    signal_pos = ref_pos - 1  # enforce T+1 (no using today's close)
    need = max(lookback_12m_days, sma_window_days, vol_window_days) + 1
    if signal_pos < need:
        raise ValueError("Insufficient history for lookbacks at the requested date.")

    # Prices for lookbacks (all trading-day aligned)
    p_signal = adj_close.iloc[signal_pos]                        # P(t-1)
    p_6m = adj_close.iloc[signal_pos - lookback_6m_days]         # P(t-126)
    p_12m = adj_close.iloc[signal_pos - lookback_12m_days]       # P(t-252)
    p_1m = adj_close.iloc[signal_pos - lookback_1m_days]         # P(t-21)

    # Returns
    ret_6m = (p_signal / p_6m) - 1.0
    ret_12m_ex1 = (p_1m / p_12m) - 1.0  # excludes last month

    # Volatility to t-1
    daily_ret = adj_close.pct_change()
    vol = daily_ret.iloc[signal_pos - (vol_window_days - 1): signal_pos + 1].std() * math.sqrt(TRADING_DAYS_PER_YEAR)

    # SMA200 at t-1
    sma200_series = adj_close.rolling(window=sma_window_days, min_periods=sma_window_days).mean().iloc[signal_pos]
    above_200dma = p_signal > sma200_series

    # Base filters
    filters = above_200dma & (ret_6m > 0) & (ret_12m_ex1 > 0)

    # Liquidity: avg turnover (₹ Cr) last 60 sessions to t-1
    if min_avg_turnover_cr is not None:
        aligned_close = close.reindex(idx)
        aligned_volume = volume.reindex(idx)
        avg_turnover_cr = (aligned_close * aligned_volume).rolling(60).mean().iloc[signal_pos] / 1e7
        filters = filters & (avg_turnover_cr >= float(min_avg_turnover_cr))

    if price_cap is not None:
        filters = filters & (p_signal <= float(price_cap))

    # Score
    risk_adj_vol = vol + 0.01
    momentum_score = (0.4 * ret_6m + 0.6 * ret_12m_ex1) / risk_adj_vol

    mom_df = pd.DataFrame({
        "Price": p_signal,
        "Ret_6M": ret_6m,
        "Ret_12M_ex1": ret_12m_ex1,
        "Vol_Ann": vol,
        "MomentumScore": momentum_score,
        "SMA200": sma200_series,
        "Above_200DMA": above_200dma,
    }).replace([np.inf, -np.inf], np.nan)

    mom_df = mom_df[filters].dropna(subset=["MomentumScore"]).sort_values("MomentumScore", ascending=False)

    reference = {
        "signal_date": pd.Timestamp(idx[signal_pos]).date().isoformat(),
        "entry_date": pd.Timestamp(idx[signal_pos + 1]).date().isoformat(),
        "lb_6m_days": lookback_6m_days,
        "lb_12m_days": lookback_12m_days,
        "lb_1m_days": lookback_1m_days,
        "sma_window_days": sma_window_days,
        "vol_window_days": vol_window_days,
        "trading_days_per_year": TRADING_DAYS_PER_YEAR,
    }
    return mom_df, reference

# ------------ Allocation ------------
def suggest_allocations(top_df: pd.DataFrame, capital: float, top_n: int) -> pd.DataFrame:
    if top_df.empty:
        return pd.DataFrame(columns=["Ticker","Price","Qty","Amount","AllocationPct","Unutilized"])
    n = min(top_n, len(top_df))
    per_slot = float(capital) / float(n) if n > 0 else 0.0
    out = []
    for t, row in top_df.head(n).iterrows():
        price = float(row["Price"])
        qty = int(per_slot // price) if price > 0 else 0
        amt = qty * price
        alloc_pct = (amt / capital) * 100.0 if capital > 0 else 0.0
        unutil = per_slot - amt
        out.append({"Ticker": t, "Price": price, "Qty": qty, "Amount": amt,
                    "AllocationPct": alloc_pct, "Unutilized": unutil})
    return pd.DataFrame(out)

# ------------ Schemas ------------
class StockData(BaseModel):
    ticker: str
    rank: int
    price: float
    qty: int
    amount: float
    allocation_pct: float
    momentum_score: float
    ret_6m: float
    ret_12m_ex1: float
    vol_ann: float
    above_200dma: bool

class RunResponse(BaseModel):
    reference: Dict[str, str]
    eligible_count: int
    selected_count: int
    stocks: List[StockData]
    summary: Dict[str, float]
    csv_paths: Optional[Dict[str, str]] = None

class LevelData(BaseModel):
    ticker: str
    last_close: float
    atr14: float
    stop_2atr: float
    stop_1_5atr: float
    target_2atr: float

class LevelsResponse(BaseModel):
    as_of: str
    levels: List[LevelData]

# ------------ Health ------------
@app.get("/health")
def health():
    return {"status": "ok", "ts": now_local().isoformat(), "version": app.version}

# ------------ /run ------------
@app.get("/run", response_model=RunResponse)
def run_strategy(
    capital: Optional[float] = Query(None, description="Override total capital (INR)"),
    top_n: Optional[int] = Query(None, description="Override number of picks"),
    price_cap: Optional[float] = Query(None, description="Exclude stocks above this price"),
    ref_date: Optional[str] = Query(None, description="YYYY-MM-DD; if omitted uses last available session"),
    min_momentum_score: Optional[float] = Query(None, description="Optional floor on score"),
):
    _capital = float(capital) if capital is not None else float(cfg["capital"])
    _top_n = int(top_n) if top_n is not None else int(cfg["top_n"])
    if _top_n <= 0 or _top_n > 50:
        raise HTTPException(status_code=400, detail="top_n must be between 1 and 50.")

    # 1) Universe
    universe = load_universe()
    if len(universe) == 0:
        raise HTTPException(status_code=500, detail="Empty universe.")

    # 2) Date range (wide enough for all windows)
    end_dt = pd.Timestamp(ref_date) if ref_date else pd.Timestamp(now_local().date())
    start_dt = end_dt - timedelta(days=int(cfg["history_days"]))

    # 3) Download (Adj Close, Close, Volume) + actions
    adj_close, close, volume, splits_map, divs_map = yf_download_full(universe, start=start_dt, end=end_dt + timedelta(days=1))
    if adj_close.empty:
        raise HTTPException(status_code=500, detail="No price data downloaded.")

    # 4) Resolve safe reference date from actual trading sessions
    safe_ref_date = resolve_ref_date_from_prices(adj_close, requested_ref_date=end_dt)

    # 5) Corporate action anomaly guard
    suspects = detect_corporate_action_anomalies(adj_close, close, splits_map)
    if suspects:
        log.warning("Dropping suspected mis-adjusted tickers: %s", ", ".join(suspects))
        adj_close = adj_close.drop(columns=[c for c in suspects if c in adj_close.columns], errors="ignore")
        close = close.drop(columns=[c for c in suspects if c in close.columns], errors="ignore")
        volume = volume.drop(columns=[c for c in suspects if c in volume.columns], errors="ignore")
    if adj_close.empty:
        raise HTTPException(status_code=500, detail="All tickers filtered by anomaly guard. Check data source.")

    # 6) Momentum frame (India-safe)
    mom_df, ref_info = compute_momentum_frame_india(
        adj_close=adj_close,
        close=close,
        volume=volume,
        ref_date=safe_ref_date,
        lookback_6m_days=int(cfg["lookback_6m_days"]),
        lookback_12m_days=int(cfg["lookback_12m_days"]),
        lookback_1m_days=int(cfg["lookback_1m_days"]),
        sma_window_days=int(cfg["sma_window_days"]),
        vol_window_days=int(cfg["vol_window_days"]),
        min_avg_turnover_cr=(float(cfg["min_avg_turnover_cr"]) if cfg.get("min_avg_turnover_cr") is not None else None),
        price_cap=(float(price_cap) if price_cap is not None else None),
    )
    if mom_df.empty:
        raise HTTPException(status_code=404, detail="No eligible stocks after filters.")

    # Optional score floor
    if min_momentum_score is not None:
        mom_df = mom_df[mom_df["MomentumScore"] >= float(min_momentum_score)]
        if mom_df.empty:
            raise HTTPException(status_code=404, detail="No stocks meet the minimum momentum score.")

    # 7) Rank, pick top N, allocate
    top_n_used = min(_top_n, len(mom_df))
    top_df = mom_df.head(top_n_used).copy()
    alloc_df = suggest_allocations(top_df, _capital, top_n_used)
    alloc_df["Rank"] = np.arange(1, len(alloc_df) + 1)

    # 8) Build response stocks
    stocks = []
    # Join alloc back on ticker index
    joined = top_df.join(alloc_df.set_index("Ticker"), how="left")
    for t, r in joined.iterrows():
        stocks.append(StockData(
            ticker=t,
            rank=int(r.get("Rank", 0)),
            price=float(r["Price"]),
            qty=int(r.get("Qty", 0)),
            amount=float(r.get("Amount", 0.0)),
            allocation_pct=float(r.get("AllocationPct", 0.0)),
            momentum_score=float(r["MomentumScore"]),
            ret_6m=float(r["Ret_6M"]),
            ret_12m_ex1=float(r["Ret_12M_ex1"]),
            vol_ann=float(r["Vol_Ann"]),
            above_200dma=bool(r["Above_200DMA"]),
        ))

    # 9) CSV outputs (optional)
    csv_paths = {}
    if cfg.get("write_csv", True):
        stamp = ref_info["signal_date"]
        outdir = cfg["output_dir"]
        os.makedirs(outdir, exist_ok=True)
        prices_path = os.path.join(outdir, f"prices_adjclose_{stamp}.csv")
        scores_path = os.path.join(outdir, f"momentum_scores_{stamp}.csv")
        eligible_path = os.path.join(outdir, f"eligible_{stamp}.csv")
        top_path = os.path.join(outdir, f"top_selection_{stamp}.csv")
        ref_path = os.path.join(outdir, f"reference_{stamp}.csv")

        try:
            adj_close.to_csv(prices_path)
            mom_df.to_csv(scores_path)
            mom_df.index.to_series().to_frame("Ticker").to_csv(eligible_path, index=False)
            pd.DataFrame([s.dict() for s in stocks]).to_csv(top_path, index=False)
            pd.DataFrame([ref_info]).to_csv(ref_path, index=False)
            csv_paths = {
                "prices_adjclose": prices_path,
                "momentum_scores": scores_path,
                "eligible": eligible_path,
                "top_selection": top_path,
                "reference": ref_path,
            }
        except Exception as e:
            log.warning("CSV write failed: %s", e)

    # 10) Summary
    total_amount = sum(s.amount for s in stocks)
    utilization_pct = (total_amount / _capital) * 100.0 if _capital > 0 else 0.0
    summary = {
        "capital": _capital,
        "utilization_pct": round(utilization_pct, 2),
        "avg_momentum_score": float(np.mean([s.momentum_score for s in stocks])) if stocks else 0.0,
    }

    return RunResponse(
        reference=ref_info,
        eligible_count=int(len(mom_df)),
        selected_count=int(len(stocks)),
        stocks=stocks,
        summary=summary,
        csv_paths=csv_paths or None
    )

# ------------ /levels (ATR) ------------
@app.get("/levels", response_model=LevelsResponse)
def levels(tickers: str = Query(..., description="Comma-separated tickers (e.g., BEL.NS,MFSL.NS)")):
    syms = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not syms:
        raise HTTPException(status_code=400, detail="No tickers provided.")

    out: List[LevelData] = []
    end_dt = pd.Timestamp(now_local().date())
    start_dt = end_dt - timedelta(days=420)

    for t in syms:
        if not (t.endswith(".NS") or t.endswith(".BSE")):
            t = t + ".NS"
        try:
            df = yf.download(t, start=start_dt, end=end_dt + timedelta(days=1), auto_adjust=False, progress=False, actions=False)
            if df is None or df.empty or not {"High","Low","Close"}.issubset(df.columns):
                continue
            df = df.dropna(subset=["High","Low","Close"]).copy()
            df["PrevClose"] = df["Close"].shift(1)
            df["TR"] = np.maximum(df["High"] - df["Low"],
                           np.maximum((df["High"] - df["PrevClose"]).abs(),
                                      (df["Low"] - df["PrevClose"]).abs()))
            df["ATR14"] = df["TR"].rolling(14, min_periods=14).mean()
            last = df.dropna().iloc[-1]
            last_close = float(last["Close"])
            atr = float(last["ATR14"])
            out.append(LevelData(
                ticker=t,
                last_close=last_close,
                atr14=atr,
                stop_2atr=last_close - 2*atr,
                stop_1_5atr=last_close - 1.5*atr,
                target_2atr=last_close + 2*atr,
            ))
        except Exception:
            continue

    if not out:
        raise HTTPException(status_code=404, detail="No levels computed (insufficient data).")

    return LevelsResponse(as_of=str(end_dt.date()), levels=out)

# ------------ Main ------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
