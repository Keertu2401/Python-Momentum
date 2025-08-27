from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import yfinance as yf
import yaml
import os
from functools import lru_cache
from datetime import datetime
from dateutil.tz import gettz
from typing import List, Dict, Any

# --------------------------------------------------
# Config helpers
# --------------------------------------------------
CFG_PATH = os.getenv("MOMENTUM_API_CONFIG", "config.yaml")

@lru_cache(maxsize=1)
def load_config() -> Dict[str, Any]:
    """Load YAML config once and cache it."""
    with open(CFG_PATH, "r") as f:
        return yaml.safe_load(f)

cfg = load_config()
# Ensure output directory exists upfront
os.makedirs(cfg.get("output_dir", "output"), exist_ok=True)

# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI(title="Momentum Strategy API", version="1.1.0")

# --------------------------------------------------
# Utility functions
# --------------------------------------------------

def now_local() -> pd.Timestamp:
    """Current date normalized to local timezone specified in config."""
    return pd.Timestamp.now(tz=gettz(cfg.get("tz", "Asia/Kolkata"))).normalize()

@lru_cache(maxsize=1)
def load_universe() -> List[str]:
    """Return list of NSE tickers (with .NS suffix). The result is cached."""
    url = cfg["universe_source"]
    try:
        df = pd.read_csv(url)
    except Exception:
        # Fallback to local relative CSV shipped with repo (if available)
        df = pd.read_csv("ind_nifty500list.csv")
    col = "Symbol" if "Symbol" in df.columns else "SYMBOL"
    syms = df[col].astype(str).str.strip().tolist()
    return [s + ".NS" for s in syms]


def yf_download(tickers: List[str], start: pd.Timestamp, end: pd.Timestamp, batch: int = 50) -> pd.DataFrame:
    """Download adjusted prices in batches and return a wide DataFrame of Close prices."""
    frames: List[pd.DataFrame] = []
    # yfinance struggles with very large requests; batching gives better reliability
    for i in range(0, len(tickers), batch):
        chunk = tickers[i : i + batch]
        # yfinance end date is exclusive -> add +1 day to include data up to `end`
        raw = yf.download(
            chunk,
            start=start,
            end=end + pd.Timedelta(days=1),
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if raw.empty:
            continue

        # Multi-index (field, ticker) format when multiple tickers supplied
        if isinstance(raw.columns, pd.MultiIndex):
            if ("Close" in raw.columns.get_level_values(0)):
                df = raw["Close"].copy()
            else:
                # Fallback in rare cases where Close not present
                first_field = raw.columns.get_level_values(0).unique()[0]
                df = raw[first_field].copy()
        else:
            # Single ticker DataFrame
            df = raw.copy()
            # Ensure column name equals ticker for consistency
            df = df[["Close"]] if "Close" in df.columns else df.iloc[:, [0]]
            df.columns = chunk  # `chunk` length is 1 here
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1)
    out = out.sort_index().ffill()  # Forward-fill missing daily data
    out = out.loc[:, out.notna().any()]  # drop entirely-empty columns
    return out


def compute_indices(prices: pd.DataFrame, ref_date: pd.Timestamp):
    """Return row indices aligned to price DataFrame for required look-back windows."""
    # Find the position of the reference date (last trading day <= ref_date)
    if ref_date not in prices.index:
        pos = prices.index.get_indexer([ref_date], method="pad")[0]
    else:
        pos = prices.index.get_loc(ref_date)

    need = max(cfg["lookback_12m_days"], cfg["sma_window_days"], cfg["vol_window_days"])
    if pos < need:
        raise ValueError(
            f"Not enough history at {ref_date.date()} (need {need} trading days, got {pos})."
        )

    return {
        "today": pos,
        "6m": pos - cfg["lookback_6m_days"],
        "12m": pos - cfg["lookback_12m_days"],
        "1m_back": pos - cfg["exclude_1m_days"],
    }


def momentum_frame(prices: pd.DataFrame, ref_date: pd.Timestamp):
    """Compute momentum metrics and return detailed DataFrame + meta reference DataFrame."""

    idx = compute_indices(prices, ref_date)

    today = prices.iloc[idx["today"]]
    sixm = prices.iloc[idx["6m"]]
    twlv = prices.iloc[idx["12m"]]
    one_m = prices.iloc[idx["1m_back"]]

    # Tick symbols with complete data across all look-back points
    valid = today.dropna().index.intersection(sixm.dropna().index)
    valid = valid.intersection(twlv.dropna().index).intersection(one_m.dropna().index)

    if valid.empty:
        raise ValueError("No tickers with complete history for calculations.")

    # ---------------- Statistical measures ----------------

    # Volatility: std of daily pct-change over last 126 trading days, annualised
    pct_window = prices[valid].iloc[idx["today"] - cfg["vol_window_days"] : idx["today"]]
    pct = pct_window.pct_change().dropna(how="all")
    vol = pct.std() * np.sqrt(252) * 100

    # Returns
    ret_6m = (today[valid] / sixm[valid] - 1.0) * 100
    ret_12m_ex1 = (one_m[valid] / twlv[valid] - 1.0) * 100

    # 200-DMA trend filter
    sma_slice = prices[valid].iloc[idx["today"] - cfg["sma_window_days"] : idx["today"]]
    sma_200 = sma_slice.mean()
    above_200dma = today[valid] > sma_200

    score = (ret_6m + ret_12m_ex1) / vol.replace(0, np.nan)

    df = (
        pd.concat(
            {
                "Price": today[valid],
                "6M_Return(%)": ret_6m,
                "12M_ex1_Return(%)": ret_12m_ex1,
                "Volatility(%)": vol,
                "MomentumScore": score,
                "Above_200DMA": above_200dma,
                "Positive_6M": ret_6m > 0,
            },
            axis=1,
        )
        .dropna()
    )

    # Reference snapshot (helps external automation pipelines)
    ref = pd.DataFrame(
        [
            {
                "ReferenceDate": prices.index[idx["today"]],
                "TodayDate": prices.index[idx["today"]],
                "Date_6M": prices.index[idx["6m"]],
                "Date_12M": prices.index[idx["12m"]],
                "Date_1MBack": prices.index[idx["1m_back"]],
                "UniverseCount": prices.shape[1],
                "EligibleCount": int(((df["Above_200DMA"]) & (df["Positive_6M"])).sum()),
            }
        ]
    )

    return df, ref


def suggest_allocations(portfolio: pd.DataFrame, capital: float, top_n: int) -> pd.DataFrame:
    """Equal-weight allocation across *selected* tickers."""
    alloc_per = capital / max(1, top_n)
    qty = (alloc_per // portfolio["Price"]).astype(int)
    amt = qty * portfolio["Price"]
    out = portfolio.copy()
    out["Qty"] = qty
    out["Amount"] = amt
    return out

# --------------------------------------------------
# Pydantic models (for docs & validation)
# --------------------------------------------------
class RunResponse(BaseModel):
    reference_dates: Dict[str, str]
    eligible: List[Dict[str, Any]]
    top: List[Dict[str, Any]]
    csv_paths: Dict[str, str]


# --------------------------------------------------
# API Endpoints
# --------------------------------------------------

@app.get("/health")
async def health():
    return {"ok": True, "ts": datetime.now().isoformat()}


@app.get("/run", response_model=RunResponse)
async def run(
    capital: float | None = Query(None, description="Override capital (default from config)"),
    top_n: int | None = Query(None, description="Override top_n (default from config)"),
    price_cap: float | None = Query(
        None, description="Price cap in INR; set null to disable filter"
    ),
    ref_date: str = Query("", description="Reference date YYYY-MM-DD; default=last trading day"),
):
    """Main momentum computation endpoint."""
    # ----- Resolve runtime parameters -----
    _capital = capital if capital is not None else cfg["capital"]
    _top_n = top_n if top_n is not None else cfg["top_n"]
    _cap = price_cap if price_cap is not None else cfg["price_cap"]
    out_dir = cfg["output_dir"]

    # ----- Universe & price history -----
    universe = load_universe()
    end_dt = pd.Timestamp(ref_date) if ref_date else now_local()
    start_dt = end_dt - pd.Timedelta(days=cfg["history_days"])

    prices = yf_download(universe, start_dt, end_dt)
    if prices.empty or prices.shape[0] < max(
        cfg["lookback_12m_days"], cfg["sma_window_days"]
    ):
        raise HTTPException(status_code=422, detail="Insufficient price history retrieved.")

    # ----- Momentum calculations -----
    mom_df, ref_df = momentum_frame(prices, end_dt)

    # Eligibility filters
    eligible = mom_df[mom_df["Above_200DMA"] & mom_df["Positive_6M"]].copy()
    if _cap is not None:
        eligible = eligible[eligible["Price"] <= _cap]

    ranked = eligible.sort_values("MomentumScore", ascending=False)
    top_sel = ranked.head(_top_n)
    top_sel = suggest_allocations(top_sel, _capital, _top_n)
    top_sel.insert(0, "Rank", range(1, len(top_sel) + 1))

    # ----- Optional CSV outputs -----
    csv_paths: Dict[str, str] = {}
    if cfg["write_csv"]:
        prices.to_csv(os.path.join(out_dir, "phase2_prices_snapshot.csv"))
        mom_df.to_csv(os.path.join(out_dir, "phase3_momentum_scores.csv"))
        eligible.to_csv(os.path.join(out_dir, "phase4_filtered_eligible.csv"))
        top_sel.to_csv(os.path.join(out_dir, "phase4_top_selection.csv"))
        ref_df.to_csv(os.path.join(out_dir, "phase3_reference_dates.csv"), index=False)
        csv_paths = {
            "prices": "phase2_prices_snapshot.csv",
            "scores": "phase3_momentum_scores.csv",
            "eligible": "phase4_filtered_eligible.csv",
            "top": "phase4_top_selection.csv",
            "reference_dates": "phase3_reference_dates.csv",
        }

    # ----- Build JSON payloads (round values for readability) -----
    eligible_json = (
        eligible.reset_index()
        .rename(columns={"index": "Ticker"})
        .round(6)
        .to_dict(orient="records")
    )
    top_json = (
        top_sel.reset_index().rename(columns={"index": "Ticker"}).round(6).to_dict("records")
    )

    return {
        "reference_dates": ref_df.iloc[0].astype(str).to_dict(),
        "eligible": eligible_json,
        "top": top_json,
        "csv_paths": csv_paths,
    }


@app.get("/levels")
async def levels(
    tickers: str = Query(..., description="Comma-separated tickers, e.g., BEL.NS,MFSL.NS")
):
    """Return simple 2×ATR14 stop suggestions for provided tickers."""
    tick_list = [t.strip() for t in tickers.split(",") if t.strip()]
    if not tick_list:
        raise HTTPException(status_code=400, detail="No valid tickers supplied.")

    start = now_local() - pd.Timedelta(days=180)
    end = now_local()

    rows: List[Dict[str, Any]] = []
    for t in tick_list:
        raw = yf.download(
            t,
            start=start,
            end=end + pd.Timedelta(days=1),
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if raw.empty or not {"High", "Low", "Close"}.issubset(raw.columns):
            continue

        df = raw.copy()
        tr = np.maximum(
            df["High"] - df["Low"],
            np.maximum(
                (df["High"] - df["Close"].shift()).abs(),
                (df["Low"] - df["Close"].shift()).abs(),
            ),
        )
        atr = tr.rolling(14).mean()
        last_close = float(df.iloc[-1]["Close"])
        last_atr = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else None
        stop = float(last_close - 2 * last_atr) if last_atr is not None else None

        rows.append({"Ticker": t, "Close": last_close, "ATR14": last_atr, "Stop_2xATR": stop})

    return {"levels": rows}