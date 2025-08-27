# Momentum Strategy API

A FastAPI service that computes momentum-based stock selection for NSE tickers using the NIFTY 500 universe.

## Quickstart

```bash
# install dependencies (preferably in a virtual environment)
pip install -r requirements.txt

# launch API (reload optional)
uvicorn app:app --host 0.0.0.0 --port 8000
```

Navigate to `http://localhost:8000/docs` for interactive API docs.

## Endpoints

* `/health` – basic health-check.
* `/run` – runs momentum calculation and returns eligible + top selections.
* `/levels?tickers=INFY.NS,TCS.NS` – ATR-based stop suggestions.

## Configuration

Modify `config.yaml` to tweak look-back windows, capital assumptions, or output directory. Set `MOMENTUM_API_CONFIG` environment variable to load a custom config path.
