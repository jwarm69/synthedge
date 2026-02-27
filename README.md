# SynthEdge - BTC Edge Scanner

Real-time edge detection for Kalshi BTC prediction markets, powered by SynthData's decentralized ML forecasting network (Bittensor Subnet 50).

Built for the [SynthData Predictive Intelligence Hackathon](https://www.synthdata.co).

## What It Does

SynthEdge combines three independent signal sources to find mispriced Kalshi contracts:

1. **SynthData API** - Aggregated forecasts from 200+ competing ML models on Bittensor, updated every 60 seconds
2. **Local Ensemble** - XGBoost + LightGBM + CatBoost trained on historical crypto data with walk-forward validation
3. **Polymarket** - Cross-exchange probability reference from the SynthData API response

When sources **agree**, the system applies an agreement boost and sizes positions using Kelly criterion. When they **disagree**, it sits out.

## Features

- **Live Edge Scanner** - Scans 1h and 15min Kalshi BTC markets for pricing edges
- **3-Way Signal Comparison** - SynthData vs Local Ensemble vs Polymarket with unanimous agreement detection
- **Agreement-Boosted Blending** - Confidence scales up when independent sources agree
- **Price Distribution Fan Charts** - SynthData's 1000 Monte Carlo paths visualized with Kalshi strikes overlaid
- **Paper Trading** - Click-to-record paper trades with automatic P&L tracking
- **Auto-Recorder** - Background thread that continuously scans and logs edge opportunities
- **Kelly Position Sizing** - Bankroll-optimal contract sizing for each opportunity

## Architecture

```
SynthData API ──→ synthdata_client.py ──┐
(200+ models)     (15min + 1h forecasts) │
                  (percentiles, vol)      ├──→ signal_blender.py ──→ edge_detector.py ──→ dashboard_v2.py
Local Ensemble ──→ predict_v2.py ────────┘    (agreement boost)      (model vs market)     (Streamlit)
(XGB+LGB+Cat)                                                              │
                                                                    kalshi_api.py
Kalshi API ──→ kalshi_api.py                                    (1h + 15min markets)
(live pricing)  (contract structure)

Polymarket ──→ (cross-exchange reference via SynthData API)
Auto-Recorder ──→ auto_recorder.py (background, 60s)
P&L Tracker ──→ pnl_tracker.py (CSV signal log + settlement)
```

## Quick Start

```bash
# Clone and set up
git clone https://github.com/jwarm69/synthedge.git
cd synthedge

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Launch dashboard
./run_synthedge.sh
```

The dashboard runs at `http://localhost:8503`.

## Requirements

- Python 3.11+
- SynthData API key (set `SYNTHDATA_API_KEY` env var, or use the default in `run_synthedge.sh`)
- Kalshi API access (optional, for live market pricing)

Key packages: `streamlit`, `xgboost`, `lightgbm`, `catboost`, `plotly`, `pandas`, `requests`

## Training Models

Pre-trained models for BTC/ETH are included in `models/`. To retrain:

```bash
source .venv/bin/activate
python src/train_v2.py --coin btc
python src/train_v2.py --coin eth
```

## Project Structure

```
src/
  dashboard_v2.py      # Streamlit dashboard (main entry point)
  synthdata_client.py   # SynthData API wrapper with caching
  signal_blender.py     # Agreement-boosted probability fusion
  edge_detector.py      # Model vs market edge scanning
  pnl_tracker.py        # Signal recording + P&L evaluation
  auto_recorder.py      # Background signal recording thread
  predict_v2.py         # Local ensemble prediction pipeline
  kalshi_api.py         # Kalshi market data + position sizing
  data_fetch.py         # Historical OHLCV data fetching
  features.py           # Feature engineering (100+ features)
  train_v2.py           # Model training with walk-forward CV

configs/                # Model hyperparameters
models/                 # Pre-trained model files (.joblib)
data/                   # Raw data, SynthData cache, signal logs
```

## License

MIT
