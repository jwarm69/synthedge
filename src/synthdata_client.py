"""
SynthData API client for decentralized ML forecasts.

Wraps SynthData's Bittensor Subnet 50 API which aggregates predictions
from 200+ competing ML models. Provides directional forecasts, price
percentiles, and volatility estimates for BTC.

Authentication: SYNTHDATA_API_KEY env var (header: Authorization: Apikey {key})
Base URL: https://api.synthdata.co
"""

import os
import json
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional


BASE_URL = "https://api.synthdata.co"
CACHE_DIR = Path(__file__).parent.parent / "data" / "synthdata"
CACHE_TTL_SECONDS = 60


def _get_api_key() -> str:
    key = os.environ.get("SYNTHDATA_API_KEY", "")
    if not key:
        raise ValueError("SYNTHDATA_API_KEY environment variable not set")
    return key


def _headers() -> dict:
    return {
        "Authorization": f"Apikey {_get_api_key()}",
        "Accept": "application/json",
    }


def _cache_path(endpoint: str, asset: str, horizon: str = "") -> Path:
    safe_name = endpoint.replace("/", "_").strip("_")
    suffix = f"_{horizon}" if horizon else ""
    return CACHE_DIR / f"{safe_name}_{asset}{suffix}.json"


def _read_cache(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        mtime = path.stat().st_mtime
        if time.time() - mtime > CACHE_TTL_SECONDS:
            return None
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _write_cache(path: Path, data: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception:
        pass


def _api_get(endpoint: str, params: dict = None, timeout: int = 15) -> Optional[dict]:
    """Make authenticated GET request to SynthData API."""
    url = f"{BASE_URL}{endpoint}"
    try:
        resp = requests.get(url, headers=_headers(), params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        print(f"SynthData API HTTP error on {endpoint}: {e}")
        return None
    except requests.exceptions.ConnectionError:
        print(f"SynthData API connection error on {endpoint}")
        return None
    except Exception as e:
        print(f"SynthData API error on {endpoint}: {e}")
        return None


def get_directional_forecast(asset: str = "BTC", horizon: str = "1h") -> dict:
    """
    Get directional probability forecast from SynthData.

    Args:
        asset: "BTC" (or "ETH")
        horizon: "1h" or "15min"

    Returns:
        {
            "status": "ok"|"unavailable",
            "prob_up": float,   # probability price goes up
            "prob_down": float, # probability price goes down
            "direction": "UP"|"DOWN",
            "confidence": float, # 0-1 scale
            "horizon": str,
            "timestamp": str,
            "source": "synthdata"
        }
    """
    # Map horizon to endpoint
    if horizon in ("5min", "5m"):
        endpoint = "/insights/polymarket/up-down/5min"
        hz = "5min"
    elif horizon in ("15min", "15m"):
        endpoint = "/insights/polymarket/up-down/15min"
        hz = "15min"
    else:
        endpoint = "/insights/polymarket/up-down/hourly"
        hz = "1h"

    cache_key = _cache_path("directional", asset, hz)

    # Try cache first
    cached = _read_cache(cache_key)
    if cached and cached.get("status") == "ok":
        return cached

    # Call API
    raw = _api_get(endpoint, params={"asset": asset.upper()})

    if raw is None:
        # Fallback to stale cache
        if cache_key.exists():
            try:
                with open(cache_key) as f:
                    stale = json.load(f)
                stale["status"] = "stale"
                return stale
            except Exception:
                pass
        return {"status": "unavailable", "horizon": hz, "source": "synthdata"}

    # Parse response - real API returns:
    # {"synth_probability_up": 0.598, "synth_outcome": "Up",
    #  "current_price": 67492.45, "start_price": 67480.61, ...}
    prob_up = _extract_prob_up(raw)
    prob_down = 1.0 - prob_up

    result = {
        "status": "ok",
        "prob_up": prob_up,
        "prob_down": prob_down,
        "direction": "UP" if prob_up > 0.5 else "DOWN",
        "confidence": abs(prob_up - 0.5) * 2,
        "horizon": hz,
        "timestamp": datetime.utcnow().isoformat(),
        "source": "synthdata",
        "current_price": raw.get("current_price", 0),
        "start_price": raw.get("start_price", 0),
        "synth_outcome": raw.get("synth_outcome", ""),
        "polymarket_prob_up": raw.get("polymarket_probability_up", 0),
        "event_end_time": raw.get("event_end_time", ""),
        "raw": raw,
    }

    _write_cache(cache_key, result)
    return result


def _extract_prob_up(raw: dict) -> float:
    """Extract probability of UP from API response.

    Real API returns: {"synth_probability_up": 0.598, "synth_outcome": "Up", ...}
    """
    # Primary: synth_probability_up (real API format)
    if "synth_probability_up" in raw:
        return float(raw["synth_probability_up"])

    # Fallback field names
    for key in ("prob_up", "probability_up", "up_probability", "p_up", "yes_probability"):
        if key in raw:
            return float(raw[key])

    # Nested under common wrappers
    for wrapper in ("data", "result", "prediction", "forecast"):
        if wrapper in raw and isinstance(raw[wrapper], dict):
            inner = raw[wrapper]
            if "synth_probability_up" in inner:
                return float(inner["synth_probability_up"])
            for key in ("prob_up", "probability_up", "up_probability"):
                if key in inner:
                    return float(inner[key])

    # If response has "up" and "down" keys
    if "up" in raw and "down" in raw:
        return float(raw["up"])

    # If response is a list, take the latest
    if isinstance(raw, list) and raw:
        return _extract_prob_up(raw[-1])

    print(f"Warning: Could not parse SynthData response format: {list(raw.keys())}")
    return 0.5


def get_price_percentiles(asset: str = "BTC", horizon: str = "1h") -> dict:
    """
    Get price distribution percentiles from SynthData's 1000 Monte Carlo paths.

    Returns:
        {
            "status": "ok"|"unavailable",
            "percentiles": {"p5": float, "p10": float, ..., "p95": float},
            "current_price": float,
            "expected_move_pct": float,  # p75-p25 range as % of current
            "horizon": str,
        }
    """
    hz = "15min" if horizon in ("15min", "15m") else "1h"
    cache_key = _cache_path("percentiles", asset, hz)

    cached = _read_cache(cache_key)
    if cached and cached.get("status") == "ok":
        return cached

    # Percentiles endpoint doesn't take a horizon param
    raw = _api_get("/insights/prediction-percentiles", params={
        "asset": asset.upper(),
    })

    if raw is None:
        if cache_key.exists():
            try:
                with open(cache_key) as f:
                    stale = json.load(f)
                stale["status"] = "stale"
                return stale
            except Exception:
                pass
        return {"status": "unavailable", "horizon": hz, "source": "synthdata",
                "percentiles": {}, "current_price": 0, "expected_move_pct": 0}

    # Parse - real API returns:
    # {"current_price": 67492.45, "forecast_future": {"percentiles": [
    #   {"0.05": 67328.73, "0.2": 67420.92, "0.5": 67488.75, ...}, ...]}}
    percentiles = _extract_percentiles(raw)
    current_price = float(raw.get("current_price", 0))

    # Calculate expected move
    p25 = percentiles.get("p25", 0)
    p75 = percentiles.get("p75", 0)
    expected_move_pct = ((p75 - p25) / current_price * 100) if current_price else 0

    result = {
        "status": "ok",
        "percentiles": percentiles,
        "current_price": current_price,
        "expected_move_pct": expected_move_pct,
        "horizon": hz,
        "timestamp": datetime.utcnow().isoformat(),
        "source": "synthdata",
        "raw": raw,
    }

    _write_cache(cache_key, result)
    return result


def _extract_percentiles(raw: dict) -> dict:
    """Extract percentile values from API response.

    Real API returns: {"forecast_future": {"percentiles": [
        {"0.05": price, "0.2": price, "0.5": price, "0.8": price, "0.95": price, ...},
        ... (multiple timesteps)
    ]}}
    We take the LAST timestep (furthest horizon point).
    """
    pcts = {}

    # Map from API decimal keys to our p-format
    decimal_to_pkey = {
        "0.005": "p0.5", "0.05": "p5", "0.2": "p20", "0.35": "p35",
        "0.5": "p50", "0.65": "p65", "0.8": "p80", "0.95": "p95", "0.995": "p99.5",
    }
    # Also map to our standard names
    decimal_to_standard = {
        "0.05": "p5", "0.2": "p20", "0.35": "p25",  # approximate p25
        "0.5": "p50", "0.65": "p75",  # approximate p75
        "0.8": "p80", "0.95": "p95",
    }

    # Real API: forecast_future.percentiles is a list of timestep dicts
    forecast = raw.get("forecast_future", {})
    pct_list = forecast.get("percentiles", [])

    if isinstance(pct_list, list) and pct_list:
        # Take the last timestep (1h or 15min horizon endpoint)
        last_step = pct_list[-1] if len(pct_list) > 0 else {}

        for decimal_key, price in last_step.items():
            try:
                p_val = float(decimal_key)
                p_int = int(p_val * 100)
                pcts[f"p{p_int}"] = float(price)
            except (ValueError, TypeError):
                continue

        # Create convenient aliases: p25~p20 or p35, p75~p65 or p80
        if "p25" not in pcts and "p20" in pcts:
            pcts["p25"] = pcts["p20"]
        if "p25" not in pcts and "p35" in pcts:
            pcts["p25"] = pcts["p35"]
        if "p75" not in pcts and "p80" in pcts:
            pcts["p75"] = pcts["p80"]
        if "p75" not in pcts and "p65" in pcts:
            pcts["p75"] = pcts["p65"]
        if "p10" not in pcts and "p5" in pcts:
            pcts["p10"] = (pcts.get("p5", 0) + pcts.get("p20", pcts.get("p5", 0))) / 2
        if "p90" not in pcts and "p95" in pcts:
            pcts["p90"] = (pcts.get("p80", pcts.get("p95", 0)) + pcts.get("p95", 0)) / 2

        return pcts

    # Fallback: check flat format
    target_pcts = [5, 10, 25, 50, 75, 90, 95]
    sources = [raw]
    for wrapper in ("data", "result", "percentiles", "prediction"):
        if wrapper in raw and isinstance(raw[wrapper], dict):
            sources.append(raw[wrapper])

    for src in sources:
        for p in target_pcts:
            key = f"p{p}"
            if key not in pcts:
                for candidate in (f"p{p}", f"p{p:02d}", f"percentile_{p}", str(p)):
                    if candidate in src:
                        pcts[key] = float(src[candidate])
                        break

    return pcts


def _extract_field(raw: dict, candidates: tuple) -> float:
    """Extract a numeric field by trying multiple key names."""
    sources = [raw]
    for wrapper in ("data", "result"):
        if wrapper in raw and isinstance(raw[wrapper], dict):
            sources.append(raw[wrapper])

    for src in sources:
        for key in candidates:
            if key in src:
                try:
                    return float(src[key])
                except (ValueError, TypeError):
                    continue
    return 0.0


def get_volatility_forecast(asset: str = "BTC", horizon: str = "1h") -> dict:
    """
    Get volatility forecast from SynthData.

    Returns:
        {
            "status": "ok"|"unavailable",
            "forward_vol": float,    # predicted forward volatility
            "realized_vol": float,   # recent realized volatility
            "vol_ratio": float,      # forward/realized (>1 = expanding, <1 = contracting)
            "horizon": str,
        }
    """
    hz = "15min" if horizon in ("15min", "15m") else "1h"
    cache_key = _cache_path("volatility", asset, hz)

    cached = _read_cache(cache_key)
    if cached and cached.get("status") == "ok":
        return cached

    # Volatility endpoint doesn't take a horizon param
    raw = _api_get("/insights/volatility", params={
        "asset": asset.upper(),
    })

    if raw is None:
        if cache_key.exists():
            try:
                with open(cache_key) as f:
                    stale = json.load(f)
                stale["status"] = "stale"
                return stale
            except Exception:
                pass
        return {"status": "unavailable", "horizon": hz, "source": "synthdata"}

    # Real API returns:
    # {"realized": {"average_volatility": x}, "forecast_future": {"average_volatility": y}, ...}
    forecast_future = raw.get("forecast_future", {})
    realized_data = raw.get("realized", {})

    forward_vol = float(forecast_future.get("average_volatility", 0) or
                        _extract_field(raw, ("forward_vol", "predicted_volatility", "forecast_vol")))
    realized_vol = float(realized_data.get("average_volatility", 0) or
                         _extract_field(raw, ("realized_vol", "realized_volatility", "historical_vol")))
    vol_ratio = (forward_vol / realized_vol) if realized_vol > 0 else 1.0

    result = {
        "status": "ok",
        "forward_vol": forward_vol,
        "realized_vol": realized_vol,
        "vol_ratio": vol_ratio,
        "horizon": hz,
        "timestamp": datetime.utcnow().isoformat(),
        "source": "synthdata",
        "raw": raw,
    }

    _write_cache(cache_key, result)
    return result


def get_leaderboard() -> dict:
    """
    Get the latest miner leaderboard from Bittensor Subnet 50.

    Returns:
        {
            "status": "ok"|"unavailable",
            "miners": [{"uid": int, "hotkey": str, "crps_score": float, ...}],
            "top_count": int,
            "timestamp": str,
        }
    """
    cache_key = _cache_path("leaderboard", "global")

    cached = _read_cache(cache_key)
    if cached and cached.get("status") == "ok":
        return cached

    raw = _api_get("/v2/leaderboard/latest")

    if raw is None:
        if cache_key.exists():
            try:
                with open(cache_key) as f:
                    stale = json.load(f)
                stale["status"] = "stale"
                return stale
            except Exception:
                pass
        return {"status": "unavailable", "miners": [], "top_count": 0}

    # Parse response — may be a list of miners or wrapped in a dict
    miners = []
    if isinstance(raw, list):
        miners = raw
    elif isinstance(raw, dict):
        for key in ("miners", "data", "leaderboard", "results"):
            if key in raw and isinstance(raw[key], list):
                miners = raw[key]
                break
        if not miners and "uid" in raw:
            miners = [raw]

    result = {
        "status": "ok",
        "miners": miners[:50],  # cap at 50
        "top_count": len(miners),
        "timestamp": datetime.utcnow().isoformat(),
        "source": "synthdata",
        "raw": raw if isinstance(raw, dict) else {"data": raw},
    }

    _write_cache(cache_key, result)
    return result


def get_meta_leaderboard() -> dict:
    """
    Get meta-leaderboard (aggregated performance over rolling windows).

    Returns:
        {
            "status": "ok"|"unavailable",
            "rankings": [{"uid": int, "meta_score": float, ...}],
            "timestamp": str,
        }
    """
    cache_key = _cache_path("meta_leaderboard", "global")

    cached = _read_cache(cache_key)
    if cached and cached.get("status") == "ok":
        return cached

    raw = _api_get("/v2/meta-leaderboard/latest")

    if raw is None:
        if cache_key.exists():
            try:
                with open(cache_key) as f:
                    stale = json.load(f)
                stale["status"] = "stale"
                return stale
            except Exception:
                pass
        return {"status": "unavailable", "rankings": []}

    rankings = []
    if isinstance(raw, list):
        rankings = raw
    elif isinstance(raw, dict):
        for key in ("rankings", "data", "leaderboard", "results", "miners"):
            if key in raw and isinstance(raw[key], list):
                rankings = raw[key]
                break

    result = {
        "status": "ok",
        "rankings": rankings[:50],
        "timestamp": datetime.utcnow().isoformat(),
        "source": "synthdata",
        "raw": raw if isinstance(raw, dict) else {"data": raw},
    }

    _write_cache(cache_key, result)
    return result


def get_validation_scores() -> dict:
    """
    Get latest validation scores across miners.

    Returns:
        {
            "status": "ok"|"unavailable",
            "scores": [{"uid": int, "score": float, ...}],
            "avg_score": float,
            "score_variance": float,
            "timestamp": str,
        }
    """
    cache_key = _cache_path("validation_scores", "global")

    cached = _read_cache(cache_key)
    if cached and cached.get("status") == "ok":
        return cached

    raw = _api_get("/validation/scores/latest")

    if raw is None:
        if cache_key.exists():
            try:
                with open(cache_key) as f:
                    stale = json.load(f)
                stale["status"] = "stale"
                return stale
            except Exception:
                pass
        return {"status": "unavailable", "scores": [], "avg_score": 0, "score_variance": 0}

    scores = []
    if isinstance(raw, list):
        scores = raw
    elif isinstance(raw, dict):
        for key in ("scores", "data", "results", "validations"):
            if key in raw and isinstance(raw[key], list):
                scores = raw[key]
                break

    # Compute aggregate stats
    score_vals = []
    for s in scores:
        for key in ("score", "crps_score", "validation_score", "value"):
            if key in s:
                try:
                    score_vals.append(float(s[key]))
                except (ValueError, TypeError):
                    pass
                break

    avg_score = sum(score_vals) / len(score_vals) if score_vals else 0
    variance = (
        sum((v - avg_score) ** 2 for v in score_vals) / len(score_vals)
        if score_vals else 0
    )

    result = {
        "status": "ok",
        "scores": scores[:50],
        "avg_score": avg_score,
        "score_variance": variance,
        "timestamp": datetime.utcnow().isoformat(),
        "source": "synthdata",
        "raw": raw if isinstance(raw, dict) else {"data": raw},
    }

    _write_cache(cache_key, result)
    return result


def get_all_signals(asset: str = "BTC") -> dict:
    """
    Fetch all SynthData signals for both horizons.

    Returns combined dict with keys:
        "1h": {"directional": {...}, "percentiles": {...}, "volatility": {...}}
        "15min": {"directional": {...}, "percentiles": {...}, "volatility": {...}}
        "status": "ok"|"partial"|"unavailable"
    """
    result = {"status": "unavailable"}
    ok_count = 0
    total = 0

    for hz in ("1h", "15min", "5min"):
        hz_data = {}

        directional = get_directional_forecast(asset, hz)
        hz_data["directional"] = directional
        total += 1
        if directional.get("status") in ("ok", "stale"):
            ok_count += 1

        percentiles = get_price_percentiles(asset, hz)
        hz_data["percentiles"] = percentiles
        total += 1
        if percentiles.get("status") in ("ok", "stale"):
            ok_count += 1

        volatility = get_volatility_forecast(asset, hz)
        hz_data["volatility"] = volatility
        total += 1
        if volatility.get("status") in ("ok", "stale"):
            ok_count += 1

        result[hz] = hz_data

    if ok_count == total:
        result["status"] = "ok"
    elif ok_count > 0:
        result["status"] = "partial"
    else:
        result["status"] = "unavailable"

    result["timestamp"] = datetime.utcnow().isoformat()
    return result


if __name__ == "__main__":
    print("Testing SynthData API client...")
    print(f"API Key set: {'Yes' if os.environ.get('SYNTHDATA_API_KEY') else 'No'}")
    print()

    # Test directional forecasts
    for hz in ("1h", "15min"):
        print(f"--- Directional Forecast ({hz}) ---")
        d = get_directional_forecast("BTC", hz)
        print(f"  Status: {d['status']}")
        if d["status"] == "ok":
            print(f"  Direction: {d['direction']}")
            print(f"  P(UP): {d['prob_up']:.3f}")
            print(f"  Confidence: {d['confidence']:.3f}")
        print()

    # Test percentiles
    print("--- Price Percentiles (1h) ---")
    p = get_price_percentiles("BTC", "1h")
    print(f"  Status: {p['status']}")
    if p["status"] == "ok":
        for k, v in p.get("percentiles", {}).items():
            print(f"  {k}: ${v:,.2f}")
        print(f"  Expected move: {p['expected_move_pct']:.2f}%")
    print()

    # Test volatility
    print("--- Volatility Forecast (1h) ---")
    v = get_volatility_forecast("BTC", "1h")
    print(f"  Status: {v['status']}")
    if v["status"] == "ok":
        print(f"  Forward vol: {v['forward_vol']:.4f}")
        print(f"  Realized vol: {v['realized_vol']:.4f}")
        print(f"  Vol ratio: {v['vol_ratio']:.2f}")
