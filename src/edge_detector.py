"""
Edge detector: compare blended model probabilities against Kalshi market prices.

Finds opportunities where our combined signal (SynthData + ensemble) disagrees
with the market-implied probability, filtered by minimum edge threshold.
"""

import numpy as np
from datetime import datetime
from typing import Optional

from kalshi_api import (
    get_market_snapshot,
    expected_value_yes,
    kelly_fraction,
    position_size,
    KALSHI_TICKERS,
)
from synthdata_client import get_all_signals, get_directional_forecast, get_price_percentiles
from signal_blender import (
    blend_predictions,
    blend_synthdata_only,
    Agreement,
    EdgeQuality,
)

# Import predict_v2 for local ensemble (may fail if models not trained)
try:
    from predict_v2 import predict_with_threshold, DEFAULT_CALIBRATION
    from data_fetch import get_latest_candles
    HAS_ENSEMBLE = True
except ImportError:
    HAS_ENSEMBLE = False
    DEFAULT_CALIBRATION = "shrink"

# Minimum edge (in probability points) to consider a trade
MIN_EDGE_THRESHOLD = 0.05  # 5 cents on a $1 contract
DEFAULT_BANKROLL = 1000.0
DEFAULT_MAX_KELLY = 0.25


def estimate_strike_probability(
    blended_prob_up: float,
    current_price: float,
    strike: float,
    contract_type: str,
    percentiles: Optional[dict] = None,
) -> float:
    """
    Estimate probability that a specific Kalshi contract resolves YES.

    For "above" contracts: P(price > strike at settlement)
    For "below" contracts: P(price < strike at settlement)
    For "range" contracts: P(floor < price < cap at settlement)

    Uses SynthData percentile CDF when available, falls back to
    blended directional probability with distance scaling.
    """
    if percentiles and len(percentiles) >= 3:
        cdf = build_percentile_cdf(percentiles, current_price)
        if cdf is not None:
            if contract_type == "above":
                return max(0.01, min(0.99, 1.0 - cdf(strike)))
            elif contract_type == "below":
                return max(0.01, min(0.99, cdf(strike)))
            elif contract_type == "between":
                # For range contracts, strike is (floor, cap) tuple
                if isinstance(strike, (tuple, list)) and len(strike) == 2:
                    return max(0.01, min(0.99, cdf(strike[1]) - cdf(strike[0])))

    # Fallback: scale directional probability by distance from current price
    if contract_type == "above":
        distance_pct = (strike - current_price) / current_price
        # Further strikes are less likely
        if distance_pct > 0:
            # Strike above current: P(above) decreases with distance
            decay = np.exp(-distance_pct * 100)  # fast decay for far strikes
            return max(0.01, min(0.99, blended_prob_up * decay))
        else:
            # Strike below current: very likely to be above
            return max(0.01, min(0.99, 0.5 + blended_prob_up * 0.4))

    elif contract_type == "below":
        distance_pct = (current_price - strike) / current_price
        prob_down = 1.0 - blended_prob_up
        if distance_pct > 0:
            decay = np.exp(-distance_pct * 100)
            return max(0.01, min(0.99, prob_down * decay))
        else:
            return max(0.01, min(0.99, 0.5 + prob_down * 0.4))

    return 0.5


def build_percentile_cdf(percentiles: dict, current_price: float):
    """
    Build a CDF interpolation function from SynthData percentiles.

    Args:
        percentiles: {"p5": price, "p10": price, ..., "p95": price}
        current_price: current BTC price

    Returns:
        Callable that maps price -> cumulative probability, or None
    """
    # Extract percentile points
    pct_map = {
        "p5": 0.05, "p10": 0.10, "p25": 0.25, "p50": 0.50,
        "p75": 0.75, "p90": 0.90, "p95": 0.95,
    }

    prices = []
    probs = []

    for key, prob in sorted(pct_map.items(), key=lambda x: x[1]):
        if key in percentiles and percentiles[key] > 0:
            prices.append(percentiles[key])
            probs.append(prob)

    if len(prices) < 3:
        return None

    def cdf(x):
        """Interpolate CDF at price x."""
        if x <= prices[0]:
            return probs[0] * (x / prices[0]) if prices[0] > 0 else 0.0
        if x >= prices[-1]:
            return probs[-1] + (1.0 - probs[-1]) * min(1.0, (x - prices[-1]) / (prices[-1] * 0.01 + 1))

        # Linear interpolation between percentile points
        for i in range(len(prices) - 1):
            if prices[i] <= x <= prices[i + 1]:
                frac = (x - prices[i]) / (prices[i + 1] - prices[i]) if prices[i + 1] != prices[i] else 0
                return probs[i] + frac * (probs[i + 1] - probs[i])

        return 0.5

    return cdf


def scan_edge(
    blended_signal: dict,
    market_structure: dict,
    current_price: float,
    horizon: str = "1h",
    bankroll: float = DEFAULT_BANKROLL,
    percentiles: Optional[dict] = None,
) -> list:
    """
    Scan for edge opportunities in a single market snapshot.

    Args:
        blended_signal: output from signal_blender.blend_predictions()
        market_structure: output from kalshi_api.parse_market_structure()
        current_price: current BTC price
        horizon: "1h" or "15min"
        bankroll: bankroll for position sizing
        percentiles: SynthData percentiles dict (optional)

    Returns:
        List of edge opportunity dicts, sorted by expected value
    """
    opportunities = []
    blended_prob = blended_signal["blended_prob_up"]

    # Scan above strikes
    for strike in market_structure.get("above_strikes", []):
        floor = strike.get("floor_strike", 0)
        yes_ask = strike.get("yes_ask", 0)
        if yes_ask <= 0 or yes_ask >= 1:
            continue

        model_prob = estimate_strike_probability(
            blended_prob, current_price, floor, "above", percentiles
        )
        market_implied = yes_ask  # yes_ask ~ market's implied P(YES)
        edge = model_prob - market_implied

        if abs(edge) >= MIN_EDGE_THRESHOLD:
            ev = expected_value_yes(model_prob, yes_ask)
            kelly = kelly_fraction(model_prob, yes_ask)
            contracts = position_size(bankroll, model_prob, yes_ask)

            opportunities.append({
                "contract_type": "above",
                "strike": floor,
                "ticker": strike.get("ticker", ""),
                "subtitle": strike.get("subtitle", f"Above ${floor:,.0f}"),
                "model_prob": model_prob,
                "market_prob": market_implied,
                "edge": edge,
                "ev_per_contract": ev,
                "kelly_fraction": kelly,
                "contracts": contracts,
                "yes_ask": yes_ask,
                "side": "YES" if edge > 0 else "NO",
                "action": f"BUY {'YES' if edge > 0 else 'NO'}",
                "horizon": horizon,
                "volume": strike.get("volume", 0),
                "open_interest": strike.get("open_interest", 0),
            })

    # Scan below strikes
    for strike in market_structure.get("below_strikes", []):
        cap = strike.get("cap_strike", 0)
        yes_ask = strike.get("yes_ask", 0)
        if yes_ask <= 0 or yes_ask >= 1:
            continue

        model_prob = estimate_strike_probability(
            blended_prob, current_price, cap, "below", percentiles
        )
        market_implied = yes_ask
        edge = model_prob - market_implied

        if abs(edge) >= MIN_EDGE_THRESHOLD:
            ev = expected_value_yes(model_prob, yes_ask)
            kelly = kelly_fraction(model_prob, yes_ask)
            contracts = position_size(bankroll, model_prob, yes_ask)

            opportunities.append({
                "contract_type": "below",
                "strike": cap,
                "ticker": strike.get("ticker", ""),
                "subtitle": strike.get("subtitle", f"Below ${cap:,.0f}"),
                "model_prob": model_prob,
                "market_prob": market_implied,
                "edge": edge,
                "ev_per_contract": ev,
                "kelly_fraction": kelly,
                "contracts": contracts,
                "yes_ask": yes_ask,
                "side": "YES" if edge > 0 else "NO",
                "action": f"BUY {'YES' if edge > 0 else 'NO'}",
                "horizon": horizon,
                "volume": strike.get("volume", 0),
                "open_interest": strike.get("open_interest", 0),
            })

    # Scan range contracts
    for rng in market_structure.get("ranges", []):
        floor = rng.get("floor_strike", 0)
        cap = rng.get("cap_strike", 0)
        yes_ask = rng.get("yes_ask", 0)
        if yes_ask <= 0 or yes_ask >= 1:
            continue

        model_prob = estimate_strike_probability(
            blended_prob, current_price, (floor, cap), "between", percentiles
        )
        market_implied = yes_ask
        edge = model_prob - market_implied

        if abs(edge) >= MIN_EDGE_THRESHOLD:
            ev = expected_value_yes(model_prob, yes_ask)
            kelly = kelly_fraction(model_prob, yes_ask)
            contracts = position_size(bankroll, model_prob, yes_ask)

            opportunities.append({
                "contract_type": "range",
                "strike": f"${floor:,.0f}-${cap:,.0f}",
                "ticker": rng.get("ticker", ""),
                "subtitle": rng.get("subtitle", f"${floor:,.0f} to ${cap:,.0f}"),
                "model_prob": model_prob,
                "market_prob": market_implied,
                "edge": edge,
                "ev_per_contract": ev,
                "kelly_fraction": kelly,
                "contracts": contracts,
                "yes_ask": yes_ask,
                "side": "YES" if edge > 0 else "NO",
                "action": f"BUY {'YES' if edge > 0 else 'NO'}",
                "horizon": horizon,
                "volume": rng.get("volume", 0),
                "open_interest": rng.get("open_interest", 0),
            })

    # Sort by absolute EV descending
    opportunities.sort(key=lambda x: abs(x["ev_per_contract"]), reverse=True)
    return opportunities


def scan_all_edges(
    asset: str = "BTC",
    bankroll: float = DEFAULT_BANKROLL,
    confidence_threshold: float = 0.58,
) -> dict:
    """
    Top-level orchestrator: fetch all data sources and scan both horizons.

    Returns:
        {
            "status": "ok"|"partial"|"error",
            "asset": str,
            "current_price": float,
            "timestamp": str,
            "1h": {
                "blended_signal": dict,
                "opportunities": list,
                "market_snapshot": dict,
            },
            "15min": {
                "blended_signal": dict,
                "opportunities": list,
                "market_snapshot": dict,
            },
            "synthdata_status": str,
            "ensemble_status": str,
        }
    """
    result = {
        "status": "error",
        "asset": asset.upper(),
        "timestamp": datetime.utcnow().isoformat(),
        "synthdata_status": "unavailable",
        "ensemble_status": "unavailable",
    }

    # 1. Fetch SynthData signals
    synth_signals = get_all_signals(asset)
    result["synthdata_status"] = synth_signals.get("status", "unavailable")

    # 2. Get local ensemble prediction (1h only)
    ensemble_result = None
    current_price = 0

    if HAS_ENSEMBLE and asset.upper() in KALSHI_TICKERS:
        try:
            btc_df = get_latest_candles("btc", limit=200) if asset.upper() != "BTC" else None
            ensemble_result = predict_with_threshold(
                asset.lower(),
                btc_df=btc_df,
                confidence_threshold=confidence_threshold,
            )
            current_price = ensemble_result.get("price", 0)
            result["ensemble_status"] = "ok"
        except Exception as e:
            print(f"Ensemble prediction error: {e}")
            result["ensemble_status"] = f"error: {e}"

    # 3. Get current price from SynthData if ensemble unavailable
    if current_price == 0:
        synth_1h = synth_signals.get("1h", {})
        pct_data = synth_1h.get("percentiles", {})
        current_price = pct_data.get("current_price", 0)

    result["current_price"] = current_price

    # 4. Scan each horizon
    for hz in ("1h", "15min", "5min"):
        hz_result = {
            "blended_signal": None,
            "opportunities": [],
            "market_snapshot": None,
        }

        # Get SynthData directional for this horizon
        synth_hz = synth_signals.get(hz, {})
        synth_dir = synth_hz.get("directional", {})
        synth_prob_up = synth_dir.get("prob_up", 0.5)

        # Blend signals
        if hz == "1h" and ensemble_result:
            ens_prob_up = ensemble_result.get("calibrated_prob_up", 0.5)
            blended = blend_predictions(synth_prob_up, ens_prob_up, horizon="1h")
        elif hz in ("15min", "5min"):
            ens_dir = ensemble_result.get("direction") if ensemble_result else None
            blended = blend_synthdata_only(synth_prob_up, ens_dir, horizon=hz)
        else:
            # No ensemble, use SynthData at full weight
            blended = blend_predictions(synth_prob_up, 0.5, horizon="1h")

        hz_result["blended_signal"] = blended

        # Get Kalshi market snapshot
        snapshot = get_market_snapshot(asset.upper(), timeframe=hz)
        hz_result["market_snapshot"] = snapshot

        # Scan for edge
        if snapshot and current_price > 0:
            # Get percentiles for CDF
            pct_data = synth_hz.get("percentiles", {})
            percentiles = pct_data.get("percentiles") if pct_data.get("status") in ("ok", "stale") else None

            opportunities = scan_edge(
                blended_signal=blended,
                market_structure=snapshot["structure"],
                current_price=current_price,
                horizon=hz,
                bankroll=bankroll,
                percentiles=percentiles,
            )
            hz_result["opportunities"] = opportunities

        result[hz] = hz_result

    # Set overall status
    has_1h = bool(result.get("1h", {}).get("blended_signal"))
    has_15m = bool(result.get("15min", {}).get("blended_signal"))
    has_5m = bool(result.get("5min", {}).get("blended_signal"))
    ok_count = sum([has_1h, has_15m, has_5m])
    if ok_count == 3:
        result["status"] = "ok"
    elif ok_count > 0:
        result["status"] = "partial"

    return result


if __name__ == "__main__":
    print("Scanning for edge opportunities...\n")
    edges = scan_all_edges("BTC")

    print(f"Status: {edges['status']}")
    print(f"Asset: {edges['asset']}")
    print(f"Price: ${edges['current_price']:,.2f}" if edges['current_price'] else "Price: N/A")
    print(f"SynthData: {edges['synthdata_status']}")
    print(f"Ensemble: {edges['ensemble_status']}")

    for hz in ("1h", "15min", "5min"):
        hz_data = edges.get(hz, {})
        blended = hz_data.get("blended_signal")
        opps = hz_data.get("opportunities", [])

        print(f"\n--- {hz.upper()} ---")
        if blended:
            print(f"  Blended: {blended['blended_direction']} "
                  f"({blended['blended_prob_up']:.3f}) "
                  f"Agreement: {blended['agreement'].value} "
                  f"Quality: {blended['quality'].value}")

        if opps:
            print(f"  Edge opportunities: {len(opps)}")
            for opp in opps[:5]:
                print(f"    {opp['subtitle']}: edge={opp['edge']:+.3f} "
                      f"EV=${opp['ev_per_contract']:.4f} "
                      f"action={opp['action']}")
        else:
            print("  No edge opportunities found")
