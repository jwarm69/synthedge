"""
Live prediction script for Kalshi betting.
Uses Fear & Greed Index and cross-asset features.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime

from data_fetch import get_latest_candles, SYMBOLS
from features import prepare_features, get_feature_columns

# Import EnsembleClassifier for unpickling
try:
    from ensemble import EnsembleClassifier
except ImportError:
    pass

MODELS_DIR = Path(__file__).parent.parent / "models"


def load_model(coin: str, use_ensemble: bool = True):
    """Load trained model for a coin.

    Args:
        coin: Coin name
        use_ensemble: Try to load ensemble model first (default True)
    """
    if use_ensemble:
        ensemble_path = MODELS_DIR / f"{coin}_ensemble.joblib"
        if ensemble_path.exists():
            return joblib.load(ensemble_path)

    model_path = MODELS_DIR / f"{coin}_xgb.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"No model for {coin}. Run train.py or ensemble.py first.")
    return joblib.load(model_path)


def load_model_meta(coin: str) -> dict:
    """Load model metadata (features used, etc.)."""
    meta_path = MODELS_DIR / f"{coin}_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    # Default for old models without metadata
    return {
        "features": get_feature_columns(include_fgi=False, include_cross_asset=False),
        "include_fgi": False,
        "include_cross_asset": False
    }


def predict_single(coin: str, btc_df: pd.DataFrame = None, verbose: bool = True) -> dict:
    """
    Make prediction for a single coin.

    Args:
        coin: Coin to predict
        btc_df: BTC data for cross-asset features (optional)
        verbose: Print output

    Returns:
        Dict with coin, probability, direction, confidence
    """
    # Load model and metadata
    model = load_model(coin)
    meta = load_model_meta(coin)

    # Get latest data (need enough for indicator warmup + FGI merge)
    df = get_latest_candles(coin, limit=200)

    # For altcoins, load BTC data for cross-asset features
    if meta.get("include_cross_asset", False) and coin.lower() != "btc":
        if btc_df is None:
            btc_df = get_latest_candles("btc", limit=200)
    else:
        btc_df = None

    # Engineer features
    df = prepare_features(
        df,
        lookahead=1,
        btc_df=btc_df,
        include_fgi=meta.get("include_fgi", True),
        include_onchain=meta.get("include_onchain", False)
    )

    # Get latest complete row
    latest = df.iloc[-1:]

    # Get features from metadata or default
    features = meta.get("features", get_feature_columns())

    # Filter to available features
    available_features = [f for f in features if f in latest.columns]

    X = latest[available_features]
    prob_up = model.predict_proba(X)[0, 1]

    direction = "UP" if prob_up > 0.5 else "DOWN"
    confidence = abs(prob_up - 0.5) * 2  # Scale to 0-1

    result = {
        "coin": coin.upper(),
        "symbol": f"{SYMBOLS[coin.lower()]}USD",
        "timestamp": latest.index[0],
        "current_price": latest["close"].values[0],
        "prob_up": prob_up,
        "prob_down": 1 - prob_up,
        "direction": direction,
        "confidence": confidence,
    }

    if verbose:
        print(f"\n{coin.upper()} ({SYMBOLS[coin.lower()]}USD)")
        print(f"  Current Price: ${result['current_price']:,.2f}")
        print(f"  Prediction:    {direction} ({prob_up*100:.1f}% up / {(1-prob_up)*100:.1f}% down)")
        print(f"  Confidence:    {confidence*100:.1f}%")

    return result


def predict_all(verbose: bool = True) -> list:
    """Make predictions for all coins."""
    results = []

    if verbose:
        print("=" * 60)
        print(f"CRYPTO PREDICTIONS - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 60)

    # Load BTC data once for cross-asset features
    try:
        btc_df = get_latest_candles("btc", limit=200)
    except Exception:
        btc_df = None

    for coin in SYMBOLS.keys():
        try:
            result = predict_single(coin, btc_df=btc_df, verbose=verbose)
            results.append(result)
        except FileNotFoundError as e:
            print(f"\n{coin.upper()}: No model found - run train.py first")
        except Exception as e:
            print(f"\n{coin.upper()}: Error - {e}")

    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY - Next Hour Predictions")
        print("=" * 60)
        for r in results:
            arrow = "🔼" if r["direction"] == "UP" else "🔽"
            print(f"{r['coin']:5} {arrow} {r['direction']:4} ({r['prob_up']*100:5.1f}% up) - Confidence: {r['confidence']*100:.0f}%")

    return results


def kalshi_recommendation(results: list, min_confidence: float = 0.6) -> list:
    """
    Generate Kalshi betting recommendations.

    Args:
        results: List of prediction results
        min_confidence: Minimum confidence to recommend a bet

    Returns:
        List of recommendations
    """
    print("\n" + "=" * 60)
    print("KALSHI RECOMMENDATIONS")
    print("=" * 60)
    print(f"(Only showing predictions with >{min_confidence*100:.0f}% confidence)")
    print()

    recommendations = []
    for r in results:
        if r["confidence"] >= min_confidence:
            rec = {
                "coin": r["coin"],
                "action": f"BET {r['direction']}",
                "probability": r["prob_up"] if r["direction"] == "UP" else r["prob_down"],
                "confidence": r["confidence"]
            }
            recommendations.append(rec)

            print(f"✅ {r['coin']}: BET {r['direction']} (Model: {rec['probability']*100:.0f}% confident)")
        else:
            print(f"⏸️  {r['coin']}: SKIP (low confidence: {r['confidence']*100:.0f}%)")

    if not recommendations:
        print("\nNo high-confidence bets right now. Consider waiting.")

    return recommendations


def kalshi_live_contracts(results: list, min_confidence: float = 0.55) -> list:
    """
    Fetch live Kalshi contracts and show specific betting opportunities.

    Args:
        results: List of prediction results
        min_confidence: Minimum confidence to show contracts

    Returns:
        List of contract recommendations
    """
    try:
        from kalshi_api import (
            get_next_hourly_event,
            get_markets_for_event,
            parse_market_structure,
            KALSHI_TICKERS
        )
    except ImportError:
        print("\nKalshi API module not available")
        return []

    print("\n" + "=" * 60)
    print("KALSHI LIVE CONTRACTS")
    print("=" * 60)

    contract_recs = []

    for r in results:
        coin = r["coin"]

        # Only BTC and ETH have Kalshi contracts
        if coin not in KALSHI_TICKERS:
            continue

        if r["confidence"] < min_confidence:
            continue

        print(f"\n{coin} - Predicted {r['direction']} ({r['prob_up']*100:.1f}% up)")
        print(f"Current Price: ${r['current_price']:,.2f}")

        # Get Kalshi contract data
        event = get_next_hourly_event(coin)
        if not event:
            print("  No upcoming contracts found")
            continue

        print(f"Next Settlement: {event.get('sub_title', '')}")

        markets = get_markets_for_event(event.get("event_ticker"))
        if not markets:
            print("  No markets available")
            continue

        structure = parse_market_structure(markets)

        # Find relevant contracts based on prediction
        current_price = r["current_price"]

        if r["direction"] == "UP":
            # Look for "above current price" contracts
            print(f"\n  Recommended Contracts (betting UP):")

            # Find range containing current price
            for rng in structure["ranges"]:
                floor = rng.get("floor_strike", 0)
                cap = rng.get("cap_strike", 0)
                if floor <= current_price < cap:
                    print(f"    Current range: {rng['subtitle']}")
                    print(f"      Ticker: {rng['ticker']}")
                    if rng["yes_ask"] > 0:
                        print(f"      Ask: ${rng['yes_ask']:.2f} | If win: $1.00 | ROI: {(1/rng['yes_ask']-1)*100:.0f}%")
                    break

            # Show above-current strike
            for strike in structure["above_strikes"]:
                if strike.get("floor_strike", 0) > current_price:
                    print(f"    Above ${strike['floor_strike']:,.0f}:")
                    print(f"      Ticker: {strike['ticker']}")
                    if strike["yes_ask"] > 0:
                        print(f"      Ask: ${strike['yes_ask']:.2f} | If win: $1.00 | ROI: {(1/strike['yes_ask']-1)*100:.0f}%")
                    contract_recs.append({
                        "coin": coin,
                        "direction": "UP",
                        "ticker": strike["ticker"],
                        "strike": strike.get("floor_strike"),
                        "ask": strike["yes_ask"],
                        "model_prob": r["prob_up"]
                    })
                    break

        else:  # DOWN
            print(f"\n  Recommended Contracts (betting DOWN):")

            # Show below-current strike
            for strike in structure["below_strikes"]:
                if strike.get("cap_strike", float("inf")) < current_price:
                    print(f"    Below ${strike['cap_strike']:,.0f}:")
                    print(f"      Ticker: {strike['ticker']}")
                    if strike["yes_ask"] > 0:
                        print(f"      Ask: ${strike['yes_ask']:.2f} | If win: $1.00 | ROI: {(1/strike['yes_ask']-1)*100:.0f}%")
                    contract_recs.append({
                        "coin": coin,
                        "direction": "DOWN",
                        "ticker": strike["ticker"],
                        "strike": strike.get("cap_strike"),
                        "ask": strike["yes_ask"],
                        "model_prob": r["prob_down"]
                    })
                    break

    if not contract_recs:
        print("\nNo actionable contracts found. Markets may be closed or low confidence.")

    return contract_recs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Crypto price predictions for Kalshi")
    parser.add_argument("--coin", type=str, default="all", help="Coin to predict (btc, eth, doge, zec, or 'all')")
    parser.add_argument("--min-confidence", type=float, default=0.55, help="Minimum confidence for recommendations")
    parser.add_argument("--kalshi", action="store_true", help="Show live Kalshi contracts")

    args = parser.parse_args()

    if args.coin == "all":
        results = predict_all()
        kalshi_recommendation(results, min_confidence=args.min_confidence)

        if args.kalshi:
            kalshi_live_contracts(results, min_confidence=args.min_confidence)
    else:
        result = predict_single(args.coin)
        if args.kalshi:
            kalshi_live_contracts([result], min_confidence=args.min_confidence)
