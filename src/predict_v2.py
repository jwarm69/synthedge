"""
Enhanced prediction script v2 with:
- Threshold-based selective betting
- Confidence calibration
- Multi-horizon predictions
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
from typing import Optional

from data_fetch import get_latest_candles, SYMBOLS
from features import prepare_features, get_feature_columns

# Import for unpickling
try:
    from ensemble import EnsembleClassifier
except ImportError:
    pass

MODELS_DIR = Path(__file__).parent.parent / "models"
DEFAULT_CALIBRATION = "platt"


def load_model_v2(coin: str):
    """Load v2 trained model, fallback to v1."""
    v2_path = MODELS_DIR / f"{coin}_ensemble_v2.joblib"
    if v2_path.exists():
        return joblib.load(v2_path), "v2"

    v1_path = MODELS_DIR / f"{coin}_ensemble.joblib"
    if v1_path.exists():
        return joblib.load(v1_path), "v1"

    raise FileNotFoundError(f"No model for {coin}")


def load_meta_v2(coin: str) -> dict:
    """Load v2 model metadata."""
    v2_path = MODELS_DIR / f"{coin}_meta_v2.json"
    if v2_path.exists():
        with open(v2_path) as f:
            return json.load(f)

    v1_path = MODELS_DIR / f"{coin}_meta.json"
    if v1_path.exists():
        with open(v1_path) as f:
            return json.load(f)

    return {"features": get_feature_columns()}

def load_calibration(coin: str) -> dict:
    """Load stored calibration models if available."""
    calib_path = MODELS_DIR / f"{coin}_calibration.joblib"
    if calib_path.exists():
        try:
            return joblib.load(calib_path)
        except Exception:
            return {}
    return {}


def calibrate_probability(
    prob: float,
    calibration: str = "none",
    calibrators: Optional[dict] = None
) -> float:
    """
    Calibrate raw model probability.

    Many models output overconfident probabilities.
    This helps adjust them to more realistic values.
    """
    if calibration == "none":
        return prob

    if calibration == "platt":
        if calibrators and calibrators.get("platt") is not None:
            model = calibrators["platt"]
            return float(model.predict_proba([[prob]])[:, 1][0])
        return 0.5 + (prob - 0.5) * 0.7

    if calibration == "isotonic":
        if calibrators and calibrators.get("isotonic") is not None:
            model = calibrators["isotonic"]
            return float(model.predict([prob])[0])
        return 0.5 + (prob - 0.5) * 0.7

    if calibration == "shrink":
        # Shrink towards 0.5 (more conservative)
        return 0.5 + (prob - 0.5) * 0.7

    return prob


def calibrate_probabilities(
    probs: np.ndarray,
    calibration: str = "none",
    calibrators: Optional[dict] = None
) -> np.ndarray:
    """Vectorized calibration for arrays."""
    if calibration in ("platt", "isotonic") and calibrators:
        if calibration == "platt" and calibrators.get("platt") is not None:
            return calibrators["platt"].predict_proba(probs.reshape(-1, 1))[:, 1]
        if calibration == "isotonic" and calibrators.get("isotonic") is not None:
            return calibrators["isotonic"].predict(probs)

    if calibration == "shrink":
        return 0.5 + (probs - 0.5) * 0.7
    if calibration in ("platt", "isotonic"):
        return 0.5 + (probs - 0.5) * 0.7
    return probs


def predict_with_threshold(
    coin: str,
    btc_df: pd.DataFrame = None,
    confidence_threshold: float = 0.55,
    calibration: str = DEFAULT_CALIBRATION
) -> dict:
    """
    Make prediction with threshold-based recommendation.

    Only recommends betting when model confidence exceeds threshold.

    Args:
        coin: Coin to predict
        btc_df: BTC data for cross-asset features
        confidence_threshold: Min confidence to recommend (0.5-1.0)
        calibration: Probability calibration method

    Returns:
        Dict with prediction, confidence, and recommendation
    """
    model, version = load_model_v2(coin)
    meta = load_meta_v2(coin)

    # Get latest data
    df = get_latest_candles(coin, limit=200)

    # For altcoins, get BTC data
    include_cross_asset = coin.lower() != "btc"
    if include_cross_asset and btc_df is None:
        btc_df = get_latest_candles("btc", limit=200)

    # Prepare features
    df = prepare_features(
        df,
        lookahead=1,
        btc_df=btc_df if include_cross_asset else None,
        include_fgi=meta.get("include_fgi", True),
        include_onchain=meta.get("include_onchain", False),
        include_lags=meta.get("include_lags", True),
        include_multitimeframe=meta.get("include_multitimeframe", True),
        include_macro=meta.get("include_macro", False),
        include_flows=meta.get("include_flows", False),
        include_orderbook=meta.get("include_orderbook", False)
    )

    # Get features
    features = meta.get("features", get_feature_columns())
    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]

    latest = df.iloc[-1:].copy()

    # Fill missing features with 0 (neutral values)
    for f in missing:
        latest[f] = 0

    X = latest[features]

    # Get prediction
    prob_up = model.predict_proba(X)[0, 1]

    # Calibrate
    calibrators = load_calibration(coin) if calibration in ("platt", "isotonic") else None
    prob_up_cal = calibrate_probability(prob_up, calibration, calibrators=calibrators)

    # Determine direction and confidence
    direction = "UP" if prob_up_cal > 0.5 else "DOWN"
    confidence = abs(prob_up_cal - 0.5) * 2  # 0 to 1 scale

    # Recommendation based on threshold
    threshold_conf = max(0.0, min(1.0, (confidence_threshold - 0.5) * 2))
    meets_threshold = confidence >= threshold_conf

    if meets_threshold:
        recommendation = f"BET {direction}"
        action = "trade"
    else:
        recommendation = "SKIP - Low confidence"
        action = "skip"

    result = {
        "coin": coin.upper(),
        "timestamp": latest.index[0],
        "price": latest["close"].values[0],
        "model_version": version,
        "raw_prob_up": float(prob_up),
        "calibrated_prob_up": float(prob_up_cal),
        "direction": direction,
        "confidence": float(confidence),
        "confidence_pct": f"{confidence*100:.1f}%",
        "threshold": confidence_threshold,
        "threshold_confidence": float(threshold_conf),
        "meets_threshold": meets_threshold,
        "recommendation": recommendation,
        "action": action,
        "calibration": calibration,
    }

    return result


def backtest_threshold(
    coin: str,
    threshold: float = 0.55,
    days_back: int = 90,
    calibration: str = DEFAULT_CALIBRATION,
    assumed_yes_ask: float = 0.5,
    fee_per_contract: float = 0.0,
    bankroll: float = 1000.0,
    max_kelly_fraction: float = 0.25,
    position_sizing: str = "kelly"
) -> dict:
    """
    Backtest threshold-based strategy.

    Calculates:
    - Accuracy when model confidence > threshold
    - Coverage (% of predictions above threshold)
    - Expected value using assumed Kalshi pricing + fees

    Args:
        coin: Coin to backtest
        threshold: Confidence threshold
        days_back: Days of history to test
        assumed_yes_ask: Assumed yes-ask price for EV estimates
        fee_per_contract: Fee per contract
        bankroll: Starting bankroll for position sizing
        max_kelly_fraction: Cap on Kelly fraction
        position_sizing: "kelly" or "fixed"

    Returns:
        Dict with backtest metrics
    """
    from data_fetch import load_data

    model, version = load_model_v2(coin)
    meta = load_meta_v2(coin)

    # Load historical data
    df = load_data(coin)

    # For altcoins, get BTC
    btc_df = None
    if coin.lower() != "btc":
        btc_df = load_data("btc")

    # Prepare features
    df = prepare_features(
        df,
        lookahead=1,
        btc_df=btc_df,
        include_fgi=meta.get("include_fgi", True),
        include_onchain=meta.get("include_onchain", False),
        include_lags=meta.get("include_lags", True),
        include_multitimeframe=meta.get("include_multitimeframe", True),
        include_macro=meta.get("include_macro", False),
        include_flows=meta.get("include_flows", False),
        include_orderbook=meta.get("include_orderbook", False)
    )

    # Use last N days
    hours = days_back * 24
    df = df.tail(hours)

    features = meta.get("features", get_feature_columns())
    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]

    # Fill missing features with 0 (neutral values)
    df_filled = df.copy()
    for f in missing:
        df_filled[f] = 0

    X = df_filled[features]
    y = df["target"]

    # Get predictions
    probs_raw = model.predict_proba(X)[:, 1]
    calibrators = load_calibration(coin) if calibration in ("platt", "isotonic") else None
    probs = calibrate_probabilities(probs_raw, calibration, calibrators=calibrators)
    preds = (probs > 0.5).astype(int)

    # Calculate confidence
    confidences = np.abs(probs - 0.5) * 2
    threshold_conf = max(0.0, min(1.0, (threshold - 0.5) * 2))

    # Filter by threshold
    above_threshold = confidences >= threshold_conf

    # Metrics
    total_predictions = len(df)
    threshold_predictions = above_threshold.sum()
    coverage = threshold_predictions / total_predictions

    if threshold_predictions > 0:
        threshold_accuracy = (preds[above_threshold] == y[above_threshold]).mean()
    else:
        threshold_accuracy = 0

    overall_accuracy = (preds == y).mean()

    # EV and position sizing
    from kalshi_api import expected_value_yes, position_size

    prob_yes = np.where(preds == 1, probs, 1 - probs)
    ev_per_contract = np.array([
        expected_value_yes(p, assumed_yes_ask, fee_per_contract=fee_per_contract)
        for p in prob_yes
    ])

    expected_pnl_total = 0.0
    realized_pnl_total = 0.0
    contract_counts = []
    bankroll_now = bankroll

    for i, is_trade in enumerate(above_threshold):
        if not is_trade:
            continue

        if position_sizing == "kelly":
            contracts = position_size(
                bankroll_now,
                prob_yes[i],
                assumed_yes_ask,
                fee_per_contract=fee_per_contract,
                max_fraction=max_kelly_fraction
            )
        else:
            contracts = 1

        contract_counts.append(contracts)
        expected_pnl_total += ev_per_contract[i] * contracts

        if contracts == 0:
            continue

        cost = (assumed_yes_ask + fee_per_contract) * contracts
        win = preds[i] == y.iloc[i]
        pnl = (contracts - cost) if win else -cost
        realized_pnl_total += pnl
        bankroll_now += pnl

    result = {
        "coin": coin.upper(),
        "days_back": days_back,
        "threshold": threshold,
        "total_predictions": total_predictions,
        "threshold_predictions": threshold_predictions,
        "coverage": coverage,
        "coverage_pct": f"{coverage*100:.1f}%",
        "overall_accuracy": overall_accuracy,
        "overall_accuracy_pct": f"{overall_accuracy*100:.1f}%",
        "threshold_accuracy": threshold_accuracy,
        "threshold_accuracy_pct": f"{threshold_accuracy*100:.1f}%",
        "accuracy_improvement": threshold_accuracy - overall_accuracy,
        "improvement_pct": f"{(threshold_accuracy - overall_accuracy)*100:+.1f}%",
        "assumed_yes_ask": assumed_yes_ask,
        "fee_per_contract": fee_per_contract,
        "bankroll_start": bankroll,
        "bankroll_end": bankroll_now,
        "expected_pnl_total": expected_pnl_total,
        "realized_pnl_total": realized_pnl_total,
        "avg_ev_per_contract": float(ev_per_contract[above_threshold].mean()) if threshold_predictions else 0.0,
        "avg_contracts": float(np.mean(contract_counts)) if contract_counts else 0.0,
        "calibration": calibration,
        "position_sizing": position_sizing,
    }

    return result


def find_optimal_threshold(
    coin: str,
    days_back: int = 90,
    calibration: str = DEFAULT_CALIBRATION,
    optimize_metric: str = "ev",
    assumed_yes_ask: float = 0.5,
    fee_per_contract: float = 0.0,
    bankroll: float = 1000.0,
    max_kelly_fraction: float = 0.25,
    position_sizing: str = "kelly"
) -> dict:
    """Find the threshold that maximizes EV or accuracy while maintaining coverage."""
    results = []

    for threshold in np.arange(0.50, 0.70, 0.02):
        bt = backtest_threshold(
            coin,
            threshold,
            days_back,
            calibration=calibration,
            assumed_yes_ask=assumed_yes_ask,
            fee_per_contract=fee_per_contract,
            bankroll=bankroll,
            max_kelly_fraction=max_kelly_fraction,
            position_sizing=position_sizing
        )
        if optimize_metric == "ev":
            bt["score"] = bt["expected_pnl_total"]
        else:
            bt["score"] = bt["threshold_accuracy"] * np.sqrt(bt["coverage"])
        results.append(bt)

    # Find best threshold
    results_df = pd.DataFrame(results)
    best_idx = results_df["score"].idxmax()
    best = results_df.loc[best_idx].to_dict()

    return {
        "coin": coin,
        "optimal_threshold": best["threshold"],
        "optimal_accuracy": best["threshold_accuracy"],
        "optimal_coverage": best["coverage"],
        "optimal_score": best["score"],
        "optimize_metric": optimize_metric,
        "all_thresholds": results
    }


def predict_all_selective(
    confidence_threshold: float = 0.58,
    calibration: str = DEFAULT_CALIBRATION,
    verbose: bool = True
) -> list:
    """
    Make predictions for all coins with selective betting.

    Only returns recommendations for coins where confidence exceeds threshold.
    """
    results = []

    if verbose:
        print("=" * 65)
        print(f"SELECTIVE PREDICTIONS - Threshold: {confidence_threshold*100:.0f}%")
        print(f"Calibration: {calibration}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 65)

    # Load BTC once
    btc_df = get_latest_candles("btc", limit=200)

    for coin in SYMBOLS.keys():
        try:
            result = predict_with_threshold(
                coin,
                btc_df=btc_df,
                confidence_threshold=confidence_threshold,
                calibration=calibration
            )
            results.append(result)

            if verbose:
                emoji = "✅" if result["action"] == "trade" else "⏸️"
                arrow = "🔼" if result["direction"] == "UP" else "🔽"

                print(f"\n{emoji} {result['coin']} - ${result['price']:,.2f}")
                print(f"   {arrow} {result['direction']} | Confidence: {result['confidence_pct']}")
                print(f"   → {result['recommendation']}")

        except Exception as e:
            if verbose:
                print(f"\n❌ {coin.upper()}: Error - {e}")

    # Summary
    if verbose:
        trades = [r for r in results if r["action"] == "trade"]
        skips = [r for r in results if r["action"] == "skip"]

        print("\n" + "=" * 65)
        print("SUMMARY")
        print("=" * 65)
        print(f"Recommended trades: {len(trades)}")
        print(f"Skipped (low confidence): {len(skips)}")

        if trades:
            print("\n📊 ACTIONABLE PREDICTIONS:")
            for t in sorted(trades, key=lambda x: x["confidence"], reverse=True):
                arrow = "🔼" if t["direction"] == "UP" else "🔽"
                print(f"   {t['coin']}: {arrow} {t['direction']} ({t['confidence_pct']} confidence)")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Selective predictions")
    parser.add_argument("--coin", type=str, default="all", help="Coin to predict")
    parser.add_argument("--threshold", type=float, default=0.58, help="Confidence threshold")
    parser.add_argument("--calibration", type=str, default=DEFAULT_CALIBRATION, help="Calibration method")
    parser.add_argument("--backtest", action="store_true", help="Run backtest instead of live prediction")
    parser.add_argument("--days", type=int, default=90, help="Days for backtest")
    parser.add_argument("--optimize", action="store_true", help="Find optimal threshold")
    parser.add_argument("--optimize-metric", type=str, default="ev", help="Optimize metric (ev|accuracy)")
    parser.add_argument("--assumed-yes-ask", type=float, default=0.5, help="Assumed Kalshi yes-ask for backtests")
    parser.add_argument("--fee-per-contract", type=float, default=0.0, help="Fee per contract")
    parser.add_argument("--bankroll", type=float, default=1000.0, help="Bankroll for position sizing")
    parser.add_argument("--max-kelly", type=float, default=0.25, help="Max Kelly fraction")
    parser.add_argument("--position-sizing", type=str, default="kelly", help="Position sizing (kelly|fixed)")

    args = parser.parse_args()

    if args.optimize:
        print("Finding optimal thresholds...")
        for coin in SYMBOLS.keys():
            try:
                opt = find_optimal_threshold(
                    coin,
                    args.days,
                    calibration=args.calibration,
                    optimize_metric=args.optimize_metric,
                    assumed_yes_ask=args.assumed_yes_ask,
                    fee_per_contract=args.fee_per_contract,
                    bankroll=args.bankroll,
                    max_kelly_fraction=args.max_kelly,
                    position_sizing=args.position_sizing
                )
                print(f"\n{coin.upper()}:")
                print(f"  Optimal threshold: {opt['optimal_threshold']*100:.0f}%")
                print(f"  Accuracy at threshold: {opt['optimal_accuracy']*100:.1f}%")
                print(f"  Coverage: {opt['optimal_coverage']*100:.1f}%")
            except Exception as e:
                print(f"{coin.upper()}: Error - {e}")

    elif args.backtest:
        print(f"Backtesting with {args.threshold*100:.0f}% threshold over {args.days} days...\n")

        for coin in SYMBOLS.keys():
            try:
                bt = backtest_threshold(
                    coin,
                    args.threshold,
                    args.days,
                    calibration=args.calibration,
                    assumed_yes_ask=args.assumed_yes_ask,
                    fee_per_contract=args.fee_per_contract,
                    bankroll=args.bankroll,
                    max_kelly_fraction=args.max_kelly,
                    position_sizing=args.position_sizing
                )
                print(f"{bt['coin']}:")
                print(f"  Overall accuracy: {bt['overall_accuracy_pct']}")
                print(f"  Threshold accuracy: {bt['threshold_accuracy_pct']} ({bt['improvement_pct']})")
                print(f"  Coverage: {bt['coverage_pct']} ({bt['threshold_predictions']}/{bt['total_predictions']} trades)")
                print(f"  Expected PnL: ${bt['expected_pnl_total']:.4f}")
                print(f"  Realized PnL: ${bt['realized_pnl_total']:.4f}")
                print()
            except Exception as e:
                print(f"{coin.upper()}: Error - {e}\n")

    elif args.coin == "all":
        predict_all_selective(args.threshold, calibration=args.calibration)

    else:
        result = predict_with_threshold(
            args.coin,
            confidence_threshold=args.threshold,
            calibration=args.calibration
        )
        print(json.dumps(result, indent=2, default=str))
