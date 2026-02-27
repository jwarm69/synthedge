"""
Kalshi EV logging and evaluation workflow.

Records live signals with Kalshi yes-ask prices and evaluates realized PnL
after settlement using historical price data.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

from data_fetch import get_latest_candles, load_data
from kalshi_api import (
    get_next_hourly_event,
    get_markets_for_event,
    parse_market_structure,
    KALSHI_TICKERS,
    expected_value_yes,
    position_size,
    DEFAULT_FEE_PER_CONTRACT,
    DEFAULT_MAX_KELLY_FRACTION
)
from predict_v2 import predict_with_threshold, DEFAULT_CALIBRATION

DATA_DIR = Path(__file__).parent.parent / "data" / "kalshi"
SIGNALS_FILE = DATA_DIR / "signals.csv"
RESULTS_FILE = DATA_DIR / "signal_results.csv"


def _parse_strike_date(date_str: str) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.replace(tzinfo=None)
    except ValueError:
        return None


def _select_contract(structure: dict, current_price: float, direction: str) -> Optional[Dict[str, Any]]:
    if direction == "UP":
        for strike in structure.get("above_strikes", []):
            if strike.get("floor_strike", 0) > current_price and strike.get("yes_ask", 0) > 0:
                return {
                    "type": "above",
                    "strike": strike.get("floor_strike"),
                    "ticker": strike.get("ticker"),
                    "yes_ask": strike.get("yes_ask", 0),
                    "yes_bid": strike.get("yes_bid", 0)
                }
    elif direction == "DOWN":
        for strike in structure.get("below_strikes", []):
            if strike.get("cap_strike", float("inf")) < current_price and strike.get("yes_ask", 0) > 0:
                return {
                    "type": "below",
                    "strike": strike.get("cap_strike"),
                    "ticker": strike.get("ticker"),
                    "yes_ask": strike.get("yes_ask", 0),
                    "yes_bid": strike.get("yes_bid", 0)
                }
    return None


def _append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def record_kalshi_signal(
    coin: str,
    confidence_threshold: float = 0.58,
    calibration: str = DEFAULT_CALIBRATION,
    fee_per_contract: float = DEFAULT_FEE_PER_CONTRACT,
    bankroll: float = 1000.0,
    max_kelly_fraction: float = DEFAULT_MAX_KELLY_FRACTION
) -> dict:
    """
    Record a live signal with Kalshi market pricing.

    Returns:
        Dict with signal info and EV.
    """
    coin = coin.lower()
    if coin.upper() not in KALSHI_TICKERS:
        return {"coin": coin, "status": "unsupported"}

    # Model prediction
    btc_df = get_latest_candles("btc", limit=200) if coin != "btc" else None
    prediction = predict_with_threshold(
        coin,
        btc_df=btc_df,
        confidence_threshold=confidence_threshold,
        calibration=calibration
    )

    event = get_next_hourly_event(coin.upper())
    if not event:
        return {"coin": coin, "status": "no_event"}

    markets = get_markets_for_event(event.get("event_ticker"))
    if not markets:
        return {"coin": coin, "status": "no_markets"}

    structure = parse_market_structure(markets)
    contract = _select_contract(structure, prediction["price"], prediction["direction"])

    prob_up = prediction["calibrated_prob_up"]
    prob_yes = prob_up if prediction["direction"] == "UP" else 1 - prob_up
    yes_ask = contract["yes_ask"] if contract else 0
    ev_per_contract = (
        expected_value_yes(prob_yes, yes_ask, fee_per_contract=fee_per_contract)
        if yes_ask else 0
    )
    contracts = (
        position_size(
            bankroll,
            prob_yes,
            yes_ask,
            fee_per_contract=fee_per_contract,
            max_fraction=max_kelly_fraction
        )
        if yes_ask else 0
    )

    row = {
        "logged_at": datetime.utcnow().isoformat(),
        "coin": coin,
        "direction": prediction["direction"],
        "action": prediction["action"],
        "confidence": prediction["confidence"],
        "confidence_threshold": confidence_threshold,
        "calibration": calibration,
        "current_price": prediction["price"],
        "model_prob_up": prediction["raw_prob_up"],
        "model_prob_up_cal": prob_up,
        "event_ticker": event.get("event_ticker"),
        "settlement_time": event.get("strike_date"),
        "contract_type": contract["type"] if contract else "",
        "contract_ticker": contract["ticker"] if contract else "",
        "strike": contract["strike"] if contract else "",
        "yes_ask": yes_ask,
        "yes_bid": contract["yes_bid"] if contract else 0,
        "fee_per_contract": fee_per_contract,
        "bankroll": bankroll,
        "max_kelly_fraction": max_kelly_fraction,
        "contracts": contracts,
        "ev_per_contract": ev_per_contract,
        "expected_pnl": ev_per_contract * contracts
    }

    _append_row(SIGNALS_FILE, row)
    row["status"] = "recorded"
    return row


def record_all_signals(
    confidence_threshold: float = 0.58,
    calibration: str = DEFAULT_CALIBRATION,
    fee_per_contract: float = DEFAULT_FEE_PER_CONTRACT,
    bankroll: float = 1000.0,
    max_kelly_fraction: float = DEFAULT_MAX_KELLY_FRACTION
) -> list:
    """Record signals for all Kalshi-supported coins."""
    results = []
    for coin in KALSHI_TICKERS.keys():
        results.append(record_kalshi_signal(
            coin.lower(),
            confidence_threshold=confidence_threshold,
            calibration=calibration,
            fee_per_contract=fee_per_contract,
            bankroll=bankroll,
            max_kelly_fraction=max_kelly_fraction
        ))
    return results


def _get_settlement_price(coin: str, settlement_time: datetime) -> Optional[float]:
    df = load_data(coin)
    df = df.sort_index()
    if settlement_time not in df.index:
        df = df[df.index <= settlement_time]
        if df.empty:
            return None
        return float(df.iloc[-1]["close"])
    return float(df.loc[settlement_time]["close"])


def evaluate_signals() -> dict:
    """
    Evaluate logged signals and write results to RESULTS_FILE.
    """
    if not SIGNALS_FILE.exists():
        return {"status": "no_signals"}

    signals = pd.read_csv(SIGNALS_FILE)
    if signals.empty:
        return {"status": "no_signals"}

    results = []
    now = datetime.utcnow()

    for _, row in signals.iterrows():
        settlement_time = _parse_strike_date(str(row.get("settlement_time", "")))
        if not settlement_time or settlement_time > now:
            continue

        coin = str(row["coin"]).lower()
        contract_type = row.get("contract_type")
        strike = float(row["strike"]) if str(row.get("strike", "")).strip() else None

        if contract_type not in ("above", "below") or strike is None:
            continue

        settlement_price = _get_settlement_price(coin, settlement_time)
        if settlement_price is None:
            continue

        yes_ask = float(row.get("yes_ask", 0))
        fee_per_contract = float(row.get("fee_per_contract", 0))
        contracts = int(row.get("contracts", 0))
        if yes_ask <= 0:
            continue

        if contract_type == "above":
            win = settlement_price > strike
        else:
            win = settlement_price < strike

        if contracts <= 0:
            pnl = 0.0
        else:
            cost = (yes_ask + fee_per_contract) * contracts
            pnl = (1 - (yes_ask + fee_per_contract)) * contracts if win else -cost

        result_row = dict(row)
        result_row.update({
            "settled": True,
            "settlement_price": settlement_price,
            "won": win,
            "pnl": pnl
        })
        results.append(result_row)

    if results:
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)

    if not results:
        return {"status": "no_settled_signals"}

    df = pd.DataFrame(results)
    return {
        "status": "ok",
        "settled": len(df),
        "win_rate": float((df["won"]).mean()),
        "total_pnl": float(df["pnl"].sum()),
        "avg_pnl": float(df["pnl"].mean()),
        "avg_ev": float(df["ev_per_contract"].mean())
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kalshi EV logging and evaluation")
    parser.add_argument("--record", action="store_true", help="Record a new signal")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate settled signals")
    parser.add_argument("--coin", type=str, default="all", help="Coin to record (btc or eth)")
    parser.add_argument("--threshold", type=float, default=0.58, help="Confidence threshold")
    parser.add_argument("--calibration", type=str, default=DEFAULT_CALIBRATION, help="Calibration method")
    parser.add_argument("--fee-per-contract", type=float, default=DEFAULT_FEE_PER_CONTRACT, help="Fee per contract")
    parser.add_argument("--bankroll", type=float, default=1000.0, help="Bankroll for sizing")
    parser.add_argument("--max-kelly", type=float, default=DEFAULT_MAX_KELLY_FRACTION, help="Max Kelly fraction")

    args = parser.parse_args()

    if args.record:
        if args.coin == "all":
            records = record_all_signals(
                confidence_threshold=args.threshold,
                calibration=args.calibration,
                fee_per_contract=args.fee_per_contract,
                bankroll=args.bankroll,
                max_kelly_fraction=args.max_kelly
            )
            for r in records:
                print(r)
        else:
            print(record_kalshi_signal(
                args.coin,
                confidence_threshold=args.threshold,
                calibration=args.calibration,
                fee_per_contract=args.fee_per_contract,
                bankroll=args.bankroll,
                max_kelly_fraction=args.max_kelly
            ))

    if args.evaluate:
        print(evaluate_signals())
