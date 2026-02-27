"""
P&L tracker for SynthEdge signals.

Records each signal with its source attribution (SynthData, ensemble, blended),
agreement level, and edge. Evaluates settled trades and provides performance
breakdowns by horizon, agreement type, and signal source.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "tracking"
SIGNALS_FILE = DATA_DIR / "live_signals.csv"
SETTLED_FILE = DATA_DIR / "settled_trades.csv"


SIGNAL_FIELDS = [
    "signal_id",
    "logged_at",
    "asset",
    "horizon",
    "blended_direction",
    "blended_prob_up",
    "blended_confidence",
    "synthdata_prob_up",
    "ensemble_prob_up",
    "agreement",
    "agreement_score",
    "quality",
    "boost_applied",
    "current_price",
    "contract_type",
    "contract_ticker",
    "strike",
    "subtitle",
    "model_prob",
    "market_prob",
    "edge",
    "ev_per_contract",
    "kelly_fraction",
    "contracts",
    "yes_ask",
    "side",
    "action",
    "settlement_time",
    "bankroll",
]

SETTLED_FIELDS = SIGNAL_FIELDS + [
    "settlement_price",
    "resolved_yes",
    "won",
    "pnl",
    "settled_at",
]


def _ensure_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _append_row(path: Path, row: dict, fields: list) -> None:
    _ensure_dir()
    write_header = not path.exists()
    # Ensure all fields present
    for f in fields:
        if f not in row:
            row[f] = ""
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def _next_signal_id() -> str:
    """Generate sequential signal ID."""
    if not SIGNALS_FILE.exists():
        return "SE-0001"
    try:
        df = pd.read_csv(SIGNALS_FILE)
        if df.empty or "signal_id" not in df.columns:
            return "SE-0001"
        last_id = df["signal_id"].iloc[-1]
        num = int(str(last_id).split("-")[-1]) + 1
        return f"SE-{num:04d}"
    except Exception:
        return f"SE-{int(datetime.utcnow().timestamp()) % 10000:04d}"


def record_signal(
    blended_signal: dict,
    edge_opportunity: dict,
    asset: str = "BTC",
    current_price: float = 0,
    settlement_time: Optional[datetime] = None,
    bankroll: float = 1000.0,
) -> dict:
    """
    Record a signal with full source attribution.

    Args:
        blended_signal: output from signal_blender.blend_predictions()
        edge_opportunity: single opportunity from edge_detector.scan_edge()
        asset: "BTC" or "ETH"
        current_price: current asset price
        settlement_time: when the contract settles
        bankroll: bankroll used for sizing

    Returns:
        The recorded signal row dict
    """
    signal_id = _next_signal_id()

    row = {
        "signal_id": signal_id,
        "logged_at": datetime.utcnow().isoformat(),
        "asset": asset.upper(),
        "horizon": blended_signal.get("horizon", "1h"),
        "blended_direction": blended_signal.get("blended_direction", ""),
        "blended_prob_up": blended_signal.get("blended_prob_up", 0.5),
        "blended_confidence": blended_signal.get("blended_confidence", 0),
        "synthdata_prob_up": blended_signal.get("synthdata_prob_up", 0.5),
        "ensemble_prob_up": blended_signal.get("ensemble_prob_up", 0.5),
        "agreement": blended_signal.get("agreement", "").value if hasattr(blended_signal.get("agreement", ""), "value") else str(blended_signal.get("agreement", "")),
        "agreement_score": blended_signal.get("agreement_score", 0),
        "quality": blended_signal.get("quality", "").value if hasattr(blended_signal.get("quality", ""), "value") else str(blended_signal.get("quality", "")),
        "boost_applied": blended_signal.get("boost_applied", 0),
        "current_price": current_price,
        "contract_type": edge_opportunity.get("contract_type", ""),
        "contract_ticker": edge_opportunity.get("ticker", ""),
        "strike": edge_opportunity.get("strike", ""),
        "subtitle": edge_opportunity.get("subtitle", ""),
        "model_prob": edge_opportunity.get("model_prob", 0),
        "market_prob": edge_opportunity.get("market_prob", 0),
        "edge": edge_opportunity.get("edge", 0),
        "ev_per_contract": edge_opportunity.get("ev_per_contract", 0),
        "kelly_fraction": edge_opportunity.get("kelly_fraction", 0),
        "contracts": edge_opportunity.get("contracts", 0),
        "yes_ask": edge_opportunity.get("yes_ask", 0),
        "side": edge_opportunity.get("side", ""),
        "action": edge_opportunity.get("action", ""),
        "settlement_time": settlement_time.isoformat() if settlement_time else "",
        "bankroll": bankroll,
    }

    _append_row(SIGNALS_FILE, row, SIGNAL_FIELDS)
    return row


def evaluate_settled() -> dict:
    """
    Check which signals have resolved and compute P&L.

    Reads live_signals.csv, checks if settlement time has passed,
    determines outcome, and writes to settled_trades.csv.

    Returns:
        Summary dict with counts and P&L
    """
    if not SIGNALS_FILE.exists():
        return {"status": "no_signals", "settled": 0}

    try:
        signals = pd.read_csv(SIGNALS_FILE)
    except Exception:
        return {"status": "error_reading", "settled": 0}

    if signals.empty:
        return {"status": "no_signals", "settled": 0}

    # Load already-settled IDs to avoid duplicates
    settled_ids = set()
    if SETTLED_FILE.exists():
        try:
            existing = pd.read_csv(SETTLED_FILE)
            if "signal_id" in existing.columns:
                settled_ids = set(existing["signal_id"].values)
        except Exception:
            pass

    now = datetime.utcnow()
    new_settled = []

    for _, row in signals.iterrows():
        signal_id = row.get("signal_id", "")
        if signal_id in settled_ids:
            continue

        settlement_str = str(row.get("settlement_time", ""))
        if not settlement_str or settlement_str == "nan":
            continue

        try:
            settlement_time = datetime.fromisoformat(settlement_str.replace("Z", "+00:00"))
            if hasattr(settlement_time, "tzinfo") and settlement_time.tzinfo:
                settlement_time = settlement_time.replace(tzinfo=None)
        except Exception:
            continue

        if settlement_time > now:
            continue

        # Try to get settlement price
        settlement_price = _get_settlement_price(
            str(row.get("asset", "btc")).lower(),
            settlement_time
        )
        if settlement_price is None:
            continue

        # Determine outcome
        contract_type = row.get("contract_type", "")
        strike = row.get("strike", "")
        yes_ask = float(row.get("yes_ask", 0))

        resolved_yes = _resolve_contract(contract_type, strike, settlement_price)
        if resolved_yes is None:
            continue

        side = row.get("side", "YES")
        won = (resolved_yes and side == "YES") or (not resolved_yes and side == "NO")

        contracts = int(row.get("contracts", 0))
        if contracts > 0 and yes_ask > 0:
            cost = yes_ask * contracts
            if won:
                pnl = (1.0 - yes_ask) * contracts if side == "YES" else yes_ask * contracts
            else:
                pnl = -cost if side == "YES" else -(1.0 - yes_ask) * contracts
        else:
            pnl = 0.0

        settled_row = dict(row)
        settled_row.update({
            "settlement_price": settlement_price,
            "resolved_yes": resolved_yes,
            "won": won,
            "pnl": pnl,
            "settled_at": datetime.utcnow().isoformat(),
        })
        new_settled.append(settled_row)

    # Write new settled trades
    for s in new_settled:
        _append_row(SETTLED_FILE, s, SETTLED_FIELDS)

    return {
        "status": "ok",
        "newly_settled": len(new_settled),
        "total_settled": len(settled_ids) + len(new_settled),
    }


def _get_settlement_price(coin: str, settlement_time: datetime) -> Optional[float]:
    """Get the price at settlement time."""
    try:
        from data_fetch import load_data
        df = load_data(coin)
        df = df.sort_index()
        # Find closest price at or before settlement
        mask = df.index <= settlement_time
        if mask.any():
            return float(df.loc[mask].iloc[-1]["close"])
    except Exception:
        pass
    return None


def _resolve_contract(contract_type: str, strike, settlement_price: float) -> Optional[bool]:
    """Determine if a contract resolved YES."""
    try:
        if contract_type == "above":
            return settlement_price > float(strike)
        elif contract_type == "below":
            return settlement_price < float(strike)
        elif contract_type == "range":
            # Strike is "floor-cap" string
            parts = str(strike).replace("$", "").replace(",", "").split("-")
            if len(parts) == 2:
                floor_val = float(parts[0])
                cap_val = float(parts[1])
                return floor_val < settlement_price < cap_val
    except (ValueError, TypeError):
        pass
    return None


def get_performance_summary() -> dict:
    """
    Get comprehensive performance summary with breakdowns.

    Returns:
        {
            "total_signals": int,
            "total_settled": int,
            "pending": int,
            "win_rate": float,
            "total_pnl": float,
            "avg_pnl": float,
            "by_horizon": {"1h": {...}, "15min": {...}},
            "by_agreement": {"STRONG_AGREE": {...}, ...},
            "source_attribution": {"synthdata_alpha": float, "ensemble_alpha": float},
        }
    """
    result = {
        "total_signals": 0,
        "total_settled": 0,
        "pending": 0,
        "win_rate": 0.0,
        "total_pnl": 0.0,
        "avg_pnl": 0.0,
        "by_horizon": {},
        "by_agreement": {},
    }

    # Count total signals
    if SIGNALS_FILE.exists():
        try:
            signals = pd.read_csv(SIGNALS_FILE)
            result["total_signals"] = len(signals)
        except Exception:
            pass

    # Load settled trades
    if not SETTLED_FILE.exists():
        result["pending"] = result["total_signals"]
        return result

    try:
        df = pd.read_csv(SETTLED_FILE)
    except Exception:
        result["pending"] = result["total_signals"]
        return result

    if df.empty:
        result["pending"] = result["total_signals"]
        return result

    result["total_settled"] = len(df)
    result["pending"] = result["total_signals"] - result["total_settled"]

    # Overall stats
    if "won" in df.columns:
        df["won"] = df["won"].astype(bool)
        result["win_rate"] = float(df["won"].mean())
    if "pnl" in df.columns:
        result["total_pnl"] = float(df["pnl"].sum())
        result["avg_pnl"] = float(df["pnl"].mean())

    # By horizon
    if "horizon" in df.columns:
        for hz, group in df.groupby("horizon"):
            hz_stats = {
                "count": len(group),
                "win_rate": float(group["won"].mean()) if "won" in group.columns else 0,
                "total_pnl": float(group["pnl"].sum()) if "pnl" in group.columns else 0,
                "avg_pnl": float(group["pnl"].mean()) if "pnl" in group.columns else 0,
            }
            result["by_horizon"][str(hz)] = hz_stats

    # By agreement
    if "agreement" in df.columns:
        for agreement, group in df.groupby("agreement"):
            ag_stats = {
                "count": len(group),
                "win_rate": float(group["won"].mean()) if "won" in group.columns else 0,
                "total_pnl": float(group["pnl"].sum()) if "pnl" in group.columns else 0,
                "avg_pnl": float(group["pnl"].mean()) if "pnl" in group.columns else 0,
                "avg_edge": float(group["edge"].mean()) if "edge" in group.columns else 0,
            }
            result["by_agreement"][str(agreement)] = ag_stats

    return result


def get_signal_history(limit: int = 50) -> pd.DataFrame:
    """Get recent signal history for display."""
    if not SIGNALS_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(SIGNALS_FILE)
        return df.tail(limit)
    except Exception:
        return pd.DataFrame()


def get_settled_history(limit: int = 50) -> pd.DataFrame:
    """Get recent settled trade history for display."""
    if not SETTLED_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(SETTLED_FILE)
        return df.tail(limit)
    except Exception:
        return pd.DataFrame()


if __name__ == "__main__":
    print("=== P&L Tracker ===\n")

    # Evaluate any pending signals
    eval_result = evaluate_settled()
    print(f"Evaluation: {eval_result}")

    # Show summary
    summary = get_performance_summary()
    print(f"\nTotal signals: {summary['total_signals']}")
    print(f"Settled: {summary['total_settled']}")
    print(f"Pending: {summary['pending']}")
    print(f"Win rate: {summary['win_rate']*100:.1f}%")
    print(f"Total PnL: ${summary['total_pnl']:.2f}")

    if summary["by_horizon"]:
        print("\nBy Horizon:")
        for hz, stats in summary["by_horizon"].items():
            print(f"  {hz}: {stats['count']} trades, "
                  f"WR={stats['win_rate']*100:.1f}%, "
                  f"PnL=${stats['total_pnl']:.2f}")

    if summary["by_agreement"]:
        print("\nBy Agreement:")
        for ag, stats in summary["by_agreement"].items():
            print(f"  {ag}: {stats['count']} trades, "
                  f"WR={stats['win_rate']*100:.1f}%, "
                  f"PnL=${stats['total_pnl']:.2f}")
