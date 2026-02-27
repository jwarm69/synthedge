"""
Backtest simulator for SynthEdge signals.

Replays recorded signals with configurable parameters (min_edge,
agreement filter, kelly cap) to compute hypothetical P&L.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data" / "tracking"
SIGNALS_FILE = DATA_DIR / "live_signals.csv"
SETTLED_FILE = DATA_DIR / "settled_trades.csv"


def _resolve_contract(contract_type: str, strike, settlement_price: float) -> Optional[bool]:
    """Determine if a contract resolved YES."""
    try:
        if contract_type == "above":
            return settlement_price > float(strike)
        elif contract_type == "below":
            return settlement_price < float(strike)
        elif contract_type == "range":
            parts = str(strike).replace("$", "").replace(",", "").split("-")
            if len(parts) == 2:
                floor_val = float(parts[0])
                cap_val = float(parts[1])
                return floor_val < settlement_price < cap_val
    except (ValueError, TypeError):
        pass
    return None


def run_backtest(
    min_edge: float = 0.05,
    agreement_filter: str = "ALL",
    kelly_cap: float = 0.25,
    bankroll: float = 1000.0,
) -> dict:
    """
    Run backtest on recorded signal history with configurable parameters.

    Args:
        min_edge: Minimum absolute edge to take a trade (0.03 - 0.15)
        agreement_filter: "ALL", "AGREE+", or "STRONG_AGREE"
            ALL = take all signals
            AGREE+ = only AGREE or STRONG_AGREE
            STRONG_AGREE = only STRONG_AGREE
        kelly_cap: Maximum Kelly fraction to risk per trade (0.05 - 0.50)
        bankroll: Starting bankroll

    Returns:
        {
            "params": {"min_edge": float, "agreement_filter": str, "kelly_cap": float},
            "win_rate": float,
            "total_pnl": float,
            "sharpe": float,
            "max_drawdown": float,
            "n_trades": int,
            "n_filtered": int,
            "equity_curve": [float, ...],
            "timestamps": [str, ...],
            "trade_log": [dict, ...],
        }
    """
    result = {
        "params": {
            "min_edge": min_edge,
            "agreement_filter": agreement_filter,
            "kelly_cap": kelly_cap,
        },
        "win_rate": 0.0,
        "total_pnl": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "n_trades": 0,
        "n_filtered": 0,
        "equity_curve": [],
        "timestamps": [],
        "trade_log": [],
    }

    # Try settled trades first, fall back to signals
    df = _load_settled_data()
    if df is None or df.empty:
        return result

    # Ensure numeric columns
    numeric_cols = ["edge", "kelly_fraction", "yes_ask", "settlement_price", "pnl",
                    "blended_prob_up", "model_prob", "market_prob"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Apply filters
    total_signals = len(df)

    # Edge filter
    if "edge" in df.columns:
        df = df[df["edge"].abs() >= min_edge]

    # Agreement filter
    agreement_accept = {
        "ALL": ["STRONG_AGREE", "AGREE", "NEUTRAL", "DISAGREE", "STRONG_DISAGREE"],
        "AGREE+": ["STRONG_AGREE", "AGREE"],
        "STRONG_AGREE": ["STRONG_AGREE"],
    }
    accepted = agreement_accept.get(agreement_filter, agreement_accept["ALL"])
    if "agreement" in df.columns:
        df = df[df["agreement"].isin(accepted)]

    result["n_filtered"] = total_signals - len(df)

    if df.empty:
        return result

    # Replay trades
    current_bankroll = bankroll
    peak_bankroll = bankroll
    max_drawdown = 0.0
    equity_curve = [bankroll]
    timestamps = []
    trade_log = []
    pnl_list = []
    wins = 0

    for _, row in df.iterrows():
        yes_ask = float(row.get("yes_ask", 0))
        settlement_price = float(row.get("settlement_price", 0))
        contract_type = str(row.get("contract_type", ""))
        strike = row.get("strike", "")
        side = str(row.get("side", "YES"))
        kelly = min(float(row.get("kelly_fraction", 0)), kelly_cap)

        if yes_ask <= 0 or yes_ask >= 1 or kelly <= 0:
            continue

        # Resolve outcome
        resolved_yes = _resolve_contract(contract_type, strike, settlement_price)
        if resolved_yes is None:
            continue

        # Position size using capped Kelly
        risk_amount = current_bankroll * kelly
        n_contracts = max(1, int(risk_amount / yes_ask))

        won = (resolved_yes and side == "YES") or (not resolved_yes and side == "NO")

        if won:
            wins += 1
            trade_pnl = (1.0 - yes_ask) * n_contracts if side == "YES" else yes_ask * n_contracts
        else:
            trade_pnl = -yes_ask * n_contracts if side == "YES" else -(1.0 - yes_ask) * n_contracts

        pnl_list.append(trade_pnl)
        current_bankroll += trade_pnl
        peak_bankroll = max(peak_bankroll, current_bankroll)
        drawdown = (peak_bankroll - current_bankroll) / peak_bankroll if peak_bankroll > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)

        equity_curve.append(current_bankroll)
        timestamps.append(str(row.get("settled_at", "")))

        trade_log.append({
            "signal_id": str(row.get("signal_id", "")),
            "won": won,
            "pnl": trade_pnl,
            "contracts": n_contracts,
            "edge": float(row.get("edge", 0)),
            "agreement": str(row.get("agreement", "")),
            "bankroll_after": current_bankroll,
        })

    n_trades = len(pnl_list)
    result["n_trades"] = n_trades
    result["equity_curve"] = equity_curve
    result["timestamps"] = timestamps
    result["trade_log"] = trade_log
    result["max_drawdown"] = max_drawdown

    if n_trades > 0:
        result["win_rate"] = wins / n_trades
        result["total_pnl"] = sum(pnl_list)

        # Sharpe ratio
        if n_trades > 1:
            pnl_arr = np.array(pnl_list)
            mean_ret = pnl_arr.mean()
            std_ret = pnl_arr.std()
            result["sharpe"] = (mean_ret / std_ret * (252 ** 0.5)) if std_ret > 0 else 0

    return result


def _load_settled_data() -> Optional[pd.DataFrame]:
    """Load settled trades data."""
    if SETTLED_FILE.exists():
        try:
            df = pd.read_csv(SETTLED_FILE)
            if not df.empty and "settlement_price" in df.columns:
                return df
        except Exception:
            pass
    return None


def run_parameter_sweep(
    edge_range: tuple = (0.03, 0.05, 0.08, 0.10, 0.15),
    agreement_options: tuple = ("ALL", "AGREE+", "STRONG_AGREE"),
    kelly_range: tuple = (0.10, 0.15, 0.25, 0.35, 0.50),
) -> list:
    """
    Run backtest across all parameter combinations.

    Returns list of result dicts sorted by Sharpe ratio.
    """
    results = []
    for edge in edge_range:
        for agreement in agreement_options:
            for kelly in kelly_range:
                bt = run_backtest(
                    min_edge=edge,
                    agreement_filter=agreement,
                    kelly_cap=kelly,
                )
                if bt["n_trades"] > 0:
                    results.append(bt)

    results.sort(key=lambda x: x.get("sharpe", 0), reverse=True)
    return results


if __name__ == "__main__":
    print("=== SynthEdge Backtester ===\n")

    bt = run_backtest()
    print(f"Params: edge>{bt['params']['min_edge']}, "
          f"agreement={bt['params']['agreement_filter']}, "
          f"kelly_cap={bt['params']['kelly_cap']}")
    print(f"Trades: {bt['n_trades']} (filtered: {bt['n_filtered']})")
    print(f"Win Rate: {bt['win_rate']*100:.1f}%")
    print(f"Total PnL: ${bt['total_pnl']:.2f}")
    print(f"Sharpe: {bt['sharpe']:.2f}")
    print(f"Max Drawdown: {bt['max_drawdown']*100:.1f}%")

    if bt["n_trades"] > 0:
        print(f"\nEquity: ${bt['equity_curve'][0]:.0f} -> ${bt['equity_curve'][-1]:.0f}")

    print("\n--- Parameter Sweep ---")
    sweep = run_parameter_sweep()
    for r in sweep[:5]:
        p = r["params"]
        print(f"  edge>{p['min_edge']:.2f} | {p['agreement_filter']:14s} | "
              f"kelly={p['kelly_cap']:.2f} | "
              f"WR={r['win_rate']*100:.1f}% | "
              f"PnL=${r['total_pnl']:.2f} | "
              f"Sharpe={r['sharpe']:.2f} | "
              f"n={r['n_trades']}")
