"""
Alert-first SynthEdge bot runner.

Scans for edge opportunities on a fixed interval, applies stricter filters,
notifies on high-quality setups, and records qualifying signals for paper P&L.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

from edge_detector import scan_all_edges
from pnl_tracker import record_signal


DEFAULT_CONFIG: dict[str, Any] = {
    "asset": "BTC",
    "interval_seconds": 60,
    "bankroll": 1000.0,
    "horizons": ["1h"],
    "max_alerts_per_cycle": 2,
    "cooldown_minutes": 15,
    "record_signals": True,
    "state_file": "data/tracking/bot_state.json",
    "filters": {
        "min_abs_edge": 0.04,
        "min_blended_confidence": 0.10,
        "min_agreement_score": 0.05,
        "min_yes_ask": 0.05,
        "max_yes_ask": 0.95,
        "max_spread": 0.25,
        "min_contract_volume": 0,
        "min_open_interest": 0,
        "min_total_event_volume": 0,
    },
    "notify": {
        "console": True,
        "telegram": {
            "enabled": False,
            "bot_token_env": "TELEGRAM_BOT_TOKEN",
            "chat_id_env": "TELEGRAM_CHAT_ID",
        },
        "webhook": {
            "enabled": False,
            "url_env": "SYNTHEDGE_WEBHOOK_URL",
        },
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _utc_now() -> datetime:
    """Return timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _normalize_horizon(horizon: str) -> str:
    h = str(horizon).strip().lower()
    if h in ("15m", "15min"):
        return "15min"
    if h in ("5m", "5min"):
        return "5min"
    return "1h"


def load_config(path: str) -> dict[str, Any]:
    cfg = deepcopy(DEFAULT_CONFIG)
    config_path = Path(path)
    if config_path.exists():
        with config_path.open() as f:
            user_cfg = json.load(f)
        cfg = _deep_merge(cfg, user_cfg)
    cfg["horizons"] = list(dict.fromkeys(_normalize_horizon(h) for h in cfg.get("horizons", ["1h"])))
    if not cfg["horizons"]:
        cfg["horizons"] = ["1h"]
    return cfg


def load_state(path: str) -> dict[str, str]:
    state_path = Path(path)
    if not state_path.exists():
        return {}
    try:
        with state_path.open() as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_state(path: str, state: dict[str, str]) -> None:
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("w") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def _flatten_markets(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    ticker_map: dict[str, dict[str, Any]] = {}
    structure = (snapshot or {}).get("structure", {})
    for bucket in ("above_strikes", "below_strikes", "ranges"):
        for market in structure.get(bucket, []):
            ticker = str(market.get("ticker", "")).strip()
            if ticker:
                ticker_map[ticker] = market
    return ticker_map


def _in_cooldown(state: dict[str, str], key: str, now: datetime, cooldown_minutes: int) -> bool:
    last_sent = state.get(key)
    if not last_sent:
        return False
    try:
        last_dt = datetime.fromisoformat(last_sent)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
    except Exception:
        return False
    return now - last_dt < timedelta(minutes=cooldown_minutes)


def _passes_filters(
    opportunity: dict[str, Any],
    blended: dict[str, Any],
    snapshot: dict[str, Any],
    market_lookup: dict[str, dict[str, Any]],
    filters: dict[str, Any],
) -> bool:
    edge = abs(float(opportunity.get("edge", 0.0)))
    if edge < float(filters.get("min_abs_edge", 0.0)):
        return False

    confidence = float(blended.get("blended_confidence", 0.0))
    if confidence < float(filters.get("min_blended_confidence", 0.0)):
        return False

    agreement_score = float(blended.get("agreement_score", 0.0))
    if agreement_score < float(filters.get("min_agreement_score", 0.0)):
        return False

    yes_ask = float(opportunity.get("yes_ask", 0.0))
    if yes_ask < float(filters.get("min_yes_ask", 0.0)):
        return False
    if yes_ask > float(filters.get("max_yes_ask", 1.0)):
        return False

    min_vol = float(filters.get("min_contract_volume", 0.0))
    min_oi = float(filters.get("min_open_interest", 0.0))
    if float(opportunity.get("volume", 0.0)) < min_vol:
        return False
    if float(opportunity.get("open_interest", 0.0)) < min_oi:
        return False

    min_event_vol = float(filters.get("min_total_event_volume", 0.0))
    total_event_vol = float((snapshot.get("liquidity", {}) or {}).get("total_volume", 0.0))
    if total_event_vol < min_event_vol:
        return False

    ticker = str(opportunity.get("ticker", ""))
    market = market_lookup.get(ticker, {})
    yes_bid = float(market.get("yes_bid", 0.0))
    max_spread = float(filters.get("max_spread", 1.0))
    if yes_bid > 0 and yes_ask > 0 and (yes_ask - yes_bid) > max_spread:
        return False

    return True


def _alert_key(asset: str, horizon: str, opportunity: dict[str, Any]) -> str:
    ticker = str(opportunity.get("ticker", "UNKNOWN"))
    side = str(opportunity.get("side", "NA"))
    return f"{asset}:{horizon}:{ticker}:{side}"


def _format_alert_line(asset: str, horizon: str, opportunity: dict[str, Any], blended: dict[str, Any]) -> str:
    agreement = blended.get("agreement")
    if hasattr(agreement, "value"):
        agreement = agreement.value
    return (
        f"{asset} {horizon} {opportunity.get('action', '')} {opportunity.get('ticker', '')} | "
        f"edge={opportunity.get('edge', 0.0):+.3f} "
        f"model={opportunity.get('model_prob', 0.0):.3f} "
        f"market={opportunity.get('market_prob', 0.0):.3f} "
        f"conf={blended.get('blended_confidence', 0.0):.3f} "
        f"agreement={agreement}"
    )


def _notify_console(enabled: bool, message: str) -> None:
    if enabled:
        print(message)


def _notify_telegram(notify_cfg: dict[str, Any], message: str) -> None:
    tg_cfg = notify_cfg.get("telegram", {})
    if not tg_cfg.get("enabled", False):
        return

    token = os.getenv(tg_cfg.get("bot_token_env", "TELEGRAM_BOT_TOKEN"), "")
    chat_id = os.getenv(tg_cfg.get("chat_id_env", "TELEGRAM_CHAT_ID"), "")
    if not token or not chat_id:
        print("Telegram notification skipped: missing bot token or chat id env var.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=10).raise_for_status()
    except Exception as exc:
        print(f"Telegram notification failed: {exc}")


def _notify_webhook(notify_cfg: dict[str, Any], payload: dict[str, Any]) -> None:
    wh_cfg = notify_cfg.get("webhook", {})
    if not wh_cfg.get("enabled", False):
        return

    url = os.getenv(wh_cfg.get("url_env", "SYNTHEDGE_WEBHOOK_URL"), "")
    if not url:
        print("Webhook notification skipped: missing webhook URL env var.")
        return

    try:
        requests.post(url, json=payload, timeout=10).raise_for_status()
    except Exception as exc:
        print(f"Webhook notification failed: {exc}")


def run_cycle(config: dict[str, Any], state: dict[str, str]) -> int:
    now = _utc_now()
    asset = str(config.get("asset", "BTC")).upper()
    bankroll = float(config.get("bankroll", 1000.0))
    cooldown_minutes = int(config.get("cooldown_minutes", 15))
    max_alerts = int(config.get("max_alerts_per_cycle", 2))
    filters = config.get("filters", {})
    notify_cfg = config.get("notify", {})
    record_enabled = bool(config.get("record_signals", True))

    edges = scan_all_edges(asset=asset, bankroll=bankroll)
    status = edges.get("status", "error")
    if status not in ("ok", "partial"):
        print(f"[{now.isoformat()}] Scan status={status}; skipping cycle.")
        return 0

    candidates: list[dict[str, Any]] = []
    for horizon in config.get("horizons", ["1h"]):
        hz_data = edges.get(horizon, {})
        blended = hz_data.get("blended_signal")
        opportunities = hz_data.get("opportunities", [])
        snapshot = hz_data.get("market_snapshot") or {}
        if not blended or not opportunities:
            continue

        market_lookup = _flatten_markets(snapshot)
        for opportunity in opportunities:
            if not _passes_filters(opportunity, blended, snapshot, market_lookup, filters):
                continue
            key = _alert_key(asset, horizon, opportunity)
            if _in_cooldown(state, key, now, cooldown_minutes):
                continue
            candidates.append(
                {
                    "score": abs(float(opportunity.get("edge", 0.0))),
                    "horizon": horizon,
                    "opportunity": opportunity,
                    "blended": blended,
                    "snapshot": snapshot,
                    "key": key,
                }
            )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = candidates[:max_alerts]

    sent_count = 0
    for item in selected:
        horizon = item["horizon"]
        opportunity = item["opportunity"]
        blended = item["blended"]
        snapshot = item["snapshot"]
        key = item["key"]

        alert_line = _format_alert_line(asset, horizon, opportunity, blended)
        msg = f"[{now.strftime('%Y-%m-%d %H:%M:%S')} UTC] {alert_line}"

        _notify_console(bool(notify_cfg.get("console", True)), msg)
        _notify_telegram(notify_cfg, msg)
        _notify_webhook(
            notify_cfg,
            {
                "timestamp": now.isoformat(),
                "asset": asset,
                "horizon": horizon,
                "message": alert_line,
                "opportunity": opportunity,
            },
        )

        if record_enabled:
            settlement_time = snapshot.get("settlement_time")
            if settlement_time and not isinstance(settlement_time, datetime):
                settlement_time = None
            try:
                record_signal(
                    blended_signal=blended,
                    edge_opportunity=opportunity,
                    asset=asset,
                    current_price=float(edges.get("current_price", 0.0)),
                    settlement_time=settlement_time,
                    bankroll=bankroll,
                )
            except Exception as exc:
                print(f"Record signal failed for {opportunity.get('ticker', '?')}: {exc}")

        state[key] = now.isoformat()
        sent_count += 1

    print(
        f"[{now.strftime('%Y-%m-%d %H:%M:%S')} UTC] "
        f"scan={status} candidates={len(candidates)} alerts_sent={sent_count}"
    )
    return sent_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SynthEdge alert bot loop.")
    parser.add_argument("--config", default="configs/bot.json", help="Path to JSON config file.")
    parser.add_argument("--once", action="store_true", help="Run one scan cycle and exit.")
    parser.add_argument("--cycles", type=int, default=0, help="Run N cycles then exit (0 = infinite).")
    parser.add_argument("--interval", type=int, default=None, help="Override interval seconds.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.interval is not None and args.interval > 0:
        config["interval_seconds"] = int(args.interval)
    interval_seconds = int(config.get("interval_seconds", 60))

    state_file = str(config.get("state_file", "data/tracking/bot_state.json"))
    state = load_state(state_file)

    cycle_count = 0
    print(
        f"SynthEdge Bot started | asset={config.get('asset')} "
        f"horizons={config.get('horizons')} interval={interval_seconds}s"
    )

    try:
        while True:
            run_cycle(config, state)
            save_state(state_file, state)
            cycle_count += 1

            if args.once:
                break
            if args.cycles > 0 and cycle_count >= args.cycles:
                break
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("Bot stopped by user.")


if __name__ == "__main__":
    main()
