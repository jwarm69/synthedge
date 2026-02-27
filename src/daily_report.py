"""
Generate a daily performance report for SynthEdge alerting.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pnl_tracker import (
    DATA_DIR,
    evaluate_settled,
    get_performance_summary,
    get_settled_history,
    get_signal_history,
    get_source_comparison,
)


def _parse_date(date_str: str | None) -> str:
    if not date_str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")
    datetime.strptime(date_str, "%Y-%m-%d")
    return date_str


def _to_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def _safe_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _top_loss_patterns(settled_today: pd.DataFrame, top_n: int = 3) -> list[tuple[str, int]]:
    if settled_today.empty or "won" not in settled_today.columns:
        return []

    losers = settled_today[settled_today["won"].astype(str).str.lower().isin(["false", "0"])]
    if losers.empty:
        return []

    patterns: list[tuple[str, int]] = []
    for cols in (["horizon", "agreement"], ["contract_type", "side"], ["horizon", "contract_type"]):
        available = [c for c in cols if c in losers.columns]
        if not available:
            continue
        grouped = (
            losers.groupby(available)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        for _, row in grouped.head(top_n).iterrows():
            label = " | ".join(f"{c}={row[c]}" for c in available)
            patterns.append((label, int(row["count"])))

    # Deduplicate labels while preserving order
    seen = set()
    unique_patterns = []
    for label, count in patterns:
        if label in seen:
            continue
        seen.add(label)
        unique_patterns.append((label, count))
    return unique_patterns[:top_n]


def generate_report(report_date: str) -> str:
    settlement_eval = evaluate_settled()
    summary = get_performance_summary()
    source_cmp = get_source_comparison()

    signals = get_signal_history(limit=100000)
    settled = get_settled_history(limit=100000)

    if not signals.empty and "logged_at" in signals.columns:
        signals = signals.copy()
        signals["report_date"] = _to_date(signals["logged_at"])
        signals_today = signals[signals["report_date"] == report_date]
    else:
        signals_today = pd.DataFrame()

    if not settled.empty and "settled_at" in settled.columns:
        settled = settled.copy()
        settled["report_date"] = _to_date(settled["settled_at"])
        settled_today = settled[settled["report_date"] == report_date]
    else:
        settled_today = pd.DataFrame()

    alerts_today = len(signals_today)
    if alerts_today:
        edge_series = pd.to_numeric(signals_today.get("edge", pd.Series(dtype=float)), errors="coerce").abs()
        avg_edge_today = float(edge_series.mean()) if not edge_series.empty else 0.0
        if pd.isna(avg_edge_today):
            avg_edge_today = 0.0
    else:
        avg_edge_today = 0.0

    settled_count = len(settled_today)
    if settled_count and "won" in settled_today.columns:
        won_series = settled_today["won"].astype(str).str.lower().isin(["true", "1"])
        daily_win_rate = float(won_series.mean())
    else:
        daily_win_rate = 0.0

    if settled_count and "pnl" in settled_today.columns:
        daily_pnl = float(pd.to_numeric(settled_today["pnl"], errors="coerce").fillna(0).sum())
    else:
        daily_pnl = 0.0

    top_patterns = _top_loss_patterns(settled_today)

    lines = [
        f"# SynthEdge Daily Report - {report_date} (UTC)",
        "",
        f"Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Settlement evaluation result: {settlement_eval}",
        "",
        "## Daily Snapshot",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Alerts logged ({report_date}) | {alerts_today} |",
        f"| Avg abs edge ({report_date}) | {avg_edge_today:.3f} |",
        f"| Trades settled ({report_date}) | {settled_count} |",
        f"| Win rate ({report_date}) | {_safe_pct(daily_win_rate)} |",
        f"| PnL ({report_date}) | ${daily_pnl:.2f} |",
        "",
        "## Overall Totals",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Total signals | {summary.get('total_signals', 0)} |",
        f"| Total settled | {summary.get('total_settled', 0)} |",
        f"| Pending | {summary.get('pending', 0)} |",
        f"| Overall win rate | {_safe_pct(float(summary.get('win_rate', 0.0)))} |",
        f"| Overall total PnL | ${float(summary.get('total_pnl', 0.0)):.2f} |",
        "",
        "## Source Comparison (Overall)",
        "",
        "| Source | Trades | Win Rate | Total PnL | Sharpe |",
        "| --- | --- | --- | --- | --- |",
        f"| SynthData | {source_cmp.get('synthdata', {}).get('n_trades', 0)} | "
        f"{_safe_pct(float(source_cmp.get('synthdata', {}).get('win_rate', 0.0)))} | "
        f"${float(source_cmp.get('synthdata', {}).get('total_pnl', 0.0)):.2f} | "
        f"{float(source_cmp.get('synthdata', {}).get('sharpe', 0.0)):.2f} |",
        f"| Ensemble | {source_cmp.get('ensemble', {}).get('n_trades', 0)} | "
        f"{_safe_pct(float(source_cmp.get('ensemble', {}).get('win_rate', 0.0)))} | "
        f"${float(source_cmp.get('ensemble', {}).get('total_pnl', 0.0)):.2f} | "
        f"{float(source_cmp.get('ensemble', {}).get('sharpe', 0.0)):.2f} |",
        f"| Blended | {source_cmp.get('blended', {}).get('n_trades', 0)} | "
        f"{_safe_pct(float(source_cmp.get('blended', {}).get('win_rate', 0.0)))} | "
        f"${float(source_cmp.get('blended', {}).get('total_pnl', 0.0)):.2f} | "
        f"{float(source_cmp.get('blended', {}).get('sharpe', 0.0)):.2f} |",
        "",
        f"Alpha note: {source_cmp.get('alpha', 'N/A')}",
        "",
        "## False-Positive Patterns",
        "",
    ]

    if top_patterns:
        lines.extend([
            "| Pattern | Loss Count |",
            "| --- | --- |",
        ])
        for label, count in top_patterns:
            lines.append(f"| {label} | {count} |")
    else:
        lines.append("No clear loss pattern from settled trades for this date.")

    lines.extend([
        "",
        "## Next Filter Tweaks",
        "",
        "1. Raise `min_abs_edge` if daily alert volume is high but win rate is weak.",
        "2. Raise `min_agreement_score` if most losses are low-consensus signals.",
        "3. Tighten `max_spread` if losses cluster in wide-spread contracts.",
    ])

    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SynthEdge daily report markdown.")
    parser.add_argument("--date", default=None, help="UTC report date (YYYY-MM-DD). Defaults to current UTC date.")
    parser.add_argument("--output", default=None, help="Output file path. Default: data/reports/daily_report_<date>.md")
    parser.add_argument("--stdout-only", action="store_true", help="Print report only; do not write file.")
    args = parser.parse_args()

    report_date = _parse_date(args.date)
    report_text = generate_report(report_date)

    print(report_text)

    if args.stdout_only:
        return

    output_path = Path(args.output) if args.output else (DATA_DIR.parent / "reports" / f"daily_report_{report_date}.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text)
    print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()
