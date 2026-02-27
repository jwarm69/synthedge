"""
Auto Signal Recorder - Background thread that continuously scans for edges
and records signals to the P&L tracker.

Runs alongside the Streamlit dashboard, scanning every interval_seconds.
"""

import threading
import time
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("auto_recorder")

_recorder_thread = None
_recorder_stop = threading.Event()


def _recording_loop(
    asset: str,
    bankroll: float,
    interval_seconds: int,
    min_edge: float,
):
    """Main recording loop - runs in background thread."""
    from edge_detector import scan_all_edges
    from pnl_tracker import record_signal

    logger.info(f"Auto-recorder started: asset={asset}, interval={interval_seconds}s")

    while not _recorder_stop.is_set():
        try:
            edges = scan_all_edges(asset, bankroll=bankroll)
            if edges and edges.get("status") in ("ok", "partial"):
                current_price = edges.get("current_price", 0)
                recorded = 0

                for hz in ("1h", "15min"):
                    hz_data = edges.get(hz, {})
                    blended = hz_data.get("blended_signal")
                    opportunities = hz_data.get("opportunities", [])
                    snapshot = hz_data.get("market_snapshot")

                    if not blended or not opportunities:
                        continue

                    # Estimate settlement time from market snapshot
                    settlement_time = None
                    if snapshot:
                        ttl_min = snapshot.get("time_to_settlement_min", 0)
                        if ttl_min > 0:
                            settlement_time = datetime.utcnow() + timedelta(minutes=ttl_min)

                    for opp in opportunities:
                        edge = abs(opp.get("edge", 0))
                        if edge < min_edge:
                            continue

                        record_signal(
                            blended_signal=blended,
                            edge_opportunity=opp,
                            asset=asset,
                            current_price=current_price,
                            settlement_time=settlement_time,
                            bankroll=bankroll,
                        )
                        recorded += 1

                if recorded > 0:
                    logger.info(f"Recorded {recorded} signals at {datetime.utcnow().strftime('%H:%M:%S')}")

        except Exception as e:
            logger.warning(f"Auto-recorder error: {e}")

        _recorder_stop.wait(interval_seconds)

    logger.info("Auto-recorder stopped.")


def start_recorder(
    asset: str = "BTC",
    bankroll: float = 1000.0,
    interval_seconds: int = 60,
    min_edge: float = 0.05,
) -> bool:
    """Start the background recording thread.

    Returns True if started, False if already running.
    """
    global _recorder_thread

    if _recorder_thread is not None and _recorder_thread.is_alive():
        return False

    _recorder_stop.clear()
    _recorder_thread = threading.Thread(
        target=_recording_loop,
        args=(asset, bankroll, interval_seconds, min_edge),
        daemon=True,
        name="auto_recorder",
    )
    _recorder_thread.start()
    return True


def stop_recorder() -> bool:
    """Stop the background recording thread.

    Returns True if stopped, False if wasn't running.
    """
    global _recorder_thread

    if _recorder_thread is None or not _recorder_thread.is_alive():
        return False

    _recorder_stop.set()
    _recorder_thread.join(timeout=5)
    _recorder_thread = None
    return True


def is_running() -> bool:
    """Check if the recorder is currently running."""
    return _recorder_thread is not None and _recorder_thread.is_alive()


def get_status() -> dict:
    """Get recorder status for display."""
    return {
        "running": is_running(),
        "thread_name": _recorder_thread.name if _recorder_thread else None,
    }
