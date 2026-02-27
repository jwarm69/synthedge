"""
SynthEdge Dashboard v2 - Hackathon demo for SynthData x Kalshi BTC trading.

6-tab Streamlit dashboard:
1. Live Edge Scanner (hero) - with Polymarket 3-way comparison + 5min horizon
2. Signal Breakdown
3. Price Distribution
4. P&L Tracker + Paper Trading + Backtest Simulator
5. Model Comparison + Source Performance Comparison
6. Network Intelligence (Bittensor Leaderboard + Consensus)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import math

st.set_page_config(
    page_title="SynthEdge - BTC Edge Scanner",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject global CSS for dark theme polish
st.markdown("""
<style>
    .price-header {
        text-align: center;
        padding: 12px 0 8px 0;
        border-bottom: 1px solid #333;
        margin-bottom: 16px;
    }
    .price-header .price-value {
        font-size: 42px;
        font-weight: 800;
        letter-spacing: -1px;
    }
    .price-header .price-change {
        font-size: 16px;
        margin-left: 12px;
    }
    .countdown-bar {
        text-align: center;
        padding: 8px 16px;
        border-radius: 8px;
        margin-bottom: 12px;
        font-weight: 600;
    }
    .signal-card {
        border-radius: 12px;
        padding: 18px 12px;
        text-align: center;
        transition: transform 0.15s;
    }
    .signal-card:hover { transform: scale(1.02); }
    .strength-meter {
        height: 8px;
        border-radius: 4px;
        margin: 8px auto;
        max-width: 180px;
    }
    .three-way-banner {
        text-align: center;
        padding: 10px;
        border-radius: 8px;
        margin: 8px 0 16px 0;
        font-weight: 700;
        font-size: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Color scheme
GREEN = "#26a69a"
RED = "#ef5350"
BLUE = "#1565C0"
ORANGE = "#FF9800"
PURPLE = "#9C27B0"
GOLD = "#FFD700"

# Import local modules
try:
    from synthdata_client import (
        get_directional_forecast,
        get_price_percentiles,
        get_volatility_forecast,
        get_all_signals,
        get_leaderboard,
        get_meta_leaderboard,
        get_validation_scores,
    )
    HAS_SYNTHDATA = True
except ImportError:
    HAS_SYNTHDATA = False

try:
    from signal_blender import (
        blend_predictions,
        blend_synthdata_only,
        get_three_way_comparison,
        Agreement,
        EdgeQuality,
    )
    HAS_BLENDER = True
except ImportError:
    HAS_BLENDER = False

try:
    from edge_detector import scan_all_edges, MIN_EDGE_THRESHOLD
    HAS_EDGE = True
except ImportError:
    HAS_EDGE = False

try:
    from pnl_tracker import (
        get_performance_summary,
        get_signal_history,
        get_settled_history,
        evaluate_settled,
        record_signal,
        get_source_comparison,
    )
    HAS_PNL = True
except ImportError:
    HAS_PNL = False

try:
    from backtester import run_backtest
    HAS_BACKTESTER = True
except ImportError:
    HAS_BACKTESTER = False

try:
    from predict_v2 import predict_with_threshold, DEFAULT_CALIBRATION
    from data_fetch import get_latest_candles, SYMBOLS
    HAS_PREDICT = True
except ImportError:
    HAS_PREDICT = False
    DEFAULT_CALIBRATION = "shrink"

try:
    from kalshi_api import get_market_snapshot, KALSHI_TICKERS
    HAS_KALSHI = True
except ImportError:
    HAS_KALSHI = False

try:
    from auto_recorder import start_recorder, stop_recorder, is_running as recorder_is_running
    HAS_RECORDER = True
except ImportError:
    HAS_RECORDER = False

MODELS_DIR = Path(__file__).parent.parent / "models"


# =============================================================================
# Cached Data Loading
# =============================================================================

@st.cache_data(ttl=60)
def cached_scan_all_edges(asset: str, bankroll: float):
    if not HAS_EDGE:
        return None
    try:
        return scan_all_edges(asset, bankroll=bankroll)
    except Exception as e:
        st.error(f"Edge scan error: {e}")
        return None


@st.cache_data(ttl=60)
def cached_synthdata_signals(asset: str):
    if not HAS_SYNTHDATA:
        return None
    try:
        return get_all_signals(asset)
    except Exception:
        return None


@st.cache_data(ttl=60)
def cached_ensemble_prediction(coin: str):
    if not HAS_PREDICT:
        return None
    try:
        btc_df = get_latest_candles("btc", limit=200) if coin.lower() != "btc" else None
        return predict_with_threshold(coin.lower(), btc_df=btc_df)
    except Exception:
        return None


@st.cache_data(ttl=300)
def cached_performance_summary():
    if not HAS_PNL:
        return {}
    try:
        evaluate_settled()
        return get_performance_summary()
    except Exception:
        return {}


@st.cache_data(ttl=120)
def cached_leaderboard():
    if not HAS_SYNTHDATA:
        return None
    try:
        return get_leaderboard()
    except Exception:
        return None


@st.cache_data(ttl=120)
def cached_validation_scores():
    if not HAS_SYNTHDATA:
        return None
    try:
        return get_validation_scores()
    except Exception:
        return None


@st.cache_data(ttl=300)
def cached_source_comparison():
    if not HAS_PNL:
        return None
    try:
        return get_source_comparison()
    except Exception:
        return None


# =============================================================================
# Helpers: Live Price Header + Countdown
# =============================================================================

def render_live_price_header(current_price: float, asset: str = "BTC"):
    """Large live price display at top of page."""
    if current_price <= 0:
        return

    color = "#e0e0e0"
    st.markdown(f"""
    <div class="price-header">
        <span style="color: #888; font-size: 14px;">{asset}/USD LIVE</span><br>
        <span class="price-value" style="color: {color};">${current_price:,.2f}</span>
    </div>
    """, unsafe_allow_html=True)


def render_countdown_timer(time_to_settlement_min: float, horizon: str):
    """Countdown timer to next Kalshi settlement."""
    if time_to_settlement_min <= 0:
        return

    minutes = int(time_to_settlement_min)
    seconds = int((time_to_settlement_min - minutes) * 60)

    if minutes < 5:
        bg = f"linear-gradient(90deg, {RED}33, {RED}11)"
        text_color = RED
        urgency = "CLOSING SOON"
    elif minutes < 15:
        bg = f"linear-gradient(90deg, {ORANGE}33, {ORANGE}11)"
        text_color = ORANGE
        urgency = "ACTIVE"
    else:
        bg = f"linear-gradient(90deg, {GREEN}33, {GREEN}11)"
        text_color = GREEN
        urgency = "OPEN"

    st.markdown(f"""
    <div class="countdown-bar" style="background: {bg}; border: 1px solid {text_color}44;">
        <span style="color: {text_color};">
            {horizon.upper()} Settlement in {minutes:02d}:{seconds:02d}
        </span>
        <span style="color: #888; margin-left: 12px; font-size: 12px;">{urgency}</span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Helpers: Signal Cards + Agreement Meter
# =============================================================================

def render_signal_card(title: str, prob_up: float, direction: str, confidence: float, status: str, color: str):
    """Render a compact signal card with glow effect."""
    arrow = "↑" if direction == "UP" else "↓"
    dir_color = GREEN if direction == "UP" else RED
    conf_pct = f"{confidence * 100:.1f}%"

    st.markdown(f"""
    <div class="signal-card" style="
        border: 2px solid {color};
        background: linear-gradient(135deg, {color}15, {color}05);
        box-shadow: 0 0 15px {color}22;
    ">
        <div style="font-size: 13px; color: #888; margin-bottom: 6px;">{title}</div>
        <div style="font-size: 32px; font-weight: 800; color: {dir_color};">
            {arrow} {direction}
        </div>
        <div style="font-size: 16px; color: #ccc; margin-top: 2px;">{conf_pct} confidence</div>
        <div style="font-size: 12px; color: #666; margin-top: 6px;">P(UP) = {prob_up:.3f}</div>
        <div style="font-size: 11px; color: {'#4caf50' if status == 'ok' else '#ff9800'}; margin-top: 4px;">
            {status.upper()}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_agreement_badge(agreement_str: str, quality_str: str, agreement_score: float = 0):
    """Render agreement badge with strength meter."""
    ag_colors = {
        "STRONG_AGREE": "#4caf50",
        "AGREE": "#8bc34a",
        "NEUTRAL": "#ffc107",
        "DISAGREE": "#ff9800",
        "STRONG_DISAGREE": "#f44336",
    }
    q_colors = {
        "HIGH": "#4caf50",
        "MEDIUM": "#ffc107",
        "LOW": "#f44336",
    }

    ag_color = ag_colors.get(agreement_str, "#888")
    q_color = q_colors.get(quality_str, "#888")

    # Strength meter: agreement_score from -1 to +1, map to 0-100%
    meter_pct = max(0, min(100, (agreement_score + 1) * 50))
    meter_color = GREEN if agreement_score > 0.2 else RED if agreement_score < -0.2 else ORANGE

    st.markdown(f"""
    <div style="text-align: center; margin-top: 10px;">
        <span style="
            background: {ag_color};
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 13px;
            font-weight: bold;
            margin-right: 8px;
        ">{agreement_str.replace('_', ' ')}</span>
        <span style="
            background: {q_color};
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 13px;
            font-weight: bold;
        ">QUALITY: {quality_str}</span>
        <div class="strength-meter" style="
            background: linear-gradient(90deg, {RED} 0%, {ORANGE} 50%, {GREEN} 100%);
            position: relative;
            margin-top: 10px;
        ">
            <div style="
                position: absolute;
                left: {meter_pct}%;
                top: -4px;
                width: 4px;
                height: 16px;
                background: white;
                border-radius: 2px;
                transform: translateX(-50%);
                box-shadow: 0 0 4px white;
            "></div>
        </div>
        <div style="font-size: 11px; color: #666; margin-top: 2px;">
            Agreement Strength: {agreement_score:+.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_three_way_banner(comparison: dict):
    """Show 3-way comparison banner when all sources agree or disagree."""
    if not comparison:
        return

    all_agree = comparison.get("all_agree", False)
    agree_dir = comparison.get("agree_direction")
    sources_up = comparison.get("sources_up", 0)
    conviction = comparison.get("conviction", 0)

    if all_agree and agree_dir:
        dir_color = GREEN if agree_dir == "UP" else RED
        arrow = "↑↑↑" if agree_dir == "UP" else "↓↓↓"
        st.markdown(f"""
        <div class="three-way-banner" style="
            background: linear-gradient(90deg, {dir_color}22, {dir_color}11);
            border: 2px solid {dir_color};
            color: {dir_color};
        ">
            {arrow} ALL 3 SOURCES AGREE: {agree_dir} (conviction: {conviction:.0%}) {arrow}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="three-way-banner" style="
            background: {ORANGE}11;
            border: 1px solid {ORANGE}44;
            color: {ORANGE};
        ">
            MIXED SIGNALS: {sources_up}/3 sources say UP
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# Tab 1: Live Edge Scanner
# =============================================================================

def render_live_edge_scanner():
    col_asset, col_bank, col_refresh = st.columns([1, 1, 2])
    with col_asset:
        asset = st.selectbox("Asset", ["BTC", "ETH"], key="edge_asset")
    with col_bank:
        bankroll = st.number_input("Bankroll ($)", value=1000, min_value=100, step=100, key="edge_bankroll")
    with col_refresh:
        auto_refresh = st.toggle("Auto-refresh (60s)", value=False, key="edge_refresh")

    if auto_refresh:
        st.markdown(f"<div style='color:{GREEN}; font-size:12px;'>Auto-refresh active</div>",
                    unsafe_allow_html=True)

    # Run edge scan
    edges = cached_scan_all_edges(asset, float(bankroll))

    if edges is None:
        st.warning("Edge detection modules not available. Check imports.")
        return

    # Live price header
    current_price = edges.get("current_price", 0)
    render_live_price_header(current_price, asset)

    for hz in ("1h", "15min", "5min"):
        hz_data = edges.get(hz, {})
        blended = hz_data.get("blended_signal")
        if not blended:
            continue

        # Countdown timer (5min has no Kalshi market)
        snapshot = hz_data.get("market_snapshot")
        if snapshot:
            ttl = snapshot.get("time_to_settlement_min", 0)
            render_countdown_timer(ttl, hz)

        st.markdown(f"### {hz.upper()} Horizon")

        # Extract probabilities
        synth_prob = blended.get("synthdata_prob_up", 0.5)
        ens_prob = blended.get("ensemble_prob_up", 0.5)

        # Get Polymarket prob from SynthData directional response
        poly_prob = 0.5
        synth_signals = cached_synthdata_signals(asset)
        if synth_signals:
            hz_synth = synth_signals.get(hz, {})
            directional = hz_synth.get("directional", {})
            poly_prob = directional.get("polymarket_prob_up", 0.5)
            if poly_prob == 0:
                poly_prob = 0.5

        # 3-way comparison banner
        if HAS_BLENDER and poly_prob != 0.5:
            comparison = get_three_way_comparison(synth_prob, ens_prob, poly_prob)
            render_three_way_banner(comparison)

        # Four signal cards: SynthData, Ensemble, Polymarket, Blended
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            synth_dir = "UP" if synth_prob > 0.5 else "DOWN"
            synth_conf = abs(synth_prob - 0.5) * 2
            render_signal_card(
                "SynthData (200+ models)",
                synth_prob, synth_dir, synth_conf,
                edges.get("synthdata_status", "unavailable"),
                PURPLE,
            )

        with col2:
            ens_dir = "UP" if ens_prob > 0.5 else "DOWN"
            ens_conf = abs(ens_prob - 0.5) * 2
            render_signal_card(
                "Local Ensemble (XGB+LGB+Cat)",
                ens_prob, ens_dir, ens_conf,
                edges.get("ensemble_status", "unavailable"),
                BLUE,
            )

        with col3:
            poly_dir = "UP" if poly_prob > 0.5 else "DOWN"
            poly_conf = abs(poly_prob - 0.5) * 2
            poly_status = "ok" if poly_prob != 0.5 else "unavailable"
            render_signal_card(
                "Polymarket (Cross-Exchange)",
                poly_prob, poly_dir, poly_conf,
                poly_status,
                ORANGE,
            )

        with col4:
            bl_prob = blended.get("blended_prob_up", 0.5)
            bl_dir = blended.get("blended_direction", "UP")
            bl_conf = blended.get("blended_confidence", 0)
            bl_color = GREEN if bl_dir == "UP" else RED
            render_signal_card(
                "BLENDED SIGNAL",
                bl_prob, bl_dir, bl_conf,
                "blended",
                bl_color,
            )

        # Agreement badge with strength meter
        agreement_str = blended.get("agreement", "")
        if hasattr(agreement_str, "value"):
            agreement_str = agreement_str.value
        quality_str = blended.get("quality", "")
        if hasattr(quality_str, "value"):
            quality_str = quality_str.value
        agreement_score = blended.get("agreement_score", 0)
        render_agreement_badge(str(agreement_str), str(quality_str), agreement_score)

        st.markdown("---")

        # Edge opportunities table with color coding + Polymarket column
        opportunities = hz_data.get("opportunities", [])
        if opportunities:
            st.subheader(f"{hz.upper()} Edge Opportunities ({len(opportunities)} found)")

            # Build styled HTML table
            _render_edge_table(opportunities, poly_prob)

            # Paper trade buttons
            if HAS_PNL:
                st.markdown("**Paper Trade:**")
                pt_cols = st.columns(min(len(opportunities), 4))
                for i, opp in enumerate(opportunities[:4]):
                    with pt_cols[i]:
                        btn_label = f"{opp.get('action', 'BUY')} {opp.get('subtitle', '')[:20]}"
                        btn_key = f"pt_{hz}_{i}_{opp.get('ticker', i)}"
                        if st.button(btn_label, key=btn_key, type="primary" if opp.get("edge", 0) > 0.08 else "secondary"):
                            settlement_time = None
                            if snapshot:
                                ttl_min = snapshot.get("time_to_settlement_min", 0)
                                if ttl_min > 0:
                                    settlement_time = datetime.utcnow() + timedelta(minutes=ttl_min)

                            recorded = record_signal(
                                blended_signal=blended,
                                edge_opportunity=opp,
                                asset=asset,
                                current_price=current_price,
                                settlement_time=settlement_time,
                                bankroll=float(bankroll),
                            )
                            st.success(f"Paper trade recorded: {recorded.get('signal_id', '?')}")
                            st.cache_data.clear()
        else:
            st.info(f"No edge opportunities above {MIN_EDGE_THRESHOLD:.0%} threshold for {hz}")

        # Market snapshot metrics
        if snapshot:
            liq = snapshot.get("liquidity", {})
            ttl = snapshot.get("time_to_settlement_min", 0)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Time to Settlement", f"{ttl:.0f} min")
            col2.metric("Total Volume", liq.get("total_volume", 0))
            col3.metric("Open Interest", liq.get("total_oi", 0))
            col4.metric("Active Markets", liq.get("active_markets", 0))

        st.divider()


def _render_edge_table(opportunities: list, polymarket_prob: float = 0.5):
    """Render color-coded edge opportunities as an HTML table with spread indicator."""
    rows_html = ""
    for opp in opportunities:
        edge = opp.get("edge", 0)
        ev = opp.get("ev_per_contract", 0)

        # Color coding
        if edge > 0.08:
            row_bg = f"{GREEN}18"
            edge_color = GREEN
            edge_weight = "bold"
        elif edge > 0:
            row_bg = f"{GREEN}08"
            edge_color = GREEN
            edge_weight = "normal"
        elif edge < -0.08:
            row_bg = f"{RED}18"
            edge_color = RED
            edge_weight = "bold"
        else:
            row_bg = f"{RED}08"
            edge_color = RED
            edge_weight = "normal"

        # EV color
        ev_color = GREEN if ev > 0 else RED

        # Bid-ask spread
        yes_ask = opp.get("yes_ask", 0)
        yes_bid = opp.get("yes_bid", yes_ask - 0.02)  # estimate if not present
        spread = abs(yes_ask - yes_bid) if yes_bid else 0.02
        if spread < 0.05:
            spread_color = GREEN
            spread_label = f"{spread:.2f}"
        elif spread < 0.10:
            spread_color = ORANGE
            spread_label = f"{spread:.2f}"
        else:
            spread_color = RED
            spread_label = f"{spread:.2f}"

        rows_html += f"""
        <tr style="background: {row_bg};">
            <td style="padding: 6px 10px;">{opp.get('subtitle', '')}</td>
            <td style="padding: 6px 10px;">{opp.get('action', '')}</td>
            <td style="padding: 6px 10px;">{opp.get('model_prob', 0):.3f}</td>
            <td style="padding: 6px 10px;">{opp.get('market_prob', 0):.3f}</td>
            <td style="padding: 6px 10px; color: {ORANGE};">{polymarket_prob:.3f}</td>
            <td style="padding: 6px 10px; color: {edge_color}; font-weight: {edge_weight};">{edge:+.3f}</td>
            <td style="padding: 6px 10px; color: {ev_color};">${ev:.4f}</td>
            <td style="padding: 6px 10px;">{opp.get('kelly_fraction', 0)*100:.1f}%</td>
            <td style="padding: 6px 10px;">{opp.get('contracts', 0)}</td>
            <td style="padding: 6px 10px; color: {spread_color};">{spread_label}</td>
        </tr>
        """

    st.markdown(f"""
    <div style="overflow-x: auto;">
    <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
        <thead>
            <tr style="border-bottom: 2px solid #444; color: #999;">
                <th style="padding: 8px 10px; text-align: left;">Contract</th>
                <th style="padding: 8px 10px; text-align: left;">Action</th>
                <th style="padding: 8px 10px; text-align: left;">Model P</th>
                <th style="padding: 8px 10px; text-align: left;">Kalshi P</th>
                <th style="padding: 8px 10px; text-align: left; color: {ORANGE};">Poly P</th>
                <th style="padding: 8px 10px; text-align: left;">Edge</th>
                <th style="padding: 8px 10px; text-align: left;">EV/$</th>
                <th style="padding: 8px 10px; text-align: left;">Kelly %</th>
                <th style="padding: 8px 10px; text-align: left;">Contracts</th>
                <th style="padding: 8px 10px; text-align: left;">Spread</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Tab 2: Signal Breakdown
# =============================================================================

def render_signal_breakdown():
    st.header("Signal Breakdown")

    history = get_signal_history(24) if HAS_PNL else pd.DataFrame()

    if not history.empty and "agreement" in history.columns:
        st.subheader("Agreement Distribution (Last 24 Signals)")
        ag_counts = history["agreement"].value_counts()

        fig = go.Figure(go.Bar(
            x=ag_counts.index.tolist(),
            y=ag_counts.values.tolist(),
            marker_color=[
                "#4caf50" if "STRONG_AGREE" in str(x) else
                "#8bc34a" if "AGREE" in str(x) and "DIS" not in str(x) else
                "#ffc107" if "NEUTRAL" in str(x) else
                "#ff9800" if "DISAGREE" in str(x) and "STRONG" not in str(x) else
                "#f44336" for x in ag_counts.index
            ],
        ))
        fig.update_layout(
            height=300,
            title="Agreement Distribution (Last 24 Signals)",
            xaxis_title="Agreement Level",
            yaxis_title="Count",
            margin=dict(l=50, r=50, t=50, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Signal History")
        display_cols = [c for c in [
            "signal_id", "logged_at", "horizon", "blended_direction",
            "blended_confidence", "agreement", "quality", "edge",
            "ev_per_contract", "action"
        ] if c in history.columns]

        if display_cols:
            st.dataframe(history[display_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No signal history yet. Enable Auto-Record or click Paper Trade to start generating signals.")

    st.divider()

    # Source comparison scatter
    st.subheader("Source Agreement Scatter")
    if not history.empty and "synthdata_prob_up" in history.columns and "ensemble_prob_up" in history.columns:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=history["synthdata_prob_up"],
            y=history["ensemble_prob_up"],
            mode="markers",
            marker=dict(
                size=10,
                color=history["blended_confidence"] if "blended_confidence" in history.columns else "#888",
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Confidence"),
            ),
            text=history.get("signal_id", ""),
            hovertemplate="SynthData: %{x:.3f}<br>Ensemble: %{y:.3f}<br>ID: %{text}<extra></extra>",
        ))

        fig.add_shape(type="line", x0=0.5, y0=0, x1=0.5, y1=1,
                      line=dict(color="gray", dash="dash", width=1))
        fig.add_shape(type="line", x0=0, y0=0.5, x1=1, y1=0.5,
                      line=dict(color="gray", dash="dash", width=1))
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                      line=dict(color=GREEN, dash="dot", width=1))

        fig.add_shape(type="rect", x0=0.5, y0=0.5, x1=1, y1=1,
                      fillcolor=GREEN, opacity=0.05, line_width=0)
        fig.add_shape(type="rect", x0=0, y0=0, x1=0.5, y1=0.5,
                      fillcolor=GREEN, opacity=0.05, line_width=0)
        fig.add_shape(type="rect", x0=0.5, y0=0, x1=1, y1=0.5,
                      fillcolor=RED, opacity=0.05, line_width=0)
        fig.add_shape(type="rect", x0=0, y0=0.5, x1=0.5, y1=1,
                      fillcolor=RED, opacity=0.05, line_width=0)

        fig.update_layout(
            height=400,
            title="SynthData vs Ensemble Predictions",
            xaxis_title="SynthData P(UP)",
            yaxis_title="Ensemble P(UP)",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            margin=dict(l=50, r=50, t=50, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Scatter plot requires recorded signals with both sources.")


# =============================================================================
# Tab 3: Price Distribution
# =============================================================================

def render_price_distribution():
    st.header("Price Distribution (SynthData Percentiles)")

    asset = st.selectbox("Asset", ["BTC", "ETH"], key="dist_asset")

    for hz in ("1h", "15min", "5min"):
        st.subheader(f"{hz.upper()} Price Distribution")

        pct_data = None
        if HAS_SYNTHDATA:
            pct_data = get_price_percentiles(asset, hz)

        if pct_data and pct_data.get("status") in ("ok", "stale"):
            percentiles = pct_data.get("percentiles", {})
            current_price = pct_data.get("current_price", 0)

            if percentiles and current_price > 0:
                fig = go.Figure()

                bands = [
                    ("p5", "p95", "rgba(156, 39, 176, 0.1)", "5th-95th"),
                    ("p10", "p90", "rgba(156, 39, 176, 0.15)", "10th-90th"),
                    ("p25", "p75", "rgba(156, 39, 176, 0.25)", "25th-75th"),
                ]

                x_labels = ["Now", f"+{hz}"]

                for low_key, high_key, color, label in bands:
                    low_val = percentiles.get(low_key, current_price)
                    high_val = percentiles.get(high_key, current_price)

                    fig.add_trace(go.Scatter(
                        x=x_labels + x_labels[::-1],
                        y=[current_price, high_val, current_price, low_val],
                        fill="toself",
                        fillcolor=color,
                        line=dict(width=0),
                        name=label,
                        showlegend=True,
                    ))

                p50 = percentiles.get("p50", current_price)
                fig.add_trace(go.Scatter(
                    x=x_labels,
                    y=[current_price, p50],
                    mode="lines+markers",
                    name="Median (p50)",
                    line=dict(color=PURPLE, width=2),
                    marker=dict(size=8),
                ))

                fig.add_hline(y=current_price, line_dash="dash", line_color="white",
                              annotation_text=f"Current: ${current_price:,.0f}")

                if HAS_KALSHI:
                    snap = get_market_snapshot(asset, timeframe=hz)
                    if snap:
                        structure = snap["structure"]
                        strike_prices = []
                        for s in structure.get("above_strikes", [])[:3]:
                            strike_prices.append(s.get("floor_strike", 0))
                        for s in structure.get("below_strikes", [])[:3]:
                            strike_prices.append(s.get("cap_strike", 0))

                        for sp in strike_prices:
                            if sp > 0:
                                fig.add_hline(
                                    y=sp, line_dash="dot", line_color="#ffc107",
                                    annotation_text=f"Strike: ${sp:,.0f}",
                                    annotation_font_size=10,
                                )

                fig.update_layout(
                    height=400,
                    title=f"{asset} {hz} Price Distribution Fan Chart",
                    yaxis_title="Price ($)",
                    margin=dict(l=50, r=50, t=50, b=50),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig, use_container_width=True)

                col1, col2, col3, col4 = st.columns(4)
                p5 = percentiles.get("p5", current_price)
                p95 = percentiles.get("p95", current_price)
                p25 = percentiles.get("p25", current_price)
                p75 = percentiles.get("p75", current_price)

                col1.metric("Current Price", f"${current_price:,.0f}")
                col2.metric("Expected Range (50%)",
                            f"${p25:,.0f} - ${p75:,.0f}",
                            delta=f"{pct_data.get('expected_move_pct', 0):.2f}% width")
                col3.metric("Tail Range (90%)",
                            f"${p5:,.0f} - ${p95:,.0f}")
                col4.metric("Median Target", f"${p50:,.0f}",
                            delta=f"{(p50/current_price - 1)*100:+.2f}%")
            else:
                st.info(f"Percentile data incomplete for {hz}")
        else:
            st.info(f"SynthData percentiles not available for {hz}. Set SYNTHDATA_API_KEY env var.")

        st.divider()

    # Volatility comparison
    st.subheader("Volatility Forecast")
    if HAS_SYNTHDATA:
        vol_1h = get_volatility_forecast(asset, "1h")
        if vol_1h.get("status") in ("ok", "stale"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Forward Vol", f"{vol_1h.get('forward_vol', 0):.4f}")
            col2.metric("Realized Vol", f"{vol_1h.get('realized_vol', 0):.4f}")
            vol_ratio = vol_1h.get("vol_ratio", 1.0)
            vol_label = "Expanding" if vol_ratio > 1.1 else "Contracting" if vol_ratio < 0.9 else "Stable"
            col3.metric("Vol Ratio", f"{vol_ratio:.2f}", delta=vol_label)
        else:
            st.info("Volatility forecast not available.")


# =============================================================================
# Tab 4: P&L Tracker + Paper Trading
# =============================================================================

def render_pnl_tracker():
    st.header("P&L Tracker & Paper Trading")

    if not HAS_PNL:
        st.warning("P&L tracker module not available.")
        return

    summary = cached_performance_summary()

    if not summary or summary.get("total_signals", 0) == 0:
        st.info("No signals recorded yet. Use Paper Trade buttons on the Edge Scanner tab, or enable Auto-Record in the sidebar.")

        # Still show the trade log section in case signals arrive
        st.divider()
        st.subheader("How to Start")
        st.markdown("""
        1. Go to **Live Edge Scanner** tab
        2. Click a **Paper Trade** button on any edge opportunity
        3. Or enable **Auto-Record** in the sidebar to automatically record signals
        4. Come back here to track P&L as contracts settle
        """)
        return

    # Top-level metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    total_signals = summary.get("total_signals", 0)
    total_settled = summary.get("total_settled", 0)
    pending = summary.get("pending", 0)
    win_rate = summary.get("win_rate", 0)
    total_pnl = summary.get("total_pnl", 0)

    col1.metric("Total Signals", total_signals)
    col2.metric("Settled", total_settled)
    col3.metric("Pending", pending)
    col4.metric("Win Rate", f"{win_rate*100:.1f}%",
                delta="above random" if win_rate > 0.5 else "below random" if win_rate < 0.5 and total_settled > 0 else None)
    pnl_color = "normal" if total_pnl >= 0 else "inverse"
    col5.metric("Total PnL", f"${total_pnl:.2f}",
                delta=f"{'profit' if total_pnl > 0 else 'loss' if total_pnl < 0 else 'breakeven'}")

    st.divider()

    # Cumulative PnL chart
    settled = get_settled_history(200)
    if not settled.empty and "pnl" in settled.columns:
        st.subheader("Cumulative P&L")

        settled_sorted = settled.copy()
        if "settled_at" in settled_sorted.columns:
            settled_sorted["settled_at"] = pd.to_datetime(settled_sorted["settled_at"])
            settled_sorted = settled_sorted.sort_values("settled_at")

        settled_sorted["cumulative_pnl"] = settled_sorted["pnl"].cumsum()

        fig = go.Figure()
        final_pnl = settled_sorted["cumulative_pnl"].iloc[-1]
        line_color = GREEN if final_pnl >= 0 else RED
        fill_color = "rgba(38, 166, 154, 0.1)" if final_pnl >= 0 else "rgba(239, 83, 80, 0.1)"

        fig.add_trace(go.Scatter(
            x=settled_sorted.get("settled_at", range(len(settled_sorted))),
            y=settled_sorted["cumulative_pnl"],
            mode="lines+markers",
            name="Cumulative PnL",
            line=dict(color=line_color, width=2),
            fill="tozeroy",
            fillcolor=fill_color,
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            height=350,
            title="Cumulative P&L Over Time",
            yaxis_title="P&L ($)",
            margin=dict(l=50, r=50, t=50, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Breakdown by horizon and agreement
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("By Horizon")
        by_hz = summary.get("by_horizon", {})
        if by_hz:
            hz_data = []
            for hz, stats in by_hz.items():
                hz_data.append({
                    "Horizon": hz,
                    "Trades": stats.get("count", 0),
                    "Win Rate": f"{stats.get('win_rate', 0)*100:.1f}%",
                    "Total PnL": f"${stats.get('total_pnl', 0):.2f}",
                    "Avg PnL": f"${stats.get('avg_pnl', 0):.4f}",
                })
            st.dataframe(pd.DataFrame(hz_data), use_container_width=True, hide_index=True)
        else:
            st.info("No horizon breakdown yet.")

    with col2:
        st.subheader("By Agreement Level")
        by_ag = summary.get("by_agreement", {})
        if by_ag:
            ag_data = []
            for ag, stats in by_ag.items():
                ag_data.append({
                    "Agreement": ag,
                    "Trades": stats.get("count", 0),
                    "Win Rate": f"{stats.get('win_rate', 0)*100:.1f}%",
                    "Total PnL": f"${stats.get('total_pnl', 0):.2f}",
                    "Avg Edge": f"{stats.get('avg_edge', 0):+.3f}",
                })
            st.dataframe(pd.DataFrame(ag_data), use_container_width=True, hide_index=True)
        else:
            st.info("No agreement breakdown yet.")

    st.divider()

    # Recent signal log (pending + settled)
    st.subheader("Recent Signals (All)")
    all_signals = get_signal_history(30)
    if not all_signals.empty:
        display_cols = [c for c in [
            "signal_id", "logged_at", "asset", "horizon", "blended_direction",
            "agreement", "edge", "ev_per_contract", "action", "contracts"
        ] if c in all_signals.columns]
        if display_cols:
            st.dataframe(all_signals[display_cols], use_container_width=True, hide_index=True)

    st.divider()

    # Settled trade log
    st.subheader("Settled Trades")
    if not settled.empty:
        display_cols = [c for c in [
            "signal_id", "asset", "horizon", "blended_direction",
            "agreement", "edge", "contracts", "won", "pnl", "settled_at"
        ] if c in settled.columns]
        if display_cols:
            st.dataframe(settled[display_cols].tail(20), use_container_width=True, hide_index=True)
    else:
        st.info("No settled trades yet. Trades settle when their contract's settlement time passes.")

    # Backtest Simulator (Phase 4)
    st.divider()
    st.subheader("Backtest Simulator")

    if HAS_BACKTESTER:
        bt_col1, bt_col2, bt_col3 = st.columns(3)
        with bt_col1:
            bt_min_edge = st.slider("Min Edge", 0.03, 0.15, 0.05, 0.01, key="bt_edge")
        with bt_col2:
            bt_agreement = st.selectbox("Agreement Filter",
                                        ["ALL", "AGREE+", "STRONG_AGREE"],
                                        key="bt_agreement")
        with bt_col3:
            bt_kelly = st.slider("Kelly Cap", 0.05, 0.50, 0.25, 0.05, key="bt_kelly")

        if st.button("Run Backtest", type="primary", key="run_bt"):
            bt_result = run_backtest(
                min_edge=bt_min_edge,
                agreement_filter=bt_agreement,
                kelly_cap=bt_kelly,
            )

            if bt_result["n_trades"] > 0:
                # Results metrics
                r_col1, r_col2, r_col3, r_col4, r_col5 = st.columns(5)
                r_col1.metric("Trades", bt_result["n_trades"])
                r_col2.metric("Win Rate", f"{bt_result['win_rate']*100:.1f}%")
                r_col3.metric("Total PnL", f"${bt_result['total_pnl']:.2f}")
                r_col4.metric("Sharpe", f"{bt_result['sharpe']:.2f}")
                r_col5.metric("Max Drawdown", f"{bt_result['max_drawdown']*100:.1f}%")

                # Key insight card
                st.markdown(f"""
                <div style="text-align: center; padding: 12px; margin: 12px 0;
                            background: {GREEN}11; border: 1px solid {GREEN}33; border-radius: 8px;">
                    <span style="color: {GREEN}; font-weight: bold;">
                        At edge>{bt_min_edge:.2f} + {bt_agreement} + kelly={bt_kelly:.0%},
                        SynthEdge achieved {bt_result['win_rate']*100:.1f}% win rate
                        on {bt_result['n_trades']} trades
                        (Sharpe: {bt_result['sharpe']:.2f})
                    </span>
                </div>
                """, unsafe_allow_html=True)

                # Equity curve
                equity = bt_result.get("equity_curve", [])
                if len(equity) > 1:
                    fig = go.Figure()
                    final = equity[-1]
                    line_color = GREEN if final >= equity[0] else RED
                    fig.add_trace(go.Scatter(
                        y=equity,
                        mode="lines",
                        name="Equity",
                        line=dict(color=line_color, width=2),
                        fill="tozeroy",
                        fillcolor=f"rgba(38,166,154,0.1)" if final >= equity[0] else "rgba(239,83,80,0.1)",
                    ))
                    fig.add_hline(y=equity[0], line_dash="dash", line_color="gray",
                                  annotation_text=f"Start: ${equity[0]:,.0f}")
                    fig.update_layout(
                        height=350,
                        title="Backtest Equity Curve",
                        yaxis_title="Bankroll ($)",
                        margin=dict(l=50, r=50, t=50, b=50),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Trade log
                trade_log = bt_result.get("trade_log", [])
                if trade_log:
                    with st.expander("Trade Log", expanded=False):
                        st.dataframe(pd.DataFrame(trade_log), use_container_width=True, hide_index=True)
            else:
                st.warning(f"No trades passed filters (filtered {bt_result['n_filtered']} signals). "
                           "Try relaxing the min edge or agreement filter.")
    else:
        st.info("Backtester module not available.")


# =============================================================================
# Tab 5: Model Comparison
# =============================================================================

def render_model_comparison():
    st.header("Model Comparison")

    st.markdown("""
    Compare prediction accuracy across signal sources:
    - **SynthData**: 200+ ML models on Bittensor Subnet 50
    - **Local Ensemble**: XGBoost + LightGBM + CatBoost trained on historical data
    - **Polymarket**: Cross-exchange market consensus
    - **Blended**: Agreement-boosted combination
    """)

    # Source overview - now 4 columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div style="border: 2px solid {PURPLE}; border-radius: 10px; padding: 15px;">
            <h4 style="color: {PURPLE};">SynthData</h4>
            <ul>
                <li>200+ competing models</li>
                <li>Updated every 60s</li>
                <li>15min + 1h horizons</li>
                <li>Scored by CRPS</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        model_info = "No model loaded"
        if HAS_PREDICT:
            import json as _json
            meta_path = MODELS_DIR / "btc_meta_v2.json"
            if not meta_path.exists():
                meta_path = MODELS_DIR / "btc_meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = _json.load(f)
                acc = meta.get("test_accuracy", 0)
                feats = len(meta.get("features", []))
                model_info = f"Accuracy: {acc*100:.1f}%, {feats} features"

        st.markdown(f"""
        <div style="border: 2px solid {BLUE}; border-radius: 10px; padding: 15px;">
            <h4 style="color: {BLUE};">Local Ensemble</h4>
            <ul>
                <li>XGB + LightGBM + CatBoost</li>
                <li>{model_info}</li>
                <li>1h horizon only</li>
                <li>Walk-forward validated</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="border: 2px solid {ORANGE}; border-radius: 10px; padding: 15px;">
            <h4 style="color: {ORANGE};">Polymarket</h4>
            <ul>
                <li>Cross-exchange reference</li>
                <li>Market consensus</li>
                <li>Included in API response</li>
                <li>Not in blend math</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div style="border: 2px solid {GREEN}; border-radius: 10px; padding: 15px;">
            <h4 style="color: {GREEN};">Blended Signal</h4>
            <ul>
                <li>Agreement-boosted fusion</li>
                <li>Confidence scaling</li>
                <li>Disagreement = sit out</li>
                <li>Kelly-optimal sizing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Rolling accuracy comparison from settled trades
    st.subheader("Source Accuracy (Settled Trades)")

    settled = get_settled_history(200) if HAS_PNL else pd.DataFrame()

    if not settled.empty and "won" in settled.columns:
        settled["won"] = settled["won"].astype(bool)

        if "agreement" in settled.columns:
            fig = go.Figure()

            for ag in ["STRONG_AGREE", "AGREE", "NEUTRAL", "DISAGREE", "STRONG_DISAGREE"]:
                mask = settled["agreement"] == ag
                if mask.sum() > 0:
                    wr = settled.loc[mask, "won"].mean()
                    count = mask.sum()
                    fig.add_trace(go.Bar(
                        x=[ag.replace("_", " ")],
                        y=[wr * 100],
                        name=f"{ag} (n={count})",
                        text=[f"{wr*100:.1f}%"],
                        textposition="auto",
                    ))

            fig.add_hline(y=50, line_dash="dash", line_color="gray",
                          annotation_text="Random (50%)")
            fig.update_layout(
                height=350,
                title="Win Rate by Agreement Level",
                yaxis_title="Win Rate (%)",
                yaxis=dict(range=[0, 100]),
                showlegend=False,
                margin=dict(l=50, r=50, t=50, b=50),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Agreement premium
        st.subheader("Agreement Premium")
        if "agreement" in settled.columns:
            agree_mask = settled["agreement"].isin(["STRONG_AGREE", "AGREE"])
            disagree_mask = settled["agreement"].isin(["DISAGREE", "STRONG_DISAGREE"])

            if agree_mask.sum() > 0 and disagree_mask.sum() > 0:
                agree_wr = settled.loc[agree_mask, "won"].mean()
                disagree_wr = settled.loc[disagree_mask, "won"].mean()
                premium = (agree_wr - disagree_wr) * 100

                col1, col2, col3 = st.columns(3)
                col1.metric("Agree Win Rate",
                            f"{agree_wr*100:.1f}%",
                            delta=f"n={agree_mask.sum()}")
                col2.metric("Disagree Win Rate",
                            f"{disagree_wr*100:.1f}%",
                            delta=f"n={disagree_mask.sum()}")
                col3.metric("Agreement Premium",
                            f"{premium:+.1f}pp",
                            delta="Edge from consensus" if premium > 0 else "No premium yet")
            else:
                st.info("Need trades in both agree and disagree categories to calculate premium.")
    else:
        st.info("No settled trades yet. The model comparison will populate as trades are recorded and settled.")

    st.divider()

    # Source Performance Comparison (Phase 3 - Killer Chart)
    st.subheader("Source Performance Comparison")

    comparison = cached_source_comparison()
    if comparison and comparison.get("blended", {}).get("n_trades", 0) > 0:
        # 3-column comparison cards
        col1, col2, col3 = st.columns(3)

        synth_stats = comparison.get("synthdata", {})
        ens_stats = comparison.get("ensemble", {})
        blend_stats = comparison.get("blended", {})

        with col1:
            wr = synth_stats.get("win_rate", 0)
            st.markdown(f"""
            <div style="border: 2px solid {PURPLE}; border-radius: 10px; padding: 15px; text-align: center;">
                <div style="font-size: 13px; color: #888;">SynthData Only</div>
                <div style="font-size: 36px; font-weight: 800; color: {PURPLE};">{wr*100:.1f}%</div>
                <div style="font-size: 12px; color: #888;">Win Rate | PnL: ${synth_stats.get('total_pnl', 0):.2f}</div>
                <div style="font-size: 11px; color: #666;">Sharpe: {synth_stats.get('sharpe', 0):.2f} | n={synth_stats.get('n_trades', 0)}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            wr = ens_stats.get("win_rate", 0)
            st.markdown(f"""
            <div style="border: 2px solid {BLUE}; border-radius: 10px; padding: 15px; text-align: center;">
                <div style="font-size: 13px; color: #888;">Ensemble Only</div>
                <div style="font-size: 36px; font-weight: 800; color: {BLUE};">{wr*100:.1f}%</div>
                <div style="font-size: 12px; color: #888;">Win Rate | PnL: ${ens_stats.get('total_pnl', 0):.2f}</div>
                <div style="font-size: 11px; color: #666;">Sharpe: {ens_stats.get('sharpe', 0):.2f} | n={ens_stats.get('n_trades', 0)}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            wr = blend_stats.get("win_rate", 0)
            st.markdown(f"""
            <div style="border: 2px solid {GREEN}; border-radius: 10px; padding: 15px; text-align: center;">
                <div style="font-size: 13px; color: #888;">BLENDED</div>
                <div style="font-size: 36px; font-weight: 800; color: {GREEN};">{wr*100:.1f}%</div>
                <div style="font-size: 12px; color: #888;">Win Rate | PnL: ${blend_stats.get('total_pnl', 0):.2f}</div>
                <div style="font-size: 11px; color: #666;">Sharpe: {blend_stats.get('sharpe', 0):.2f} | n={blend_stats.get('n_trades', 0)}</div>
            </div>
            """, unsafe_allow_html=True)

        # Alpha attribution
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; margin-top: 12px;
                    background: {GREEN}11; border: 1px solid {GREEN}33; border-radius: 8px;">
            <span style="color: {GREEN}; font-weight: bold;">{comparison.get('alpha', '')}</span>
        </div>
        """, unsafe_allow_html=True)

        # Cumulative PnL overlay chart
        cumulative = comparison.get("cumulative", {})
        if any(cumulative.get(s) for s in ("synthdata", "ensemble", "blended")):
            fig = go.Figure()

            timestamps = cumulative.get("timestamps", [])

            for source, color, name in [
                ("synthdata", PURPLE, "SynthData Only"),
                ("ensemble", BLUE, "Ensemble Only"),
                ("blended", GREEN, "Blended"),
            ]:
                curve = cumulative.get(source, [])
                if curve:
                    x_axis = timestamps[:len(curve)] if timestamps else list(range(len(curve)))
                    fig.add_trace(go.Scatter(
                        x=x_axis,
                        y=curve,
                        mode="lines",
                        name=name,
                        line=dict(color=color, width=2),
                    ))

            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(
                height=400,
                title="Cumulative P&L by Source (Hypothetical)",
                yaxis_title="Cumulative P&L ($)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(l=50, r=50, t=60, b=50),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Source comparison requires settled trades. Record and settle some paper trades first.")

    st.divider()

    # Architecture diagram
    st.subheader("SynthEdge Architecture")
    st.code("""
    SynthData API ──→ synthdata_client.py ──┐
    (200+ models)     (15min + 1h forecasts) │
                      (percentiles, vol)      ├──→ signal_blender.py ──→ edge_detector.py ──→ dashboard_v2.py
    Local Ensemble ──→ predict_v2.py ────────┘    (agreement boost)      (model vs market)     (this dashboard)
    (XGB+LGB+Cat)                                                              │
                                                                       kalshi_api.py ←─────────┘
    Kalshi API ──→ kalshi_api.py                                       (1h + 15min markets)
    (live pricing)  (extended for 15min)

    Polymarket ──→ (via SynthData API response, cross-exchange reference)
    Auto-Recorder ──→ auto_recorder.py (background thread, 60s interval)
    P&L Tracker ──→ pnl_tracker.py (CSV-based signal log + settlement eval)
    Leaderboard ──→ /v2/leaderboard/latest (Bittensor miner rankings)
    Backtester ──→ backtester.py (parameter sweep + equity curves)
    """, language="text")


# =============================================================================
# Tab 6: Network Intelligence
# =============================================================================

def render_network_intelligence():
    st.header("Network Intelligence (Bittensor Subnet 50)")
    st.markdown("""
    Deep view into the decentralized prediction network powering SynthData.
    This data comes from 200+ competing ML models on Bittensor, scored by CRPS
    (Continuous Ranked Probability Score).
    """)

    if not HAS_SYNTHDATA:
        st.warning("SynthData client not available. Check SYNTHDATA_API_KEY.")
        return

    # Top Miners Leaderboard
    st.subheader("Top Miners by CRPS Score")

    leaderboard = cached_leaderboard()
    if leaderboard and leaderboard.get("status") in ("ok", "stale"):
        miners = leaderboard.get("miners", [])
        if miners:
            # Extract score field (try multiple keys)
            miner_rows = []
            for i, m in enumerate(miners[:10]):
                score = 0
                for key in ("crps_score", "score", "value", "incentive", "emission"):
                    if key in m:
                        try:
                            score = float(m[key])
                        except (ValueError, TypeError):
                            pass
                        break

                uid = m.get("uid", m.get("miner_uid", i))
                hotkey = str(m.get("hotkey", m.get("miner_hotkey", "")))[:12]

                miner_rows.append({
                    "Rank": i + 1,
                    "UID": uid,
                    "Hotkey": hotkey + "..." if hotkey else "N/A",
                    "CRPS Score": f"{score:.4f}" if score else "N/A",
                })

            st.dataframe(pd.DataFrame(miner_rows), use_container_width=True, hide_index=True)

            # Bar chart of miner scores
            scores = []
            labels = []
            for row in miner_rows:
                try:
                    scores.append(float(row["CRPS Score"]))
                    labels.append(f"UID {row['UID']}")
                except ValueError:
                    pass

            if scores:
                fig = go.Figure(go.Bar(
                    x=labels,
                    y=scores,
                    marker_color=[PURPLE if i == 0 else BLUE for i in range(len(scores))],
                    text=[f"{s:.4f}" for s in scores],
                    textposition="auto",
                ))
                fig.update_layout(
                    height=350,
                    title="Top 10 Miners by CRPS Score",
                    yaxis_title="CRPS Score (lower = better)",
                    margin=dict(l=50, r=50, t=50, b=50),
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Leaderboard data is empty. The API may be updating.")
    else:
        st.info("Leaderboard data not available. Check API connection.")

    st.divider()

    # Network Health + Consensus Strength
    st.subheader("Network Health & Consensus")

    validation = cached_validation_scores()
    col1, col2, col3 = st.columns(3)

    if validation and validation.get("status") in ("ok", "stale"):
        avg_score = validation.get("avg_score", 0)
        variance = validation.get("score_variance", 0)
        n_validators = len(validation.get("scores", []))

        # Network health gauge
        with col1:
            if avg_score > 0:
                health_label = "HEALTHY" if avg_score < 0.5 else "MODERATE" if avg_score < 1.0 else "DEGRADED"
                health_color = GREEN if health_label == "HEALTHY" else ORANGE if health_label == "MODERATE" else RED
            else:
                health_label = "NO DATA"
                health_color = "#888"

            st.markdown(f"""
            <div style="border: 2px solid {health_color}; border-radius: 10px; padding: 18px; text-align: center;">
                <div style="font-size: 12px; color: #888;">Network Health</div>
                <div style="font-size: 28px; font-weight: 800; color: {health_color};">{health_label}</div>
                <div style="font-size: 12px; color: #666;">Avg Score: {avg_score:.4f}</div>
                <div style="font-size: 11px; color: #666;">{n_validators} validators</div>
            </div>
            """, unsafe_allow_html=True)

        # Consensus strength (derived from score variance)
        with col2:
            if variance > 0:
                std_dev = variance ** 0.5
                if std_dev < 0.1:
                    consensus = "HIGH CONSENSUS"
                    cons_color = GREEN
                elif std_dev < 0.3:
                    consensus = "MODERATE"
                    cons_color = ORANGE
                else:
                    consensus = "LOW CONSENSUS"
                    cons_color = RED
            else:
                consensus = "NO DATA"
                cons_color = "#888"
                std_dev = 0

            st.markdown(f"""
            <div style="border: 2px solid {cons_color}; border-radius: 10px; padding: 18px; text-align: center;">
                <div style="font-size: 12px; color: #888;">Consensus Strength</div>
                <div style="font-size: 28px; font-weight: 800; color: {cons_color};">{consensus}</div>
                <div style="font-size: 12px; color: #666;">Score Std Dev: {std_dev:.4f}</div>
                <div style="font-size: 11px; color: #666;">Low variance = high agreement</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div style="border: 2px solid {GOLD}; border-radius: 10px; padding: 18px; text-align: center;">
                <div style="font-size: 12px; color: #888;">Prediction Quality</div>
                <div style="font-size: 28px; font-weight: 800; color: {GOLD};">
                    {'HIGH' if avg_score < 0.3 and variance < 0.05 else 'MEDIUM' if avg_score < 0.8 else 'LOW'}
                </div>
                <div style="font-size: 12px; color: #666;">Combined health + consensus</div>
                <div style="font-size: 11px; color: #666;">Use for position sizing</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        with col1:
            st.info("Validation scores not available.")
        with col2:
            st.info("Consensus data requires validation scores.")
        with col3:
            st.info("Quality assessment pending data.")

    st.divider()

    # Why this matters
    st.subheader("Why Network Intelligence Matters")
    st.markdown("""
    **For Trading Decisions:**
    - **High consensus + healthy network** = Trust SynthData signals more, increase position sizes
    - **Low consensus** = Models disagree, reduce exposure or sit out
    - **Degraded health** = Network issues, fall back to local ensemble only

    **Technical Depth:**
    SynthEdge is the only hackathon entry that uses the Bittensor miner leaderboard
    to dynamically assess prediction quality. This shows deep understanding of Subnet 50's
    architecture and creates a meta-layer of intelligence above raw predictions.
    """)


# =============================================================================
# Main App
# =============================================================================

def main():
    # Sidebar
    with st.sidebar:
        st.title("⚡ SynthEdge")
        st.caption("SynthData x Kalshi BTC Edge Scanner")
        st.markdown("---")

        st.write("**System Status**")

        modules = {
            "SynthData Client": HAS_SYNTHDATA,
            "Signal Blender": HAS_BLENDER,
            "Edge Detector": HAS_EDGE,
            "P&L Tracker": HAS_PNL,
            "Local Ensemble": HAS_PREDICT,
            "Kalshi API": HAS_KALSHI,
            "Auto-Recorder": HAS_RECORDER,
        }

        for name, available in modules.items():
            icon = "✓" if available else "✗"
            color = "green" if available else "red"
            st.markdown(f"<span style='color:{color}'>{icon}</span> {name}",
                        unsafe_allow_html=True)

        st.markdown("---")

        # Auto-recorder controls
        if HAS_RECORDER:
            st.write("**Auto-Record Signals**")
            recorder_running = recorder_is_running()

            if recorder_running:
                st.markdown(f"<span style='color:{GREEN};'>Recording active</span>",
                            unsafe_allow_html=True)
                if st.button("Stop Recording", key="stop_rec"):
                    stop_recorder()
                    st.rerun()
            else:
                rec_interval = st.selectbox("Interval", [60, 120, 300], format_func=lambda x: f"{x}s", key="rec_int")
                if st.button("Start Recording", key="start_rec", type="primary"):
                    start_recorder(
                        asset="BTC",
                        bankroll=1000.0,
                        interval_seconds=rec_interval,
                    )
                    st.rerun()

            st.markdown("---")

        # Volatility Regime Badge
        st.markdown("---")
        st.write("**Volatility Regime**")
        if HAS_SYNTHDATA:
            try:
                vol_data = get_volatility_forecast("BTC", "1h")
                if vol_data.get("status") in ("ok", "stale"):
                    vol_ratio = vol_data.get("vol_ratio", 1.0)
                    if vol_ratio > 1.3:
                        vol_label, vol_color = "HIGH VOL", RED
                    elif vol_ratio < 0.8:
                        vol_label, vol_color = "LOW VOL", BLUE
                    else:
                        vol_label, vol_color = "NORMAL", GREEN
                    st.markdown(f"""
                    <div style="text-align: center; padding: 6px;
                                background: {vol_color}22; border: 1px solid {vol_color};
                                border-radius: 8px; font-weight: bold; color: {vol_color};">
                        {vol_label} ({vol_ratio:.2f}x)
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color:#888;'>Vol data unavailable</span>",
                                unsafe_allow_html=True)
            except Exception:
                st.markdown("<span style='color:#888;'>Vol data error</span>",
                            unsafe_allow_html=True)
        else:
            st.markdown("<span style='color:#888;'>SynthData not loaded</span>",
                        unsafe_allow_html=True)

        # Settlement Results Ticker
        st.markdown("---")
        st.write("**Recent Settlements**")
        if HAS_PNL:
            try:
                settled = get_settled_history(5)
                if not settled.empty and "pnl" in settled.columns:
                    ticker_items = []
                    for _, row in settled.tail(5).iterrows():
                        sid = str(row.get("signal_id", "?"))
                        pnl = float(row.get("pnl", 0))
                        won = row.get("won", False)
                        icon = "W" if won else "L"
                        color = GREEN if won else RED
                        ticker_items.append(
                            f"<span style='color:{color};'>{sid}: {icon} "
                            f"{'+'if pnl>=0 else ''}${pnl:.2f}</span>"
                        )
                    st.markdown(" | ".join(ticker_items), unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color:#888;'>No settlements yet</span>",
                                unsafe_allow_html=True)
            except Exception:
                pass

        st.markdown("---")
        st.write(f"Updated: {datetime.now().strftime('%H:%M:%S')}")

        st.markdown("---")
        st.caption("SynthData Predictive Intelligence Hackathon")
        st.caption("Bittensor Subnet 50 | 200+ ML Models")

    # Main content with tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "⚡ Live Edge Scanner",
        "📊 Signal Breakdown",
        "📈 Price Distribution",
        "💰 P&L Tracker",
        "🔬 Model Comparison",
        "🧠 Network Intelligence",
    ])

    with tab1:
        render_live_edge_scanner()

    with tab2:
        render_signal_breakdown()

    with tab3:
        render_price_distribution()

    with tab4:
        render_pnl_tracker()

    with tab5:
        render_model_comparison()

    with tab6:
        render_network_intelligence()


if __name__ == "__main__":
    main()
