"""
Streamlit dashboard for Crypto-Kalshi Predictor.
Visualizes price data, predictions, on-chain metrics, and Kalshi contracts.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json

# Set page config first (must be first Streamlit command)
st.set_page_config(
    page_title="Crypto Kalshi Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import local modules
from data_fetch import load_data, get_latest_candles, SYMBOLS
from features import prepare_features, get_feature_columns
from charts import (
    create_candlestick_chart,
    create_rsi_chart,
    create_macd_chart,
    create_fgi_timeline,
    create_mvrv_chart,
    create_sopr_chart,
    create_prediction_gauge,
    create_feature_importance,
    create_accuracy_chart,
    create_prediction_histogram
)

# Import prediction functions
try:
    from predict_v2 import (
        predict_with_threshold,
        load_model_v2,
        load_meta_v2,
        backtest_threshold,
        DEFAULT_CALIBRATION
    )
    from ensemble import EnsembleClassifier
    HAS_PREDICT = True
except ImportError as e:
    HAS_PREDICT = False
    DEFAULT_CALIBRATION = "shrink"
    st.warning(f"Prediction module not available: {e}")

DEFAULT_THRESHOLD = 0.58

# Import Kalshi API
try:
    from kalshi_api import (
        get_next_hourly_event,
        get_markets_for_event,
        parse_market_structure,
        KALSHI_TICKERS
    )
    HAS_KALSHI = True
except ImportError:
    HAS_KALSHI = False

MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"
KALSHI_DATA_DIR = DATA_DIR / "kalshi"
KALSHI_SIGNALS_FILE = KALSHI_DATA_DIR / "signals.csv"
KALSHI_RESULTS_FILE = KALSHI_DATA_DIR / "signal_results.csv"


# =============================================================================
# Cached Data Loading
# =============================================================================

@st.cache_data(ttl=300)  # 5 min cache
def load_price_data(coin: str, limit: int = None) -> pd.DataFrame:
    """Load price data from parquet or API."""
    try:
        df = load_data(coin)
        if limit:
            df = df.tail(limit)
        return df
    except Exception as e:
        st.error(f"Error loading {coin} data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)  # 1 min cache for live data
def get_live_data(coin: str, limit: int = 200) -> pd.DataFrame:
    """Get latest candles from API."""
    try:
        return get_latest_candles(coin, limit=limit)
    except Exception as e:
        st.error(f"Error fetching live {coin} data: {e}")
        return pd.DataFrame()


@st.cache_resource
def load_cached_model(coin: str):
    """Load trained model."""
    if not HAS_PREDICT:
        return None
    try:
        model, _ = load_model_v2(coin)
        return model
    except Exception as e:
        st.error(f"No model for {coin}: {e}")
        return None


@st.cache_data(ttl=3600)  # 1 hour cache
def load_onchain_data() -> pd.DataFrame:
    """Load on-chain metrics."""
    try:
        from onchain import get_all_onchain_metrics
        return get_all_onchain_metrics()
    except Exception as e:
        st.warning(f"On-chain data not available: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_fgi_data() -> pd.DataFrame:
    """Load Fear & Greed Index data."""
    try:
        from fear_greed import load_fgi_data as load_fgi
        return load_fgi()
    except Exception as e:
        st.warning(f"FGI data not available: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_kalshi_results() -> pd.DataFrame:
    if KALSHI_RESULTS_FILE.exists():
        try:
            return pd.read_csv(KALSHI_RESULTS_FILE)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_kalshi_signals() -> pd.DataFrame:
    if KALSHI_SIGNALS_FILE.exists():
        try:
            return pd.read_csv(KALSHI_SIGNALS_FILE)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_threshold_accuracy(
    coin: str,
    threshold: float = DEFAULT_THRESHOLD,
    calibration: str = DEFAULT_CALIBRATION
) -> float:
    """Compute threshold accuracy using the same calibration as live predictions."""
    if not HAS_PREDICT:
        return 0.0
    try:
        bt = backtest_threshold(
            coin,
            threshold=threshold,
            days_back=90,
            calibration=calibration
        )
        return float(bt.get("threshold_accuracy", 0.0))
    except Exception:
        return 0.0


# =============================================================================
# Tab 1: Live Predictions
# =============================================================================

def render_live_predictions():
    """Render the live predictions tab."""
    st.header("Live Predictions")

    # Coin selector
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        coin = st.selectbox("Select Coin", list(SYMBOLS.keys()), format_func=str.upper)
    with col2:
        auto_refresh = st.toggle("Auto-refresh (60s)", value=False)

    if auto_refresh:
        st.info("Auto-refresh enabled. Page will update every 60 seconds.")
        # Note: actual refresh happens via st.rerun() with a timer

    # Get live prediction
    if HAS_PREDICT:
        try:
            # Load BTC data for cross-asset features
            btc_df = get_live_data("btc", limit=200) if coin.lower() != "btc" else None

            result = predict_with_threshold(
                coin,
                btc_df=btc_df,
                confidence_threshold=DEFAULT_THRESHOLD,
                calibration=DEFAULT_CALIBRATION
            )

            # Display current price and prediction
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    label=f"{coin.upper()} Price",
                    value=f"${result['price']:,.2f}",
                )

            with col2:
                direction_emoji = "🔼" if result["direction"] == "UP" else "🔽"
                st.metric(
                    label="Prediction",
                    value=f"{direction_emoji} {result['direction']}",
                    delta=f"{result['calibrated_prob_up']*100:.1f}% up"
                )

            with col3:
                st.metric(
                    label="Confidence",
                    value=result['confidence_pct'],
                )

            with col4:
                st.metric(
                    label="Timestamp",
                    value=result['timestamp'].strftime("%H:%M") if hasattr(result['timestamp'], 'strftime') else str(result['timestamp'])[:16]
                )

            # SELECTIVE BETTING RECOMMENDATION
            st.divider()
            st.subheader("Betting Recommendation")

            threshold_acc = load_threshold_accuracy(
                coin.lower(),
                threshold=DEFAULT_THRESHOLD,
                calibration=DEFAULT_CALIBRATION
            ) or 0.65

            if result["action"] == "trade":
                st.success(f"### ✅ BET {result['direction']}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Confidence", result['confidence_pct'])
                with col2:
                    st.metric("Threshold", f"{DEFAULT_THRESHOLD*100:.0f}%")
                with col3:
                    st.metric("Expected Accuracy", f"{threshold_acc*100:.1f}%")
                st.info(f"Model confidence exceeds threshold. Historical accuracy at this confidence level: **{threshold_acc*100:.1f}%**")
            else:
                st.warning("### ⏸️ SKIP - Low Confidence")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Confidence", result['confidence_pct'])
                with col2:
                    st.metric("Threshold", f"{DEFAULT_THRESHOLD*100:.0f}%")
                with col3:
                    st.metric("Raw Accuracy", "~53%")
                st.info("Confidence below 58% threshold. Wait for a higher-confidence signal to improve expected accuracy.")

            st.divider()

            # Prediction gauge
            st.subheader("Prediction Confidence Gauge")
            gauge_fig = create_prediction_gauge(result["calibrated_prob_up"], result["direction"])
            st.plotly_chart(gauge_fig, use_container_width=True)

            # Model breakdown
            st.subheader("Model Details")
            meta = load_meta_v2(coin)
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Model Version:** {result.get('model_version', 'v2')}")
                st.write(f"**Trained:** {meta.get('trained_at', 'Unknown')[:10] if meta.get('trained_at') else 'Unknown'}")
                st.write(f"**Raw Accuracy:** {meta.get('test_accuracy', 0)*100:.1f}%")
            with col2:
                st.write(f"**Threshold Accuracy:** {threshold_acc*100:.1f}%")
                st.write(f"**Features:** {len(meta.get('features', []))}")
                st.write(f"**Calibration:** {DEFAULT_CALIBRATION}")

        except Exception as e:
            st.error(f"Error making prediction: {e}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.warning("Prediction module not available. Train models first.")

    # Kalshi Contracts Section
    st.divider()
    st.subheader("Kalshi Contract Recommendations")

    if HAS_KALSHI and coin.upper() in KALSHI_TICKERS:
        try:
            event = get_next_hourly_event(coin.upper())
            if event:
                st.write(f"**Next Settlement:** {event.get('sub_title', 'Unknown')}")

                markets = get_markets_for_event(event.get("event_ticker"))
                if markets:
                    structure = parse_market_structure(markets)
                    current_price = result["price"] if HAS_PREDICT else 0

                    # Display contracts as a table
                    contracts_data = []

                    # Above strikes
                    for strike in structure.get("above_strikes", [])[:3]:
                        strike_val = strike.get('floor_strike', 0)
                        contracts_data.append({
                            "Type": "Above",
                            "Strike": f"${strike_val:,.0f}",
                            "Meaning": f"{coin.upper()} will be ABOVE ${strike_val:,.0f} at settlement",
                            "Ticker": strike.get("ticker", ""),
                            "Yes Ask": f"${strike.get('yes_ask', 0):.2f}",
                            "ROI if Win": f"{(1/strike['yes_ask']-1)*100:.0f}%" if strike.get('yes_ask', 0) > 0 else "N/A"
                        })

                    # Below strikes
                    for strike in structure.get("below_strikes", [])[:3]:
                        strike_val = strike.get('cap_strike', 0)
                        contracts_data.append({
                            "Type": "Below",
                            "Strike": f"${strike_val:,.0f}",
                            "Meaning": f"{coin.upper()} will be BELOW ${strike_val:,.0f} at settlement",
                            "Ticker": strike.get("ticker", ""),
                            "Yes Ask": f"${strike.get('yes_ask', 0):.2f}",
                            "ROI if Win": f"{(1/strike['yes_ask']-1)*100:.0f}%" if strike.get('yes_ask', 0) > 0 else "N/A"
                        })

                    if contracts_data:
                        df_contracts = pd.DataFrame(contracts_data)
                        st.dataframe(df_contracts, use_container_width=True, hide_index=True)

                        # Recommendation
                        if HAS_PREDICT and result["meets_threshold"]:
                            rec_type = "Above" if result["direction"] == "UP" else "Below"
                            st.success(f"**Recommended:** Look at {rec_type} contracts based on model prediction")
                    else:
                        st.info("No contracts available at this time")
                else:
                    st.info("No markets available for this event")
            else:
                st.info("No upcoming hourly contracts found")
        except Exception as e:
            st.warning(f"Could not fetch Kalshi data: {e}")
    elif coin.upper() not in ["BTC", "ETH"]:
        st.info(f"Kalshi only offers contracts for BTC and ETH, not {coin.upper()}")
    else:
        st.warning("Kalshi API not available")

    # Kalshi EV Summary
    st.divider()
    st.subheader("Kalshi EV Summary")
    results_df = load_kalshi_results()
    signals_df = load_kalshi_signals()

    if not results_df.empty:
        settled = len(results_df)
        win_rate = results_df["won"].mean() if "won" in results_df.columns else 0
        total_pnl = results_df["pnl"].sum() if "pnl" in results_df.columns else 0
        avg_pnl = results_df["pnl"].mean() if "pnl" in results_df.columns else 0
        avg_ev = results_df["ev_per_contract"].mean() if "ev_per_contract" in results_df.columns else 0

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Settled Trades", settled)
        col2.metric("Win Rate", f"{win_rate*100:.1f}%")
        col3.metric("Total PnL", f"${total_pnl:.2f}")
        col4.metric("Avg PnL", f"${avg_pnl:.3f}")
        col5.metric("Avg EV/Contract", f"${avg_ev:.3f}")

        st.dataframe(results_df.tail(20), use_container_width=True, hide_index=True)
    elif not signals_df.empty:
        st.info("Signals logged but no settled trades yet.")
        st.dataframe(signals_df.tail(20), use_container_width=True, hide_index=True)
    else:
        st.info("No Kalshi signals found. Run kalshi_backtest.py to log signals.")


# =============================================================================
# Tab 2: Price Charts & Technicals
# =============================================================================

def render_price_charts():
    """Render the price charts and technicals tab."""
    st.header("Price Charts & Technical Analysis")

    # Controls
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        coin = st.selectbox("Coin", list(SYMBOLS.keys()), format_func=str.upper, key="chart_coin")
    with col2:
        timeframe = st.selectbox("Timeframe", ["7 days", "30 days", "90 days", "1 year"], index=1)
    with col3:
        show_sma = st.toggle("Show SMA", value=True)
    with col4:
        show_bb = st.toggle("Show Bollinger Bands", value=False)

    # Calculate limit based on timeframe
    timeframe_hours = {
        "7 days": 24 * 7,
        "30 days": 24 * 30,
        "90 days": 24 * 90,
        "1 year": 24 * 365
    }
    limit = timeframe_hours.get(timeframe, 24 * 30)

    # Load data
    df = load_price_data(coin, limit=limit)

    if df.empty:
        st.warning("No data available")
        return

    # Add technical indicators
    df = prepare_features(df, lookahead=1, include_fgi=False, include_onchain=False)

    # Candlestick chart
    st.subheader(f"{coin.upper()} Price Chart")
    candle_fig = create_candlestick_chart(df, coin, show_sma=show_sma, show_bb=show_bb)
    st.plotly_chart(candle_fig, use_container_width=True)

    # Technical indicators in columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("RSI (14)")
        rsi_fig = create_rsi_chart(df)
        st.plotly_chart(rsi_fig, use_container_width=True)

        # Current RSI value
        if "rsi_14" in df.columns:
            current_rsi = df["rsi_14"].iloc[-1]
            if current_rsi > 70:
                st.warning(f"RSI: {current_rsi:.1f} - Overbought")
            elif current_rsi < 30:
                st.success(f"RSI: {current_rsi:.1f} - Oversold")
            else:
                st.info(f"RSI: {current_rsi:.1f} - Neutral")

    with col2:
        st.subheader("MACD")
        macd_fig = create_macd_chart(df)
        st.plotly_chart(macd_fig, use_container_width=True)

        # MACD signal
        if "macd" in df.columns and "macd_signal" in df.columns:
            macd = df["macd"].iloc[-1]
            signal = df["macd_signal"].iloc[-1]
            if macd > signal:
                st.success("MACD above signal line - Bullish")
            else:
                st.warning("MACD below signal line - Bearish")

    # Price statistics
    st.subheader("Price Statistics")
    col1, col2, col3, col4 = st.columns(4)

    current_price = df["close"].iloc[-1]
    high_24h = df["high"].tail(24).max()
    low_24h = df["low"].tail(24).min()
    change_24h = (current_price / df["close"].iloc[-25] - 1) * 100 if len(df) > 25 else 0

    with col1:
        st.metric("Current Price", f"${current_price:,.2f}")
    with col2:
        st.metric("24h High", f"${high_24h:,.2f}")
    with col3:
        st.metric("24h Low", f"${low_24h:,.2f}")
    with col4:
        st.metric("24h Change", f"{change_24h:+.2f}%")


# =============================================================================
# Tab 3: On-Chain & Sentiment
# =============================================================================

def render_onchain_sentiment():
    """Render the on-chain metrics and sentiment tab."""
    st.header("On-Chain Metrics & Sentiment")

    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        days_back = st.selectbox("Time Period", ["30 days", "90 days", "180 days", "1 year"], index=1)
    with col2:
        st.info("On-chain metrics are BTC-specific")

    days_map = {"30 days": 30, "90 days": 90, "180 days": 180, "1 year": 365}
    num_days = days_map.get(days_back, 90)

    # Fear & Greed Index
    st.subheader("Fear & Greed Index")
    fgi_df = load_fgi_data()

    if not fgi_df.empty:
        fgi_df = fgi_df.tail(num_days)

        # Current value badge
        if "fgi_value" in fgi_df.columns or "value" in fgi_df.columns:
            value_col = "fgi_value" if "fgi_value" in fgi_df.columns else "value"
            current_fgi = fgi_df[value_col].iloc[-1]

            if current_fgi <= 25:
                sentiment = "Extreme Fear"
                color = "red"
            elif current_fgi <= 45:
                sentiment = "Fear"
                color = "orange"
            elif current_fgi <= 55:
                sentiment = "Neutral"
                color = "gray"
            elif current_fgi <= 75:
                sentiment = "Greed"
                color = "lightgreen"
            else:
                sentiment = "Extreme Greed"
                color = "green"

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Value", f"{current_fgi:.0f}")
            with col2:
                st.metric("Sentiment", sentiment)
            with col3:
                if len(fgi_df) > 7:
                    change_7d = current_fgi - fgi_df[value_col].iloc[-8]
                    st.metric("7-Day Change", f"{change_7d:+.0f}")

        # FGI chart
        # Rename column for chart if needed
        if "value" in fgi_df.columns and "fgi_value" not in fgi_df.columns:
            fgi_df = fgi_df.rename(columns={"value": "fgi_value"})

        fgi_fig = create_fgi_timeline(fgi_df)
        st.plotly_chart(fgi_fig, use_container_width=True)
    else:
        st.warning("Fear & Greed data not available. Run fear_greed.py to fetch data.")

    st.divider()

    # On-chain metrics
    st.subheader("On-Chain Metrics (BTC)")
    onchain_df = load_onchain_data()

    if not onchain_df.empty:
        onchain_df = onchain_df.tail(num_days)

        col1, col2 = st.columns(2)

        with col1:
            # MVRV Z-Score
            st.write("**MVRV Z-Score**")
            if "mvrv_zscore" in onchain_df.columns or "mvrvZscore" in onchain_df.columns:
                mvrv_fig = create_mvrv_chart(onchain_df)
                st.plotly_chart(mvrv_fig, use_container_width=True)

                col_name = "mvrv_zscore" if "mvrv_zscore" in onchain_df.columns else "mvrvZscore"
                current_mvrv = onchain_df[col_name].iloc[-1]
                if current_mvrv > 7:
                    st.error(f"MVRV: {current_mvrv:.2f} - Extreme (historically signals tops)")
                elif current_mvrv > 2.5:
                    st.warning(f"MVRV: {current_mvrv:.2f} - Overbought zone")
                elif current_mvrv < 0:
                    st.success(f"MVRV: {current_mvrv:.2f} - Undervalued zone")
                else:
                    st.info(f"MVRV: {current_mvrv:.2f} - Neutral")
            else:
                st.info("MVRV data not available")

        with col2:
            # SOPR
            st.write("**SOPR (Spent Output Profit Ratio)**")
            if "sopr" in onchain_df.columns:
                sopr_fig = create_sopr_chart(onchain_df)
                st.plotly_chart(sopr_fig, use_container_width=True)

                current_sopr = onchain_df["sopr"].iloc[-1]
                if current_sopr > 1.05:
                    st.success(f"SOPR: {current_sopr:.3f} - Holders taking profits")
                elif current_sopr < 0.95:
                    st.error(f"SOPR: {current_sopr:.3f} - Holders selling at loss")
                else:
                    st.info(f"SOPR: {current_sopr:.3f} - Near break-even")
            else:
                st.info("SOPR data not available")

        # Active Addresses
        if "active_addresses" in onchain_df.columns:
            st.write("**Active Addresses**")
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=onchain_df.index,
                y=onchain_df["active_addresses"],
                name="Active Addresses",
                line=dict(color="#1565C0", width=2)
            ))
            fig.update_layout(
                height=300,
                title="Daily Active Addresses",
                margin=dict(l=50, r=50, t=50, b=30)
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("On-chain data not available. Run onchain.py to fetch data.")


# =============================================================================
# Tab 4: Model Performance
# =============================================================================

def render_model_performance():
    """Render the model performance tab."""
    st.header("Model Performance Analysis")

    # Model selector
    coin = st.selectbox("Select Model", list(SYMBOLS.keys()), format_func=str.upper, key="perf_coin")

    # Load model metadata
    meta_path = MODELS_DIR / f"{coin}_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

        # Model info cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Test Accuracy", f"{meta.get('test_accuracy', 0)*100:.1f}%")
        with col2:
            st.metric("Test AUC", f"{meta.get('test_auc', 0):.4f}")
        with col3:
            st.metric("Model Type", meta.get('model_type', 'xgboost').upper())
        with col4:
            st.metric("Features", len(meta.get('features', [])))

        st.divider()

        # Feature Importance
        st.subheader("Feature Importance")
        model = load_cached_model(coin)

        if model is not None:
            try:
                # Get feature importance from XGBoost model
                if hasattr(model, 'models'):  # Ensemble
                    xgb_model = model.models.get('xgb')
                    if xgb_model and hasattr(xgb_model, 'feature_importances_'):
                        features = meta.get('features', [])
                        importances = dict(zip(features, xgb_model.feature_importances_))
                        fig = create_feature_importance(importances, top_n=15)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Feature importance not available for this model")
                elif hasattr(model, 'feature_importances_'):
                    features = meta.get('features', [])
                    importances = dict(zip(features, model.feature_importances_))
                    fig = create_feature_importance(importances, top_n=15)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Feature importance not available for this model type")
            except Exception as e:
                st.warning(f"Could not extract feature importance: {e}")

        st.divider()

        # Model configuration
        st.subheader("Model Configuration")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Training Info:**")
            st.write(f"- Trained: {meta.get('trained_at', 'Unknown')[:19] if meta.get('trained_at') else 'Unknown'}")
            st.write(f"- Include FGI: {'Yes' if meta.get('include_fgi') else 'No'}")
            st.write(f"- Include Cross-Asset: {'Yes' if meta.get('include_cross_asset') else 'No'}")
            st.write(f"- Include On-Chain: {'Yes' if meta.get('include_onchain') else 'No'}")

        with col2:
            st.write("**Models in Ensemble:**")
            for m in meta.get('models', ['xgb']):
                st.write(f"- {m.upper()}")

        # Feature list
        with st.expander("All Features Used"):
            features = meta.get('features', [])
            col1, col2, col3 = st.columns(3)
            third = len(features) // 3
            with col1:
                for f in features[:third]:
                    st.write(f"- {f}")
            with col2:
                for f in features[third:2*third]:
                    st.write(f"- {f}")
            with col3:
                for f in features[2*third:]:
                    st.write(f"- {f}")

    else:
        st.warning(f"No model metadata found for {coin.upper()}. Train the model first.")

    st.divider()

    # All models comparison
    st.subheader("All Models Comparison")
    comparison_data = []

    for c in SYMBOLS.keys():
        meta_path = MODELS_DIR / f"{c}_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                m = json.load(f)
                comparison_data.append({
                    "Coin": c.upper(),
                    "Accuracy": f"{m.get('test_accuracy', 0)*100:.1f}%",
                    "AUC": f"{m.get('test_auc', 0):.4f}",
                    "Type": m.get('model_type', 'xgb').upper(),
                    "Features": len(m.get('features', [])),
                    "FGI": "Yes" if m.get('include_fgi') else "No"
                })

    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)
    else:
        st.info("No trained models found. Run train.py or ensemble.py first.")


# =============================================================================
# Main App
# =============================================================================

def main():
    """Main Streamlit app."""
    # Sidebar
    with st.sidebar:
        st.title("Crypto Kalshi Predictor")
        st.markdown("---")

        st.write("**Quick Stats**")

        # Show model status
        models_found = 0
        for coin in SYMBOLS.keys():
            if (MODELS_DIR / f"{coin}_ensemble.joblib").exists() or (MODELS_DIR / f"{coin}_xgb.joblib").exists():
                models_found += 1
        st.write(f"Models Trained: {models_found}/{len(SYMBOLS)}")

        # Current time
        st.write(f"Updated: {datetime.now().strftime('%H:%M:%S')}")

        st.markdown("---")
        st.write("**Navigation**")
        st.write("Use tabs above to switch views")

        st.markdown("---")
        st.caption("Built with Streamlit & Plotly")
        st.caption("Models: XGBoost + LightGBM + CatBoost")

    # Main content with tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Live Predictions",
        "📈 Price Charts",
        "⛓️ On-Chain & Sentiment",
        "🎯 Model Performance"
    ])

    with tab1:
        render_live_predictions()

    with tab2:
        render_price_charts()

    with tab3:
        render_onchain_sentiment()

    with tab4:
        render_model_performance()


if __name__ == "__main__":
    main()
