"""
Plotly chart helpers for the Streamlit dashboard.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def create_candlestick_chart(
    df: pd.DataFrame,
    coin: str,
    show_sma: bool = True,
    show_bb: bool = False,
    height: int = 500
) -> go.Figure:
    """
    Create interactive candlestick chart with volume and optional overlays.

    Args:
        df: DataFrame with OHLCV data
        coin: Coin name for title
        show_sma: Show SMA 20/50 overlays
        show_bb: Show Bollinger Bands
        height: Chart height in pixels
    """
    # Create subplots with volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=(f"{coin.upper()} Price", "Volume")
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350"
        ),
        row=1, col=1
    )

    # SMA overlays
    if show_sma and "sma_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["sma_20"],
                name="SMA 20",
                line=dict(color="#2196F3", width=1)
            ),
            row=1, col=1
        )
    if show_sma and "sma_50" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["sma_50"],
                name="SMA 50",
                line=dict(color="#FF9800", width=1)
            ),
            row=1, col=1
        )

    # Bollinger Bands
    if show_bb and "bb_upper" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["bb_upper"],
                name="BB Upper",
                line=dict(color="#9C27B0", width=1, dash="dash"),
                opacity=0.5
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["bb_lower"],
                name="BB Lower",
                line=dict(color="#9C27B0", width=1, dash="dash"),
                fill="tonexty",
                fillcolor="rgba(156, 39, 176, 0.1)",
                opacity=0.5
            ),
            row=1, col=1
        )

    # Volume bars
    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(df["close"], df["open"])]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["volume"],
            name="Volume",
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=height,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=50, b=50)
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128,128,128,0.2)")

    return fig


def create_rsi_chart(df: pd.DataFrame, height: int = 200) -> go.Figure:
    """Create RSI chart with overbought/oversold zones."""
    fig = go.Figure()

    # RSI line
    if "rsi_14" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["rsi_14"],
                name="RSI (14)",
                line=dict(color="#673AB7", width=2)
            )
        )

    # Overbought/oversold zones
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5)

    # Shaded zones
    fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, line_width=0)

    fig.update_layout(
        height=height,
        title="RSI (14)",
        yaxis=dict(range=[0, 100]),
        showlegend=False,
        margin=dict(l=50, r=50, t=40, b=30)
    )

    return fig


def create_macd_chart(df: pd.DataFrame, height: int = 200) -> go.Figure:
    """Create MACD chart with histogram and signal line."""
    fig = go.Figure()

    if "macd" in df.columns:
        # MACD histogram
        colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["macd_hist"]]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["macd_hist"],
                name="Histogram",
                marker_color=colors,
                opacity=0.7
            )
        )

        # MACD line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["macd"],
                name="MACD",
                line=dict(color="#2196F3", width=1.5)
            )
        )

        # Signal line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["macd_signal"],
                name="Signal",
                line=dict(color="#FF9800", width=1.5)
            )
        )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        height=height,
        title="MACD (12, 26, 9)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=50, t=40, b=30)
    )

    return fig


def create_fgi_timeline(df: pd.DataFrame, height: int = 300) -> go.Figure:
    """Create Fear & Greed Index timeline with color gradient."""
    fig = go.Figure()

    if "fgi_value" in df.columns:
        # Color based on FGI value
        colors = []
        for v in df["fgi_value"]:
            if v <= 25:
                colors.append("#d32f2f")  # Extreme Fear - dark red
            elif v <= 45:
                colors.append("#f57c00")  # Fear - orange
            elif v <= 55:
                colors.append("#ffc107")  # Neutral - yellow
            elif v <= 75:
                colors.append("#8bc34a")  # Greed - light green
            else:
                colors.append("#2e7d32")  # Extreme Greed - dark green

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["fgi_value"],
                mode="lines+markers",
                name="Fear & Greed",
                line=dict(color="#555", width=1),
                marker=dict(color=colors, size=4)
            )
        )

    # Zone lines
    fig.add_hline(y=25, line_dash="dot", line_color="red", opacity=0.5)
    fig.add_hline(y=45, line_dash="dot", line_color="orange", opacity=0.5)
    fig.add_hline(y=55, line_dash="dot", line_color="green", opacity=0.5)
    fig.add_hline(y=75, line_dash="dot", line_color="darkgreen", opacity=0.5)

    # Zone shading
    fig.add_hrect(y0=0, y1=25, fillcolor="red", opacity=0.1, line_width=0,
                  annotation_text="Extreme Fear", annotation_position="left")
    fig.add_hrect(y0=75, y1=100, fillcolor="green", opacity=0.1, line_width=0,
                  annotation_text="Extreme Greed", annotation_position="left")

    fig.update_layout(
        height=height,
        title="Fear & Greed Index",
        yaxis=dict(range=[0, 100]),
        showlegend=False,
        margin=dict(l=50, r=50, t=40, b=30)
    )

    return fig


def create_mvrv_chart(df: pd.DataFrame, height: int = 300) -> go.Figure:
    """Create MVRV Z-Score chart with threshold zones."""
    fig = go.Figure()

    col = "mvrv_zscore" if "mvrv_zscore" in df.columns else "mvrvZscore"
    if col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                name="MVRV Z-Score",
                line=dict(color="#1565C0", width=2)
            )
        )

    # Threshold zones
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=2.5, line_dash="dash", line_color="orange", annotation_text="Overbought")
    fig.add_hline(y=7, line_dash="dash", line_color="red", annotation_text="Extreme")

    # Shading
    fig.add_hrect(y0=2.5, y1=10, fillcolor="red", opacity=0.1, line_width=0)
    fig.add_hrect(y0=-2, y1=0, fillcolor="green", opacity=0.1, line_width=0)

    fig.update_layout(
        height=height,
        title="MVRV Z-Score (BTC)",
        showlegend=False,
        margin=dict(l=50, r=50, t=40, b=30)
    )

    return fig


def create_sopr_chart(df: pd.DataFrame, height: int = 300) -> go.Figure:
    """Create SOPR chart with profit/loss shading."""
    fig = go.Figure()

    if "sopr" in df.columns:
        # Fill above/below 1
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["sopr"],
                name="SOPR",
                line=dict(color="#1565C0", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(21, 101, 192, 0.2)"
            )
        )

        # 7-day MA if available
        if "sopr_ma7" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["sopr_ma7"],
                    name="7-day MA",
                    line=dict(color="#FF9800", width=1.5, dash="dash")
                )
            )

    # Profit/loss line
    fig.add_hline(y=1, line_dash="dash", line_color="gray",
                  annotation_text="Break-even")

    fig.update_layout(
        height=height,
        title="SOPR (Spent Output Profit Ratio)",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=50, t=40, b=30)
    )

    return fig


def create_prediction_gauge(prob_up: float, direction: str) -> go.Figure:
    """Create a gauge showing prediction confidence."""
    # Calculate gauge value (0-100 scale)
    confidence = abs(prob_up - 0.5) * 2 * 100

    # Color based on direction
    if direction == "UP":
        bar_color = "#26a69a"
        gauge_color = "green"
    else:
        bar_color = "#ef5350"
        gauge_color = "red"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob_up * 100,
        title={"text": f"Prediction: {direction}"},
        delta={"reference": 50, "valueformat": ".1f"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": bar_color},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "gray",
            "steps": [
                {"range": [0, 30], "color": "rgba(239, 83, 80, 0.3)"},
                {"range": [30, 50], "color": "rgba(239, 83, 80, 0.1)"},
                {"range": [50, 70], "color": "rgba(38, 166, 154, 0.1)"},
                {"range": [70, 100], "color": "rgba(38, 166, 154, 0.3)"}
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": prob_up * 100
            }
        }
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=30, r=30, t=50, b=30)
    )

    return fig


def create_feature_importance(importances: dict, top_n: int = 15) -> go.Figure:
    """Create horizontal bar chart of feature importances."""
    # Sort and get top N
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features = [f[0] for f in sorted_features]
    values = [f[1] for f in sorted_features]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker_color="#1565C0"
    ))

    fig.update_layout(
        height=400,
        title=f"Top {top_n} Feature Importances",
        xaxis_title="Importance",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=150, r=50, t=50, b=50)
    )

    return fig


def create_accuracy_chart(accuracy_data: pd.DataFrame, height: int = 300) -> go.Figure:
    """Create rolling accuracy chart over time."""
    fig = go.Figure()

    for coin in accuracy_data.columns:
        fig.add_trace(
            go.Scatter(
                x=accuracy_data.index,
                y=accuracy_data[coin] * 100,
                name=coin.upper(),
                mode="lines"
            )
        )

    # 50% baseline
    fig.add_hline(y=50, line_dash="dash", line_color="gray",
                  annotation_text="Random Baseline")

    fig.update_layout(
        height=height,
        title="30-Day Rolling Accuracy",
        yaxis_title="Accuracy (%)",
        yaxis=dict(range=[40, 70]),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=50, t=50, b=30)
    )

    return fig


def create_prediction_histogram(predictions: pd.Series, height: int = 250) -> go.Figure:
    """Create histogram of prediction probabilities."""
    fig = go.Figure(go.Histogram(
        x=predictions,
        nbinsx=30,
        marker_color="#1565C0",
        opacity=0.7
    ))

    fig.add_vline(x=0.5, line_dash="dash", line_color="red",
                  annotation_text="Decision Boundary")

    fig.update_layout(
        height=height,
        title="Prediction Distribution",
        xaxis_title="Probability of UP",
        yaxis_title="Count",
        xaxis=dict(range=[0, 1]),
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig
