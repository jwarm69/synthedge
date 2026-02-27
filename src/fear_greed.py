"""
Fear & Greed Index API wrapper.
Free API from alternative.me - daily sentiment data.
"""

import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

API_URL = "https://api.alternative.me/fng/"
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def fetch_fear_greed(limit: int = 365) -> pd.DataFrame:
    """
    Fetch Fear & Greed Index data.

    Args:
        limit: Number of days to fetch (max ~2000)

    Returns:
        DataFrame with columns: timestamp, value, value_classification
    """
    params = {
        "limit": limit,
        "format": "json"
    }

    response = requests.get(API_URL, params=params)
    response.raise_for_status()

    data = response.json()

    if "data" not in data:
        raise ValueError(f"API Error: {data}")

    df = pd.DataFrame(data["data"])

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
    df["value"] = df["value"].astype(int)

    # Set index
    df.set_index("timestamp", inplace=True)
    df = df.sort_index()

    # Keep essential columns
    df = df[["value", "value_classification"]]
    df.columns = ["fgi_value", "fgi_class"]

    print(f"Fetched {len(df)} days of Fear & Greed data ({df.index.min().date()} to {df.index.max().date()})")

    return df


def save_fear_greed(df: pd.DataFrame) -> Path:
    """Save Fear & Greed data to parquet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DATA_DIR / "fear_greed.parquet"
    df.to_parquet(filepath)
    print(f"Saved to {filepath}")
    return filepath


def load_fear_greed() -> pd.DataFrame:
    """Load cached Fear & Greed data."""
    filepath = DATA_DIR / "fear_greed.parquet"
    if not filepath.exists():
        raise FileNotFoundError("No cached FGI data. Run fetch_fear_greed first.")
    return pd.read_parquet(filepath)


def merge_fgi_with_hourly(
    hourly_df: pd.DataFrame,
    fgi_df: pd.DataFrame = None,
    availability_lag_hours: int = 24
) -> pd.DataFrame:
    """
    Merge daily Fear & Greed Index with hourly price data.

    Uses a lagged as-of merge to avoid lookahead leakage.
    With a 24h lag, each day only sees the previous day's FGI value.

    Args:
        hourly_df: Hourly OHLCV DataFrame with datetime index
        fgi_df: Fear & Greed DataFrame (will fetch if None)
        availability_lag_hours: Hours to lag daily data before it is available

    Returns:
        Hourly DataFrame with FGI features added
    """
    if fgi_df is None:
        try:
            fgi_df = load_fear_greed()
        except FileNotFoundError:
            fgi_df = fetch_fear_greed(limit=400)
            save_fear_greed(fgi_df)

    df = hourly_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Normalize FGI index to date-only and apply availability lag.
    fgi_daily = fgi_df.copy()
    fgi_daily.index = pd.to_datetime(fgi_daily.index).normalize()
    if availability_lag_hours:
        fgi_daily.index = fgi_daily.index + pd.Timedelta(hours=availability_lag_hours)
    fgi_daily = fgi_daily.sort_index()

    # As-of merge to ensure only past available values are used.
    hourly = df.reset_index().rename(columns={"index": "timestamp"})
    daily = fgi_daily.reset_index().rename(columns={"index": "fgi_timestamp"})
    merged = pd.merge_asof(
        hourly.sort_values("timestamp"),
        daily.sort_values("fgi_timestamp")[["fgi_timestamp", "fgi_value"]],
        left_on="timestamp",
        right_on="fgi_timestamp",
        direction="backward"
    )
    merged = merged.drop(columns=["fgi_timestamp"]).set_index("timestamp")
    df = merged

    # Add derived features
    # 7-day change in FGI
    df["fgi_change_7d"] = df["fgi_value"].diff(24 * 7)  # 7 days in hours

    # FGI zones (binary features)
    df["fgi_extreme_fear"] = (df["fgi_value"] <= 25).astype(int)
    df["fgi_fear"] = ((df["fgi_value"] > 25) & (df["fgi_value"] <= 45)).astype(int)
    df["fgi_neutral"] = ((df["fgi_value"] > 45) & (df["fgi_value"] <= 55)).astype(int)
    df["fgi_greed"] = ((df["fgi_value"] > 55) & (df["fgi_value"] <= 75)).astype(int)
    df["fgi_extreme_greed"] = (df["fgi_value"] > 75).astype(int)

    # Simplified zones for model
    df["fgi_is_fear"] = (df["fgi_value"] < 45).astype(int)
    df["fgi_is_greed"] = (df["fgi_value"] > 55).astype(int)

    # Clean up - drop _date if it exists
    if "_date" in df.columns:
        df.drop(columns=["_date"], inplace=True)

    return df


def get_fgi_feature_columns() -> list:
    """Return list of FGI feature column names."""
    return [
        "fgi_value",
        "fgi_change_7d",
        "fgi_is_fear",
        "fgi_is_greed",
    ]


if __name__ == "__main__":
    # Test
    df = fetch_fear_greed(limit=365)
    print(df.head(10))
    print(f"\nCurrent FGI: {df.iloc[-1]['fgi_value']} ({df.iloc[-1]['fgi_class']})")
    save_fear_greed(df)
