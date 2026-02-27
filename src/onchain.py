"""
On-chain metrics fetcher using free APIs.
Sources:
- BGeometrics (bitcoin-data.com) - MVRV, SOPR, Active Addresses, Funding Rate
- Blockchain.info - Transaction count, Hash rate
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def fetch_bgeometrics(endpoint: str, cache_hours: int = 6) -> pd.DataFrame:
    """
    Fetch data from BGeometrics API with caching.

    Args:
        endpoint: API endpoint name (e.g., "mvrv-zscore", "sopr")
        cache_hours: Hours to cache data before refetching

    Returns:
        DataFrame with date index and metric column
    """
    cache_file = DATA_DIR / f"onchain_{endpoint}.parquet"

    # Check cache
    if cache_file.exists():
        cache_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - cache_mtime < timedelta(hours=cache_hours):
            return pd.read_parquet(cache_file)

    url = f"https://bitcoin-data.com/v1/{endpoint}"

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        df = pd.DataFrame(data)

        # Standardize column names
        if "d" in df.columns:
            df["date"] = pd.to_datetime(df["d"])
            df = df.drop(columns=["d"])
        if "unixTs" in df.columns:
            df = df.drop(columns=["unixTs"])

        df = df.set_index("date")

        # Convert all columns to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Cache the data
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file)

        return df

    except Exception as e:
        print(f"Error fetching {endpoint}: {e}")
        # Try to load from cache even if stale
        if cache_file.exists():
            return pd.read_parquet(cache_file)
        return pd.DataFrame()


def get_mvrv_zscore() -> pd.DataFrame:
    """
    Fetch MVRV Z-Score.

    MVRV = Market Value / Realized Value
    Z-Score normalizes MVRV to identify overbought/oversold conditions.

    High Z-Score (>7): Market is overheated, potential top
    Low Z-Score (<0): Market is undervalued, potential bottom
    """
    return fetch_bgeometrics("mvrv-zscore")


def get_sopr() -> pd.DataFrame:
    """
    Fetch Spent Output Profit Ratio (SOPR).

    SOPR = Realized Value / Value at Creation for spent outputs

    SOPR > 1: Coins moved at profit (bullish in uptrend)
    SOPR < 1: Coins moved at loss (bearish, capitulation)
    SOPR = 1: Break-even, often acts as support in bull markets
    """
    return fetch_bgeometrics("sopr")


def get_active_addresses() -> pd.DataFrame:
    """
    Fetch daily active addresses.

    Rising active addresses with rising price = healthy trend
    Falling active addresses with rising price = divergence (bearish)
    """
    return fetch_bgeometrics("active-addresses")


def get_funding_rate() -> pd.DataFrame:
    """
    Fetch perpetual futures funding rate.

    High positive funding (>0.1%): Longs paying shorts, crowded long
    High negative funding (<-0.1%): Shorts paying longs, crowded short
    Extreme funding often precedes reversals.
    """
    return fetch_bgeometrics("funding-rate")


def get_nupl() -> pd.DataFrame:
    """
    Fetch Net Unrealized Profit/Loss (NUPL).

    NUPL = (Market Cap - Realized Cap) / Market Cap

    >0.75: Euphoria (potential top)
    0.5-0.75: Greed
    0.25-0.5: Optimism
    0-0.25: Hope/Fear
    <0: Capitulation (potential bottom)
    """
    return fetch_bgeometrics("nupl")


def get_puell_multiple() -> pd.DataFrame:
    """
    Fetch Puell Multiple.

    Puell = Daily Coin Issuance (USD) / 365-day MA of Issuance

    >4: Miners selling heavily, potential top
    <0.5: Miners under stress, potential bottom
    """
    return fetch_bgeometrics("puell-multiple")


def get_all_onchain_metrics() -> pd.DataFrame:
    """
    Fetch all on-chain metrics and combine into single DataFrame.
    """
    metrics = {
        "mvrv_zscore": get_mvrv_zscore,
        "sopr": get_sopr,
        "active_addresses": get_active_addresses,
        "funding_rate": get_funding_rate,
        "nupl": get_nupl,
        "puell_multiple": get_puell_multiple,
    }

    dfs = []
    for name, func in metrics.items():
        print(f"Fetching {name}...")
        df = func()
        if not df.empty:
            # Rename columns to include metric name
            df.columns = [f"{name}_{col}" if col != name else name for col in df.columns]
            # If there's only one data column, rename it
            if len(df.columns) == 1:
                df.columns = [name]
            dfs.append(df)
        time.sleep(0.5)  # Be nice to the API

    if not dfs:
        return pd.DataFrame()

    # Merge all metrics on date
    result = dfs[0]
    for df in dfs[1:]:
        result = result.join(df, how="outer")

    return result.sort_index()


def merge_onchain_with_hourly(
    hourly_df: pd.DataFrame,
    onchain_df: pd.DataFrame,
    availability_lag_hours: int = 24
) -> pd.DataFrame:
    """
    Merge daily on-chain metrics with hourly price data.
    Uses a lagged as-of merge to avoid lookahead leakage.

    Args:
        hourly_df: Hourly OHLCV data with datetime index
        onchain_df: Daily on-chain metrics with date index
        availability_lag_hours: Hours to lag daily data before it is available

    Returns:
        Hourly DataFrame with on-chain features added
    """
    if onchain_df.empty:
        return hourly_df

    hourly = hourly_df.copy()
    if not isinstance(hourly.index, pd.DatetimeIndex):
        hourly.index = pd.to_datetime(hourly.index)
    hourly = hourly.sort_index()

    daily = onchain_df.copy()
    daily.index = pd.to_datetime(daily.index).normalize()
    if availability_lag_hours:
        daily.index = daily.index + pd.Timedelta(hours=availability_lag_hours)
    daily = daily.sort_index()

    hourly_reset = hourly.reset_index().rename(columns={"index": "timestamp"})
    daily_reset = daily.reset_index().rename(columns={"index": "onchain_timestamp"})

    merged = pd.merge_asof(
        hourly_reset.sort_values("timestamp"),
        daily_reset.sort_values("onchain_timestamp"),
        left_on="timestamp",
        right_on="onchain_timestamp",
        direction="backward"
    )
    merged = merged.drop(columns=["onchain_timestamp"]).set_index("timestamp")
    return merged


def add_onchain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived on-chain features to DataFrame.

    Args:
        df: DataFrame with raw on-chain metrics

    Returns:
        DataFrame with additional derived features
    """
    df = df.copy()

    # MVRV features
    if "mvrv_zscore" in df.columns or "mvrvZscore" in df.columns:
        mvrv_col = "mvrv_zscore" if "mvrv_zscore" in df.columns else "mvrvZscore"
        df["mvrv_overbought"] = (df[mvrv_col] > 2.5).astype(int)
        df["mvrv_oversold"] = (df[mvrv_col] < 0).astype(int)
        df["mvrv_extreme"] = ((df[mvrv_col] > 4) | (df[mvrv_col] < -0.5)).astype(int)

    # SOPR features
    if "sopr" in df.columns:
        df["sopr_profit"] = (df["sopr"] > 1).astype(int)
        df["sopr_loss"] = (df["sopr"] < 1).astype(int)
        # SOPR momentum
        df["sopr_ma7"] = df["sopr"].rolling(7, min_periods=1).mean()
        df["sopr_above_ma"] = (df["sopr"] > df["sopr_ma7"]).astype(int)

    # Funding rate features
    if "funding_rate" in df.columns or "fundingRate" in df.columns:
        fr_col = "funding_rate" if "funding_rate" in df.columns else "fundingRate"
        df["funding_high_long"] = (df[fr_col] > 0.05).astype(int)  # Crowded long
        df["funding_high_short"] = (df[fr_col] < -0.05).astype(int)  # Crowded short
        df["funding_extreme"] = ((df[fr_col] > 0.1) | (df[fr_col] < -0.1)).astype(int)

    # NUPL features
    if "nupl" in df.columns:
        df["nupl_euphoria"] = (df["nupl"] > 0.75).astype(int)
        df["nupl_capitulation"] = (df["nupl"] < 0).astype(int)

    return df


if __name__ == "__main__":
    print("Fetching all on-chain metrics...")
    df = get_all_onchain_metrics()

    print(f"\nFetched {len(df)} days of data")
    print(f"Columns: {list(df.columns)}")
    print(f"\nLatest values:")
    print(df.tail(5))

    # Save combined data
    output_file = DATA_DIR / "onchain_combined.parquet"
    df.to_parquet(output_file)
    print(f"\nSaved to {output_file}")
