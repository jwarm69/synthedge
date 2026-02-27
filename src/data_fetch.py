"""
Data fetching from CryptoCompare API (US-friendly, free tier).
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time

# CryptoCompare API (free, 100k calls/month)
BASE_URL = "https://min-api.cryptocompare.com/data/v2/histohour"

SYMBOLS = {
    "btc": "BTC",
    "eth": "ETH",
    "doge": "DOGE",
    "zec": "ZEC",
}

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def fetch_hourly_data(symbol: str, limit: int = 2000, to_ts: int = None) -> pd.DataFrame:
    """
    Fetch hourly OHLCV data from CryptoCompare.

    Args:
        symbol: Crypto symbol (BTC, ETH, etc.)
        limit: Number of hours (max 2000 per request)
        to_ts: End timestamp (optional)

    Returns:
        DataFrame with OHLCV data
    """
    params = {
        "fsym": symbol,
        "tsym": "USD",
        "limit": min(limit, 2000),
    }
    if to_ts:
        params["toTs"] = to_ts

    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()

    data = response.json()

    if data["Response"] != "Success":
        raise ValueError(f"API Error: {data.get('Message', 'Unknown error')}")

    df = pd.DataFrame(data["Data"]["Data"])

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("timestamp", inplace=True)

    # Rename columns to standard OHLCV
    df = df.rename(columns={
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volumefrom": "volume",
    })

    # Keep only essential columns
    df = df[["open", "high", "low", "close", "volume"]]

    # Add trades column (not available from CryptoCompare, use placeholder)
    df["trades"] = 0

    return df


def fetch_historical_data(coin: str, months: int = 6) -> pd.DataFrame:
    """
    Fetch historical hourly data for a coin.

    Args:
        coin: One of 'btc', 'eth', 'doge', 'zec'
        months: Number of months of history to fetch

    Returns:
        DataFrame with OHLCV data
    """
    symbol = SYMBOLS.get(coin.lower())
    if not symbol:
        raise ValueError(f"Unknown coin: {coin}. Valid options: {list(SYMBOLS.keys())}")

    hours_needed = months * 30 * 24  # Approximate hours
    all_data = []

    print(f"Fetching {symbol} data ({months} months ≈ {hours_needed} hours)...")

    end_ts = None
    fetched = 0

    while fetched < hours_needed:
        batch_size = min(2000, hours_needed - fetched)
        df = fetch_hourly_data(symbol, limit=batch_size, to_ts=end_ts)

        if df.empty:
            break

        all_data.append(df)
        fetched += len(df)

        # Move end timestamp back for next batch
        end_ts = int(df.index.min().timestamp()) - 1

        print(f"  Fetched {fetched} candles...")

        # Rate limiting
        time.sleep(0.2)

    if not all_data:
        raise ValueError(f"No data fetched for {symbol}")

    # Combine all batches
    result = pd.concat(all_data)
    result = result.sort_index()
    result = result[~result.index.duplicated(keep="first")]

    print(f"  Total: {len(result)} hourly candles ({result.index.min()} to {result.index.max()})")

    return result


def save_data(df: pd.DataFrame, coin: str) -> Path:
    """Save DataFrame to parquet file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DATA_DIR / f"{coin.lower()}_1h.parquet"
    df.to_parquet(filepath)
    print(f"Saved to {filepath}")
    return filepath


def load_data(coin: str) -> pd.DataFrame:
    """Load cached data from parquet file."""
    filepath = DATA_DIR / f"{coin.lower()}_1h.parquet"
    if not filepath.exists():
        raise FileNotFoundError(f"No cached data for {coin}. Run fetch_historical_data first.")
    return pd.read_parquet(filepath)


def fetch_and_cache_all(months: int = 6) -> dict:
    """Fetch and cache data for all coins."""
    results = {}
    for coin in SYMBOLS.keys():
        df = fetch_historical_data(coin, months=months)
        save_data(df, coin)
        results[coin] = df
    return results


def get_latest_candles(coin: str, limit: int = 100) -> pd.DataFrame:
    """Fetch the most recent candles (for live prediction)."""
    symbol = SYMBOLS.get(coin.lower())
    if not symbol:
        raise ValueError(f"Unknown coin: {coin}")

    return fetch_hourly_data(symbol, limit=limit)


if __name__ == "__main__":
    # Test: fetch BTC data
    df = fetch_historical_data("btc", months=6)
    print(df.head())
    print(df.tail())
    save_data(df, "btc")
