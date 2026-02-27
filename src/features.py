"""
Feature engineering for crypto price prediction.
Technical indicators, Fear & Greed Index, and cross-asset features.
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to OHLCV DataFrame.

    Args:
        df: DataFrame with columns [open, high, low, close, volume]

    Returns:
        DataFrame with additional indicator columns
    """
    df = df.copy()

    # RSI (Relative Strength Index) - multiple periods
    df["rsi_14"] = ta.rsi(df["close"], length=14)
    df["rsi_7"] = ta.rsi(df["close"], length=7)
    df["rsi_21"] = ta.rsi(df["close"], length=21)

    # MACD
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_hist"] = macd["MACDh_12_26_9"]

    # Bollinger Bands
    bbands = ta.bbands(df["close"], length=20, std=2)
    df["bb_upper"] = bbands["BBU_20_2.0_2.0"]
    df["bb_middle"] = bbands["BBM_20_2.0_2.0"]
    df["bb_lower"] = bbands["BBL_20_2.0_2.0"]
    # Position within bands (0 = at lower, 1 = at upper)
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
    # Bollinger Band width (volatility measure)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"] * 100

    # ATR (Average True Range) - volatility
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    # Normalized ATR (as % of price)
    df["atr_pct"] = df["atr_14"] / df["close"] * 100

    # ===== NEW INDICATORS =====

    # Stochastic Oscillator
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
    df["stoch_k"] = stoch["STOCHk_14_3_3"]
    df["stoch_d"] = stoch["STOCHd_14_3_3"]

    # Williams %R
    df["willr"] = ta.willr(df["high"], df["low"], df["close"], length=14)

    # ADX (Average Directional Index) - trend strength
    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    df["adx"] = adx["ADX_14"]
    df["dmp"] = adx["DMP_14"]  # +DI
    df["dmn"] = adx["DMN_14"]  # -DI

    # OBV (On-Balance Volume)
    df["obv"] = ta.obv(df["close"], df["volume"])
    # Normalized OBV change
    df["obv_pct"] = df["obv"].pct_change(24) * 100

    # CMF (Chaikin Money Flow)
    df["cmf"] = ta.cmf(df["high"], df["low"], df["close"], df["volume"], length=20)

    # MFI (Money Flow Index) - volume-weighted RSI
    df["mfi"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)

    # CCI (Commodity Channel Index)
    df["cci"] = ta.cci(df["high"], df["low"], df["close"], length=20)

    # ROC (Rate of Change)
    df["roc_12"] = ta.roc(df["close"], length=12)

    # ===== END NEW INDICATORS =====

    # Volume features
    df["volume_sma_24"] = df["volume"].rolling(24).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_24"]
    # Volume trend
    df["volume_trend"] = (df["volume"].rolling(6).mean() > df["volume"].rolling(24).mean()).astype(int)

    # Price momentum
    df["return_1h"] = df["close"].pct_change(1) * 100
    df["return_4h"] = df["close"].pct_change(4) * 100
    df["return_24h"] = df["close"].pct_change(24) * 100

    # Moving averages
    df["sma_20"] = ta.sma(df["close"], length=20)
    df["sma_50"] = ta.sma(df["close"], length=50)
    df["ema_12"] = ta.ema(df["close"], length=12)
    df["ema_26"] = ta.ema(df["close"], length=26)

    # Price relative to MAs
    df["price_vs_sma20"] = (df["close"] / df["sma_20"] - 1) * 100
    df["price_vs_sma50"] = (df["close"] / df["sma_50"] - 1) * 100
    df["price_vs_ema12"] = (df["close"] / df["ema_12"] - 1) * 100

    # Trend (SMA crossover)
    df["sma_trend"] = (df["sma_20"] > df["sma_50"]).astype(int)
    df["ema_trend"] = (df["ema_12"] > df["ema_26"]).astype(int)

    # Volatility regime (high/low volatility)
    df["volatility_regime"] = (df["atr_pct"] > df["atr_pct"].rolling(168).mean()).astype(int)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features."""
    df = df.copy()

    # Hour of day (0-23) - cyclical encoding
    hour = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Day of week (0-6) - cyclical encoding
    dow = df.index.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # Is weekend?
    df["is_weekend"] = (dow >= 5).astype(int)

    return df


def add_multitimeframe_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add higher-timeframe indicators (4h) and trend context.
    Uses resampling and forward-fill to hourly index.
    """
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    close_4h = df["close"].resample("4H").last()
    rsi_4h = ta.rsi(close_4h, length=14)
    macd_4h = ta.macd(close_4h, fast=12, slow=26, signal=9)
    sma_4h = close_4h.rolling(5).mean()

    df["rsi_14_4h"] = rsi_4h.reindex(df.index, method="ffill")
    if macd_4h is not None:
        df["macd_hist_4h"] = macd_4h["MACDh_12_26_9"].reindex(df.index, method="ffill")
    df["price_vs_sma_4h"] = (df["close"] / sma_4h.reindex(df.index, method="ffill") - 1) * 100

    return df


def add_macro_event_features(
    df: pd.DataFrame,
    macro_path: Path = DATA_DIR / "macro_events.csv",
    default_window_hours: int = 6
) -> pd.DataFrame:
    """
    Add macro/event overlay features from a local events file.

    Expected columns: timestamp, impact(optional), window_hours(optional)
    """
    if not macro_path.exists():
        return df

    events = pd.read_csv(macro_path)
    if "timestamp" not in events.columns:
        return df

    events["timestamp"] = pd.to_datetime(events["timestamp"])
    events["impact"] = pd.to_numeric(events.get("impact"), errors="coerce")
    events["window_hours"] = pd.to_numeric(events.get("window_hours"), errors="coerce")

    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df["macro_event"] = 0
    df["macro_impact"] = 0

    for _, row in events.iterrows():
        event_time = row["timestamp"]
        window = int(row["window_hours"]) if not pd.isna(row["window_hours"]) else default_window_hours
        start = event_time - pd.Timedelta(hours=window)
        end = event_time + pd.Timedelta(hours=window)
        mask = (df.index >= start) & (df.index <= end)
        df.loc[mask, "macro_event"] = 1
        if not pd.isna(row["impact"]):
            df.loc[mask, "macro_impact"] = np.maximum(df.loc[mask, "macro_impact"], row["impact"])

    return df


def add_exchange_flow_features(
    df: pd.DataFrame,
    availability_lag_hours: int = 24,
    flows_path: Path = DATA_DIR / "exchange_flows.parquet"
) -> pd.DataFrame:
    """
    Merge daily exchange flow metrics (if provided locally) into hourly data.
    """
    if not flows_path.exists():
        csv_path = flows_path.with_suffix(".csv")
        if not csv_path.exists():
            return df
        flows = pd.read_csv(csv_path)
    else:
        flows = pd.read_parquet(flows_path)
    if "timestamp" in flows.columns:
        flows = flows.set_index(pd.to_datetime(flows["timestamp"])).drop(columns=["timestamp"])
    elif "date" in flows.columns:
        flows = flows.set_index(pd.to_datetime(flows["date"])).drop(columns=["date"])
    if not isinstance(flows.index, pd.DatetimeIndex):
        flows.index = pd.to_datetime(flows.index)

    flows = flows.sort_index()
    flows.index = flows.index.normalize()
    if availability_lag_hours:
        flows.index = flows.index + pd.Timedelta(hours=availability_lag_hours)

    hourly = df.copy()
    if not isinstance(hourly.index, pd.DatetimeIndex):
        hourly.index = pd.to_datetime(hourly.index)
    hourly = hourly.sort_index()

    hourly_reset = hourly.reset_index().rename(columns={"index": "timestamp"})
    flows_reset = flows.reset_index().rename(columns={"index": "flow_timestamp"})

    merged = pd.merge_asof(
        hourly_reset.sort_values("timestamp"),
        flows_reset.sort_values("flow_timestamp"),
        left_on="timestamp",
        right_on="flow_timestamp",
        direction="backward"
    )
    merged = merged.drop(columns=["flow_timestamp"]).set_index("timestamp")
    return merged


def add_orderbook_features(
    df: pd.DataFrame,
    orderbook_path: Path = DATA_DIR / "orderbook.parquet"
) -> pd.DataFrame:
    """
    Merge optional order book features (hourly) if present locally.
    """
    if not orderbook_path.exists():
        csv_path = orderbook_path.with_suffix(".csv")
        if not csv_path.exists():
            return df
        orderbook = pd.read_csv(csv_path)
    else:
        orderbook = pd.read_parquet(orderbook_path)
    if "timestamp" in orderbook.columns:
        orderbook = orderbook.set_index(pd.to_datetime(orderbook["timestamp"])).drop(columns=["timestamp"])
    if not isinstance(orderbook.index, pd.DatetimeIndex):
        orderbook.index = pd.to_datetime(orderbook.index)

    orderbook = orderbook.sort_index()
    merged = df.copy()
    merged = merged.join(orderbook, how="left")
    return merged


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged features for temporal patterns.
    Key insight: patterns from 2h, 6h, 12h ago often predict next hour.

    Args:
        df: DataFrame with technical indicators already added

    Returns:
        DataFrame with lag features
    """
    df = df.copy()

    # Lag periods (hours ago)
    lag_periods = [2, 6, 12]

    # Key indicators to lag
    indicators_to_lag = [
        "rsi_14",
        "macd_hist",
        "bb_position",
        "volume_ratio",
        "return_1h",
    ]

    for lag in lag_periods:
        for indicator in indicators_to_lag:
            if indicator in df.columns:
                df[f"{indicator}_lag{lag}"] = df[indicator].shift(lag)

        # Return over lag period
        df[f"return_{lag}h"] = df["close"].pct_change(lag) * 100

    # Change in indicators (momentum of momentum)
    if "rsi_14" in df.columns:
        df["rsi_change_6h"] = df["rsi_14"] - df["rsi_14"].shift(6)
        df["rsi_change_12h"] = df["rsi_14"] - df["rsi_14"].shift(12)

    if "macd_hist" in df.columns:
        df["macd_hist_change_6h"] = df["macd_hist"] - df["macd_hist"].shift(6)

    # Rolling statistics (higher-order features)
    df["return_std_24h"] = df["return_1h"].rolling(24).std()
    df["return_skew_24h"] = df["return_1h"].rolling(24).skew()
    df["return_kurt_24h"] = df["return_1h"].rolling(24).kurt()

    # High/Low range features
    df["hl_range_pct"] = (df["high"] - df["low"]) / df["close"] * 100
    df["hl_range_vs_avg"] = df["hl_range_pct"] / df["hl_range_pct"].rolling(24).mean()

    # Consecutive up/down candles
    df["up_candle"] = (df["close"] > df["open"]).astype(int)
    df["consecutive_up"] = df["up_candle"].rolling(6).sum()
    df["consecutive_down"] = 6 - df["consecutive_up"]

    # Price position in day's range
    day_high = df["high"].rolling(24).max()
    day_low = df["low"].rolling(24).min()
    df["price_position_24h"] = (df["close"] - day_low) / (day_high - day_low)

    return df


def add_cross_asset_features(df: pd.DataFrame, btc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add BTC-based features for altcoin prediction.
    BTC often leads altcoin movements.

    Args:
        df: Altcoin OHLCV DataFrame
        btc_df: BTC OHLCV DataFrame (must have same index)

    Returns:
        DataFrame with BTC momentum features added
    """
    df = df.copy()

    # Ensure BTC data is aligned
    btc_aligned = btc_df.reindex(df.index)

    # BTC momentum features
    df["btc_return_1h"] = btc_aligned["close"].pct_change(1) * 100
    df["btc_return_4h"] = btc_aligned["close"].pct_change(4) * 100
    df["btc_return_24h"] = btc_aligned["close"].pct_change(24) * 100

    # BTC volatility
    df["btc_volatility_24h"] = btc_aligned["close"].pct_change().rolling(24).std() * 100

    # Relative strength: is altcoin outperforming BTC?
    alt_return_24h = df["close"].pct_change(24)
    btc_return_24h = btc_aligned["close"].pct_change(24)
    df["alt_vs_btc_24h"] = (alt_return_24h - btc_return_24h) * 100

    # Rolling correlation with BTC (24h window)
    alt_returns = df["close"].pct_change()
    btc_returns = btc_aligned["close"].pct_change()
    df["btc_corr_24h"] = alt_returns.rolling(24).corr(btc_returns)

    return df


def add_fear_greed_features(
    df: pd.DataFrame,
    availability_lag_hours: int = 24
) -> pd.DataFrame:
    """
    Add Fear & Greed Index features.

    Args:
        df: OHLCV DataFrame with datetime index
        availability_lag_hours: Hours to lag daily data before it is available

    Returns:
        DataFrame with FGI features added
    """
    from fear_greed import merge_fgi_with_hourly
    return merge_fgi_with_hourly(df, availability_lag_hours=availability_lag_hours)


def add_onchain_features(
    df: pd.DataFrame,
    availability_lag_hours: int = 24
) -> pd.DataFrame:
    """
    Add on-chain metrics (BTC only - MVRV, SOPR, Active Addresses).

    Args:
        df: OHLCV DataFrame with datetime index
        availability_lag_hours: Hours to lag daily data before it is available

    Returns:
        DataFrame with on-chain features added
    """
    from onchain import get_all_onchain_metrics, merge_onchain_with_hourly, add_onchain_features as derive_onchain

    # Fetch on-chain data
    onchain_df = get_all_onchain_metrics()

    if onchain_df.empty:
        print("Warning: No on-chain data available")
        return df

    # Merge with hourly data
    df = merge_onchain_with_hourly(df, onchain_df, availability_lag_hours=availability_lag_hours)

    # Add derived features
    df = derive_onchain(df)

    return df


def add_target(df: pd.DataFrame, lookahead: int = 1) -> pd.DataFrame:
    """
    Add target variable: did price go up in next N hours?

    Args:
        df: DataFrame with close prices
        lookahead: Hours ahead to predict (default 1)

    Returns:
        DataFrame with 'target' column (1 = up, 0 = down/flat)
    """
    df = df.copy()

    # Future return
    df["future_return"] = df["close"].shift(-lookahead) / df["close"] - 1

    # Binary target
    df["target"] = (df["future_return"] > 0).astype(int)

    return df


def add_kalman_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Kalman filter-based features for noise reduction and trend extraction.

    Features added:
    - kalman_price: Smoothed close price
    - kalman_velocity: Price trend/momentum from Kalman
    - kalman_zscore: Standardized residual (mean-reversion signal)
    - kalman_acceleration: Change in trend
    """
    from kalman import add_kalman_features as kalman_add
    return kalman_add(df)


def prepare_features(
    df: pd.DataFrame,
    lookahead: int = 1,
    btc_df: pd.DataFrame = None,
    include_fgi: bool = True,
    include_onchain: bool = False,
    include_lags: bool = True,
    include_kalman: bool = True,
    include_multitimeframe: bool = True,
    include_macro: bool = False,
    include_flows: bool = False,
    include_orderbook: bool = False,
    availability_lag_hours: int = 24
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Args:
        df: Raw OHLCV DataFrame
        lookahead: Hours ahead to predict
        btc_df: BTC DataFrame for cross-asset features (skip for BTC itself)
        include_fgi: Whether to include Fear & Greed Index features
        include_onchain: Whether to include on-chain metrics (BTC only)
        include_lags: Whether to include lag features (default True)
        include_kalman: Whether to include Kalman filter features (default True)
        include_multitimeframe: Include 4h timeframe indicators
        include_macro: Include macro/event overlay features if available
        include_flows: Include exchange flow features if available
        include_orderbook: Include order book features if available
        availability_lag_hours: Hours to lag daily metrics before availability

    Returns:
        DataFrame with all features and target, NaN rows dropped
    """
    df = add_technical_indicators(df)
    df = add_time_features(df)

    if include_multitimeframe:
        df = add_multitimeframe_features(df)

    # Add lag features for temporal patterns
    if include_lags:
        df = add_lag_features(df)

    # Add Kalman filter features (noise reduction + trend)
    if include_kalman:
        try:
            df = add_kalman_features(df)
        except Exception as e:
            print(f"Warning: Could not add Kalman features: {e}")

    # Add Fear & Greed features
    if include_fgi:
        try:
            df = add_fear_greed_features(df, availability_lag_hours=availability_lag_hours)
        except Exception as e:
            print(f"Warning: Could not add FGI features: {e}")

    # Add on-chain features (BTC only)
    if include_onchain:
        try:
            df = add_onchain_features(df, availability_lag_hours=availability_lag_hours)
        except Exception as e:
            print(f"Warning: Could not add on-chain features: {e}")

    # Add cross-asset features (for altcoins)
    if btc_df is not None:
        df = add_cross_asset_features(df, btc_df)

    if include_macro:
        try:
            df = add_macro_event_features(df)
        except Exception as e:
            print(f"Warning: Could not add macro features: {e}")

    if include_flows:
        try:
            df = add_exchange_flow_features(df, availability_lag_hours=availability_lag_hours)
        except Exception as e:
            print(f"Warning: Could not add exchange flow features: {e}")

    if include_orderbook:
        try:
            df = add_orderbook_features(df)
        except Exception as e:
            print(f"Warning: Could not add order book features: {e}")

    df = add_target(df, lookahead=lookahead)

    # Drop rows with NaN (from indicator warmup period)
    df = df.dropna()

    return df


def get_feature_columns(
    include_fgi: bool = True,
    include_cross_asset: bool = False,
    include_onchain: bool = False,
    include_lags: bool = True,
    include_kalman: bool = True,
    include_multitimeframe: bool = True,
    include_macro: bool = False,
    include_flows: bool = False,
    include_orderbook: bool = False
) -> list:
    """
    Return list of feature column names.

    Args:
        include_fgi: Include Fear & Greed features
        include_cross_asset: Include BTC cross-asset features (for altcoins)
        include_onchain: Include on-chain metrics (BTC only)
        include_lags: Include lag features
        include_kalman: Include Kalman filter features
        include_multitimeframe: Include 4h timeframe indicators
        include_macro: Include macro/event overlay features
        include_flows: Include exchange flow features
        include_orderbook: Include order book features
    """
    features = [
        # Technical indicators - RSI
        "rsi_14", "rsi_7", "rsi_21",
        # MACD
        "macd", "macd_signal", "macd_hist",
        # Bollinger Bands
        "bb_position", "bb_width",
        # Volatility
        "atr_pct", "volatility_regime",
        # Stochastic
        "stoch_k", "stoch_d",
        # Williams %R
        "willr",
        # ADX (trend strength)
        "adx", "dmp", "dmn",
        # Volume indicators
        "obv_pct", "cmf", "mfi",
        "volume_ratio", "volume_trend",
        # Other oscillators
        "cci", "roc_12",
        # Momentum
        "return_1h", "return_4h", "return_24h",
        # Moving average features
        "price_vs_sma20", "price_vs_sma50", "price_vs_ema12",
        "sma_trend", "ema_trend",
        # Time features
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "is_weekend",
    ]

    if include_lags:
        features.extend([
            # Lagged indicators
            "rsi_14_lag2", "rsi_14_lag6", "rsi_14_lag12",
            "macd_hist_lag2", "macd_hist_lag6", "macd_hist_lag12",
            "bb_position_lag2", "bb_position_lag6", "bb_position_lag12",
            "volume_ratio_lag2", "volume_ratio_lag6", "volume_ratio_lag12",
            "return_1h_lag2", "return_1h_lag6", "return_1h_lag12",
            # Returns over lag periods
            "return_2h", "return_6h", "return_12h",
            # Momentum of momentum
            "rsi_change_6h", "rsi_change_12h",
            "macd_hist_change_6h",
            # Rolling stats
            "return_std_24h", "return_skew_24h", "return_kurt_24h",
            # Range features
            "hl_range_pct", "hl_range_vs_avg",
            # Consecutive candles
            "consecutive_up", "consecutive_down",
            # Price position
            "price_position_24h",
        ])

    if include_fgi:
        features.extend([
            "fgi_value",
            "fgi_change_7d",
            "fgi_is_fear",
            "fgi_is_greed",
        ])

    if include_onchain:
        features.extend([
            # Raw metrics
            "mvrv_zscore",
            "sopr",
            "active_addresses",
            "funding_rate",
            # Derived features
            "mvrv_overbought",
            "mvrv_oversold",
            "sopr_profit",
            "sopr_loss",
            "sopr_above_ma",
            "funding_high_long",
            "funding_high_short",
            "funding_extreme",
        ])

    if include_cross_asset:
        features.extend([
            "btc_return_1h",
            "btc_return_4h",
            "btc_return_24h",
            "btc_volatility_24h",
            "alt_vs_btc_24h",
            "btc_corr_24h",
        ])

    if include_kalman:
        features.extend([
            # Kalman filter features (noise reduction + trend)
            "kalman_velocity",      # Trend direction/momentum
            "kalman_zscore",        # Mean-reversion signal
            "kalman_acceleration",  # Change in trend
            "kalman_rsi",           # Denoised RSI
            "kalman_volume_ratio",  # Denoised volume ratio
        ])

    if include_multitimeframe:
        features.extend([
            "rsi_14_4h",
            "macd_hist_4h",
            "price_vs_sma_4h",
        ])

    if include_macro:
        features.extend([
            "macro_event",
            "macro_impact",
        ])

    if include_flows:
        features.extend([
            "exchange_inflow",
            "exchange_outflow",
            "exchange_netflow",
        ])

    if include_orderbook:
        features.extend([
            "bid_ask_spread",
            "order_imbalance",
            "depth_ratio",
        ])

    return features


if __name__ == "__main__":
    # Test with sample data
    from data_fetch import load_data

    print("Testing BTC features (no cross-asset)...")
    btc_df = load_data("btc")
    btc_features = prepare_features(btc_df, include_fgi=True)
    print(f"BTC shape: {btc_features.shape}")
    print(f"BTC features: {get_feature_columns(include_fgi=True, include_cross_asset=False)}")

    print("\nTesting ETH features (with cross-asset)...")
    eth_df = load_data("eth")
    eth_features = prepare_features(eth_df, btc_df=btc_df, include_fgi=True)
    print(f"ETH shape: {eth_features.shape}")
    print(f"ETH features: {get_feature_columns(include_fgi=True, include_cross_asset=True)}")

    print(f"\nFGI values sample:")
    print(btc_features[["fgi_value", "fgi_is_fear", "fgi_is_greed"]].tail(10))
