"""
Kalman Filter for cryptocurrency price/feature denoising.

The Kalman filter is optimal for estimating the true state of a
noisy signal. For crypto:
- Smooths price data while preserving trends
- Reduces noise in technical indicators
- Can be used to generate alpha signals (price vs filtered price)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class KalmanFilter1D:
    """
    1D Kalman Filter for price/indicator smoothing.

    Parameters:
        process_variance (Q): How much we expect the true state to change
            - Higher = more responsive to recent data
            - Lower = smoother output
        measurement_variance (R): How noisy we believe the measurements are
            - Higher = trust measurements less, smoother output
            - Lower = follow measurements more closely
    """

    def __init__(
        self,
        process_variance: float = 1e-5,
        measurement_variance: float = 1e-2,
        initial_estimate: Optional[float] = None,
        initial_error: float = 1.0
    ):
        self.Q = process_variance  # Process noise covariance
        self.R = measurement_variance  # Measurement noise covariance
        self.x = initial_estimate  # State estimate
        self.P = initial_error  # Estimate error covariance

    def update(self, measurement: float) -> float:
        """
        Update filter with new measurement and return filtered value.
        """
        # Initialize on first measurement
        if self.x is None:
            self.x = measurement
            return self.x

        # Prediction step
        x_pred = self.x  # State prediction (random walk model)
        P_pred = self.P + self.Q  # Error prediction

        # Update step
        K = P_pred / (P_pred + self.R)  # Kalman gain
        self.x = x_pred + K * (measurement - x_pred)  # Updated estimate
        self.P = (1 - K) * P_pred  # Updated error

        return self.x

    def filter_series(self, series: pd.Series) -> pd.Series:
        """
        Apply Kalman filter to entire series.
        """
        filtered = []
        for value in series:
            if pd.isna(value):
                filtered.append(np.nan)
            else:
                filtered.append(self.update(value))
        return pd.Series(filtered, index=series.index)


class KalmanFilterWithVelocity:
    """
    Kalman Filter with velocity (trend) estimation.

    State: [price, velocity]
    This model assumes price follows: price(t) = price(t-1) + velocity(t-1)

    Better for trending markets as it can anticipate continuation.
    """

    def __init__(
        self,
        process_variance: float = 1e-4,
        measurement_variance: float = 1e-2,
        dt: float = 1.0
    ):
        self.Q = process_variance
        self.R = measurement_variance
        self.dt = dt

        # State transition matrix [price, velocity]
        self.F = np.array([
            [1, dt],
            [0, 1]
        ])

        # Measurement matrix (we only observe price)
        self.H = np.array([[1, 0]])

        # Process noise covariance
        self.Q_matrix = np.array([
            [dt**4/4, dt**3/2],
            [dt**3/2, dt**2]
        ]) * self.Q

        # State and covariance
        self.x = None  # [price, velocity]
        self.P = np.eye(2) * 1000  # Initial uncertainty

    def update(self, measurement: float) -> Tuple[float, float]:
        """
        Update filter and return (filtered_price, estimated_velocity).
        """
        if self.x is None:
            self.x = np.array([[measurement], [0]])
            return measurement, 0.0

        # Prediction
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q_matrix

        # Measurement residual
        z = np.array([[measurement]])
        y = z - self.H @ x_pred

        # Kalman gain
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T / S

        # Update
        self.x = x_pred + K @ y
        self.P = (np.eye(2) - K @ self.H) @ P_pred

        return float(self.x[0, 0]), float(self.x[1, 0])

    def filter_series(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Apply filter to series, return (filtered_prices, velocities).
        """
        prices = []
        velocities = []

        for value in series:
            if pd.isna(value):
                prices.append(np.nan)
                velocities.append(np.nan)
            else:
                p, v = self.update(value)
                prices.append(p)
                velocities.append(v)

        return (
            pd.Series(prices, index=series.index),
            pd.Series(velocities, index=series.index)
        )


def add_kalman_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Kalman filter-based features to OHLCV DataFrame.

    Features added:
    - kalman_price: Smoothed close price
    - kalman_velocity: Price trend/momentum from Kalman
    - kalman_residual: Difference between actual and filtered (mean-reversion signal)
    - kalman_zscore: Standardized residual (potential alpha signal)
    - kalman_smooth_rsi: Denoised RSI
    """
    df = df.copy()

    # Price smoothing with velocity estimation
    kf_price = KalmanFilterWithVelocity(
        process_variance=1e-5,
        measurement_variance=0.01
    )
    df["kalman_price"], df["kalman_velocity"] = kf_price.filter_series(df["close"])

    # Residual = actual - filtered (mean-reversion signal)
    df["kalman_residual"] = df["close"] - df["kalman_price"]

    # Z-score of residual (normalized deviation from trend)
    residual_std = df["kalman_residual"].rolling(24).std()
    df["kalman_zscore"] = df["kalman_residual"] / residual_std

    # Velocity acceleration (change in trend)
    df["kalman_acceleration"] = df["kalman_velocity"].diff()

    # Smooth RSI if available
    if "rsi_14" in df.columns:
        kf_rsi = KalmanFilter1D(
            process_variance=1e-4,
            measurement_variance=0.5
        )
        df["kalman_rsi"] = kf_rsi.filter_series(df["rsi_14"])

    # Smooth volume ratio if available
    if "volume_ratio" in df.columns:
        kf_vol = KalmanFilter1D(
            process_variance=1e-4,
            measurement_variance=0.3
        )
        df["kalman_volume_ratio"] = kf_vol.filter_series(df["volume_ratio"])

    return df


def adaptive_kalman_filter(
    series: pd.Series,
    min_R: float = 0.001,
    max_R: float = 0.1,
    volatility_lookback: int = 24
) -> pd.Series:
    """
    Adaptive Kalman filter that adjusts measurement noise based on volatility.

    In high volatility: trust measurements less (higher R) = smoother
    In low volatility: trust measurements more (lower R) = more responsive
    """
    # Calculate rolling volatility
    returns = series.pct_change()
    volatility = returns.rolling(volatility_lookback).std()
    vol_normalized = (volatility - volatility.min()) / (volatility.max() - volatility.min() + 1e-10)

    filtered = []
    x = None
    P = 1.0
    Q = 1e-5

    for i, (value, vol) in enumerate(zip(series, vol_normalized)):
        if pd.isna(value) or pd.isna(vol):
            filtered.append(np.nan)
            continue

        # Adaptive R based on volatility
        R = min_R + vol * (max_R - min_R)

        if x is None:
            x = value
            filtered.append(x)
            continue

        # Prediction
        P_pred = P + Q

        # Update
        K = P_pred / (P_pred + R)
        x = x + K * (value - x)
        P = (1 - K) * P_pred

        filtered.append(x)

    return pd.Series(filtered, index=series.index)


if __name__ == "__main__":
    # Test with sample data
    from data_fetch import load_data

    print("Testing Kalman filter on BTC data...")
    df = load_data("btc")

    # Add Kalman features
    df = add_kalman_features(df)

    print("\nKalman features added:")
    print(df[["close", "kalman_price", "kalman_velocity", "kalman_zscore"]].tail(20))

    # Test adaptive filter
    print("\nTesting adaptive Kalman filter...")
    adaptive_prices = adaptive_kalman_filter(df["close"])

    print(f"Original std: {df['close'].std():.2f}")
    print(f"Filtered std: {adaptive_prices.std():.2f}")
    print(f"Correlation: {df['close'].corr(adaptive_prices):.4f}")
