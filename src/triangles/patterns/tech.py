from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
from pykalman import KalmanFilter
import pywt


def detect_head_shoulder(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Detect Head and Shoulder patterns using a basic rolling window approach."""
    df['high_roll_max'] = df['High'].rolling(window=window).max()
    df['low_roll_min'] = df['Low'].rolling(window=window).min()

    mask_head_shoulder = (
        (df['high_roll_max'] > df['High'].shift(1))
        & (df['high_roll_max'] > df['High'].shift(-1))
        & (df['High'] < df['High'].shift(1))
        & (df['High'] < df['High'].shift(-1))
    )
    mask_inv_head_shoulder = (
        (df['low_roll_min'] < df['Low'].shift(1))
        & (df['low_roll_min'] < df['Low'].shift(-1))
        & (df['Low'] > df['Low'].shift(1))
        & (df['Low'] > df['Low'].shift(-1))
    )

    df['head_shoulder_pattern'] = np.nan
    df.loc[mask_head_shoulder, 'head_shoulder_pattern'] = 'Head and Shoulder'
    df.loc[mask_inv_head_shoulder, 'head_shoulder_pattern'] = 'Inverse Head and Shoulder'
    return df


def detect_head_shoulder_filter(
    df: pd.DataFrame,
    window: int = 3,
    threshold: float = 0.01,
    time_delay: int = 1,
) -> pd.DataFrame:
    """Detect Head and Shoulder patterns using Savitzky-Golay smoothing.

    Smooths High/Low before detection and requires a minimum head height above
    `threshold` to reduce false positives from insignificant price moves.
    """
    df['High_smooth'] = savgol_filter(df['High'], window, 2)
    df['Low_smooth'] = savgol_filter(df['Low'], window, 2)

    df['high_roll_max'] = df['High_smooth'].rolling(window=window).max()
    df['low_roll_min'] = df['Low_smooth'].rolling(window=window).min()
    df['head_height'] = df['high_roll_max'] - df['Low'].rolling(window=window).min()
    df['inv_head_height'] = df['High'].rolling(window=window).max() - df['low_roll_min']

    mask_head_shoulder = (
        (df['head_height'] > threshold)
        & (df['high_roll_max'] > df['High_smooth'].shift(time_delay))
        & (df['high_roll_max'] > df['High_smooth'].shift(-time_delay))
        & (df['High_smooth'] < df['High_smooth'].shift(time_delay))
        & (df['High_smooth'] < df['High_smooth'].shift(-time_delay))
    )
    mask_inv_head_shoulder = (
        (df['inv_head_height'] > threshold)
        & (df['low_roll_min'] < df['Low_smooth'].shift(time_delay))
        & (df['low_roll_min'] < df['Low_smooth'].shift(-time_delay))
        & (df['Low_smooth'] > df['Low_smooth'].shift(time_delay))
        & (df['Low_smooth'] > df['Low_smooth'].shift(-time_delay))
    )

    df['head_shoulder_pattern'] = np.nan
    df.loc[mask_head_shoulder, 'head_shoulder_pattern'] = 'Head and Shoulder'
    df.loc[mask_inv_head_shoulder, 'head_shoulder_pattern'] = 'Inverse Head and Shoulder'
    return df


def kalman_smooth(series: pd.Series, n_iter: int = 10) -> np.ndarray:
    """Smooth a price series using Kalman filter with EM parameter estimation."""
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    kf = kf.em(series, n_iter=n_iter)
    state_means, _ = kf.filter(series.values)
    return state_means.flatten()


def detect_head_shoulder_kf(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Detect Head and Shoulder patterns using Kalman filter smoothing.

    The Kalman filter estimates a system's state recursively, handling noise
    and uncertainty — making it well-suited for financial time series.
    """
    df['High_smooth'] = kalman_smooth(df['High'])
    df['Low_smooth'] = kalman_smooth(df['Low'])

    df['high_roll_max'] = df['High_smooth'].rolling(window=window).max()
    df['low_roll_min'] = df['Low_smooth'].rolling(window=window).min()

    mask_head_shoulder = (
        (df['high_roll_max'] > df['High_smooth'].shift(1))
        & (df['high_roll_max'] > df['High_smooth'].shift(-1))
        & (df['High_smooth'] < df['High_smooth'].shift(1))
        & (df['High_smooth'] < df['High_smooth'].shift(-1))
    )
    mask_inv_head_shoulder = (
        (df['low_roll_min'] < df['Low_smooth'].shift(1))
        & (df['low_roll_min'] < df['Low_smooth'].shift(-1))
        & (df['Low_smooth'] > df['Low_smooth'].shift(1))
        & (df['Low_smooth'] > df['Low_smooth'].shift(-1))
    )

    df['head_shoulder_pattern'] = np.nan
    df.loc[mask_head_shoulder, 'head_shoulder_pattern'] = 'Head and Shoulder'
    df.loc[mask_inv_head_shoulder, 'head_shoulder_pattern'] = 'Inverse Head and Shoulder'
    return df


def wavelet_denoise(series: pd.Series, wavelet: str = 'db1', level: int = 1) -> np.ndarray:
    """Denoise a series via wavelet decomposition with per-level soft thresholding."""
    coeff = pywt.wavedec(series, wavelet, mode="per")
    for i in range(1, len(coeff)):
        coeff[i] = pywt.threshold(coeff[i], value=np.std(coeff[i]) / 2, mode="soft")
    return pywt.waverec(coeff, wavelet, mode="per")


def detect_head_shoulder_wavelet(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Detect Head and Shoulder patterns using wavelet denoising.

    Wavelet decomposition removes high-frequency noise while preserving
    structural features, improving robustness against market microstructure noise.
    """
    df['High_smooth'] = wavelet_denoise(df['High'], 'db1', level=1)
    df['Low_smooth'] = wavelet_denoise(df['Low'], 'db1', level=1)

    df['high_roll_max'] = df['High_smooth'].rolling(window=window).max()
    df['low_roll_min'] = df['Low_smooth'].rolling(window=window).min()

    mask_head_shoulder = (
        (df['high_roll_max'] > df['High_smooth'].shift(1))
        & (df['high_roll_max'] > df['High_smooth'].shift(-1))
        & (df['High_smooth'] < df['High_smooth'].shift(1))
        & (df['High_smooth'] < df['High_smooth'].shift(-1))
    )
    mask_inv_head_shoulder = (
        (df['low_roll_min'] < df['Low_smooth'].shift(1))
        & (df['low_roll_min'] < df['Low_smooth'].shift(-1))
        & (df['Low_smooth'] > df['Low_smooth'].shift(1))
        & (df['Low_smooth'] > df['Low_smooth'].shift(-1))
    )

    df['head_shoulder_pattern'] = np.nan
    df.loc[mask_head_shoulder, 'head_shoulder_pattern'] = 'Head and Shoulder'
    df.loc[mask_inv_head_shoulder, 'head_shoulder_pattern'] = 'Inverse Head and Shoulder'
    return df
