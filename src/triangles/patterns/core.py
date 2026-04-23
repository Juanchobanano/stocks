import pandas as pd
import numpy as np


def detect_head_shoulder(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Detect Head and Shoulder and Inverse Head and Shoulder patterns."""
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


def detect_multiple_tops_bottoms(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Detect Multiple Top and Multiple Bottom range-bound patterns."""
    df['high_roll_max'] = df['High'].rolling(window=window).max()
    df['low_roll_min'] = df['Low'].rolling(window=window).min()
    df['close_roll_max'] = df['Close'].rolling(window=window).max()
    df['close_roll_min'] = df['Close'].rolling(window=window).min()

    mask_top = (
        (df['high_roll_max'] >= df['High'].shift(1))
        & (df['close_roll_max'] < df['Close'].shift(1))
    )
    mask_bottom = (
        (df['low_roll_min'] <= df['Low'].shift(1))
        & (df['close_roll_min'] > df['Close'].shift(1))
    )

    df['multiple_top_bottom_pattern'] = np.nan
    df.loc[mask_top, 'multiple_top_bottom_pattern'] = 'Multiple Top'
    df.loc[mask_bottom, 'multiple_top_bottom_pattern'] = 'Multiple Bottom'
    return df


def calculate_support_resistance(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Compute support and resistance levels using rolling mean ± 2 standard deviations."""
    std_dev = 2

    df['high_roll_max'] = df['High'].rolling(window=window).max()
    df['low_roll_min'] = df['Low'].rolling(window=window).min()
    df['support'] = df['Low'].rolling(window=window).mean() - std_dev * df['Low'].rolling(window=window).std()
    df['resistance'] = df['High'].rolling(window=window).mean() + std_dev * df['High'].rolling(window=window).std()
    return df


def detect_triangle_pattern(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Detect Ascending and Descending Triangle patterns."""
    df['high_roll_max'] = df['High'].rolling(window=window).max()
    df['low_roll_min'] = df['Low'].rolling(window=window).min()

    mask_asc = (
        (df['high_roll_max'] >= df['High'].shift(1))
        & (df['low_roll_min'] <= df['Low'].shift(1))
        & (df['Close'] > df['Close'].shift(1))
    )
    mask_desc = (
        (df['high_roll_max'] <= df['High'].shift(1))
        & (df['low_roll_min'] >= df['Low'].shift(1))
        & (df['Close'] < df['Close'].shift(1))
    )

    df['triangle_pattern'] = np.nan
    df.loc[mask_asc, 'triangle_pattern'] = 'Ascending Triangle'
    df.loc[mask_desc, 'triangle_pattern'] = 'Descending Triangle'
    return df


def _trend_direction(x: np.ndarray) -> int:
    """Return 1 for uptrend, -1 for downtrend, 0 for flat over the window."""
    delta = x[-1] - x[0]
    return 1 if delta > 0 else (-1 if delta < 0 else 0)


def detect_wedge(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Detect Wedge Up and Wedge Down patterns."""
    df['high_roll_max'] = df['High'].rolling(window=window).max()
    df['low_roll_min'] = df['Low'].rolling(window=window).min()
    df['trend_high'] = df['High'].rolling(window=window).apply(_trend_direction)
    df['trend_low'] = df['Low'].rolling(window=window).apply(_trend_direction)

    mask_wedge_up = (
        (df['high_roll_max'] >= df['High'].shift(1))
        & (df['low_roll_min'] <= df['Low'].shift(1))
        & (df['trend_high'] == 1)
        & (df['trend_low'] == 1)
    )
    mask_wedge_down = (
        (df['high_roll_max'] <= df['High'].shift(1))
        & (df['low_roll_min'] >= df['Low'].shift(1))
        & (df['trend_high'] == -1)
        & (df['trend_low'] == -1)
    )

    df['wedge_pattern'] = np.nan
    df.loc[mask_wedge_up, 'wedge_pattern'] = 'Wedge Up'
    df.loc[mask_wedge_down, 'wedge_pattern'] = 'Wedge Down'
    return df


def detect_channel(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Detect Channel Up and Channel Down patterns within a 10% price range."""
    channel_range = 0.1

    df['high_roll_max'] = df['High'].rolling(window=window).max()
    df['low_roll_min'] = df['Low'].rolling(window=window).min()
    df['trend_high'] = df['High'].rolling(window=window).apply(_trend_direction)
    df['trend_low'] = df['Low'].rolling(window=window).apply(_trend_direction)

    midpoint = (df['high_roll_max'] + df['low_roll_min']) / 2
    within_range = (df['high_roll_max'] - df['low_roll_min']) <= channel_range * midpoint

    mask_channel_up = (
        (df['high_roll_max'] >= df['High'].shift(1))
        & (df['low_roll_min'] <= df['Low'].shift(1))
        & within_range
        & (df['trend_high'] == 1)
        & (df['trend_low'] == 1)
    )
    mask_channel_down = (
        (df['high_roll_max'] <= df['High'].shift(1))
        & (df['low_roll_min'] >= df['Low'].shift(1))
        & within_range
        & (df['trend_high'] == -1)
        & (df['trend_low'] == -1)
    )

    df['channel_pattern'] = np.nan
    df.loc[mask_channel_up, 'channel_pattern'] = 'Channel Up'
    df.loc[mask_channel_down, 'channel_pattern'] = 'Channel Down'
    return df


def detect_double_top_bottom(df: pd.DataFrame, window: int = 3, threshold: float = 0.05) -> pd.DataFrame:
    """Detect Double Top and Double Bottom patterns within a price range threshold."""
    df['high_roll_max'] = df['High'].rolling(window=window).max()
    df['low_roll_min'] = df['Low'].rolling(window=window).min()

    prev_midpoint = (df['High'].shift(1) + df['Low'].shift(1)) / 2
    next_midpoint = (df['High'].shift(-1) + df['Low'].shift(-1)) / 2
    prev_in_range = (df['High'].shift(1) - df['Low'].shift(1)) <= threshold * prev_midpoint
    next_in_range = (df['High'].shift(-1) - df['Low'].shift(-1)) <= threshold * next_midpoint

    mask_double_top = (
        (df['high_roll_max'] >= df['High'].shift(1))
        & (df['high_roll_max'] >= df['High'].shift(-1))
        & (df['High'] < df['High'].shift(1))
        & (df['High'] < df['High'].shift(-1))
        & prev_in_range
        & next_in_range
    )
    mask_double_bottom = (
        (df['low_roll_min'] <= df['Low'].shift(1))
        & (df['low_roll_min'] <= df['Low'].shift(-1))
        & (df['Low'] > df['Low'].shift(1))
        & (df['Low'] > df['Low'].shift(-1))
        & prev_in_range
        & next_in_range
    )

    df['double_pattern'] = np.nan
    df.loc[mask_double_top, 'double_pattern'] = 'Double Top'
    df.loc[mask_double_bottom, 'double_pattern'] = 'Double Bottom'
    return df


def detect_trendline(df: pd.DataFrame, window: int = 2) -> pd.DataFrame:
    """Compute a rolling linear regression trendline and classify support/resistance by slope sign."""
    df['slope'] = np.nan
    df['intercept'] = np.nan

    for i in range(window, len(df)):
        x = np.arange(i - window, i)
        y = df['Close'].iloc[i - window:i].values
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        df.at[df.index[i], 'slope'] = m
        df.at[df.index[i], 'intercept'] = c

    df['support'] = np.nan
    df['resistance'] = np.nan
    df.loc[df['slope'] > 0, 'support'] = df['Close'] * df['slope'] + df['intercept']
    df.loc[df['slope'] < 0, 'resistance'] = df['Close'] * df['slope'] + df['intercept']
    return df


def detect_triangle_geometry(
    df: pd.DataFrame,
    lookback: int = 50,
    min_touches: int = 2,
    flatness_tol: float = 0.01,
) -> dict | None:
    """
    Fit trendlines through swing highs/lows over a lookback window and classify
    the triangle pattern with full geometry for order sizing.

    Returns None when no valid triangle is found.

    Returns a dict with:
        pattern_type       – 'Ascending Triangle' | 'Descending Triangle' | 'Symmetrical Triangle'
        height             – vertical extent at the base of the triangle
        resistance_at_now  – resistance trendline value at the last bar (long entry trigger)
        support_at_now     – support trendline value at the last bar (short entry trigger)
        apex_distance_bars – bars until the two trendlines converge (positive = still forming)
        is_consolidating   – True when price is inside the trendlines and apex is ahead
    """
    window = df.tail(lookback).copy()
    if len(window) < lookback // 2:
        return None

    highs = window["High"].values
    lows = window["Low"].values
    n = len(window)
    indices = np.arange(n)

    # --- swing highs / lows (local extrema, ignoring first and last bar)
    sh_idx = [i for i in range(1, n - 1) if highs[i] >= highs[i - 1] and highs[i] >= highs[i + 1]]
    sl_idx = [i for i in range(1, n - 1) if lows[i] <= lows[i - 1] and lows[i] <= lows[i + 1]]

    if len(sh_idx) < min_touches or len(sl_idx) < min_touches:
        return None

    # --- fit trendlines via linear regression
    def _fit(x, y):
        A = np.vstack([x, np.ones(len(x))]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        return float(slope), float(intercept)

    slope_r, int_r = _fit(np.array(sh_idx), highs[sh_idx])
    slope_s, int_s = _fit(np.array(sl_idx), lows[sl_idx])

    # normalise slopes relative to average price to make tolerance price-agnostic
    avg_price = float(np.mean(highs))
    norm_slope_r = slope_r / avg_price
    norm_slope_s = slope_s / avg_price

    # --- classify
    asc = abs(norm_slope_r) < flatness_tol and norm_slope_s > flatness_tol
    desc = norm_slope_r < -flatness_tol and abs(norm_slope_s) < flatness_tol
    sym = norm_slope_r < -flatness_tol and norm_slope_s > flatness_tol

    if not (asc or desc or sym):
        return None

    if asc:
        pattern_type = "Ascending Triangle"
    elif desc:
        pattern_type = "Descending Triangle"
    else:
        pattern_type = "Symmetrical Triangle"

    # --- evaluate trendlines at last bar
    last_i = n - 1
    resistance_at_now = slope_r * last_i + int_r
    support_at_now = slope_s * last_i + int_s

    # height is the vertical gap at bar 0 of the window (widest point)
    height = float((slope_r * 0 + int_r) - (slope_s * 0 + int_s))
    if height <= 0:
        return None

    # apex: where slope_r * x + int_r == slope_s * x + int_s
    denom = slope_s - slope_r
    if abs(denom) < 1e-10:
        return None
    apex_abs = (int_r - int_s) / denom
    apex_distance_bars = float(apex_abs - last_i)

    last_close = float(window["Close"].iloc[-1])
    is_consolidating = (
        support_at_now <= last_close <= resistance_at_now
        and apex_distance_bars > 0
    )

    return {
        "pattern_type": pattern_type,
        "height": height,
        "resistance_at_now": float(resistance_at_now),
        "support_at_now": float(support_at_now),
        "apex_distance_bars": apex_distance_bars,
        "is_consolidating": is_consolidating,
    }


def find_pivots(df: pd.DataFrame) -> pd.DataFrame:
    """Identify pivot points: Higher High (HH), Lower Low (LL), Lower High (LH), Higher Low (HL).

    Expects lowercase 'high' and 'low' columns (unlike other functions which use 'High'/'Low').
    """
    high_diffs = df['high'].diff()
    low_diffs = df['low'].diff()

    higher_high_mask = (high_diffs > 0) & (high_diffs.shift(-1) < 0)
    lower_low_mask = (low_diffs < 0) & (low_diffs.shift(-1) > 0)
    lower_high_mask = (high_diffs < 0) & (high_diffs.shift(-1) > 0)
    higher_low_mask = (low_diffs > 0) & (low_diffs.shift(-1) < 0)

    df['signal'] = ''
    df.loc[higher_high_mask, 'signal'] = 'HH'
    df.loc[lower_low_mask, 'signal'] = 'LL'
    df.loc[lower_high_mask, 'signal'] = 'LH'
    df.loc[higher_low_mask, 'signal'] = 'HL'
    return df
