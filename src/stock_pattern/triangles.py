"""
Triangle pattern detection ported from stock-pattern (BennyThadikaran).

Detects Symmetrical, Ascending, and Descending triangles using pivot-based
geometric analysis. All patterns are detected at the last leg of formation,
before any breakout occurs.

Original source: https://github.com/BennyThadikaran/stock-pattern
License: GPL-3.0 (the algorithms, this reimplementation is for the host project)
"""

from __future__ import annotations

import logging
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten a MultiIndex column header (e.g. from yfinance ``auto_adjust=True``)
    to a plain ``[Open, High, Low, Close, Volume]`` DataFrame."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------


class Point(NamedTuple):
    x: pd.Timestamp
    y: float


class Coordinate(NamedTuple):
    start: Point
    end: Point


class Line(NamedTuple):
    line: Coordinate
    slope: float
    y_int: float


# ---------------------------------------------------------------------------
# Pivot detection
# ---------------------------------------------------------------------------


def get_max_min(
    df: pd.DataFrame,
    bars_left: int = 6,
    bars_right: int = 6,
    pivot_type: str = "both",
) -> pd.DataFrame:
    """
    Find swing highs and lows using a rolling window.

    A pivot high is a bar whose High is the maximum within a window of
    ``bars_left`` candles before and ``bars_right`` candles after it.
    Pivot lows are identified symmetrically on the Low.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns ``High``, ``Low``, ``Volume`` and a DatetimeIndex.
    bars_left : int
        Number of bars to the left of the candidate bar.
    bars_right : int
        Number of bars to the right of the candidate bar.
    pivot_type : str
        ``"high"``, ``"low"``, or ``"both"`` (default).

    Returns
    -------
    pd.DataFrame
        Columns ``P`` (price) and ``V`` (volume), indexed by timestamp.
    """
    df = _normalise_columns(df)
    window = bars_left + 1 + bars_right
    local_max_dt: list[pd.Timestamp] = []
    local_min_dt: list[pd.Timestamp] = []

    for win in df.rolling(window):
        if win.shape[0] < window:
            continue

        idx = win.index[bars_left + 1]  # centre candle

        if win.High.idxmax() == idx:
            local_max_dt.append(idx)

        if win.Low.idxmin() == idx:
            local_min_dt.append(idx)

    maxima = pd.DataFrame(df.loc[local_max_dt, ["High", "Volume"]])
    maxima.columns = ["P", "V"]

    minima = pd.DataFrame(df.loc[local_min_dt, ["Low", "Volume"]])
    minima.columns = ["P", "V"]

    if pivot_type == "high":
        return maxima
    if pivot_type == "low":
        return minima

    return pd.concat([maxima, minima], axis=0).sort_index()


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------


def _get_next_index(index: pd.DatetimeIndex, idx: pd.Timestamp) -> int:
    """Return the integer position *after* ``idx`` in the index."""
    pos = index.get_loc(idx)
    if isinstance(pos, slice):
        return pos.stop
    if isinstance(pos, int):
        return pos + 1
    raise TypeError(f"Expected int or slice from get_loc, got {type(pos)}")


def _get_prev_index(index: pd.DatetimeIndex, idx: pd.Timestamp) -> int:
    """Return the integer position *before* ``idx`` in the index."""
    pos = index.get_loc(idx)
    if isinstance(pos, slice):
        return pos.stop
    if isinstance(pos, int):
        return pos - 1
    raise TypeError(f"Expected int or slice from get_loc, got {type(pos)}")


# ---------------------------------------------------------------------------
# Trendline fitting
# ---------------------------------------------------------------------------


def generate_trend_line(
    series: pd.Series,
    date1: pd.Timestamp,
    date2: pd.Timestamp,
) -> Line:
    """
    Fit a straight line through two points on a series with a DatetimeIndex.

    Returns a ``Line`` whose ``.line`` Coordinate spans from ``date1`` to the
    last bar in the series.
    """
    index = series.index
    p1 = float(series[date1])
    p2 = float(series[date2])
    d1 = index.get_loc(date1)
    d2 = index.get_loc(date2)

    last_idx = index[-1]
    last_idx_pos = index.get_loc(last_idx)

    # slope = Δy / Δx
    m = (p2 - p1) / (d2 - d1)
    # y-intercept: b = y - mx
    y_intercept = p1 - m * d1

    return Line(
        line=Coordinate(
            start=Point(x=date1, y=m * d1 + y_intercept),
            end=Point(x=last_idx, y=m * last_idx_pos + y_intercept),
        ),
        slope=m,
        y_int=y_intercept,
    )


# ---------------------------------------------------------------------------
# Triangle classification
# ---------------------------------------------------------------------------


def _classify_triangle(
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
    f: float,
    avg_bar_length: float,
) -> Optional[str]:
    """
    Classify a six-point pivot sequence as a triangle pattern.

    The six points are labelled A–F walking forward through pivot highs and
    lows (see ``find_triangles`` for the extraction order).

    Parameters
    ----------
    a..f : float
        Price levels at the six pivot / close points.
    avg_bar_length : float
        Median (High - Low) over the A–D window. Used as a flatness tolerance
        for trendline alignment checks.

    Returns
    -------
    Optional[str]
        ``"Ascending"``, ``"Descending"``, ``"Symmetric"``, or ``None``.
    """
    # --- Ascending: flat upper trendline (A≈C≈E), rising lower (B < D < F) ---
    ac_flat = abs(a - c) <= avg_bar_length
    ce_flat = abs(c - e) <= avg_bar_length
    if ac_flat and ce_flat and b < d < f < e:
        return "Ascending"

    # --- Descending: flat lower trendline (B≈D), falling upper (A > C > E) ---
    bd_flat = abs(b - d) <= avg_bar_length
    if bd_flat and a > c > e > f and f >= d:
        return "Descending"

    # --- Symmetrical: converging trendlines ---
    if a > c > e and b < d < f and e > f:
        return "Symmetric"

    return None


def _classify_triangle_4p(
    a: float,
    b: float,
    c: float,
    d: float,
    f: float,
    avg_bar_length: float,
) -> Optional[str]:
    """
    Classify a four-point sequence (no third high E available after D).

    This handles triangles whose last pivot before breakout is a low rather
    than a high — common in descending and symmetric triangles.
    """
    # --- Ascending: flat upper (A≈C), rising lower (B < D) ---
    if abs(a - c) <= avg_bar_length and b < d:
        return "Ascending"

    # --- Descending: falling upper (A > C), flat lower (B≈D) ---
    if a > c and abs(b - d) <= avg_bar_length and f >= d - avg_bar_length:
        return "Descending"

    # --- Symmetric: falling upper (A > C), rising lower (B < D) ---
    if a > c > f and b < d < f:
        return "Symmetric"

    return None


# ---------------------------------------------------------------------------
# Main detection entry point
# ---------------------------------------------------------------------------


def find_triangles(
    df: pd.DataFrame,
    bars_left: int = 6,
    bars_right: int = 6,
) -> list[dict]:
    """
    Detect ALL symmetrical, ascending, or descending triangle patterns in OHLC data.

    The algorithm extracts separate pivot-high and pivot-low lists, then walks
    each high as a candidate A, finds the next low after it as B, the next
    high after B as C, and so on through E.  This keeps the upper-trendline
    points (A, C, E) anchored to actual swing highs and prevents C from
    jumping to the far end of the data in trending markets.

    Detection fires at the **last leg** of the pattern — before any breakout
    has occurred.  Multiple non-overlapping triangles may be returned, each
    anchored to a different pivot high.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with columns ``Open``, ``High``, ``Low``, ``Close``,
        ``Volume`` and a ``DatetimeIndex``.
    bars_left : int
        Left-side window for pivot detection (default 6).
    bars_right : int
        Right-side window for pivot detection (default 6).

    Returns
    -------
    list[dict]
        List of triangle result dicts (empty if none found). Each dict has keys
        ``sym``, ``pattern``, ``alt_name``, ``start``, ``end``,
        ``slope_upper``, ``slope_lower``, ``points``, ``extra_points``.
    """
    df = _normalise_columns(df)

    if df.empty:
        return None

    # --- separate pivot highs and pivot lows ---
    highs = get_max_min(df, bars_left=bars_left, bars_right=bars_right,
                        pivot_type="high")
    lows = get_max_min(df, bars_left=bars_left, bars_right=bars_right,
                       pivot_type="low")

    if highs.empty or lows.empty or len(highs) < 3 or len(lows) < 2:
        return []

    results: list[dict] = []

    ticker = df.attrs.get("ticker", "??")

    # Walk pivot highs in order; each is a candidate A
    for a_pos in range(len(highs) - 2):
        a_idx: pd.Timestamp = highs.index[a_pos]  # type: ignore[assignment]
        a = float(highs.iloc[a_pos]["P"])

        # --- B candidates: try first 2 pivot lows *after* A ---
        after_a = lows.loc[a_idx:]
        # drop the first if it shares A's date (same bar is both high and low)
        b_start = 1 if (not after_a.empty and after_a.index[0] == a_idx) else 0
        b_end = min(b_start + 2, len(after_a))
        if b_end - b_start < 1:
            continue

        for b_off in range(b_start, b_end):
            b_idx: pd.Timestamp = after_a.index[b_off]  # type: ignore[assignment]
            b = float(after_a.iloc[b_off]["P"])

            # --- C candidates: try first 3 pivot highs *after* B ---
            after_b = highs.loc[b_idx:]
            c_end = min(3, len(after_b))
            if c_end < 1:
                continue

            for c_offset in range(c_end):
                c_idx = after_b.index[c_offset]
                c = float(after_b.iloc[c_offset]["P"])

                # --- D candidates: try first 2 pivot lows *after* C ---
                after_c = lows.loc[c_idx:]
                if after_c.empty:
                    continue
                d_start = 1 if after_c.index[0] == c_idx else 0
                d_end = min(d_start + 2, len(after_c))
                if d_end - d_start < 1:
                    continue

                for d_off in range(d_start, d_end):
                    d_idx = after_c.index[d_off]
                    d = float(after_c.loc[d_idx, "P"])

                    # --- adaptive tolerance ---
                    df_slice = df.loc[a_idx:d_idx]
                    if len(df_slice) < 5:
                        continue
                    avg_bar_length: float = float(
                        (df_slice["High"] - df_slice["Low"]).median()
                    )
                    if avg_bar_length <= 0:
                        continue

                    # --- E = first pivot high *after* D (if any) ---
                    after_d = highs.loc[d_idx:]
                    has_e = bool(
                        not after_d.empty
                        and not (len(after_d) == 1 and after_d.index[0] == d_idx)
                    )

                    if has_e:
                        if after_d.index[0] == d_idx:
                            if len(after_d) < 2:
                                has_e = False
                            else:
                                e_idx = after_d.index[1]
                        else:
                            e_idx = after_d.index[0]

                    if has_e:
                        e = float(after_d.loc[e_idx, "P"])
                        f_idx = e_idx
                        f = float(df.at[f_idx, "Close"])
                        endpoint_idx = e_idx
                        triangle = _classify_triangle(a, b, c, d, e, f, avg_bar_length)
                    else:
                        # No high after D — try 4-point classification
                        e_idx = d_idx
                        e = d
                        f_idx = d_idx
                        f = float(df.at[f_idx, "Close"])
                        endpoint_idx = d_idx
                        triangle = _classify_triangle_4p(a, b, c, d, f, avg_bar_length)

                    if triangle is None:
                        continue

                    # --- pivot anchoring check ---
                    if has_e:
                        if (
                            b == df.at[b_idx, "High"]
                            or c == df.at[c_idx, "Low"]
                            or d == df.at[d_idx, "High"]
                            or e == df.at[e_idx, "Low"]
                        ):
                            continue
                    else:
                        if (
                            b == df.at[b_idx, "High"]
                            or c == df.at[c_idx, "Low"]
                            or d == df.at[d_idx, "High"]
                        ):
                            continue

                    # --- duration ratio check (use endpoint = E or D) ---
                    upper_duration = (endpoint_idx - a_idx).days
                    lower_duration = (endpoint_idx - b_idx).days
                    ratio_limit = 2.5 if not has_e else 1.8
                    if (
                        max(upper_duration, lower_duration)
                        / max(min(upper_duration, lower_duration), 1)
                        > ratio_limit
                    ):
                        continue

                    upper = generate_trend_line(df.High, a_idx, c_idx)
                    lower = generate_trend_line(df.Low, b_idx, d_idx)

                    # trendlines must not have crossed by endpoint
                    ep_pos = df.index.get_loc(endpoint_idx)
                    if isinstance(ep_pos, slice):
                        ep_pos = ep_pos.stop
                    if upper.slope * ep_pos + upper.y_int < lower.slope * ep_pos + lower.y_int:
                        continue

                    # reject nearly-flat trendlines
                    if triangle == "Ascending" and (
                        upper.slope > 0.1 and lower.slope < 0.2
                    ):
                        continue
                    if triangle == "Descending" and (
                        lower.slope < -0.1 and upper.slope > -0.2
                    ):
                        continue
                    if triangle == "Symmetric" and (
                        upper.slope > -0.2 and lower.slope < 0.2
                    ):
                        continue

                    # --- price must not breach trendlines between A and endpoint ---
                    a_int = df.index.get_loc(a_idx)
                    ep_int = df.index.get_loc(endpoint_idx)
                    if isinstance(a_int, slice):
                        a_int = a_int.start
                    if isinstance(ep_int, slice):
                        ep_int = ep_int.stop
                    pos_w = np.arange(a_int, ep_int + 1)
                    upper_line = upper.slope * pos_w + upper.y_int
                    lower_line = lower.slope * pos_w + lower.y_int
                    window_close = df.iloc[a_int : ep_int + 1]["Close"]
                    if (window_close > upper_line).any() or (
                        window_close < lower_line
                    ).any():
                        continue

                    log.debug("%s — %s triangle detected", ticker, triangle)

                    results.append({
                        "sym": ticker,
                        "pattern": "TRNG",
                        "alt_name": triangle,
                        "complete": has_e,  # True if 5-point (real E), False if 4-point fallback
                        "start": a_idx,
                        "end": endpoint_idx,
                        "df_start": df.index[0],
                        "df_end": df.index[-1],
                        "slope_upper": upper.slope,
                        "slope_lower": lower.slope,
                        "points": {
                            "A": (a_idx, a),
                            "B": (b_idx, b),
                            "C": (c_idx, c),
                            "D": (d_idx, d),
                            "E": (e_idx, e),
                            "F": (f_idx, f),
                        },
                        "extra_points": {
                            "upper_start": upper.line.start,
                            "upper_end": upper.line.end,
                            "lower_start": lower.line.start,
                            "lower_end": lower.line.end,
                        },
                    })

    # Dedup: keep at most one triangle per (end, pattern) — the one with the
    # earliest A (longest formation, tightest trendlines). Prefer 5-point over
    # 4-point when the same A produces both.
    if not results:
        return results

    # Sort: earliest A first, 5-point before 4-point
    results.sort(key=lambda r: (r["start"], not r.get("complete", True)))

    # Pass 1: one triangle per A timestamp
    seen_a: set[pd.Timestamp] = set()
    by_a: list[dict] = []
    for r in results:
        if r["start"] not in seen_a:
            seen_a.add(r["start"])
            by_a.append(r)

    # Pass 2: one triangle per (end, alt_name) — keep earliest A (first in sorted order)
    seen_ep: dict[tuple, int] = {}  # (end_ts, pattern) -> index in by_a
    unique: list[dict] = []
    for r in by_a:
        key = (r["end"], r["alt_name"])
        if key not in seen_ep:
            seen_ep[key] = len(unique)
            unique.append(r)
        # else: later A with same end+pattern → skip (keep the earlier one)

    return unique


# ---------------------------------------------------------------------------
# Trade level calculation
# ---------------------------------------------------------------------------


def calculate_trade_levels(
    result: dict,
    df: pd.DataFrame,
    entry_pct: float = 0.02,
) -> dict:
    """
    Calculate entry, stop-loss, and take-profit levels for a detected triangle.

    Parameters
    ----------
    result : dict
        The dict returned by ``find_triangles()``.
    df : pd.DataFrame
        The same OHLC DataFrame passed to ``find_triangles()`` (needed for
        integer-position lookups on the trendlines).
    entry_pct : float
        Fraction of the triangle height above/below the trendline to place
        the entry order (default 2%).

    Returns
    -------
    dict
        Keys: ``direction``, ``entry``, ``stop_loss``, ``take_profit_half``,
        ``take_profit_full``, ``height``.  For Symmetric triangles, also
        includes ``entry_short``, ``stop_loss_short``,
        ``take_profit_half_short``, ``take_profit_full_short``.
    """
    df = _normalise_columns(df)

    triangle_type: str = result["alt_name"]
    points: dict = result["points"]
    upper_slope: float = result["slope_upper"]
    lower_slope: float = result["slope_lower"]
    endpoint_ts = result["end"]

    # --- integer positions for slope-intercept maths ---
    a_ts = points["A"][0]
    b_ts = points["B"][0]
    d_ts = points["D"][0]

    def _pos(ts):
        p = df.index.get_loc(ts)
        return p.start if isinstance(p, slice) else p

    a_pos = _pos(a_ts)
    b_pos = _pos(b_ts)
    end_pos = _pos(endpoint_ts)
    if isinstance(end_pos, slice):
        end_pos = end_pos.stop

    # y-intercepts (in integer-position space)
    upper_y_int = points["A"][1] - upper_slope * a_pos
    lower_y_int = points["B"][1] - lower_slope * b_pos

    # trendline values at pattern end
    upper_at_end = upper_slope * end_pos + upper_y_int
    lower_at_end = lower_slope * end_pos + lower_y_int

    # height = widest distance between trendlines (at pattern start)
    start_pos = min(a_pos, b_pos)
    height = abs(
        (upper_slope * start_pos + upper_y_int)
        - (lower_slope * start_pos + lower_y_int)
    )
    if height <= 0:
        height = abs(points["A"][1] - points["B"][1])

    # --- direction and entry ---
    # Ascending (flat top, rising floor) → bearish breakdown → SHORT
    # Descending (falling top, flat floor) → bullish breakout → LONG
    if triangle_type == "Ascending":
        direction = "SHORT"
        entry = lower_at_end - entry_pct * height
        stop = upper_at_end + 0.01 * height
        tp_half = entry - height / 2
        tp_full = entry - height
        return {
            "direction": direction,
            "entry": round(entry, 2),
            "stop_loss": round(stop, 2),
            "take_profit_half": round(tp_half, 2),
            "take_profit_full": round(tp_full, 2),
            "height": round(height, 2),
        }
    elif triangle_type == "Descending":
        direction = "LONG"
        entry = upper_at_end + entry_pct * height
        stop = lower_at_end - 0.01 * height
        tp_half = entry + height / 2
        tp_full = entry + height
        return {
            "direction": direction,
            "entry": round(entry, 2),
            "stop_loss": round(stop, 2),
            "take_profit_half": round(tp_half, 2),
            "take_profit_full": round(tp_full, 2),
            "height": round(height, 2),
        }
    else:  # Symmetric — can break either way; show both long and short levels
        entry_long = upper_at_end + entry_pct * height
        stop_long = lower_at_end - 0.01 * height
        tp_half_long = entry_long + height / 2
        tp_full_long = entry_long + height

        entry_short = lower_at_end - entry_pct * height
        stop_short = upper_at_end + 0.01 * height
        tp_half_short = entry_short - height / 2
        tp_full_short = entry_short - height

        return {
            "direction": "SYMMETRIC",
            "entry": round(entry_long, 2),
            "stop_loss": round(stop_long, 2),
            "take_profit_half": round(tp_half_long, 2),
            "take_profit_full": round(tp_full_long, 2),
            "entry_short": round(entry_short, 2),
            "stop_loss_short": round(stop_short, 2),
            "take_profit_half_short": round(tp_half_short, 2),
            "take_profit_full_short": round(tp_full_short, 2),
            "height": round(height, 2),
        }


# ---------------------------------------------------------------------------
# Volume trend helper
# ---------------------------------------------------------------------------


def compute_volume_trend(
    df: pd.DataFrame,
    result: dict,
) -> dict:
    """
    Compute the volume trend (linear regression slope) over a triangle's
    formation period (A → endpoint).

    Returns a dict with ``slope`` (float), ``trend`` ("declining" / "flat" /
    "rising"), and ``r_squared`` (float).
    """
    a_ts = result["points"]["A"][0]
    end_ts = result["end"]

    try:
        a_pos = df.index.get_loc(a_ts)
        e_pos = df.index.get_loc(end_ts)
    except KeyError:
        return {"slope": 0.0, "trend": "unknown", "r_squared": 0.0}

    if isinstance(a_pos, slice):
        a_pos = a_pos.start
    if isinstance(e_pos, slice):
        e_pos = e_pos.stop

    if e_pos <= a_pos + 2:
        return {"slope": 0.0, "trend": "flat", "r_squared": 0.0}

    vol_slice = df["Volume"].iloc[a_pos : e_pos + 1]
    xs = list(range(a_pos, e_pos + 1))
    ys = vol_slice.values.astype(float)

    n = len(xs)
    x_mean = sum(xs) / n
    y_mean = ys.mean()
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den_x = sum((x - x_mean) ** 2 for x in xs)
    den_y = sum((y - y_mean) ** 2 for y in ys)

    if den_x == 0:
        return {"slope": 0.0, "trend": "flat", "r_squared": 0.0}

    slope = num / den_x

    # R²
    if den_y == 0:
        r2 = 0.0
    else:
        r2 = (num ** 2) / (den_x * den_y)

    # Classify trend relative to mean volume scale
    rel = abs(slope) / max(y_mean, 1) * 100 if y_mean > 0 else 0
    if rel < 0.5:
        trend = "flat"
    elif slope < 0:
        trend = "declining"
    else:
        trend = "rising"

    return {"slope": round(float(slope), 2), "trend": trend, "r_squared": round(float(r2), 4)}
