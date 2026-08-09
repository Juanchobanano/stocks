"""
Minimal pattern plotter — renders a single triangle detection result
as a candlestick chart with trendlines and A–F labels.

Requires mplfinance (``pip install mplfinance``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mplfinance as mpf
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
# Public API
# ---------------------------------------------------------------------------


def plot_triangle(
    df: pd.DataFrame,
    result: dict | list[dict],
    *,
    save_to: Optional[str | Path] = None,
    bars_before: int = 30,
    entry_pct: float = 0.02,
    figscale: float = 1.2,
) -> None:
    """
    Render one or more triangle patterns on a candlestick chart with trade levels.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with columns ``Open``, ``High``, ``Low``, ``Close``,
        ``Volume`` and a ``DatetimeIndex``.
    result : dict or list[dict]
        A single result dict from ``find_triangles()`` or a list of them.
    save_to : str or Path, optional
        File path to save the chart (PNG). If ``None``, displays interactively.
    bars_before : int
        How many bars before the earliest point A to include.
    entry_pct : float
        Fraction of triangle height above/below trendline for entry (default 2%).
    figscale : float
        Scaling factor for the figure size.
    """
    from src.stock_pattern.triangles import calculate_trade_levels

    df = _normalise_columns(df)

    # Normalise to list
    if isinstance(result, dict):
        results = [result]
    else:
        results = list(result)

    if not results:
        log.warning("No triangle results to plot.")
        return

    sym = results[0].get("sym", "??")

    # --- colour palette for multiple triangles ---
    palette = ["midnightblue", "darkorange", "forestgreen", "darkviolet", "crimson"]

    # --- determine chart slice (earliest A to last bar) ---
    earliest_a_idx = None
    for r in results:
        a_ts = r["points"]["A"][0]
        if earliest_a_idx is None or a_ts < earliest_a_idx:
            earliest_a_idx = a_ts

    a_pos = df.index.get_loc(earliest_a_idx)
    if isinstance(a_pos, slice):
        a_pos = a_pos.start
    start = max(0, a_pos - bars_before)
    end = len(df) - 1
    chart_df = df.iloc[start : end + 1]

    if chart_df.empty:
        log.warning("Empty chart slice — check your data.")
        return

    # --- compute trade levels for the *last* (most recent) triangle ---
    last_result = max(results, key=lambda r: r["end"])
    levels = calculate_trade_levels(last_result, df, entry_pct=entry_pct)

    # --- build title ---
    pattern_names = [r["alt_name"] for r in results]
    if len(results) == 1:
        status = "Completed" if results[0].get("complete", True) else "Forming"
        title = f"{sym} — {pattern_names[0]} ({levels['direction']}) — {status}"
    else:
        title = f"{sym} — {len(results)} triangles: {', '.join(pattern_names)}"

    # --- build alines for every triangle ---
    alines_data: list[list[tuple]] = []
    alines_colors: list[str] = []
    alines_lw: list[float] = []
    alines_alpha: list[float] = []

    for i, r in enumerate(results):
        color = palette[i % len(palette)]
        end_ts = r["end"]
        end_orig = df.index.get_loc(end_ts)
        if isinstance(end_orig, slice):
            end_orig = end_orig.stop

        extra = r["extra_points"]

        # upper trendline
        u_start_ts = extra["upper_start"].x
        u_slope = r["slope_upper"]
        u_start_orig = df.index.get_loc(u_start_ts)
        if isinstance(u_start_orig, slice):
            u_start_orig = u_start_orig.start
        u_y_int = extra["upper_start"].y - u_slope * u_start_orig
        upper_line = [
            (u_start_ts, extra["upper_start"].y),
            (end_ts, u_slope * end_orig + u_y_int),
        ]

        # lower trendline
        l_start_ts = extra["lower_start"].x
        l_slope = r["slope_lower"]
        l_start_orig = df.index.get_loc(l_start_ts)
        if isinstance(l_start_orig, slice):
            l_start_orig = l_start_orig.start
        l_y_int = extra["lower_start"].y - l_slope * l_start_orig
        lower_line = [
            (l_start_ts, extra["lower_start"].y),
            (end_ts, l_slope * end_orig + l_y_int),
        ]

        alines_data.append(upper_line)
        alines_data.append(lower_line)
        alines_colors.extend([color, color])
        alines_lw.extend([1.5, 1.5])
        alines_alpha.extend([0.8, 0.8])

    alines = {
        "alines": alines_data,
        "colors": alines_colors,
        "linewidths": alines_lw,
        "alpha": alines_alpha,
    }

    # --- horizontal trade level lines (from last triangle only) ---
    is_symmetric = levels.get("direction") == "SYMMETRIC"
    if is_symmetric:
        hlines = dict(
            hlines=[
                levels["entry"],              levels["stop_loss"],
                levels["take_profit_half"],   levels["take_profit_full"],
                levels["entry_short"],        levels["stop_loss_short"],
                levels["take_profit_half_short"], levels["take_profit_full_short"],
            ],
            colors=[
                "dodgerblue", "crimson", "forestgreen", "forestgreen",
                "orange", "orangered", "limegreen", "limegreen",
            ],
            linestyle=["--", "--", "-.", "-.", "--", "--", "-.", "-."],
            linewidths=[1.2, 1.2, 0.9, 0.9, 1.2, 1.2, 0.9, 0.9],
            alpha=0.7,
        )
    else:
        hlines = dict(
            hlines=[
                levels["entry"],
                levels["stop_loss"],
                levels["take_profit_half"],
                levels["take_profit_full"],
            ],
            colors=["dodgerblue", "crimson", "forestgreen", "forestgreen"],
            linestyle=["--", "--", "-.", "-."],
            linewidths=[1.2, 1.2, 0.9, 0.9],
            alpha=0.7,
        )

    # --- plot ---
    fig, axs = mpf.plot(
        chart_df,
        type="candle",
        style="tradingview",
        volume=True,
        alines=alines,
        hlines=hlines,
        title=title,
        returnfig=True,
        figscale=figscale,
        warn_too_much_data=100000,
    )

    main_ax = axs[0] if isinstance(axs, (list, tuple)) else axs

    # --- annotate A–F points for every triangle ---
    for i, r in enumerate(results):
        color = palette[i % len(palette)]
        points = r["points"]
        # Build label prefix: "1", "2", ... for 2+ triangles; empty for single
        prefix = f"{i + 1}" if len(results) > 1 else ""

        for label in ("A", "B", "C", "D", "E", "F"):
            x_ts, y = points[label]
            try:
                x_pos = chart_df.index.get_loc(x_ts)
            except KeyError:
                continue

            display_label = f"{prefix}{label}"

            if label == "F":
                offset = (15, 0)
            elif y >= chart_df.at[x_ts, "High"] * 0.999:
                offset = (0, 15)
            else:
                offset = (0, -15)

            main_ax.annotate(
                display_label,
                xy=(x_pos, y),
                xytext=offset,
                textcoords="offset pixels",
                fontweight="bold",
                ha="center",
                color=color,
            )

    # --- label trade levels at the right edge ---
    last_x = len(chart_df) - 1
    entry = levels["entry"]
    if entry > 0:
        tp_half_pct = abs(levels["take_profit_half"] - entry) / entry * 100
        tp_full_pct = abs(levels["take_profit_full"] - entry) / entry * 100
        stop_pct = abs(levels["stop_loss"] - entry) / entry * 100
    else:
        tp_half_pct = tp_full_pct = stop_pct = 0.0

    if is_symmetric and entry > 0:
        stop_s_pct = abs(levels["stop_loss_short"] - levels["entry_short"]) / levels["entry_short"] * 100
        tp_half_s_pct = abs(levels["take_profit_half_short"] - levels["entry_short"]) / levels["entry_short"] * 100
        tp_full_s_pct = abs(levels["take_profit_full_short"] - levels["entry_short"]) / levels["entry_short"] * 100
        level_labels = [
            (levels["take_profit_full"], f"TP full L  {levels['take_profit_full']:.2f}  (+{tp_full_pct:.1f}%)", "forestgreen"),
            (levels["take_profit_half"], f"TP half L  {levels['take_profit_half']:.2f}  (+{tp_half_pct:.1f}%)", "forestgreen"),
            (levels["entry"], f"Entry L  {levels['entry']:.2f}", "dodgerblue"),
            (levels["stop_loss"], f"Stop L  {levels['stop_loss']:.2f}  (-{stop_pct:.1f}%)", "crimson"),
            (levels["take_profit_full_short"], f"TP full S  {levels['take_profit_full_short']:.2f}  (+{tp_full_s_pct:.1f}%)", "limegreen"),
            (levels["take_profit_half_short"], f"TP half S  {levels['take_profit_half_short']:.2f}  (+{tp_half_s_pct:.1f}%)", "limegreen"),
            (levels["entry_short"], f"Entry S  {levels['entry_short']:.2f}", "orange"),
            (levels["stop_loss_short"], f"Stop S  {levels['stop_loss_short']:.2f}  (-{stop_s_pct:.1f}%)", "orangered"),
        ]
    else:
        level_labels = [
            (levels["take_profit_full"], f"TP full  {levels['take_profit_full']:.2f}  (+{tp_full_pct:.1f}%)", "forestgreen"),
            (levels["take_profit_half"], f"TP half  {levels['take_profit_half']:.2f}  (+{tp_half_pct:.1f}%)", "forestgreen"),
            (levels["entry"], f"Entry  {levels['entry']:.2f}", "dodgerblue"),
            (levels["stop_loss"], f"Stop  {levels['stop_loss']:.2f}  (-{stop_pct:.1f}%)", "crimson"),
        ]
    for price, text, color in level_labels:
        main_ax.annotate(
            text,
            xy=(last_x, price),
            xytext=(8, 0),
            textcoords="offset pixels",
            fontsize=7,
            va="center",
            color=color,
            fontweight="bold",
        )

    # --- volume trend lines (confirming volume contraction during formation) ---
    vol_ax = None
    if isinstance(axs, (list, tuple)):
        for a in axs:
            label = a.get_ylabel() if hasattr(a, "get_ylabel") else ""
            if label.startswith("Volume"):
                vol_ax = a
                break
    if vol_ax is not None:
        for i, r in enumerate(results):
            color = palette[i % len(palette)]
            a_ts = r["points"]["A"][0]
            end_ts = r["end"]
            try:
                a_vpos = chart_df.index.get_loc(a_ts)
                e_vpos = chart_df.index.get_loc(end_ts)
            except KeyError:
                continue
            if isinstance(a_vpos, slice):
                a_vpos = a_vpos.start
            if isinstance(e_vpos, slice):
                e_vpos = e_vpos.stop
            if e_vpos <= a_vpos:
                continue
            vol_slice = chart_df["Volume"].iloc[a_vpos : e_vpos + 1]
            xs = range(a_vpos, e_vpos + 1)
            ys = vol_slice.values.astype(float)
            if len(ys) < 3:
                continue
            # linear regression: vol = m * pos + b
            x_mean = sum(xs) / len(xs)
            y_mean = ys.mean()
            num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
            den = sum((x - x_mean) ** 2 for x in xs)
            m = num / den if den != 0 else 0.0
            b = y_mean - m * x_mean
            trend_line = [m * x + b for x in (a_vpos, e_vpos)]
            vol_ax.plot(
                [a_vpos, e_vpos],
                trend_line,
                color=color,
                linewidth=1.2,
                linestyle="--",
                alpha=0.7,
            )
            # small label at the right end
            direction = "↓" if m < 0 else "↑"
            prefix = f"{i + 1}" if len(results) > 1 else ""
            vol_ax.annotate(
                f"{prefix}Vol {direction}",
                xy=(e_vpos, trend_line[1]),
                xytext=(4, 0),
                textcoords="offset pixels",
                fontsize=6,
                va="center",
                color=color,
                fontweight="bold",
            )

    main_ax.xaxis.set_visible(False)

    if save_to is not None:
        path = Path(save_to)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info("Chart saved to %s", path)
        plt.close(fig)
    else:
        mpf.show()


def plot_candlestick(
    df: pd.DataFrame,
    *,
    save_to: Optional[str | Path] = None,
    title: Optional[str] = None,
    figscale: float = 1.2,
) -> None:
    """
    Render a plain candlestick chart — no pattern overlays.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with columns ``Open``, ``High``, ``Low``, ``Close``,
        ``Volume`` and a ``DatetimeIndex``.
    save_to : str or Path, optional
        File path to save the chart (PNG). If ``None``, displays interactively.
    title : str, optional
        Chart title. Defaults to the ticker in ``df.attrs['ticker']``.
    figscale : float
        Scaling factor for the figure size.
    """
    df = _normalise_columns(df)

    ticker = df.attrs.get("ticker", "??")
    if title is None:
        title = f"{ticker}"

    fig, _axs = mpf.plot(
        df,
        type="candle",
        style="tradingview",
        volume=True,
        title=title,
        returnfig=True,
        figscale=figscale,
        warn_too_much_data=100000,
    )

    if save_to is not None:
        path = Path(save_to)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info("Chart saved to %s", path)
        plt.close(fig)
    else:
        mpf.show()
