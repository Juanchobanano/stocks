#!/usr/bin/env python3
"""Scan for triangle patterns and plot each one found.

Usage:
    python scan_triangles.py               # uses defaults from StockPatternConfig
    python scan_triangles.py --start 2025-06-01 --end 2026-08-07
    python scan_triangles.py --symbols AAPL MSFT GOOGL
    python scan_triangles.py --min-beta 1.5
"""

import argparse
import json
from datetime import datetime as _dt, timedelta as _td
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from src.stock_pattern import (
    StockPatternConfig,
    calculate_trade_levels,
    compute_volume_trend,
    find_triangles,
    plot_candlestick,
    plot_triangle,
)
from src.stock_pattern.config import SYMBOL_SECTORS


def compute_betas(
    symbols: list[str],
    start: str,
    end: str,
) -> dict[str, float]:
    """Compute beta for each symbol vs SPY over the given date range."""
    print("Computing betas vs SPY ...", end=" ", flush=True)

    # Download all at once
    all_syms = ["SPY"] + symbols
    data = yf.download(all_syms, start=start, end=end, auto_adjust=True, progress=False)

    if data.empty:
        print("no data")
        return {}

    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        close = data.xs("Close", axis=1, level=0)
    else:
        close = data["Close"]

    returns = close.pct_change(fill_method=None).dropna(how="all")
    spy_ret = returns["SPY"]
    spy_var = float(spy_ret.var())

    betas: dict[str, float] = {}
    for sym in symbols:
        if sym not in returns.columns:
            continue
        sym_ret = returns[sym].dropna()
        common = sym_ret.index.intersection(spy_ret.index)
        if len(common) < 20 or spy_var == 0:
            continue
        cov = float(sym_ret.loc[common].cov(spy_ret.loc[common]))
        betas[sym] = round(cov / spy_var, 2)

    print(f"{len(betas)} computed")
    return betas


def _generate_pine(
    sym: str,
    result: dict,
    levels: dict,
    extra: dict,
    pattern: str,
    direction: str,
    status: str,
    start_date: str,
    end_date: str,
) -> str:
    """Generate a Pine Script v5 indicator for a single triangle pattern."""
    points = result["points"]

    def _ts(date_str: str) -> str:
        y, m, d = date_str.split("-")
        return f"timestamp({int(y)}, {int(m)}, {int(d)}, 0, 0)"

    def _fmt_p(v: float) -> str:
        return f"{v:.2f}"

    lines: list[str] = []

    # ── header ──
    direction_label = {"LONG": "LONG (bullish)", "SHORT": "SHORT (bearish)", "SYMMETRIC": "SYMMETRIC (both)"}.get(direction, direction)
    status_label = "FORMING" if not result.get("complete", True) else "COMPLETED"
    pattern_desc = {
        "Ascending":  "flat top A→C, rising floor B→D",
        "Descending": "flat floor B→D, falling ceiling A→C",
        "Symmetric":  "converging lines A→C and B→D",
    }.get(pattern, "")

    lines.append("//@version=5")
    lines.append(f"// {sym} — {pattern} Triangle △  ({direction_label} setup)")
    lines.append(f"// Pattern:  {pattern_desc}")
    lines.append(f"// Status:   {status_label}")
    lines.append(f"// Direction: {direction}")
    lines.append(f'indicator("{sym} Triangle", overlay=true, max_lines_count=500, max_labels_count=500)')

    # ── timestamps ──
    lines.append("")
    lines.append("// ── Pivot timestamps ──────────────────────────────────────────")
    a_date = str(points["A"][0].date()) if hasattr(points["A"][0], "date") else str(points["A"][0])[:10]
    b_date = str(points["B"][0].date()) if hasattr(points["B"][0], "date") else str(points["B"][0])[:10]
    c_date = str(points["C"][0].date()) if hasattr(points["C"][0], "date") else str(points["C"][0])[:10]
    d_date = str(points["D"][0].date()) if hasattr(points["D"][0], "date") else str(points["D"][0])[:10]
    f_date = str(points["F"][0].date()) if hasattr(points["F"][0], "date") else str(points["F"][0])[:10]

    lines.append(f"tA = {_ts(a_date)}")
    lines.append(f"tB = {_ts(b_date)}")
    lines.append(f"tC = {_ts(c_date)}")
    lines.append(f"tD = {_ts(d_date)}")

    # ── prices ──
    lines.append("")
    lines.append("// ── Pivot prices ──────────────────────────────────────────────")
    lines.append(f"pA = {_fmt_p(float(points['A'][1]))}")
    lines.append(f"pB = {_fmt_p(float(points['B'][1]))}")
    lines.append(f"pC = {_fmt_p(float(points['C'][1]))}")
    lines.append(f"pD = {_fmt_p(float(points['D'][1]))}")
    lines.append(f"pF = {_fmt_p(float(points['F'][1]))}")

    # ── trendline endpoints ──
    lines.append("")
    lines.append("// ── Trendline endpoints at current bar ────────────────────────")
    upper_end_y = extra.get("upper_end")
    lower_end_y = extra.get("lower_end")
    if upper_end_y is not None:
        lines.append(f"pUpperEnd = {_fmt_p(upper_end_y.y)}")
    if lower_end_y is not None:
        lines.append(f"pLowerEnd = {_fmt_p(lower_end_y.y)}")

    # ── trendlines ──
    lines.append("")
    lines.append("// ── TRENDLINES — from first pivot to current bar ──────────────")
    lines.append("if barstate.islast")
    if upper_end_y is not None:
        lines.append("    // Upper: first high → current bar")
        lines.append(f"    line.new(tA, pA, last_bar_time, pUpperEnd, xloc=xloc.bar_time,")
        lines.append(f'             color=color.white, width=4)')
    if lower_end_y is not None:
        lines.append("    // Lower: first low → current bar")
        lines.append(f"    line.new(tB, pB, last_bar_time, pLowerEnd, xloc=xloc.bar_time,")
        lines.append(f'             color=color.new(#ffff00, 0), width=4)')

    # ── horizontal trade levels ──
    lines.append("")
    lines.append("// ── HORIZONTAL TRADE LEVELS ───────────────────────────────────")
    tp_full_pct = abs(levels["take_profit_full"] - levels["entry"]) / levels["entry"] * 100 if levels["entry"] > 0 else 0
    tp_half_pct = abs(levels["take_profit_half"] - levels["entry"]) / levels["entry"] * 100 if levels["entry"] > 0 else 0
    stop_pct = abs(levels["stop_loss"] - levels["entry"]) / levels["entry"] * 100 if levels["entry"] > 0 else 0
    lines.append(f'hline({_fmt_p(levels["take_profit_full"])}, "TP full  +{tp_full_pct:.1f}%",  color.new(#ffff00, 20), hline.style_dashed, 4)')
    lines.append(f'hline({_fmt_p(levels["take_profit_half"])}, "TP half  +{tp_half_pct:.1f}%",  color.new(#ffff00, 20), hline.style_dashed, 4)')
    lines.append(f'hline({_fmt_p(levels["entry"])}, "Entry  {_fmt_p(levels["entry"])}",    color.white,             hline.style_solid,  4)')
    lines.append(f'hline({_fmt_p(levels["stop_loss"])}, "Stop  -{stop_pct:.1f}%",     color.new(#ffff00, 20),  hline.style_dashed, 4)')

    # ── symmetric short levels (only if symmetric) ──
    if direction == "SYMMETRIC" and levels.get("entry_short", 0) > 0:
        lines.append("")
        lines.append("// ── SHORT trade levels ────────────────────────────────────────")
        s_entry = levels["entry_short"]
        s_tp_full_pct = abs(levels["take_profit_full_short"] - s_entry) / s_entry * 100
        s_tp_half_pct = abs(levels["take_profit_half_short"] - s_entry) / s_entry * 100
        s_stop_pct = abs(levels["stop_loss_short"] - s_entry) / s_entry * 100
        lines.append(f'hline({_fmt_p(levels["take_profit_full_short"])}, "TP full S  +{s_tp_full_pct:.1f}%",  color.new(#ffff00, 20), hline.style_dashed, 4)')
        lines.append(f'hline({_fmt_p(levels["take_profit_half_short"])}, "TP half S  +{s_tp_half_pct:.1f}%",  color.new(#ffff00, 20), hline.style_dashed, 4)')
        lines.append(f'hline({_fmt_p(levels["entry_short"])}, "Entry S  {_fmt_p(levels["entry_short"])}",    color.white,             hline.style_solid,  4)')
        lines.append(f'hline({_fmt_p(levels["stop_loss_short"])}, "Stop S  -{s_stop_pct:.1f}%",     color.new(#ffff00, 20),  hline.style_dashed, 4)')

    # ── point labels ──
    lines.append("")
    lines.append("// ── POINT LABELS ───────────────────────────────────────────────")
    lines.append("if barstate.islast")
    lines.append("    c = color.new(color.white, 80)")
    lines.append("    // Highs — label below")
    lines.append(f"    label.new(tA, pA, \"A\", xloc=xloc.bar_time, color=c, textcolor=color.white, style=label.style_label_down, size=size.normal)")
    lines.append(f"    label.new(tC, pC, \"C\", xloc=xloc.bar_time, color=c, textcolor=color.white, style=label.style_label_down, size=size.normal)")
    lines.append("    // Lows — label above")
    lines.append(f"    label.new(tB, pB, \"B\", xloc=xloc.bar_time, color=c, textcolor=color.white, style=label.style_label_up,   size=size.normal)")
    lines.append(f"    label.new(tD, pD, \"D\", xloc=xloc.bar_time, color=c, textcolor=color.white, style=label.style_label_up,   size=size.normal)")
    lines.append(f"    label.new(tD, pD, \"E\", xloc=xloc.bar_time, color=c, textcolor=color.white, style=label.style_label_up,   size=size.normal)")
    lines.append(f"    label.new(tD, pF, \"F\", xloc=xloc.bar_time, color=c, textcolor=color.white, style=label.style_label_up,   size=size.normal)")

    # ── right-edge trade level labels ──
    # Build label data and resolve overlaps: if two label prices are within
    # 2 % of each other, the higher one is pushed above its line and the
    # lower one below its line (by adjusting the y coordinate ±1.5 %).
    lbl_defs: list[dict] = []
    lbl_defs.append({"col": 2, "price": levels["take_profit_full"],  "text": f"TP full  {_fmt_p(levels['take_profit_full'])}  (+{tp_full_pct:.1f}%)",  "tc": "#ffff00"})
    lbl_defs.append({"col": 2, "price": levels["take_profit_half"],  "text": f"TP half  {_fmt_p(levels['take_profit_half'])}  (+{tp_half_pct:.1f}%)",  "tc": "#ffff00"})
    lbl_defs.append({"col": 2, "price": levels["entry"],             "text": f"Entry L  {_fmt_p(levels['entry'])}",                                     "tc": "white"})
    lbl_defs.append({"col": 2, "price": levels["stop_loss"],         "text": f"Stop L  {_fmt_p(levels['stop_loss'])}  (-{stop_pct:.1f}%)",              "tc": "#ffff00"})
    if direction == "SYMMETRIC" and levels.get("entry_short", 0) > 0:
        lbl_defs.append({"col": 10, "price": levels["take_profit_full_short"],  "text": f"TP full S  {_fmt_p(levels['take_profit_full_short'])}  (+{s_tp_full_pct:.1f}%)",  "tc": "#ffff00"})
        lbl_defs.append({"col": 10, "price": levels["take_profit_half_short"],  "text": f"TP half S  {_fmt_p(levels['take_profit_half_short'])}  (+{s_tp_half_pct:.1f}%)",  "tc": "#ffff00"})
        lbl_defs.append({"col": 10, "price": levels["entry_short"],             "text": f"Entry S  {_fmt_p(levels['entry_short'])}",                                     "tc": "white"})
        lbl_defs.append({"col": 10, "price": levels["stop_loss_short"],         "text": f"Stop S  {_fmt_p(levels['stop_loss_short'])}  (-{s_stop_pct:.1f}%)",              "tc": "#ffff00"})

    # Sort by price, detect close pairs, assign above/below offsets
    close_threshold = 0.02  # 2% price difference → consider them close
    lbl_defs.sort(key=lambda d: d["price"])
    for i in range(len(lbl_defs) - 1):
        p_lo = lbl_defs[i]["price"]
        p_hi = lbl_defs[i + 1]["price"]
        if p_hi > 0 and (p_hi - p_lo) / p_hi < close_threshold:
            # Push higher label ABOVE its line, lower label BELOW its line
            if lbl_defs[i + 1].get("yadj") is None:
                lbl_defs[i + 1]["yadj"] = +0.025  # above
            if lbl_defs[i].get("yadj") is None:
                lbl_defs[i]["yadj"] = -0.030  # below

    lines.append("")
    lines.append("// ── RIGHT-EDGE LEVEL LABELS ────────────────────────────────────")
    lines.append("if barstate.islast")
    for d in sorted(lbl_defs, key=lambda d: (d["col"], -d["price"])):
        yadj = d.get("yadj", 0)
        if yadj != 0:
            y_val = f"{_fmt_p(d['price'])} * 1.025" if yadj > 0 else f"{_fmt_p(d['price'])} * 0.970"
        else:
            y_val = _fmt_p(d["price"])
        tc = d["tc"]
        # hex colors use bare literals; named colors need color. prefix
        if tc.startswith("#"):
            bg_val = f"color.new({tc}, 90)"
            text_val = f"color.new({tc}, 0)"
        else:
            bg_val = f"color.new(color.{tc}, 90)"
            text_val = f"color.new(color.{tc}, 0)"
        lines.append(f"    label.new(bar_index + {d['col']}, {y_val}, \"{d['text']}\", color={bg_val}, textcolor={text_val}, style=label.style_none, size=size.large)")

    return "\n".join(lines) + "\n"


def _serialize_levels(levels: dict) -> dict:
    """Flatten trade levels into a JSON-serialisable dict."""
    out = {
        "direction": levels["direction"],
        "entry": levels["entry"],
        "stop_loss": levels["stop_loss"],
        "take_profit_half": levels["take_profit_half"],
        "take_profit_full": levels["take_profit_full"],
        "height": levels["height"],
    }
    # symmetric has both long and short
    if levels.get("direction") == "SYMMETRIC":
        out["entry_short"] = levels.get("entry_short")
        out["stop_loss_short"] = levels.get("stop_loss_short")
        out["take_profit_half_short"] = levels.get("take_profit_half_short")
        out["take_profit_full_short"] = levels.get("take_profit_full_short")
    return out


def main():
    cfg = StockPatternConfig()

    parser = argparse.ArgumentParser(description="Scan stocks for triangle patterns")
    parser.add_argument("--start", default=cfg.start_date, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=cfg.end_date, help="End date YYYY-MM-DD")
    parser.add_argument("--symbols", nargs="*", default=list(cfg.symbols), help="Tickers to scan")
    parser.add_argument("--bars-left", type=int, default=cfg.bars_left)
    parser.add_argument("--bars-right", type=int, default=cfg.bars_right)
    parser.add_argument("--plots-dir", default=cfg.plots_dir, help="Output directory for charts")
    parser.add_argument("--entry-pct", type=float, default=cfg.entry_pct,
                        help="%% of triangle height above/below trendline for entry")
    parser.add_argument("--min-beta", type=float, default=cfg.min_beta,
                        help="Minimum beta vs SPY to include (default: no filter)")
    parser.add_argument("--interval", default=cfg.interval,
                        help="Candle interval: 1d (daily) or 1h (hourly)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Re-download and re-plot even if the output file already exists")
    parser.add_argument("--validate", action="store_true",
                        help="Send forming patterns to Claude VLM for visual validation")
    args = parser.parse_args()

    # For sub-daily intervals, default to a shorter lookback (60 days)
    # unless the user explicitly set --start. Intervals 1d/1mo/1wk keep full range.
    _sub_daily = {"1h", "3h", "4h", "1m", "5m", "15m", "30m", "60m", "90m"}
    if args.interval in _sub_daily and args.start == cfg.start_date:
        end_dt = _dt.strptime(args.end, "%Y-%m-%d")
        args.start = (end_dt - _td(days=60)).strftime("%Y-%m-%d")
        print(f"{args.interval} mode: auto-adjusted start → {args.start}\n")

    symbols = list(args.symbols)
    betas: dict[str, float] = {}

    if args.min_beta > 0:
        # Crypto + forex pairs trade 24/7 — SPY beta meaningless; auto-include
        non_equity = [s for s in symbols if s.endswith("-USD") or s.endswith("=X")]
        equity = [s for s in symbols if s not in non_equity]
        betas = compute_betas(equity, args.start, args.end)
        symbols = non_equity + [s for s in equity if betas.get(s, 0) >= args.min_beta]
        if not symbols:
            print(f"No symbols with beta >= {args.min_beta}")
            return
        print(f"Filtered to {len(symbols)} symbols with beta >= {args.min_beta}\n")

    plots_dir = Path(args.plots_dir)

    found = 0
    today_str = _dt.now().strftime("%Y-%m-%d")
    skipped = 0
    for sym in symbols:
        # --- cache: skip if plot already exists for today ---
        if not args.no_cache:
            existing = list((plots_dir / today_str).rglob(f"{sym.lower()}_triangle.png"))
            if existing:
                rel = existing[0].relative_to(plots_dir)
                print(f"{sym:6s} — cached ({rel})")
                skipped += 1
                continue

        # Some intervals (3h, 4h) aren't natively supported — resample from 1h
        _RESAMPLE_MAP = {"3h": "1h", "4h": "1h"}
        dl_interval = _RESAMPLE_MAP.get(args.interval, args.interval)
        df = yf.download(
            sym, start=args.start, end=args.end, interval=dl_interval,
            auto_adjust=True, progress=False
        )
        if df.empty:
            print(f"{sym:6s} — no data")
            continue

        if args.interval in _RESAMPLE_MAP:
            # flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.resample(args.interval).agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }).dropna()

        df.attrs["ticker"] = sym
        results = find_triangles(df, bars_left=args.bars_left, bars_right=args.bars_right)

        beta_str = f" β={betas[sym]:.2f}" if sym in betas else ""

        if not results:
            print(f"{sym:6s} — no triangle{beta_str}")
        else:
            # Folder: {sector}/{last_pattern}/{completed_or_forming}/
            sector = SYMBOL_SECTORS.get(sym, "other")
            last = max(results, key=lambda r: r["end"])
            sub = last["alt_name"].lower()
            status = "completed" if last.get("complete", True) else "forming"
            path = plots_dir / today_str / args.interval / status / sector / sub / f"{sym.lower()}_triangle.png"
            plot_triangle(df, results, save_to=str(path), entry_pct=args.entry_pct)

            # --- write metadata JSON next to the PNG ---
            json_path = path.with_suffix(".json")
            meta: dict = {
                "symbol": sym,
                "sector": sector,
                "beta": betas.get(sym),
                "triangles": [],
            }
            for r in results:
                levels = calculate_trade_levels(r, df, entry_pct=args.entry_pct)
                vol = compute_volume_trend(df, r)
                meta["triangles"].append({
                    "pattern": r["alt_name"],
                    "complete": r.get("complete", True),
                    "direction": levels["direction"],
                    "start": str(r["points"]["A"][0].date()),
                    "end": str(r["end"].date()),
                    "duration_days": (r["end"] - r["points"]["A"][0]).days,
                    "trade_levels": _serialize_levels(levels),
                    "volume_trend": vol,
                    "points": {
                        k: [str(v[0].date()), round(float(v[1]), 4)]
                        for k, v in r["points"].items()
                    },
                })
            json_path.write_text(json.dumps(meta, indent=2, default=str))

            # --- write Pine Script next to the PNG/JSON ---
            pine_path = path.with_suffix(".pine")
            # use the last (most recent) triangle for the Pine Script
            last_r = max(results, key=lambda r: r["end"])
            last_levels = calculate_trade_levels(last_r, df, entry_pct=args.entry_pct)
            pine_src = _generate_pine(
                sym=sym,
                result=last_r,
                levels=last_levels,
                extra=last_r.get("extra_points", {}),
                pattern=last_r["alt_name"],
                direction=last_levels["direction"],
                status="completed" if last_r.get("complete", True) else "forming",
                start_date=str(last_r["points"]["A"][0].date()),
                end_date=str(last_r["end"].date()),
            )
            pine_path.write_text(pine_src)
            # ----------------------------------------------------------
            # --- VLM validation (forming triangles only) ---
            if args.validate and status == "forming":
                try:
                    # Generate a plain candlestick chart (no annotations)
                    # using the same window as the annotated chart
                    earliest_a_idx = None
                    for r_ in results:
                        a_ts = r_["points"]["A"][0]
                        if earliest_a_idx is None or a_ts < earliest_a_idx:
                            earliest_a_idx = a_ts
                    a_pos = df.index.get_loc(earliest_a_idx)
                    if isinstance(a_pos, slice):
                        a_pos = a_pos.start
                    start_idx = max(0, a_pos - 30)  # 30 bars before, same as plot_triangle default
                    chart_df = df.iloc[start_idx:]

                    plain_path = Path(str(path).replace("_triangle.png", "_plain.png"))
                    plot_candlestick(
                        chart_df,
                        save_to=str(plain_path),
                        title=f"{sym} — {args.interval}",
                    )

                    from src.stock_pattern.validate import validate_triangle
                    validation = validate_triangle(plain_path, json_path, no_cache=args.no_cache)
                    if validation.get("claude_found_triangle"):
                        agree = "✅ match" if validation.get("pattern_agree") else "⚠️  diff"
                        print(f"       VLM: {agree} algo={validation.get('algo_pattern')} claude={validation.get('claude_pattern')}")
                        if validation.get("confidence_score"):
                            print(f"       VLM: confidence={validation.get('confidence_score')}/100 rec={validation.get('recommendation')}")
                    else:
                        notes = validation.get("validation_notes", "")
                        print(f"       VLM: ❌ no triangle found — {notes[:80]}")
                except Exception as exc:
                    print(f"       VLM: API error — {exc}")
            # ----------------------------------------------------------
            parts = []
            for r in results:
                a_date = r["points"]["A"][0].date()
                e_date = r["points"]["E"][0].date()
                days = (e_date - a_date).days
                parts.append(f"{r['alt_name']} {a_date}→{e_date} ({days}d)")
            print(f"{sym:6s} {', '.join(parts)}  → {path}{beta_str}")
            found += 1

    print(f"\n{found} triangle(s) found, {skipped} cached, across {len(symbols)} symbols (min beta={args.min_beta}).")


if __name__ == "__main__":
    main()
