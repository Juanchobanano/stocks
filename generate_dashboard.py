#!/usr/bin/env python3
"""Generate a self-contained HTML dashboard from all scan + validation results.

Reads every ``_triangle.json`` (algorithm results) and its companion
``_validation.json`` (Claude verdict, if present) under ``plots/`` and writes
a single ``docs/index.html`` ready for GitHub Pages.

Directory structure:
    plots/{YYYY-MM-DD}/{timeframe}/{status}/{sector}/{pattern}/{symbol}_triangle.json  (new)
    plots/{timeframe}/{status}/{sector}/{pattern}/{symbol}_triangle.json               (legacy)

Usage:
    python generate_dashboard.py
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PLOTS_DIR = ROOT / "plots"
OUTPUT = ROOT / "docs" / "index.html"

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TF_ORDER = {"1mo": 0, "1d": 1, "4h": 2, "3h": 3, "1h": 4}


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def collect_rows() -> dict[str, list[dict]]:
    """Walk plots/ and collect forming-triangle rows, grouped by sector.

    Only *forming* (4-point, complete=False) patterns are included.
    Rows are sorted by date (newest first), then timeframe (longest first).
    Sectors are ordered by pattern count (descending).
    """
    raw: list[dict] = []

    for json_path in sorted(PLOTS_DIR.rglob("*_triangle.json")):
        meta = _load_json(json_path)
        if not meta or not meta.get("triangles"):
            continue

        rel = json_path.relative_to(PLOTS_DIR)
        parts = rel.parts

        if DATE_RE.match(parts[0]):
            scan_date = parts[0]
            timeframe = parts[1]
            status = parts[2]
            sector = parts[3]
        else:
            scan_date = ""
            timeframe = parts[0]
            status = parts[1]
            sector = parts[2]

        tri = max(meta["triangles"], key=lambda t: t.get("end", ""))
        if tri.get("complete", True):
            continue

        sym = meta["symbol"]
        algo_pattern = tri.get("pattern", "?")
        algo_direction = tri.get("direction", "?")
        img_rel = f"plots/{rel}".replace("_triangle.json", "_triangle.png")

        val_path = Path(str(json_path).replace("_triangle.json", "_validation.json"))
        val = _load_json(val_path) if val_path.exists() else None

        claude_found = val.get("claude_found_triangle") if val else None
        claude_pattern = val.get("claude_pattern", "") if val else ""
        claude_direction = val.get("claude_direction", "") if val else ""
        agree = val.get("pattern_agree") if val else None
        notes = val.get("validation_notes", "") if val else ""
        research = val.get("symbol_research", "") if val else ""
        sentiment = val.get("market_sentiment", "") if val else ""
        supporting = val.get("supporting_signals", []) if val else []
        opposing = val.get("opposing_signals", []) if val else []
        presence_conf = val.get("presence_confidence") if val else None
        pattern_quality = val.get("pattern_quality") if val else None
        research_conf = val.get("research_confidence") if val else None
        # Backward compat: old format had a single "confidence_score" field
        legacy_conf = val.get("confidence_score") if val else None
        recommendation = val.get("recommendation", "") if val else ""
        validated_at = val.get("validated_at", "") if val else ""

        # Pick the most meaningful score for the table column:
        #   disputed → presence_confidence (how sure Claude is it's NOT a triangle)
        #   agreed   → pattern_quality (how clean/tradable the triangle is)
        if claude_found is True:
            display_score = pattern_quality
            score_label = "Quality"
        elif claude_found is False:
            display_score = presence_conf
            score_label = "Certainty"
        else:
            display_score = legacy_conf or presence_conf
            score_label = "Confidence"

        if claude_found is True and agree:
            row_cls, verdict_label = "ok", "Both agree"
        elif claude_found is True and not agree:
            row_cls, verdict_label = "diff", "Both detect, differ on type"
        elif claude_found is False:
            row_cls, verdict_label = "disp", "Disputed"
        else:
            row_cls, verdict_label = "", "Not reviewed"

        display_date = (validated_at[:10]) if validated_at else scan_date

        raw.append({
            "symbol": sym,
            "sector": sector,
            "timeframe": timeframe,
            "algo_pattern": algo_pattern,
            "algo_direction": algo_direction,
            "img_rel": img_rel,
            "claude_found": claude_found,
            "claude_pattern": claude_pattern,
            "claude_direction": claude_direction,
            "agree": agree,
            "row_cls": row_cls,
            "verdict_label": verdict_label,
            "notes": notes,
            "research": research,
            "sentiment": sentiment,
            "supporting": supporting,
            "opposing": opposing,
            "presence_conf": presence_conf,
            "pattern_quality": pattern_quality,
            "research_conf": research_conf,
            "display_score": display_score,
            "score_label": score_label,
            "recommendation": recommendation,
            "display_date": display_date,
        })

    # Deduplicate: keep newest date per (symbol, timeframe).
    # On tie, prefer the row with a confidence score (validation data).
    seen: dict[tuple[str, str], dict] = {}
    for r in raw:
        key = (r["symbol"], r["timeframe"])
        if key not in seen:
            seen[key] = r
        else:
            curr = seen[key]
            if r["display_date"] > curr["display_date"]:
                seen[key] = r
            elif r["display_date"] == curr["display_date"] and r["display_score"] is not None and curr["display_score"] is None:
                seen[key] = r
    deduped = list(seen.values())

    # Sort: date DESC, then TF (longest first), then confidence DESC
    deduped.sort(key=lambda r: r["display_score"] or 0, reverse=True)
    deduped.sort(key=lambda r: TF_ORDER.get(r["timeframe"], 99))
    deduped.sort(key=lambda r: r["display_date"] or "", reverse=True)

    # Group by sector (sectors ordered by count desc)
    sector_counts: dict[str, int] = {}
    for r in deduped:
        sector_counts[r["sector"]] = sector_counts.get(r["sector"], 0) + 1
    sector_order = sorted(sector_counts, key=lambda s: (-sector_counts[s], s))

    grouped: dict[str, list[dict]] = OrderedDict()
    for sector in sector_order:
        grouped[sector] = [r for r in deduped if r["sector"] == sector]

    return grouped


def build_html(grouped: dict[str, list[dict]]) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    all_rows = [r for rows in grouped.values() for r in rows]
    total = len(all_rows)
    confirmed = sum(1 for r in all_rows if r["claude_found"] is True)
    rejected = sum(1 for r in all_rows if r["claude_found"] is False)
    pending = total - confirmed - rejected

    tfs = sorted({r["timeframe"] for r in all_rows})
    tf_options = "\n".join(f'<option value="{t}">{t}</option>' for t in tfs)

    dates = sorted({r["display_date"] for r in all_rows if r["display_date"]}, reverse=True)
    date_options = "\n".join(f'<option value="{d}">{d}</option>' for d in dates)

    # Number of columns (must match colspan on sector-hdr and detail-row)
    COLSPAN = 10

    rows_html_parts: list[str] = []
    for sector, sector_rows in grouped.items():
        count = len(sector_rows)
        rows_html_parts.append(
            f'<tr class="sector-hdr" data-sector="{sector}">'
            f'<td colspan="{COLSPAN}"><span class="sector-name">{sector}</span> '
            f'<span class="sector-count">{count} pattern{"s" if count != 1 else ""}</span></td>'
            f'</tr>'
        )
        for r in sector_rows:
            cls = r["row_cls"]
            rec = r["recommendation"]
            rec_cls = {"enter": "rec-enter", "monitor": "rec-monitor", "avoid": "rec-avoid"}.get(rec, "")
            conf_str = f'{r["score_label"]}: {r["display_score"]}' if r["display_score"] is not None else "—"
            date_disp = r["display_date"] or "—"

            # Detail section
            detail: list[str] = []
            detail.append(
                f'<div class="detail-label">Chart</div>'
                f'<a href="{r["img_rel"]}" target="_blank">'
                f'<img src="{r["img_rel"]}" class="chart-thumb" loading="lazy" '
                f'alt="{r["symbol"]} chart" title="Click for full size"></a>'
            )
            for label, value in [
                ("Claude Notes", r["notes"]),
                ("Symbol Research", r["research"]),
                ("Market Sentiment", r["sentiment"]),
            ]:
                if value:
                    detail.append(f'<div class="detail-label">{label}</div><div class="detail-text">{value}</div>')
            # Show both scores in detail
            if r["presence_conf"] is not None:
                detail.append(f'<div class="detail-label">Presence certainty</div><div class="detail-text">{r["presence_conf"]}/100 — how sure Claude is that its found/not-found call is correct</div>')
            if r["pattern_quality"] is not None:
                detail.append(f'<div class="detail-label">Pattern quality</div><div class="detail-text">{r["pattern_quality"]}/100 — how clean/tradable this specific triangle is</div>')
            if r["research_conf"] is not None:
                detail.append(f'<div class="detail-label">Research confidence</div><div class="detail-text">{r["research_conf"]}/100 — Phase B research conviction</div>')
            if r["supporting"]:
                items = "".join(f"<li>{s}</li>" for s in r["supporting"])
                detail.append(f'<div class="detail-label">Supporting Signals</div><ul>{items}</ul>')
            if r["opposing"]:
                items = "".join(f"<li>{s}</li>" for s in r["opposing"])
                detail.append(f'<div class="detail-label">Opposing Signals</div><ul>{items}</ul>')

            rows_html_parts.append(
                f'<tr class="row-{cls}"'
                f' data-tf="{r["timeframe"]}"'
                f' data-date="{date_disp}"'
                f' data-sector="{r["sector"]}"'
                f' data-verdict="{cls}"'
                f' onclick="this.classList.toggle(\'expanded\')">'
                f'<td>{date_disp}</td>'
                f'<td class="sym">{r["symbol"]}</td>'
                f'<td>{r["timeframe"]}</td>'
                f'<td>{r["algo_pattern"]}</td>'
                f'<td>{r["algo_direction"]}</td>'
                f'<td class="verdict-col">{r["verdict_label"]}</td>'
                f'<td>{r["claude_pattern"] or "—"}</td>'
                f'<td>{r["claude_direction"] or "—"}</td>'
                f'<td>{conf_str}</td>'
                f'<td class="{rec_cls}">{rec or "—"}</td>'
                f'</tr>'
            )
            rows_html_parts.append(
                f'<tr class="detail-row"'
                f' data-tf="{r["timeframe"]}"'
                f' data-date="{date_disp}"'
                f' data-sector="{r["sector"]}">'
                f'<td colspan="{COLSPAN}"><div class="detail-box">{"".join(detail)}</div></td>'
                f'</tr>'
            )

    rows_html = "\n".join(rows_html_parts)

    return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Triangle Pattern Scanner</title>
<style>
:root {{
    --bg: #131722;
    --surface: #1e222d;
    --border: #2a2e39;
    --text: #d1d4dc;
    --muted: #787b86;
    --green: #22c55e;
    --green-bg: rgba(34, 197, 94, 0.08);
    --blue: #3b82f6;
    --blue-bg: rgba(59, 130, 246, 0.08);
    --amber: #f59e0b;
    --amber-bg: rgba(245, 158, 11, 0.08);
    --yellow: #eab308;
}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{
    background:var(--bg);color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    line-height:1.5;padding:24px;max-width:1500px;margin:0 auto;
}}
h1{{font-size:1.5rem;margin-bottom:4px}}
h1 span{{color:var(--muted);font-weight:400;font-size:.875rem}}
.legend{{display:flex;gap:18px;margin:12px 0 16px;flex-wrap:wrap;font-size:.75rem;color:var(--muted)}}
.legend span{{display:flex;align-items:center;gap:6px}}
.legend .swatch{{width:14px;height:14px;border-radius:3px;flex-shrink:0}}
.swatch-ok{{background:var(--green)}}
.swatch-diff{{background:var(--blue)}}
.swatch-disp{{background:var(--amber)}}
.swatch-none{{background:var(--border)}}
.stats{{display:flex;gap:16px;margin:12px 0 20px;flex-wrap:wrap}}
.stat{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px 20px;min-width:100px}}
.stat .num{{font-size:1.5rem;font-weight:700}}
.stat .label{{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}}
.stat.ok .num{{color:var(--green)}}
.stat.disp .num{{color:var(--amber)}}
.stat.info .num{{color:var(--blue)}}
.stat.warn .num{{color:var(--yellow)}}
.controls{{display:flex;gap:12px;margin-bottom:16px;align-items:center;flex-wrap:wrap}}
.controls select,.controls input{{background:var(--surface);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:6px 12px;font-size:.8rem}}
.controls label{{font-size:.75rem;color:var(--muted)}}
.table-wrap{{overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:.78rem}}
th{{text-align:left;padding:7px 8px;border-bottom:2px solid var(--border);color:var(--muted);font-weight:600;font-size:.68rem;text-transform:uppercase;letter-spacing:.5px;white-space:nowrap}}
td{{padding:7px 8px;border-bottom:1px solid var(--border);white-space:nowrap;vertical-align:middle}}
tr{{cursor:pointer;transition:background .15s}}
tr:hover{{filter:brightness(1.15)}}
.sector-hdr{{cursor:default;background:var(--surface)}}
.sector-hdr:hover{{filter:none}}
.sector-hdr td{{padding:10px 8px;border-bottom:2px solid var(--border);font-weight:600;font-size:.85rem}}
.sector-name{{text-transform:capitalize}}
.sector-count{{color:var(--muted);font-weight:400;font-size:.7rem;margin-left:6px}}
.row-ok{{background:var(--green-bg)}}
.row-diff{{background:var(--blue-bg)}}
.row-disp{{background:var(--amber-bg)}}
.sym{{font-weight:700}}
.verdict-col{{font-size:.72rem;color:var(--muted)}}
.rec-enter{{color:var(--green);font-weight:600}}
.rec-monitor{{color:var(--yellow)}}
.rec-avoid{{color:#ef4444}}
.detail-row{{display:none}}
tr.expanded+.detail-row{{display:table-row}}
tr.expanded+.detail-row.hidden{{display:none}}
.detail-row td{{padding:0;background:var(--surface);border-bottom:2px solid var(--border);white-space:normal}}
.detail-box{{padding:16px 20px;max-width:950px;max-height:600px;overflow-y:auto;display:flex;gap:20px;flex-wrap:wrap}}
.chart-thumb{{max-width:1000px;max-height:680px;border-radius:6px;border:1px solid var(--border);transition:transform .2s}}
.chart-thumb:hover{{transform:scale(1.02)}}
.detail-label{{font-size:.65rem;text-transform:uppercase;letter-spacing:.5px;color:var(--muted);margin-top:12px;margin-bottom:4px}}
.detail-label:first-child{{margin-top:0}}
.detail-text{{font-size:.82rem;line-height:1.6;max-height:280px;overflow-y:auto;word-wrap:break-word;overflow-wrap:break-word}}
.detail-box ul{{margin:4px 0 0 18px;font-size:.82rem;line-height:1.6}}
.detail-box li{{margin-bottom:2px}}
.hidden{{display:none!important}}
.footer{{margin-top:24px;font-size:.7rem;color:var(--muted);text-align:center}}
</style>
</head>
<body>

<h1>&#x25B3; Triangle Pattern Scanner <span>forming patterns (4-point)</span></h1>

<div class="legend">
    <span><span class="swatch swatch-ok"></span> Both agree on a triangle</span>
    <span><span class="swatch swatch-diff"></span> Both detect, differ on type</span>
    <span><span class="swatch swatch-disp"></span> Disputed — Claude doesn't see it</span>
    <span><span class="swatch swatch-none"></span> Not reviewed by Claude</span>
</div>

<div class="stats">
    <div class="stat info"><div class="num">{total}</div><div class="label">Forming Patterns</div></div>
    <div class="stat ok"><div class="num">{confirmed}</div><div class="label">Both Agree</div></div>
    <div class="stat disp"><div class="num">{rejected}</div><div class="label">Disputed</div></div>
    <div class="stat warn"><div class="num">{pending}</div><div class="label">Not Reviewed</div></div>
</div>

<div class="controls">
    <label>Date:</label>
    <select id="filter-date">
        <option value="all">All</option>
        {date_options}
    </select>
    <label>Timeframe:</label>
    <select id="filter-tf">
        <option value="all">All</option>
        {tf_options}
    </select>
    <label>Verdict:</label>
    <select id="filter-verdict">
        <option value="all">All</option>
        <option value="ok">Both agree</option>
        <option value="diff">Both detect, differ</option>
        <option value="disp">Disputed</option>
        <option value="none">Not reviewed</option>
    </select>
    <input type="text" id="filter-search" placeholder="Search symbol...">
</div>

<div class="table-wrap">
<table id="main-table">
    <thead>
        <tr>
            <th>Date</th>
            <th>Symbol</th>
            <th>TF</th>
            <th>Pattern</th>
            <th>Dir</th>
            <th>Verdict</th>
            <th>Claude Pattern</th>
            <th>Claude Dir</th>
            <th>Conf</th>
            <th>Rec</th>
        </tr>
    </thead>
    <tbody>
{rows_html}
    </tbody>
</table>
</div>

<div class="footer">
    Last updated: {now} &mdash; Click any row to see the chart and full analysis &middot; Sorted by date (newest first) then TF
</div>

<script>
(function() {{
    var dateSel   = document.getElementById('filter-date');
    var tfSel     = document.getElementById('filter-tf');
    var verdictSel = document.getElementById('filter-verdict');
    var searchInp = document.getElementById('filter-search');
    var tbody     = document.querySelector('#main-table tbody');

    function applyFilters() {{
        var dateVal    = dateSel.value;
        var tfVal      = tfSel.value;
        var verdictVal = verdictSel.value;
        var search     = searchInp.value.toLowerCase();

        var rows = tbody.querySelectorAll('tr');
        var prevSectorHdr = null;
        var sectorHasVisible = false;

        for (var i = 0; i < rows.length; i++) {{
            var row = rows[i];

            // --- sector headers ---
            if (row.classList.contains('sector-hdr')) {{
                if (prevSectorHdr) {{
                    prevSectorHdr.classList.toggle('hidden', !sectorHasVisible);
                }}
                prevSectorHdr = row;
                sectorHasVisible = false;
                continue;
            }}

            // --- detail rows: mirror their parent ---
            if (row.classList.contains('detail-row')) {{
                var prev = row.previousElementSibling;
                if (prev && !prev.classList.contains('hidden') && !prev.classList.contains('sector-hdr')) {{
                    row.classList.remove('hidden');
                }} else {{
                    row.classList.add('hidden');
                    if (prev && !prev.classList.contains('sector-hdr')) prev.classList.remove('expanded');
                }}
                continue;
            }}

            // --- data rows ---
            var rowDate    = row.getAttribute('data-date') || '';
            var rowTf      = row.getAttribute('data-tf') || '';
            var rowVerdict = row.getAttribute('data-verdict') || '';

            var dateOk    = dateVal === 'all' || rowDate === dateVal;
            var tfOk      = tfVal   === 'all' || rowTf   === tfVal;
            var verdictOk = verdictVal === 'all' || rowVerdict === verdictVal;
            if (verdictVal === 'none') verdictOk = (rowVerdict === '');

            var symEl = row.querySelector('.sym');
            var sym = symEl ? symEl.textContent.toLowerCase() : '';
            var searchOk = !search || sym.indexOf(search) !== -1;

            var show = dateOk && tfOk && verdictOk && searchOk;

            if (show) {{
                row.classList.remove('hidden');
                sectorHasVisible = true;
            }} else {{
                row.classList.add('hidden');
                row.classList.remove('expanded');
                var next = row.nextElementSibling;
                if (next && next.classList.contains('detail-row')) {{
                    next.classList.add('hidden');
                }}
            }}
        }}

        if (prevSectorHdr) {{
            prevSectorHdr.classList.toggle('hidden', !sectorHasVisible);
        }}
    }}

    dateSel.addEventListener('change', applyFilters);
    tfSel.addEventListener('change', applyFilters);
    verdictSel.addEventListener('change', applyFilters);
    searchInp.addEventListener('input', applyFilters);

    // initial pass
    applyFilters();
}})();
</script>

</body>
</html>"""


def main():
    grouped = collect_rows()
    total = sum(len(v) for v in grouped.values())
    print(f"Collected {total} forming pattern(s) across {len(grouped)} sector(s)")

    html = build_html(grouped)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html)
    print(f"Dashboard written -> {OUTPUT}")


if __name__ == "__main__":
    main()
