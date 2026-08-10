"""
VLM validation for triangle patterns using Claude Vision API.

Two-phase flow:
  Phase A — send a *plain* candlestick chart (no trendlines) to Claude and
            ask it to independently identify any triangle patterns.
  Phase B — deep research (text-only, only if Claude confirms the pattern).

Public API:
    validate_triangle(plain_png_path, json_path, *, model) -> dict
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Auto-load .env from project root
# ---------------------------------------------------------------------------
def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
_API_BASE = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"


def _get_api_key() -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")
    return api_key


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------
def _parse_json(text: str) -> dict:
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        return json.loads(m.group(1))
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"Could not extract JSON from response: {text[:200]}")


# ---------------------------------------------------------------------------
# Low-level API call with retry
# ---------------------------------------------------------------------------
def _call_claude(
    api_key: str,
    *,
    model: str,
    system: str,
    messages: list[dict],
    max_tokens: int = 1024,
    max_retries: int = 3,
) -> str:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": _API_VERSION,
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(_API_BASE, headers=headers, json=body, timeout=60)
            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                if attempt < max_retries:
                    delay = 2 ** (attempt + 1)
                    log.warning("API %s — retrying in %ds", resp.status_code, delay)
                    time.sleep(delay)
                    continue
            resp.raise_for_status()
            data = resp.json()
            for block in data.get("content", []):
                if block.get("type") == "text":
                    return block["text"]
            return ""
        except (requests.ConnectionError, requests.Timeout) as e:
            last_error = e
            if attempt < max_retries:
                delay = 2 ** (attempt + 1)
                log.warning("Connection error — retrying in %ds", delay)
                time.sleep(delay)
                continue
        except requests.HTTPError:
            raise

    raise RuntimeError(f"API call failed after {max_retries} retries") from last_error


# ---------------------------------------------------------------------------
# Phase A — independent pattern detection (plain chart, no annotations)
# ---------------------------------------------------------------------------
_PHASE_A_SYSTEM = """\
You are an expert technical analyst. You will be shown a plain candlestick \
chart with NO trendlines or annotations. Your job is to visually scan the \
chart and identify whether any triangle patterns are forming.

Triangle types to look for:
- **Ascending**: a roughly flat or gently sloping upper resistance line, \
  with a rising lower support line (higher lows)
- **Descending**: a roughly flat or gently sloping lower support line, \
  with a falling upper resistance line (lower highs)
- **Symmetric**: both upper and lower trendlines converging toward each other

Real-world patterns are rarely textbook-perfect. Be flexible:
- Trendlines may have a slight slope instead of being perfectly flat
- Touch points don't need to hit the line exactly — nearby is fine (within ~1-2 candles)
- Two touches per side is enough; three is ideal but not required
- If the overall structure clearly shows price compressing into a converging wedge, \
  that counts — mark it as a triangle

Only reject patterns that are clearly NOT triangles: parallel channels, \
random noise, already broken out with no containment, or fewer than two \
touches on each side.

Look at the RIGHT SIDE of the chart — is price compressing into a triangle \
apex? Is the pattern still forming (price inside), or has it already broken out?

IMPORTANT — separate two distinct judgments:

1. **Presence certainty** (0-100): how sure are you that your found/not-found \
   verdict is correct? A high score means "I'm very confident in my answer, \
   whichever way it goes." You can be 90% certain a triangle is NOT there \
   (clear downtrend, clean channel, obvious breakout). You can be 90% certain \
   one IS there (textbook touches, clear convergence). A low score means \
   "it's ambiguous — I could be wrong either way."

2. **Pattern quality** (0-100): ONLY if you found a triangle. How clean and \
   tradable is this specific pattern? Consider: number of touch points (3+ per \
   side = high), convergence clarity, how well price respects the boundaries, \
   how much room is left before the apex. A textbook pattern gets 85+; a messy \
   but recognizable one gets 50-60. If you did NOT find a triangle, set this \
   to null — do NOT report quality for a pattern you rejected.

Reply with a single JSON object."""


def _build_phase_a_prompt(meta: dict) -> str:
    """Build the user prompt for phase A — independent detection."""
    triangles = meta.get("triangles", [])
    forming = [t for t in triangles if not t.get("complete", True)]
    target = forming[-1] if forming else triangles[-1]

    our_pattern = target.get("pattern", "?")
    our_direction = target.get("direction", "?")

    return f"""Symbol: {meta['symbol']}  |  Interval: {meta.get('interval', '?')}
Sector: {meta.get('sector', '?')}

Look at this plain candlestick chart carefully. Do you see any triangle pattern \
(ascending, descending, or symmetric) forming near the right edge?

For reference, an automated algorithm flagged this as a possible **{our_pattern}** \
triangle ({our_direction} direction). But algorithms have no judgment — you do. \
Use your visual expertise to independently assess whether the structure is credible.

Be flexible: real-world triangles are rarely textbook-perfect. If the overall \
structure resembles a converging pattern with price compressing between two \
trendlines, that's a valid find.

Reply with JSON only:
{{
  "found_triangle": true/false,
  "pattern_type": "Ascending" | "Descending" | "Symmetric" | null,
  "direction": "LONG" | "SHORT" | "SYMMETRIC" | null,
  "presence_confidence": 0-100 — how certain you are that your found_triangle verdict is correct, REGARDLESS of which way you ruled. High = very confident in your yes OR no. Low = ambiguous, could go either way.,
  "pattern_quality": 0-100 if found_triangle is true, otherwise null — if you found a triangle, how clean/tradable is it? Touch points (3+ per side = high), convergence clarity, respect of boundaries, room before apex.
  "notes": "Describe what you see — where are the trendlines? How many touch points? Is price still inside or has it broken out? If you disagree with the algorithm, explain why."
}}"""


def validate_pattern(
    plain_png_path: Path,
    json_path: Path,
    *,
    model: str = "claude-sonnet-5",
) -> dict:
    """Phase A: send a plain candlestick chart to Claude for independent detection.

    Returns dict with found_triangle, pattern_type, direction, notes.
    """
    api_key = _get_api_key()

    with open(plain_png_path, "rb") as fh:
        img_bytes = fh.read()
    b64 = base64.b64encode(img_bytes).decode("ascii")

    meta = json.loads(json_path.read_text())
    symbol = meta["symbol"]

    prompt = _build_phase_a_prompt(meta)

    log.info("Phase A — independent detection for %s", symbol)

    raw = _call_claude(
        api_key,
        model=model,
        system=_PHASE_A_SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64,
                    },
                },
                {"type": "text", "text": prompt},
            ],
        }],
        max_tokens=1024,
    )

    try:
        result = _parse_json(raw)
    except (json.JSONDecodeError, ValueError):
        log.warning("Phase A — could not parse JSON, saving raw text")
        result = {"found_triangle": None, "pattern_type": None, "direction": None, "notes": raw}

    result.setdefault("found_triangle", None)
    result.setdefault("pattern_type", None)
    result.setdefault("direction", None)
    result.setdefault("presence_confidence", None)
    result.setdefault("pattern_quality", None)
    result.setdefault("notes", "")
    result["symbol"] = symbol

    # Carry forward our algorithm's detection for comparison
    triangles = meta.get("triangles", [])
    forming = [t for t in triangles if not t.get("complete", True)]
    target = forming[-1] if forming else triangles[-1]
    result["algo_pattern"] = target.get("pattern", "?")
    result["algo_direction"] = target.get("direction", "?")

    return result


# ---------------------------------------------------------------------------
# Phase B — deep research (text-only, no image)
# ---------------------------------------------------------------------------
_PHASE_B_SYSTEM = """\
You are a quantitative research analyst reviewing a validated triangle pattern \
for a trading decision. Be concise and evidence-based. Reply with a single JSON object."""


def _build_phase_b_prompt(meta: dict, claude_pattern: str, claude_direction: str) -> str:
    triangles = meta.get("triangles", [])
    forming = [t for t in triangles if not t.get("complete", True)]
    target = forming[-1] if forming else triangles[-1]

    trade_levels = target.get("trade_levels", {})

    return f"""Symbol: {meta['symbol']}
Sector: {meta.get('sector', '?')}
Pattern: {claude_pattern} triangle  |  Direction: {claude_direction}
Entry: {trade_levels.get('entry', '?')}  |  Stop: {trade_levels.get('stop_loss', '?')}
TP targets: {trade_levels.get('take_profit_half', '?')} / {trade_levels.get('take_profit_full', '?')}

The trader is considering a {claude_direction} trade on this {claude_pattern} triangle \
that was independently confirmed by visual inspection.

Please:
1. Briefly research what this symbol/company does and any recent relevant context
2. Assess the current market sentiment (bullish / neutral / bearish)
3. List 2-5 signals that SUPPORT the {claude_direction} trade
4. List 2-5 signals that OPPOSE or warn against the {claude_direction} trade
5. Provide a confidence score (0-100) and recommendation (monitor / enter / avoid)

Reply with JSON only:
{{
  "symbol_research": "...",
  "market_sentiment": "bullish" | "neutral" | "bearish",
  "supporting_signals": ["...", "..."],
  "opposing_signals": ["...", "..."],
  "confidence_score": 65,
  "recommendation": "monitor" | "enter" | "avoid"
}}"""


def research_symbol(
    json_path: Path,
    *,
    claude_pattern: str = "",
    claude_direction: str = "",
    model: str = "claude-sonnet-5",
) -> dict:
    """Phase B: deep research on the symbol (text-only)."""
    api_key = _get_api_key()
    meta = json.loads(json_path.read_text())
    symbol = meta["symbol"]

    prompt = _build_phase_b_prompt(meta, claude_pattern, claude_direction)

    log.info("Phase B — researching %s", symbol)

    raw = _call_claude(
        api_key,
        model=model,
        system=_PHASE_B_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )

    try:
        result = _parse_json(raw)
    except (json.JSONDecodeError, ValueError):
        log.warning("Phase B — could not parse JSON, saving raw text")
        result = {
            "symbol_research": raw,
            "market_sentiment": "unknown",
            "supporting_signals": [],
            "opposing_signals": [],
            "confidence_score": None,
            "recommendation": "unknown",
        }

    return result


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------
def validate_triangle(
    plain_png_path: Path,
    json_path: Path,
    *,
    model: str = "claude-sonnet-5",
    no_cache: bool = False,
) -> dict:
    """Validate a forming triangle by asking Claude to independently detect patterns.

    Parameters
    ----------
    plain_png_path : Path
        Path to a PLAIN candlestick chart PNG (no trendlines or annotations).
    json_path : Path
        Path to the companion JSON metadata file.
    model : str
        Claude model to use.
    no_cache : bool
        If True, re-validate even if a cached validation exists.

    Returns
    -------
    dict
        Always saved to ``{symbol}_validation.json``.
    """
    # --- cache check ---
    val_path = Path(str(json_path).replace("_triangle.json", "_validation.json"))
    if not no_cache and val_path.exists():
        png_mtime = plain_png_path.stat().st_mtime
        val_mtime = val_path.stat().st_mtime
        if val_mtime >= png_mtime:
            log.info("Validation cache hit for %s", val_path.name)
            return json.loads(val_path.read_text())

    # --- Phase A: independent detection from plain chart ---
    phase_a = validate_pattern(plain_png_path, json_path, model=model)

    symbol = phase_a.get("symbol", "?")
    found = phase_a.get("found_triangle", False)
    claude_pattern = phase_a.get("pattern_type", "?") or "?"
    claude_direction = phase_a.get("direction", "?") or "?"
    algo_pattern = phase_a.get("algo_pattern", "?")
    algo_direction = phase_a.get("algo_direction", "?")

    # Phase A: two distinct scores
    #   presence_confidence — how sure Claude is about its found/not-found verdict
    #   pattern_quality    — how clean/tradable the pattern is (only if found)
    presence_conf = phase_a.get("presence_confidence")
    if presence_conf is not None:
        try:
            presence_conf = int(presence_conf)
        except (TypeError, ValueError):
            pass

    pattern_quality = phase_a.get("pattern_quality")
    if pattern_quality is not None:
        try:
            pattern_quality = int(pattern_quality)
        except (TypeError, ValueError):
            pass

    result = {
        "symbol": symbol,
        "algo_pattern": algo_pattern,
        "algo_direction": algo_direction,
        "claude_found_triangle": found,
        "claude_pattern": claude_pattern,
        "claude_direction": claude_direction,
        "pattern_agree": (found and claude_pattern == algo_pattern),
        "presence_confidence": presence_conf,
        "pattern_quality": pattern_quality,
        "validation_notes": phase_a.get("notes", ""),
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }

    if not found:
        val_path.write_text(json.dumps(result, indent=2, default=str))
        return result

    # --- Phase B: research (only if Claude confirmed a triangle) ---
    phase_b = research_symbol(
        json_path,
        claude_pattern=claude_pattern,
        claude_direction=claude_direction,
        model=model,
    )

    result.update({
        "symbol_research": phase_b.get("symbol_research", ""),
        "market_sentiment": phase_b.get("market_sentiment", "unknown"),
        "supporting_signals": phase_b.get("supporting_signals", []),
        "opposing_signals": phase_b.get("opposing_signals", []),
        "recommendation": phase_b.get("recommendation", "unknown"),
    })
    # Phase B research confidence (separate from Phase A visual scores)
    pb_conf = phase_b.get("confidence_score")
    if pb_conf is not None:
        result["research_confidence"] = pb_conf

    val_path.write_text(json.dumps(result, indent=2, default=str))
    return result
