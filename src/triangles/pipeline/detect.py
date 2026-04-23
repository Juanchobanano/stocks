import logging
from typing import Dict
import pandas as pd
from src.triangles.patterns.core import detect_triangle_geometry
from src.triangles.pipeline.config import TrianglePipelineConfig

log = logging.getLogger(__name__)


def run(
    bars: Dict[str, pd.DataFrame],
    cfg: TrianglePipelineConfig,
) -> Dict[str, dict]:
    """
    Detect triangle geometry for each symbol.

    Returns a dict mapping ticker → geometry dict (only hits, never None).
    """
    results: Dict[str, dict] = {}

    for ticker, df in bars.items():
        try:
            geometry = detect_triangle_geometry(df, lookback=cfg.lookback_bars)
            if geometry is not None:
                results[ticker] = geometry
                log.info(
                    f"{ticker}: {geometry['pattern_type']} detected "
                    f"(height={geometry['height']:.2f}, "
                    f"apex_in={geometry['apex_distance_bars']:.1f}bars, "
                    f"consolidating={geometry['is_consolidating']})"
                )
        except Exception as exc:
            log.warning(f"{ticker}: detection error — {exc}")

    log.info(f"Detection complete: {len(results)} pattern(s) found across {len(bars)} symbols.")
    return results
