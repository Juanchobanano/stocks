"""
Stock pattern detection — geometric chart pattern recognition on OHLC data.

Currently provides triangle detection (symmetrical, ascending, descending)
ported from BennyThadikaran/stock-pattern, plus a minimal candlestick plotter.
"""

from src.stock_pattern.config import StockPatternConfig
from src.stock_pattern.plot import plot_candlestick, plot_triangle
from src.stock_pattern.triangles import (
    Coordinate,
    Line,
    Point,
    calculate_trade_levels,
    compute_volume_trend,
    find_triangles,
    generate_trend_line,
    get_max_min,
)
from src.stock_pattern.validate import validate_triangle

__all__ = [
    "calculate_trade_levels",
    "compute_volume_trend",
    "find_triangles",
    "get_max_min",
    "generate_trend_line",
    "plot_candlestick",
    "plot_triangle",
    "validate_triangle",
    "StockPatternConfig",
    "Coordinate",
    "Line",
    "Point",
]
