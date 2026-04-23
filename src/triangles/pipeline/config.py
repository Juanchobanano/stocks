from dataclasses import dataclass


@dataclass
class TrianglePipelineConfig:
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497
    ib_client_id: int = 2
    scanner_n_symbols: int = 100
    scanner_code: str = "MOST_ACTIVE"
    lookback_bars: int = 90
    bar_size: str = "1 day"
    min_bars_to_apex: int = 5
    volume_lookback: int = 20
    volume_ratio_threshold: float = 0.9
    position_pct: float = 0.02
    dry_run: bool = True
    db_path: str = "triangles_pipeline.db"
