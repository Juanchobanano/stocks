import logging
import sys
from src.quantls.config import Config
from src.quantls.engine import LiveEngine
from src.broker import IBBroker
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


cfg = Config(
    polygon_api_key="YOUR_POLYGON_KEY",
    universe_size=100,
    total_positions=40,
    max_gross_leverage=1.0,
    lookback_days=126,
    lasso_alpha=0.01,
    ml_weight=1.5,
)

IB_PORT = 7497          # paper: 7497 / live: 7496 / GW paper: 4002 / GW live: 4001
ORDER_TYPE = "MKT"      # "MKT" or "LMT"
STAGGER_SECONDS = 0.5   # seconds between order submissions
DRY_RUN = True          # set False to submit real orders
FRACTIONAL = True       # send decimal share quantities


def retrain() -> None:
    """
    Fit the Lasso model on the most recent lookback window and save it to disk.
    Run this Friday EOD, after market close.
    """
    log.info("=== RETRAIN ===")
    engine = LiveEngine(cfg)
    engine.retrain()
    log.info("Model saved. Run 'python production.py trade' at next market open.")


def trade() -> None:
    """
    Load the saved model, generate target weights, and rebalance via IB.
    Run this Friday morning, after market open.
    """
    log.info("=== TRADE ===")
    engine = LiveEngine(cfg)
    weights = engine.generate_signals()

    log.info(f"Target weights ({len(weights)} positions):\n{weights.sort_values().to_string()}")
    log.info(
        f"Longs: {(weights > 0).sum()}  "
        f"Shorts: {(weights < 0).sum()}  "
        f"Gross leverage: {weights.abs().sum():.4f}"
    )

    with IBBroker(
        port=IB_PORT,
        order_type=ORDER_TYPE,
        stagger_seconds=STAGGER_SECONDS,
        dry_run=DRY_RUN,
        fractional=FRACTIONAL,
    ) as broker:
        broker.rebalance(weights)


COMMANDS = {"retrain": retrain, "trade": trade}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        print(f"Usage: python production.py [{' | '.join(COMMANDS)}]")
        sys.exit(1)

    COMMANDS[sys.argv[1]]()
