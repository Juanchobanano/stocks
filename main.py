import logging
from src.quantls.config import Config
from src.quantls.engine import Backtest
from src.quantls.reporting import print_summary
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    cfg = Config(
        universe_size=100,
        total_positions=40,
        start_date="2022-01-01",
        end_date="2023-12-31",
    )

    bt = Backtest(cfg)
    results = bt.run()

    print_summary(results, cfg)
    results.to_csv("backtest_results.csv")
    log.info("Results saved to backtest_results.csv")


if __name__ == "__main__":
    main()
