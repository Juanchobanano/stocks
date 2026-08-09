from dataclasses import dataclass, field


# Symbol → market sector (used for plot folder hierarchy)
SYMBOL_SECTORS: dict[str, str] = {
        # Mega-cap / FAANG+
        "AAPL": "mega-cap", "MSFT": "mega-cap", "NVDA": "mega-cap",
        "GOOGL": "mega-cap", "AMZN": "mega-cap", "META": "mega-cap",
        "TSLA": "mega-cap",
        # Semiconductors
        "AVGO": "semiconductors", "AMD": "semiconductors", "INTC": "semiconductors",
        "QCOM": "semiconductors", "TXN": "semiconductors", "AMAT": "semiconductors",
        "LRCX": "semiconductors", "KLAC": "semiconductors", "MU": "semiconductors",
        "ADI": "semiconductors", "MRVL": "semiconductors", "ON": "semiconductors",
        "MCHP": "semiconductors", "MPWR": "semiconductors", "TER": "semiconductors",
        "ENTG": "semiconductors", "STM": "semiconductors", "NXPI": "semiconductors",
        "SWKS": "semiconductors",
        # Software / Cloud
        "ADBE": "software", "CRM": "software", "ORCL": "software",
        "NOW": "software", "SNOW": "software", "CRWD": "software",
        "DDOG": "software", "ZS": "software", "PANW": "software",
        "WDAY": "software", "TEAM": "software", "PLTR": "software",
        "MDB": "software", "NET": "software", "DOCU": "software",
        "ZM": "software", "OKTA": "software", "TWLO": "software",
        "DT": "software", "HUBS": "software", "PATH": "software",
        "GTLB": "software",
        # Internet / E-commerce
        "NFLX": "internet", "SHOP": "internet", "UBER": "internet",
        "ABNB": "internet", "SNAP": "internet", "PINS": "internet",
        "DASH": "internet", "RBLX": "internet", "COIN": "internet",
        # Hardware / Networking
        "CSCO": "hardware", "ANET": "hardware", "DELL": "hardware",
        "HPQ": "hardware", "IBM": "hardware", "SMCI": "hardware",
        "STX": "hardware", "WDC": "hardware", "NTAP": "hardware",
        # Fintech
        "PYPL": "fintech", "SOFI": "fintech", "AFRM": "fintech",
        "TOST": "fintech",
        # Space
        "RKLB": "space", "ASTS": "space", "LUNR": "space",
        "IRDM": "space", "GSAT": "space", "PL": "space",
        "RDW": "space", "BKSY": "space", "SPCE": "space", "VSAT": "space",
        # EV / Auto tech
        "LCID": "ev", "RIVN": "ev", "XPEV": "ev", "NIO": "ev",
        # Clean energy tech
        "ENPH": "clean-energy", "FSLR": "clean-energy",
        "RUN": "clean-energy", "SEDG": "clean-energy",
        # Crypto / Blockchain equities
        "MSTR": "crypto-equity", "MARA": "crypto-equity", "RIOT": "crypto-equity",
        "HUT": "crypto-equity", "CLSK": "crypto-equity", "WULF": "crypto-equity",
        "IREN": "crypto-equity", "CIFR": "crypto-equity", "CORZ": "crypto-equity",
        "BTBT": "crypto-equity", "HOOD": "crypto-equity",
        # Crypto majors
        "BTC-USD": "crypto", "ETH-USD": "crypto", "XRP-USD": "crypto",
        "SOL-USD": "crypto", "DOGE-USD": "crypto", "ADA-USD": "crypto",
        "AVAX-USD": "crypto", "DOT-USD": "crypto", "LINK-USD": "crypto",
        "LTC-USD": "crypto", "BCH-USD": "crypto", "XLM-USD": "crypto",
        "ATOM-USD": "crypto", "NEAR-USD": "crypto", "ICP-USD": "crypto",
        "FIL-USD": "crypto", "ARB-USD": "crypto", "OP-USD": "crypto",
        "AAVE-USD": "crypto", "ALGO-USD": "crypto", "VET-USD": "crypto",
        "SAND-USD": "crypto", "MANA-USD": "crypto", "SHIB-USD": "crypto",
        "ETC-USD": "crypto",
        # Banks / Financials
        "JPM": "banks", "BAC": "banks", "GS": "banks", "MS": "banks",
        "BX": "banks", "C": "banks",
        # Energy
        "OXY": "energy", "DVN": "energy", "HAL": "energy", "SLB": "energy",
        "XOM": "energy", "CVX": "energy",
        # Industrials / Airlines / Cruise
        "CAT": "industrials", "GE": "industrials", "BA": "industrials",
        "CCL": "industrials", "RCL": "industrials", "DAL": "industrials",
        # Gambling / Entertainment
        "DKNG": "entertainment", "SPOT": "entertainment", "DIS": "entertainment",
        # Adtech / Enterprise
        "TTD": "adtech", "U": "adtech", "SNPS": "adtech", "CDNS": "adtech",
        # Biotech
        "MRNA": "biotech", "BNTX": "biotech",
        # AI / Emerging tech
        "ARM": "ai", "AI": "ai", "SOUN": "ai",
        # Retail / Consumer
        "WMT": "retail", "TGT": "retail", "COST": "retail",
        "HD": "retail", "LOW": "retail", "MCD": "retail",
        "SBUX": "retail", "NKE": "retail", "LULU": "retail",
        # Defense / Aerospace
        "RTX": "defense", "LMT": "defense", "NOC": "defense",
        "GD": "defense", "LHX": "defense", "TDG": "defense",
        "HII": "defense",
        # Healthcare / Pharma
        "PFE": "healthcare", "JNJ": "healthcare", "UNH": "healthcare",
        "ABBV": "healthcare", "LLY": "healthcare", "GILD": "healthcare",
        "REGN": "healthcare", "BIIB": "healthcare",
        # Commodities / Metals / Mining
        "GLD": "commodities", "SLV": "commodities", "GDX": "commodities",
        "FCX": "commodities", "NEM": "commodities", "AA": "commodities",
        "CLF": "commodities",
        # Forex majors
        "EURUSD=X": "forex", "GBPUSD=X": "forex", "USDJPY=X": "forex",
        "USDCHF=X": "forex", "AUDUSD=X": "forex", "USDCAD=X": "forex",
        "NZDUSD=X": "forex", "EURGBP=X": "forex", "EURJPY=X": "forex",
        "GBPJPY=X": "forex",
    }


@dataclass
class StockPatternConfig:
    """Configuration for the stock-pattern scanner."""

    # --- data window ---
    start_date: str = "2026-01-01"
    end_date: str = "2026-08-07"

    # --- pivot detection ---
    bars_left: int = 6
    bars_right: int = 6

    # --- symbols to scan ---
    symbols: tuple[str, ...] = (
        # Mega-cap / FAANG+
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
        # Semiconductors
        "AVGO", "AMD", "INTC", "QCOM", "TXN", "AMAT", "LRCX",
        "KLAC", "MU", "ADI", "MRVL", "ON", "MCHP", "MPWR", "TER",
        "ENTG", "STM", "NXPI", "SWKS",
        # Software / Cloud
        "ADBE", "CRM", "ORCL", "NOW", "SNOW", "CRWD", "DDOG",
        "ZS", "PANW", "WDAY", "TEAM", "PLTR", "MDB", "NET",
        "DOCU", "ZM", "OKTA", "TWLO", "DT", "HUBS", "PATH",
        "GTLB",
        # Internet / E-commerce
        "NFLX", "SHOP", "UBER", "ABNB", "SNAP", "PINS",
        "DASH", "RBLX", "COIN",
        # Hardware / Networking
        "CSCO", "ANET", "DELL", "HPQ", "IBM", "SMCI",
        "STX", "WDC", "NTAP",
        # Fintech
        "PYPL", "SOFI", "AFRM", "TOST",
        # Space
        "RKLB", "ASTS", "LUNR", "IRDM", "GSAT", "PL",
        "RDW", "BKSY", "SPCE", "VSAT",
        # EV / Auto tech
        "LCID", "RIVN", "XPEV", "NIO",
        # Clean energy tech
        "ENPH", "FSLR", "RUN", "SEDG",
        # Crypto / Blockchain equities
        "MSTR", "MARA", "RIOT", "HUT", "CLSK",
        "WULF", "IREN", "CIFR", "CORZ", "BTBT", "HOOD",
        # Crypto majors (yfinance pairs)
        "BTC-USD", "ETH-USD", "XRP-USD", "SOL-USD", "DOGE-USD",
        "ADA-USD", "AVAX-USD", "DOT-USD", "LINK-USD",
        "LTC-USD", "BCH-USD", "XLM-USD", "ATOM-USD", "NEAR-USD",
        "ICP-USD", "FIL-USD", "ARB-USD", "OP-USD",
        "AAVE-USD", "ALGO-USD", "VET-USD",
        "SAND-USD", "MANA-USD", "SHIB-USD", "ETC-USD",
        # Banks / Financials
        "JPM", "BAC", "GS", "MS", "BX", "C",
        # Energy
        "OXY", "DVN", "HAL", "SLB", "XOM", "CVX",
        # Industrials / Airlines / Cruise
        "CAT", "GE", "BA", "CCL", "RCL", "DAL",
        # Gambling / Entertainment
        "DKNG", "SPOT", "DIS",
        # Adtech / Enterprise
        "TTD", "U", "SNPS", "CDNS",
        # Biotech (volatile)
        "MRNA", "BNTX",
        # AI / Emerging tech
        "ARM", "AI", "SOUN",
        # Retail / Consumer
        "WMT", "TGT", "COST", "HD", "LOW", "MCD",
        "SBUX", "NKE", "LULU",
        # Defense / Aerospace
        "RTX", "LMT", "NOC", "GD", "LHX", "TDG", "HII",
        # Healthcare / Pharma
        "PFE", "JNJ", "UNH", "ABBV", "LLY", "GILD", "REGN", "BIIB",
        # Commodities / Metals / Mining
        "GLD", "SLV", "GDX", "FCX", "NEM", "AA", "CLF",
        # Forex majors
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X",
        "USDCAD=X", "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X",
    )

    # --- filters ---
    min_beta: float = 1.2

    # --- data interval ---
    interval: str = "1h"  # "1d" for daily, "1h" for hourly

    # --- trade levels ---
    entry_pct: float = 0.02

    # --- output ---
    plots_dir: str = "plots"
