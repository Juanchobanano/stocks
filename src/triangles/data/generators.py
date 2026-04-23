import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def generate_sample_df_with_pattern(pattern: str) -> pd.DataFrame:
    """Return a small synthetic OHLCV DataFrame containing the requested pattern."""
    date_rng = pd.date_range(start='2020-01-01', end='2020-01-10', freq='D')
    data: dict = {'date': date_rng}

    if pattern == 'Head and Shoulder':
        data['Open'] = [90, 85, 80, 90, 85, 80, 75, 80, 85, 90]
        data['High'] = [95, 90, 85, 95, 90, 85, 80, 85, 90, 95]
        data['Low'] = [80, 75, 70, 80, 75, 70, 65, 70, 75, 80]
        data['Close'] = [90, 85, 80, 90, 85, 80, 75, 80, 85, 90]
        data['Volume'] = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    elif pattern == 'Inverse Head and Shoulder':
        data['Open'] = [20, 25, 30, 20, 25, 30, 35, 30, 25, 20]
        data['High'] = [25, 30, 35, 25, 30, 35, 40, 35, 30, 25]
        data['Low'] = [15, 20, 25, 15, 20, 25, 30, 25, 20, 15]
        data['Close'] = [20, 25, 30, 20, 25, 30, 35, 30, 25, 20]
        data['Volume'] = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    else:
        data['High'] = [95, 90, 85, 95, 90, 85, 80, 85, 90, 95]
        data['Low'] = [80, 75, 70, 80, 75, 70, 65, 70, 75, 80]

    return pd.DataFrame(data)


def generate_random_data(length: int) -> dict:
    """Generate a dict of random OHLCV lists with realistic price ranges."""
    close_values = np.random.randint(100, 200, length).tolist()
    return {
        'Open': [v - random.randint(0, 10) for v in close_values],
        'High': [v + random.randint(0, 10) for v in close_values],
        'Low': [v - random.randint(0, 10) for v in close_values],
        'Close': close_values,
        'Volume': np.random.randint(1000, 2000, length).tolist(),
    }


def inject_patterns(data: dict) -> dict:
    """Inject Head and Shoulder (indices 3,5,7) and Inverse Head and Shoulder (13,15,17) into data.

    Assumes a 20-point series. Shoulder/head heights are randomized within realistic ranges.
    """
    shoulder_height = random.randint(120, 140)
    head_height = random.randint(150, 170)
    inv_shoulder_depth = random.randint(60, 80)
    inv_head_depth = random.randint(40, 60)

    data['High'][3] = shoulder_height
    data['Close'][3] = shoulder_height - random.randint(0, 5)

    data['High'][5] = head_height
    data['Close'][5] = head_height - random.randint(0, 5)

    data['High'][7] = shoulder_height
    data['Close'][7] = shoulder_height - random.randint(0, 5)

    data['Low'][13] = inv_shoulder_depth
    data['Close'][13] = inv_shoulder_depth + random.randint(0, 5)

    data['Low'][15] = inv_head_depth
    data['Close'][15] = inv_head_depth + random.randint(0, 5)

    data['Low'][17] = inv_shoulder_depth
    data['Close'][17] = inv_shoulder_depth + random.randint(0, 5)

    return data


def generate_data_head_shoulder(n: int) -> pd.DataFrame:
    """Generate n synthetic 20-day windows, each containing HS and I-HS patterns, concatenated."""
    start_date = datetime.now()
    frames = []

    for _ in range(n):
        data = generate_random_data(20)
        data = inject_patterns(data)
        temp_df = pd.DataFrame(data)
        temp_df['Datetime'] = pd.date_range(start=start_date, periods=20, freq='D')
        start_date += timedelta(days=20)
        frames.append(temp_df)

    df = pd.concat(frames, ignore_index=True)
    return df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
