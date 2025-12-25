import numpy as np
import pandas as pd

def naive_last_value_forecast(train: pd.DataFrame, horizon: int) -> np.ndarray:
    last = float(train["value"].iloc[-1])
    return np.full((horizon,), last, dtype=float)

def seasonal_naive_forecast(train: pd.DataFrame, horizon: int, season: int = 7) -> np.ndarray:
    values = train["value"].to_numpy()
    if len(values) < season:
        return naive_last_value_forecast(train, horizon)
    last_season = values[-season:]
    reps = int(np.ceil(horizon / season))
    return np.tile(last_season, reps)[:horizon].astype(float)
