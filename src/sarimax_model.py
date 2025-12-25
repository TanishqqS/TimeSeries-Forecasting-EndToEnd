import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_sarimax(train: pd.DataFrame, order=(1,1,1), seasonal_order=(1,1,1,7)):
    model = SARIMAX(
        train["value"].astype(float),
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return model.fit(disp=False)

def forecast_sarimax(model_res, horizon: int) -> np.ndarray:
    pred = model_res.forecast(steps=horizon)
    return np.asarray(pred, dtype=float)
