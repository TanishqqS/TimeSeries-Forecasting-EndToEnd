import numpy as np

def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

def compute_metrics(y_true, y_pred) -> dict:
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred), "MAPE(%)": mape(y_true, y_pred)}
