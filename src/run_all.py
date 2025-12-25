"""Run the full pipeline and write outputs to results/ and figures/.

Run:
    python -m src.run_all
"""
import os
import pandas as pd
import torch

from .config import Config
from .data_loader import load_series, time_split
from .baselines import naive_last_value_forecast, seasonal_naive_forecast
from .sarimax_model import fit_sarimax, forecast_sarimax
from .lstm_model import train_lstm, forecast_lstm
from .evaluation import compute_metrics
from .visualization import plot_series, plot_forecast, plot_metric_bars

def main():
    cfg = Config()
    df = load_series(cfg.data_path, cfg.date_col, cfg.value_col)
    train, val, test = time_split(df, test_days=cfg.test_days, val_days=cfg.val_days)

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    plot_series(df, out_path="figures/01_series.png", title="Daily Energy (Synthetic)")

    horizon = len(test)
    test_dates = test["date"].to_numpy()
    y_true = test["value"].to_numpy(float)

    hist_df = pd.concat([train, val])

    yhat_naive = naive_last_value_forecast(hist_df, horizon=horizon)
    yhat_seasonal = seasonal_naive_forecast(hist_df, horizon=horizon, season=7)

    sarimax_res = fit_sarimax(hist_df, order=cfg.sarimax_order, seasonal_order=cfg.sarimax_seasonal_order)
    yhat_sarimax = forecast_sarimax(sarimax_res, horizon=horizon)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, norm_stats = train_lstm(
        train_values=train["value"].to_numpy(float),
        val_values=val["value"].to_numpy(float),
        lookback=cfg.lookback,
        hidden_size=cfg.lstm_hidden,
        num_layers=cfg.lstm_layers,
        epochs=cfg.lstm_epochs,
        batch_size=cfg.lstm_batch_size,
        lr=cfg.lstm_lr,
        device=device
    )
    yhat_lstm = forecast_lstm(model, history_values=hist_df["value"].to_numpy(float),
                             horizon=horizon, lookback=cfg.lookback, norm_stats=norm_stats, device=device)

    rows = []
    for name, yhat in [
        ("Naive", yhat_naive),
        ("SeasonalNaive(7)", yhat_seasonal),
        ("SARIMAX", yhat_sarimax),
        ("LSTM", yhat_lstm),
    ]:
        m = compute_metrics(y_true, yhat)
        m["model"] = name
        rows.append(m)

    metrics_df = pd.DataFrame(rows)[["model", "MAE", "RMSE", "MAPE(%)"]]
    metrics_df.to_csv("results/metrics.csv", index=False)

    plot_forecast(
        test_dates, y_true,
        forecasts={"Naive": yhat_naive, "SeasonalNaive(7)": yhat_seasonal, "SARIMAX": yhat_sarimax, "LSTM": yhat_lstm},
        out_path="figures/02_forecasts.png",
        title="Forecast vs Actual (Test Window)"
    )
    plot_metric_bars(metrics_df, out_dir="figures", title="Model Metrics")
    print("Done. See results/metrics.csv and figures/*.png")

if __name__ == "__main__":
    main()
