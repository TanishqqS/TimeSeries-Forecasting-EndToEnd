import os
import matplotlib.pyplot as plt
import pandas as pd

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def plot_series(df: pd.DataFrame, out_path: str = None, title: str = "Time Series"):
    plt.figure(figsize=(10,4))
    plt.plot(df["date"], df["value"])
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.tight_layout()
    if out_path:
        _ensure_dir(os.path.dirname(out_path))
        plt.savefig(out_path, dpi=200)
    plt.show()

def plot_forecast(dates, y_true, forecasts: dict, out_path: str = None, title: str = "Forecast vs Actual"):
    plt.figure(figsize=(10,4))
    plt.plot(dates, y_true, label="Actual")
    for name, yhat in forecasts.items():
        plt.plot(dates, yhat, label=name)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    if out_path:
        _ensure_dir(os.path.dirname(out_path))
        plt.savefig(out_path, dpi=200)
    plt.show()

def plot_metric_bars(metrics_df: pd.DataFrame, out_dir: str = "figures", title: str = "Model Metrics"):
    _ensure_dir(out_dir)
    for metric in [c for c in metrics_df.columns if c != "model"]:
        plt.figure(figsize=(7,4))
        plt.bar(metrics_df["model"], metrics_df[metric])
        plt.title(f"{title}: {metric}")
        plt.ylabel(metric)
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"03_metrics_{metric.replace('%','pct')}.png"), dpi=200)
        plt.show()
