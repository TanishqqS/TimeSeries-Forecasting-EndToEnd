from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    data_path: str = "data/raw/daily_energy_synthetic.csv"
    date_col: str = "date"
    value_col: str = "value"

    # Time-based split
    test_days: int = 90
    val_days: int = 60

    # SARIMAX defaults
    sarimax_order: tuple = (1, 1, 1)
    sarimax_seasonal_order: tuple = (1, 1, 1, 7)  # weekly

    # LSTM defaults
    lookback: int = 30
    lstm_hidden: int = 64
    lstm_layers: int = 1
    lstm_epochs: int = 15
    lstm_batch_size: int = 64
    lstm_lr: float = 1e-3
