import pandas as pd

def load_series(csv_path: str, date_col: str = "date", value_col: str = "value") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    return df[[date_col, value_col]].rename(columns={date_col: "date", value_col: "value"})

def time_split(df: pd.DataFrame, test_days: int, val_days: int):
    if len(df) <= test_days + val_days + 30:
        raise ValueError("Dataset too small for requested split sizes.")
    train_end = len(df) - (test_days + val_days)
    val_end = len(df) - test_days
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test
