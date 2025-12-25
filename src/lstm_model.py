import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class SeqDataset(Dataset):
    def __init__(self, series: np.ndarray, lookback: int):
        self.series = series.astype(np.float32)
        self.lookback = lookback

    def __len__(self):
        return len(self.series) - self.lookback

    def __getitem__(self, idx):
        x = self.series[idx:idx+self.lookback]
        y = self.series[idx+self.lookback]
        return torch.from_numpy(x).unsqueeze(-1), torch.tensor([y], dtype=torch.float32)

class LSTMForecaster(nn.Module):
    def __init__(self, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

def train_lstm(train_values: np.ndarray, val_values: np.ndarray, lookback: int,
               hidden_size: int = 64, num_layers: int = 1,
               epochs: int = 10, batch_size: int = 64, lr: float = 1e-3,
               device: str = "cpu"):
    mu, sigma = float(train_values.mean()), float(train_values.std() + 1e-8)
    train_norm = (train_values - mu) / sigma
    val_norm = (val_values - mu) / sigma

    train_ds = SeqDataset(train_norm, lookback)
    val_ds = SeqDataset(val_norm, lookback)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = LSTMForecaster(hidden_size=hidden_size, num_layers=num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for _ in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())
        avg_val = float(np.mean(val_losses)) if val_losses else float("inf")
        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, (mu, sigma)

def forecast_lstm(model: nn.Module, history_values: np.ndarray, horizon: int, lookback: int, norm_stats, device: str = "cpu") -> np.ndarray:
    mu, sigma = norm_stats
    hist = history_values.astype(np.float32)
    hist_norm = (hist - mu) / sigma

    window = hist_norm[-lookback:].copy()
    preds = []

    model.eval()
    with torch.no_grad():
        for _ in range(horizon):
            xb = torch.from_numpy(window).unsqueeze(0).unsqueeze(-1).to(device)
            yhat = model(xb).item()
            preds.append(yhat)
            window = np.roll(window, -1)
            window[-1] = yhat

    preds = np.array(preds, dtype=np.float32) * sigma + mu
    return preds.astype(float)
