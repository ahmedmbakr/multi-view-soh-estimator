import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

NUMBER_OF_FREQUENCY_POINTS = 61  # Expected number of frequency points in each EIS measurement

class TinyNyquistCNN(nn.Module):
    """
    Input: [B, 2, NUMBER_OF_FREQUENCY_POINTS]  -> scalar SOH in 0..100
    """
    def __init__(self, in_ch=2, p_drop=0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(in_ch, 16, kernel_size=5, padding=2, bias=False),
            nn.GELU(),
            nn.BatchNorm1d(16),
            nn.Dropout(p_drop),

            nn.Conv1d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 32, kernel_size=7, padding=3, bias=False),
            nn.GELU(),
            nn.BatchNorm1d(32),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)  # -> [B, 32, 1]
        self.head = nn.Sequential(
            nn.Flatten(),          # -> [B, 32]
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(16, 1),
        )

    def forward(self, x):  # x: [B, 2, NUMBER_OF_FREQUENCY_POINTS]
        z = self.backbone(x)
        z = self.gap(z)
        y = self.head(z)      # [B, 1]
        return y.squeeze(-1)  # [B]

# ---------- Dataset ----------
class NyquistArrayDataset(Dataset):
    """
    X: numpy array [N, NUMBER_OF_FREQUENCY_POINTS, 2]  where last dim = [Z_real, Z_imag]
    y: numpy array [N]  SOH in 0..100
    Normalizes per-channel, per-frequency using training stats.
    """
    def __init__(self, X, y=None, mean=None, std=None):
        X = np.asarray(X, dtype=np.float32)
        assert X.ndim == 3 and X.shape[1] == NUMBER_OF_FREQUENCY_POINTS and X.shape[2] == 2, \
            "Expected X shape (N, NUMBER_OF_FREQUENCY_POINTS, 2) with last dim [Z_real, Z_imag]"

        # move channels last -> channels first for Conv1d
        X = np.transpose(X, (0, 2, 1))   # [N, 2, NUMBER_OF_FREQUENCY_POINTS]

        if mean is None or std is None:
            mean = X.mean(axis=0, keepdims=True)            # [1, 2, NUMBER_OF_FREQUENCY_POINTS]
            std  = X.std(axis=0, keepdims=True) + 1e-8      # [1, 2, NUMBER_OF_FREQUENCY_POINTS]
        self.mean, self.std = mean, std

        self.X = (X - self.mean) / self.std
        self.y = None if y is None else np.asarray(y, dtype=np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        xb = torch.from_numpy(self.X[idx])     # [2, NUMBER_OF_FREQUENCY_POINTS]
        if self.y is None:
            return xb
        yb = torch.tensor(self.y[idx])         # scalar
        return xb, yb

# ---------- Metrics ----------
def _compute_regression_metrics(y_true, y_pred):
    """Compute RMSE, MAE, MAPE given numpy arrays of targets and predictions."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    eps = 1e-6
    mape = float(np.mean(np.abs(err) / (np.abs(y_true) + eps)) * 100.0)
    return {"rmse": rmse, "mae": mae, "mape": mape}

@torch.no_grad()
def evaluate_metrics_on_loader(model, loader, device=None):
    """Run model on a DataLoader of (xb, yb) and return RMSE/MAE/MAPE."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        yhat = model(xb)
        preds.append(yhat.detach().cpu().numpy())
        trues.append(yb.detach().cpu().numpy())
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)
    return _compute_regression_metrics(y_true, y_pred)

# ---------- Training / Eval ----------
def train_nyquist_cnn_from_arrays(
    train_X, train_y, val_X=None, val_y=None,
    epochs=200, batch_size=32, lr=1e-3, weight_decay=1e-4,
    huber_delta=1.0, early_patience=20, device=None
):
    print("Training TinyNyquistCNN model with the following parameters:")
    print(f"  epochs={epochs}, batch_size={batch_size}, lr={lr}, weight_decay={weight_decay}")
    print(f"  huber_delta={huber_delta}, early_patience={early_patience}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Fit normalization on train only
    tmp_ds = NyquistArrayDataset(train_X, train_y)
    mean, std = tmp_ds.mean, tmp_ds.std

    train_ds = NyquistArrayDataset(train_X, train_y, mean=mean, std=std)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_ld = None
    if val_X is not None and val_y is not None:
        val_ds = NyquistArrayDataset(val_X, val_y, mean=mean, std=std)
        val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = TinyNyquistCNN().to(device)
    criterion = nn.HuberLoss(delta=huber_delta)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(10, epochs))

    best_val = math.inf
    best_state = None
    wait = 0
    logs = {"train_loss": [], "val_loss": [], "val_rmse": [], "val_mae": [], "val_mape": []}

    for ep in range(epochs):
        # --- train ---
        model.train()
        tr_total = 0.0
        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)                   # SOH in 0..100
            loss = criterion(pred, yb)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            tr_total += loss.item() * xb.size(0)
        tr_loss = tr_total / len(train_ds)
        logs["train_loss"].append(tr_loss)

        # --- val ---
        if val_ld is not None:
            model.eval()
            va_total = 0.0
            with torch.no_grad():
                for xb, yb in val_ld:
                    xb, yb = xb.to(device), yb.to(device)
                    va_total += criterion(model(xb), yb).item() * xb.size(0)
            va_loss = va_total / len(val_ld.dataset)
            logs["val_loss"].append(va_loss)

            # compute validation metrics (RMSE/MAE/MAPE)
            val_metrics = evaluate_metrics_on_loader(model, val_ld, device=device)
            logs["val_rmse"].append(val_metrics["rmse"])
            logs["val_mae"].append(val_metrics["mae"])
            logs["val_mape"].append(val_metrics["mape"])

            if va_loss < best_val - 1e-6:
                best_val = va_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= early_patience:
                    break

        sched.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, (mean, std), logs

@torch.no_grad()
def predict_soh_arrays(model, X, norm_stats, batch_size=64, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = norm_stats
    ds = NyquistArrayDataset(X, y=None, mean=mean, std=std)
    ld = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model.eval().to(device)
    out = []
    for xb in ld:
        xb = xb.to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0)  # SOH in 0..100
