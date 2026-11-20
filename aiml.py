"""
Quantum ML for Time-Series Forecasting - beginner project
Compares: Naive baseline  vs  LSTM  vs  Hybrid (LSTM + quantum layer)

Install first:
    pip install torch pennylane scikit-learn matplotlib numpy pandas

Run:
    python quantum_forecasting_project.py
"""

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------------------------------------------------------------
# SETTINGS (change these to experiment)
# ----------------------------------------------------------------------
LOOKBACK = 10      # how many past values the model sees
N_QUBITS = 4       # keep small: 4 to 8
Q_LAYERS = 2       # depth of the quantum circuit
HIDDEN = 8         # LSTM hidden size
EPOCHS = 20
LR = 0.01
BATCH = 32
SEEDS = [0, 1, 2]  # repeat experiments with different random seeds


# ----------------------------------------------------------------------
# STEP 1: GET DATA
# A synthetic "temperature-like" series so you can run without downloads.
# To use real data, replace this function (see bottom of the guide).
# ----------------------------------------------------------------------
def load_data(n=1000):
    rng = np.random.default_rng(42)
    t = np.arange(n)
    series = 20 + 8 * np.sin(2 * np.pi * t / 50) + 0.01 * t + rng.normal(0, 0.8, n)
    return series.astype(np.float32)


# ----------------------------------------------------------------------
# STEP 2: SPLIT BY TIME, THEN SCALE (fit scaler on training data only)
# ----------------------------------------------------------------------
def split_and_scale(series):
    n = len(series)
    i_train, i_val = int(n * 0.70), int(n * 0.85)

    train, val, test = series[:i_train], series[i_train:i_val], series[i_val:]

    scaler = MinMaxScaler()
    scaler.fit(train.reshape(-1, 1))          # learn the scale from TRAIN only

    def tf(a):
        return scaler.transform(a.reshape(-1, 1)).astype(np.float32).ravel()

    return tf(train), tf(val), tf(test), scaler


# ----------------------------------------------------------------------
# STEP 3: SLIDING WINDOWS
# Input = last LOOKBACK values, target = the next value
# ----------------------------------------------------------------------
def make_windows(arr, lookback=LOOKBACK):
    X, y = [], []
    for i in range(len(arr) - lookback):
        X.append(arr[i:i + lookback])
        y.append(arr[i + lookback])
    X = torch.tensor(np.array(X)).unsqueeze(-1)   # shape (N, lookback, 1)
    y = torch.tensor(np.array(y)).unsqueeze(-1)   # shape (N, 1)
    return X, y


# ----------------------------------------------------------------------
# STEP 4: CLASSICAL MODEL (LSTM)
# ----------------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)           # out: (batch, time, hidden)
        return self.fc(out[:, -1, :])   # use the last time step


# ----------------------------------------------------------------------
# STEP 5: QUANTUM LAYER + HYBRID MODEL
# ----------------------------------------------------------------------
dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # 1) encode classical numbers as rotation angles
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    # 2) trainable rotations + entanglement
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    # 3) measure: one number between -1 and +1 per qubit
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


class HybridModel(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.to_qubits = nn.Linear(hidden, N_QUBITS)       # compress to N_QUBITS numbers
        weight_shapes = {"weights": (Q_LAYERS, N_QUBITS)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.out = nn.Linear(N_QUBITS, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        z = self.to_qubits(out[:, -1, :])
        z = torch.tanh(z) * np.pi          # keep angles in [-pi, pi]
        q = self.qlayer(z).float()         # quantum layer
        return self.out(q)


# ----------------------------------------------------------------------
# STEP 6: TRAINING LOOP
# ----------------------------------------------------------------------
def train(model, Xtr, ytr, Xva, yva, name=""):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    val_history = []

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva), yva).item()
        val_history.append(val_loss)
        print(f"[{name}] epoch {epoch + 1:2d}/{EPOCHS}  val loss: {val_loss:.5f}")

    return val_history


# ----------------------------------------------------------------------
# STEP 7: EVALUATION (convert back to real units, then compute errors)
# ----------------------------------------------------------------------
def evaluate(pred_scaled, true_scaled, scaler):
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    true = scaler.inverse_transform(true_scaled.reshape(-1, 1)).ravel()
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    return mae, rmse, pred, true


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------
# STEP 8: RUN EVERYTHING
# ----------------------------------------------------------------------
def main():
    series = load_data()
    train_s, val_s, test_s, scaler = split_and_scale(series)
    Xtr, ytr = make_windows(train_s)
    Xva, yva = make_windows(val_s)
    Xte, yte = make_windows(test_s)
    print("Train/Val/Test windows:", len(Xtr), len(Xva), len(Xte))

    # Baseline: "next value = last value in the window"
    naive_pred = Xte[:, -1, 0].numpy()
    n_mae, n_rmse, _, _ = evaluate(naive_pred, yte.squeeze().numpy(), scaler)
    print(f"\nNAIVE baseline  MAE={n_mae:.3f}  RMSE={n_rmse:.3f}")

    results = {"LSTM": [], "Hybrid": []}
    last_preds, last_hist = {}, {}

    for seed in SEEDS:
        for name, cls in [("LSTM", LSTMModel), ("Hybrid", HybridModel)]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = cls()
            print(f"\n=== {name} | seed {seed} | trainable params: {count_params(model)} ===")
            hist = train(model, Xtr, ytr, Xva, yva, name=name)

            model.eval()
            with torch.no_grad():
                pred = model(Xte).squeeze().numpy()
            mae, rmse, pred_real, true_real = evaluate(pred, yte.squeeze().numpy(), scaler)
            results[name].append((mae, rmse))
            last_preds[name], last_hist[name] = pred_real, hist

    # ---- summary table ----
    print("\n================ RESULTS (mean +/- std over seeds) ================")
    print(f"{'Naive':8s} MAE={n_mae:.3f}  RMSE={n_rmse:.3f}")
    for name, vals in results.items():
        arr = np.array(vals)
        print(f"{name:8s} MAE={arr[:, 0].mean():.3f}+/-{arr[:, 0].std():.3f}   "
              f"RMSE={arr[:, 1].mean():.3f}+/-{arr[:, 1].std():.3f}")

    # ---- plots ----
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    ax[0].plot(true_real, label="Actual", color="black")
    for name, p in last_preds.items():
        ax[0].plot(p, label=name, alpha=0.8)
    ax[0].set_title("Test set: actual vs predicted")
    ax[0].set_xlabel("time step")
    ax[0].set_ylabel("value")
    ax[0].legend()

    for name, h in last_hist.items():
        ax[1].plot(h, label=name)
    ax[1].set_title("Validation loss during training")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("MSE (scaled)")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig("results.png", dpi=150)
    print("\nSaved plot to results.png")


if __name__ == "__main__":
    main()