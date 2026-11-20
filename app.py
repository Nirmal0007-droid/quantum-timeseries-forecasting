"""
Quantum ML for Time-Series Forecasting - Interactive Dashboard (Streamlit)

Install:
    python -m pip install streamlit plotly torch pennylane scikit-learn pandas numpy

Run:
    python -m streamlit run app.py
"""

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pennylane as qml
import streamlit as st
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# ----------------------------------------------------------------------
# PAGE SETUP + STYLE
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Quantum Time-Series Forecasting",
    page_icon="⚛️",
    layout="wide",
)

st.markdown(
    """
<style>
.block-container {padding-top: 1.4rem; max-width: 1300px;}
.hero {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    padding: 1.6rem 2rem; border-radius: 14px; margin-bottom: 1.2rem;
}
.hero h1 {margin: 0; font-size: 1.9rem; color: #ffffff;}
.hero p {margin: .35rem 0 0; color: #d5e3ea; font-size: 1rem;}
div[data-testid="stMetric"] {
    background: rgba(128,128,128,0.08);
    border: 1px solid rgba(128,128,128,0.22);
    padding: 12px 16px; border-radius: 10px;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>⚛️ Quantum Machine Learning for Time-Series Forecasting</h1>
  <p>Hybrid quantum-classical models compared with classical deep-learning baselines
  &nbsp;|&nbsp; Advanced Machine Learning &amp; Deep Learning</p>
</div>
""",
    unsafe_allow_html=True,
)

HYBRID = "Hybrid (LSTM + Quantum)"
ALL_MODELS = ["Naive", "LSTM", "GRU", HYBRID]


# ----------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------
def make_synthetic(n, noise, period):
    rng = np.random.default_rng(42)
    t = np.arange(n)
    s = 20 + 8 * np.sin(2 * np.pi * t / period) + 0.01 * t + rng.normal(0, noise, n)
    return pd.DataFrame({"value": s.astype(np.float32)})


def split_scale(series, lookback):
    n = len(series)
    i1, i2 = int(n * 0.70), int(n * 0.85)
    scaler = MinMaxScaler().fit(series[:i1].reshape(-1, 1))   # fit on TRAIN only

    def sc(a):
        return scaler.transform(a.reshape(-1, 1)).astype(np.float32).ravel()

    parts = (sc(series[:i1]), sc(series[i1:i2]), sc(series[i2:]))
    if min(len(p) for p in parts) < lookback + 10:
        raise ValueError("Not enough data for this lookback window. "
                         "Use more data points or a smaller lookback.")
    return parts, scaler, (i1, i2)


def make_windows(arr, lookback):
    X = np.array([arr[i:i + lookback] for i in range(len(arr) - lookback)])
    y = np.array([arr[i + lookback] for i in range(len(arr) - lookback)])
    return torch.tensor(X).unsqueeze(-1), torch.tensor(y).unsqueeze(-1)


# ----------------------------------------------------------------------
# MODELS
# ----------------------------------------------------------------------
class RNNModel(nn.Module):
    """Classical baseline: LSTM or GRU followed by a linear output layer."""

    def __init__(self, kind, hidden):
        super().__init__()
        cell = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = cell(input_size=1, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


def build_circuit(n_qubits):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


class HybridModel(nn.Module):
    """LSTM -> Linear (to n_qubits numbers) -> Variational Quantum Circuit -> Linear."""

    def __init__(self, hidden, n_qubits, q_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.to_q = nn.Linear(hidden, n_qubits)
        self.circuit = build_circuit(n_qubits)
        self.qlayer = qml.qnn.TorchLayer(self.circuit, {"weights": (q_layers, n_qubits)})
        self.out = nn.Linear(n_qubits, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        z = torch.tanh(self.to_q(out[:, -1, :])) * np.pi   # angles in [-pi, pi]
        return self.out(self.qlayer(z).float())


def build_model(name, cfg):
    if name == "LSTM":
        return RNNModel("lstm", cfg["hidden"])
    if name == "GRU":
        return RNNModel("gru", cfg["hidden"])
    return HybridModel(cfg["hidden"], cfg["n_qubits"], cfg["q_layers"])


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------
# TRAINING + EVALUATION
# ----------------------------------------------------------------------
def train_model(model, Xtr, ytr, Xva, yva, epochs, lr, batch, on_epoch):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_hist, val_hist = [], []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        run, nb = 0.0, 0
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            run += loss.item()
            nb += 1
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xva), yva).item()
        train_hist.append(run / nb)
        val_hist.append(val)
        on_epoch(ep + 1, run / nb, val)
    return train_hist, val_hist


def evaluate(pred_scaled, true_scaled, scaler):
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    true = scaler.inverse_transform(true_scaled.reshape(-1, 1)).ravel()
    mae = float(mean_absolute_error(true, pred))
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    mape = float(np.mean(np.abs((true - pred) / np.maximum(np.abs(true), 1e-8))) * 100)
    return mae, rmse, mape, pred, true


def run_experiment(series, cfg, models, seeds, bar, status):
    (tr, va, te), scaler, split_idx = split_scale(series, cfg["lookback"])
    Xtr, ytr = make_windows(tr, cfg["lookback"])
    Xva, yva = make_windows(va, cfg["lookback"])
    Xte, yte = make_windows(te, cfg["lookback"])
    yte_np = yte.squeeze(-1).numpy()

    trainable = [m for m in models if m != "Naive"]
    total_steps = max(1, len(trainable) * seeds * cfg["epochs"])
    done = {"n": 0}

    rows, preds, hists = [], {}, {}
    true_real = None

    if "Naive" in models:
        mae, rmse, mape, pred, true_real = evaluate(Xte[:, -1, 0].numpy(), yte_np, scaler)
        rows.append(dict(Model="Naive", Seed=0, MAE=mae, RMSE=rmse, MAPE=mape,
                         Params=0, Time=0.0))
        preds["Naive"] = pred

    for name in trainable:
        for seed in range(seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = build_model(name, cfg)

            def on_epoch(ep, tl, vl, name=name, seed=seed):
                done["n"] += 1
                bar.progress(min(1.0, done["n"] / total_steps))
                status.write(f"Training **{name}** | seed {seed + 1}/{seeds} | "
                             f"epoch {ep}/{cfg['epochs']} | val loss {vl:.5f}")

            t0 = time.time()
            th, vh = train_model(model, Xtr, ytr, Xva, yva,
                                 cfg["epochs"], cfg["lr"], cfg["batch"], on_epoch)
            elapsed = time.time() - t0

            model.eval()
            with torch.no_grad():
                p = model(Xte).squeeze(-1).numpy()
            mae, rmse, mape, pred, true_real = evaluate(p, yte_np, scaler)
            rows.append(dict(Model=name, Seed=seed, MAE=mae, RMSE=rmse, MAPE=mape,
                             Params=count_params(model), Time=elapsed))
            if seed == 0:
                preds[name] = pred
                hists[name] = (th, vh)

    bar.progress(1.0)
    status.write("Done.")
    return dict(rows=pd.DataFrame(rows), preds=preds, hists=hists,
                true=true_real, split_idx=split_idx, cfg=cfg, seeds=seeds)


# ----------------------------------------------------------------------
# SIDEBAR (controls)
# ----------------------------------------------------------------------
st.sidebar.header("1. Data")
source = st.sidebar.radio("Data source", ["Synthetic demo", "Upload CSV"])

if source == "Synthetic demo":
    n_points = st.sidebar.slider("Number of points", 300, 3000, 1000, step=100)
    noise = st.sidebar.slider("Noise level", 0.0, 3.0, 0.8, step=0.1)
    period = st.sidebar.slider("Seasonal period", 10, 200, 50, step=5)
    df = make_synthetic(n_points, noise, period)
    value_col, x_axis = "value", df.index.values
    data_name = "Synthetic series"
else:
    up = st.sidebar.file_uploader("CSV file", type=["csv"])
    if up is None:
        st.info("Upload a CSV file in the sidebar, or switch to the synthetic demo.")
        st.stop()
    raw = pd.read_csv(up)
    num_cols = raw.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        st.error("No numeric columns found in this file.")
        st.stop()
    value_col = st.sidebar.selectbox("Value column (to forecast)", num_cols)
    date_col = st.sidebar.selectbox("Date column (optional)", ["(none)"] + raw.columns.tolist())
    df = raw.copy()
    if date_col != "(none)":
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col)
    df = df.dropna(subset=[value_col]).reset_index(drop=True)
    x_axis = df[date_col].values if date_col != "(none)" else df.index.values
    data_name = up.name

series = df[value_col].values.astype(np.float32)

st.sidebar.header("2. Models")
models = st.sidebar.multiselect("Models to compare", ALL_MODELS, default=ALL_MODELS)
lookback = st.sidebar.slider("Lookback window", 3, 60, 10)
hidden = st.sidebar.slider("LSTM/GRU hidden size", 4, 64, 8)

st.sidebar.header("3. Quantum circuit")
n_qubits = st.sidebar.slider("Qubits", 2, 8, 4)
q_layers = st.sidebar.slider("Circuit layers", 1, 4, 2)

st.sidebar.header("4. Training")
epochs = st.sidebar.slider("Epochs", 5, 60, 20)
lr = st.sidebar.select_slider("Learning rate", [0.001, 0.003, 0.005, 0.01, 0.02, 0.05], value=0.01)
batch = st.sidebar.select_slider("Batch size", [16, 32, 64, 128], value=32)
seeds = st.sidebar.slider("Random seeds (repeats)", 1, 5, 3)

if HYBRID in models:
    st.sidebar.caption("The quantum model runs on a simulator, so it trains slower than the classical models.")

run_clicked = st.sidebar.button("▶ Run experiment", type="primary")

cfg = dict(lookback=lookback, hidden=hidden, n_qubits=n_qubits, q_layers=q_layers,
           epochs=epochs, lr=lr, batch=batch)

# ----------------------------------------------------------------------
# RUN
# ----------------------------------------------------------------------
if run_clicked:
    if not models:
        st.warning("Select at least one model.")
    else:
        bar = st.progress(0.0)
        status = st.empty()
        try:
            st.session_state["results"] = run_experiment(series, cfg, models, seeds, bar, status)
            st.session_state["data_name"] = data_name
        except ValueError as e:
            st.error(str(e))
        bar.empty()
        status.empty()

# ----------------------------------------------------------------------
# TABS
# ----------------------------------------------------------------------
tab_data, tab_results, tab_train, tab_quantum, tab_about = st.tabs(
    ["📊 Data", "🏆 Results", "📉 Training", "⚛️ Quantum circuit", "ℹ️ About"]
)

# ---- DATA TAB ----
with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", f"{len(series):,}")
    c2.metric("Mean", f"{series.mean():.2f}")
    c3.metric("Std. deviation", f"{series.std():.2f}")
    c4.metric("Min / Max", f"{series.min():.1f} / {series.max():.1f}")

    i1, i2 = int(len(series) * 0.70), int(len(series) * 0.85)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_axis, y=series, mode="lines", name=value_col))
    fig.add_vrect(x0=x_axis[0], x1=x_axis[i1 - 1], fillcolor="#2ca02c", opacity=0.07,
                  line_width=0, annotation_text="Train 70%", annotation_position="top left")
    fig.add_vrect(x0=x_axis[i1], x1=x_axis[i2 - 1], fillcolor="#ff7f0e", opacity=0.09,
                  line_width=0, annotation_text="Validation 15%", annotation_position="top left")
    fig.add_vrect(x0=x_axis[i2], x1=x_axis[-1], fillcolor="#d62728", opacity=0.07,
                  line_width=0, annotation_text="Test 15%", annotation_position="top left")
    fig.update_layout(title=f"{data_name}: chronological train / validation / test split",
                      height=420, margin=dict(l=10, r=10, t=50, b=10), xaxis_title="Time",
                      yaxis_title=value_col)
    st.plotly_chart(fig)
    st.caption("The split is chronological (never shuffled) and the scaler is fitted on the "
               "training part only, which prevents data leakage.")
    with st.expander("Preview data"):
        st.dataframe(df.head(100))

# ---- RESULTS TAB ----
with tab_results:
    res = st.session_state.get("results")
    if res is None:
        st.info("Set your options in the sidebar and press **Run experiment**.")
    else:
        rows = res["rows"]
        summary = rows.groupby("Model", sort=False).agg(
            MAE=("MAE", "mean"), MAE_sd=("MAE", "std"),
            RMSE=("RMSE", "mean"), RMSE_sd=("RMSE", "std"),
            MAPE=("MAPE", "mean"), Params=("Params", "first"), Time=("Time", "mean"),
        ).fillna(0.0).reset_index()

        best = summary.loc[summary["RMSE"].idxmin()]
        naive_row = summary[summary["Model"] == "Naive"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best model (RMSE)", best["Model"])
        c2.metric("Best RMSE", f"{best['RMSE']:.3f}")
        c3.metric("Best MAE", f"{summary['MAE'].min():.3f}")
        if len(naive_row) and best["Model"] != "Naive":
            gain = (1 - best["RMSE"] / naive_row["RMSE"].iloc[0]) * 100
            c4.metric("Improvement vs naive", f"{gain:.1f}%")
        else:
            c4.metric("Seeds used", res["seeds"])

        st.subheader("Comparison table")
        show = pd.DataFrame({
            "Model": summary["Model"],
            "MAE": [f"{m:.3f} ± {s:.3f}" for m, s in zip(summary["MAE"], summary["MAE_sd"])],
            "RMSE": [f"{m:.3f} ± {s:.3f}" for m, s in zip(summary["RMSE"], summary["RMSE_sd"])],
            "MAPE (%)": summary["MAPE"].round(2),
            "Trainable params": summary["Params"].astype(int),
            "Train time (s)": summary["Time"].round(1),
        })
        st.dataframe(show, hide_index=True)
        st.caption("Values are mean ± standard deviation across random seeds. "
                   "MAPE can be misleading when values are close to zero.")

        left, right = st.columns(2)
        with left:
            bar_fig = go.Figure(go.Bar(x=summary["Model"], y=summary["RMSE"],
                                       error_y=dict(type="data", array=summary["RMSE_sd"])))
            bar_fig.update_layout(title="RMSE by model (lower is better)", height=380,
                                  margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(bar_fig)
        with right:
            time_fig = go.Figure(go.Bar(x=summary["Model"], y=summary["Time"]))
            time_fig.update_layout(title="Training time (seconds)", height=380,
                                   margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(time_fig)

        st.subheader("Forecast vs actual (test set)")
        true = res["true"]
        n_show = st.slider("Points to display", 30, len(true), min(len(true), 150))
        fc = go.Figure()
        fc.add_trace(go.Scatter(y=true[:n_show], mode="lines", name="Actual",
                                line=dict(color="black", width=2)))
        for name, p in res["preds"].items():
            fc.add_trace(go.Scatter(y=p[:n_show], mode="lines", name=name,
                                    line=dict(dash="dot" if name == "Naive" else "solid")))
        fc.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10),
                         xaxis_title="Test time step", yaxis_title=value_col,
                         legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fc)

        st.download_button("⬇ Download raw results (CSV)",
                           rows.to_csv(index=False).encode("utf-8"),
                           file_name="forecasting_results.csv", mime="text/csv")

# ---- TRAINING TAB ----
with tab_train:
    res = st.session_state.get("results")
    if res is None or not res["hists"]:
        st.info("Run an experiment with at least one trainable model to see training curves.")
    else:
        fig = go.Figure()
        for name, (th, vh) in res["hists"].items():
            fig.add_trace(go.Scatter(y=vh, mode="lines", name=f"{name} (validation)"))
            fig.add_trace(go.Scatter(y=th, mode="lines", name=f"{name} (train)",
                                     line=dict(dash="dash"), opacity=0.55))
        fig.update_layout(title="Loss during training (seed 1)", height=450,
                          xaxis_title="Epoch", yaxis_title="MSE loss (scaled data)",
                          margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig)
        st.caption("If validation loss rises while training loss keeps falling, the model is overfitting.")

# ---- QUANTUM TAB ----
with tab_quantum:
    st.subheader("Variational quantum circuit used in the hybrid model")
    st.markdown(
        f"- **Qubits:** {n_qubits}  \n- **Trainable layers:** {q_layers}  \n"
        f"- **Trainable quantum parameters:** {n_qubits * q_layers}  \n"
        f"- **Encoding:** angle encoding  \n- **Readout:** Pauli-Z expectation value per qubit"
    )
    try:
        circ = build_circuit(n_qubits)
        diagram = qml.draw(circ)(torch.zeros(n_qubits), torch.zeros(q_layers, n_qubits))
        st.code(diagram, language="text")
    except Exception as e:  # drawing is optional
        st.warning(f"Could not draw the circuit: {e}")

    st.markdown("**Model pipeline**")
    st.code(
        f"Past {lookback} values\n"
        f"   -> LSTM (hidden size {hidden})\n"
        f"   -> Linear layer (compress to {n_qubits} numbers)\n"
        f"   -> Angle encoding into {n_qubits} qubits\n"
        f"   -> Variational circuit ({q_layers} layers, entangling gates)\n"
        f"   -> {n_qubits} expectation values\n"
        f"   -> Linear layer -> next-step forecast",
        language="text",
    )

# ---- ABOUT TAB ----
with tab_about:
    st.markdown(
        """
### Project summary
This dashboard compares **hybrid quantum-classical** forecasting models with **classical
deep-learning baselines** on time-series data.

**Research question:** Does inserting a variational quantum layer into an LSTM improve forecasting
accuracy or parameter efficiency compared with purely classical models?

### Method
1. Chronological 70/15/15 train/validation/test split; scaler fitted on training data only.
2. Sliding-window supervised dataset (lookback window to next value).
3. Models: Naive baseline, LSTM, GRU, and a Hybrid LSTM + variational quantum circuit.
4. Training with Adam and MSE loss; repeated over several random seeds.
5. Metrics: MAE, RMSE, MAPE; plus trainable parameter count and training time.

### Limitations
- Quantum circuits run on a classical **simulator**, limited to a few qubits.
- Simulation is slower than classical training; no hardware speed-up is claimed.
- On small problems, quantum models typically **match** rather than beat classical ones.
        """
    )
