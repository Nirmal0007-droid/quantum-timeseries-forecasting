# Quantum Machine Learning for Time-Series Forecasting

Hybrid quantum-classical forecasting models compared with classical deep-learning baselines.
Project for **Advanced Machine Learning and Deep Learning** (PG).

**Live demo:** _paste your Streamlit link here_

## What it does
- Forecasts a time series with four approaches: **Naive**, **LSTM**, **GRU**, and a **Hybrid LSTM + variational quantum circuit**
- Chronological 70/15/15 train/validation/test split, scaler fitted on training data only
- Metrics: MAE, RMSE, MAPE, trainable parameter count, training time, mean ± std over random seeds
- Interactive dashboard (Streamlit) with data, results, training curves, and the quantum circuit diagram
- Quantum circuits run on a **simulator** (PennyLane), so no quantum hardware is needed

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard |
| `aiml.py` | Command-line version of the experiment |
| `requirements.txt` | Python dependencies |
| `run_app.bat` | One-click launcher for Windows |

## Run locally
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m streamlit run app.py
```

## Method
1. Sliding-window dataset (lookback window to next value)
2. Models trained with Adam and MSE loss
3. Hybrid model: LSTM, then a linear layer to N numbers, then angle encoding into N qubits, then a variational circuit (entangling layers), then Pauli-Z expectation values, then a linear output layer
4. Experiments repeated over several random seeds

## Limitations
- Simulated quantum circuits are limited to a few qubits and train slower than classical models
- On small problems, quantum models typically match rather than beat classical ones
