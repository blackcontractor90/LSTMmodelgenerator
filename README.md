# Rainfall Model Generator

This repository contains a standalone Python script that trains a regression model on a CSV dataset and exports it to ONNX format. It is intentionally separated from any Java/GUI projects — this repo is focused only on training and exporting the model artifact(s) so they can be consumed by other projects (for example, a desktop app or a server-side service).

Important: The current script is an example of training an sklearn MLPRegressor (a feed-forward neural network). It is not an LSTM implementation. If you need an LSTM-based model, use a Keras/PyTorch implementation and a different ONNX conversion workflow.

Contents
- generate_rainfall_model.py — trains a StandardScaler + MLPRegressor pipeline and converts it to ONNX.
- requirements.txt — Python packages used while developing and testing the script.

Quick one-line repo description (for GitHub About)
Train an sklearn MLP regression pipeline on CSV rainfall data and export the model to ONNX for downstream consumption.

Why this repo is separate
Keeping the training/export code in its own repository keeps concerns separated:
- training & model export (Python, sklearn, skl2onnx) are isolated and reproducible,
- downstream consumers (Java apps, web services) only need the resulting ONNX and scaler artifacts,
- CI/CD, model versioning, and retraining pipelines can be added without touching UI code.

Requirements
- Python 3.8+
- Packages (see requirements.txt). Recommended to run in a virtual environment.

How to run
1. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

2. Put your training CSV in the repo (default filename `usa_rain2425.csv`) or change DATA_PATH at the top of the script.

3. Ensure the CSV contains a numeric target column named `rainfall` (or change TARGET_COLUMN in the script), and that all feature columns are numeric.

4. Run the script:
```bash
python generate_rainfall_model.py
```
- The script trains the pipeline and writes an ONNX model file named `rainfallmodel.onnx` by default.

Recommended additions (to the script)
- Save scaler artifacts so external consumers can apply identical preprocessing. Example (add after model.fit):
```python
scaler = regressor.named_steps['preprocessor'].named_transformers_['num']
import numpy as np
np.savetxt("scaler_mean.csv", scaler.mean_[None, :], delimiter=",")
np.savetxt("scaler_scale.csv", scaler.scale_[None, :], delimiter=",")
print("Saved scaler_mean.csv and scaler_scale.csv")
```

- Prefer a single named input tensor for ONNX consumers:
```python
from skl2onnx.common.data_types import FloatTensorType
initial_type = [("input", FloatTensorType([None, len(numeric_features)]))]
onnx_model = convert_sklearn(regressor, initial_types=initial_type)
```
This makes it easier for Java/ONNX runtimes that expect a single input array.

Notes and caveats
- The script currently trains an sklearn MLPRegressor; it is not a sequence model. If you intended to train an LSTM, use Keras/PyTorch and convert with tf2onnx or torch.onnx.
- The ONNX initial types and input naming matter for downstream code — verify the model's input shape and name using netron or the onnx python package.
- The script assumes all feature columns are numeric; it does not perform imputation or categorical encoding by default.
- Consider adding a train/test split and evaluation metrics (RMSE/MAE) before exporting the final model.

Suggested next steps
- Add CLI argument parsing (argparse) to specify data path, output path, hyperparameters.
- Save scaler mean/scale CSVs (recommended) so non-Python consumers can scale inputs identically.
- Add a small sample dataset (data/sample.csv) and a smoke-test script to validate predictions.
- Optionally add unit tests and a model validation step before exporting.

License
- Add a LICENSE file (MIT recommended if you want permissive reuse).

Maintainer
- @blackcontractor90
