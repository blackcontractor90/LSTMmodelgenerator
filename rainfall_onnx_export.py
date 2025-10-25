import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import numpy as np
import os
import warnings
from sklearn.exceptions import ConvergenceWarning

# Configurations
DATA_PATH = "msiaraindatasetv1.csv"
ONNX_MODEL_PATH = "msiarainfallmodel.onnx"
TARGET_COLUMN = "rainfall"
RANDOM_STATE = 42
HIDDEN_LAYERS = (64, 32)
MAX_ITER = 500

# Update these to match your CSV column names exactly!
FEATURES = [
    "state",            # categorical
    "height",           # numeric
    "minMeanTemp",      # numeric
    "maxMeanTemp",      # numeric
    "meanRelHum"        # numeric                              
]

def main():
    # Load data
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"CSV file '{DATA_PATH}' not found.")

    df = pd.read_csv(DATA_PATH)

    # Replace '-' with NaN in all relevant columns
    for col in [TARGET_COLUMN] + FEATURES:
        df[col] = df[col].replace("-", np.nan)

    # Clean numeric columns: remove commas, parentheses, 'm', and convert to float
    for col in ["height", "minMeanTemp", "maxMeanTemp", "meanRelHum"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
            .str.replace("m", "", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean target column
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in the data.")

    # Drop rows with missing target or features
    df_clean = df.dropna(subset=[TARGET_COLUMN] + FEATURES)

    # One-hot encode 'state'
    X_raw = df_clean[FEATURES]
    X = pd.get_dummies(X_raw, columns=['state'])
    print("DEBUG: Feature order:", list(X.columns))
    X.columns.to_series().to_csv('feature_order.csv', index=False)
    y = df_clean[TARGET_COLUMN]

    # Preprocessing (only StandardScaler needed)
    preprocessor = StandardScaler()

    # Model pipeline
    regressor = Pipeline([
        ("scaler", preprocessor),
        ("regressor", MLPRegressor(hidden_layer_sizes=HIDDEN_LAYERS,
                                   max_iter=MAX_ITER,
                                   random_state=RANDOM_STATE))
    ])

    # Fit model (with warnings suppressed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        regressor.fit(X, y)

    scaler = regressor.named_steps['scaler']
    np.savetxt("scaler_mean.csv", scaler.mean_.reshape(1, -1), delimiter=",")
    np.savetxt("scaler_scale.csv", scaler.scale_.reshape(1, -1), delimiter=",")

    # ONNX initial_types: now a single float tensor with n_features columns
    n_features = X.shape[1]
    initial_type = [("float_input", FloatTensorType([None, n_features]))]

    onnx_model = convert_sklearn(
        regressor, 
        initial_types=initial_type,
        options={id(regressor): {'input_name': "float_input"}}
    )

    # Save model
    with open(ONNX_MODEL_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"Saved {ONNX_MODEL_PATH} successfully. n_features={n_features}")

    # === DEBUG SECTION: Print and compare a single prediction (for row 0) ===
    print("\n--- DEBUG: Single Row Prediction Check ---")
    row_idx = 0  # You can change this index for another row
    X_row = X.iloc[row_idx]
    y_row = y.iloc[row_idx]

    # Print raw features (after one-hot, before scaling)
    print("PYTHON RAW FEATURES:", X_row.values.tolist())

    # Convert to DataFrame to preserve feature names for scaler (fixes warning)
    X_row_df = pd.DataFrame([X_row.values], columns=X.columns)
    scaled_features = scaler.transform(X_row_df)[0]
    print("PYTHON SCALED FEATURES:", scaled_features.tolist())

    # Make prediction
    pred = regressor.predict(X_row_df)[0]
    print("PYTHON PREDICTION:", pred)
    print("PYTHON ACTUAL:", y_row)

    # Also print state column for clarity
    print("PYTHON STATE VALUE:", X_raw.iloc[row_idx]['state'])

if __name__ == "__main__":
    main()
