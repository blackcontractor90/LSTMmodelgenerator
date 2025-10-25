import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import numpy as np
import os
import warnings
from sklearn.exceptions import ConvergenceWarning

# Configurations
DATA_PATH = "usa_rain2425.csv"
ONNX_MODEL_PATH = "rainfallmodel.onnx"
TARGET_COLUMN = "rainfall"
RANDOM_STATE = 42
HIDDEN_LAYERS = (64, 32)
MAX_ITER = 500

def main():
    # Load data
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"CSV file '{DATA_PATH}' not found.")

    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in the data.")

    # Drop rows with missing target
    df_clean = df.dropna(subset=[TARGET_COLUMN])

    # Separate features and target
    X = df_clean.drop(columns=[TARGET_COLUMN])
    y = df_clean[TARGET_COLUMN]

    # Check: all features are numeric
    if not all(np.issubdtype(X[col].dtype, np.number) for col in X.columns):
        raise ValueError("All feature columns must be numeric.")

    numeric_features = X.columns.tolist()
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features)
    ])

    # Model pipeline
    regressor = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', MLPRegressor(hidden_layer_sizes=HIDDEN_LAYERS,
                                   max_iter=MAX_ITER,
                                   random_state=RANDOM_STATE))
    ])

    # Fit model (with warnings suppressed)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        regressor.fit(X, y)

    # Convert to ONNX (fix initial_types to use column names)
    initial_type = [(col, FloatTensorType([None, 1])) for col in numeric_features]
    onnx_model = convert_sklearn(regressor, initial_types=initial_type)

    # Save model
    with open(ONNX_MODEL_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"Saved {ONNX_MODEL_PATH} successfully.")

if __name__ == "__main__":
    main()
