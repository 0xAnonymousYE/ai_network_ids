"""
test_model.py
-------------
CLI tool to test trained ML model on new CSV file.

Usage:
    python test_model.py --file data/test.csv
"""

import os
import json
import pandas as pd
import joblib
from train_model import prepare_input_df, load_model_artifacts

# -------------------- Paths --------------------
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
ENCODERS_PATH = os.path.join(MODELS_DIR, "encoders.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "features.json")

# -------------------- Helper Functions --------------------
def load_model_artifacts():
    """Load trained model, encoders, and features from disk."""
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH) if os.path.exists(ENCODERS_PATH) else None
    features_list = None
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH,'r') as f:
            features_list = json.load(f)
    return model, encoders, features_list

def predict_on_csv(model, encoders, features_list, csv_df):
    """Predict classes for CSV and return results with new 'prediction' column."""
    df_input = csv_df.drop(columns=['class'], errors='ignore')
    df_prepared = prepare_input_df(df_input, encoders, features_list)
    preds = model.predict(df_prepared)
    df_results = csv_df.copy()
    df_results['prediction'] = preds
    return df_results

# -------------------- CLI --------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test trained ML model on CSV")
    parser.add_argument("--file", required=True, help="CSV file path to test")
    args = parser.parse_args()

    # Load model
    model, encoders, features_list = load_model_artifacts()
    if model is None:
        print("[ERROR] No trained model found. Train the model first.")
        exit()

    # Load test CSV
    test_df = pd.read_csv(args.file)
    print(f"[INFO] Test dataset loaded: {test_df.shape[0]} rows, {test_df.shape[1]} columns")

    # Predict
    results_df = predict_on_csv(model, encoders, features_list, test_df)
    print("[INFO] Prediction completed. Showing first 10 results:\n")
    print(results_df.head(10))

    # Save results CSV
    output_file = "predictions.csv"
    results_df.to_csv(output_file, index=False)
    print(f"[INFO] Predictions saved to {output_file}")
