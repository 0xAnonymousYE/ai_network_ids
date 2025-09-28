"""
train_model.py
---------------
CLI tool to train a machine learning model for network IDS or any tabular dataset.

Usage:
    python train_model.py --file data/train.csv --target class --algo RandomForest --test-size 20
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# -------------------- Paths --------------------
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
ENCODERS_PATH = os.path.join(MODELS_DIR, "encoders.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "features.json")

# -------------------- Helper Functions --------------------
def save_model(model, encoders, features_list):
    """Save trained model, label encoders, and feature list to disk."""
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    with open(FEATURES_PATH, 'w') as f:
        json.dump(features_list, f)
    print(f"[INFO] Model, encoders, and features saved in '{MODELS_DIR}'")

def apply_label_encoders(df, encoders):
    """Apply previously fitted LabelEncoders to dataframe."""
    if encoders is None: return df
    df2 = df.copy()
    for col, le in encoders.items():
        if col not in df2.columns: continue
        vals = df2[col].astype(str).tolist()
        classes = list(le.classes_)
        new_vals = []
        for v in vals:
            if v in classes: new_vals.append(v)
            elif 'other' in classes: new_vals.append('other')
            else: classes.append(v); new_vals.append(v)
        try: le.classes_ = np.array(classes)
        except: pass
        try: df2[col] = le.transform(new_vals)
        except: df2[col] = 0
    return df2

def prepare_input_df(raw_df, encoders, features_list):
    """Prepare dataframe for prediction: encode labels, convert to numeric, reindex columns."""
    df = apply_label_encoders(raw_df.copy(), encoders)
    for c in df.columns:
        try: df[c] = pd.to_numeric(df[c])
        except: pass
    if features_list is not None:
        df = df.reindex(columns=features_list, fill_value=0)
    return df

# -------------------- CLI --------------------
def main():
    parser = argparse.ArgumentParser(description="Train ML model from CSV")
    parser.add_argument("--file", required=True, help="CSV file path")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--algo", default="RandomForest", choices=["RandomForest","DecisionTree","GradientBoosting"], help="Algorithm choice")
    parser.add_argument("--test-size", type=int, default=20, help="Test size percentage")
    args = parser.parse_args()

    # Load dataset
    df = pd.read_csv(args.file)
    print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Identify features and target
    y_col = args.target
    X_cols = [c for c in df.columns if c != y_col]

    # Encode categorical columns
    encoders = {}
    for col in df.select_dtypes(include='object').columns:
        if col == y_col: continue
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Split dataset
    test_ratio = args.test_size / 100
    X = df[X_cols]
    y = df[y_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)

    # Initialize model
    if args.algo == "RandomForest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif args.algo == "DecisionTree":
        model = DecisionTreeClassifier(random_state=42)
    elif args.algo == "GradientBoosting":
        model = GradientBoostingClassifier(random_state=42)

    # Train model
    model.fit(X_train, y_train)
    print("[INFO] Training completed.")

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[RESULT] Test Accuracy: {acc*100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    save_model(model, encoders, X_cols)

if __name__ == "__main__":
    main()
