````markdown
# 🤖 AI Trainer & Live Network IDS - Usage Guide

This document explains how to use all components of the project.

---

## 1️⃣ Streamlit App: `ai_model_streamlit_complete_fixed.py`

Launch the interactive dashboard:

```bash
streamlit run ai_model_streamlit_complete_fixed.py
````

### Tabs Overview:

**Training Tab**

* Upload a CSV dataset.
* Select the target column.
* Choose ML algorithm: RandomForest, DecisionTree, GradientBoosting.
* Adjust test size (%).
* Click "Start Training".
* View classification report, confusion matrix, feature importance.
* Model, encoders, and features are saved automatically in `models/`.

**Testing Tab**

* **Option 1: Upload CSV**

  * Upload a CSV file to predict.
  * Click "Predict CSV" to see results.
  * Download predictions as CSV.
* **Option 2: Live Network Capture**

  * Select network interface (requires admin/root privileges).
  * Apply BPF filter if needed.
  * Start sniffing to capture live packets.
  * Alerts are generated for suspicious activity.
  * Metrics displayed: Total Packets, Alerts, Alert Rate.
  * Table shows last captured packets with prediction labels.

---

## 2️⃣ Command-Line Interface (CLI) Training: `train_model_cli.py`

Train a model directly from the command line.

**Usage Example:**

```bash
python train_model_cli.py --input data/sample_train.csv --target class --algorithm RandomForest --test-size 0.2
```

**Arguments:**

* `--input`: Path to CSV training file
* `--target`: Target column name
* `--algorithm`: RandomForest, DecisionTree, or GradientBoosting
* `--test-size`: Fraction of data for testing (default 0.2)

**Output:**

* `models/model.pkl` → Trained model
* `models/encoders.pkl` → Encoders for categorical columns
* `models/features.json` → List of features used

**Notes:**

* Automatically handles categorical encoding.
* Displays training accuracy, classification report, and confusion matrix.

---

## 3️⃣ Command-Line Interface (CLI) Testing: `test_model_cli.py`

Test a trained model on new CSV data.

**Usage Example:**

```bash
python test_model_cli.py --input data/sample_test.csv --output predictions.csv
```

**Arguments:**

* `--input`: Path to CSV file to predict
* `--output`: Path to save predictions CSV

**Output:**

* CSV file with a new column `prediction` containing predicted labels.

**Notes:**

* Ensure the model is trained before testing.
* Automatically applies encoders and aligns features to match training.

---

## 4️⃣ Sample Data

* `data/sample_train.csv` → Example training dataset
* `data/sample_test.csv` → Example testing dataset

---

## 5️⃣ Requirements

* Python >= 3.9
* pandas
* numpy
* scikit-learn
* streamlit
* matplotlib
* altair
* scapy
* psutil
* joblib

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 6️⃣ Additional Notes

* CLI scripts are ideal for automated workflows or servers without GUI.
* Streamlit app provides interactive visualization and live packet monitoring.
* Live network capture may require admin/root privileges.
* All models, encoders, and features are stored in the `models/` folder.
* Ensure your CSV datasets have consistent columns with training data.

---

## 7️⃣ References

* Explore sample datasets in `data/`.
* Visualizations from Streamlit dashboard can be used in reports.
* For full project explanation and screenshots, see the main README.

```
