````markdown
# 🤖 AI Trainer & Live Network IDS

**Advanced Machine Learning Model Trainer & Real-Time Network Intrusion Detection System (IDS) using Python and Streamlit**

---

## 📝 Project Overview

This project is a complete **AI-powered network monitoring and intrusion detection system** built with **Python**.  
It includes both **CLI-based scripts** and a **Streamlit interactive dashboard** to:

- Train **machine learning models** (RandomForest, DecisionTree, GradientBoosting) on CSV network datasets.
- Test trained models on new datasets.
- Run a **live Network IDS** to monitor and classify packets in real-time.
- Visualize **feature importance, confusion matrix, and model performance** metrics.

**Keywords:** AI, Machine Learning, Network IDS, Streamlit, Python, Cybersecurity, Real-Time Packet Capture, Anomaly Detection, Data Science

---

## 🚀 Features

- **CLI scripts for training (`train_model.py`) and testing (`test_model.py`)**
- **Streamlit dashboard (`ai_model_streamlit_complete_fixed.py`)** for interactive training/testing
- **Live network packet capture and anomaly detection**
- **Classification report and confusion matrix**
- **Feature importance visualization**
- **Real-time alerts for suspicious network activity**

---

## 🖼 Screenshots

**Streamlit Training Tab:**  
![Training Tab](docs/screenshots/training_tab_dashboard.png)

**Streamlit Testing CSV:**  
![Testing Tab](docs/screenshots/testing_csv.png)

**Streamlit Live Network IDS:**  
![Live IDS](docs/screenshots/live_ids_alerts.png)

*Replace placeholders with actual screenshots after running the app.*

---

## 🛠 Installation

Clone the repository:

```bash
git clone https://github.com/0xAnonymousYE/ai_network_ids.git
cd ai_network_ids
````

Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📂 Usage

### 1️⃣ Streamlit Interface

Run the Streamlit app:

```bash
streamlit run ai_model_streamlit_complete_fixed.py
```

**Tabs Overview:**

* **Training Tab:** Train models on your dataset, visualize performance, and save trained artifacts.
* **Testing Tab:** Test models on new CSV data or capture live network packets to detect anomalies.
* **Live IDS:** Monitor traffic in real-time and generate alerts for suspicious activity.

### 2️⃣ CLI Scripts

#### a) Training

Train a model directly from CSV using `train_model.py`:

```bash
python train_model.py --file data/sample_train.csv --target class --algo RandomForest --test-size 20
```

* `--file` : path to training CSV
* `--target` : target column name
* `--algo` : algorithm (`RandomForest`, `DecisionTree`, `GradientBoosting`)
* `--test-size` : test split percentage

> Saves trained model, encoders, and feature list in `models/` directory.

#### b) Testing

Test a CSV dataset using `test_model.py`:

```bash
python test_model.py --file data/sample_test.csv
```

* Produces `predictions.csv` with predicted classes
* Uses the trained model saved in `models/`

---

## 📊 Sample Data

* `data/sample_train.csv` – example training dataset
* `data/sample_test.csv` – example testing dataset

---

## 🧩 Requirements

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

Install all requirements:

```bash
pip install -r requirements.txt
```

---

## 📌 License

MIT License © 2025 Your Name

---

## 🌐 SEO Tips for This Project

* Topics on GitHub: `AI, Machine-Learning, Network-IDS, CLI, Streamlit, Python, Cybersecurity, Data-Science`
* Include **screenshots and usage examples** in `docs/screenshots/`
* Maintain **updated commits and documentation** to improve visibility
* Share project links in **social media, forums, and developer communities**

```

