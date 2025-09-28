# ai_model_streamlit_complete_fixed.py
import os
import json
import queue
import threading
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import altair as alt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Scapy imports
try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP, Raw
except Exception:
    sniff = None
    IP = TCP = UDP = ICMP = Raw = None

# -------------------- Paths --------------------
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
ENCODERS_PATH = os.path.join(MODELS_DIR, "encoders.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "features.json")

# -------------------- Helper Functions --------------------
def save_model(model, encoders, features_list):
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    with open(FEATURES_PATH, 'w') as f:
        json.dump(features_list, f)

def load_model_artifacts():
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH) if os.path.exists(ENCODERS_PATH) else None
    features_list = None
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH,'r') as f:
            features_list = json.load(f)
    return model, encoders, features_list

def apply_label_encoders(df, encoders):
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
    df = apply_label_encoders(raw_df.copy(), encoders)
    for c in df.columns:
        try: df[c] = pd.to_numeric(df[c])
        except: pass
    if features_list is not None:
        df = df.reindex(columns=features_list, fill_value=0)
    return df

def predict_on_csv(model, encoders, features_list, csv_df):
    df_input = csv_df.drop(columns=['class'], errors='ignore')
    df_prepared = prepare_input_df(df_input, encoders, features_list)
    preds = model.predict(df_prepared)
    df_results = csv_df.copy()
    df_results['prediction'] = preds
    return df_results

def plot_feature_importance(model, X_cols):
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        df_imp = pd.DataFrame({'Feature': X_cols, 'Importance': imp}).sort_values(by='Importance', ascending=False)
        fig, ax = plt.subplots(figsize=(8,5))
        ax.barh(df_imp['Feature'], df_imp['Importance'], color='skyblue')
        ax.invert_yaxis()
        ax.set_title("Feature Importance")
        st.pyplot(fig)

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="AI Trainer & Network IDS", layout="wide")
st.title("🤖 Advanced AI Model Trainer & Live Network IDS")

tab_train, tab_test = st.tabs(["🛠 Model Training", "🔎 Model Testing & Live IDS"])

# ==================== Tab 1: Training ====================
with tab_train:
    st.subheader("1️⃣ Upload CSV for Training")
    train_file = st.file_uploader("Upload CSV File", type=["csv"])
    if train_file:
        df = pd.read_csv(train_file)
        st.write("Preview of dataset (first 10 rows):")
        st.dataframe(df.head(10))
        y_col = st.selectbox("Select Target Column", options=df.columns.tolist())
        X_cols = [c for c in df.columns if c != y_col]

        algo_choice = st.selectbox("Select Algorithm", ["RandomForest", "DecisionTree", "GradientBoosting"])
        test_size = st.slider("Test Size (%)", min_value=10, max_value=50, value=20, step=5)
        test_ratio = test_size/100

        if st.button("🚀 Start Training"):
            # Encode categorical
            encoders = {}
            for col in df.select_dtypes(include='object').columns:
                if col == y_col: continue
                le = LabelEncoder()
                df[col] = df[col].astype(str)
                df[col] = le.fit_transform(df[col])
                encoders[col] = le

            # Split
            X = df[X_cols]
            y = df[y_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)

            # Train model
            if algo_choice == "RandomForest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif algo_choice == "DecisionTree":
                model = DecisionTreeClassifier(random_state=42)
            elif algo_choice == "GradientBoosting":
                model = GradientBoostingClassifier(random_state=42)

            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.success(f"✅ Training completed. Test Accuracy: {acc*100:.2f}%")

            st.subheader("Classification Report")
            st.text(classification_report(y_test, y_pred))

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6,5))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax, cmap='Blues', colorbar=True)
            st.pyplot(fig)

            # Feature importance
            plot_feature_importance(model, X_cols)

            # Save model
            save_model(model, encoders, X_cols)
            st.info(f"Model, encoders, and features saved automatically in '{MODELS_DIR}'")

# ==================== Tab 2: Testing / Live IDS ====================
with tab_test:
    st.subheader("2️⃣ Model Testing / Live IDS")
    model, encoders, features_list = load_model_artifacts()
    if model is None:
        st.error("⚠️ No trained model found. Train model first.")
    else:
        st.success("✅ Model loaded successfully.")
        test_option = st.radio("Choose Testing Method", ["Upload CSV", "Live Network Capture"])

        if test_option == "Upload CSV":
            test_file = st.file_uploader("Upload CSV to Test", type=["csv"], key="test_csv")
            if test_file:
                test_df = pd.read_csv(test_file)
                if st.button("▶️ Predict CSV"):
                    results_df = predict_on_csv(model, encoders, features_list, test_df)
                    st.dataframe(results_df.head(200))
                    st.bar_chart(results_df['prediction'].value_counts())
                    csv_bytes = results_df.to_csv(index=False).encode()
                    st.download_button("⬇️ Download Results CSV", csv_bytes, file_name="predictions.csv")

        elif test_option == "Live Network Capture":
            import psutil
            iface_list = ["<default>"] + list(psutil.net_if_addrs().keys()) if hasattr(psutil,'net_if_addrs') else ["<default>"]
            iface_choice = st.selectbox("Choose interface", iface_list)
            chosen_iface = None if iface_choice=="<default>" else iface_choice
            filter_expr = st.text_input("BPF filter", value="ip").strip() or "ip"

            # Session state defaults
            if 'sniffer_thread' not in st.session_state: st.session_state['sniffer_thread']=None
            if 'sniffer_queue' not in st.session_state: st.session_state['sniffer_queue']=queue.Queue()
            if 'sniffer_stop' not in st.session_state: st.session_state['sniffer_stop']=threading.Event()
            if 'live_summary' not in st.session_state: st.session_state['live_summary']={"total":0,"alerts":0}
            if 'live_history' not in st.session_state: st.session_state['live_history']=[]

            class SnifferThread(threading.Thread):
                def __init__(self,q,iface=None,filter_expr="ip",stop_event=None):
                    super().__init__(daemon=True)
                    self.q=q
                    self.iface=iface
                    self.filter_expr=filter_expr
                    self.stop_event=stop_event or threading.Event()
                def run(self):
                    if sniff is None:
                        self.q.put(("error","Scapy not installed"))
                        return
                    def prn(pkt):
                        if self.stop_event.is_set(): return
                        self.q.put(("packet",pkt))
                    try:
                        sniff(filter=self.filter_expr, iface=self.iface, prn=prn, store=False)
                    except Exception as e:
                        self.q.put(("error",f"sniff error: {e}"))

            col1,col2 = st.columns([1,2])
            with col1:
                sniffer_thread = st.session_state.get('sniffer_thread')
                if sniffer_thread is None or not sniffer_thread.is_alive():
                    if st.button("▶️ Start Sniffing"):
                        # Reset queue and stop event
                        st.session_state['sniffer_queue'] = queue.Queue()
                        st.session_state['sniffer_stop'] = threading.Event()
                        th = SnifferThread(
                            q=st.session_state['sniffer_queue'],
                            iface=chosen_iface,
                            filter_expr=filter_expr,
                            stop_event=st.session_state['sniffer_stop']
                        )
                        st.session_state['sniffer_thread'] = th
                        th.start()

                        # Initialize LiveIDS
                        class LiveIDS:
                            def __init__(self, model, encoders, features_list):
                                self.model = model
                                self.encoders = encoders
                                self.features_list = features_list
                                self.queue = st.session_state['sniffer_queue']
                                self.stop_event = st.session_state['sniffer_stop']
                                self.prev_pkt_time = {}
                                self.history = st.session_state['live_history']
                                self.summary = st.session_state['live_summary']
                                self.stats = {}
                                self.window_time = 1.0

                            def extract_features(self, pkt):
                                f = {feat:0 for feat in self.features_list}
                                if IP not in pkt: return f
                                src, dst = pkt[IP].src, pkt[IP].dst
                                proto_map = {1:'ICMP', 6:'TCP', 17:'UDP'}
                                proto = proto_map.get(pkt[IP].proto, 'other')
                                f['protocol_type'] = proto
                                now = datetime.now()
                                f['duration'] = (now - self.prev_pkt_time.get(src, now)).total_seconds()
                                self.prev_pkt_time[src] = now

                                if TCP in pkt:
                                    tcp = pkt[TCP]
                                    f['service'] = tcp.dport
                                    f['flag'] = str(tcp.flags)
                                    f['src_bytes'] = len(pkt[Raw].load) if Raw in pkt else 0
                                    f['dst_bytes'] = 0
                                    f['urgent'] = tcp.urgptr
                                    f['land'] = int(src==dst and tcp.sport==tcp.dport)
                                elif UDP in pkt:
                                    udp = pkt[UDP]
                                    f['service'] = udp.dport
                                    f['flag'] = 'OTH'
                                    f['src_bytes'] = len(pkt[Raw].load) if Raw in pkt else 0
                                elif ICMP in pkt:
                                    f['service'] = 'eco_i'
                                    f['flag'] = 'OTH'
                                    f['src_bytes'] = len(pkt[Raw].load) if Raw in pkt else 0

                                self.stats.setdefault(src, []).append(now)
                                recent = [t for t in self.stats[src] if (now - t).total_seconds() <= self.window_time]
                                f['count'] = len(recent)
                                f['srv_count'] = len(recent)
                                self.stats[src] = recent
                                return f

                            def process_packet(self, pkt):
                                feats = self.extract_features(pkt)
                                df_in = pd.DataFrame([feats])
                                df_prep = prepare_input_df(df_in, self.encoders, self.features_list)
                                pred = self.model.predict(df_prep)[0]
                                src = pkt[IP].src if IP in pkt else 'N/A'
                                dst = pkt[IP].dst if IP in pkt else 'N/A'
                                self.summary['total'] += 1
                                if str(pred).lower() in ["attack","anomaly","malicious","1","true"]:
                                    self.summary['alerts'] += 1
                                event = {"time":datetime.now().strftime("%H:%M:%S"), "src":src, "dst":dst, "pred":str(pred)}
                                self.history.append(event)
                                self.history = self.history[-500:]
                                return event

                            def process_queue(self, max_items=50):
                                processed=[]
                                for _ in range(max_items):
                                    try:
                                        item=self.queue.get_nowait()
                                    except:
                                        break
                                    if not item: continue
                                    tag,payload = item
                                    if tag=="error":
                                        processed.append({"time":datetime.now().strftime("%H:%M:%S"),"src":"N/A","dst":"N/A","pred":"ERROR"})
                                    elif tag=="packet":
                                        try:
                                            event = self.process_packet(payload)
                                            processed.append(event)
                                        except:
                                            processed.append({"time":datetime.now().strftime("%H:%M:%S"),"src":"N/A","dst":"N/A","pred":"ERROR"})
                                return processed

                        st.session_state['live_ids'] = LiveIDS(model, encoders, features_list)
                        st.success(f"Started sniffing on iface: {chosen_iface or 'default'}")
                else:
                    if st.button("⏸️ Stop Sniffing"):
                        st.session_state['sniffer_stop'].set()
                        st.session_state['sniffer_thread'] = None
                        st.success("Sniffing stopped successfully")

            # Display Live IDS updates
            recent_area = st.empty()
            table_area = st.empty()
            chart_area = st.empty()

            live_ids = st.session_state.get('live_ids')
            if live_ids:
                new_events = live_ids.process_queue()
                # Metrics
                col1,col2,col3 = st.columns(3)
                col1.metric("Total Packets", live_ids.summary['total'])
                col2.metric("Alerts", live_ids.summary['alerts'])
                total = max(live_ids.summary['total'],1)
                col3.metric("Alert Rate", f"{live_ids.summary['alerts']/total*100:.2f}%")

                # Table display
                if live_ids.history:
                    df_hist = pd.DataFrame(live_ids.history)
                    def highlight_alert(val):
                        color = 'background-color: tomato' if str(val).lower() in ["attack","anomaly","malicious","1","true"] else ''
                        return color
                    table_area.dataframe(df_hist.tail(50).style.map(highlight_alert, subset=['pred']))
                else:
                    table_area.info("No packets captured yet.")

st.caption("Live capture may require admin/root privileges.")
