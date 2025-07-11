# qce_app.py
import streamlit as st
from datetime import datetime
import random

# -----------------------
# User Management System
# -----------------------
class User:
    def __init__(self, name, token):
        self.name = name
        self.token = token
        self.history = []

    def add_history(self, result):
        self.history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "result": result
        })

# -----------------------
# Session Logging System
# -----------------------
class SessionLog:
    def __init__(self):
        self.logs = []

    def record(self, user, result):
        log_entry = {
            "user": user.name,
            "timestamp": datetime.utcnow().isoformat(),
            "result": result
        }
        self.logs.append(log_entry)
        user.add_history(result)
        return log_entry

# -----------------------
# Energy Threshold Configuration
# -----------------------
class SystemConfig:
    def __init__(self):
        self.discordance_threshold = 0.7
        self.intent_threshold = 0.88
        self.expected_harmonics = ["alpha", "beta"]

# -----------------------
# Consent and Access Evaluation System
# -----------------------
class QuantumConsentEngine:
    def __init__(self, config):
        self.config = config

    def evaluate_consent(self, intent_score, discordance, harmonics):
        passed = (
            intent_score >= self.config.intent_threshold and
            discordance <= self.config.discordance_threshold and
            harmonics in self.config.expected_harmonics
        )
        return passed

# -----------------------
# Streamlit Interface
# -----------------------
st.set_page_config(page_title="Quantum Consent Engine", layout="centered")
st.title("🔮 Quantum Consent Engine (QCE)")
st.markdown("Consent verification system based on energy and awareness")

# System Instances
session_log = SessionLog()
system_config = SystemConfig()
qce = QuantumConsentEngine(system_config)

# User Input Section
st.subheader("🧑‍💼 User Registration")
name = st.text_input("User Name")
token = st.text_input("Awareness Token")

# Energy Analysis Input Section
st.subheader("🌀 Energy During Consent Request")
intent_score = st.slider("Intent Score", 0.0, 1.0, 0.5)
discordance = st.slider("Discordance", 0.0, 1.0, 0.5)
harmonics = st.selectbox("Energy Wave (Harmonics)", ["alpha", "beta", "theta", "gamma", "delta"])

# Submit Button
if st.button("📥 Evaluate and Record"):
    if name and token:
        user = User(name, token)
        result = {
            "intent_score": intent_score,
            "discordance": discordance,
            "harmonics": harmonics
        }

        log_entry = session_log.record(user, result)
        passed = qce.evaluate_consent(intent_score, discordance, harmonics)

        if passed:
            st.success("✅ Energy threshold passed — Consent granted")
        else:
            st.warning("⚠️ Threshold not passed — Please reconsider your mental field")

        st.json(log_entry)
    else:
        st.error("Please enter both name and token")

# Display Recent Logs
st.subheader("📜 Record History")
if session_log.logs:
    for entry in reversed(session_log.logs[-5:]):
        st.markdown(f"**{entry['user']}** @ `{entry['timestamp']}`")
        st.json(entry['result'])
else:
    st.write("No data has been recorded yet")
