import os
import streamlit as st
import pandas as pd
import mysql.connector
from mysql.connector import Error
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import json
import re

# ── LOAD INTENTS ─────────────────────────────────────────────────────────
with open("intents.json", "r", encoding="utf-8") as f:
    INTENTS = json.load(f)["intents"]

# ── CONFIG & MODEL LOADING ──────────────────────────────────────────────────
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Sripriya@7385",
    "database": "echonova_feedback_db"
}
model = joblib.load("customer_satisfaction_linear_model.pkl")
scaler = joblib.load("scaler.pkl")

# ── SESSION STATE DEFAULTS ────────────────────────────────────────────────
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

# ── DB CONNECTION ─────────────────────────────────────────────────────────
def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

# ── PREPROCESS & SENTIMENT ────────────────────────────────────────────────
def preprocess_input(data):
    df = pd.DataFrame([data])
    df["Gender"] = LabelEncoder().fit_transform(df["Gender"])
    df["Region"] = LabelEncoder().fit_transform(df["Region"])
    return scaler.transform(df)

def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    if polarity < 0:
        return "Negative"
    return "Neutral"

# ── CHATBOT RESPONSE USING JSON INTENTS ────────────────────────────────────
def get_chatbot_response(user_input):
    text = user_input.lower().strip()
    for intent in INTENTS:
        for kw in intent.get("keywords", []):
            if re.search(rf"\b{re.escape(kw)}\b", text):
                return intent["response"]
    sentiment = analyze_sentiment(user_input)
    if sentiment == "Positive":
        return next(i["response"] for i in INTENTS if i.get("tag") == "fallback_positive")
    if sentiment == "Negative":
        return next(i["response"] for i in INTENTS if i.get("tag") == "fallback_negative")
    return next(i["response"] for i in INTENTS if i.get("tag") == "fallback_default")

# ── PAGE CONFIG & GLOBAL STYLES ──────────────────────────────────────────
st.set_page_config(page_title="EchoNova Feedback Predictor", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] > .main {
        background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), url('bg.png');
        background-size: cover;
        background-position: center;
    }
    </style>
    """, unsafe_allow_html=True
)

# ── RENDER HEADER fn ─────────────────────────────────────────────────────
def render_header(title: str, show_tagline: bool = True):
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("logo.png", width=100)
    with col2:
        st.markdown(f"## {title}")
        if show_tagline:
            st.markdown(
                "<small style='color:lightgray;'>Voices Heard. Experiences Transformed.</small>",
                unsafe_allow_html=True
            )
    st.markdown("---")

# ── SIDEBAR NAVIGATION ───────────────────────────────────────────────────
page = st.sidebar.selectbox("Navigate", ["Home", "Chatbot", "Admin"])

# ── HOME PAGE ─────────────────────────────────────────────────────────────
if page == "Home":
    render_header("EchoNova Customer Feedback Portal")
    st.subheader("Submit Your Feedback")
    with st.form("feedback_form"):
        age = st.number_input("Age", 10, 100, 10)
        gender = st.selectbox("Gender", ["Male", "Female"])
        region = st.selectbox("Region", ["Andhra Pradesh","Telangana","Tamil Nadu","Karnataka","Kerala"])
        net = st.slider("Network Coverage (1–10)", 1, 10, 0)
        speed = st.slider("Internet Speed (1–10)", 1, 10, 0)
        support = st.slider("Customer Support (1–10)", 1, 10, 0)
        app_exp = st.slider("App Experience (1–10)", 1, 10, 0)
        plan = st.slider("Plan Satisfaction (1–10)", 1, 10, 0)
        call_q = st.slider("Call Quality (1–10)", 1, 10, 0)
        duration = st.number_input("Years with EchoNova", 0.0, 50.0, 0.0)
        feedback = st.text_area("Additional Feedback", height=100)
        submitted = st.form_submit_button("Submit")
    if submitted:
        inp = {
            "Age": age, "Gender": gender, "Region": region,
            "NetworkCoverage": net, "InternetSpeed": speed,
            "CustomerSupport": support, "AppExperience": app_exp,
            "PlanSatisfaction": plan, "CallQuality": call_q,
            "Duration": duration
        }
        X_scaled = preprocess_input(inp)
        score = float(model.predict(X_scaled)[0])
        sentiment = analyze_sentiment(feedback)
        churn = "High" if score < 50 else "Low"
        try:
            conn = get_db_connection(); cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO feedback_entries
                  (Age,Gender,Region,NetworkCoverage,InternetSpeed,
                   CustomerSupport,AppExperience,PlanSatisfaction,
                   CallQuality,Duration,FeedbackText,
                   PredictedScore,Sentiment,ChurnRisk)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (age, gender, region, net, speed, support,
                 app_exp, plan, call_q, duration,
                 feedback, score, sentiment, churn)
            )
            conn.commit(); cur.close(); conn.close()
            st.success("Feedback submitted successfully!")
        except Error as e:
            st.error(f"Database error: {e}")

# ── ADMIN PAGE ────────────────────────────────────────────────────────────
elif page == "Admin":
    render_header("Admin Dashboard", show_tagline=False)
    if not st.session_state.admin_authenticated:
        user = st.text_input("Username", key="adm_user")
        pw = st.text_input("Password", type="password", key="adm_pwd")
        if st.button("Login", key="adm_login"):
            if user == "admin" and pw == "admin123":
                st.session_state.admin_authenticated = True
                st.success("Logged in as Admin")
            else:
                st.error("Invalid credentials")
    else:
        try:
            conn = get_db_connection()
            df_latest = pd.read_sql("SELECT * FROM feedback_entries ORDER BY submission_time DESC LIMIT 10", conn)
            df_all = pd.read_sql("SELECT PredictedScore,Sentiment,ChurnRisk FROM feedback_entries", conn)
            conn.close()
            st.subheader("Latest Feedback Entries"); st.dataframe(df_latest)
            st.subheader("Satisfaction Score Over Time"); st.line_chart(df_all['PredictedScore'])
            st.subheader("Sentiment Distribution"); st.bar_chart(df_all['Sentiment'].value_counts())
            st.subheader("Churn Risk Distribution"); st.bar_chart(df_all['ChurnRisk'].value_counts())
        except Error as e:
            st.error(f"Database error: {e}")

# ── CHATBOT PAGE ─────────────────────────────────────────────────────────
elif page == "Chatbot":
    render_header("EchoNova Chatbot Assistant", show_tagline=False)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    for sender, msg in st.session_state.chat_history:
        st.chat_message('user' if sender=='You' else 'assistant').write(msg)
    user_msg = st.chat_input("Type your message...")
    if user_msg:
        st.chat_message('user').write(user_msg)
        resp = get_chatbot_response(user_msg)
        st.chat_message('assistant').write(resp)
        st.session_state.chat_history.append(("You", user_msg))
        st.session_state.chat_history.append(("Bot", resp))
        try:
            conn = get_db_connection(); cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO feedback_entries
                  (Age,Gender,Region,NetworkCoverage,InternetSpeed,
                   CustomerSupport,AppExperience,PlanSatisfaction,
                   CallQuality,Duration,FeedbackText,
                   PredictedScore,Sentiment,ChurnRisk)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (0,'Male','Unknown',0,0,0,0,0,0,0.0,
                 user_msg,
                 60 if analyze_sentiment(user_msg)=='Positive' else 40 if analyze_sentiment(user_msg)=='Negative' else 50,
                 analyze_sentiment(user_msg), 'Low')
            )
            conn.commit(); cur.close(); conn.close()
        except Error as e:
            st.error(f"Database error: {e}")
