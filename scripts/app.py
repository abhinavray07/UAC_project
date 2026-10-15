import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="UAC Care Load Forecasting", layout="wide")
st.title("Predictive Forecasting: Care Load & Placement Demand")

@st.cache_data
def load_data():
    hist = pd.read_csv('../data/hhs_filled.csv', index_col='Date', parse_dates=True)
    care_fc = pd.read_csv('../outputs/care_load_forecast.csv', index_col=0, parse_dates=True)
    discharge_fc = pd.read_csv('../outputs/discharge_forecast.csv', index_col=0, parse_dates=True)
    scoreboard = pd.read_csv('../outputs/combined_scoreboard.csv')
    kpis = pd.read_csv('../outputs/kpi_summary.csv')
    return hist, care_fc, discharge_fc, scoreboard, kpis

hist, care_fc, discharge_fc, scoreboard, kpis = load_data()

# --- Controls ---
st.sidebar.header("Controls")
horizon = st.sidebar.selectbox("Forecast horizon", [1, 7, 14], index=2)
model_choice = st.sidebar.selectbox("Model", scoreboard['model'].unique())
scenario = st.sidebar.radio("Scenario", ["Baseline", "Capacity stress (upper bound)"])

target_col = 'upper' if scenario == "Capacity stress (upper bound)" else 'mean'
care_slice = care_fc.iloc[:horizon]
discharge_slice = discharge_fc.iloc[:horizon]

# --- Module 1: Care Load Forecast ---
st.subheader("Future Care Load Forecast")
st.caption("Children in HHS Care, actual history plus forecast with 90% uncertainty band.")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(hist.index[-30:], hist['Children in HHS Care'].iloc[-30:], label='History')
ax.plot(care_slice.index, care_slice[target_col], label='Forecast', color='C1')
ax.fill_between(care_slice.index, care_slice['lower'], care_slice['upper'], alpha=0.2, color='C1')
ax.legend()
st.pyplot(fig)

# --- Module 2: Discharge Demand Forecast Panel ---
st.subheader("Discharge Demand Forecast")
st.caption("Children discharged from HHS Care, forecast with 90% uncertainty band.")

fig_d, ax_d = plt.subplots(figsize=(12, 4))
ax_d.plot(hist.index[-30:], hist['Children discharged from HHS Care'].iloc[-30:], label='History')
ax_d.plot(discharge_slice.index, discharge_slice[target_col], label='Forecast', color='C2')
ax_d.fill_between(discharge_slice.index, discharge_slice['lower'], discharge_slice['upper'], alpha=0.2, color='C2')
ax_d.legend()
st.pyplot(fig_d)

# --- Module 3: Model Selection & Comparison ---
st.subheader("Model Selection & Comparison")
st.caption(f"Accuracy at the {horizon}-day horizon. Selected model: {model_choice}.")

horizon_scores = scoreboard[scoreboard['horizon'] == horizon].sort_values('MAE').reset_index(drop=True)
st.dataframe(
    horizon_scores.style.apply(
        lambda row: ['background-color: #fff3cd' if row['model'] == model_choice else '' for _ in row],
        axis=1
    )
)

# --- Module 4: Confidence Interval Visualization + KPIs ---
st.subheader("Key Performance Indicators")
st.dataframe(kpis)

st.subheader("Uncertainty Band Detail")
st.caption(f"Prediction interval width over the selected {horizon}-day horizon.")

fig2, ax2 = plt.subplots(figsize=(12, 3))
ax2.fill_between(care_slice.index, care_slice['lower'], care_slice['upper'], alpha=0.3)
ax2.plot(care_slice.index, care_slice['mean'])
st.pyplot(fig2)