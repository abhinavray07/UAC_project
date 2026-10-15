import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

st.set_page_config(page_title="UAC Care Load Forecasting", layout="wide")
st.title("Predictive Forecasting: Care Load & Placement Demand")


@st.cache_data
def load_data():
    required_files = {
        "Historical data": DATA_DIR / "hhs_filled.csv",
        "Care-load forecast": OUTPUT_DIR / "care_load_forecast.csv",
        "Discharge forecast": OUTPUT_DIR / "discharge_forecast.csv",
        "Model scoreboard": OUTPUT_DIR / "combined_scoreboard.csv",
        "KPI summary": OUTPUT_DIR / "kpi_summary.csv",
    }

    # Give a clear error if a file is missing from the repository.
    missing_files = [
        f"{name}: {path}"
        for name, path in required_files.items()
        if not path.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "The following required files were not found:\n"
            + "\n".join(missing_files)
            + "\nCheck that these files exist in the repository and are committed to GitHub."
        )

    hist = pd.read_csv(
        required_files["Historical data"],
        index_col="Date",
        parse_dates=True,
    )
    care_fc = pd.read_csv(
        required_files["Care-load forecast"],
        index_col=0,
        parse_dates=True,
    )
    discharge_fc = pd.read_csv(
        required_files["Discharge forecast"],
        index_col=0,
        parse_dates=True,
    )
    scoreboard = pd.read_csv(required_files["Model scoreboard"])
    kpis = pd.read_csv(required_files["KPI summary"])

    return hist, care_fc, discharge_fc, scoreboard, kpis


hist, care_fc, discharge_fc, scoreboard, kpis = load_data()

# --- Validate expected columns ---
required_hist_cols = [
    "Children in HHS Care",
    "Children discharged from HHS Care",
]
required_forecast_cols = ["mean", "lower", "upper"]
required_scoreboard_cols = ["model", "horizon", "MAE"]

missing_hist_cols = [c for c in required_hist_cols if c not in hist.columns]
if missing_hist_cols:
    st.error(f"Historical data is missing columns: {missing_hist_cols}")
    st.stop()

for name, forecast in [
    ("Care-load forecast", care_fc),
    ("Discharge forecast", discharge_fc),
]:
    missing = [c for c in required_forecast_cols if c not in forecast.columns]
    if missing:
        st.error(f"{name} is missing columns: {missing}")
        st.stop()

missing_scoreboard_cols = [
    c for c in required_scoreboard_cols if c not in scoreboard.columns
]
if missing_scoreboard_cols:
    st.error(f"Model scoreboard is missing columns: {missing_scoreboard_cols}")
    st.stop()

if scoreboard.empty:
    st.error("The model scoreboard is empty. Check combined_scoreboard.csv.")
    st.stop()

if care_fc.empty or discharge_fc.empty:
    st.error("One or more forecast files are empty.")
    st.stop()

# --- Controls ---
st.sidebar.header("Controls")

available_horizons = [h for h in [1, 7, 14] if h <= min(len(care_fc), len(discharge_fc))]
if not available_horizons:
    st.error("The forecast files do not contain any forecast rows.")
    st.stop()

horizon = st.sidebar.selectbox(
    "Forecast horizon",
    available_horizons,
    index=len(available_horizons) - 1,
)

models = scoreboard["model"].dropna().unique().tolist()
if not models:
    st.error("No models are available in the model scoreboard.")
    st.stop()

model_choice = st.sidebar.selectbox("Model", models)
scenario = st.sidebar.radio(
    "Scenario",
    ["Baseline", "Capacity stress (upper bound)"],
)

target_col = "upper" if scenario == "Capacity stress (upper bound)" else "mean"

care_slice = care_fc.iloc[:horizon]
discharge_slice = discharge_fc.iloc[:horizon]

# --- Module 1: Care Load Forecast ---
st.subheader("Future Care Load Forecast")
st.caption(
    "Children in HHS Care: historical values and forecast with a 90% uncertainty band."
)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(
    hist.index[-30:],
    hist["Children in HHS Care"].iloc[-30:],
    label="History",
)
ax.plot(
    care_slice.index,
    care_slice[target_col],
    label="Forecast",
    color="C1",
)
ax.fill_between(
    care_slice.index,
    care_slice["lower"],
    care_slice["upper"],
    alpha=0.2,
    color="C1",
)
ax.set_xlabel("Date")
ax.set_ylabel("Children")
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

# --- Module 2: Discharge Demand Forecast ---
st.subheader("Discharge Demand Forecast")
st.caption(
    "Children discharged from HHS Care: forecast with a 90% uncertainty band."
)

fig_d, ax_d = plt.subplots(figsize=(12, 4))
ax_d.plot(
    hist.index[-30:],
    hist["Children discharged from HHS Care"].iloc[-30:],
    label="History",
)
ax_d.plot(
    discharge_slice.index,
    discharge_slice[target_col],
    label="Forecast",
    color="C2",
)
ax_d.fill_between(
    discharge_slice.index,
    discharge_slice["lower"],
    discharge_slice["upper"],
    alpha=0.2,
    color="C2",
)
ax_d.set_xlabel("Date")
ax_d.set_ylabel("Children discharged")
ax_d.legend()
fig_d.tight_layout()
st.pyplot(fig_d)
plt.close(fig_d)

# --- Module 3: Model Selection & Comparison ---
st.subheader("Model Selection & Comparison")
st.caption(
    f"Accuracy at the {horizon}-day horizon. "
    f"Selected model: {model_choice}."
)

horizon_scores = (
    scoreboard[scoreboard["horizon"] == horizon]
    .sort_values("MAE")
    .reset_index(drop=True)
)

if horizon_scores.empty:
    st.warning(
        f"No model evaluation results are available for the {horizon}-day horizon."
    )
else:
    st.dataframe(
        horizon_scores.style.apply(
            lambda row: [
                "background-color: #fff3cd"
                if row["model"] == model_choice
                else ""
                for _ in row
            ],
            axis=1,
        ),
        use_container_width=True,
    )

# --- Module 4: KPIs ---
st.subheader("Key Performance Indicators")
st.dataframe(kpis, use_container_width=True)

# --- Module 5: Uncertainty Band ---
st.subheader("Uncertainty Band Detail")
st.caption(f"Prediction interval width over the selected {horizon}-day horizon.")

fig2, ax2 = plt.subplots(figsize=(12, 3))
ax2.fill_between(
    care_slice.index,
    care_slice["lower"],
    care_slice["upper"],
    alpha=0.3,
    label="90% prediction interval",
)
ax2.plot(
    care_slice.index,
    care_slice["mean"],
    label="Forecast mean",
)
ax2.set_xlabel("Date")
ax2.set_ylabel("Children in care")
ax2.legend()
fig2.tight_layout()
st.pyplot(fig2)
plt.close(fig2)