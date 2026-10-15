from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# --------------------------------------------------
# App configuration
# --------------------------------------------------
st.set_page_config(
    page_title="UAC Care Load Forecasting",
    page_icon="📊",
    layout="wide",
)

st.title("Predictive Forecasting: Care Load & Placement Demand")
st.caption(
    "Explore historical care-load trends, discharge forecasts, "
    "model accuracy, and forecast uncertainty."
)


# --------------------------------------------------
# Paths
# Resolve paths from this file rather than the
# current working directory.
# --------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

HISTORY_FILE = DATA_DIR / "hhs_filled.csv"
CARE_FORECAST_FILE = OUTPUT_DIR / "care_load_forecast.csv"
DISCHARGE_FORECAST_FILE = OUTPUT_DIR / "discharge_forecast.csv"
SCOREBOARD_FILE = OUTPUT_DIR / "combined_scoreboard.csv"
KPI_FILE = OUTPUT_DIR / "kpi_summary.csv"


# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_data():
    required_files = [
        HISTORY_FILE,
        CARE_FORECAST_FILE,
        DISCHARGE_FORECAST_FILE,
        SCOREBOARD_FILE,
        KPI_FILE,
    ]

    missing_files = [str(path.relative_to(PROJECT_ROOT))
                     for path in required_files if not path.is_file()]

    if missing_files:
        raise FileNotFoundError(
            "The following required files are missing from the repository:\n"
            + "\n".join(f"- {file}" for file in missing_files)
            + "\n\nCheck that these files are committed to GitHub "
              "and that their names and capitalization match."
        )

    hist = pd.read_csv(
        HISTORY_FILE,
        index_col="Date",
        parse_dates=True,
    )

    care_fc = pd.read_csv(
        CARE_FORECAST_FILE,
        index_col=0,
        parse_dates=True,
    )

    discharge_fc = pd.read_csv(
        DISCHARGE_FORECAST_FILE,
        index_col=0,
        parse_dates=True,
    )

    scoreboard = pd.read_csv(SCOREBOARD_FILE)
    kpis = pd.read_csv(KPI_FILE)

    # Validate the columns used by the dashboard.
    required_history_columns = [
        "Children in HHS Care",
        "Children discharged from HHS Care",
    ]
    required_forecast_columns = ["lower", "mean", "upper"]
    required_scoreboard_columns = ["model", "horizon", "MAE"]

    missing_history_columns = [
        col for col in required_history_columns if col not in hist.columns
    ]
    missing_care_columns = [
        col for col in required_forecast_columns if col not in care_fc.columns
    ]
    missing_discharge_columns = [
        col
        for col in required_forecast_columns
        if col not in discharge_fc.columns
    ]
    missing_scoreboard_columns = [
        col
        for col in required_scoreboard_columns
        if col not in scoreboard.columns
    ]

    if missing_history_columns:
        raise ValueError(
            f"Missing columns in hhs_filled.csv: {missing_history_columns}"
        )

    if missing_care_columns:
        raise ValueError(
            "Missing columns in care_load_forecast.csv: "
            f"{missing_care_columns}"
        )

    if missing_discharge_columns:
        raise ValueError(
            "Missing columns in discharge_forecast.csv: "
            f"{missing_discharge_columns}"
        )

    if missing_scoreboard_columns:
        raise ValueError(
            f"Missing columns in combined_scoreboard.csv: "
            f"{missing_scoreboard_columns}"
        )

    if hist.empty or care_fc.empty or discharge_fc.empty:
        raise ValueError(
            "One or more history or forecast datasets are empty."
        )

    if scoreboard.empty:
        raise ValueError("The model scoreboard contains no rows.")

    return hist, care_fc, discharge_fc, scoreboard, kpis


try:
    hist, care_fc, discharge_fc, scoreboard, kpis = load_data()
except Exception as exc:
    st.error("The dashboard could not load its data.")
    st.exception(exc)
    st.stop()


# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
st.sidebar.header("Controls")

available_horizons = [h for h in [1, 7, 14]
                      if h <= len(care_fc) and h <= len(discharge_fc)]

if not available_horizons:
    st.error("The forecast files do not contain enough rows to display.")
    st.stop()

horizon = st.sidebar.selectbox(
    "Forecast horizon (days)",
    available_horizons,
    index=len(available_horizons) - 1,
)

available_models = (
    scoreboard["model"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

if not available_models:
    st.error("No model names were found in the model scoreboard.")
    st.stop()

model_choice = st.sidebar.selectbox(
    "Model to highlight in the comparison",
    available_models,
)

scenario = st.sidebar.radio(
    "Scenario",
    ["Baseline", "Capacity stress (upper bound)"],
)

target_col = (
    "upper"
    if scenario == "Capacity stress (upper bound)"
    else "mean"
)

care_slice = care_fc.iloc[:horizon].copy()
discharge_slice = discharge_fc.iloc[:horizon].copy()

st.sidebar.caption(
    "The model selector highlights a model in the comparison table. "
    "The forecast charts display the precomputed forecasts stored in "
    "the repository."
)


# --------------------------------------------------
# Module 1: Care load forecast
# --------------------------------------------------
st.subheader("Future Care Load Forecast")

st.caption(
    "Historical children in HHS care and the forecast. "
    "The shaded region represents the supplied 90% uncertainty interval."
)

fig, ax = plt.subplots(figsize=(12, 4))

history_care = hist["Children in HHS Care"].dropna().iloc[-30:]

ax.plot(
    history_care.index,
    history_care,
    label="Historical care load",
    color="C0",
)

ax.plot(
    care_slice.index,
    care_slice[target_col],
    label=(
        "Upper-bound scenario"
        if scenario == "Capacity stress (upper bound)"
        else "Baseline forecast"
    ),
    color="C1",
    marker="o",
)

ax.fill_between(
    care_slice.index,
    care_slice["lower"],
    care_slice["upper"],
    alpha=0.2,
    color="C1",
    label="90% uncertainty interval",
)

ax.set_xlabel("Date")
ax.set_ylabel("Children in HHS care")
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)


# --------------------------------------------------
# Module 2: Discharge demand forecast
# --------------------------------------------------
st.subheader("Discharge Demand Forecast")

st.caption(
    "Historical discharges and the forecast with the supplied "
    "90% uncertainty interval."
)

fig_d, ax_d = plt.subplots(figsize=(12, 4))

history_discharge = (
    hist["Children discharged from HHS Care"].dropna().iloc[-30:]
)

ax_d.plot(
    history_discharge.index,
    history_discharge,
    label="Historical discharges",
    color="C0",
)

ax_d.plot(
    discharge_slice.index,
    discharge_slice[target_col],
    label=(
        "Upper-bound scenario"
        if scenario == "Capacity stress (upper bound)"
        else "Baseline forecast"
    ),
    color="C2",
    marker="o",
)

ax_d.fill_between(
    discharge_slice.index,
    discharge_slice["lower"],
    discharge_slice["upper"],
    alpha=0.2,
    color="C2",
    label="90% uncertainty interval",
)

ax_d.set_xlabel("Date")
ax_d.set_ylabel("Children discharged")
ax_d.legend()
ax_d.grid(alpha=0.2)
fig_d.tight_layout()

st.pyplot(fig_d)
plt.close(fig_d)


# --------------------------------------------------
# Module 3: Model selection and comparison
# --------------------------------------------------
st.subheader("Model Selection & Comparison")

st.caption(
    f"Showing model evaluation results for the {horizon}-day horizon. "
    f"Selected model: {model_choice}."
)

horizon_scores = scoreboard[
    pd.to_numeric(scoreboard["horizon"], errors="coerce") == horizon
].copy()

horizon_scores["MAE"] = pd.to_numeric(
    horizon_scores["MAE"], errors="coerce"
)

horizon_scores = (
    horizon_scores
    .sort_values("MAE", na_position="last")
    .reset_index(drop=True)
)

if horizon_scores.empty:
    st.warning(
        f"No model evaluation results were found for the {horizon}-day horizon."
    )
else:
    def highlight_selected_model(row):
        if str(row["model"]) == model_choice:
            return ["background-color: #fff3cd"] * len(row)
        return [""] * len(row)

    st.dataframe(
        horizon_scores.style.apply(
            highlight_selected_model,
            axis=1,
        ),
        width="stretch",
        hide_index=True,
    )

    valid_scores = horizon_scores.dropna(subset=["MAE"])

    if not valid_scores.empty:
        best_row = valid_scores.loc[valid_scores["MAE"].idxmin()]
        st.metric(
            label="Lowest MAE at this horizon",
            value=f"{best_row['MAE']:.3f}",
            help=f"Model: {best_row['model']}",
        )


# --------------------------------------------------
# Module 4: Key performance indicators
# --------------------------------------------------
st.subheader("Key Performance Indicators")

if kpis.empty:
    st.info("The KPI summary file contains no rows.")
else:
    st.dataframe(
        kpis,
        width="stretch",
        hide_index=True,
    )


# --------------------------------------------------
# Module 5: Forecast uncertainty
# --------------------------------------------------
st.subheader("Care Load: Uncertainty Band Detail")

st.caption(
    f"Forecast interval over the selected {horizon}-day horizon. "
    "The interval width is upper minus lower."
)

uncertainty = care_slice["upper"] - care_slice["lower"]

fig2, ax2 = plt.subplots(figsize=(12, 3))

ax2.fill_between(
    care_slice.index,
    care_slice["lower"],
    care_slice["upper"],
    alpha=0.25,
    color="C1",
    label="90% uncertainty interval",
)

ax2.plot(
    care_slice.index,
    care_slice["mean"],
    color="C1",
    marker="o",
    label="Forecast mean",
)

ax2.set_xlabel("Date")
ax2.set_ylabel("Children in HHS care")
ax2.legend()
ax2.grid(alpha=0.2)
fig2.tight_layout()

st.pyplot(fig2)
plt.close(fig2)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Average forecast",
        f"{care_slice['mean'].mean():,.1f}",
    )

with col2:
    st.metric(
        "Maximum upper bound",
        f"{care_slice['upper'].max():,.1f}",
    )

with col3:
    st.metric(
        "Average interval width",
        f"{uncertainty.mean():,.1f}",
    )


# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()

st.caption(
    "Forecasts and uncertainty intervals are based on the precomputed "
    "outputs in this project. They should be interpreted as analytical "
    "estimates, not as official operational capacity or placement decisions."
)