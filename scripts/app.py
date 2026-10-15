from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# ==================================================
# 1. APP CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="UAC Care Load Forecasting",
    page_icon="📊",
    layout="wide",
)

st.title("Predictive Forecasting: Care Load & Placement Demand")
st.caption(
    "Explore historical care-load trends, discharge forecasts, "
    "model accuracy, key performance indicators, and forecast uncertainty."
)


# ==================================================
# 2. PROJECT PATHS
# Resolve all paths from the repository root.
# Expected location: scripts/app.py
# ==================================================
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

HISTORY_FILE = DATA_DIR / "hhs_filled.csv"
CARE_FORECAST_FILE = OUTPUT_DIR / "care_load_forecast.csv"
DISCHARGE_FORECAST_FILE = OUTPUT_DIR / "discharge_forecast.csv"
SCOREBOARD_FILE = OUTPUT_DIR / "combined_scoreboard.csv"
KPI_FILE = OUTPUT_DIR / "kpi_summary.csv"


# ==================================================
# 3. DATA LOADING AND VALIDATION
# ==================================================
@st.cache_data
def load_data():
    """Load project CSV files and validate their schemas."""

    required_files = {
        "Historical data": HISTORY_FILE,
        "Care-load forecast": CARE_FORECAST_FILE,
        "Discharge forecast": DISCHARGE_FORECAST_FILE,
        "Model scoreboard": SCOREBOARD_FILE,
        "KPI summary": KPI_FILE,
    }

    missing_files = [
        f"{label}: {path.relative_to(PROJECT_ROOT)}"
        for label, path in required_files.items()
        if not path.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required project files were not found:\n\n"
            + "\n".join(f"- {item}" for item in missing_files)
            + "\n\nVerify that these files exist in the repository, "
            "that their capitalization matches, and that they have "
            "been committed and pushed to GitHub."
        )

    # Read historical observations.
    hist = pd.read_csv(HISTORY_FILE)

    if "Date" not in hist.columns:
        raise ValueError(
            "hhs_filled.csv must contain a column named 'Date'. "
            f"Found columns: {hist.columns.tolist()}"
        )

    hist["Date"] = pd.to_datetime(hist["Date"], errors="coerce")
    hist = hist.dropna(subset=["Date"])
    hist = hist.set_index("Date").sort_index()

    # Read forecast files.
    def read_forecast(path, label):
        df = pd.read_csv(path)

        # Support files whose date column is named Date, or whose
        # first column contains the forecast dates.
        if df.empty:
            raise ValueError(f"{label} is empty: {path.name}")

        date_column = next(
            (
                col for col in df.columns
                if str(col).strip().lower() in {
                    "date", "ds", "datetime", "timestamp"
                }
            ),
            df.columns[0],
        )

        parsed_dates = pd.to_datetime(
            df[date_column],
            errors="coerce",
        )

        # If the first column is not a date index, report the issue
        # rather than silently plotting against row numbers.
        if parsed_dates.isna().any():
            raise ValueError(
                f"Could not parse all forecast dates in {path.name}. "
                f"Column used: '{date_column}'. "
                "Check the CSV's date/index format."
            )

        df = df.drop(columns=[date_column])
        df.index = parsed_dates
        df.index.name = "Date"
        df = df.sort_index()

        return df

    care_fc = read_forecast(
        CARE_FORECAST_FILE,
        "Care-load forecast",
    )

    discharge_fc = read_forecast(
        DISCHARGE_FORECAST_FILE,
        "Discharge forecast",
    )

    scoreboard = pd.read_csv(SCOREBOARD_FILE)
    kpis = pd.read_csv(KPI_FILE)

    # ----------------------------------------------
    # Validate historical data
    # ----------------------------------------------
    required_history_columns = [
        "Children in HHS Care",
        "Children discharged from HHS Care",
    ]

    missing = [
        col for col in required_history_columns
        if col not in hist.columns
    ]

    if missing:
        raise ValueError(
            f"hhs_filled.csv is missing columns: {missing}. "
            f"Available columns: {hist.columns.tolist()}"
        )

    # ----------------------------------------------
    # Validate forecast schemas
    # ----------------------------------------------
    required_forecast_columns = ["lower", "mean", "upper"]

    for label, df, filename in [
        ("Care-load forecast", care_fc, CARE_FORECAST_FILE.name),
        (
            "Discharge forecast",
            discharge_fc,
            DISCHARGE_FORECAST_FILE.name,
        ),
    ]:
        missing = [
            col for col in required_forecast_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"{filename} is missing columns: {missing}. "
                f"Available columns: {df.columns.tolist()}"
            )

        if df.empty:
            raise ValueError(f"{label} contains no forecast rows.")

        # Ensure chart values are numeric.
        for col in required_forecast_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if df[required_forecast_columns].isna().any().any():
            raise ValueError(
                f"{filename} contains missing or non-numeric forecast "
                "values in lower, mean, or upper."
            )

        if (df["lower"] > df["upper"]).any():
            raise ValueError(
                f"{filename} contains lower bounds greater than upper bounds."
            )

    # ----------------------------------------------
    # Validate scoreboard
    # ----------------------------------------------
    required_scoreboard_columns = ["model", "horizon", "MAE"]

    missing = [
        col for col in required_scoreboard_columns
        if col not in scoreboard.columns
    ]

    if missing:
        raise ValueError(
            "combined_scoreboard.csv is missing columns: "
            f"{missing}. Available columns: {scoreboard.columns.tolist()}"
        )

    if scoreboard.empty:
        raise ValueError("combined_scoreboard.csv contains no rows.")

    scoreboard["horizon"] = pd.to_numeric(
        scoreboard["horizon"],
        errors="coerce",
    )

    scoreboard["MAE"] = pd.to_numeric(
        scoreboard["MAE"],
        errors="coerce",
    )

    scoreboard["model"] = scoreboard["model"].astype("string")

    if scoreboard["model"].dropna().empty:
        raise ValueError(
            "combined_scoreboard.csv does not contain any valid model names."
        )

    # ----------------------------------------------
    # Validate history and KPI data
    # ----------------------------------------------
    if hist.empty:
        raise ValueError(
            "hhs_filled.csv contains no usable rows with valid dates."
        )

    return hist, care_fc, discharge_fc, scoreboard, kpis


try:
    hist, care_fc, discharge_fc, scoreboard, kpis = load_data()

except Exception as exc:
    st.error("The dashboard could not load its data.")
    st.exception(exc)

    with st.expander("Repository paths being checked"):
        st.code(
            "\n".join(
                [
                    f"Project root: {PROJECT_ROOT}",
                    f"History: {HISTORY_FILE}",
                    f"Care forecast: {CARE_FORECAST_FILE}",
                    f"Discharge forecast: {DISCHARGE_FORECAST_FILE}",
                    f"Scoreboard: {SCOREBOARD_FILE}",
                    f"KPI summary: {KPI_FILE}",
                ]
            )
        )

    st.stop()


# ==================================================
# 4. SIDEBAR CONTROLS
# ==================================================
st.sidebar.header("Controls")

available_horizons = [
    horizon
    for horizon in [1, 7, 14]
    if horizon <= len(care_fc) and horizon <= len(discharge_fc)
]

if not available_horizons:
    st.error(
        "The forecast files do not contain enough rows for a "
        "1-, 7-, or 14-day forecast."
    )
    st.stop()

horizon = st.sidebar.selectbox(
    "Forecast horizon (days)",
    available_horizons,
    index=len(available_horizons) - 1,
)

available_models = sorted(
    scoreboard["model"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

if not available_models:
    st.error("No valid model names were found in the model scoreboard.")
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
    "The selected model is highlighted in the comparison table. "
    "Forecast charts use the precomputed forecast files in the repository."
)


# ==================================================
# 5. HELPER FUNCTIONS
# ==================================================
def plot_forecast(
    historical_series,
    forecast_df,
    target_column,
    title,
    y_label,
    forecast_color,
    forecast_label,
):
    """Plot historical observations and forecast uncertainty."""

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        historical_series.index,
        historical_series.values,
        label="Historical observations",
        color="C0",
        marker=".",
        linewidth=1.8,
    )

    ax.plot(
        forecast_df.index,
        forecast_df[target_column],
        label=forecast_label,
        color=forecast_color,
        marker="o",
        linewidth=2,
    )

    ax.fill_between(
        forecast_df.index,
        forecast_df["lower"].to_numpy(dtype=float),
        forecast_df["upper"].to_numpy(dtype=float),
        alpha=0.2,
        color=forecast_color,
        label="Supplied uncertainty interval",
    )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(y_label)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()

    return fig


forecast_label = (
    "Upper-bound scenario"
    if scenario == "Capacity stress (upper bound)"
    else "Baseline forecast"
)


# ==================================================
# 6. MODULE 1: CARE LOAD FORECAST
# ==================================================
st.subheader("Future Care Load Forecast")

st.caption(
    "Historical children in HHS care and the selected forecast. "
    "The shaded region shows the supplied uncertainty interval."
)

history_care = (
    hist["Children in HHS Care"]
    .pipe(pd.to_numeric, errors="coerce")
    .dropna()
    .iloc[-30:]
)

if history_care.empty:
    st.warning(
        "No numeric historical care-load observations are available to plot."
    )
else:
    fig = plot_forecast(
        historical_series=history_care,
        forecast_df=care_slice,
        target_column=target_col,
        title="Historical and Forecast Care Load",
        y_label="Children in HHS care",
        forecast_color="C1",
        forecast_label=forecast_label,
    )

    st.pyplot(fig, width="stretch")
    plt.close(fig)


# ==================================================
# 7. MODULE 2: DISCHARGE DEMAND FORECAST
# ==================================================
st.subheader("Discharge Demand Forecast")

st.caption(
    "Historical discharges and the selected forecast, "
    "including the supplied uncertainty interval."
)

history_discharge = (
    hist["Children discharged from HHS Care"]
    .pipe(pd.to_numeric, errors="coerce")
    .dropna()
    .iloc[-30:]
)

if history_discharge.empty:
    st.warning(
        "No numeric historical discharge observations are available to plot."
    )
else:
    fig_d = plot_forecast(
        historical_series=history_discharge,
        forecast_df=discharge_slice,
        target_column=target_col,
        title="Historical and Forecast Discharges",
        y_label="Children discharged",
        forecast_color="C2",
        forecast_label=forecast_label,
    )

    st.pyplot(fig_d, width="stretch")
    plt.close(fig_d)


# ==================================================
# 8. MODULE 3: MODEL SELECTION AND COMPARISON
# ==================================================
st.subheader("Model Selection & Comparison")

st.caption(
    f"Evaluation results for the {horizon}-day horizon. "
    f"Selected model: {model_choice}."
)

horizon_scores = scoreboard[
    scoreboard["horizon"] == horizon
].copy()

horizon_scores = (
    horizon_scores
    .sort_values("MAE", na_position="last")
    .reset_index(drop=True)
)

if horizon_scores.empty:
    st.warning(
        f"No model evaluation results were found for the "
        f"{horizon}-day horizon."
    )

else:

    def highlight_selected_model(row):
        if str(row["model"]) == model_choice:
            return ["background-color: #4a3f00; color: #ffffff; font-weight: 700;"] * len(row)
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

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:
            st.metric(
                "Lowest MAE at this horizon",
                f"{best_row['MAE']:.3f}",
            )

        with metric_col2:
            selected_scores = valid_scores[
                valid_scores["model"].astype(str) == model_choice
            ]

            if not selected_scores.empty:
                st.metric(
                    "Selected model MAE",
                    f"{selected_scores['MAE'].iloc[0]:.3f}",
                )
            else:
                st.metric(
                    "Selected model MAE",
                    "Not available",
                )


# ==================================================
# 9. MODULE 4: KEY PERFORMANCE INDICATORS
# ==================================================
st.subheader("Key Performance Indicators")

if kpis.empty:
    st.info("The KPI summary file contains no rows.")

else:
    st.dataframe(
        kpis,
        width="stretch",
        hide_index=True,
    )


# ==================================================
# 10. MODULE 5: FORECAST UNCERTAINTY
# ==================================================
st.subheader("Care Load: Uncertainty Band Detail")

st.caption(
    f"Forecast interval over the selected {horizon}-day horizon. "
    "Interval width is calculated as upper bound minus lower bound."
)

uncertainty = care_slice["upper"] - care_slice["lower"]

fig2, ax2 = plt.subplots(figsize=(12, 3))

ax2.fill_between(
    care_slice.index,
    care_slice["lower"].to_numpy(dtype=float),
    care_slice["upper"].to_numpy(dtype=float),
    alpha=0.25,
    color="C1",
    label="Supplied uncertainty interval",
)

ax2.plot(
    care_slice.index,
    care_slice["mean"],
    color="C1",
    marker="o",
    label="Forecast mean",
)

ax2.set_title("Care Load Forecast Uncertainty")
ax2.set_xlabel("Date")
ax2.set_ylabel("Children in HHS care")
ax2.legend()
ax2.grid(alpha=0.2)
fig2.autofmt_xdate()
fig2.tight_layout()

st.pyplot(fig2, width="stretch")
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


# ==================================================
# 11. FOOTER
# ==================================================
st.divider()

st.caption(
    "Forecasts and uncertainty intervals are based on the precomputed "
    "outputs in this project. They should be interpreted as analytical "
    "estimates, not as official operational capacity or placement decisions."
)
