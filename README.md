# UAC Care Load & Placement Demand Forecasting

A time-series forecasting project that uses historical data from the U.S. Department of Health and Human Services (HHS) Unaccompanied Alien Children (UAC) Program to explore care-load trends, discharge demand, and short-term forecasting.

The project combines data auditing, exploratory data analysis, statistical forecasting, machine learning, uncertainty estimation, and an interactive Streamlit dashboard to support data-informed capacity planning.

## Objectives

- Forecast the number of children in HHS care over short-term horizons.
- Estimate future discharge demand.
- Explore historical trends and patterns in care and transfer data.
- Compare statistical and machine-learning forecasting approaches.
- Evaluate model performance across different forecast horizons.
- Quantify forecast uncertainty using prediction intervals.
- Present forecasts, evaluation metrics, and operational indicators in an interactive dashboard.

## Dashboard Features

The Streamlit dashboard provides the following capabilities:

- **Care-load forecasting:** Visualize historical care counts alongside future projections and 90% uncertainty intervals.
- **Discharge forecasting:** Examine projected discharges and their associated uncertainty.
- **Flexible forecast horizons:** Switch between 1-day, 7-day, and 14-day forecasts.
- **Model comparison:** Compare model performance at the selected forecast horizon.
- **Capacity stress scenario:** Inspect the upper forecast bound as a conservative planning scenario.
- **KPI summaries:** Review key performance indicators and forecast uncertainty.

The dashboard loads precomputed forecasts and evaluation results from the `outputs/` directory.

## Repository Structure

```text
UAC_project/
├── data/
│   ├── HHS_Unaccompanied_Alien_Children_Program.csv
│   ├── hhs_cleaned.csv
│   └── hhs_filled.csv
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_statistical_models.ipynb
│   ├── 04_ml_models.ipynb
│   ├── 05_kpis_uncertainty.ipynb
│   └── 06_discharge_forecast.ipynb
├── outputs/
│   ├── care_load_forecast.csv
│   ├── discharge_forecast.csv
│   ├── combined_scoreboard.csv
│   ├── statistical_scoreboard.csv
│   ├── kpi_summary.csv
│   └── figures/
└── scripts/
    └── app.py
```

### Directory Overview

| Directory | Purpose |
|---|---|
| `data/` | Source, cleaned, and processed datasets |
| `notebooks/` | Sequential data analysis, modelling, and evaluation workflow |
| `outputs/` | Forecasts, model scoreboards, KPI summaries, and visualizations |
| `scripts/` | Streamlit dashboard application |

## Methodology

The project follows a structured forecasting workflow.

### 1. Data Auditing and Preparation

Inspect the source dataset, identify missing values, examine data quality, and prepare cleaned and filled datasets for analysis.

### 2. Exploratory Data Analysis

Investigate historical time-series behaviour, trends, changes in care load, and relationships among relevant program indicators.

### 3. Statistical Forecasting

Develop and evaluate statistical time-series models to establish forecasting baselines and capture temporal patterns.

### 4. Machine-Learning Models

Explore machine-learning approaches and compare their predictive performance with statistical models.

### 5. Model Evaluation

Compare forecast errors across different horizons and use evaluation scoreboards to assess model performance.

### 6. Uncertainty and KPI Analysis

Examine forecast intervals and summarize key performance indicators to communicate both expected outcomes and potential variation.

### 7. Discharge Forecasting

Model discharge demand separately to provide another perspective on future care-system movement.

### 8. Interactive Visualization

Present forecast results, uncertainty bands, and model comparisons through a Streamlit dashboard.

For the complete analysis, start with `01_data_audit.ipynb` and proceed through the notebooks in numerical order.

## Data

The repository contains the original dataset and derived cleaned and filled versions.

The analysis focuses on program indicators relating to:

- Children apprehended or placed in CBP custody
- Children in CBP custody
- Transfers out of CBP custody
- Children in HHS care
- Children discharged from HHS care

Refer to the source dataset and data-audit notebook for the exact column definitions, date coverage, and preprocessing decisions.

## Getting Started

### Prerequisites

- Python 3.10 or later recommended
- Git
- Jupyter Notebook
- Required Python data-science and visualization libraries

### 1. Clone the Repository

```bash
git clone https://github.com/abhinavray07/UAC_project.git
cd UAC_project
```

### 2. Create a Virtual Environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install the core libraries used by the notebooks and dashboard:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn statsmodels jupyter streamlit
```

Additional dependencies may be required depending on the models and packages used in individual notebooks.

### 4. Run the Notebooks

From the repository root:

```bash
jupyter notebook
```

Open the notebooks in `notebooks/` and execute them in numerical order. Some notebooks may depend on processed datasets or outputs generated in earlier stages.

### 5. Launch the Dashboard

Run the application from the `scripts/` directory:

```bash
cd scripts
streamlit run app.py
```

Open the local URL printed by Streamlit, typically:

`http://localhost:8501`

**Important:** The dashboard expects the required data and forecast CSV files to exist in `data/` and `outputs/`. Generate or restore these files before launching the application if they are missing.

## Model Evaluation and Forecast Uncertainty

The `outputs/` directory contains model scoreboards, forecast results, and KPI summaries.

The dashboard compares models at the selected forecast horizon and sorts the displayed scoreboard by Mean Absolute Error (MAE).

Forecast evaluation should respect the chronological structure of time-series data. A time-ordered validation strategy helps prevent information from the future from leaking into model training.

Prediction intervals communicate uncertainty around the forecasts. Wider intervals generally indicate greater uncertainty, while the upper-bound scenario provides a conservative view for capacity planning.

Consult the modelling notebooks for the exact validation strategy, model configurations, and metric calculations.

## Technology Stack

- **Python** — core programming language
- **Pandas and NumPy** — data manipulation and numerical computation
- **Matplotlib and Seaborn** — data visualization
- **Statsmodels** — statistical modelling and time-series analysis
- **Scikit-learn** — machine-learning models and evaluation
- **Jupyter Notebook** — experimentation and analysis
- **Streamlit** — interactive dashboard

## Limitations and Responsible Use

- Forecasts are estimates based on historical data, not guarantees of future demand.
- Policy changes, operational disruptions, and unusual events may not be captured by historical patterns.
- The capacity-stress scenario uses the forecast's upper bound; it is not a separate causal model of capacity requirements.
- Model performance depends on the data, validation procedure, and evaluation period.
- The project is an analytical prototype, not an official HHS or CBP system.
- Forecasts should not be used as the sole basis for operational, legal, or individual-level decisions.

## Future Improvements

- Add a pinned dependency file for reproducible installations.
- Document data provenance, update frequency, and missing-value treatment.
- Add automated tests for preprocessing and forecast-output validation.
- Create a unified pipeline for regenerating forecasts and launching the dashboard.
- Monitor forecast errors as new observations become available.

## Author

**Abhinav Ray**

[GitHub Profile](https://github.com/abhinavray07) · [Project Repository](https://github.com/abhinavray07/UAC_project)

---

This project explores how time-series analysis and predictive modelling can help interpret historical care demand and support more informed capacity planning.
