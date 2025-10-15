# Forecast of NZ Total Household Deposits

This repo consists of the relevant code for developing 4 forecasting models to predict the current month, 3-month, 6-month and 12-month horizons of the NZ total household deposits. 

## Content Table
1. [Introduction](#introduction) 
2. [Assumptions](#assumptions)
3. [Summary of Methodology](#summary-of-methodology) 
4. [Input](#input)
5. [Output](#output)
6. [StreamLit App](#streamlit-app)

## Introduction

This project develops forecasting models to predict NZ total household deposits for different time horizons. It helps financial analysts understand and anticipate trends in household savings. The data is based entirely on public data.

## Assumptions
Key assumptions and constraints for this project include:
- **Public data limitation**: All data used in this project is sourced from publicly available datasets. This inherently limits the volume and scope of data. The target dataset is updated monthly, and as of October 2025, only 321 monthly observations are available (with the latest data point being from August 2025). This relatively small dataset imposes constraints on model complexity and robustness.
- **Data publication lag**: There is a typical 2-month delay in data publication by the Reserve Bank of New Zealand (RBNZ). As a result, a clear distinction must be made between the calendar forecast horizon and the actual data horizon. For example, although the current month is October 2025, the latest available data is from August 2025 — meaning a forecast for “October” requires a 2-month-ahead prediction. A table below outlines the mapping between calendar months and model forecast horizons.

  |Actual Horizon (Calendar)|Model Forecast Horizon (Data)|
  |---|---|
  |Current Month (0-Month)|2-Month|
  |3-Month|5-Month|
  |6-Month|8-Month|
  |12-Month|14-Month|

- **Handling of quarterly data**: Some input variables are only available at a quarterly frequency. To align these with the monthly target series, interpolation or duplication techniques are used. In cases where the latest quarterly data has not yet been published, forecasted values of those features are used instead. This introduces additional uncertainty into the model.
- **Inclusion of the COVID-19 period**: The dataset includes the period impacted by the COVID-19 pandemic (2020–2021), during which household deposit behavior and broader macroeconomic conditions may have been disrupted. A visual inspection was conducted to identify and exclude any datasets that appeared to be heavily affected during this period. However, it is possible that some outliers remain undetected. These anomalies may influence the model’s overall performance and its ability to generalize to future periods.


## Summary of Methodology
The overall methodology and data pipeline for this project involves the following:

![Data Pipeline](https://github.com/elizabeth050594/deposits_forecast_prod/blob/main/assets/img/data-pipeline.png)

**A. Data Collection**  
- Initial dataset: 49 macroeconomic & financial variables.  
- Monthly data: Jan 2004 – Aug 2025; quarterly data up to Jun 2025.  
- See Data Log for variable details and update frequencies.

**B. Data Preprocessing**  
- Handled missing values via interpolation or forward-fill.  
- Interpolated quarterly data to monthly frequency linearly.  
- Standardized and merged all variables into a unified, time-aligned dataset.

**C. Feature Reduction (See EDA notebook for details)**  
**Stage 1: Statistical Filtering**  
- Applied STL decomposition (trend & seasonal).  
- Selected features based on max Pearson correlation (>0.9) and significant Granger causality (p<`0.05`).  

**Stage 2: Manual Inspection**  
- Removed features with COVID-19 anomalies after visual review.  
- Reduced to 16 features.

**Stage 3: Forecast Availability**  
- Excluded features lacking reliable future data.  
- Kept only monthly or forecastable quarterly variables.

**D. Feature Engineering**  
- Created time series features: lags (e.g., 1, 2, 8, 12 months), differences, rate of change, rolling stats (mean, std, min, max).  
- Added datetime features: year, quarter, month.

**E. Model Framework**  
**Feature Selection**  
- Residual learning on AutoETS forecasts using Gradient Boosting Regressor (GBR).  
- Used expanding window CV; GBR trained on residuals.  
- SHAP values computed to rank feature importance across folds.

**Hyperparameter Tuning**  
- Used Optuna for tuning model parameters separately for each forecast horizon.

**Evaluation**  
- Backtested on last 12 months (hold-out test set).  
- Metrics: RMSE, MAE, Max Error.

**Forecasting**  
- Produced forecasts for 2, 5, 8, and 14-month horizons based on current data.

## Input
The final features used after selection and model development are the target variable `household_deposits` and the `household_loans` feature. Both datasets are monthly and sourced from the Reserve Bank of New Zealand (RBNZ). For more details, refer to the Data Log.

## Output
- Forecasted results via the notebook `00_run_forecasting_pipeline`
- Validation results and plots via the notebook `01_run_validation_pipeline`

## How to Use
- For a detailed walkthrough of the model development and analysis, please refer to the accompanying notebooks.
- To generate forecasts using the latest available data, run the notebook `00_run_forecasting_pipeline`.
- To evaluate model performance on the most recent 12 months, run `01_run_validation_pipeline`, which produces error metrics and validation results by default.

## Updating the Data
Currently, data updates require appending new entries to data/raw/total_household_deposits_monthly_data.csv. **Need to check with stakeholders if they want a more user-friendly update process (e.g., a table or dashboard) or if manual CSV updates are sufficient.

## StreamLit
