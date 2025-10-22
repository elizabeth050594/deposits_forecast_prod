import streamlit as st
import pandas as pd

# Set page config to use wide layout
st.set_page_config(
    page_title="NZ National Household Deposits Forecast",
    layout="wide"
)
st.title("About the Project")
st.write("This project develops forecasting models to predict NZ total household deposits for 2, 5, 8 and 14-month horizons. This corresponds to . The target is the `total household deposits` found on the [RBNZ website](https://www.rbnz.govt.nz/statistics/series/registered-banks/banks-assets-loans-by-sector). The data and the model is based entirely on publicly available features.")

st.markdown("### Key Assumptions")
st.markdown("""Key assumptions and constraints for this project include:
- **Public data limitation**:    
    All data used in this project is sourced from publicly available datasets. This inherently limits the volume and scope of data. The target dataset is updated monthly, and as of October 2025, only 321 monthly observations are available (with the latest data point being from August 2025). This relatively small dataset imposes constraints on model complexity and robustness.
- **Data publication lag**: 
            
    There is a typical 2-month delay in data publication by the Reserve Bank of New Zealand (RBNZ). As a result, a clear distinction must be made between the calendar forecast horizon and the actual data horizon. For example, if the current month is October 2025, the latest available data is only until August 2025 — meaning a forecast for the “current month" (October) requires a 2-month-ahead prediction. The table below outlines the mapping between calendar months and model forecast horizons.

  |Actual Horizon (Calendar)|Model Forecast Horizon (Data)|
  |---|---|
  |Current Month (0-Month)|2-Month|
  |3-Month|5-Month|
  |6-Month|8-Month|
  |12-Month|14-Month|

- **Handling of quarterly data**: 
            
    Some input variables are only available at a quarterly frequency. To align these with the monthly target series, interpolation or duplication techniques are used. In cases where the latest quarterly data has not yet been published, forecasted values of those features are used instead. This introduces additional uncertainty into the model.
- **Inclusion of the COVID-19 period**: 
            
    The dataset includes the period impacted by the COVID-19 pandemic (2020–2021), during which household deposit behavior and broader macroeconomic conditions may have been disrupted. A visual inspection was conducted to identify and exclude any datasets that appeared to be heavily affected during this period. However, it is possible that some outliers remain undetected. These anomalies may influence the model’s overall performance and its ability to generalize to future periods.
""")

st.markdown("### Project Framework")
st.image("assets/img/data-pipeline.png", caption="Data Pipeline", use_container_width="stretch")

st.markdown("### Model Descriptions")
st.write('The model uses two main predictors as input. The table below shows the latest model features and performances.')

data = {
    "Model": ["Current month", "3-month", "6-month", "12-month"],
    "Features": [
        "Household deposits: Rate of change, month-month differences, standard deviation",
        "Household deposits: Rate of change, month-month differences",
        "Household deposits: Rate of change, month-month differences",
        "Household deposits: Rate of change, month-month differences\nHousehold loans: Rate of change"
    ],
    "Average Error (RMSE) NZDm": ["$1,020", "$1,873", "$2,888", "$2,777"],
    "% Error": ["0.29%", "0.59%", "1.00%", "0.91%"]
}

df = pd.DataFrame(data)
st.dataframe(df.set_index('Model'))
