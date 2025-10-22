import pandas as pd
import streamlit as st

@st.cache_data
def read_data(path):
    """Generic CSV reader with caching."""
    return pd.read_csv(path)

@st.cache_data
def load_all_data():
    """Load and return all necessary datasets."""
    data = {}
    data['forecasts'] = read_data('data/final_forecast_df.csv')
    data['historic'] = read_data('data/legacy_combined_df_for_eda.csv')
    data['legacy'] = read_data('data/legacy_combined_df_for_eda.csv')
    data['sector'] = read_data('data/deposits_rbnz.csv')
    data['val_metric'] = read_data('data/all_forecasts_summary.csv')

    # Load validation results by horizon
    horizons = [2, 5, 8, 14]
    data['val_dict'] = {
        h: read_data(f'data/model_evaluation_results_h{h}.csv')
        for h in horizons
    }

    return data
