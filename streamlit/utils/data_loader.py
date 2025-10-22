import pandas as pd
import streamlit as st
from pathlib import Path

@st.cache_data
def read_data(path):
    """Generic CSV reader with caching."""
    return pd.read_csv(path)

@st.cache_data
def load_all_data():
    """Load and return all necessary datasets."""
    data = {}
    base_dir = Path(__file__).parent.parent.parent
    data['forecasts'] = read_data(base_dir/'outputs/forecasts/final_forecast_df.csv')
    data['historic'] = read_data(base_dir/'data/processed/combined_total_household_data_interpolate.csv')
    data['legacy'] = read_data(base_dir/'data/processed/legacy_combined_df_for_eda.csv')
    data['sector'] = read_data(base_dir/'data/raw/deposits_rbnz.csv')
    data['val_metric'] = read_data(base_dir/'outputs/validations/all_forecasts_summary.csv')

    # Load validation results by horizon
    horizons = [2, 5, 8, 14]
    data['val_dict'] = {
        h: read_data(base_dir/f'outputs/validations/model_forecast_h{h}/model_evaluation_results_h{h}.csv')
        for h in horizons
    }

    return data
