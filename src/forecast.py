from src.feature_engineering import prepare_features
from src.train_model import build_forecaster
import pandas as pd
import random
import numpy as np


# Set seeds
random.seed(42)
np.random.seed(42)

def forecast_future(df, target='household_deposits', forecast_horizon=12, sp=12, n_estimators=200, learning_rate=0.01, max_depth=4, min_samples_leaf=1, subsample=1, window_length=160):
    """
    Fits the time series model for the specified horizon on the full data and predicts h steps ahead.

    Parameters:
    - df (pd.DataFrame): Processed input DataFrame with features including 'date', target and predictors as well as engineered features
    - target (str): Target variable to forecast
    - forecast_horizon (int): Number of months ahead to forecast (e.g. 2 = 2-month ahead, from the data perpsective)
    - sp (int): Seasonal period (set to 12 for the deposits data as strong yearly seasonaility observed)
    - n_estimators, learning_rate, max_depth, min_samples_leaf, subsample: parameters for GBT
    - window_length (int): Number of past observations used in GBT residual correction

    Returns:
    - forecast_df (pd.DataFrame): Forecasted value with datetime index
    - forecaster: Trained forecasting model
    """

    # Fit full model on all data
    forecaster = build_forecaster(sp, n_estimators, learning_rate, max_depth, window_length, min_samples_leaf, subsample)
    y = df[target]
    X = df.drop(columns=[target])
    forecaster.fit(y, X, fh = [forecast_horizon])

    # Forecast h steps ahead
    y_pred = forecaster.predict(fh=[forecast_horizon], X=X)

    # Compute the forecast date
    forecast_date = (df.index[-1] + forecast_horizon).to_timestamp()

    forecast_df = pd.DataFrame({
        "forecast": [y_pred.values[0]]
    }, index=[forecast_date])

    return forecast_df, forecaster

def get_forecast_params(h, raw_df):
    """
    Returns:
    - adjusted raw_df (e.g. shortened if using earlier point for horizon 1)
    - actual forecast horizon to use (2 for horizon 1, otherwise the original)
    """
    if h == 1:
        df = raw_df[:-1].copy()        # Drop latest to simulate previous month
        actual_h = 2            # Still forecasting 2 months ahead
    else:
        df = raw_df.copy()
        actual_h = h
    return df, actual_h

def train_and_forecast(df, h, param_dict, target):
    """
    Wrapper for running the forecast pipeline.
    """
    # Get model params, data, and adjusted forecast horizon
    df_subset, actual_h = get_forecast_params(h, df)

    # Generate the forecast
    forecast_df, forecaster = forecast_future(
        df=df_subset,
        target=target,
        forecast_horizon=actual_h,
        sp=param_dict['sp'],
        n_estimators=param_dict['n_estimators'],
        learning_rate=param_dict['learning_rate'],
        max_depth=param_dict['max_depth'],
        min_samples_leaf=param_dict['min_samples_leaf'],
        subsample=param_dict['subsample'],
        window_length=param_dict['window_length']
    )

    return forecast_df, forecaster