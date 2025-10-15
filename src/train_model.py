import random
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.compose import TransformedTargetForecaster, make_reduction
from sktime.forecasting.residual_booster import ResidualBoostingForecaster
from sktime.forecasting.base import ForecastingHorizon

# Set seeds
random.seed(42)
np.random.seed(42)

def build_forecaster(sp, n_estimators, learning_rate, max_depth, window_length, min_samples_leaf, subsample):
    """
    Builds the combined forecasting model (AutoETS + Gradient Boosting residual corrector).

    Parameters:
    - sp (int): Seasonal period (e.g. 12 for monthly data with yearly seasonality)
    - n_estimators (int): Number of boosting trees
    - learning_rate (float): Learning rate for GBT
    - max_depth (int): Max tree depth
    - window_length (int): Number of past time points used in the residual booster
    - min_samples_leaf (int): Minimum samples in leaf node
    - subsample (float): Fraction of samples used in boosting
     
    Returns:
    - sktime.forecasting.compose._pipeline.ForecasterPipeline: A composite forecaster object
    """
    base_forecaster = AutoETS(auto=True, sp=sp, n_jobs=-1)

    regressor = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf = min_samples_leaf, 
        subsample = subsample, 
        random_state=42
    )
    
    residual_model = make_reduction(regressor, window_length=window_length, strategy="direct")
    residual_forecaster = TransformedTargetForecaster([
        ("scaler", MinMaxScaler()),
        ("regressor", residual_model)
    ])

    full_forecaster = ResidualBoostingForecaster(base_forecaster, residual_forecaster)
    return full_forecaster