import pandas as pd
from src.feature_engineering import prepare_features

def load_and_prepare_data(path, target, params):
    """
    Load the raw dataset and prepare the features necesary for each model. 

    Parameters:
    - path (string): Path to the raw dataset
    - target (string): Name of the target column
    - params (dict): Dictionary of the model parameters
     
    Returns:
    - pd.DataFrame: Processed and indexed DataFrame ready for forecasting
    """
    raw_df = pd.read_csv(path)
    raw_df = raw_df.sort_values('date')
    predictors = params['predictors']
    df = raw_df[['date',target]+predictors].copy()
    
    df = prepare_features(
        df,
        target,
        lags=params['best_lags'],
        rolling_windows=params['best_rolling_windows'],
        use_cols=params['best_features']
    )
    return df