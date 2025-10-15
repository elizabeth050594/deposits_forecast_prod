import pandas as pd

def create_features(df, target_col, predictor_cols, lags=[1,3,4, 6,12], rolling_windows=[3,4, 6,12]):
    """
    Creates lag-based and rolling statistical features for both the target and predictor columns.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with 'date', target, and predictor columns
    - target_col (str): Name of the target column
    - predictor_cols (list): List of predictor column names
    - lags (list): Lags to apply for lag, diff, and pct change features
    - rolling_windows (list): Window sizes for rolling statistics (mean, std, min, max)

    Returns:
    - pd.DataFrame: DataFrame with engineered features and datetime features (year, month, quarter)
    """
    # Ensure date is in datetime format
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df.set_index('date', inplace=True)

    new_features = {}

    # For each predictor (and target), compute lags, rolling stats, etc.
    for col in [target_col] + predictor_cols:
        # Lag features
        for lag in lags:
            new_features[f'{col}_lag{lag}'] = df[col].shift(lag) # lag

        # Differences and pct changes
        for lag in lags:
            new_features[f'{col}_diff{lag}'] = df[col].diff(lag) # difference
            if (df[col] != 0).all():
                new_features[f'{col}_roc{lag}'] = df[col].pct_change(lag) # rate of change

        # Rolling stats
        for win in rolling_windows:
            new_features[f'{col}_ma{win}'] = df[col].rolling(win).mean()
            new_features[f'{col}_std{win}'] = df[col].rolling(win).std()
            new_features[f'{col}_min{win}'] = df[col].rolling(win).min()
            new_features[f'{col}_max{win}'] = df[col].rolling(win).max()

    # Combine original and new features        
    df_new = pd.concat([df]+ [pd.DataFrame(new_features, index=df.index)], axis=1)

    # Add datetime features
    df_new['year'] = df_new.index.year
    df_new['month'] = df_new.index.month
    df_new['quarter'] = df_new.index.quarter

    # Drop rows with missing values caused by shifting/rolling
    df_new = df_new.dropna()
    df_new.reset_index(inplace=True)

    return df_new

def prepare_features(df, target, lags, rolling_windows, use_cols):
    """
    Wrapper around create_features to:
    - Generate features
    - Subset to required columns
    - Set proper monthly period index

    Parameters:
    - df (pd.DataFrame): Input DataFrame with 'date', target and predictors
    - target (str): Target column name
    - lags (list): List of lag values
    - rolling_windows (list): List of rolling window sizes
    - use_cols (list): Final features to retain

    Returns:
    - pd.DataFrame: Processed and indexed DataFrame ready for forecasting
    """

    predictor_cols = [x for x in df.columns if x not in ['date', target]]
    df = create_features(df, target, predictor_cols, lags=lags, rolling_windows=rolling_windows)
    df = df[['date', target] + use_cols].copy()
    df = df.set_index('date').asfreq('MS')  # Set frequency to monthly start
    df.index = df.index.to_period("M")  # Use PeriodIndex (monthly) as required for sktime

    return df