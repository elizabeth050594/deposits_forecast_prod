import pandas as pd
from src.feature_engineering import prepare_features
from pathlib import Path

def process_quarterlies_duplicate(df):
    # List to hold monthly duplicated rows
    monthly_rows = []

    for _, row in df.iterrows():
        # Use the quarter-end date from the original row
        quarter_end = pd.to_datetime(row['date'])

        # Generate 3 monthly dates ending at the quarter-end (i.e., previous two months + current)
        months = pd.date_range(end=quarter_end, periods=3, freq='MS')

        # Duplicate the row for each month in the quarter
        for m in months:
            new_row = row.copy()
            new_row['date'] = m
            monthly_rows.append(new_row)

    # Create new DataFrame with monthly frequency
    monthly_df = pd.DataFrame(monthly_rows)

    # Format 'date' column to 'YYYY-MM'
    monthly_df['date'] = pd.to_datetime(monthly_df['date']).dt.strftime('%Y-%m')

    return monthly_df


def process_quarterlies_interpolate(df):
    non_date_cols = [x for x in df.columns if x != 'date']
    
    # Create a complete monthly date range
    months = pd.date_range(start=df.date.min(), end=df.date.max(), freq='MS')
    month_df = pd.DataFrame(months, columns=['date'])
    month_df['date'] = month_df['date'].dt.strftime('%Y-%m')

    # Merge with original DataFrame
    month_df = month_df.merge(df, on='date', how='left')

    # Interpolate with direction both ways
    month_df[non_date_cols] = month_df[non_date_cols].interpolate(limit_direction='both')

    return month_df


def clean_and_process_df(df):
    # Standardize column names
    df.columns = [col.lower() for col in df.columns]
    df.columns = (
        df.columns
        .str.replace(r"\s*\(.*?\)", "", regex=True)   # remove text in parentheses
        .str.replace(" ", "_", regex=False)           # replace spaces with underscores
        .str.replace("-", "_", regex=False)           # replace hyphens with underscores
        .str.replace(".", "", regex=False)            # remove dots
        .str.replace("\t", "", regex=False)            # remove tabs
    )

    # Convert 'date' column to datetime and format
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce').dt.strftime('%Y-%m')

    # Drop rows where all elements are NaN
    df = df.dropna(how='all')

    # Convert applicable columns to numeric
    for col in df.columns:
        if col == 'date':
            continue
        if df[col].dtype == object:
            # Remove commas if present
            if df[col].astype(str).str.contains(",", na=False).any():
                df[col] = df[col].str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def update_data(monthly_df, quarterly_df, output_path = 'data/processed'):
    """
    Process and combine the raw inputs into one dataframe. 

    Parameters:
     - monthly_df: df of the monthly data
     - quarterly_df: df of the quarterly data
    Returns:
     - combined_df: df of the combined data
    """

    # clean and process df
    monthly_df = clean_and_process_df(monthly_df)
    quarterly_df = clean_and_process_df(quarterly_df)

    # process quarterly data
    duplicate_quarterly_df = process_quarterlies_duplicate(quarterly_df) # duplicate quarterly data
    interpolated_quarterly_df = process_quarterlies_interpolate(quarterly_df) # interpolate quarterly data

    # combine and save
    combined_data_duplicated = monthly_df.merge(duplicate_quarterly_df, on = 'date', how = 'left').sort_values(by='date')
    combined_data_interpolated = monthly_df.merge(interpolated_quarterly_df, on = 'date', how = 'left').sort_values(by='date')

    combined_data_duplicated.to_csv(Path(output_path)/'combined_total_household_data_duplicate.csv', index = False)
    combined_data_interpolated.to_csv(Path(output_path)/'combined_total_household_data_interpolate.csv', index = False)


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