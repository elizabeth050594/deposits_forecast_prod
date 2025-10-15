
from src.feature_engineering import prepare_features
from src.train_model import build_forecaster
from sktime.split import temporal_train_test_split
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.forecasting.model_evaluation import evaluate
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error, mean_squared_error
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import os


def plot_forecast_results(results_df, title="Forecast vs Actual", save_path=None):
    """
    Plots actual vs. predicted values along with RMSE, MAPE, and Max Error.

    Parameters:
    - results_df (pd.DataFrame): DataFrame with columns ["y_true", "y_pred"] and a datetime index
    - title (str): Plot title
    - save_path (str or Path, optional): Path to save the plot. If None, just displays it.

    Returns:
    - tuple: (RMSE, MAPE, Max Error) as floats
    """

    y_true = results_df["y_true"]
    y_pred = results_df["y_pred"]

    # Calculate error metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    max_error = np.max(np.abs(y_true - y_pred))

    # Create plot
    plt.figure(figsize=(15, 5))
    plt.plot(results_df.index, y_true, label="Actual", marker='o', linestyle='--', color='black')
    plt.plot(results_df.index, y_pred, label=f"Predicted RMSE={rmse:.1f}, MAPE={mape:.2f}%, Max Err={max_error:.1f}",
             marker='o', color='tab:green')
    
    # Annotate forecast errors above or below the points
    for i in range(len(results_df)):
        date = results_df.index[i]
        actual = y_true.iloc[i]
        predicted = y_pred.iloc[i]
        diff = actual - predicted
        plt.text(date, max(actual, predicted) + 1000, f"{diff:+.1f}", color='red', fontsize=12,
                 ha='center', va='bottom' if diff > 0 else 'top')

    # Axis and plot settings
    plt.xlabel("Date")
    plt.ylabel("Household Deposits (NZDm)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save or show
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

    return rmse, mape, max_error

def run_residual_boosting_pipeline(df, target='household_deposits', test_size=12, window_length=160, h=2, sp=12, n_estimators=200, learning_rate=0.01, max_depth=4, min_samples_leaf = 1, subsample = 1, plot_save_path=None):

    """
    End-to-end pipeline for evaluating the Residual Boosting Forecasting model
    using AutoETS as base model and GradientBoosting for residual correction.

    Parameters:
    - df (pd.DataFrame): Processed input DataFrame with features including 'date', target and predictors as well as engineered features
    - target (str): Target variable name
    - test_size (int): Number of months to reserve for test set (must be >= h)
    - window_length (int): Number of past points used in GBT residual correction
    - h (int): Forecast horizon (e.g. 2 = 2-month ahead, from the data perpsective)
    - sp (int): Seasonal period (set to 12 for the deposits data as strong yearly seasonaility observed)
    - n_estimators, learning_rate, max_depth, min_samples_leaf, subsample: parameters for GBT
    - plot_save_path (str or Path): Optional path to save the forecast plot

    Returns:
    - plots the actual versus prediction (plot not returned)
    - tuple: (RMSE, MAPE, Max Error) from forecast evaluation
    """

    # Train-test split
    y_train, y_test = temporal_train_test_split(df, test_size=test_size)

    # Build forecasting pipeline
    forecaster = build_forecaster(sp, n_estimators, learning_rate, max_depth, window_length, min_samples_leaf, subsample)

    # Define expanding CV
    splitter = ExpandingWindowSplitter(
        initial_window=len(y_train) - h + 1,
        step_length=1,
        fh=[h]
    )

    # Evaluate with cross-validation
    cv_results = evaluate(
        forecaster=forecaster,
        y=df[target],
        X=df.drop(columns=[target]),
        cv=splitter,
        strategy="refit",
        return_data=True,
    )

    # Extract predictions
    results = pd.DataFrame({
        "y_true": [s.values[0] for s in cv_results["y_test"]],
        "y_pred": [s.values[0] for s in cv_results["y_pred"]]
    })
    results.index = y_test.index
    results.index = results.index.to_timestamp()

    # Plot results
    rmse, mape, max_error = plot_forecast_results(results, title="AutoETS + Residual Boosting Forecast", save_path = plot_save_path)

    return rmse, mape, max_error, results 

def evaluate_models_on_test(df, h, param_dict, target, test_size=12, output_path=None):
    """
    Wrapper for running run residual boosting pipeline. 

    Parameters:
    - df (pd.DataFrame): Input data with features, including target and engineered features
    - h (int): Forecast horizon
    - param_dict (dict): Dictionary of model hyperparameters, expected keys:
        'window_length', 'sp', 'n_estimators', 'learning_rate', 'max_depth',
        'min_samples_leaf', 'subsample'
    - target (str): Target variable name
    - test_size (int): Number of months reserved for test set
    - output_path (str or Path, optional): If provided, save the forecast plot and results CSV here

    Returns:
    - tuple: (rmse, mape, max_error, results_df)
    """

    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / f'model_evaluation_results_h{h}'
        plot_path = output_path.with_suffix('.png')
        csv_path = output_path.with_suffix('.csv')

    # Run forecasting pipeline
    rmse, mape, max_error, results = run_residual_boosting_pipeline(
        df=df,
        target=target,
        test_size=test_size,
        window_length=param_dict['window_length'],
        h=h,
        sp=param_dict['sp'],
        n_estimators=param_dict['n_estimators'],
        learning_rate=param_dict['learning_rate'],
        max_depth=param_dict['max_depth'],
        min_samples_leaf=param_dict['min_samples_leaf'],
        subsample=param_dict['subsample'],
        plot_save_path=plot_path
    )

    # Save results to CSV if path provided
    if csv_path:
        results.to_csv(csv_path)
        print(f"Saved forecast results to {csv_path}")

    return rmse, mape, max_error, results

    


