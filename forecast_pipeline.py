import json
import pandas as pd
import mlflow
from pathlib import Path

from ../src.data import load_and_prepare_data
from ../src.forecasting import train_and_forecast

CONFIG_PATH = Path("configs/")

def run_all_models_forecast():
    all_forecasts = []
    for config_file in CONFIG_PATH.glob("*.json"):
        with open(config_file) as f:
            params = json.load(f)

        model_name = params["model_name"]
        target = params['target']
        horizon = params['horizon']

        print(f"Running pipeline for: {model_name}")

        df = load_and_prepare_data(
            path="data/processed/combined_total_household_data_interpolate.csv",
            target=target,
            params=params
        )

        # Start MLflow run to log each model separately
        with mlflow.start_run(run_name=model_name):
            forecast_df, model = train_and_forecast(df, h=horizon, param_dict=params, target=target)

            # Add horizon column and log forecast as artifact (CSV)
            forecast_df["horizon"] = horizon
            all_forecasts.append(forecast_df)

            # Log model parameters and artifacts
            mlflow.log_params(params)
            mlflow.sklearn.log_model(model, artifact_path="model")

            # Log forecast CSV artifact
            csv_path = f"forecast_{model_name}.csv"
            forecast_df.to_csv(csv_path)
            mlflow.log_artifact(csv_path)

    # Combine all forecasts for return/logging
    final_forecast_df = pd.concat(all_forecasts).sort_index().reset_index().rename(columns={"index": "date"})

    # Optionally save combined forecasts for easy access
    final_forecast_df.to_csv("final_forecasts.csv", index=False)
    mlflow.log_artifact("final_forecasts.csv")

    return final_forecast_df

if __name__ == "__main__":
    run_all_models_forecast()
