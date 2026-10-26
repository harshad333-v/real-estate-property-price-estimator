
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.data_split import load_and_split_data
from src.model_pipeline import build_preprocessor


RANDOM_STATE = 42
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")


def build_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        ),
    }


def evaluate_model(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def print_error_diagnostics(y_true, y_pred):
    """Print absolute validation prediction errors in lakhs."""
    errors = np.abs(
        np.asarray(y_true).reshape(-1)
        - np.asarray(y_pred).reshape(-1)
    )

    print("\nValidation error diagnostics:")
    print(f"Median absolute error: {np.median(errors):,.2f} lakhs")
    print(
        "90th percentile absolute error: "
        f"{np.percentile(errors, 90):,.2f} lakhs"
    )
    print(
        "95th percentile absolute error: "
        f"{np.percentile(errors, 95):,.2f} lakhs"
    )
    print(f"Maximum absolute error: {np.max(errors):,.2f} lakhs")


def save_worst_predictions(X_valid, y_true, y_pred):
    """Save validation errors with their original property features."""
    validation_errors = pd.DataFrame(
        {
            "Actual_Price_Lakhs": np.asarray(y_true).reshape(-1),
            "Predicted_Price_Lakhs": np.asarray(y_pred).reshape(-1),
        },
        index=X_valid.index,
    )

    validation_errors["Absolute_Error_Lakhs"] = np.abs(
        validation_errors["Actual_Price_Lakhs"]
        - validation_errors["Predicted_Price_Lakhs"]
    )

    # Join using original row indices to retain the correct property.
    validation_errors = validation_errors.join(
        X_valid,
        how="left",
        validate="one_to_one",
    )

    validation_errors = validation_errors.sort_values(
        "Absolute_Error_Lakhs",
        ascending=False,
    )

    error_report_path = REPORT_DIR / "random_forest_validation_errors.csv"
    validation_errors.to_csv(
        error_report_path,
        index_label="Original_Row_Index",
    )

    print("\nTop 10 Random Forest prediction errors:")
    print(validation_errors.head(10).to_string())

    print(f"\nSaved detailed errors: {error_report_path}")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_valid, y_train, y_valid = load_and_split_data()

    results = []
    fitted_models = {}

    for name, estimator in build_models().items():
        print(f"\n{'=' * 60}")
        print(f"Training {name}...")
        start = time.perf_counter()

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_valid)

        metrics = evaluate_model(y_valid, predictions)
        elapsed = time.perf_counter() - start

        row = {
            "Model": name,
            **metrics,
            "Training_And_Evaluation_Seconds": round(elapsed, 2),
        }

        results.append(row)
        fitted_models[name] = pipeline

        print("\nValidation metrics:")
        print(pd.Series(row).to_string())

        print_error_diagnostics(y_valid, predictions)

        if name == "Random Forest":
            save_worst_predictions(X_valid, y_valid, predictions)

    results_df = (
        pd.DataFrame(results)
        .sort_values("MAE")
        .reset_index(drop=True)
    )

    results_path = REPORT_DIR / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    print(f"\n{'=' * 60}")
    print("MODEL COMPARISON (sorted by MAE)")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["Model"]
    best_pipeline = fitted_models[best_name]

    model_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(best_pipeline, model_path)

    print(f"\nBest model by validation MAE: {best_name}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {results_path}")


if __name__ == "__main__":
    main()
