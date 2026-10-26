from datetime import datetime, timezone
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.data_split import load_and_split_data
from src.model_pipeline import build_preprocessor


RANDOM_STATE = 42
DATA_PATH = Path("data/raw/train.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
PRICE_BANDS = [-np.inf, 10, 30, 75, 150, np.inf]
PRICE_BAND_LABELS = ["<=10L", "10-30L", "30-75L", "75-150L", ">150L"]


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
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def build_price_segment_report(y_true, y_pred):
    """Evaluate errors by actual-price band to expose uneven model performance."""
    frame = pd.DataFrame({
        "actual_price_lakhs": np.asarray(y_true, dtype=float).reshape(-1),
        "predicted_price_lakhs": np.asarray(y_pred, dtype=float).reshape(-1),
    })
    frame["price_band"] = pd.cut(
        frame["actual_price_lakhs"],
        bins=PRICE_BANDS,
        labels=PRICE_BAND_LABELS,
        include_lowest=True,
        right=True,
    )

    rows = []
    for band in PRICE_BAND_LABELS:
        subset = frame.loc[frame["price_band"] == band]
        if subset.empty:
            continue
        row = {
            "actual_price_band": band,
            "samples": int(len(subset)),
            "MAE_lakhs": float(mean_absolute_error(
                subset["actual_price_lakhs"], subset["predicted_price_lakhs"]
            )),
            "RMSE_lakhs": float(np.sqrt(mean_squared_error(
                subset["actual_price_lakhs"], subset["predicted_price_lakhs"]
            ))),
        }
        row["R2"] = (
            float(r2_score(subset["actual_price_lakhs"], subset["predicted_price_lakhs"]))
            if len(subset) >= 2 else None
        )
        rows.append(row)
    return rows


def print_error_diagnostics(y_true, y_pred):
    """Print absolute validation prediction errors in lakhs."""
    errors = np.abs(np.asarray(y_true).reshape(-1) - np.asarray(y_pred).reshape(-1))
    print("\nValidation error diagnostics:")
    print(f"Median absolute error: {np.median(errors):,.2f} lakhs")
    print(f"90th percentile absolute error: {np.percentile(errors, 90):,.2f} lakhs")
    print(f"95th percentile absolute error: {np.percentile(errors, 95):,.2f} lakhs")
    print(f"Maximum absolute error: {np.max(errors):,.2f} lakhs")


def save_worst_predictions(X_valid, y_true, y_pred):
    """Save validation errors with their original property features."""
    validation_errors = pd.DataFrame({
        "Actual_Price_Lakhs": np.asarray(y_true).reshape(-1),
        "Predicted_Price_Lakhs": np.asarray(y_pred).reshape(-1),
    }, index=X_valid.index)
    validation_errors["Absolute_Error_Lakhs"] = np.abs(
        validation_errors["Actual_Price_Lakhs"] - validation_errors["Predicted_Price_Lakhs"]
    )
    validation_errors = validation_errors.join(X_valid, how="left", validate="one_to_one")
    validation_errors = validation_errors.sort_values("Absolute_Error_Lakhs", ascending=False)
    error_report_path = REPORT_DIR / "random_forest_validation_errors.csv"
    validation_errors.to_csv(error_report_path, index_label="Original_Row_Index")
    print("\nTop 10 Random Forest prediction errors:")
    print(validation_errors.head(10).to_string())
    print(f"\nSaved detailed errors: {error_report_path}")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_valid, y_train, y_valid = load_and_split_data(
        train_path=str(DATA_PATH), test_size=0.20, random_state=RANDOM_STATE
    )

    results = []
    fitted_models = {}
    validation_predictions = {}

    for name, estimator in build_models().items():
        print(f"\n{'=' * 60}\nTraining {name}...")
        start = time.perf_counter()
        pipeline = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ])
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
        validation_predictions[name] = predictions
        print("\nValidation metrics:")
        print(pd.Series(row).to_string())
        print_error_diagnostics(y_valid, predictions)
        if name == "Random Forest":
            save_worst_predictions(X_valid, y_valid, predictions)

    results_df = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
    results_path = REPORT_DIR / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)

    best_name = str(results_df.iloc[0]["Model"])
    best_pipeline = fitted_models[best_name]
    best_predictions = validation_predictions[best_name]
    model_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(best_pipeline, model_path)

    metrics_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATA_PATH),
        "target": "TARGET(PRICE_IN_LACS)",
        "target_unit": "INR lakhs",
        "split": {
            "strategy": "random holdout",
            "validation_fraction": 0.20,
            "random_state": RANDOM_STATE,
            "training_rows": int(len(X_train)),
            "validation_rows": int(len(X_valid)),
        },
        "selection_metric": "MAE (lower is better)",
        "selected_model": best_name,
        "models": results,
        "selected_model_price_segment_metrics": build_price_segment_report(
            y_valid, best_predictions
        ),
    }
    (REPORT_DIR / "model_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )
    pd.DataFrame(metrics_payload["selected_model_price_segment_metrics"]).to_csv(
        REPORT_DIR / "price_segment_metrics.csv", index=False
    )

    model_card = {
        "model_name": best_name,
        "artifact_path": str(model_path),
        "generated_at_utc": metrics_payload["generated_at_utc"],
        "intended_use": "Educational estimates of Indian residential listing prices.",
        "not_intended_use": [
            "Certified property valuation",
            "Loan underwriting without independent validation",
            "Replacing professional or locality-specific appraisal",
        ],
        "training_data": {
            "path": str(DATA_PATH),
            "target": "TARGET(PRICE_IN_LACS)",
            "training_rows": int(len(X_train)),
            "validation_rows": int(len(X_valid)),
        },
        "validation_metrics": next(item for item in results if item["Model"] == best_name),
        "known_limitations": [
            "Random holdout validation is not a substitute for temporal or geographic external validation.",
            "Extreme property prices may dominate RMSE.",
            "Coordinate column interpretation must be verified against the source dataset.",
            "Predictions may be unreliable for rare localities or out-of-distribution properties.",
        ],
    }
    (REPORT_DIR / "model_card.json").write_text(
        json.dumps(model_card, indent=2), encoding="utf-8"
    )

    print(f"\n{'=' * 60}\nMODEL COMPARISON (sorted by MAE)")
    print(results_df.to_string(index=False))
    print(f"\nBest model by validation MAE: {best_name}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {results_path}, reports/model_metrics.json")
    print("Saved reports/model_card.json and reports/price_segment_metrics.csv")


if __name__ == "__main__":
    main()
