from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from last_meter_nyc.paths import MODELS_DIR

BASE_DIR = MODELS_DIR.parent
MODEL_DIR = MODELS_DIR
TRAINED_MODEL_DIR = MODEL_DIR / "trained_models"

RANDOM_STATE = 42
TEST_SIZE = 0.2

CAR_MODEL_FEATURES = [
    "a1_Floors_norm",
    "CurbCrowdingPenalty",
    "a2_RoadToDeliveryDistance_norm",
    "ShapePenalty_norm",
    "BuildingTypePenalty_norm",
]

AMR_MODEL_FEATURES = [
    "a1_Floors_norm",
    "a2_RoadToDeliveryDistance_norm",
    "ShapePenalty_norm",
    "BuildingTypePenalty_norm",
    "b1_Population_norm",
    "b2_PedestrianPresence_norm",
]

MODEL_TARGET_SPECS = {
    "car_last_meter_mean_s": {
        "features": CAR_MODEL_FEATURES,
        "training_target": "base_time_mean_s",
        "output_column": "car_last_meter_mean_s",
        "source_column": "car_last_meter_source",
        "bundle_name": "car_model.joblib",
    },
    "amr_last_meter_mean_s": {
        "features": AMR_MODEL_FEATURES,
        "training_target": "base_amr_time_mean_s",
        "output_column": "amr_last_meter_mean_s",
        "source_column": "amr_last_meter_source",
        "bundle_name": "amr_model.joblib",
    },
}


def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_regression_models() -> dict[str, Pipeline]:
    return {
        "linear_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", LinearRegression()),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        random_state=RANDOM_STATE,
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=3,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def clean_training_frame(training_df: pd.DataFrame, feature_cols: list[str], target_col: str) -> pd.DataFrame:
    required_cols = ["bin", target_col, *feature_cols]
    work = training_df[required_cols].copy()
    for col in [target_col, *feature_cols]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["bin", target_col]).copy()
    work = work[work[target_col] > 0].copy()
    return work


def train_best_model(training_df: pd.DataFrame, feature_cols: list[str], target_col: str) -> tuple[dict, pd.DataFrame]:
    work = clean_training_frame(training_df, feature_cols, target_col)
    X = work[feature_cols].copy()
    y = work[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    metric_rows = []
    best_name = None
    best_score = None

    for model_name, pipeline in build_regression_models().items():
        pipeline.fit(X_train, y_train)
        pred_test = pipeline.predict(X_test)
        row = {
            "target": target_col,
            "model": model_name,
            "n_rows": len(work),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "rmse_test_s": rmse(y_test, pred_test),
            "mae_test_s": float(mean_absolute_error(y_test, pred_test)),
            "r2_test": float(r2_score(y_test, pred_test)),
        }
        metric_rows.append(row)
        score = (row["r2_test"], -row["rmse_test_s"])
        if best_score is None or score > best_score:
            best_score = score
            best_name = model_name

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        by=["r2_test", "rmse_test_s"],
        ascending=[False, True],
    ).reset_index(drop=True)

    final_pipeline = build_regression_models()[str(best_name)]
    final_pipeline.fit(X, y)

    bundle = {
        "model_name": str(best_name),
        "target_col": target_col,
        "feature_cols": list(feature_cols),
        "pipeline": final_pipeline,
        "metrics": metrics_df.to_dict(orient="records"),
        "n_rows": int(len(work)),
    }
    return bundle, metrics_df


def save_model_bundle(bundle: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)


def load_model_bundle(path: Path) -> dict:
    if path.exists():
        return joblib.load(path)

    fallback_map = {
        "car_model.joblib": "car_base_model.joblib",
        "amr_model.joblib": "amr_base_model.joblib",
    }
    fallback_name = fallback_map.get(path.name)
    if fallback_name is not None:
        fallback_path = path.with_name(fallback_name)
        if fallback_path.exists():
            return joblib.load(fallback_path)

    raise FileNotFoundError(f"Missing trained model bundle: {path}")


def predict_with_bundle(bundle: dict, prediction_df: pd.DataFrame) -> pd.Series:
    feature_cols = list(bundle["feature_cols"])
    X = prediction_df.reindex(columns=feature_cols).copy()
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    preds = bundle["pipeline"].predict(X)
    return pd.Series(preds, index=prediction_df.index, dtype=float)
