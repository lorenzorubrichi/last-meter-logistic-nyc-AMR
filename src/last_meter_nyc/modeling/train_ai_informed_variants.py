from __future__ import annotations

import argparse
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

from amr_last_meter_sim_ai_penalty import OUTPUT_AMR_STATS_XLSX, run_simulation as run_amr_simulation, save_results as save_amr_results
from building_scene_preview_ai_penalty import AI_FEATURES_XLSX, OUTPUT_STATS_XLSX as CAR_STATS_XLSX, run_simulation as run_car_simulation, save_results as save_car_results
from last_meter_nyc.paths import MODELS_DIR, STREETVIEW_OUTPUT_DIR


BASE_DIR = MODELS_DIR.parent
MODEL_DIR = MODELS_DIR
SHARED_DIR = MODEL_DIR / "ai_penalty_shared"
NO_AI_DIR = MODEL_DIR / "ai_penalty_no_ai_features"
WITH_AI_DIR = MODEL_DIR / "ai_penalty_with_ai_features"
AI_FEATURES_CSV = STREETVIEW_OUTPUT_DIR / "streetview_visual_features_api.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

DEFAULT_N_BUILDINGS = 10000
DEFAULT_N_RUNS = 100

CAR_BASE_FEATURES = [
    "a1_Floors_norm",
    "CurbCrowdingPenalty",
    "a2_RoadToDeliveryDistance_norm",
    "ShapePenalty_norm",
    "BuildingTypePenalty_norm",
]

AMR_BASE_FEATURES = [
    "a1_Floors_norm",
    "a2_RoadToDeliveryDistance_norm",
    "ShapePenalty_norm",
    "BuildingTypePenalty_norm",
    "b1_Population_norm",
    "b2_PedestrianPresence_norm",
]

AI_FEATURES = [
    "ai_stairs_present",
    "ai_gate_present",
    "ai_ramp_present",
    "ai_access_barrier_mean",
]

CAR_SEGMENTS = [
    ("car_all_buildings", "All buildings", None),
    ("car_amr_feasible_only", "AMR-feasible buildings only", True),
    ("car_amr_not_feasible_only", "AMR-not-feasible buildings only", False),
]

AMR_SEGMENTS = [
    ("amr_amr_feasible_only", "AMR-feasible buildings only", True),
]


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


def to_bool_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )


def load_amr_reach_lookup() -> pd.DataFrame:
    if not AI_FEATURES_CSV.exists():
        raise FileNotFoundError(f"Missing AI features CSV: {AI_FEATURES_CSV}")
    df = pd.read_csv(AI_FEATURES_CSV, dtype=str, usecols=["bin", "amr_can_reach_door"])
    df["bin"] = pd.to_numeric(df["bin"], errors="coerce").astype("Int64")
    df["amr_can_reach_door"] = to_bool_series(df["amr_can_reach_door"])
    df = df.dropna(subset=["bin", "amr_can_reach_door"]).drop_duplicates(subset=["bin"], keep="last")
    return df


def attach_amr_reach(df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["bin"] = pd.to_numeric(out["bin"], errors="coerce").astype("Int64")
    out = out.merge(lookup, on="bin", how="left")
    return out


def clean_training_frame(training_df: pd.DataFrame, feature_cols: list[str], target_col: str) -> pd.DataFrame:
    required_cols = ["bin", target_col, *feature_cols, "amr_can_reach_door"]
    work = training_df[required_cols].copy()
    for col in [target_col, *feature_cols]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["bin", target_col]).copy()
    work = work[work[target_col] > 0].copy()
    return work


def train_best_model(
    training_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    output_column: str,
    segment_key: str,
    segment_label: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
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
    test_prediction_tables: list[pd.DataFrame] = []
    best_name = None
    best_score = None

    test_meta = work.loc[X_test.index, ["bin", "amr_can_reach_door"]].copy()

    for model_name, pipeline in build_regression_models().items():
        pipeline.fit(X_train, y_train)
        pred_test = pipeline.predict(X_test)
        row = {
            "segment_key": segment_key,
            "segment_label": segment_label,
            "output_column": output_column,
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
        preds_df = test_meta.copy()
        preds_df["output_column"] = output_column
        preds_df["segment_key"] = segment_key
        preds_df["segment_label"] = segment_label
        preds_df["model"] = model_name
        preds_df["actual_time_s"] = y_test.values
        preds_df["predicted_time_s"] = pred_test
        preds_df["abs_error_s"] = np.abs(preds_df["predicted_time_s"] - preds_df["actual_time_s"])
        test_prediction_tables.append(preds_df)

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

    full_predictions = final_pipeline.predict(X)
    modeled_times_df = work[["bin", "amr_can_reach_door"]].copy()
    modeled_times_df["output_column"] = output_column
    modeled_times_df["segment_key"] = segment_key
    modeled_times_df["segment_label"] = segment_label
    modeled_times_df["chosen_model"] = str(best_name)
    modeled_times_df["actual_time_s"] = y.values
    modeled_times_df["predicted_time_s"] = full_predictions
    modeled_times_df["abs_error_s"] = np.abs(modeled_times_df["predicted_time_s"] - modeled_times_df["actual_time_s"])
    modeled_times_df["row_scope"] = "full_segment"

    best_test_predictions_df = pd.concat(test_prediction_tables, ignore_index=True)
    best_test_predictions_df = best_test_predictions_df[best_test_predictions_df["model"] == str(best_name)].copy()
    best_test_predictions_df["row_scope"] = "test_only"

    bundle = {
        "model_name": str(best_name),
        "target_col": target_col,
        "output_column": output_column,
        "segment_key": segment_key,
        "segment_label": segment_label,
        "feature_cols": list(feature_cols),
        "pipeline": final_pipeline,
        "metrics": metrics_df.to_dict(orient="records"),
        "n_rows": int(len(work)),
    }
    modeled_times_export = pd.concat([modeled_times_df, best_test_predictions_df], ignore_index=True, sort=False)
    return bundle, metrics_df, modeled_times_export


def save_model_bundle(bundle: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)


def build_variant_specs(include_ai: bool) -> dict[str, dict]:
    car_features = CAR_BASE_FEATURES + (AI_FEATURES if include_ai else [])
    amr_features = AMR_BASE_FEATURES + (AI_FEATURES if include_ai else [])
    return {
        "car_last_meter_mean_s": {
            "features": car_features,
            "training_target": "base_time_mean_s",
            "bundle_name": "car_model.joblib",
            "segments": CAR_SEGMENTS,
        },
        "amr_last_meter_mean_s": {
            "features": amr_features,
            "training_target": "base_amr_time_mean_s",
            "bundle_name": "amr_model.joblib",
            "segments": AMR_SEGMENTS,
        },
    }


def filter_segment(training_df: pd.DataFrame, feasible_flag: bool | None) -> pd.DataFrame:
    if feasible_flag is None:
        return training_df.copy()
    return training_df[training_df["amr_can_reach_door"] == feasible_flag].copy()


def train_variant(car_df: pd.DataFrame, amr_df: pd.DataFrame, include_ai: bool, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = build_variant_specs(include_ai=include_ai)
    metric_tables: list[pd.DataFrame] = []
    chosen_rows: list[dict] = []
    modeled_time_tables: list[pd.DataFrame] = []
    trained_dir = output_dir / "trained_models"

    for output_name, spec in specs.items():
        source_df = car_df if output_name.startswith("car_") else amr_df
        for segment_key, segment_label, feasible_flag in spec["segments"]:
            segment_df = filter_segment(source_df, feasible_flag)
            if segment_df.empty:
                print(f"[SKIP] Variant={'with_ai' if include_ai else 'no_ai'} target={output_name} segment={segment_key} (0 rows)")
                continue

            print(
                f"[TRAIN] Variant={'with_ai' if include_ai else 'no_ai'} "
                f"target={output_name} segment={segment_key} rows={len(segment_df)}"
            )
            bundle, metrics_df, modeled_times_df = train_best_model(
                training_df=segment_df,
                feature_cols=spec["features"],
                target_col=spec["training_target"],
                output_column=output_name,
                segment_key=segment_key,
                segment_label=segment_label,
            )
            bundle_name = spec["bundle_name"].replace(".joblib", f"__{segment_key}.joblib")
            save_model_bundle(bundle, trained_dir / bundle_name)
            chosen_rows.append(
                {
                    "segment_key": segment_key,
                    "segment_label": segment_label,
                    "output_column": output_name,
                    "training_target": spec["training_target"],
                    "chosen_model": bundle["model_name"],
                    "bundle_file": bundle_name,
                    "n_rows": bundle["n_rows"],
                    "include_ai_features": include_ai,
                    "amr_feasible_filter": feasible_flag,
                }
            )
            metric_tables.append(metrics_df.assign(include_ai_features=include_ai, amr_feasible_filter=feasible_flag))
            modeled_times_df["chosen_model"] = bundle["model_name"]
            modeled_times_df["include_ai_features"] = include_ai
            modeled_times_df["amr_feasible_filter"] = feasible_flag
            modeled_time_tables.append(modeled_times_df)

    chosen_df = pd.DataFrame(chosen_rows)
    metrics_df = pd.concat(metric_tables, ignore_index=True) if metric_tables else pd.DataFrame()
    modeled_times_export = pd.concat(modeled_time_tables, ignore_index=True) if modeled_time_tables else pd.DataFrame()
    report_path = output_dir / "trained_last_meter_models.xlsx"
    params_df = pd.DataFrame(
        [
            {"parameter": "include_ai_features", "value": include_ai},
            {"parameter": "car_stats_xlsx", "value": str(CAR_STATS_XLSX)},
            {"parameter": "amr_stats_xlsx", "value": str(OUTPUT_AMR_STATS_XLSX)},
            {"parameter": "ai_features_xlsx", "value": str(AI_FEATURES_XLSX)},
            {"parameter": "trained_model_dir", "value": str(trained_dir)},
            {"parameter": "segmentation_rule", "value": "car: all / amr feasible / amr not feasible; amr: feasible only"},
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        chosen_df.to_excel(writer, index=False, sheet_name="chosen_models")
        metrics_df.to_excel(writer, index=False, sheet_name="model_metrics")
        modeled_times_export.to_excel(writer, index=False, sheet_name="modeled_times")
        params_df.to_excel(writer, index=False, sheet_name="training_parameters")
    return chosen_df, metrics_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AI-penalized simulations once, then train two regression variants with and without AI predictors."
    )
    parser.add_argument("--n-buildings", type=int, default=DEFAULT_N_BUILDINGS, help="Number of buildings to simulate.")
    parser.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS, help="Monte Carlo runs per building.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[DATA] AI-covered raw+AI subset will be saved in: {SHARED_DIR}")
    print(f"[SIM] Car AI-penalized simulation on {args.n_buildings} buildings, {args.n_runs} runs...")
    car_df = run_car_simulation(
        n_buildings=max(1, args.n_buildings),
        n_runs=max(1, args.n_runs),
        seed=args.seed,
    )
    save_car_results(car_df)

    print(f"[SIM] AMR AI-penalized simulation on {args.n_buildings} buildings, {args.n_runs} runs...")
    amr_df = run_amr_simulation(
        n_buildings=max(1, args.n_buildings),
        n_runs=max(1, args.n_runs),
        seed=args.seed,
    )
    save_amr_results(amr_df)

    reach_lookup = load_amr_reach_lookup()
    car_df = attach_amr_reach(car_df, reach_lookup)
    amr_df = attach_amr_reach(amr_df, reach_lookup)
    car_df = car_df.dropna(subset=["amr_can_reach_door"]).copy()
    amr_df = amr_df.dropna(subset=["amr_can_reach_door"]).copy()

    train_variant(car_df, amr_df, include_ai=False, output_dir=NO_AI_DIR)
    train_variant(car_df, amr_df, include_ai=True, output_dir=WITH_AI_DIR)

    print(f"[DONE] Shared simulation outputs: {SHARED_DIR}")
    print(f"[DONE] Regression variant without AI features: {NO_AI_DIR}")
    print(f"[DONE] Regression variant with AI features: {WITH_AI_DIR}")


if __name__ == "__main__":
    main()
