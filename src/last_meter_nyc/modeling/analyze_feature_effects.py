from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from datetime import datetime

import pandas as pd
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.model_selection import train_test_split

from amr_last_meter_sim import OUTPUT_AMR_STATS_XLSX
from amr_last_meter_sim_ai_penalty import OUTPUT_AMR_STATS_XLSX as AI_OUTPUT_AMR_STATS_XLSX
from building_scene_preview import OUTPUT_STATS_XLSX as CAR_STATS_XLSX
from building_scene_preview_ai_penalty import OUTPUT_STATS_XLSX as AI_CAR_STATS_XLSX
from last_meter_model_utils import (
    MODEL_TARGET_SPECS,
    RANDOM_STATE,
    TEST_SIZE,
    build_regression_models,
    clean_training_frame,
)
from train_last_meter_models_ai_variants import AI_FEATURES, AMR_BASE_FEATURES, CAR_BASE_FEATURES
from last_meter_nyc.paths import MODELS_DIR


BASE_DIR = MODELS_DIR.parent
MODEL_DIR = MODELS_DIR
OUTPUT_DIR = MODEL_DIR / "feature_effect_analysis"
OUTPUT_XLSX = OUTPUT_DIR / "feature_effect_analysis.xlsx"
FALLBACK_OUTPUT_XLSX = OUTPUT_DIR / "feature_effect_analysis_new.xlsx"

TRAINING_SOURCES = {
    "car_last_meter_mean_s": {
        "path": CAR_STATS_XLSX,
        "sheet": "car_last_meter",
    },
    "amr_last_meter_mean_s": {
        "path": OUTPUT_AMR_STATS_XLSX,
        "sheet": "amr_last_meter",
    },
}

VARIANT_CONFIG = {
    "baseline": {
        "output_dir": OUTPUT_DIR,
        "output_xlsx": OUTPUT_XLSX,
        "fallback_output_xlsx": FALLBACK_OUTPUT_XLSX,
        "training_sources": TRAINING_SOURCES,
        "model_specs": MODEL_TARGET_SPECS,
    },
    "ai_no_features": {
        "output_dir": MODEL_DIR / "ai_penalty_no_ai_features",
        "output_xlsx": MODEL_DIR / "ai_penalty_no_ai_features" / "feature_effect_analysis.xlsx",
        "fallback_output_xlsx": MODEL_DIR / "ai_penalty_no_ai_features" / "feature_effect_analysis_new.xlsx",
        "training_sources": {
            "car_last_meter_mean_s": {"path": AI_CAR_STATS_XLSX, "sheet": "car_last_meter"},
            "amr_last_meter_mean_s": {"path": AI_OUTPUT_AMR_STATS_XLSX, "sheet": "amr_last_meter"},
        },
        "model_specs": {
            "car_last_meter_mean_s": {"features": list(CAR_BASE_FEATURES), "training_target": "base_time_mean_s"},
            "amr_last_meter_mean_s": {"features": list(AMR_BASE_FEATURES), "training_target": "base_amr_time_mean_s"},
        },
    },
    "ai_with_features": {
        "output_dir": MODEL_DIR / "ai_penalty_with_ai_features",
        "output_xlsx": MODEL_DIR / "ai_penalty_with_ai_features" / "feature_effect_analysis.xlsx",
        "fallback_output_xlsx": MODEL_DIR / "ai_penalty_with_ai_features" / "feature_effect_analysis_new.xlsx",
        "training_sources": {
            "car_last_meter_mean_s": {"path": AI_CAR_STATS_XLSX, "sheet": "car_last_meter"},
            "amr_last_meter_mean_s": {"path": AI_OUTPUT_AMR_STATS_XLSX, "sheet": "amr_last_meter"},
        },
        "model_specs": {
            "car_last_meter_mean_s": {"features": list(CAR_BASE_FEATURES) + list(AI_FEATURES), "training_target": "base_time_mean_s"},
            "amr_last_meter_mean_s": {"features": list(AMR_BASE_FEATURES) + list(AI_FEATURES), "training_target": "base_amr_time_mean_s"},
        },
    },
}

FRIENDLY_FEATURE_NAMES = {
    "a1_Floors_norm": "Normalized Floors",
    "CurbCrowdingPenalty": "Curb Crowding",
    "EntranceDistance_norm": "Entrance Distance",
    "ShapePenalty_norm": "Building Shape",
    "BuildingTypePenalty_norm": "Building Type",
    "b1_Population_norm": "Population Density",
    "b2_PedestrianPresence_norm": "Pedestrian Presence",
    "b3_UrbanActivity_norm": "Urban Activity",
    "ai_stairs_present": "AI Stairs",
    "ai_gate_present": "AI Gate",
    "ai_ramp_present": "AI Ramp",
    "ai_access_barrier_mean": "AI Barrier Mean",
}


def load_training_frame(output_column: str, training_sources: dict[str, dict]) -> pd.DataFrame:
    source = training_sources[output_column]
    return pd.read_excel(source["path"], sheet_name=source["sheet"])


def add_friendly_names(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["feature_label"] = work["feature"].map(FRIENDLY_FEATURE_NAMES).fillna(work["feature"])
    return work


def build_pdp_table(
    pipeline,
    X: pd.DataFrame,
    output_column: str,
    features: list[str],
) -> pd.DataFrame:
    pdp_rows: list[dict] = []
    for feature in features:
        pd_result = partial_dependence(
            pipeline,
            X,
            features=[feature],
            kind="average",
        )
        grid_values = pd_result["grid_values"][0]
        averages = pd_result["average"][0]
        for x_value, y_value in zip(grid_values, averages):
            pdp_rows.append(
                {
                    "output_column": output_column,
                    "feature": feature,
                    "feature_label": FRIENDLY_FEATURE_NAMES.get(feature, feature),
                    "feature_value": float(x_value),
                    "partial_dependence": float(y_value),
                }
            )
    return pd.DataFrame(pdp_rows)


def build_pairwise_pdp_table(
    pipeline,
    X: pd.DataFrame,
    output_column: str,
    feature_a: str,
    feature_b: str,
) -> pd.DataFrame:
    idx_a = list(X.columns).index(feature_a)
    idx_b = list(X.columns).index(feature_b)
    pd_result = partial_dependence(
        pipeline,
        X,
        features=[(idx_a, idx_b)],
        kind="average",
    )
    grid_a = pd_result["grid_values"][0]
    grid_b = pd_result["grid_values"][1]
    average_grid = pd_result["average"][0]

    rows: list[dict] = []
    for i, value_a in enumerate(grid_a):
        for j, value_b in enumerate(grid_b):
            rows.append(
                {
                    "output_column": output_column,
                    "feature_x": feature_a,
                    "feature_x_label": FRIENDLY_FEATURE_NAMES.get(feature_a, feature_a),
                    "feature_x_value": float(value_a),
                    "feature_y": feature_b,
                    "feature_y_label": FRIENDLY_FEATURE_NAMES.get(feature_b, feature_b),
                    "feature_y_value": float(value_b),
                    "partial_dependence": float(average_grid[i, j]),
                }
            )
    return pd.DataFrame(rows)


def analyze_target(
    output_column: str,
    spec: dict,
    training_sources: dict[str, dict],
    top_k: int,
    pdp_top_k: int,
    n_repeats: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], object, pd.DataFrame]:
    feature_cols = list(spec["features"])
    target_col = spec["training_target"]
    training_df = load_training_frame(output_column, training_sources)

    work = clean_training_frame(training_df, feature_cols, target_col)
    X = work[feature_cols].copy()
    y = work[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    gb_pipeline = build_regression_models()["gradient_boosting"]
    gb_pipeline.fit(X_train, y_train)

    gb_model = gb_pipeline.named_steps["model"]
    gb_importance_df = pd.DataFrame(
        {
            "output_column": output_column,
            "feature": feature_cols,
            "gb_importance": gb_model.feature_importances_,
        }
    ).sort_values("gb_importance", ascending=False, ignore_index=True)
    gb_importance_df = add_friendly_names(gb_importance_df)

    top_gb_features = gb_importance_df["feature"].head(top_k).tolist()

    perm = permutation_importance(
        gb_pipeline,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        scoring="r2",
        n_jobs=1,
    )
    permutation_df = pd.DataFrame(
        {
            "output_column": output_column,
            "feature": feature_cols,
            "permutation_importance_mean": perm.importances_mean,
            "permutation_importance_std": perm.importances_std,
            "selected_by_gb_top_k": [feature in top_gb_features for feature in feature_cols],
        }
    ).sort_values("permutation_importance_mean", ascending=False, ignore_index=True)
    permutation_df = add_friendly_names(permutation_df)

    confirmed_top = (
        permutation_df[permutation_df["feature"].isin(top_gb_features)]
        .sort_values("permutation_importance_mean", ascending=False)
        ["feature"]
        .head(pdp_top_k)
        .tolist()
    )
    if not confirmed_top:
        confirmed_top = gb_importance_df["feature"].head(min(pdp_top_k, len(feature_cols))).tolist()

    full_gb_pipeline = build_regression_models()["gradient_boosting"]
    full_gb_pipeline.fit(X, y)
    pdp_df = build_pdp_table(
        pipeline=full_gb_pipeline,
        X=X,
        output_column=output_column,
        features=confirmed_top,
    )

    return gb_importance_df, permutation_df, pdp_df, confirmed_top, full_gb_pipeline, X


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze feature effects for the last-meter models: "
            "Gradient Boosting feature importance, permutation importance, and PDPs."
        )
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top Gradient Boosting features to keep before permutation confirmation.",
    )
    parser.add_argument(
        "--pdp-top-k",
        type=int,
        default=3,
        help="Number of confirmed features to plot with partial dependence.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=20,
        help="Permutation importance repeats.",
    )
    parser.add_argument(
        "--variant",
        choices=list(VARIANT_CONFIG.keys()),
        default="baseline",
        help="Which experiment branch to analyze.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variant_cfg = VARIANT_CONFIG[args.variant]
    output_dir = variant_cfg["output_dir"]
    output_xlsx_default = variant_cfg["output_xlsx"]
    fallback_output_xlsx = variant_cfg["fallback_output_xlsx"]
    training_sources = variant_cfg["training_sources"]
    model_specs = variant_cfg["model_specs"]
    output_dir.mkdir(parents=True, exist_ok=True)

    gb_tables: list[pd.DataFrame] = []
    perm_tables: list[pd.DataFrame] = []
    pdp_tables: list[pd.DataFrame] = []
    pairwise_tables: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict] = []

    for output_column, spec in model_specs.items():
        gb_df, perm_df, pdp_df, confirmed_top, full_gb_pipeline, X_full = analyze_target(
            output_column=output_column,
            spec=spec,
            training_sources=training_sources,
            top_k=max(1, args.top_k),
            pdp_top_k=max(1, args.pdp_top_k),
            n_repeats=max(1, args.n_repeats),
        )
        gb_tables.append(gb_df)
        perm_tables.append(perm_df)
        pdp_tables.append(pdp_df)
        summary_rows.append(
            {
                "variant": args.variant,
                "output_column": output_column,
                "top_k_from_gradient_boosting": ", ".join(gb_df["feature_label"].head(args.top_k).tolist()),
                "confirmed_for_pdp": ", ".join(FRIENDLY_FEATURE_NAMES.get(col, col) for col in confirmed_top),
            }
        )
        print(f"[ANALYZE] {output_column}: PDP data computed for {len(confirmed_top)} features")

        if output_column == "car_last_meter_mean_s" and len(confirmed_top) >= 2:
            feature_a, feature_b = confirmed_top[:2]
            pairwise_tables["car_heatmap"] = build_pairwise_pdp_table(
                pipeline=full_gb_pipeline,
                X=X_full,
                output_column=output_column,
                feature_a=feature_a,
                feature_b=feature_b,
            )

        if output_column == "amr_last_meter_mean_s" and len(confirmed_top) >= 3:
            for idx, (feature_a, feature_b) in enumerate(combinations(confirmed_top[:3], 2), start=1):
                pairwise_tables[f"amr_heatmap_{idx}"] = build_pairwise_pdp_table(
                    pipeline=full_gb_pipeline,
                    X=X_full,
                    output_column=output_column,
                    feature_a=feature_a,
                    feature_b=feature_b,
                )

    summary_df = pd.DataFrame(summary_rows)
    all_gb_df = pd.concat(gb_tables, ignore_index=True)
    all_perm_df = pd.concat(perm_tables, ignore_index=True)
    all_pdp_df = pd.concat(pdp_tables, ignore_index=True)
    params_df = pd.DataFrame(
        [
            {"parameter": "top_k", "value": args.top_k},
            {"parameter": "pdp_top_k", "value": args.pdp_top_k},
            {"parameter": "n_repeats", "value": args.n_repeats},
            {"parameter": "random_state", "value": RANDOM_STATE},
            {"parameter": "test_size", "value": TEST_SIZE},
            {"parameter": "variant", "value": args.variant},
        ]
    )

    output_xlsx = output_xlsx_default
    try:
        writer = pd.ExcelWriter(output_xlsx, engine="openpyxl")
    except PermissionError:
        try:
            output_xlsx = fallback_output_xlsx
            writer = pd.ExcelWriter(output_xlsx, engine="openpyxl")
        except PermissionError:
            stamped_name = f"feature_effect_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            output_xlsx = output_dir / stamped_name
            writer = pd.ExcelWriter(output_xlsx, engine="openpyxl")

    with writer:
        summary_df.to_excel(writer, index=False, sheet_name="summary")
        all_gb_df.to_excel(writer, index=False, sheet_name="gb_importance")
        all_perm_df.to_excel(writer, index=False, sheet_name="permutation_importance")
        all_pdp_df.to_excel(writer, index=False, sheet_name="pdp_data")
        if "car_heatmap" in pairwise_tables:
            pairwise_tables["car_heatmap"].to_excel(writer, index=False, sheet_name="car_heatmap")
        if "amr_heatmap_1" in pairwise_tables:
            pairwise_tables["amr_heatmap_1"].to_excel(writer, index=False, sheet_name="amr_heatmap_1")
        if "amr_heatmap_2" in pairwise_tables:
            pairwise_tables["amr_heatmap_2"].to_excel(writer, index=False, sheet_name="amr_heatmap_2")
        if "amr_heatmap_3" in pairwise_tables:
            pairwise_tables["amr_heatmap_3"].to_excel(writer, index=False, sheet_name="amr_heatmap_3")
        params_df.to_excel(writer, index=False, sheet_name="parameters")

    print(f"[ANALYZE] Excel report saved: {output_xlsx}")


if __name__ == "__main__":
    main()
