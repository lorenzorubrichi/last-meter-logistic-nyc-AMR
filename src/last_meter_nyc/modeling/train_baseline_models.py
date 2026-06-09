from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from amr_last_meter_sim import OUTPUT_AMR_STATS_XLSX, run_simulation as run_amr_simulation, save_results as save_amr_results
from building_scene_preview import OUTPUT_STATS_XLSX as CAR_STATS_XLSX, run_simulation as run_car_simulation, save_results as save_car_results
from last_meter_model_utils import MODEL_TARGET_SPECS, TRAINED_MODEL_DIR, save_model_bundle, train_best_model
from last_meter_nyc.paths import MODELS_DIR


BASE_DIR = MODELS_DIR.parent
MODEL_DIR = MODELS_DIR
OUTPUT_XLSX = MODEL_DIR / "trained_last_meter_models.xlsx"

DEFAULT_N_BUILDINGS = 10000
DEFAULT_N_RUNS = 100


def train_and_save_models(car_df: pd.DataFrame, amr_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_tables: list[pd.DataFrame] = []
    chosen_rows: list[dict] = []

    for output_name, spec in MODEL_TARGET_SPECS.items():
        training_df = car_df if output_name.startswith("car_") else amr_df
        print(f"[TRAIN] Training target: {output_name}")
        bundle, metrics_df = train_best_model(
            training_df=training_df,
            feature_cols=spec["features"],
            target_col=spec["training_target"],
        )
        save_model_bundle(bundle, TRAINED_MODEL_DIR / spec["bundle_name"])
        chosen_rows.append(
            {
                "output_column": output_name,
                "training_target": spec["training_target"],
                "chosen_model": bundle["model_name"],
                "bundle_file": spec["bundle_name"],
                "n_rows": bundle["n_rows"],
            }
        )
        metric_tables.append(metrics_df.assign(output_column=output_name))

    return pd.DataFrame(chosen_rows), pd.concat(metric_tables, ignore_index=True)


def write_training_report(chosen_df: pd.DataFrame, metrics_df: pd.DataFrame, args: argparse.Namespace) -> None:
    params_df = pd.DataFrame(
        [
            {"parameter": "n_buildings", "value": args.n_buildings},
            {"parameter": "n_runs", "value": args.n_runs},
            {"parameter": "seed", "value": args.seed},
            {"parameter": "car_stats_xlsx", "value": str(CAR_STATS_XLSX)},
            {"parameter": "amr_stats_xlsx", "value": str(OUTPUT_AMR_STATS_XLSX)},
            {"parameter": "trained_model_dir", "value": str(TRAINED_MODEL_DIR)},
        ]
    )

    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        chosen_df.to_excel(writer, index=False, sheet_name="chosen_models")
        metrics_df.to_excel(writer, index=False, sheet_name="model_metrics")
        params_df.to_excel(writer, index=False, sheet_name="training_parameters")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate one base car scenario and one base AMR scenario, then train/save reusable regression models."
    )
    parser.add_argument("--n-buildings", type=int, default=DEFAULT_N_BUILDINGS, help="Number of buildings to simulate.")
    parser.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS, help="Monte Carlo runs per building.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[TRAIN] Simulating car last-meter on {args.n_buildings} buildings, {args.n_runs} runs each...")
    car_df = run_car_simulation(
        n_buildings=max(1, args.n_buildings),
        n_runs=max(1, args.n_runs),
        seed=args.seed,
    )
    save_car_results(car_df)

    print(f"[TRAIN] Simulating AMR last-meter on {args.n_buildings} buildings, {args.n_runs} runs each...")
    amr_df = run_amr_simulation(
        n_buildings=max(1, args.n_buildings),
        n_runs=max(1, args.n_runs),
        seed=args.seed,
    )
    save_amr_results(amr_df)

    chosen_df, metrics_df = train_and_save_models(car_df, amr_df)
    write_training_report(chosen_df, metrics_df, args)
    print(f"[TRAIN] Report saved: {OUTPUT_XLSX}")
    print(f"[TRAIN] Model bundles saved in: {TRAINED_MODEL_DIR}")


if __name__ == "__main__":
    main()
