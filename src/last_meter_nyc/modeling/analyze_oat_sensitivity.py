from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import pandas as pd

import amr_last_meter_sim as amr
import building_scene_preview as car
import amr_last_meter_sim_ai_penalty as amr_ai
import building_scene_preview_ai_penalty as car_ai
from last_meter_nyc.paths import MODELS_DIR


BASE_DIR = MODELS_DIR.parent
MODEL_DIR = MODELS_DIR
OUTPUT_DIR = MODEL_DIR / "oat_sensitivity"
OUTPUT_XLSX = OUTPUT_DIR / "last_meter_oat_sensitivity.xlsx"
FALLBACK_OUTPUT_XLSX = OUTPUT_DIR / "last_meter_oat_sensitivity_new.xlsx"

VARIANT_OUTPUTS = {
    "baseline": {
        "output_dir": OUTPUT_DIR,
        "output_xlsx": OUTPUT_XLSX,
        "fallback_output_xlsx": FALLBACK_OUTPUT_XLSX,
    },
    "ai_no_features": {
        "output_dir": MODEL_DIR / "ai_penalty_no_ai_features",
        "output_xlsx": MODEL_DIR / "ai_penalty_no_ai_features" / "last_meter_oat_sensitivity.xlsx",
        "fallback_output_xlsx": MODEL_DIR / "ai_penalty_no_ai_features" / "last_meter_oat_sensitivity_new.xlsx",
    },
    "ai_with_features": {
        "output_dir": MODEL_DIR / "ai_penalty_with_ai_features",
        "output_xlsx": MODEL_DIR / "ai_penalty_with_ai_features" / "last_meter_oat_sensitivity.xlsx",
        "fallback_output_xlsx": MODEL_DIR / "ai_penalty_with_ai_features" / "last_meter_oat_sensitivity_new.xlsx",
    },
}


AMR_SPECS = {
    "response_prob_scale": {
        "label": "Customer response probability scale",
        "kind": "prob_scale",
        "base": 1.0,
        "low": 0.80,
        "high": 1.20,
    },
    "customer_response_timeout_s": {
        "label": "Customer response timeout (s)",
        "kind": "scalar",
        "patch_arg": "customer_response_timeout_s",
        "base": float(amr.CUSTOMER_RESPONSE_TIMEOUT_S),
        "low": float(amr.CUSTOMER_RESPONSE_TIMEOUT_S) * 0.80,
        "high": float(amr.CUSTOMER_RESPONSE_TIMEOUT_S) * 1.20,
    },
    "helper_search_timeout_s": {
        "label": "Helper search timeout (s)",
        "kind": "scalar",
        "patch_arg": "helper_search_timeout_s",
        "base": float(amr.HELPER_SEARCH_TIMEOUT_S),
        "low": float(amr.HELPER_SEARCH_TIMEOUT_S) * 0.80,
        "high": float(amr.HELPER_SEARCH_TIMEOUT_S) * 1.20,
    },
    "customer_walking_speed_mps": {
        "label": "Customer walking speed (m/s)",
        "kind": "scalar",
        "patch_arg": "customer_walking_speed_mps",
        "base": float(amr.CUSTOMER_WALKING_SPEED_MPS),
        "low": float(amr.CUSTOMER_WALKING_SPEED_MPS) * 0.80,
        "high": float(amr.CUSTOMER_WALKING_SPEED_MPS) * 1.20,
    },
    "elevator_wait_scale": {
        "label": "Elevator wait scale",
        "kind": "elevator_scale",
        "base": 1.0,
        "low": 0.80,
        "high": 1.20,
    },
}


CAR_SPECS = {
    "walking_speed_mps": {
        "label": "Courier walking speed (m/s)",
        "kind": "scalar",
        "patch_arg": "walking_speed_mps",
        "base": float(car.WALKING_SPEED_MPS),
        "low": float(car.WALKING_SPEED_MPS) * 0.80,
        "high": float(car.WALKING_SPEED_MPS) * 1.20,
    },
    "drop_off_time_s": {
        "label": "Drop-off time (s)",
        "kind": "scalar",
        "patch_arg": "drop_off_time_s",
        "base": float(car.DROP_OFF_TIME_S),
        "low": float(car.DROP_OFF_TIME_S) * 0.80,
        "high": float(car.DROP_OFF_TIME_S) * 1.20,
    },
    "indoor_walking_speed_mps": {
        "label": "Indoor walking speed (m/s)",
        "kind": "scalar",
        "patch_arg": "indoor_walking_speed_mps",
        "base": float(car.INDOOR_WALKING_SPEED_MPS),
        "low": float(car.INDOOR_WALKING_SPEED_MPS) * 0.80,
        "high": float(car.INDOOR_WALKING_SPEED_MPS) * 1.20,
    },
    "elevator_floor_time_s": {
        "label": "Elevator floor travel time (s)",
        "kind": "scalar",
        "patch_arg": "elevator_floor_time_s",
        "base": float(car.ELEVATOR_FLOOR_TIME_S),
        "low": float(car.ELEVATOR_FLOOR_TIME_S) * 0.80,
        "high": float(car.ELEVATOR_FLOOR_TIME_S) * 1.20,
    },
}


@contextmanager
def patched_amr_parameters(
    *,
    response_prob_scale: float | None = None,
    customer_response_timeout_s: float | None = None,
    helper_search_timeout_s: float | None = None,
    customer_walking_speed_mps: float | None = None,
    elevator_wait_scale: float | None = None,
):
    original_response_prob = deepcopy(amr.CUSTOMER_RESPONSE_PROB)
    original_customer_timeout = amr.CUSTOMER_RESPONSE_TIMEOUT_S
    original_helper_timeout = amr.HELPER_SEARCH_TIMEOUT_S
    original_customer_speed = amr.CUSTOMER_WALKING_SPEED_MPS
    original_elevator_wait = deepcopy(amr.CUSTOMER_ELEVATOR_WAIT_RANGE_S)

    try:
        if response_prob_scale is not None:
            amr.CUSTOMER_RESPONSE_PROB = {
                key: max(0.0, min(1.0, float(value) * float(response_prob_scale)))
                for key, value in original_response_prob.items()
            }
        if customer_response_timeout_s is not None:
            amr.CUSTOMER_RESPONSE_TIMEOUT_S = float(customer_response_timeout_s)
        if helper_search_timeout_s is not None:
            amr.HELPER_SEARCH_TIMEOUT_S = float(helper_search_timeout_s)
        if customer_walking_speed_mps is not None:
            amr.CUSTOMER_WALKING_SPEED_MPS = float(customer_walking_speed_mps)
        if elevator_wait_scale is not None:
            scale = float(elevator_wait_scale)
            amr.CUSTOMER_ELEVATOR_WAIT_RANGE_S = {
                key: (float(low) * scale, float(high) * scale)
                for key, (low, high) in original_elevator_wait.items()
            }
        yield
    finally:
        amr.CUSTOMER_RESPONSE_PROB = original_response_prob
        amr.CUSTOMER_RESPONSE_TIMEOUT_S = original_customer_timeout
        amr.HELPER_SEARCH_TIMEOUT_S = original_helper_timeout
        amr.CUSTOMER_WALKING_SPEED_MPS = original_customer_speed
        amr.CUSTOMER_ELEVATOR_WAIT_RANGE_S = original_elevator_wait


@contextmanager
def patched_car_parameters(
    *,
    walking_speed_mps: float | None = None,
    drop_off_time_s: float | None = None,
    indoor_walking_speed_mps: float | None = None,
    elevator_floor_time_s: float | None = None,
):
    original_walking_speed = car.WALKING_SPEED_MPS
    original_drop_off = car.DROP_OFF_TIME_S
    original_indoor_speed = car.INDOOR_WALKING_SPEED_MPS
    original_elevator_floor = car.ELEVATOR_FLOOR_TIME_S

    try:
        if walking_speed_mps is not None:
            car.WALKING_SPEED_MPS = float(walking_speed_mps)
        if drop_off_time_s is not None:
            car.DROP_OFF_TIME_S = float(drop_off_time_s)
        if indoor_walking_speed_mps is not None:
            car.INDOOR_WALKING_SPEED_MPS = float(indoor_walking_speed_mps)
        if elevator_floor_time_s is not None:
            car.ELEVATOR_FLOOR_TIME_S = float(elevator_floor_time_s)
        yield
    finally:
        car.WALKING_SPEED_MPS = original_walking_speed
        car.DROP_OFF_TIME_S = original_drop_off
        car.INDOOR_WALKING_SPEED_MPS = original_indoor_speed
        car.ELEVATOR_FLOOR_TIME_S = original_elevator_floor


def summarize_amr(df: pd.DataFrame, parameter_name: str, level_name: str, parameter_value: float) -> dict:
    return {
        "mode": "AMR",
        "parameter": parameter_name,
        "parameter_label": AMR_SPECS[parameter_name]["label"],
        "level": level_name,
        "parameter_value": parameter_value,
        "n_buildings": int(len(df)),
        "mean_time_s": float(df["amr_time_mean_s"].mean()),
        "median_time_s": float(df["amr_time_mean_s"].median()),
        "std_building_mean_time_s": float(df["amr_time_mean_s"].std(ddof=1)) if len(df) > 1 else 0.0,
        "mean_response_rate": float(df["response_rate"].mean()),
        "mean_helper_rate": float(df["helper_rate"].mean()),
        "mean_helper_cycles": float(df["helper_cycles_mean"].mean()),
    }


def summarize_car(df: pd.DataFrame, parameter_name: str, level_name: str, parameter_value: float) -> dict:
    return {
        "mode": "Car",
        "parameter": parameter_name,
        "parameter_label": CAR_SPECS[parameter_name]["label"],
        "level": level_name,
        "parameter_value": parameter_value,
        "n_buildings": int(len(df)),
        "mean_time_s": float(df["base_time_mean_s"].mean()),
        "median_time_s": float(df["base_time_mean_s"].median()),
        "std_building_mean_time_s": float(df["base_time_mean_s"].std(ddof=1)) if len(df) > 1 else 0.0,
        "mean_distance_m": float(df["base_distance_mean_m"].mean()),
        "mean_floor": float(df["base_floor_mean"].mean()),
    }


def run_amr_oat(parameter_name: str, *, n_buildings: int, n_runs: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = AMR_SPECS[parameter_name]
    summary_rows: list[dict] = []
    building_frames: list[pd.DataFrame] = []

    for level_name in ["low", "base", "high"]:
        value = float(spec[level_name])
        patch_kwargs = {
            "response_prob_scale": None,
            "customer_response_timeout_s": None,
            "helper_search_timeout_s": None,
            "customer_walking_speed_mps": None,
            "elevator_wait_scale": None,
        }
        if spec["kind"] == "scalar":
            patch_kwargs[spec["patch_arg"]] = value
        elif spec["kind"] == "prob_scale":
            patch_kwargs["response_prob_scale"] = value
        elif spec["kind"] == "elevator_scale":
            patch_kwargs["elevator_wait_scale"] = value

        with patched_amr_parameters(**patch_kwargs):
            df = ACTIVE_AMR_MODULE.run_simulation(
                n_buildings=max(1, n_buildings),
                n_runs=max(1, n_runs),
                seed=seed,
            ).copy()

        df["mode"] = "AMR"
        df["parameter"] = parameter_name
        df["parameter_label"] = spec["label"]
        df["level"] = level_name
        df["parameter_value"] = value
        summary_rows.append(summarize_amr(df, parameter_name, level_name, value))
        building_frames.append(df)
        print(f"[OAT][AMR] {parameter_name} - {level_name}: {len(df)} buildings")

    summary_df = pd.DataFrame(summary_rows)
    base_mean = float(summary_df.loc[summary_df["level"] == "base", "mean_time_s"].iloc[0])
    summary_df["delta_vs_base_s"] = summary_df["mean_time_s"] - base_mean
    summary_df["pct_change_vs_base"] = (summary_df["mean_time_s"] / base_mean - 1.0) * 100.0
    return summary_df, pd.concat(building_frames, ignore_index=True)


def run_car_oat(parameter_name: str, *, n_buildings: int, n_runs: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = CAR_SPECS[parameter_name]
    summary_rows: list[dict] = []
    building_frames: list[pd.DataFrame] = []

    for level_name in ["low", "base", "high"]:
        value = float(spec[level_name])
        patch_kwargs = {
            "walking_speed_mps": None,
            "drop_off_time_s": None,
            "indoor_walking_speed_mps": None,
            "elevator_floor_time_s": None,
        }
        patch_kwargs[spec["patch_arg"]] = value

        with patched_car_parameters(**patch_kwargs):
            df = ACTIVE_CAR_MODULE.run_simulation(
                n_buildings=max(1, n_buildings),
                n_runs=max(1, n_runs),
                seed=seed,
            ).copy()

        df["mode"] = "Car"
        df["parameter"] = parameter_name
        df["parameter_label"] = spec["label"]
        df["level"] = level_name
        df["parameter_value"] = value
        summary_rows.append(summarize_car(df, parameter_name, level_name, value))
        building_frames.append(df)
        print(f"[OAT][CAR] {parameter_name} - {level_name}: {len(df)} buildings")

    summary_df = pd.DataFrame(summary_rows)
    base_mean = float(summary_df.loc[summary_df["level"] == "base", "mean_time_s"].iloc[0])
    summary_df["delta_vs_base_s"] = summary_df["mean_time_s"] - base_mean
    summary_df["pct_change_vs_base"] = (summary_df["mean_time_s"] / base_mean - 1.0) * 100.0
    return summary_df, pd.concat(building_frames, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-at-a-time sensitivity analysis for AMR and car last-meter simulations.")
    parser.add_argument("--n-buildings", type=int, default=200, help="Number of buildings in each sensitivity sample.")
    parser.add_argument("--n-runs", type=int, default=100, help="Monte Carlo runs per building.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for all scenarios.")
    parser.add_argument(
        "--variant",
        choices=list(VARIANT_OUTPUTS.keys()),
        default="baseline",
        help="Which experiment branch to analyze.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variant_output = VARIANT_OUTPUTS[args.variant]
    output_dir = variant_output["output_dir"]
    output_xlsx_default = variant_output["output_xlsx"]
    fallback_output_xlsx = variant_output["fallback_output_xlsx"]
    output_dir.mkdir(parents=True, exist_ok=True)

    global ACTIVE_AMR_MODULE, ACTIVE_CAR_MODULE
    ACTIVE_AMR_MODULE = amr if args.variant == "baseline" else amr_ai
    ACTIVE_CAR_MODULE = car if args.variant == "baseline" else car_ai

    all_summary: list[pd.DataFrame] = []
    all_buildings: list[pd.DataFrame] = []

    for parameter_name, spec in AMR_SPECS.items():
        summary_df, building_df = run_amr_oat(
            parameter_name,
            n_buildings=args.n_buildings,
            n_runs=args.n_runs,
            seed=args.seed,
        )
        all_summary.append(summary_df)
        all_buildings.append(building_df)

    for parameter_name, spec in CAR_SPECS.items():
        summary_df, building_df = run_car_oat(
            parameter_name,
            n_buildings=args.n_buildings,
            n_runs=args.n_runs,
            seed=args.seed,
        )
        all_summary.append(summary_df)
        all_buildings.append(building_df)

    summary_all = pd.concat(all_summary, ignore_index=True)
    building_all = pd.concat(all_buildings, ignore_index=True)
    parameter_ranges_df = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "mode": "AMR",
                        "parameter": name,
                        "parameter_label": spec["label"],
                        "base": spec["base"],
                        "low": spec["low"],
                        "high": spec["high"],
                    }
                    for name, spec in AMR_SPECS.items()
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "mode": "Car",
                        "parameter": name,
                        "parameter_label": spec["label"],
                        "base": spec["base"],
                        "low": spec["low"],
                        "high": spec["high"],
                    }
                    for name, spec in CAR_SPECS.items()
                ]
            ),
        ],
        ignore_index=True,
    )
    run_params_df = pd.DataFrame(
        [
            {"parameter": "n_buildings", "value": args.n_buildings},
            {"parameter": "n_runs", "value": args.n_runs},
            {"parameter": "seed", "value": args.seed},
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
            stamped_name = f"last_meter_oat_sensitivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            output_xlsx = output_dir / stamped_name
            writer = pd.ExcelWriter(output_xlsx, engine="openpyxl")

    with writer:
        summary_all.to_excel(writer, index=False, sheet_name="scenario_summary")
        building_all.to_excel(writer, index=False, sheet_name="building_results")
        parameter_ranges_df.to_excel(writer, index=False, sheet_name="parameter_ranges")
        run_params_df.to_excel(writer, index=False, sheet_name="run_parameters")

    print(f"[OAT] Excel saved: {output_xlsx}")


if __name__ == "__main__":
    main()
