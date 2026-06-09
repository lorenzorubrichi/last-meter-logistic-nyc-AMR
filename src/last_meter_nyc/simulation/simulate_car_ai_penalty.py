from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

import building_scene_preview as base
from last_meter_nyc.paths import AI_EXPERIMENT_DIR, STREETVIEW_OUTPUT_DIR


EXPERIMENT_DIR = AI_EXPERIMENT_DIR
OUTPUT_STATS_XLSX = EXPERIMENT_DIR / "car_last_meter_stats_ai_penalty.xlsx"
AI_FEATURES_XLSX = EXPERIMENT_DIR / "ai_augmented_features.xlsx"
AI_MODELING_SUBSET_XLSX = EXPERIMENT_DIR / "ai_modeled_building_subset.xlsx"
AI_MODELING_SUBSET_CSV = EXPERIMENT_DIR / "ai_modeled_building_subset.csv"
AI_FEATURES_CSV = STREETVIEW_OUTPUT_DIR / "streetview_visual_features_api.csv"

# Easy-to-edit AI penalties for the car scenario
CAR_AI_STAIRS_PENALTY_S = 20.0
CAR_AI_GATE_PENALTY_S = 15.0
CAR_AI_RAMP_PENALTY_S = 10.0

AMR_FEATURES_SHEET = "amr_features"
CAR_FEATURES_SHEET = "car_features"


def ai_bool_to_float(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0})
        .astype("float64")
    )


def load_ai_features() -> pd.DataFrame:
    if not AI_FEATURES_CSV.exists():
        return pd.DataFrame(
            columns=[
                "bin",
                "ai_stairs_present",
                "ai_gate_present",
                "ai_ramp_present",
                "ai_access_barrier_mean",
            ]
        )
    df = pd.read_csv(AI_FEATURES_CSV)
    if "bin" not in df.columns:
        raise ValueError(f"Missing 'bin' column in {AI_FEATURES_CSV}")
    df["bin"] = pd.to_numeric(df["bin"], errors="coerce").astype("Int64")
    df["ai_stairs_present"] = ai_bool_to_float(df.get("stairs_present", pd.Series(index=df.index, dtype=object))).fillna(0.0)
    df["ai_gate_present"] = ai_bool_to_float(df.get("gate_present", pd.Series(index=df.index, dtype=object))).fillna(0.0)
    df["ai_ramp_present"] = ai_bool_to_float(df.get("ramp_present", pd.Series(index=df.index, dtype=object))).fillna(0.0)
    df["ai_access_barrier_mean"] = (
        df[["ai_stairs_present", "ai_gate_present", "ai_ramp_present"]]
        .mean(axis=1)
        .fillna(0.0)
    )
    return (
        df[
            [
                "bin",
                "ai_stairs_present",
                "ai_gate_present",
                "ai_ramp_present",
                "ai_access_barrier_mean",
            ]
        ]
        .dropna(subset=["bin"])
        .drop_duplicates(subset=["bin"])
        .reset_index(drop=True)
    )


def build_ai_augmented_feature_workbook() -> tuple[pd.DataFrame, pd.DataFrame]:
    amr_features = base.load_sheet_features(base.AMR_FEATURES_SHEET).copy()
    car_features = base.load_sheet_features(base.CAR_FEATURES_SHEET).copy()
    ai_features = load_ai_features()
    amr_aug = amr_features.merge(ai_features, on="bin", how="inner")
    car_aug = car_features.merge(ai_features, on="bin", how="inner")
    for df in [amr_aug, car_aug]:
        for col in ["ai_stairs_present", "ai_gate_present", "ai_ramp_present", "ai_access_barrier_mean"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    AI_FEATURES_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(AI_FEATURES_XLSX, engine="openpyxl") as writer:
        amr_aug.to_excel(writer, index=False, sheet_name=AMR_FEATURES_SHEET)
        car_aug.to_excel(writer, index=False, sheet_name=CAR_FEATURES_SHEET)

    modeling_subset = amr_aug.merge(
        car_aug[
            [
                c
                for c in [
                    "bin",
                    "raw_n_active_meters",
                    "ParkingScarcity_advantage",
                    "raw_n_regulation_signs",
                    "CurbRestriction_advantage",
                    "raw_bf_vehicle_ty",
                    "CommercialCurbContext",
                    "raw_number_park_lanes",
                    "raw_curb_crowding_sum",
                    "CurbCrowdingPenalty",
                ]
                if c in car_aug.columns
            ]
        ],
        on="bin",
        how="left",
    )
    with pd.ExcelWriter(AI_MODELING_SUBSET_XLSX, engine="openpyxl") as writer:
        modeling_subset.to_excel(writer, index=False, sheet_name="ai_modeled_buildings")
    modeling_subset.to_csv(AI_MODELING_SUBSET_CSV, index=False)
    return amr_aug, car_aug


def choose_random_bins_from_list(bin_values: list[int], n_buildings: int, seed: int | None) -> list[int]:
    rng = random.Random(seed)
    sample_size = min(max(1, n_buildings), len(bin_values))
    return rng.sample(bin_values, k=sample_size)


def compute_car_ai_penalty_s(ai_row: pd.Series) -> float:
    stairs = float(pd.to_numeric(ai_row.get("ai_stairs_present", 0.0), errors="coerce") or 0.0)
    gate = float(pd.to_numeric(ai_row.get("ai_gate_present", 0.0), errors="coerce") or 0.0)
    ramp = float(pd.to_numeric(ai_row.get("ai_ramp_present", 0.0), errors="coerce") or 0.0)
    return (
        stairs * CAR_AI_STAIRS_PENALTY_S
        + gate * CAR_AI_GATE_PENALTY_S
        + ramp * CAR_AI_RAMP_PENALTY_S
    )


def run_simulation(n_buildings: int, n_runs: int, seed: int | None) -> pd.DataFrame:
    buildings, streets = base.load_buildings_and_streets()
    amr_features_aug, car_features_aug = build_ai_augmented_feature_workbook()
    rng = random.Random(seed)

    results = []
    eligible = base.eligible_buildings(buildings)
    eligible["bin"] = pd.to_numeric(eligible["bin"], errors="coerce").astype("Int64")
    eligible_bins = set(int(v) for v in base.pd_to_numeric(eligible["bin"]).dropna().tolist())
    ai_bins = sorted(set(int(v) for v in pd.to_numeric(car_features_aug["bin"], errors="coerce").dropna().tolist()))
    common_bins = sorted(set(ai_bins).intersection(eligible_bins))
    selected_bins = choose_random_bins_from_list(common_bins, n_buildings=n_buildings, seed=seed)

    for building_bin in selected_bins:
        building = eligible[base.pd_to_numeric(eligible["bin"]) == building_bin].iloc[0]
        delivery_point = base.delivery_point_from_building(building)
        street = base.nearest_street(building, streets)
        parking_capacity = base.estimate_parking_capacity(street.geometry)
        parking_spots = base.generate_parking_spots(street.geometry, delivery_point, n_spots=parking_capacity)

        car_feature_match = car_features_aug[car_features_aug["bin"] == building_bin]
        amr_feature_match = amr_features_aug[amr_features_aug["bin"] == building_bin]

        base_ratio = 0.0
        floors_norm = 0.0
        road_to_delivery_distance_norm = 0.0
        ai_row = pd.Series(dtype=float)

        if not car_feature_match.empty:
            ai_row = car_feature_match.iloc[0]
            if "CurbCrowdingPenalty" in car_feature_match.columns:
                penalty = pd.to_numeric(car_feature_match.iloc[0]["CurbCrowdingPenalty"], errors="coerce")
                if pd.notna(penalty):
                    base_ratio = float(penalty)
            if "a2_RoadToDeliveryDistance_norm" in car_feature_match.columns:
                distance_norm = pd.to_numeric(car_feature_match.iloc[0]["a2_RoadToDeliveryDistance_norm"], errors="coerce")
                if pd.notna(distance_norm):
                    road_to_delivery_distance_norm = float(distance_norm)

        if not amr_feature_match.empty:
            if ai_row.empty:
                ai_row = amr_feature_match.iloc[0]
            if "a1_Floors_norm" in amr_feature_match.columns:
                floors_value = pd.to_numeric(amr_feature_match.iloc[0]["a1_Floors_norm"], errors="coerce")
                if pd.notna(floors_value):
                    floors_norm = float(floors_value)
            if road_to_delivery_distance_norm == 0.0 and "a2_RoadToDeliveryDistance_norm" in amr_feature_match.columns:
                distance_norm = pd.to_numeric(amr_feature_match.iloc[0]["a2_RoadToDeliveryDistance_norm"], errors="coerce")
                if pd.notna(distance_norm):
                    road_to_delivery_distance_norm = float(distance_norm)

        base_ratio = max(0.0, min(base.MAX_OCCUPANCY_RATIO, base_ratio))
        car_ai_penalty_s = compute_car_ai_penalty_s(ai_row)

        base_runs = []
        for _ in range(n_runs):
            run = base.simulate_one_run(
                parking_spots,
                delivery_point,
                street.geometry,
                building.geometry,
                building.get("landuse"),
                building.get("numfloors"),
                occupancy_ratio=base_ratio,
                rng=rng,
            )
            if run.get("total_time_s") is not None:
                run["total_time_s"] = float(run["total_time_s"]) + car_ai_penalty_s
            base_runs.append(run)

        row = {
            "bin": building_bin,
            "delivery_lon": float(building.get("delivery_lon")),
            "delivery_lat": float(building.get("delivery_lat")),
            "a1_Floors_norm": floors_norm,
            "CurbCrowdingPenalty": base_ratio,
            "a2_RoadToDeliveryDistance_norm": road_to_delivery_distance_norm,
            "ShapePenalty_norm": float(base.estimate_shape_penalty(building.geometry)),
            "BuildingTypePenalty_norm": float(base.building_type_penalty(building.get("landuse"))),
            "parking_capacity": parking_capacity,
            "base_occupancy_ratio": base_ratio,
            "ai_stairs_present": float(pd.to_numeric(ai_row.get("ai_stairs_present", 0.0), errors="coerce") or 0.0),
            "ai_gate_present": float(pd.to_numeric(ai_row.get("ai_gate_present", 0.0), errors="coerce") or 0.0),
            "ai_ramp_present": float(pd.to_numeric(ai_row.get("ai_ramp_present", 0.0), errors="coerce") or 0.0),
            "ai_access_barrier_mean": float(pd.to_numeric(ai_row.get("ai_access_barrier_mean", 0.0), errors="coerce") or 0.0),
            "car_ai_penalty_s": car_ai_penalty_s,
        }
        row.update(base.summarize_runs(base_runs, "base"))
        results.append(row)

    return pd.DataFrame(results)


def save_results(df: pd.DataFrame) -> None:
    OUTPUT_STATS_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_STATS_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="car_last_meter", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI-penalized Monte Carlo simulation for car last-meter time.")
    parser.add_argument("--n-buildings", type=int, default=500, help="Number of buildings to evaluate.")
    parser.add_argument("--n-runs", type=int, default=100, help="Monte Carlo runs per building.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_simulation(
        n_buildings=max(1, args.n_buildings),
        n_runs=max(1, args.n_runs),
        seed=args.seed,
    )
    save_results(results)
    print(f"Excel saved: {OUTPUT_STATS_XLSX}")
    print(f"AI-augmented features saved: {AI_FEATURES_XLSX}")
    print(f"AI-modeled subset saved: {AI_MODELING_SUBSET_XLSX}")


if __name__ == "__main__":
    main()
