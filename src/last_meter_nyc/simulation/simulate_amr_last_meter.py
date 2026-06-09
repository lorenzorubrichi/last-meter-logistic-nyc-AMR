from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from building_scene_preview import (
    FT_TO_M,
    INDOOR_WALKING_SPEED_MPS,
    ELEVATOR_FLOOR_TIME_S,
    OUTPUT_STATS_XLSX as CAR_STATS_XLSX,
    delivery_point_from_building,
    eligible_buildings,
    elevator_point_from_building,
    entrance_point_from_building,
    landuse_wait_class,
    load_buildings_and_streets,
    sample_building_floor,
    sample_internal_dropoff_point,
)
from last_meter_nyc.paths import MODELS_DIR


BASE_DIR = MODELS_DIR.parent
MODEL_DIR = MODELS_DIR
OUTPUT_AMR_STATS_XLSX = MODEL_DIR / "amr_last_meter_stats.xlsx"
AMR_FEATURES_XLSX = MODEL_DIR / "OUTPUTamr_features.xlsx"
AMR_FEATURES_SHEET = "amr_features"

# Easy-to-edit simulation knobs
DEFAULT_N_BUILDINGS = 500
DEFAULT_N_RUNS = 100

AMR_SIDEWALK_SPEED_MPS = 1.2
CUSTOMER_RESPONSE_TIMEOUT_S = 300.0
HELPER_SEARCH_TIMEOUT_S = 120.0
MAX_HELP_CYCLES = 2
CUSTOMER_HANDOFF_TIME_S = 30.0
CUSTOMER_WALKING_SPEED_MPS = 1.2
HELPER_WALKING_SPEED_MPS = INDOOR_WALKING_SPEED_MPS
CUSTOMER_ELEVATOR_FLOOR_TIME_S = ELEVATOR_FLOOR_TIME_S

CUSTOMER_RESPONSE_PROB = {
    "residential": 0.90,
    "mixed_commercial": 0.75,
    "industrial_utility": 0.45,
    "public_other": 0.60,
}

CUSTOMER_ELEVATOR_WAIT_RANGE_S = {
    "residential": (15.0, 45.0),
    "mixed_commercial": (10.0, 35.0),
    "industrial_utility": (20.0, 60.0),
    "public_other": (25.0, 75.0),
}


def choose_random_bins_from_list(bin_values: list[int], n_buildings: int, seed: int | None) -> list[int]:
    rng = random.Random(seed)
    sample_size = min(max(1, n_buildings), len(bin_values))
    return rng.sample(bin_values, k=sample_size)


def sample_customer_response(landuse_value, rng: random.Random) -> tuple[bool, float | None]:
    wait_class = landuse_wait_class(landuse_value)
    p_response = CUSTOMER_RESPONSE_PROB.get(wait_class, 0.50)
    responded = rng.random() < p_response
    if not responded:
        return False, None
    return True, rng.uniform(0.0, CUSTOMER_RESPONSE_TIMEOUT_S)


def sample_customer_elevator_wait_s(landuse_value, rng: random.Random) -> float:
    wait_class = landuse_wait_class(landuse_value)
    low, high = CUSTOMER_ELEVATOR_WAIT_RANGE_S.get(wait_class, (25.0, 75.0))
    return rng.uniform(low, high)


def sample_helper_found(probability: float, rng: random.Random) -> bool:
    probability = max(0.0, min(1.0, probability))
    return rng.random() < probability


def build_customer_route_metrics(
    entrance_point,
    elevator_point,
    internal_dropoff,
    building_floor: int,
    landuse_value,
    rng: random.Random,
) -> dict:
    if building_floor > 1:
        dropoff_to_elevator_m = internal_dropoff.distance(elevator_point) * FT_TO_M
        elevator_to_entrance_m = elevator_point.distance(entrance_point) * FT_TO_M
        customer_path_m = dropoff_to_elevator_m + elevator_to_entrance_m
        elevator_wait_time_s = sample_customer_elevator_wait_s(landuse_value, rng)
        elevator_travel_time_s = building_floor * CUSTOMER_ELEVATOR_FLOOR_TIME_S
    else:
        customer_path_m = internal_dropoff.distance(entrance_point) * FT_TO_M
        elevator_wait_time_s = 0.0
        elevator_travel_time_s = 0.0

    customer_walk_time_s = customer_path_m / CUSTOMER_WALKING_SPEED_MPS
    return {
        "customer_path_m": customer_path_m,
        "elevator_wait_time_s": elevator_wait_time_s,
        "elevator_travel_time_s": elevator_travel_time_s,
        "customer_walk_time_s": customer_walk_time_s,
    }


def build_helper_route_metrics(
    entrance_point,
    elevator_point,
    internal_dropoff,
    building_floor: int,
    landuse_value,
    rng: random.Random,
) -> dict:
    if building_floor > 1:
        entrance_to_elevator_m = entrance_point.distance(elevator_point) * FT_TO_M
        elevator_to_dropoff_m = elevator_point.distance(internal_dropoff) * FT_TO_M
        helper_one_way_m = entrance_to_elevator_m + elevator_to_dropoff_m
        elevator_wait_time_s = sample_customer_elevator_wait_s(landuse_value, rng)
        elevator_travel_time_s = 2.0 * building_floor * CUSTOMER_ELEVATOR_FLOOR_TIME_S
    else:
        helper_one_way_m = entrance_point.distance(internal_dropoff) * FT_TO_M
        elevator_wait_time_s = 0.0
        elevator_travel_time_s = 0.0

    helper_round_trip_m = 2.0 * helper_one_way_m
    helper_walk_time_s = helper_round_trip_m / HELPER_WALKING_SPEED_MPS
    return {
        "helper_path_m": helper_round_trip_m,
        "elevator_wait_time_s": elevator_wait_time_s,
        "elevator_travel_time_s": elevator_travel_time_s,
        "helper_walk_time_s": helper_walk_time_s,
    }


def simulate_one_amr_run(
    building,
    amr_feature_row: pd.Series,
    rng: random.Random,
    helper_prob_boost_delta: float = 0.0,
) -> dict:
    delivery_point = delivery_point_from_building(building)
    entrance_point = entrance_point_from_building(building.geometry, delivery_point)
    elevator_point = elevator_point_from_building(building.geometry)
    internal_dropoff = sample_internal_dropoff_point(building.geometry, rng)
    building_floor = sample_building_floor(building.get("numfloors"), rng)
    sidewalk_to_entrance_m = entrance_point.distance(delivery_point) * FT_TO_M
    amr_approach_time_s = sidewalk_to_entrance_m / AMR_SIDEWALK_SPEED_MPS
    helper_prob_base = float(
        pd.to_numeric(
            amr_feature_row.reindex(
                ["b1_Population_norm", "b2_PedestrianPresence_norm", "b3_UrbanActivity_norm"]
            ),
            errors="coerce",
        )
        .fillna(0.0)
        .mean()
    )
    helper_prob = max(0.0, min(1.0, helper_prob_base + helper_prob_boost_delta))

    total_time_s = amr_approach_time_s
    customer_path_m = None
    helper_path_m = None
    elevator_wait_time_s = None
    helper_found = False
    helper_cycles = 0

    responded, response_time_s = sample_customer_response(building.get("landuse"), rng)
    if responded and response_time_s is not None:
        route = build_customer_route_metrics(
            entrance_point,
            elevator_point,
            internal_dropoff,
            building_floor,
            building.get("landuse"),
            rng,
        )
        total_time_s += (
            float(response_time_s)
            + route["customer_walk_time_s"]
            + route["elevator_wait_time_s"]
            + route["elevator_travel_time_s"]
            + CUSTOMER_HANDOFF_TIME_S
        )
        customer_path_m = route["customer_path_m"]
        elevator_wait_time_s = route["elevator_wait_time_s"]
        response_time_out = float(response_time_s)
    else:
        total_time_s += CUSTOMER_RESPONSE_TIMEOUT_S
        response_time_out = CUSTOMER_RESPONSE_TIMEOUT_S
        for cycle_idx in range(MAX_HELP_CYCLES):
            helper_cycles = cycle_idx + 1
            total_time_s += HELPER_SEARCH_TIMEOUT_S
            if sample_helper_found(helper_prob, rng):
                helper_found = True
                route = build_helper_route_metrics(
                    entrance_point,
                    elevator_point,
                    internal_dropoff,
                    building_floor,
                    building.get("landuse"),
                    rng,
                )
                total_time_s += (
                    route["helper_walk_time_s"]
                    + (2.0 * route["elevator_wait_time_s"])
                    + route["elevator_travel_time_s"]
                    + CUSTOMER_HANDOFF_TIME_S
                )
                helper_path_m = route["helper_path_m"]
                elevator_wait_time_s = route["elevator_wait_time_s"]
                break
            if cycle_idx < MAX_HELP_CYCLES - 1:
                total_time_s += CUSTOMER_RESPONSE_TIMEOUT_S

    return {
        "customer_responded": bool(responded),
        "helper_found": helper_found,
        "helper_cycles": helper_cycles,
        "helper_probability_base": helper_prob_base,
        "helper_probability_used": helper_prob,
        "sidewalk_to_entrance_m": sidewalk_to_entrance_m,
        "amr_approach_time_s": amr_approach_time_s,
        "response_time_s": response_time_out,
        "building_floor": building_floor,
        "customer_path_m": customer_path_m,
        "helper_path_m": helper_path_m,
        "elevator_wait_time_s": elevator_wait_time_s,
        "total_time_s": total_time_s,
    }


def summarize_runs(records: list[dict], scenario_name: str) -> dict:
    responded_flags = [bool(r["customer_responded"]) for r in records]
    helper_flags = [bool(r["helper_found"]) for r in records]
    responded_records = [r for r in records if r["customer_responded"]]
    helper_records = [r for r in records if r["helper_found"]]

    approach_times = pd.Series([r["amr_approach_time_s"] for r in records], dtype=float)
    sidewalk_distances = pd.Series([r["sidewalk_to_entrance_m"] for r in records], dtype=float)
    response_times = pd.Series([r["response_time_s"] for r in records], dtype=float)
    customer_paths = pd.Series([r["customer_path_m"] for r in responded_records if r["customer_path_m"] is not None], dtype=float)
    helper_paths = pd.Series([r["helper_path_m"] for r in helper_records if r["helper_path_m"] is not None], dtype=float)
    responded_elevator_waits = pd.Series([r["elevator_wait_time_s"] for r in responded_records if r["elevator_wait_time_s"] is not None], dtype=float)
    helper_elevator_waits = pd.Series([r["elevator_wait_time_s"] for r in helper_records if r["elevator_wait_time_s"] is not None], dtype=float)
    floors = pd.Series([r["building_floor"] for r in records], dtype=float)
    helper_cycles = pd.Series([r["helper_cycles"] for r in records], dtype=float)
    helper_probabilities_base = pd.Series([r["helper_probability_base"] for r in records], dtype=float)
    helper_probabilities_used = pd.Series([r["helper_probability_used"] for r in records], dtype=float)
    total_times = pd.Series([r["total_time_s"] for r in records], dtype=float)

    return {
        f"{scenario_name}_n_runs": len(records),
        f"{scenario_name}_response_rate": float(sum(responded_flags) / len(records)) if records else 0.0,
        f"{scenario_name}_helper_rate": float(sum(helper_flags) / len(records)) if records else 0.0,
        f"{scenario_name}_responded_runs": int(len(responded_records)),
        f"{scenario_name}_helper_runs": int(len(helper_records)),
        f"{scenario_name}_helper_probability_base_mean": float(helper_probabilities_base.mean()) if not helper_probabilities_base.empty else None,
        f"{scenario_name}_helper_probability_used_mean": float(helper_probabilities_used.mean()) if not helper_probabilities_used.empty else None,
        f"{scenario_name}_helper_cycles_mean": float(helper_cycles.mean()) if not helper_cycles.empty else None,
        f"{scenario_name}_sidewalk_to_entrance_mean_m": float(sidewalk_distances.mean()) if not sidewalk_distances.empty else None,
        f"{scenario_name}_sidewalk_to_entrance_std_m": float(sidewalk_distances.std(ddof=1)) if len(sidewalk_distances) > 1 else 0.0 if len(sidewalk_distances) == 1 else None,
        f"{scenario_name}_amr_approach_time_mean_s": float(approach_times.mean()) if not approach_times.empty else None,
        f"{scenario_name}_amr_approach_time_std_s": float(approach_times.std(ddof=1)) if len(approach_times) > 1 else 0.0 if len(approach_times) == 1 else None,
        f"{scenario_name}_response_time_mean_s": float(response_times.mean()) if not response_times.empty else None,
        f"{scenario_name}_response_time_std_s": float(response_times.std(ddof=1)) if len(response_times) > 1 else 0.0 if len(response_times) == 1 else None,
        f"{scenario_name}_customer_path_mean_m": float(customer_paths.mean()) if not customer_paths.empty else None,
        f"{scenario_name}_customer_path_std_m": float(customer_paths.std(ddof=1)) if len(customer_paths) > 1 else 0.0 if len(customer_paths) == 1 else None,
        f"{scenario_name}_helper_path_mean_m": float(helper_paths.mean()) if not helper_paths.empty else None,
        f"{scenario_name}_helper_path_std_m": float(helper_paths.std(ddof=1)) if len(helper_paths) > 1 else 0.0 if len(helper_paths) == 1 else None,
        f"{scenario_name}_customer_elevator_wait_mean_s": float(responded_elevator_waits.mean()) if not responded_elevator_waits.empty else None,
        f"{scenario_name}_customer_elevator_wait_std_s": float(responded_elevator_waits.std(ddof=1)) if len(responded_elevator_waits) > 1 else 0.0 if len(responded_elevator_waits) == 1 else None,
        f"{scenario_name}_helper_elevator_wait_mean_s": float(helper_elevator_waits.mean()) if not helper_elevator_waits.empty else None,
        f"{scenario_name}_helper_elevator_wait_std_s": float(helper_elevator_waits.std(ddof=1)) if len(helper_elevator_waits) > 1 else 0.0 if len(helper_elevator_waits) == 1 else None,
        f"{scenario_name}_floor_mean": float(floors.mean()) if not floors.empty else None,
        f"{scenario_name}_floor_std": float(floors.std(ddof=1)) if len(floors) > 1 else 0.0 if len(floors) == 1 else None,
        f"{scenario_name}_amr_time_mean_s": float(total_times.mean()) if not total_times.empty else None,
        f"{scenario_name}_amr_time_std_s": float(total_times.std(ddof=1)) if len(total_times) > 1 else 0.0 if len(total_times) == 1 else None,
    }


def run_simulation(
    n_buildings: int,
    n_runs: int,
    seed: int | None,
) -> pd.DataFrame:
    buildings, _ = load_buildings_and_streets()
    buildings = buildings.copy()
    buildings["bin"] = pd.to_numeric(buildings["bin"], errors="coerce").astype("Int64")
    car_stats = pd.read_excel(CAR_STATS_XLSX, sheet_name="car_last_meter")
    car_stats["bin"] = pd.to_numeric(car_stats["bin"], errors="coerce").astype("Int64")
    amr_features = pd.read_excel(AMR_FEATURES_XLSX, sheet_name=AMR_FEATURES_SHEET)
    amr_features["bin"] = pd.to_numeric(amr_features["bin"], errors="coerce").astype("Int64")
    eligible = eligible_buildings(buildings)
    eligible_bins = set(int(v) for v in pd.to_numeric(eligible["bin"]).dropna().tolist())
    car_bins = [int(v) for v in pd.to_numeric(car_stats["bin"]).dropna().tolist()]
    common_bins = sorted(set(car_bins).intersection(eligible_bins))
    selected_bins = choose_random_bins_from_list(common_bins, n_buildings=n_buildings, seed=seed)
    rng = random.Random(seed)
    results = []

    for building_bin in selected_bins:
        building_match = buildings[buildings["bin"] == building_bin]
        if building_match.empty:
            continue
        building = building_match.iloc[0]
        amr_feature_match = amr_features[amr_features["bin"] == building_bin]
        amr_feature_row = amr_feature_match.iloc[0] if not amr_feature_match.empty else pd.Series(dtype=float)
        base_runs = [
            simulate_one_amr_run(building, amr_feature_row, rng, helper_prob_boost_delta=0.0)
            for _ in range(n_runs)
        ]

        row = {
            "bin": int(building_bin),
            "delivery_lon": float(building.get("delivery_lon")),
            "delivery_lat": float(building.get("delivery_lat")),
            "landuse": building.get("landuse"),
        }

        stats_match = car_stats[car_stats["bin"] == building_bin]
        if not stats_match.empty:
            for col in [
                "a1_Floors_norm",
                "CurbCrowdingPenalty",
                "a2_RoadToDeliveryDistance_norm",
                "ShapePenalty_norm",
                "BuildingTypePenalty_norm",
            ]:
                if col in stats_match.columns:
                    row[col] = stats_match.iloc[0][col]

        for col in ["b1_Population_norm", "b2_PedestrianPresence_norm", "b3_UrbanActivity_norm"]:
            if col in amr_feature_row.index:
                row[col] = amr_feature_row[col]

        row.update(summarize_runs(base_runs, "base"))
        row["response_rate"] = row.get("base_response_rate")
        row["helper_rate"] = row.get("base_helper_rate")
        row["helper_probability_mean"] = row.get("base_helper_probability_used_mean")
        row["helper_cycles_mean"] = row.get("base_helper_cycles_mean")
        row["amr_time_mean_s"] = row.get("base_amr_time_mean_s")
        row["amr_time_std_s"] = row.get("base_amr_time_std_s")
        results.append(row)

    return pd.DataFrame(results)


def save_results(df: pd.DataFrame) -> None:
    OUTPUT_AMR_STATS_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_AMR_STATS_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="amr_last_meter", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo simulation for AMR last-meter customer response.")
    parser.add_argument("--n-buildings", type=int, default=DEFAULT_N_BUILDINGS, help="Number of buildings to evaluate.")
    parser.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS, help="Monte Carlo runs per building.")
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
    print(f"Excel saved: {OUTPUT_AMR_STATS_XLSX}")


if __name__ == "__main__":
    main()
