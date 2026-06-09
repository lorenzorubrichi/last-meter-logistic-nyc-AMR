from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

import amr_last_meter_sim as base
from building_scene_preview_ai_penalty import AI_FEATURES_XLSX, AMR_FEATURES_SHEET
from last_meter_nyc.paths import AI_EXPERIMENT_DIR


EXPERIMENT_DIR = AI_EXPERIMENT_DIR
OUTPUT_AMR_STATS_XLSX = EXPERIMENT_DIR / "amr_last_meter_stats_ai_penalty.xlsx"

# Easy-to-edit AI penalties for the AMR scenario
AMR_AI_RAMP_PENALTY_S = 20.0


def compute_amr_ai_penalty_s(ai_row: pd.Series) -> float:
    ramp = float(pd.to_numeric(ai_row.get("ai_ramp_present", 0.0), errors="coerce") or 0.0)
    return ramp * AMR_AI_RAMP_PENALTY_S


def run_simulation(n_buildings: int, n_runs: int, seed: int | None) -> pd.DataFrame:
    buildings, _ = base.load_buildings_and_streets()
    buildings = buildings.copy()
    buildings["bin"] = pd.to_numeric(buildings["bin"], errors="coerce").astype("Int64")

    car_stats = pd.read_excel(EXPERIMENT_DIR / "car_last_meter_stats_ai_penalty.xlsx", sheet_name="car_last_meter")
    car_stats["bin"] = pd.to_numeric(car_stats["bin"], errors="coerce").astype("Int64")

    amr_features = pd.read_excel(AI_FEATURES_XLSX, sheet_name=AMR_FEATURES_SHEET)
    amr_features["bin"] = pd.to_numeric(amr_features["bin"], errors="coerce").astype("Int64")

    eligible = base.eligible_buildings(buildings)
    eligible_bins = set(int(v) for v in pd.to_numeric(eligible["bin"]).dropna().tolist())
    car_bins = [int(v) for v in pd.to_numeric(car_stats["bin"]).dropna().tolist()]
    common_bins = sorted(set(car_bins).intersection(eligible_bins))
    selected_bins = base.choose_random_bins_from_list(common_bins, n_buildings=n_buildings, seed=seed)
    rng = random.Random(seed)
    results = []

    for building_bin in selected_bins:
        building_match = buildings[buildings["bin"] == building_bin]
        if building_match.empty:
            continue
        building = building_match.iloc[0]

        amr_feature_match = amr_features[amr_features["bin"] == building_bin]
        if amr_feature_match.empty:
            continue
        amr_feature_row = amr_feature_match.iloc[0]
        amr_ai_penalty_s = compute_amr_ai_penalty_s(amr_feature_row)

        base_runs = []
        for _ in range(n_runs):
            run = base.simulate_one_amr_run(building, amr_feature_row, rng, helper_prob_boost_delta=0.0)
            if run.get("total_time_s") is not None:
                run["total_time_s"] = float(run["total_time_s"]) + amr_ai_penalty_s
            base_runs.append(run)

        row = {
            "bin": int(building_bin),
            "delivery_lon": float(building.get("delivery_lon")),
            "delivery_lat": float(building.get("delivery_lat")),
            "landuse": building.get("landuse"),
            "ai_stairs_present": float(pd.to_numeric(amr_feature_row.get("ai_stairs_present", 0.0), errors="coerce") or 0.0),
            "ai_gate_present": float(pd.to_numeric(amr_feature_row.get("ai_gate_present", 0.0), errors="coerce") or 0.0),
            "ai_ramp_present": float(pd.to_numeric(amr_feature_row.get("ai_ramp_present", 0.0), errors="coerce") or 0.0),
            "ai_access_barrier_mean": float(pd.to_numeric(amr_feature_row.get("ai_access_barrier_mean", 0.0), errors="coerce") or 0.0),
            "amr_ai_penalty_s": amr_ai_penalty_s,
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

        row.update(base.summarize_runs(base_runs, "base"))
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
    parser = argparse.ArgumentParser(description="AI-penalized Monte Carlo simulation for AMR last-meter response.")
    parser.add_argument("--n-buildings", type=int, default=base.DEFAULT_N_BUILDINGS, help="Number of buildings to evaluate.")
    parser.add_argument("--n-runs", type=int, default=base.DEFAULT_N_RUNS, help="Monte Carlo runs per building.")
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
