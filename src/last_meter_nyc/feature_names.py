"""Public feature names and backward-compatible aliases.

The research code originally used compact internal names while the analysis
was evolving. Public exports use descriptive names.

Naming convention:
- columns ending in `_R` are features actually used as regression inputs in the
  current baseline or AI-informed model specifications;
- columns without `_R` are raw, descriptive, intermediate, or output fields.
"""

REGRESSION_FEATURES_LEGACY = {
    "a1_Floors_norm",
    "a2_RoadToDeliveryDistance_norm",
    "ShapePenalty_norm",
    "BuildingTypePenalty_norm",
    "b1_Population_norm",
    "b2_PedestrianPresence_norm",
    "CurbCrowdingPenalty",
    "ai_stairs_present",
    "ai_gate_present",
    "ai_ramp_present",
    "ai_access_barrier_mean",
}


FEATURE_RENAME_MAP = {
    "a1_Floors_norm": "building_floor_count_norm_R",
    "a2_RoadToDeliveryDistance_norm": "road_to_delivery_distance_norm_R",
    "a3_AddressUncertainty": "missing_address_point_flag",
    "ShapePenalty_norm": "building_shape_complexity_norm_R",
    "BuildingTypePenalty_norm": "building_landuse_access_penalty_norm_R",
    "b1_Population_norm": "neighborhood_population_intensity_norm_R",
    "b2_PedestrianPresence_norm": "pedestrian_corridor_presence_norm_R",
    "b3_UrbanActivity_norm": "urban_activity_landuse_norm",
    "c1_PedestrianPenalty": "low_pedestrian_presence_penalty",
    "c2_SidewalkAbsencePenalty": "sidewalk_absence_penalty",
    "raw_n_active_meters": "active_parking_meter_count",
    "ParkingScarcity_advantage": "parking_meter_scarcity_penalty",
    "raw_n_regulation_signs": "curb_restriction_sign_count",
    "CurbRestriction_advantage": "curb_restriction_intensity_norm",
    "CommercialCurbContext": "commercial_or_truck_curb_context_flag",
    "raw_number_park_lanes": "parking_lane_count",
    "raw_curb_crowding_sum": "curb_difficulty_component_sum",
    "CurbCrowdingPenalty": "curb_parking_difficulty_penalty_R",
    "image_usable": "streetview_image_usable_flag",
    "stairs_present": "streetview_stairs_detected",
    "gate_present": "streetview_gate_detected",
    "ramp_present": "streetview_ramp_or_barrier_detected",
    "amr_can_reach_door": "streetview_amr_can_reach_door_flag",
    "ai_stairs_present": "streetview_stairs_present_R",
    "ai_gate_present": "streetview_gate_present_R",
    "ai_ramp_present": "streetview_ramp_or_barrier_present_R",
    "ai_access_barrier_mean": "streetview_access_barrier_score_R",
    "ai_stairs_present_car": "car_streetview_stairs_present_R",
    "ai_gate_present_car": "car_streetview_gate_present_R",
    "ai_ramp_present_car": "car_streetview_ramp_or_barrier_present_R",
    "ai_access_barrier_mean_car": "car_streetview_access_barrier_score_R",
    "ai_stairs_present_amr": "amr_streetview_stairs_present_R",
    "ai_gate_present_amr": "amr_streetview_gate_present_R",
    "ai_ramp_present_amr": "amr_streetview_ramp_or_barrier_present_R",
    "ai_access_barrier_mean_amr": "amr_streetview_access_barrier_score_R",
    "car_last_meter_mean_s": "predicted_car_last_meter_time_s",
    "amr_last_meter_mean_s": "predicted_amr_last_meter_time_s",
    "base_time_mean_s": "simulated_car_last_meter_time_s",
    "base_time_std_s": "simulated_car_last_meter_time_std_s",
    "amr_time_mean_s": "simulated_amr_last_meter_time_s",
    "amr_time_std_s": "simulated_amr_last_meter_time_std_s",
}


FEATURE_DESCRIPTIONS = {
    "building_floor_count_norm_R": "Regression feature: normalized building verticality derived from number of floors.",
    "road_to_delivery_distance_norm_R": "Regression feature: normalized distance between the building delivery point and nearest street context.",
    "missing_address_point_flag": "1 when no official address point was available and a fallback delivery point was used.",
    "building_shape_complexity_norm_R": "Regression feature: footprint elongation/shape penalty derived from the minimum rotated rectangle.",
    "building_landuse_access_penalty_norm_R": "Regression feature: rule-based access penalty derived from building land-use class.",
    "neighborhood_population_intensity_norm_R": "Regression feature: normalized neighborhood population proxy.",
    "pedestrian_corridor_presence_norm_R": "Regression feature: NYC pedestrian corridor importance mapped to a 0-1 scale.",
    "urban_activity_landuse_norm": "Urban activity proxy based on land-use class. Exported for context, not used in the current baseline regression feature set.",
    "low_pedestrian_presence_penalty": "Inverse pedestrian-presence penalty. Exported as an intermediate feature.",
    "sidewalk_absence_penalty": "1 when the nearest sidewalk is not reachable within the selected threshold. Exported as an intermediate feature.",
    "active_parking_meter_count": "Count of active on-street parking meters near the associated street segment.",
    "parking_meter_scarcity_penalty": "Inverse normalized active-meter count; higher values mean less visible metered supply.",
    "curb_restriction_sign_count": "Count of restrictive curb signs near the associated street segment.",
    "curb_restriction_intensity_norm": "Normalized restrictive-sign count.",
    "commercial_or_truck_curb_context_flag": "1 when nearest ParkNYC blockface indicates commercial/truck context.",
    "parking_lane_count": "Number of parking lanes from the associated street context, when available.",
    "curb_difficulty_component_sum": "Sum of parking scarcity, restriction intensity, and commercial/truck context.",
    "curb_parking_difficulty_penalty_R": "Regression feature: final curbside parking difficulty score used as simulated parking occupancy.",
    "streetview_image_usable_flag": "AI-derived quality flag for whether the image is usable for entrance assessment.",
    "streetview_stairs_detected": "Raw AI-derived indicator for visible stairs at or near the entrance.",
    "streetview_gate_detected": "Raw AI-derived indicator for visible gate/fence/barrier at or near the entrance.",
    "streetview_ramp_or_barrier_detected": "Raw AI-derived indicator for ramp or ramp-like access barrier.",
    "streetview_amr_can_reach_door_flag": "AI-derived assessment of whether an AMR can plausibly reach the door. Used for segmentation/filtering, not as a direct regression input.",
    "streetview_stairs_present_R": "AI-informed regression feature: visible stairs at or near the entrance.",
    "streetview_gate_present_R": "AI-informed regression feature: visible gate/fence/barrier at or near the entrance.",
    "streetview_ramp_or_barrier_present_R": "AI-informed regression feature: ramp or ramp-like access barrier.",
    "streetview_access_barrier_score_R": "AI-informed regression feature: average of selected access-barrier indicators.",
    "car_streetview_stairs_present_R": "Car AI-informed regression feature: visible stairs.",
    "car_streetview_gate_present_R": "Car AI-informed regression feature: visible gate/fence/barrier.",
    "car_streetview_ramp_or_barrier_present_R": "Car AI-informed regression feature: ramp or ramp-like access barrier.",
    "car_streetview_access_barrier_score_R": "Car AI-informed regression feature: average access-barrier score.",
    "amr_streetview_stairs_present_R": "AMR AI-informed regression feature: visible stairs.",
    "amr_streetview_gate_present_R": "AMR AI-informed regression feature: visible gate/fence/barrier.",
    "amr_streetview_ramp_or_barrier_present_R": "AMR AI-informed regression feature: ramp or ramp-like access barrier.",
    "amr_streetview_access_barrier_score_R": "AMR AI-informed regression feature: average access-barrier score.",
    "predicted_car_last_meter_time_s": "Regression-predicted car last-meter time in seconds.",
    "predicted_amr_last_meter_time_s": "Regression-predicted AMR last-meter time in seconds.",
    "simulated_car_last_meter_time_s": "Monte Carlo simulated car last-meter time in seconds.",
    "simulated_car_last_meter_time_std_s": "Standard deviation of simulated car last-meter time in seconds.",
    "simulated_amr_last_meter_time_s": "Monte Carlo simulated AMR last-meter time in seconds.",
    "simulated_amr_last_meter_time_std_s": "Standard deviation of simulated AMR last-meter time in seconds.",
}


def rename_for_publication(columns: list[str]) -> dict[str, str]:
    """Return a rename dictionary for columns present in a dataframe."""
    rename_map: dict[str, str] = {}
    suffixes = ("_x", "_y", "_left", "_right")
    for column in columns:
        if column in FEATURE_RENAME_MAP:
            rename_map[column] = FEATURE_RENAME_MAP[column]
            continue
        for suffix in suffixes:
            if column.endswith(suffix):
                base = column[: -len(suffix)]
                if base in FEATURE_RENAME_MAP:
                    public_base = FEATURE_RENAME_MAP[base]
                    if public_base.endswith("_R"):
                        rename_map[column] = f"{public_base[:-2]}{suffix}_R"
                    else:
                        rename_map[column] = f"{public_base}{suffix}"
                    break
    return rename_map
