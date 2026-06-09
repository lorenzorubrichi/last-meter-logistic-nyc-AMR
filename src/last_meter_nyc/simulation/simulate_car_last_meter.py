from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.ops import nearest_points

from last_meter_nyc.paths import MODELS_DIR, VISUALIZATION_DIR

VIS_DIR = VISUALIZATION_DIR
MODEL_DIR = MODELS_DIR

BUILDINGS_GEOJSON = VIS_DIR / "OUTPUTbuildings.geojson"
STREET_CONTEXT_GEOJSON = VIS_DIR / "OUTPUTstreet_context.geojson"
AMR_FEATURES_XLSX = MODEL_DIR / "OUTPUTamr_features.xlsx"
AMR_FEATURES_SHEET = "amr_features"
CAR_FEATURES_SHEET = "car_features"
OUTPUT_STATS_XLSX = MODEL_DIR / "car_last_meter_stats.xlsx"

WORK_CRS = "EPSG:2263"
DISPLAY_CRS = "EPSG:4326"
FT_TO_M = 0.3048

ROAD_HALF_WIDTH_FT = 18.0
PARKING_DEPTH_FT = 8.0
PARKING_LENGTH_FT = 20.0
PARKING_GAP_FT = 2.0
PARKING_MODULE_FT = PARKING_LENGTH_FT + PARKING_GAP_FT
ROAD_SEARCH_BUFFER_FT = 180.0

WALKING_SPEED_MPS = 1.4
DROP_OFF_TIME_S = 60.0
MAX_OCCUPANCY_RATIO = 0.95
INDOOR_WALKING_SPEED_MPS = 1.2
ELEVATOR_FLOOR_TIME_S = 4.0

LANDUSE_WAIT_CLASSES = {
    "residential": {1, 2, 3},
    "mixed_commercial": {4, 5},
    "industrial_utility": {6, 7},
    "public_other": {8, 9, 10, 11},
}

BUILDING_TYPE_PENALTY_MAP = {
    "mixed_commercial": 0.00,
    "residential": 0.33,
    "industrial_utility": 0.66,
    "public_other": 1.00,
}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def pd_to_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def load_buildings_and_streets() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    for path in [BUILDINGS_GEOJSON, STREET_CONTEXT_GEOJSON]:
        require_file(path)

    buildings = gpd.read_file(BUILDINGS_GEOJSON).to_crs(WORK_CRS)
    streets = gpd.read_file(STREET_CONTEXT_GEOJSON).to_crs(WORK_CRS)

    if buildings.empty:
        raise ValueError("OUTPUTbuildings.geojson is empty")
    if streets.empty:
        raise ValueError("OUTPUTstreet_context.geojson is empty")

    return buildings, streets


def load_sheet_features(sheet_name: str) -> pd.DataFrame:
    require_file(AMR_FEATURES_XLSX)
    df = pd.read_excel(AMR_FEATURES_XLSX, sheet_name=sheet_name)
    if "bin" not in df.columns:
        raise ValueError(f"'{sheet_name}' must contain a 'bin' column")
    df["bin"] = pd.to_numeric(df["bin"], errors="coerce").astype("Int64")
    return df


def landuse_wait_class(landuse_value) -> str:
    landuse_num = pd.to_numeric(landuse_value, errors="coerce")
    if pd.isna(landuse_num):
        return "public_other"

    landuse_int = int(landuse_num)
    for class_name, class_values in LANDUSE_WAIT_CLASSES.items():
        if landuse_int in class_values:
            return class_name
    return "public_other"


def sample_entry_wait_time_s(landuse_value, rng: random.Random) -> float:
    wait_class = landuse_wait_class(landuse_value)
    if wait_class == "residential":
        return rng.uniform(30.0, 120.0)
    if wait_class == "mixed_commercial":
        return rng.uniform(20.0, 90.0)
    if wait_class == "industrial_utility":
        return rng.uniform(45.0, 150.0)
    return rng.uniform(60.0, 180.0)


def sample_elevator_wait_time_s(landuse_value, rng: random.Random) -> float:
    wait_class = landuse_wait_class(landuse_value)
    if wait_class == "residential":
        return rng.uniform(15.0, 45.0)
    if wait_class == "mixed_commercial":
        return rng.uniform(10.0, 35.0)
    if wait_class == "industrial_utility":
        return rng.uniform(20.0, 60.0)
    return rng.uniform(25.0, 75.0)


def building_type_penalty(landuse_value) -> float:
    return BUILDING_TYPE_PENALTY_MAP.get(landuse_wait_class(landuse_value), 1.0)


def eligible_buildings(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    delivery_lon = pd.to_numeric(buildings.get("delivery_lon"), errors="coerce")
    delivery_lat = pd.to_numeric(buildings.get("delivery_lat"), errors="coerce")
    out = buildings[delivery_lon.notna() & delivery_lat.notna()].copy()
    if out.empty:
        raise ValueError("No buildings with valid delivery point were found")
    return out


def geometry_to_lines(geom) -> list[LineString]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return [g for g in geom.geoms if not g.is_empty]
    if isinstance(geom, Polygon):
        return [LineString(geom.exterior.coords)]
    if isinstance(geom, MultiPolygon):
        return [LineString(poly.exterior.coords) for poly in geom.geoms if not poly.is_empty]
    return []


def delivery_point_from_building(building_row: gpd.GeoSeries) -> Point:
    lon = pd.to_numeric(building_row.get("delivery_lon"), errors="coerce")
    lat = pd.to_numeric(building_row.get("delivery_lat"), errors="coerce")
    if pd.notna(lon) and pd.notna(lat):
        delivery_gs = gpd.GeoSeries([Point(float(lon), float(lat))], crs=DISPLAY_CRS).to_crs(WORK_CRS)
        return delivery_gs.iloc[0]
    return Point(building_row.geometry.centroid.x, building_row.geometry.centroid.y)


def as_polygon(geom) -> Polygon | None:
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        return max(list(geom.geoms), key=lambda g: g.area)
    return None


def nearest_street(building_row: gpd.GeoSeries, streets: gpd.GeoDataFrame) -> gpd.GeoSeries:
    building_geom = building_row.geometry

    if "street_context_id" in building_row.index and "street_context_id" in streets.columns:
        street_id = building_row.get("street_context_id")
        if street_id is not None:
            exact = streets[streets["street_context_id"] == street_id]
            if not exact.empty:
                return exact.iloc[0]

    nearby = streets[streets.geometry.distance(building_geom) <= ROAD_SEARCH_BUFFER_FT]
    search = nearby if not nearby.empty else streets
    distances = search.geometry.distance(building_geom)
    return search.loc[distances.idxmin()]


def nearest_segment_angle(line: LineString, point: Point) -> tuple[Point, float]:
    coords = list(line.coords)
    best_point = line.interpolate(line.project(point))
    best_angle = 0.0
    best_dist = math.inf

    for a, b in zip(coords[:-1], coords[1:]):
        seg = LineString([a, b])
        proj = seg.interpolate(seg.project(point))
        dist = proj.distance(point)
        if dist < best_dist and seg.length > 0:
            best_dist = dist
            best_point = proj
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            best_angle = math.degrees(math.atan2(dy, dx))

    return best_point, best_angle


def make_parking_spot(center_x: float, center_y: float, angle_deg: float) -> Polygon:
    rect = Polygon(
        [
            (-PARKING_LENGTH_FT / 2, -PARKING_DEPTH_FT / 2),
            (PARKING_LENGTH_FT / 2, -PARKING_DEPTH_FT / 2),
            (PARKING_LENGTH_FT / 2, PARKING_DEPTH_FT / 2),
            (-PARKING_LENGTH_FT / 2, PARKING_DEPTH_FT / 2),
        ]
    )
    return translate(rotate(rect, angle_deg, origin=(0, 0)), xoff=center_x, yoff=center_y)


def estimate_parking_capacity(street_geom) -> int:
    lines = geometry_to_lines(street_geom)
    if not lines:
        return 0
    line = max(lines, key=lambda g: g.length)
    return max(1, int(line.length // PARKING_MODULE_FT))


def estimate_shape_penalty(building_geom) -> float:
    polygon = as_polygon(building_geom)
    if polygon is None:
        return 1.0

    minimum_rotated = polygon.minimum_rotated_rectangle
    rect_coords = list(minimum_rotated.exterior.coords)[:-1]
    if len(rect_coords) != 4:
        return 1.0

    lengths = []
    for a, b in zip(rect_coords, rect_coords[1:] + rect_coords[:1]):
        lengths.append(LineString([a, b]).length)
    lengths = sorted(lengths, reverse=True)
    long_axis = lengths[0] if lengths else 1.0
    short_axis = lengths[-1] if lengths else 1.0
    if long_axis <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - (short_axis / long_axis)))


def entrance_point_from_building(building_geom, external_delivery_point: Point) -> Point:
    polygon = as_polygon(building_geom)
    if polygon is None:
        return external_delivery_point
    return nearest_points(polygon.boundary, external_delivery_point)[0]


def elevator_point_from_building(building_geom) -> Point:
    polygon = as_polygon(building_geom)
    if polygon is None:
        return Point(0, 0)
    return polygon.centroid


def sample_internal_dropoff_point(building_geom, rng: random.Random, max_attempts: int = 50) -> Point:
    polygon = as_polygon(building_geom)
    if polygon is None:
        return Point(0, 0)

    centroid = polygon.centroid
    minimum_rotated = polygon.minimum_rotated_rectangle
    rect_coords = list(minimum_rotated.exterior.coords)[:-1]

    if len(rect_coords) != 4:
        return centroid

    edges = []
    for a, b in zip(rect_coords, rect_coords[1:] + rect_coords[:1]):
        edge = LineString([a, b])
        edges.append((edge.length, a, b))
    edges.sort(key=lambda item: item[0], reverse=True)

    _, start_pt, end_pt = edges[0]
    major_dx = end_pt[0] - start_pt[0]
    major_dy = end_pt[1] - start_pt[1]
    major_norm = math.hypot(major_dx, major_dy) or 1.0
    major_ux = major_dx / major_norm
    major_uy = major_dy / major_norm

    minor_dx = -major_uy
    minor_dy = major_ux

    minx, miny, maxx, maxy = minimum_rotated.bounds
    span_x = max(maxx - minx, 1.0)
    span_y = max(maxy - miny, 1.0)
    major_sigma = max(span_x, span_y) / 4.5
    minor_sigma = min(span_x, span_y) / 8.0

    for _ in range(max_attempts):
        major_offset = rng.gauss(0.0, major_sigma)
        minor_offset = rng.gauss(0.0, minor_sigma)
        x = centroid.x + (major_offset * major_ux) + (minor_offset * minor_dx)
        y = centroid.y + (major_offset * major_uy) + (minor_offset * minor_dy)
        candidate = Point(x, y)
        if polygon.contains(candidate):
            return candidate

    return centroid


def sample_building_floor(numfloors_value, rng: random.Random) -> int:
    numfloors = pd.to_numeric(numfloors_value, errors="coerce")
    if pd.isna(numfloors):
        return 1
    max_floor = max(1, int(round(float(numfloors))))
    return rng.randint(1, max_floor)


def generate_parking_spots(street_geom, delivery_point: Point, n_spots: int) -> list[Polygon]:
    lines = geometry_to_lines(street_geom)
    if not lines:
        return []

    line = max(lines, key=lambda g: g.length)
    curb_point, angle = nearest_segment_angle(line, delivery_point)

    tangent_dx = math.cos(math.radians(angle))
    tangent_dy = math.sin(math.radians(angle))

    road_proj = line.interpolate(line.project(delivery_point))
    normal_dx = delivery_point.x - road_proj.x
    normal_dy = delivery_point.y - road_proj.y
    norm = math.hypot(normal_dx, normal_dy) or 1.0
    normal_dx /= norm
    normal_dy /= norm

    offset_from_center = ROAD_HALF_WIDTH_FT - (PARKING_DEPTH_FT / 2)
    anchor_x = curb_point.x + normal_dx * offset_from_center
    anchor_y = curb_point.y + normal_dy * offset_from_center

    total_span = (n_spots - 1) * PARKING_MODULE_FT
    start_shift = -total_span / 2

    spots = []
    for i in range(n_spots):
        shift = start_shift + i * PARKING_MODULE_FT
        cx = anchor_x + tangent_dx * shift
        cy = anchor_y + tangent_dy * shift
        spots.append(make_parking_spot(cx, cy, angle))
    return spots


def simulate_one_run(
    parking_spots: list[Polygon],
    delivery_point: Point,
    street_geom,
    building_geom,
    landuse_value,
    numfloors_value,
    occupancy_ratio: float,
    rng: random.Random,
) -> dict:
    if not parking_spots:
        return {"walking_distance_m": None, "total_time_s": None}

    occupancy_ratio = max(0.0, min(MAX_OCCUPANCY_RATIO, occupancy_ratio))
    occupied_count = min(len(parking_spots) - 1, round(len(parking_spots) * occupancy_ratio))
    occupied_ids = set(rng.sample(range(len(parking_spots)), k=occupied_count)) if occupied_count > 0 else set()

    available_spots = [spot for idx, spot in enumerate(parking_spots) if idx not in occupied_ids]
    if not available_spots:
        return {
            "walking_distance_m": None,
            "entry_wait_time_s": None,
            "elevator_wait_time_s": None,
            "building_floor": None,
            "indoor_distance_m": None,
            "total_time_s": None,
        }

    chosen_spot = min(available_spots, key=lambda spot: spot.centroid.distance(delivery_point))
    street_lines = geometry_to_lines(street_geom)
    if street_lines:
        line = max(street_lines, key=lambda g: g.length)
        parking_proj = line.interpolate(line.project(chosen_spot.centroid))
        delivery_proj = line.interpolate(line.project(delivery_point))
        along_sidewalk_ft = abs(line.project(delivery_proj) - line.project(parking_proj))
        direct_to_delivery_ft = delivery_proj.distance(delivery_point)
        one_way_distance_m = (along_sidewalk_ft + direct_to_delivery_ft) * FT_TO_M
    else:
        one_way_distance_m = chosen_spot.centroid.distance(delivery_point) * FT_TO_M

    entrance_point = entrance_point_from_building(building_geom, delivery_point)
    elevator_point = elevator_point_from_building(building_geom)
    internal_dropoff = sample_internal_dropoff_point(building_geom, rng)
    building_floor = sample_building_floor(numfloors_value, rng)

    entry_wait_time_s = sample_entry_wait_time_s(landuse_value, rng)
    if building_floor > 1:
        entrance_to_elevator_m = entrance_point.distance(elevator_point) * FT_TO_M
        elevator_to_dropoff_m = elevator_point.distance(internal_dropoff) * FT_TO_M
        indoor_one_way_m = entrance_to_elevator_m + elevator_to_dropoff_m
        elevator_wait_time_s = sample_elevator_wait_time_s(landuse_value, rng)
        elevator_travel_time_s = 2.0 * building_floor * ELEVATOR_FLOOR_TIME_S
    else:
        indoor_one_way_m = entrance_point.distance(internal_dropoff) * FT_TO_M
        elevator_wait_time_s = 0.0
        elevator_travel_time_s = 0.0

    indoor_round_trip_m = 2.0 * indoor_one_way_m
    indoor_walk_time_s = indoor_round_trip_m / INDOOR_WALKING_SPEED_MPS

    total_time_s = (
        (2.0 * one_way_distance_m / WALKING_SPEED_MPS)
        + entry_wait_time_s
        + (2.0 * elevator_wait_time_s)
        + elevator_travel_time_s
        + indoor_walk_time_s
        + DROP_OFF_TIME_S
    )
    return {
        "walking_distance_m": one_way_distance_m,
        "entry_wait_time_s": entry_wait_time_s,
        "elevator_wait_time_s": elevator_wait_time_s,
        "building_floor": building_floor,
        "indoor_distance_m": indoor_round_trip_m,
        "total_time_s": total_time_s,
    }


def summarize_runs(records: list[dict], scenario_name: str) -> dict:
    distances = [r["walking_distance_m"] for r in records if r["walking_distance_m"] is not None]
    times = [r["total_time_s"] for r in records if r["total_time_s"] is not None]
    indoor_distances = [r["indoor_distance_m"] for r in records if r["indoor_distance_m"] is not None]
    floors = [r["building_floor"] for r in records if r["building_floor"] is not None]

    distance_series = pd.Series(distances, dtype=float)
    time_series = pd.Series(times, dtype=float)
    indoor_distance_series = pd.Series(indoor_distances, dtype=float)
    floor_series = pd.Series(floors, dtype=float)

    return {
        f"{scenario_name}_n_valid_runs": int(len(distances)),
        f"{scenario_name}_distance_mean_m": float(distance_series.mean()) if not distance_series.empty else None,
        f"{scenario_name}_distance_std_m": float(distance_series.std(ddof=1)) if len(distance_series) > 1 else 0.0 if len(distance_series) == 1 else None,
        f"{scenario_name}_indoor_distance_mean_m": float(indoor_distance_series.mean()) if not indoor_distance_series.empty else None,
        f"{scenario_name}_indoor_distance_std_m": float(indoor_distance_series.std(ddof=1)) if len(indoor_distance_series) > 1 else 0.0 if len(indoor_distance_series) == 1 else None,
        f"{scenario_name}_floor_mean": float(floor_series.mean()) if not floor_series.empty else None,
        f"{scenario_name}_floor_std": float(floor_series.std(ddof=1)) if len(floor_series) > 1 else 0.0 if len(floor_series) == 1 else None,
        f"{scenario_name}_time_mean_s": float(time_series.mean()) if not time_series.empty else None,
        f"{scenario_name}_time_std_s": float(time_series.std(ddof=1)) if len(time_series) > 1 else 0.0 if len(time_series) == 1 else None,
    }


def choose_random_bins(buildings: gpd.GeoDataFrame, n_buildings: int, seed: int | None) -> list[int]:
    eligible = eligible_buildings(buildings)
    rng = random.Random(seed)
    bin_values = [int(v) for v in pd_to_numeric(eligible["bin"]).dropna().tolist()]
    sample_size = min(max(1, n_buildings), len(bin_values))
    return rng.sample(bin_values, k=sample_size)


def run_simulation(
    n_buildings: int,
    n_runs: int,
    seed: int | None,
) -> pd.DataFrame:
    buildings, streets = load_buildings_and_streets()
    amr_features = load_sheet_features(AMR_FEATURES_SHEET)
    car_features = load_sheet_features(CAR_FEATURES_SHEET)
    selected_bins = choose_random_bins(buildings, n_buildings=n_buildings, seed=seed)
    rng = random.Random(seed)

    results = []
    for building_bin in selected_bins:
        building = eligible_buildings(buildings)[pd_to_numeric(eligible_buildings(buildings)["bin"]) == building_bin].iloc[0]
        delivery_point = delivery_point_from_building(building)
        street = nearest_street(building, streets)
        parking_capacity = estimate_parking_capacity(street.geometry)
        parking_spots = generate_parking_spots(street.geometry, delivery_point, n_spots=parking_capacity)

        car_feature_match = car_features[car_features["bin"] == building_bin]
        amr_feature_match = amr_features[amr_features["bin"] == building_bin]
        base_ratio = 0.0
        floors_norm = 0.0
        road_to_delivery_distance_norm = 0.0
        if not car_feature_match.empty and "CurbCrowdingPenalty" in car_feature_match.columns:
            penalty = pd.to_numeric(car_feature_match.iloc[0]["CurbCrowdingPenalty"], errors="coerce")
            if pd.notna(penalty):
                base_ratio = float(penalty)
            if "a2_RoadToDeliveryDistance_norm" in car_feature_match.columns:
                distance_norm = pd.to_numeric(
                    car_feature_match.iloc[0]["a2_RoadToDeliveryDistance_norm"],
                    errors="coerce",
                )
                if pd.notna(distance_norm):
                    road_to_delivery_distance_norm = float(distance_norm)
        if not amr_feature_match.empty and "a1_Floors_norm" in amr_feature_match.columns:
            floors_value = pd.to_numeric(amr_feature_match.iloc[0]["a1_Floors_norm"], errors="coerce")
            if pd.notna(floors_value):
                floors_norm = float(floors_value)
            if road_to_delivery_distance_norm == 0.0 and "a2_RoadToDeliveryDistance_norm" in amr_feature_match.columns:
                distance_norm = pd.to_numeric(
                    amr_feature_match.iloc[0]["a2_RoadToDeliveryDistance_norm"],
                    errors="coerce",
                )
                if pd.notna(distance_norm):
                    road_to_delivery_distance_norm = float(distance_norm)
        base_ratio = max(0.0, min(MAX_OCCUPANCY_RATIO, base_ratio))

        base_runs = [
            simulate_one_run(
                parking_spots,
                delivery_point,
                street.geometry,
                building.geometry,
                building.get("landuse"),
                building.get("numfloors"),
                occupancy_ratio=base_ratio,
                rng=rng,
            )
            for _ in range(n_runs)
        ]

        row = {
            "bin": building_bin,
            "delivery_lon": float(building.get("delivery_lon")),
            "delivery_lat": float(building.get("delivery_lat")),
            "a1_Floors_norm": floors_norm,
            "CurbCrowdingPenalty": base_ratio,
            "a2_RoadToDeliveryDistance_norm": road_to_delivery_distance_norm,
            "ShapePenalty_norm": float(estimate_shape_penalty(building.geometry)),
            "BuildingTypePenalty_norm": float(building_type_penalty(building.get("landuse"))),
            "parking_capacity": parking_capacity,
            "base_occupancy_ratio": base_ratio,
        }
        row.update(summarize_runs(base_runs, "base"))
        results.append(row)

    return pd.DataFrame(results)


def save_results(df: pd.DataFrame) -> None:
    OUTPUT_STATS_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_STATS_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="car_last_meter", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monte Carlo simulation for car last-meter walking distance/time from parking to delivery point."
    )
    parser.add_argument("--n-buildings", type=int, default=500, help="Number of buildings to evaluate.")
    parser.add_argument("--n-runs", type=int, default=100, help="Monte Carlo runs per building and per scenario.")
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

    for _, row in results.iterrows():
        print(
            f"BIN {int(row['bin'])} | "
            f"base_distance_mean={row['base_distance_mean_m']:.2f} m | "
            f"base_time_mean={row['base_time_mean_s']:.2f} s"
        )

    print(f"Excel saved: {OUTPUT_STATS_XLSX}")


if __name__ == "__main__":
    main()
