from __future__ import annotations

from pathlib import Path

import branca.colormap as bcm
import folium
import geopandas as gpd
import pandas as pd

from last_meter_nyc.paths import MODELS_DIR, RAW_DATA_DIR, VISUALIZATION_DIR

MODEL_DIR = MODELS_DIR
VIS_DIR = VISUALIZATION_DIR
RAWDATA_DIR = RAW_DATA_DIR

FINAL_DB_XLSX = MODEL_DIR / "final_last_meter_database.xlsx"
FINAL_DB_SHEET = "last_meter_database"
BUILDINGS_GEOJSON = VIS_DIR / "OUTPUTbuildings.geojson"
NTA_GEOJSON = RAWDATA_DIR / "nyc_neighborhoods.geojson"

OUTPUT_HTML = VIS_DIR / "final_last_meter_map.html"
OUTPUT_GEOJSON = VIS_DIR / "final_last_meter_map.geojson"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def load_final_database() -> pd.DataFrame:
    require_file(FINAL_DB_XLSX)
    df = pd.read_excel(FINAL_DB_XLSX, sheet_name=FINAL_DB_SHEET)
    required = [
        "bin",
        "address_lon",
        "address_lat",
        "car_last_meter_mean_s",
        "amr_last_meter_mean_s",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in final_last_meter_database.xlsx: {missing}")

    df["bin"] = pd.to_numeric(df["bin"], errors="coerce").astype("Int64")
    df["car_last_meter_mean_s"] = pd.to_numeric(df["car_last_meter_mean_s"], errors="coerce")
    df["amr_last_meter_mean_s"] = pd.to_numeric(df["amr_last_meter_mean_s"], errors="coerce")
    df = df.dropna(subset=["bin"]).copy()
    df["compare_ratio"] = df["car_last_meter_mean_s"] / df["amr_last_meter_mean_s"]
    return df


def load_buildings() -> gpd.GeoDataFrame:
    require_file(BUILDINGS_GEOJSON)
    gdf = gpd.read_file(BUILDINGS_GEOJSON)
    if "bin" not in gdf.columns:
        raise ValueError("OUTPUTbuildings.geojson missing 'bin'")
    gdf["bin"] = pd.to_numeric(gdf["bin"], errors="coerce").astype("Int64")
    return gdf


def load_relevant_nta(buildings_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    require_file(NTA_GEOJSON)
    nta = gpd.read_file(NTA_GEOJSON).to_crs(buildings_gdf.crs)
    union_geom = buildings_gdf.unary_union
    return nta[nta.intersects(union_geom)].copy()


def merge_geometries(final_df: pd.DataFrame, buildings_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    merged = buildings_gdf.merge(final_df, how="inner", on="bin")
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=buildings_gdf.crs)


def tooltip_html(row: pd.Series, mode_label: str, displayed_value: float, compare_ratio: float | None = None) -> str:
    parts = [
        f"<strong>BIN:</strong> {row['bin']}",
        f"<strong>View:</strong> {mode_label}",
        f"<strong>Displayed value:</strong> {displayed_value:.1f}",
    ]
    if compare_ratio is not None:
        parts.append(f"<strong>Car / AMR ratio:</strong> {compare_ratio:.3f}")

    for label, col in [
        ("Address lon", "address_lon"),
        ("Address lat", "address_lat"),
        ("Floors raw", "raw_numfloors"),
        ("Distance to sidewalk raw (m)", "raw_distance_to_sidewalk_m"),
        ("Car last-meter time (s)", "car_last_meter_mean_s"),
        ("AMR last-meter time (s)", "amr_last_meter_mean_s"),
    ]:
        if col in row.index and pd.notna(row[col]):
            value = row[col]
            if isinstance(value, float):
                parts.append(f"<strong>{label}:</strong> {value:.1f}")
            else:
                parts.append(f"<strong>{label}:</strong> {value}")

    return "<br>".join(parts)


def add_layer(m: folium.Map, buildings_gdf: gpd.GeoDataFrame, layer_name: str, value_col: str, tooltip_mode: str, colormap: bcm.LinearColormap, show: bool = False, compare: bool = False) -> tuple[str, str]:
    feature_group = folium.FeatureGroup(name=layer_name, overlay=True, control=False, show=show)

    for _, row in buildings_gdf.iterrows():
        value = pd.to_numeric(row.get(value_col), errors="coerce")
        if pd.isna(value):
            continue
        color_value = max(0.5, min(1.5, float(value))) if compare else float(value)
        color = colormap(color_value)
        tooltip = tooltip_html(
            row,
            mode_label=tooltip_mode,
            displayed_value=float(value),
            compare_ratio=float(value) if compare else None,
        )
        folium.GeoJson(
            row["geometry"].__geo_interface__,
            style_function=lambda _f, c=color: {
                "fillColor": c,
                "color": "#444444",
                "weight": 0.4,
                "fillOpacity": 0.78,
            },
            tooltip=tooltip,
        ).add_to(feature_group)

    feature_group.add_to(m)
    return layer_name, feature_group.get_name()


def add_custom_controls(m: folium.Map, map_js_name: str, layer_js_map: dict[str, str]) -> None:
    custom_html = """
    <div id="scenario-panel" style="
        position: fixed;
        top: 18px;
        right: 18px;
        z-index: 9999;
        background: white;
        border: 1px solid #bbb;
        border-radius: 10px;
        padding: 12px 14px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.18);
        font-size: 13px;
        min-width: 220px;
    ">
      <div style="font-weight: 700; margin-bottom: 8px;">Last-meter map</div>
      <div style="margin-bottom: 8px;">
        <div style="font-weight: 600;">View</div>
        <label><input type="radio" name="viewMode" value="car" checked> Last mile with car</label><br>
        <label><input type="radio" name="viewMode" value="amr"> Last mile with AMR</label><br>
        <label><input type="radio" name="viewMode" value="compare"> Compare</label>
      </div>
      <div id="legend-box" style="margin-top:10px; padding-top:8px; border-top:1px solid #ddd;">
        <div id="legend-title" style="font-weight:600; margin-bottom:4px;">Legend</div>
        <div style="display:flex; align-items:center; gap:6px;">
          <span style="display:inline-block; width:18px; height:10px; background:#1a9850;"></span><span id="legend-low">Lower time</span>
        </div>
        <div style="display:flex; align-items:center; gap:6px;">
          <span style="display:inline-block; width:18px; height:10px; background:#fee08b;"></span><span id="legend-mid">Medium</span>
        </div>
        <div style="display:flex; align-items:center; gap:6px;">
          <span style="display:inline-block; width:18px; height:10px; background:#d73027;"></span><span id="legend-high">Higher time</span>
        </div>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(custom_html))

    script = """
    (function() {
      const mapName = '__MAP_JS_NAME__';
      const layerVarNames = {
        car: '__CAR__',
        amr: '__AMR__',
        compare: '__COMPARE__'
      };

      function selectedValue(name) {
        const el = document.querySelector('input[name="' + name + '"]:checked');
        return el ? el.value : null;
      }

      function wireControls() {
        const mapRef = window[mapName];
        if (!mapRef) return false;

        const layerMap = {};
        for (const [key, varName] of Object.entries(layerVarNames)) {
          const layer = window[varName];
          if (!layer) return false;
          layerMap[key] = layer;
        }

        function removeAllLayers() {
          Object.values(layerMap).forEach(function(layer) {
            if (layer && mapRef.hasLayer(layer)) {
              mapRef.removeLayer(layer);
            }
          });
        }

        function updateLegend(view) {
          const legendTitle = document.getElementById('legend-title');
          const legendLow = document.getElementById('legend-low');
          const legendMid = document.getElementById('legend-mid');
          const legendHigh = document.getElementById('legend-high');
          if (!legendTitle || !legendLow || !legendMid || !legendHigh) return;

          if (view === 'compare') {
            legendTitle.textContent = 'Legend: Car / AMR ratio';
            legendLow.textContent = 'Car faster';
            legendMid.textContent = 'Similar';
            legendHigh.textContent = 'AMR faster';
          } else {
            legendTitle.textContent = 'Legend: Last-meter time';
            legendLow.textContent = 'Lower time';
            legendMid.textContent = 'Medium';
            legendHigh.textContent = 'Higher time';
          }
        }

        function updateView() {
          const view = selectedValue('viewMode') || 'car';
          removeAllLayers();
          if (layerMap[view]) mapRef.addLayer(layerMap[view]);
          updateLegend(view);
        }

        document.querySelectorAll('input[name="viewMode"]').forEach(function(el) {
          el.addEventListener('change', updateView);
        });

        updateView();
        return true;
      }

      let attempts = 0;
      const timer = window.setInterval(function() {
        attempts += 1;
        if (wireControls() || attempts > 50) {
          window.clearInterval(timer);
        }
      }, 200);
    })();
    """
    script = (script
        .replace('__MAP_JS_NAME__', map_js_name)
        .replace('__CAR__', layer_js_map['car'])
        .replace('__AMR__', layer_js_map['amr'])
        .replace('__COMPARE__', layer_js_map['compare']))
    m.get_root().script.add_child(folium.Element(script))


def make_map(buildings_gdf: gpd.GeoDataFrame, nta_gdf: gpd.GeoDataFrame) -> folium.Map:
    center = [float(buildings_gdf.geometry.centroid.y.mean()), float(buildings_gdf.geometry.centroid.x.mean())]
    m = folium.Map(location=center, zoom_start=15, tiles='CartoDB positron')

    if not nta_gdf.empty:
        folium.GeoJson(
            nta_gdf.to_json(),
            name='Neighborhood boundary',
            style_function=lambda _f: {
                'fillOpacity': 0.0,
                'color': '#1f1f1f',
                'weight': 2,
            },
        ).add_to(m)

    time_series = pd.concat([
        pd.to_numeric(buildings_gdf['car_last_meter_mean_s'], errors='coerce'),
        pd.to_numeric(buildings_gdf['amr_last_meter_mean_s'], errors='coerce'),
    ], ignore_index=True).dropna()
    time_low = float(time_series.quantile(0.05)) if not time_series.empty else 0.0
    time_high = float(time_series.quantile(0.95)) if not time_series.empty else 1.0
    if time_high <= time_low:
        time_high = time_low + 1.0

    time_colormap = bcm.LinearColormap(colors=['#1a9850', '#fee08b', '#d73027'], vmin=time_low, vmax=time_high)
    compare_colormap = bcm.LinearColormap(colors=['#8b0000', '#f4d35e', '#006d2c'], vmin=0.5, vmax=1.5)

    layer_js_map = dict([
        add_layer(m, buildings_gdf, 'car', 'car_last_meter_mean_s', 'Car', time_colormap, show=True),
        add_layer(m, buildings_gdf, 'amr', 'amr_last_meter_mean_s', 'AMR', time_colormap),
        add_layer(m, buildings_gdf, 'compare', 'compare_ratio', 'Compare', compare_colormap, compare=True),
    ])

    add_custom_controls(m, m.get_name(), layer_js_map)
    return m


def main() -> None:
    final_df = load_final_database()
    buildings_gdf = load_buildings()
    merged_gdf = merge_geometries(final_df, buildings_gdf)
    nta_gdf = load_relevant_nta(merged_gdf)

    OUTPUT_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    merged_gdf.to_file(OUTPUT_GEOJSON, driver='GeoJSON')

    m = make_map(merged_gdf, nta_gdf)
    m.save(str(OUTPUT_HTML))

    print(f'Mapped buildings: {len(merged_gdf)}')
    print(f'GeoJSON saved: {OUTPUT_GEOJSON}')
    print(f'HTML map saved: {OUTPUT_HTML}')


if __name__ == '__main__':
    main()
