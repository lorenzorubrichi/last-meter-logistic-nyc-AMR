from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from last_meter_nyc.paths import MODELS_DIR, VISUALIZATION_DIR

MODEL_DIR = MODELS_DIR
FEATURE_XLSX = MODEL_DIR / "feature_effect_analysis" / "feature_effect_analysis.xlsx"
SENSITIVITY_XLSX = MODEL_DIR / "oat_sensitivity" / "last_meter_oat_sensitivity.xlsx"
OUTPUT_DIR = VISUALIZATION_DIR / "presentation_plots"
AMR_PDP_PNG = OUTPUT_DIR / "amr_pdp_plots.png"
AMR_SENSITIVITY_PNG = OUTPUT_DIR / "amr_sensitivity_errorbars.png"

FEATURE_AXIS_LABELS = {
    "b3_UrbanActivity_norm": "Urban activity score (normalized)",
    "BuildingTypePenalty_norm": "Building type difficulty score (normalized)",
    "a1_Floors_norm": "Number of floors (normalized)",
    "b1_Population_norm": "Population density score (normalized)",
    "b2_PedestrianPresence_norm": "Pedestrian presence score (normalized)",
}


def load_amr_pdp() -> pd.DataFrame:
    df = pd.read_excel(FEATURE_XLSX, sheet_name="pdp_data")
    return df[df["output_column"] == "amr_last_meter_mean_s"].copy()


def load_amr_sensitivity() -> pd.DataFrame:
    df = pd.read_excel(SENSITIVITY_XLSX, sheet_name="scenario_summary")
    return df[df["mode"] == "AMR"].copy()


def plot_amr_pdp(df: pd.DataFrame) -> None:
    features = df["feature_label"].dropna().unique().tolist()
    features = sorted(features, key=lambda x: ["Urban Activity", "Building Type", "Normalized Floors"].index(x) if x in ["Urban Activity", "Building Type", "Normalized Floors"] else 999)
    fig, axes = plt.subplots(len(features), 1, figsize=(9, 3.3 * len(features)), sharex=False)
    if len(features) == 1:
        axes = [axes]

    for ax, feature_label in zip(axes, features):
        work = df[df["feature_label"] == feature_label].sort_values("feature_value")
        ax.plot(work["feature_value"], work["partial_dependence"], color="#2563eb", linewidth=2.2)
        ax.set_title(feature_label, fontsize=14, pad=10)
        feature_code = work["feature"].iloc[0]
        ax.set_xlabel(FEATURE_AXIS_LABELS.get(feature_code, feature_label), fontsize=11)
        ax.set_ylabel("Partial dependence", fontsize=11)
        ax.grid(alpha=0.2, linestyle="--")

    fig.suptitle("AMR Partial Dependence Plots", fontsize=18, y=0.995)
    fig.tight_layout()
    fig.savefig(AMR_PDP_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_amr_sensitivity(df: pd.DataFrame) -> None:
    parameter_order = [
        "Customer response timeout (s)",
        "Customer response probability scale",
        "Helper search timeout (s)",
        "Elevator wait scale",
        "Customer walking speed (m/s)",
    ]
    level_colors = {
        "low": "#d97706",
        "base": "#2563eb",
        "high": "#059669",
    }
    level_offsets = {
        "low": -0.22,
        "base": 0.0,
        "high": 0.22,
    }

    base_positions = {label: idx for idx, label in enumerate(parameter_order)}
    fig, ax = plt.subplots(figsize=(11, 6.5))

    for _, row in df.iterrows():
        label = row["parameter_label"]
        if label not in base_positions:
            continue
        y = base_positions[label] + level_offsets.get(row["level"], 0.0)
        ax.errorbar(
            x=row["mean_time_s"],
            y=y,
            xerr=row["std_building_mean_time_s"],
            fmt="o",
            markersize=8,
            color=level_colors.get(row["level"], "#111827"),
            ecolor=level_colors.get(row["level"], "#111827"),
            elinewidth=2,
            capsize=5,
        )
        ax.text(
            row["mean_time_s"] + row["std_building_mean_time_s"] + 3,
            y,
            row["level"],
            va="center",
            fontsize=9,
            color=level_colors.get(row["level"], "#111827"),
        )

    ax.set_yticks(list(base_positions.values()))
    ax.set_yticklabels(parameter_order, fontsize=11)
    ax.set_xlabel("Mean AMR last-meter time (s)", fontsize=12)
    ax.set_ylabel("Simulation parameter", fontsize=12)
    ax.set_title("AMR OAT Sensitivity Analysis\nMean time with standard-deviation whiskers", fontsize=16)
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(AMR_SENSITIVITY_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    amr_pdp_df = load_amr_pdp()
    amr_sens_df = load_amr_sensitivity()
    plot_amr_pdp(amr_pdp_df)
    plot_amr_sensitivity(amr_sens_df)
    print(f"[PLOT] Saved: {AMR_PDP_PNG}")
    print(f"[PLOT] Saved: {AMR_SENSITIVITY_PNG}")


if __name__ == "__main__":
    main()
