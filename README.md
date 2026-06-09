# NYC Last-Meter Delivery: Geospatial, Simulation, and Street View AI Pipeline

This repository contains a research pipeline for estimating last-meter delivery
times in New York City buildings. It combines geospatial feature engineering,
Monte Carlo simulation, regression models, and an optional Street View AI branch
for visual access barriers.

## 1 What This Project Does

The pipeline supports two levels of analysis:

1. **Citywide baseline pipeline**
   - builds geospatial features for NYC buildings;
   - simulates car and autonomous mobile robot (AMR) last-meter delivery times;
   - trains regression models on simulated samples;
   - predicts last-meter times for the wider building dataset.

2. **Street View AI-informed branch**
   - downloads one street-level image per building target;
   - extracts visual access features with an AI model;
   - adds AI-informed penalties to the car and AMR simulations;
   - compares regressions trained with and without visual AI features.

The visual AI branch was run on about 7,000 buildings due to API cost
constraints. The code is designed to scale incrementally to more buildings.

## 2 Repository Structure

```text
src/last_meter_nyc/

docs/
  methodology.md
  parking_availability.md
  streetview_scaling.md
  github_publication_checklist.md
  raw_data_sources.md

data/
  raw/             small schema-compatible sample extracts
  schema/          selected publication fields and descriptions
  processed/       publication-ready processed data, if included

models/            trained model bundles, if included
notebooks/         optional analysis notebooks
```

Large raw datasets, API keys, downloaded images, and cache files should not be
committed to GitHub. The files in `data/raw/` are small schema-compatible
samples; replace them with complete raw files to reproduce the full dataset.

## 3 Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, use:

```bash
source .venv/bin/activate
```

## 4 Environment Variables

Copy `.env.example` to `.env` and add local keys if you intend to run the
Street View and AI extraction steps.

```text
GOOGLE_STREETVIEW_API_KEY=...
OPENAI_API_KEY=...
```

## 5 Main Workflows

### 5.1. Baseline Citywide Workflow

```bash
python -m last_meter_nyc.data.build_geospatial_features
python -m last_meter_nyc.modeling.train_baseline_models --n-buildings 10000 --n-runs 100
python -m last_meter_nyc.data.build_complete_dataset
```

### 5.2. Street View AI Workflow

```bash
python -m last_meter_nyc.streetview_ai.build_streetview_targets
python -m last_meter_nyc.streetview_ai.prepare_streetview_requests
python -m last_meter_nyc.streetview_ai.download_streetview_images
python -m last_meter_nyc.streetview_ai.extract_visual_features --no-overwrite
python -m last_meter_nyc.streetview_ai.merge_ai_visual_features
python -m last_meter_nyc.modeling.train_ai_informed_variants
```

All public paths are repository-relative. The included raw files are samples;
for full reproduction, replace them with the complete files described in
`docs/raw_data_sources.md`.

## 6 Public Dataset Export

Internal feature names are preserved for backward compatibility, but public
exports should use clearer names and the selected field lists in:

```text
data/schema/base_fields.csv
data/schema/ai_fields.csv
```

These schema files define the final column order, the fields kept in the
published CSVs, and the descriptions used in the data dictionaries.

```bash
python -m last_meter_nyc.data.export_public_dataset ^
  --input data/processed/complete_last_meter_dataset_extended.csv ^
  --output data/processed/complete_last_meter_dataset_public.csv ^
  --dictionary-output data/processed/complete_last_meter_dataset_dictionary.csv ^
  --schema data/schema/base_fields.csv

python -m last_meter_nyc.data.export_public_dataset ^
  --input data/processed/complete_last_meter_dataset_ai_subset.csv ^
  --output data/processed/last_meter_ai_subset_public.csv ^
  --dictionary-output data/processed/last_meter_ai_subset_dictionary.csv ^
  --schema data/schema/ai_fields.csv
```

Example rename:

```text
CurbCrowdingPenalty -> curb_parking_difficulty_penalty_R
ParkingScarcity_advantage -> parking_meter_scarcity_penalty
a1_Floors_norm -> building_floor_count_norm_R
```

## 7 Static CSV Export Tool

The repository includes `csv_filter_exporter.html`, a browser-only tool for
creating custom CSV extracts without displaying the dataset rows on the page.
It can:

- load the baseline public dataset and the Street View AI subset together when
  served through GitHub Pages or another static web host;
- join the two tables with the `bin` field;
- select fields from either table;
- show dictionary descriptions when hovering over field names;
- apply simple filters using fields from either table;
- download the joined and filtered result as a new CSV.

Exported column names are prefixed with their source table, for example
`baseline__borough` or `ai__streetview_stairs_detected`, to avoid ambiguity
when both tables contain a field with the same name.

When browsing the repository on github.com, clicking the HTML file usually
opens the source code view. To run it online, enable GitHub Pages for the
repository and open:

```text
https://<username>.github.io/<repository>/csv_filter_exporter.html
```

For local use, serve the repository folder with a small static web server and
open `csv_filter_exporter.html`; this lets the browser read the CSV files from
`data/processed/`.

## 8 Citation

If this repository is used in a thesis, paper, or derivative project, cite the
repository and the original NYC open-data sources used by the pipeline.


