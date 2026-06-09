# Scaling the Street View AI Branch

The current visual AI branch contains about 7,000 analyzed buildings because AI
image analysis has a direct API cost. This is a budget constraint, not a
structural limitation of the pipeline.

## Incremental Scaling Strategy

1. Build or extend the target list.
2. Prepare Street View camera requests.
3. Download additional images.
4. Run AI extraction with `--no-overwrite`.
5. Merge visual features again.
6. Rerun AI-informed simulations and model training.

## Recommended Commands

```bash
python -m last_meter_nyc.streetview_ai.build_streetview_targets
python -m last_meter_nyc.streetview_ai.prepare_streetview_requests
python -m last_meter_nyc.streetview_ai.download_streetview_images
python -m last_meter_nyc.streetview_ai.extract_visual_features --no-overwrite
python -m last_meter_nyc.streetview_ai.merge_ai_visual_features
python -m last_meter_nyc.modeling.train_ai_informed_variants
```

## Cost Control

For pilots:

```bash
python -m last_meter_nyc.streetview_ai.extract_visual_features --limit 100
```

For production continuation:

```bash
python -m last_meter_nyc.streetview_ai.extract_visual_features --no-overwrite
```

The `--no-overwrite` mode preserves already analyzed buildings and only fills
missing rows.

