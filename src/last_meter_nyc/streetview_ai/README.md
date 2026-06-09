# Street View AI Pipeline

This branch adds visual access features from Street View imagery.

The current experiment was run on about 7,000 buildings because AI image
analysis has a direct API cost. The scripts are designed to be incremental:
keep the existing output CSV and run `extract_visual_features.py --no-overwrite`
after adding more downloaded images.

Typical order:

1. `build_streetview_targets.py`
2. `prepare_streetview_requests.py`
3. `download_streetview_images.py`
4. `extract_visual_features.py --no-overwrite`
5. `merge_ai_visual_features.py`
6. `train_ai_informed_variants.py`

