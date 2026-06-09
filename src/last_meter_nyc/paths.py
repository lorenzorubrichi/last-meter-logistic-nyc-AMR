"""Repository-relative paths used by the public pipeline.

The original research workspace used local absolute paths. Public code should
resolve all inputs and outputs relative to the repository root.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

MODELS_DIR = REPO_ROOT / "models"
OUTPUTS_DIR = REPO_ROOT / "outputs"
VISUALIZATION_DIR = OUTPUTS_DIR / "visualization"
STREETVIEW_OUTPUT_DIR = OUTPUTS_DIR / "streetview_ai"
STREETVIEW_IMAGE_DIR = STREETVIEW_OUTPUT_DIR / "images"
STREETVIEW_REQUEST_DIR = INTERIM_DATA_DIR / "streetview_ai"
AI_EXPERIMENT_DIR = MODELS_DIR / "ai_penalty_shared"


def ensure_standard_directories() -> None:
    """Create the standard repository output directories."""
    for path in [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        SAMPLE_DATA_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        VISUALIZATION_DIR,
        STREETVIEW_OUTPUT_DIR,
        STREETVIEW_IMAGE_DIR,
        STREETVIEW_REQUEST_DIR,
        AI_EXPERIMENT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)

