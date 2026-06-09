"""Export a publication-friendly dataset with clear feature names.

This script does not recompute the pipeline. It takes an existing processed
CSV, renames ambiguous internal feature names, optionally writes a data
dictionary, and saves a clean public CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from last_meter_nyc.feature_names import FEATURE_DESCRIPTIONS, rename_for_publication


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a public dataset with clear feature names.")
    parser.add_argument("--input", type=Path, required=True, help="Processed CSV produced by the research pipeline.")
    parser.add_argument("--output", type=Path, required=True, help="Publication-ready CSV path.")
    parser.add_argument(
        "--dictionary-output",
        type=Path,
        default=None,
        help="Optional CSV data dictionary for the renamed columns.",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=None,
        help="Optional CSV with columns: column, description. Output is filtered and ordered by this schema after public renaming.",
    )
    return parser.parse_args()


def load_schema(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["column", "description"])
    if not path.exists():
        raise FileNotFoundError(f"Missing schema file: {path}")
    schema = pd.read_csv(path)
    if "column" not in schema.columns:
        raise ValueError(f"Schema must contain a 'column' field: {path}")
    if "description" not in schema.columns:
        schema["description"] = ""
    if "order" in schema.columns:
        schema = schema.sort_values("order", kind="stable")
    schema["column"] = schema["column"].astype(str).str.strip()
    schema["description"] = schema["description"].fillna("").astype(str).str.strip()
    return schema[schema["column"] != ""].copy()


def apply_schema(public_df: pd.DataFrame, schema: pd.DataFrame) -> pd.DataFrame:
    if schema.empty:
        return public_df
    requested = list(schema["column"])
    missing = [column for column in requested if column not in public_df.columns]
    if missing:
        raise ValueError(
            "Schema columns missing from public dataframe: "
            + ", ".join(missing[:30])
            + (" ..." if len(missing) > 30 else "")
        )
    return public_df[requested].copy()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, low_memory=False)
    rename_map = rename_for_publication(list(df.columns))
    public_df = df.rename(columns=rename_map)
    schema = load_schema(args.schema)
    public_df = apply_schema(public_df, schema)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    public_df.to_csv(args.output, index=False)

    if args.dictionary_output:
        schema_descriptions = {
            str(row["column"]): str(row["description"])
            for _, row in schema.iterrows()
            if str(row["description"]).strip()
        }
        dictionary_rows = []
        for column in public_df.columns:
            dictionary_rows.append(
                {
                    "column": column,
                    "legacy_name": next((old for old, new in rename_map.items() if new == column), ""),
                    "description": schema_descriptions.get(column, FEATURE_DESCRIPTIONS.get(column, "")),
                }
            )
        args.dictionary_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(dictionary_rows).to_csv(args.dictionary_output, index=False)

    print(f"Saved public dataset: {args.output}")
    if args.dictionary_output:
        print(f"Saved data dictionary: {args.dictionary_output}")


if __name__ == "__main__":
    main()
