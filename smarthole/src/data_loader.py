"""Data loading utilities for Smarthole."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

LOGGER = logging.getLogger(__name__)

EXPECTED_FILENAMES: List[str] = [
    "Dataset_Smooth.csv",
    "raw_data_plain.csv",
    "Database_rough.csv",
    "Dataset_rough2.csv",
    "Database_rough3.csv",
    "Dataset_rough4.csv",
    "raw_data.csv",
    "raw_data_pav_1_.csv",
]

REQUIRED_COLUMNS: List[str] = ["pc_time", "esp_time", "az"]


def _validate_expected_files(raw_dir: Path) -> List[Path]:
    """Validate all expected CSV files exist.

    Parameters
    ----------
    raw_dir : Path
        Directory containing raw sensor CSV files.

    Returns
    -------
    List[Path]
        Ordered list of file paths for expected files.
    """
    missing: List[str] = [name for name in EXPECTED_FILENAMES if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required raw files in '{raw_dir}': {', '.join(missing)}"
        )
    return [raw_dir / name for name in EXPECTED_FILENAMES]


def load_raw_data(raw_dir: Path) -> pd.DataFrame:
    """Load, validate, and concatenate all sensor sessions.

    Parameters
    ----------
    raw_dir : Path
        Directory containing all required CSV files.

    Returns
    -------
    pd.DataFrame
        Concatenated dataframe with columns:
        ['pc_time', 'esp_time', 'az', 'source_file'].
    """
    file_paths = _validate_expected_files(raw_dir)
    dataframes: List[pd.DataFrame] = []
    per_file_stats: List[Dict[str, float | str | int]] = []

    for file_path in file_paths:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        df.columns = [str(col).strip() for col in df.columns]

        if "az" not in df.columns:
            raise ValueError(
                f"'az' column missing in {file_path.name}. Found columns: {list(df.columns)}"
            )

        before_rows = len(df)
        df["az"] = pd.to_numeric(df["az"], errors="coerce")
        df = df.dropna(subset=["az"]).copy()
        dropped = before_rows - len(df)
        if dropped > 0:
            LOGGER.info("Dropped %d NaN/non-numeric az rows from %s", dropped, file_path.name)

        missing_required = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_required:
            raise ValueError(
                f"Missing required columns in {file_path.name}: {missing_required}. "
                f"Found columns: {list(df.columns)}"
            )

        df = df[REQUIRED_COLUMNS].copy()
        df["source_file"] = file_path.name
        dataframes.append(df)

        per_file_stats.append(
            {
                "source_file": file_path.name,
                "rows": int(len(df)),
                "az_mean": float(df["az"].mean()),
                "az_std": float(df["az"].std(ddof=0)),
                "az_min": float(df["az"].min()),
                "az_max": float(df["az"].max()),
            }
        )

    merged = pd.concat(dataframes, axis=0, ignore_index=True)
    summary_df = pd.DataFrame(per_file_stats)
    LOGGER.info("Raw data summary by file:\n%s", summary_df.to_string(index=False))
    LOGGER.info("Total rows loaded: %d", len(merged))
    return merged

