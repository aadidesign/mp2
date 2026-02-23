"""Inference utilities and road quality reporting."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
import yaml

from src.feature_engineering import create_windows
from src.model import load_model

LOGGER = logging.getLogger(__name__)
LABEL_ID_TO_NAME = {0: "Smooth", 1: "Mild Pothole", 2: "Severe Pothole"}


@dataclass
class RoadReport:
    """Container for inference summary metrics."""

    predictions: np.ndarray
    probabilities: np.ndarray
    time_delta_s: Optional[float] = None

    def __post_init__(self) -> None:
        """Compute aggregate counts, percentages, and quality score."""
        total = len(self.predictions)
        self.total_windows = total
        self.counts: Dict[int, int] = {i: int(np.sum(self.predictions == i)) for i in [0, 1, 2]}
        self.percentages: Dict[int, float] = {
            i: (100.0 * self.counts[i] / total if total > 0 else 0.0) for i in [0, 1, 2]
        }
        smooth_count = self.counts[0]
        mild_count = self.counts[1]
        self.road_quality_score = (
            100.0 * (smooth_count + 0.5 * mild_count) / total if total > 0 else 0.0
        )

    def _bar(self, pct: float, width: int = 30) -> str:
        """Return a text bar scaled by percentage."""
        filled = int(round((pct / 100.0) * width))
        return "#" * filled + "-" * (width - filled)

    def print_report(self) -> None:
        """Print final user-facing road condition summary."""
        print("\n===== ROAD CONDITION REPORT =====")
        print(f"Total windows analyzed: {self.total_windows}")
        if self.time_delta_s is not None:
            print(f"Approx duration analyzed: {self.time_delta_s:.2f} seconds")
        print("---------------------------------")
        for cls in [0, 1, 2]:
            pct = self.percentages[cls]
            print(f"{LABEL_ID_TO_NAME[cls]:14s} {pct:6.2f}% |{self._bar(pct)}|")
        print("---------------------------------")
        print(f"Road Quality Score: {self.road_quality_score:.2f}/100")
        print("=================================\n")


def _load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config from disk."""
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _validate_input_df(df: pd.DataFrame, input_csv: Path) -> pd.DataFrame:
    """Validate and clean input dataframe for inference.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded raw dataframe.
    input_csv : Path
        Source file path for error context.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with numeric az values.
    """
    df.columns = [str(c).strip() for c in df.columns]
    if "az" not in df.columns:
        raise ValueError(
            f"Input file '{input_csv.name}' is missing required column 'az'. "
            f"Found columns: {list(df.columns)}"
        )

    before_rows = len(df)
    df["az"] = pd.to_numeric(df["az"], errors="coerce")
    df = df.dropna(subset=["az"]).copy()
    dropped = before_rows - len(df)
    if dropped > 0:
        LOGGER.info("Dropped %d invalid az rows from %s", dropped, input_csv.name)

    if len(df) == 0:
        raise ValueError(f"Input file '{input_csv.name}' has no valid az rows after cleaning.")
    return df


def _write_window_tags_csv(
    input_csv: Path,
    df: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    window_size: int,
    stride: int,
    window_output_dir: Path,
) -> Path:
    """Write per-window predicted tags and probabilities to CSV.

    Parameters
    ----------
    input_csv : Path
        Original input CSV.
    df : pd.DataFrame
        Cleaned input dataframe.
    predictions : np.ndarray
        Predicted class IDs per window.
    probabilities : np.ndarray
        Predicted class probabilities per window.
    window_size : int
        Window length.
    stride : int
        Window stride.
    window_output_dir : Path
        Destination directory for prediction CSVs.

    Returns
    -------
    Path
        Saved CSV file path.
    """
    starts = np.arange(0, len(df) - window_size + 1, stride, dtype=int)
    if len(starts) != len(predictions):
        raise ValueError(
            f"Window index mismatch for '{input_csv.name}': starts={len(starts)}, "
            f"predictions={len(predictions)}"
        )

    window_df = pd.DataFrame(
        {
            "window_index": np.arange(len(predictions), dtype=int),
            "start_index": starts,
            "end_index": starts + window_size - 1,
            "predicted_label": predictions.astype(int),
            "predicted_tag": [LABEL_ID_TO_NAME[int(v)] for v in predictions],
            "prob_smooth": probabilities[:, 0],
            "prob_mild_pothole": probabilities[:, 1],
            "prob_severe_pothole": probabilities[:, 2],
        }
    )

    if "pc_time" in df.columns:
        pc_numeric = pd.to_numeric(df["pc_time"], errors="coerce")
        window_df["start_pc_time"] = pc_numeric.iloc[window_df["start_index"]].to_numpy()
        window_df["end_pc_time"] = pc_numeric.iloc[window_df["end_index"]].to_numpy()
    if "esp_time" in df.columns:
        esp_numeric = pd.to_numeric(df["esp_time"], errors="coerce")
        window_df["start_esp_time"] = esp_numeric.iloc[window_df["start_index"]].to_numpy()
        window_df["end_esp_time"] = esp_numeric.iloc[window_df["end_index"]].to_numpy()

    window_output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = window_output_dir / f"{input_csv.stem}_window_predictions.csv"
    window_df.to_csv(output_csv, index=False)
    LOGGER.info("Saved per-window tags to %s", output_csv)
    return output_csv


def predict_from_csv(
    input_csv: Path,
    model_name: str,
    config_path: Path,
    write_window_tags: bool = True,
    show_report: bool = True,
) -> RoadReport:
    """Run end-to-end inference on one raw CSV.

    Parameters
    ----------
    input_csv : Path
        Input CSV path.
    model_name : str
        Model name to load.
    config_path : Path
        Project config path.

    Returns
    -------
    RoadReport
        Computed report object with counts and quality score.
    """
    cfg = _load_config(config_path)
    model_dir = config_path.parent / cfg["paths"]["model_dir"]
    data_cfg = cfg["data"]

    df = pd.read_csv(input_csv, encoding="utf-8-sig")
    df = _validate_input_df(df, input_csv)
    df["source_file"] = "input"

    window_size = int(data_cfg["window_size"])
    stride = int(data_cfg["stride"])
    feature_matrix, _ = create_windows(
        df=df,
        window_size=window_size,
        stride=stride,
    )
    if feature_matrix.shape[0] == 0:
        raise ValueError(
            f"Input CSV '{input_csv.name}' does not produce any valid windows. "
            f"Check file length and window settings."
        )

    scaler = joblib.load(model_dir / "feature_scaler.pkl")
    x_scaled = scaler.transform(feature_matrix)
    model = load_model(model_name, model_dir)

    predictions = model.predict(x_scaled)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_scaled)
    else:
        probabilities = np.zeros((len(predictions), 3), dtype=np.float64)
        probabilities[np.arange(len(predictions)), predictions.astype(int)] = 1.0

    time_delta_s = None
    if "pc_time" in df.columns:
        pc_numeric = pd.to_numeric(df["pc_time"], errors="coerce").dropna()
        if len(pc_numeric) >= 2:
            time_delta_s = float(pc_numeric.iloc[-1] - pc_numeric.iloc[0])

    report = RoadReport(
        predictions=np.asarray(predictions, dtype=np.int64),
        probabilities=np.asarray(probabilities, dtype=np.float64),
        time_delta_s=time_delta_s,
    )
    if write_window_tags:
        predictions_dir = config_path.parent / cfg["paths"]["predictions_dir"]
        _write_window_tags_csv(
            input_csv=input_csv,
            df=df,
            predictions=report.predictions,
            probabilities=report.probabilities,
            window_size=window_size,
            stride=stride,
            window_output_dir=predictions_dir,
        )

    if show_report:
        report.print_report()
    return report


def predict_from_folder(
    config_path: Path,
    model_name: str,
    input_dir: Optional[Path] = None,
    selected_files: Optional[Iterable[str]] = None,
) -> Dict[str, RoadReport]:
    """Run inference for CSV files from a folder.

    Parameters
    ----------
    config_path : Path
        Project config path.
    model_name : str
        Model name to load.
    input_dir : Optional[Path], optional
        Directory containing input CSV files. If None, uses config custom_data_dir.
    selected_files : Optional[Iterable[str]], optional
        Specific CSV filenames to process from input_dir.

    Returns
    -------
    Dict[str, RoadReport]
        Road reports keyed by input filename.
    """
    cfg = _load_config(config_path)
    if input_dir is None:
        input_dir = config_path.parent / cfg["paths"]["custom_data_dir"]
    input_dir = input_dir.resolve()
    input_dir.mkdir(parents=True, exist_ok=True)

    if selected_files:
        requested = [name.strip() for name in selected_files if name.strip()]
        file_paths = [input_dir / name for name in requested]
        missing = [path.name for path in file_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Selected file(s) not found in '{input_dir}': {', '.join(missing)}"
            )
    else:
        file_paths = sorted(input_dir.glob("*.csv"))

    if not file_paths:
        raise FileNotFoundError(
            f"No CSV files found in '{input_dir}'. Place your files there and rerun."
        )

    reports: Dict[str, RoadReport] = {}
    print(f"\nRunning batch prediction with model: {model_name}")
    print(f"Input folder: {input_dir}")
    for file_path in file_paths:
        print(f"\n--- Predicting: {file_path.name} ---")
        report = predict_from_csv(
            input_csv=file_path,
            model_name=model_name,
            config_path=config_path,
            write_window_tags=True,
            show_report=True,
        )
        reports[file_path.name] = report

    print("\nBatch prediction complete.")
    print(f"Per-window outputs saved to: {config_path.parent / cfg['paths']['predictions_dir']}")
    return reports

