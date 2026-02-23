"""Inference utilities and road quality reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import yaml

from src.feature_engineering import create_windows
from src.model import load_model


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
        labels = {0: "Smooth", 1: "Mild Pothole", 2: "Severe Pothole"}
        for cls in [0, 1, 2]:
            pct = self.percentages[cls]
            print(f"{labels[cls]:14s} {pct:6.2f}% |{self._bar(pct)}|")
        print("---------------------------------")
        print(f"Road Quality Score: {self.road_quality_score:.2f}/100")
        print("=================================\n")


def _load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config from disk."""
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def predict_from_csv(input_csv: Path, model_name: str, config_path: Path) -> RoadReport:
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
    df.columns = [str(c).strip() for c in df.columns]
    df["az"] = pd.to_numeric(df["az"], errors="coerce")
    df = df.dropna(subset=["az"]).copy()
    df["source_file"] = "input"

    feature_matrix, _ = create_windows(
        df=df,
        window_size=int(data_cfg["window_size"]),
        stride=int(data_cfg["stride"]),
    )
    if feature_matrix.shape[0] == 0:
        raise ValueError("Input CSV does not produce any valid windows.")

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
    report.print_report()
    return report

