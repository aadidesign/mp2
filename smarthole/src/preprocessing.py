"""Train/validation/test splitting and scaling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)


def _distribution(y: np.ndarray) -> str:
    """Build compact class distribution string."""
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    parts = []
    for cls, cnt in zip(unique.tolist(), counts.tolist()):
        parts.append(f"class {cls}: {cnt} ({100.0 * cnt / total:.1f}%)")
    return ", ".join(parts)


def split_and_scale(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    val_size: float,
    random_state: int,
    scaler_save_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Split dataset with stratification and scale features leakage-free.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Label vector.
    test_size : float
        Fraction for test split.
    val_size : float
        Fraction for validation split from full dataset.
    random_state : int
        Random seed.
    scaler_save_path : Path
        Directory to save feature scaler.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]
        Scaled train/val/test arrays, corresponding labels, and scaler.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_ratio,
        stratify=y_temp,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    scaler_save_path.mkdir(parents=True, exist_ok=True)
    scaler_file = scaler_save_path / "feature_scaler.pkl"
    joblib.dump(scaler, scaler_file)
    LOGGER.info("Saved supervised feature scaler to %s", scaler_file)

    LOGGER.info("Train distribution: %s", _distribution(y_train))
    LOGGER.info("Val distribution: %s", _distribution(y_val))
    LOGGER.info("Test distribution: %s", _distribution(y_test))

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        scaler,
    )

