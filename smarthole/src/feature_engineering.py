"""Windowing and feature extraction for az acceleration signal."""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

LOGGER = logging.getLogger(__name__)

FEATURE_NAMES: List[str] = [
    "mean",
    "std_val",
    "rms",
    "p2p",
    "kurtosis",
    "skew",
    "energy",
    "iqr",
    "abs_mean",
    "median",
    "zcr",
    "crest_factor",
    "variance",
    "max_abs",
    "min_abs",
]

EPSILON = 1e-9


def extract_window_features(az: np.ndarray) -> np.ndarray:
    """Extract 15 statistical features from one az window.

    Parameters
    ----------
    az : np.ndarray
        One-dimensional array of az values for a single window.

    Returns
    -------
    np.ndarray
        Feature vector with shape (15,) and dtype float32.
    """
    signal = np.asarray(az, dtype=np.float64)
    n_samples = signal.size

    mean_val = np.mean(signal)
    std_val = np.std(signal, ddof=0)
    rms = np.sqrt(np.mean(np.square(signal)))
    p2p = np.max(signal) - np.min(signal)
    kur = kurtosis(signal, fisher=True, bias=False)
    skw = skew(signal, bias=False)
    energy = np.sum(np.square(signal))
    q75, q25 = np.percentile(signal, [75, 25])
    iqr = q75 - q25
    abs_mean = np.mean(np.abs(signal))
    median = np.median(signal)
    zcr = np.sum(np.abs(np.diff(np.sign(signal)))) / (2.0 * float(n_samples))
    max_abs = np.max(np.abs(signal))
    crest_factor = rms / (max_abs + EPSILON)
    variance = std_val**2
    min_abs = np.min(np.abs(signal))

    return np.array(
        [
            mean_val,
            std_val,
            rms,
            p2p,
            kur,
            skw,
            energy,
            iqr,
            abs_mean,
            median,
            zcr,
            crest_factor,
            variance,
            max_abs,
            min_abs,
        ],
        dtype=np.float32,
    )


def create_windows(df: pd.DataFrame, window_size: int, stride: int) -> Tuple[np.ndarray, List[str]]:
    """Create per-file sliding windows and extract features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing at least columns ['az', 'source_file'].
    window_size : int
        Number of samples per window.
    stride : int
        Step size between consecutive windows.

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        Feature matrix of shape (N_windows, 15) and source file name per window.
    """
    if window_size <= stride:
        raise ValueError(
            f"window_size ({window_size}) must be greater than stride ({stride})."
        )

    features: List[np.ndarray] = []
    source_labels: List[str] = []

    grouped = df.groupby("source_file", sort=False)
    for source_file, group in grouped:
        az_values = group["az"].to_numpy(dtype=np.float64)
        n_rows = len(az_values)
        window_count = 0

        if n_rows < window_size:
            LOGGER.warning(
                "Skipping %s: rows=%d is smaller than window_size=%d",
                source_file,
                n_rows,
                window_size,
            )
            continue

        for start in range(0, n_rows - window_size + 1, stride):
            end = start + window_size
            feature_vector = extract_window_features(az_values[start:end])
            features.append(feature_vector)
            source_labels.append(source_file)
            window_count += 1

        LOGGER.info("Extracted %d windows from %s", window_count, source_file)

    if not features:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), []

    feature_matrix = np.vstack(features).astype(np.float32)
    LOGGER.info(
        "Created feature matrix with shape (%d, %d)",
        feature_matrix.shape[0],
        feature_matrix.shape[1],
    )
    return feature_matrix, source_labels

