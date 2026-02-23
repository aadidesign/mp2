"""Unsupervised, data-driven severity labeling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)

LABEL_DISPLAY = {
    0: "Smooth",
    1: "Mild Pothole",
    2: "Severe Pothole",
}


def _rank_clusters_by_severity(
    cluster_ids: np.ndarray, severity_values: np.ndarray
) -> Dict[int, int]:
    """Rank clusters by mean severity feature and map to labels 0/1/2.

    Parameters
    ----------
    cluster_ids : np.ndarray
        Unsupervised cluster IDs for each sample.
    severity_values : np.ndarray
        Values of the severity feature (e.g., std_val) per sample.

    Returns
    -------
    Dict[int, int]
        Mapping from cluster ID to severity class label.
    """
    cluster_means = {}
    for cluster_id in sorted(np.unique(cluster_ids)):
        mask = cluster_ids == cluster_id
        cluster_means[int(cluster_id)] = float(np.mean(severity_values[mask]))

    sorted_clusters = sorted(cluster_means, key=cluster_means.get)
    cluster_to_label = {
        int(sorted_clusters[0]): 0,
        int(sorted_clusters[1]): 1,
        int(sorted_clusters[2]): 2,
    }
    return cluster_to_label


def label_by_kmeans(
    feature_matrix: np.ndarray,
    feature_names: list[str],
    severity_feature: str,
    random_state: int,
    n_init: int,
    save_path: Path,
) -> Tuple[np.ndarray, Dict[int, int], StandardScaler, KMeans]:
    """Assign labels via KMeans and cluster severity ranking.

    Parameters
    ----------
    feature_matrix : np.ndarray
        Window-level feature matrix.
    feature_names : list[str]
        Ordered feature names corresponding to matrix columns.
    severity_feature : str
        Feature used for ranking roughness severity.
    random_state : int
        Random seed for reproducibility.
    n_init : int
        Number of KMeans initializations.
    save_path : Path
        Destination path for persisted clustering artifacts.

    Returns
    -------
    Tuple[np.ndarray, Dict[int, int], StandardScaler, KMeans]
        Final labels, cluster-to-label mapping, fitted scaler, fitted KMeans.
    """
    severity_idx = feature_names.index(severity_feature)
    severity_values = feature_matrix[:, severity_idx]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(feature_matrix)

    kmeans = KMeans(
        n_clusters=3,
        n_init=n_init,
        max_iter=500,
        random_state=random_state,
    )
    cluster_ids = kmeans.fit_predict(x_scaled)
    cluster_to_label = _rank_clusters_by_severity(cluster_ids, severity_values)
    labels = np.array([cluster_to_label[int(cid)] for cid in cluster_ids], dtype=np.int64)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "scaler": scaler,
        "kmeans": kmeans,
        "cluster_to_label": cluster_to_label,
        "severity_feature": severity_feature,
        "feature_names": feature_names,
    }
    joblib.dump(payload, save_path)
    LOGGER.info("Saved KMeans labeling artifact to %s", save_path)
    return labels, cluster_to_label, scaler, kmeans


def label_by_physics_percentile(
    feature_matrix: np.ndarray, feature_names: list[str], severity_feature: str
) -> np.ndarray:
    """Assign labels by tertiles of a physical severity feature.

    Parameters
    ----------
    feature_matrix : np.ndarray
        Window-level feature matrix.
    feature_names : list[str]
        Ordered feature names corresponding to matrix columns.
    severity_feature : str
        Feature used for thresholding.

    Returns
    -------
    np.ndarray
        Severity labels in {0, 1, 2}.
    """
    severity_values = feature_matrix[:, feature_names.index(severity_feature)]
    p33, p66 = np.percentile(severity_values, [33, 66])
    labels = np.where(
        severity_values <= p33,
        0,
        np.where(severity_values <= p66, 1, 2),
    )
    return labels.astype(np.int64)


def label_by_gmm(
    feature_matrix: np.ndarray,
    feature_names: list[str],
    severity_feature: str,
    random_state: int,
) -> np.ndarray:
    """Assign labels via Gaussian Mixture clustering and severity ranking.

    Parameters
    ----------
    feature_matrix : np.ndarray
        Window-level feature matrix.
    feature_names : list[str]
        Ordered feature names corresponding to matrix columns.
    severity_feature : str
        Feature used for ranking roughness severity.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Severity labels in {0, 1, 2}.
    """
    severity_idx = feature_names.index(severity_feature)
    severity_values = feature_matrix[:, severity_idx]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(feature_matrix)

    gmm = GaussianMixture(
        n_components=3,
        covariance_type="full",
        n_init=20,
        max_iter=500,
        random_state=random_state,
    )
    cluster_ids = gmm.fit_predict(x_scaled)
    cluster_to_label = _rank_clusters_by_severity(cluster_ids, severity_values)
    labels = np.array([cluster_to_label[int(cid)] for cid in cluster_ids], dtype=np.int64)
    return labels


def assign_labels(
    feature_matrix: np.ndarray,
    feature_names: list[str],
    method: str,
    severity_feature: str,
    random_state: int,
    n_init: int,
    save_path: Path,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Dispatch to configured labeling method and log summary.

    Parameters
    ----------
    feature_matrix : np.ndarray
        Window-level feature matrix.
    feature_names : list[str]
        Ordered feature names corresponding to matrix columns.
    method : str
        One of: unsupervised_kmeans, physics_percentile, gmm.
    severity_feature : str
        Feature used as roughness proxy.
    random_state : int
        Random seed for reproducibility.
    n_init : int
        Number of initializations for KMeans.
    save_path : Path
        Path to save KMeans artifacts when used.

    Returns
    -------
    Tuple[np.ndarray, Dict[str, Any]]
        Final labels and method metadata.
    """
    metadata: Dict[str, Any] = {"method": method, "severity_feature": severity_feature}

    if method == "unsupervised_kmeans":
        labels, cluster_to_label, scaler, model = label_by_kmeans(
            feature_matrix=feature_matrix,
            feature_names=feature_names,
            severity_feature=severity_feature,
            random_state=random_state,
            n_init=n_init,
            save_path=save_path,
        )
        metadata.update(
            {"cluster_to_label": cluster_to_label, "labeling_scaler": scaler, "model": model}
        )
    elif method == "physics_percentile":
        labels = label_by_physics_percentile(feature_matrix, feature_names, severity_feature)
    elif method == "gmm":
        labels = label_by_gmm(feature_matrix, feature_names, severity_feature, random_state)
    else:
        raise ValueError(
            f"Unknown labeling method '{method}'. "
            "Supported methods: unsupervised_kmeans | physics_percentile | gmm"
        )

    unique_labels = set(np.unique(labels).tolist())
    if unique_labels != {0, 1, 2}:
        raise ValueError(f"Label assignment failed. Expected labels {{0,1,2}}, got {unique_labels}")

    total = len(labels)
    counts = {i: int(np.sum(labels == i)) for i in range(3)}
    LOGGER.info("================================================")
    LOGGER.info("LABEL ASSIGNMENT: %s", method)
    LOGGER.info("================================================")
    LOGGER.info(
        "Label 0 - Smooth        : %4d windows (%5.1f%%)",
        counts[0],
        100.0 * counts[0] / total,
    )
    LOGGER.info(
        "Label 1 - Mild Pothole  : %4d windows (%5.1f%%)",
        counts[1],
        100.0 * counts[1] / total,
    )
    LOGGER.info(
        "Label 2 - Severe Pothole: %4d windows (%5.1f%%)",
        counts[2],
        100.0 * counts[2] / total,
    )
    LOGGER.info("================================================")
    return labels, metadata

