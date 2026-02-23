"""Main training pipeline for Smarthole."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from src.data_loader import load_raw_data
from src.evaluate import (
    evaluate_model,
    full_classification_report,
    plot_feature_importance,
    plot_label_distribution,
    plot_model_comparison,
    plot_roc_curves,
    plot_signal_per_class,
    print_leaderboard,
)
from src.feature_engineering import FEATURE_NAMES, create_windows
from src.labeling import assign_labels, label_by_gmm, label_by_physics_percentile
from src.model import build_all_models, cross_validate_model, save_model, train_model
from src.preprocessing import split_and_scale

LOGGER = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config.

    Parameters
    ----------
    config_path : Path
        Config file path.

    Returns
    -------
    Dict[str, Any]
        Loaded configuration dictionary.
    """
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _log_label_diagnostics(
    labels: np.ndarray,
    source_labels: list[str],
    feature_matrix: np.ndarray,
    feature_names: list[str],
    cfg: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Log and save label diagnostics and method comparisons."""
    diag_df = pd.DataFrame(
        {
            "source_file": source_labels,
            "label": labels,
            "std_val": feature_matrix[:, feature_names.index(cfg["labeling"]["severity_feature"])],
        }
    )
    crosstab = pd.crosstab(diag_df["source_file"], diag_df["label"]).reindex(
        columns=[0, 1, 2], fill_value=0
    )
    std_means = diag_df.groupby("label")["std_val"].mean().reindex([0, 1, 2])

    LOGGER.info("Label cross-tab by source file:\n%s", crosstab.to_string())
    LOGGER.info("Mean std_val per discovered label:\n%s", std_means.to_string())

    comparison = pd.DataFrame(
        {
            "unsupervised_kmeans": labels,
            "physics_percentile": label_by_physics_percentile(
                feature_matrix, feature_names, cfg["labeling"]["severity_feature"]
            ),
            "gmm": label_by_gmm(
                feature_matrix,
                feature_names,
                cfg["labeling"]["severity_feature"],
                int(cfg["data"]["random_state"]),
            ),
        }
    )
    comparison_file = output_dir / "labeling_method_comparison.csv"
    comparison.to_csv(comparison_file, index=False)
    LOGGER.info("Saved labeling method comparison to %s", comparison_file)


def run_training_pipeline(config_path: str | Path) -> pd.DataFrame:
    """Run full Smarthole training/evaluation pipeline.

    Parameters
    ----------
    config_path : str | Path
        Path to config YAML.

    Returns
    -------
    pd.DataFrame
        Final sorted leaderboard.
    """
    config_path = Path(config_path)
    project_root = config_path.parent.resolve()
    cfg = load_config(config_path)

    raw_dir = project_root / cfg["data"]["raw_dir"]
    processed_dir = project_root / cfg["data"]["processed_dir"]
    model_dir = project_root / cfg["paths"]["model_dir"]
    output_dir = project_root / cfg["paths"]["output_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("[1/7] Loading sensor data...")
    df = load_raw_data(raw_dir)

    LOGGER.info(
        "[2/7] Extracting features (window=%d, stride=%d)...",
        int(cfg["data"]["window_size"]),
        int(cfg["data"]["stride"]),
    )
    feature_matrix, source_labels = create_windows(
        df=df,
        window_size=int(cfg["data"]["window_size"]),
        stride=int(cfg["data"]["stride"]),
    )

    LOGGER.info(
        "[3/7] Assigning severity labels using algorithm: '%s'",
        cfg["labeling"]["method"],
    )
    LOGGER.info("      (No filename-based assumptions - purely data-driven)")
    labels, label_meta = assign_labels(
        feature_matrix=feature_matrix,
        feature_names=FEATURE_NAMES,
        method=cfg["labeling"]["method"],
        severity_feature=cfg["labeling"]["severity_feature"],
        random_state=int(cfg["data"]["random_state"]),
        n_init=int(cfg["labeling"]["n_init"]),
        save_path=model_dir / "kmeans_labeling_artifact.pkl",
    )

    labeled_df = pd.DataFrame(feature_matrix, columns=FEATURE_NAMES)
    labeled_df["label"] = labels
    labeled_df["source_file"] = source_labels
    labeled_file = processed_dir / "labeled_windows.csv"
    labeled_df.to_csv(labeled_file, index=False)
    LOGGER.info("Saved labeled feature windows to %s", labeled_file)

    _log_label_diagnostics(
        labels=labels,
        source_labels=source_labels,
        feature_matrix=feature_matrix,
        feature_names=FEATURE_NAMES,
        cfg=cfg,
        output_dir=output_dir,
    )

    LOGGER.info(
        "[4/7] Splitting data (test=%.2f, val=%.2f)...",
        float(cfg["data"]["test_size"]),
        float(cfg["data"]["val_size"]),
    )
    X_train, X_val, X_test, y_train, y_val, y_test, _ = split_and_scale(
        X=feature_matrix,
        y=labels,
        test_size=float(cfg["data"]["test_size"]),
        val_size=float(cfg["data"]["val_size"]),
        random_state=int(cfg["data"]["random_state"]),
        scaler_save_path=model_dir,
    )

    LOGGER.info("[5/7] Training all models...")
    model_candidates = build_all_models(y_train=y_train, cfg=cfg)
    trained_models: Dict[str, Any] = {}
    val_scores: Dict[str, float] = {}
    for model_name, estimator in model_candidates.items():
        try:
            trained_model, val_acc, _ = train_model(
                name=model_name,
                model=estimator,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
            )
            trained_models[model_name] = trained_model
            val_scores[model_name] = val_acc
            save_model(trained_model, model_name, model_dir)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Model %s failed during training: %s", model_name, exc)

    if not trained_models:
        raise RuntimeError("No models trained successfully.")

    LOGGER.info("[6/7] Running %d-fold cross-validation on best 3 models...", int(cfg["models"]["cv_splits"]))
    top3_names = [
        name for name, _score in sorted(val_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    ]
    cv_rows = []
    for name in top3_names:
        mean_f1, std_f1 = cross_validate_model(
            name=name,
            model=trained_models[name],
            X=X_train,
            y=y_train,
            n_splits=int(cfg["models"]["cv_splits"]),
            random_state=int(cfg["data"]["random_state"]),
        )
        cv_rows.append({"model": name, "cv_f1_weighted_mean": mean_f1, "cv_f1_weighted_std": std_f1})
    cv_df = pd.DataFrame(cv_rows).sort_values("cv_f1_weighted_mean", ascending=False)
    cv_file = output_dir / "cross_validation_top3.csv"
    cv_df.to_csv(cv_file, index=False)
    LOGGER.info("Saved cross-validation summary to %s", cv_file)

    LOGGER.info("[7/7] Evaluating on held-out test set...")
    results = []
    for name, model in trained_models.items():
        result = evaluate_model(name=name, model=model, X_test=X_test, y_test=y_test, output_dir=output_dir)
        results.append(result)
        full_classification_report(model=model, X_test=X_test, y_test=y_test, name=name)
        plot_feature_importance(model=model, feature_names=FEATURE_NAMES, output_dir=output_dir, model_name=name)

    plot_roc_curves(trained_models, X_test, y_test, output_dir)
    plot_model_comparison(results, output_dir)
    plot_label_distribution(labels=labels, source_labels=source_labels, output_dir=output_dir)
    plot_signal_per_class(
        df=df,
        labels=labels,
        source_labels=source_labels,
        output_dir=output_dir,
        window_size=int(cfg["data"]["window_size"]),
    )

    leaderboard = print_leaderboard(results, output_dir)
    metadata_file = output_dir / "labeling_metadata.txt"
    metadata_file.write_text(str(label_meta), encoding="utf-8")
    LOGGER.info("Saved labeling metadata to %s", metadata_file)
    return leaderboard

