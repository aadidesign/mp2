"""Evaluation and visualization utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

LOGGER = logging.getLogger(__name__)
CLASS_NAMES = ["Smooth", "Mild Pothole", "Severe Pothole"]
CLASS_COLORS = {0: "green", 1: "orange", 2: "red"}


def evaluate_model(
    name: str,
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
) -> Dict[str, float | str]:
    """Compute core test metrics and save confusion matrix plot.

    Parameters
    ----------
    name : str
        Model name.
    model : Any
        Trained estimator.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels.
    output_dir : Path
        Plot output directory.

    Returns
    -------
    Dict[str, float | str]
        Core metrics for leaderboard.
    """
    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted"))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))

    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    else:
        decision = model.decision_function(X_test)
        if decision.ndim == 1:
            decision = np.column_stack([-decision, decision, np.zeros_like(decision)])
        y_score = decision
    roc_auc = float(
        roc_auc_score(y_test_bin, y_score, multi_class="ovr", average="macro")
    )

    _plot_confusion_matrix(name, y_test, y_pred, output_dir)
    return {
        "model": name,
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "roc_auc": roc_auc,
    }


def _plot_confusion_matrix(name: str, y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> None:
    """Plot raw and row-normalized confusion matrices.

    Parameters
    ----------
    name : str
        Model name.
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Predicted labels.
    output_dir : Path
        Plot output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_norm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2], normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[0],
    )
    axes[0].set_title(f"{name} - Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[1],
    )
    axes[1].set_title(f"{name} - Confusion Matrix (Row-Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    fig.tight_layout()
    file_path = output_dir / f"confusion_matrix_{name.replace(' ', '_')}.png"
    fig.savefig(file_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(
    models_dict: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray, output_dir: Path
) -> None:
    """Plot one-vs-rest ROC curves for each class across models.

    Parameters
    ----------
    models_dict : Dict[str, Any]
        Trained model dictionary.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels.
    output_dir : Path
        Plot output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharex=True, sharey=True)

    for class_idx, axis in enumerate(axes):
        for model_name, model in models_dict.items():
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)
            else:
                decision = model.decision_function(X_test)
                if decision.ndim == 1:
                    decision = np.column_stack([-decision, decision, np.zeros_like(decision)])
                y_score = decision

            fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], y_score[:, class_idx])
            auc = roc_auc_score(y_test_bin[:, class_idx], y_score[:, class_idx])
            axis.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")

        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.6)
        axis.set_title(f"Class {class_idx}: {CLASS_NAMES[class_idx]}")
        axis.set_xlabel("False Positive Rate")
        if class_idx == 0:
            axis.set_ylabel("True Positive Rate")
        axis.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(results_list: List[Dict[str, float | str]], output_dir: Path) -> None:
    """Plot model comparison charts for accuracy and weighted F1.

    Parameters
    ----------
    results_list : List[Dict[str, float | str]]
        Evaluation results.
    output_dir : Path
        Plot output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results_list).sort_values("f1_weighted", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    sns.barplot(data=results_df, x="accuracy", y="model", ax=axes[0], color="#4c72b0")
    axes[0].axvline(0.80, linestyle="--", color="gray")
    axes[0].set_title("Accuracy by Model")
    for idx, val in enumerate(results_df["accuracy"].tolist()):
        axes[0].text(val + 0.002, idx, f"{val:.3f}", va="center")

    sns.barplot(data=results_df, x="f1_weighted", y="model", ax=axes[1], color="#dd8452")
    axes[1].axvline(0.80, linestyle="--", color="gray")
    axes[1].set_title("Weighted F1 by Model")
    for idx, val in enumerate(results_df["f1_weighted"].tolist()):
        axes[1].text(val + 0.002, idx, f"{val:.3f}", va="center")

    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(
    model: Any, feature_names: List[str], output_dir: Path, model_name: str
) -> None:
    """Plot feature importances for tree-based models.

    Parameters
    ----------
    model : Any
        Trained estimator.
    feature_names : List[str]
        Feature names.
    output_dir : Path
        Plot output directory.
    model_name : str
        Model display name.
    """
    if not hasattr(model, "feature_importances_"):
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    ordered_names = [feature_names[i] for i in order]
    ordered_vals = importances[order]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=ordered_vals, y=ordered_names, ax=ax, color="#55a868")
    ax.set_title(f"Feature Importance: {model_name}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    file_name = f"feature_importance_{model_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    fig.savefig(output_dir / file_name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_label_distribution(labels: np.ndarray, source_labels: List[str], output_dir: Path) -> None:
    """Plot overall and per-source label distributions.

    Parameters
    ----------
    labels : np.ndarray
        Severity labels.
    source_labels : List[str]
        Source filename per window.
    output_dir : Path
        Plot output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dist_df = pd.DataFrame({"source_file": source_labels, "label": labels})
    overall = dist_df["label"].value_counts().reindex([0, 1, 2], fill_value=0)
    crosstab = pd.crosstab(dist_df["source_file"], dist_df["label"], normalize="index") * 100.0
    crosstab = crosstab.reindex(columns=[0, 1, 2], fill_value=0.0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    bars = axes[0].bar(
        [CLASS_NAMES[i] for i in [0, 1, 2]],
        overall.values,
        color=[CLASS_COLORS[i] for i in [0, 1, 2]],
    )
    axes[0].set_title("Overall Label Distribution")
    axes[0].set_ylabel("Window Count")
    for bar, val in zip(bars, overall.values.tolist()):
        axes[0].text(bar.get_x() + bar.get_width() / 2, val, str(int(val)), ha="center", va="bottom")

    left = np.zeros(len(crosstab))
    for label in [0, 1, 2]:
        vals = crosstab[label].values
        axes[1].barh(
            crosstab.index,
            vals,
            left=left,
            label=CLASS_NAMES[label],
            color=CLASS_COLORS[label],
        )
        left += vals
    axes[1].set_title("Per-Source Severity Share (%)")
    axes[1].set_xlabel("Percent")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "label_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_signal_per_class(
    df: pd.DataFrame,
    labels: np.ndarray,
    source_labels: List[str],
    output_dir: Path,
    window_size: int,
) -> None:
    """Plot representative az windows for each discovered class.

    Parameters
    ----------
    df : pd.DataFrame
        Full raw dataframe with az and source_file.
    labels : np.ndarray
        Severity label per window.
    source_labels : List[str]
        Source file per window.
    output_dir : Path
        Plot output directory.
    window_size : int
        Window length.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stride = window_size // 2

    windows_by_class: Dict[int, np.ndarray] = {}
    window_index = 0
    for source_file, group in df.groupby("source_file", sort=False):
        signal = group["az"].to_numpy(dtype=float)
        for start in range(0, len(signal) - window_size + 1, stride):
            if window_index >= len(labels):
                break
            if source_labels[window_index] != source_file:
                window_index += 1
                continue
            label = int(labels[window_index])
            if label not in windows_by_class:
                windows_by_class[label] = signal[start : start + window_size]
                if len(windows_by_class) == 3:
                    break
            window_index += 1
        if len(windows_by_class) == 3:
            break

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    x_axis = np.arange(window_size)
    for cls in [0, 1, 2]:
        axis = axes[cls]
        if cls in windows_by_class:
            axis.plot(x_axis, windows_by_class[cls], color=CLASS_COLORS[cls], linewidth=1.6)
        axis.axhline(1.0, linestyle="--", color="black", linewidth=1.0, alpha=0.7)
        axis.set_title(f"Representative Signal - {CLASS_NAMES[cls]}")
        axis.set_ylabel("az (g)")
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Sample Index")

    fig.tight_layout()
    fig.savefig(output_dir / "signal_per_class.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def print_leaderboard(results_list: List[Dict[str, float | str]], output_dir: Path) -> pd.DataFrame:
    """Print and save sorted leaderboard.

    Parameters
    ----------
    results_list : List[Dict[str, float | str]]
        Evaluation results.
    output_dir : Path
        Output directory.

    Returns
    -------
    pd.DataFrame
        Sorted leaderboard dataframe.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = pd.DataFrame(results_list).sort_values("f1_weighted", ascending=False).reset_index(drop=True)
    leaderboard.index = leaderboard.index + 1
    leaderboard.index.name = "rank"
    print("\n===== FINAL MODEL LEADERBOARD =====")
    print(leaderboard.to_string(float_format=lambda x: f"{x:.4f}"))
    leaderboard.to_csv(output_dir / "results_leaderboard.csv", index=True)
    LOGGER.info("Saved leaderboard to %s", output_dir / "results_leaderboard.csv")
    return leaderboard


def full_classification_report(model: Any, X_test: np.ndarray, y_test: np.ndarray, name: str) -> None:
    """Print sklearn classification report with class names.

    Parameters
    ----------
    model : Any
        Trained model.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels.
    name : str
        Model name.
    """
    y_pred = model.predict(X_test)
    print(f"\n===== Classification Report: {name} =====")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=CLASS_NAMES,
            digits=4,
        )
    )

