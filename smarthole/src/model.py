"""Model building, training, CV, and persistence."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight

LOGGER = logging.getLogger(__name__)


def get_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Compute balanced class weights.

    Parameters
    ----------
    y : np.ndarray
        Labels.

    Returns
    -------
    Dict[int, float]
        Class weights keyed by class index.
    """
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes.tolist(), weights.tolist())}


def build_all_models(y_train: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build all candidate models with config-driven hyperparameters.

    Parameters
    ----------
    y_train : np.ndarray
        Training labels.
    cfg : Dict[str, Any]
        Loaded config dictionary.

    Returns
    -------
    Dict[str, Any]
        Model dictionary keyed by human-readable names.
    """
    _ = get_class_weights(y_train)
    rs = int(cfg["data"]["random_state"])
    model_cfg = cfg["models"]

    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=int(model_cfg["rf_n_estimators"]),
            max_depth=model_cfg["rf_max_depth"],
            class_weight="balanced",
            random_state=rs,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=int(model_cfg["rf_n_estimators"]),
            class_weight="balanced",
            random_state=rs,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=int(model_cfg["gb_n_estimators"]),
            learning_rate=float(model_cfg["gb_learning_rate"]),
            max_depth=int(model_cfg["gb_max_depth"]),
            random_state=rs,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf",
            C=float(model_cfg["svm_c"]),
            gamma=model_cfg["svm_gamma"],
            class_weight="balanced",
            probability=True,
            random_state=rs,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=int(model_cfg["knn_k"]),
            metric="euclidean",
            n_jobs=-1,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=int(model_cfg["dt_max_depth"]),
            class_weight="balanced",
            random_state=rs,
        ),
        "Logistic Regression": LogisticRegression(
            C=float(model_cfg["lr_c"]),
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=rs,
        ),
        "Naive Bayes": GaussianNB(),
    }


def train_model(
    name: str,
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[Any, float, float]:
    """Train one model and compute validation accuracy.

    Parameters
    ----------
    name : str
        Model name.
    model : Any
        Estimator instance.
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    X_val : np.ndarray
        Validation features.
    y_val : np.ndarray
        Validation labels.

    Returns
    -------
    Tuple[Any, float, float]
        Trained model, validation accuracy, and elapsed seconds.
    """
    start = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    y_pred = model.predict(X_val)
    val_acc = float(accuracy_score(y_val, y_pred))
    LOGGER.info("%s trained in %.3fs | val_acc=%.4f", name, elapsed, val_acc)
    return model, val_acc, elapsed


def cross_validate_model(
    name: str,
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    random_state: int,
) -> Tuple[float, float]:
    """Run stratified weighted-F1 cross-validation.

    Parameters
    ----------
    name : str
        Model name.
    model : Any
        Estimator instance.
    X : np.ndarray
        Features.
    y : np.ndarray
        Labels.
    n_splits : int
        Number of CV folds.
    random_state : int
        Random seed.

    Returns
    -------
    Tuple[float, float]
        Mean and standard deviation of weighted F1.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, scoring="f1_weighted", cv=skf, n_jobs=-1)
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    LOGGER.info("%s CV weighted-F1: %.4f +/- %.4f", name, mean_score, std_score)
    return mean_score, std_score


def _model_filename(name: str) -> str:
    """Create a stable filename from model name."""
    return name.replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"


def save_model(model: Any, name: str, save_dir: Path) -> Path:
    """Persist model with joblib.

    Parameters
    ----------
    model : Any
        Trained model.
    name : str
        Model name.
    save_dir : Path
        Destination directory.

    Returns
    -------
    Path
        Path to saved model file.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / _model_filename(name)
    joblib.dump(model, file_path)
    LOGGER.info("Saved model %s -> %s", name, file_path)
    return file_path


def load_model(name: str, model_dir: Path) -> Any:
    """Load model by name from model directory.

    Parameters
    ----------
    name : str
        Model name.
    model_dir : Path
        Directory containing model files.

    Returns
    -------
    Any
        Loaded model object.
    """
    return joblib.load(model_dir / _model_filename(name))

