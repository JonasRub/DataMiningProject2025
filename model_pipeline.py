"""How to run this pipeline in a nutshell.

Use:
    python model_pipeline.py --data-path train_dataset.csv --use-optuna --n-trials 40 --do-ensemble

All arguments are optional; check ``--help`` for the full list.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from helper_KMv2 import dataCleaning, get_preprocessor

# Optional heavy imports -----------------------------------------------------
OPTUNA_AVAILABLE = False
try:  # pragma: no cover - optional dependency
    import optuna

    OPTUNA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    optuna = None

try:  # pragma: no cover - optional dependency
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover - optional dependency
    XGBClassifier = None
    warnings.warn(f"Failed to import XGBoost: {exc}")

try:  # pragma: no cover - optional dependency
    from lightgbm import LGBMClassifier
except Exception as exc:  # pragma: no cover - optional dependency
    LGBMClassifier = None
    warnings.warn(f"Failed to import LightGBM: {exc}")

try:  # pragma: no cover - optional dependency
    from catboost import CatBoostClassifier
except Exception as exc:  # pragma: no cover - optional dependency
    CatBoostClassifier = None
    warnings.warn(f"Failed to import CatBoost: {exc}")

try:  # pragma: no cover - optional dependency
    import shap

    shap.logger.setLevel(logging.WARNING)
except Exception as exc:  # pragma: no cover - optional dependency
    shap = None
    warnings.warn(f"Failed to import SHAP: {exc}")

# =============================================================================
# Global configuration
# =============================================================================
RANDOM_STATE: int = 42
np.random.seed(RANDOM_STATE)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*is deprecated and will be removed.*",
)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
LOGGER = logging.getLogger("pipeline")

METRICS = {
    "roc_auc": roc_auc_score,
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "pr_auc": average_precision_score,
    "brier": brier_score_loss,
}

PLOT_STYLE = {
    "figure.figsize": (8, 6),
    "axes.grid": True,
    "axes.titlesize": "large",
    "axes.labelsize": "medium",
}
plt.rcParams.update(PLOT_STYLE)

# =============================================================================
# Utility data classes and helpers
# =============================================================================


def ensure_dir(path: Path) -> Path:
    """Ensure that a directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class CVResult:
    """Keep track of everything we learn from cross-validation."""

    model_name: str
    oof_predictions: np.ndarray
    oof_indices: np.ndarray
    fold_metrics: pd.DataFrame
    final_estimator: BaseEstimator
    feature_importances: Optional[pd.Series] = None
    roc_curves: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None
    pr_curves: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None
    shap_values: Optional[np.ndarray] = None
    shap_expected_value: Optional[float] = None


@dataclass
class PreprocessingArtifacts:
    """Bundle the fitted preprocessing pieces so they can be reused later."""

    cleaner: BaseEstimator
    preprocessor: BaseEstimator
    feature_names: List[str]


# =============================================================================
# Data loading
# =============================================================================

def load_data(
    data_path: Path, target: str
) -> Tuple[pd.DataFrame, pd.Series, PreprocessingArtifacts]:
    """Load the raw data, run the helper preprocessing, and split features from the target."""

    if data_path.suffix == "":
        data_path = data_path.with_suffix(".csv")

    if not data_path.exists():
        candidates = []
        if data_path.suffix:
            stem = data_path.with_suffix("")
            candidates.append(stem.with_suffix(".csv"))
            candidates.append(stem.with_suffix(".parquet"))
        else:
            candidates.append(data_path.with_suffix(".csv"))
            candidates.append(data_path.with_suffix(".parquet"))

        for candidate in candidates:
            if candidate.exists():
                LOGGER.warning("Falling back to alternate file at %s", candidate)
                data_path = candidate
                break
        else:
            raise FileNotFoundError(
                f"Could not find {data_path}. Expecting train_dataset.csv as used in the EDA notebook."
            )

    LOGGER.info("Loading data from %s", data_path)
    if data_path.suffix == ".parquet":
        df_raw = pd.read_parquet(data_path)
    else:
        df_raw = pd.read_csv(data_path)

    cleaner = dataCleaning()
    df_clean = cleaner.fit_transform(df_raw)
    if target not in df_clean.columns:
        raise KeyError(f"Target column '{target}' was not found in the dataset.")

    y = df_clean[target].astype(int)
    X_raw = df_clean.drop(columns=[target])

    preprocessor = get_preprocessor()
    LOGGER.info("Fitting preprocessing pipeline from helper_KMv2.")
    X_processed = preprocessor.fit_transform(X_raw)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    feature_names = list(preprocessor.get_feature_names_out())
    X = pd.DataFrame(X_processed, columns=feature_names, index=df_clean.index)
    LOGGER.info("Shape X=%s, y=%s", X.shape, y.shape)

    preprocessing = PreprocessingArtifacts(
        cleaner=cleaner,
        preprocessor=preprocessor,
        feature_names=feature_names,
    )
    return X, y, preprocessing


# =============================================================================
# Cross-Validation Generator
# =============================================================================

def get_cv(n_splits: int = 5, seed: int = RANDOM_STATE) -> StratifiedKFold:
    """Create a deterministic ``StratifiedKFold`` instance."""

    LOGGER.info("Setting up StratifiedKFold with %s splits", n_splits)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


# =============================================================================
# Training helpers
# =============================================================================

def _compute_fold_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Calculate the common metrics we report for each fold."""

    metrics: Dict[str, float] = {}
    metrics["roc_auc"] = METRICS["roc_auc"](y_true, y_proba)
    metrics["pr_auc"] = METRICS["pr_auc"](y_true, y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    metrics["accuracy"] = METRICS["accuracy"](y_true, y_pred)
    metrics["precision"] = METRICS["precision"](y_true, y_pred, zero_division=0)
    metrics["recall"] = METRICS["recall"](y_true, y_pred)
    metrics["f1"] = METRICS["f1"](y_true, y_pred)
    metrics["brier"] = METRICS["brier"](y_true, y_proba)
    return metrics


def _aggregate_metrics(fold_metrics: List[Dict[str, float]]) -> pd.DataFrame:
    """Combine fold-level metrics into a table with mean and standard deviation."""

    df = pd.DataFrame(fold_metrics)
    summary = df.agg(["mean", "std"]).rename(index={"mean": "mean", "std": "std"})
    df = pd.concat([df, summary])
    df.index = [f"fold_{i+1}" for i in range(len(fold_metrics))] + ["mean", "std"]
    return df


def _save_metrics(metrics_df: pd.DataFrame, path: Path) -> None:
    metrics_df.to_csv(path, index=True)
    LOGGER.info("Stored metrics at %s", path)


def _save_oof_predictions(oof: np.ndarray, indices: np.ndarray, path: Path) -> None:
    df = pd.DataFrame({"index": indices, "oof_proba": oof})
    df.to_csv(path, index=False)
    LOGGER.info("Saved out-of-fold predictions to %s", path)


def _plot_cv_boxplot(results: Dict[str, pd.DataFrame], output_path: Path) -> None:
    """Draw a box plot of ROC-AUC scores across models."""

    ensure_dir(output_path.parent)
    data = []
    labels = []
    for model, df in results.items():
        folds = df.iloc[:-2]["roc_auc"].tolist()
        data.append(folds)
        labels.append(model)
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels)
    plt.ylabel("ROC-AUC")
    plt.title("Cross-validated ROC-AUC overview")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    LOGGER.info("Saved cross-validation box plot to %s", output_path)


def _plot_cv_curves(
    roc_curves: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    pr_curves: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    model_name: str,
    output_dir: Path,
) -> None:
    ensure_dir(output_dir)
    if roc_curves:
        mean_fpr = np.linspace(0, 1, 200)
        tprs = []
        aucs = []
        for fpr, tpr, _ in roc_curves:
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(np.trapz(tpr, fpr))
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        plt.figure()
        plt.plot(mean_fpr, mean_tpr, color="b", label=f"Mean ROC (AUC={mean_auc:.3f}±{std_auc:.3f})")
        plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color="b", alpha=0.2)
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC curves (cross-validation) - {model_name}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_dir / "roc_curve.png")
        plt.close()

    if pr_curves:
        mean_recall = np.linspace(0, 1, 200)
        precisions = []
        ap_scores = []
        for recall, precision, _ in pr_curves:
            precision = np.maximum.accumulate(precision[::-1])[::-1]
            interp_precision = np.interp(mean_recall, recall, precision)
            precisions.append(interp_precision)
            ap_scores.append(np.trapz(precision, recall))
        mean_precision = np.mean(precisions, axis=0)
        std_precision = np.std(precisions, axis=0)
        mean_ap = np.mean(ap_scores)
        std_ap = np.std(ap_scores)

        plt.figure()
        plt.plot(mean_recall, mean_precision, color="g", label=f"Mean PR (AP={mean_ap:.3f}±{std_ap:.3f})")
        plt.fill_between(
            mean_recall,
            np.maximum(mean_precision - std_precision, 0),
            np.minimum(mean_precision + std_precision, 1),
            color="g",
            alpha=0.2,
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-recall curves (cross-validation) - {model_name}")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(output_dir / "pr_curve.png")
        plt.close()

    LOGGER.info("Saved ROC and PR cross-validation plots under %s", output_dir)


def _plot_calibration(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    output_dir: Path,
) -> float:
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="uniform")
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Calibration Curve - {model_name}")
    plt.tight_layout()
    path = output_dir / "calibration_curve.png"
    plt.savefig(path)
    plt.close()
    brier = brier_score_loss(y_true, y_proba)
    LOGGER.info("Saved calibration plot and recorded Brier score %.4f at %s", brier, path)
    return brier


def _plot_confusion_matrices(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    output_dir: Path,
) -> Tuple[float, np.ndarray, np.ndarray]:
    thresholds = np.linspace(0, 1, 500)
    youden_scores = []
    for thr in thresholds:
        preds = (y_proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) else 0
        specificity = tn / (tn + fp) if (tn + fp) else 0
        youden_scores.append(sensitivity + specificity - 1)
    best_idx = int(np.argmax(youden_scores))
    best_threshold = thresholds[best_idx]

    def _plot_matrix(threshold: float, suffix: str) -> np.ndarray:
        preds = (y_proba >= threshold).astype(int)
        matrix = confusion_matrix(y_true, preds)
        plt.figure()
        plt.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix ({suffix}) - {model_name}")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["No", "Yes"])
        plt.yticks(tick_marks, ["No", "Yes"])
        thresh = matrix.max() / 2.
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                plt.text(j, i, format(matrix[i, j], "d"),
                         horizontalalignment="center",
                         color="white" if matrix[i, j] > thresh else "black")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        out_path = output_dir / f"confusion_matrix_{suffix}.png"
        plt.savefig(out_path)
        plt.close()
        return matrix

    ensure_dir(output_dir)
    matrix_default = _plot_matrix(0.5, "threshold_0.50")
    matrix_optimal = _plot_matrix(best_threshold, "youden")
    LOGGER.info(
        "Saved confusion matrices for the default threshold and Youden %.3f.",
        best_threshold,
    )
    return best_threshold, matrix_default, matrix_optimal


def _plot_feature_importance(
    feature_importance: pd.Series,
    model_name: str,
    output_dir: Path,
) -> None:
    ensure_dir(output_dir)
    top_features = feature_importance.sort_values(ascending=False).head(20)
    plt.figure(figsize=(10, 6))
    top_features[::-1].plot(kind="barh")
    plt.xlabel("Importance")
    plt.title(f"Feature Importance - {model_name}")
    plt.tight_layout()
    path = output_dir / "feature_importance.png"
    plt.savefig(path)
    plt.close()
    top_features.to_csv(output_dir / "feature_importance_top20.csv")
    LOGGER.info("Saved feature importance visuals and table to %s", output_dir)


# =============================================================================
# Baseline training
# =============================================================================

def train_baselines(
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    output_dir: Path,
    models: Optional[Sequence[str]] = None,
) -> Dict[str, CVResult]:
    """Train straightforward baseline models for comparison."""

    ensure_dir(output_dir)
    models = models or ["logreg", "rf"]
    results: Dict[str, CVResult] = {}
    metric_tables: Dict[str, pd.DataFrame] = {}

    for model_name in models:
        LOGGER.info("Starting baseline training for %s", model_name)
        if model_name == "logreg":
            solver = "liblinear" if X.shape[0] < 5000 else "saga"
            n_jobs = -1 if solver == "saga" else None
            base_estimator: ClassifierMixin = LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=5000,
                solver=solver,
                n_jobs=n_jobs,
            )
        elif model_name == "rf":
            base_estimator = RandomForestClassifier(
                n_estimators=500,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced_subsample",
            )
        else:
            LOGGER.warning("Unknown baseline model %s — skipping", model_name)
            continue

        fold_metrics: List[Dict[str, float]] = []
        oof_pred = np.zeros(len(y), dtype=float)
        oof_indices = np.zeros(len(y), dtype=int)
        fold_roc_curves: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        fold_pr_curves: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
            LOGGER.info("Fold %s/%s", fold, cv.n_splits)
            estimator = clone(base_estimator)
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            estimator.fit(X_train, y_train)
            proba = estimator.predict_proba(X_valid)[:, 1]
            oof_pred[valid_idx] = proba
            oof_indices[valid_idx] = valid_idx
            metrics = _compute_fold_metrics(y_valid, proba)
            fold_metrics.append(metrics)
            fpr, tpr, roc_thr = roc_curve(y_valid, proba)
            precision, recall, pr_thr = precision_recall_curve(y_valid, proba)
            fold_roc_curves.append((fpr, tpr, roc_thr))
            fold_pr_curves.append((recall, precision, pr_thr))
        metrics_df = _aggregate_metrics(fold_metrics)
        metric_tables[model_name] = metrics_df
        estimator_final = clone(base_estimator)
        estimator_final.fit(X, y)
        feature_importance = None
        if hasattr(estimator_final, "feature_importances_"):
            feature_importance = pd.Series(
                estimator_final.feature_importances_, index=X.columns
            )
        elif hasattr(estimator_final, "coef_"):
            coefs = estimator_final.coef_[0] if estimator_final.coef_.ndim > 1 else estimator_final.coef_
            feature_importance = pd.Series(coefs, index=X.columns)

        result = CVResult(
            model_name=model_name,
            oof_predictions=oof_pred,
            oof_indices=oof_indices,
            fold_metrics=metrics_df,
            final_estimator=estimator_final,
            feature_importances=feature_importance,
            roc_curves=fold_roc_curves,
            pr_curves=fold_pr_curves,
        )
        results[model_name] = result

        model_dir = output_dir / model_name
        ensure_dir(model_dir)
        _save_metrics(metrics_df, model_dir / f"metrics_{model_name}.csv")
        _save_oof_predictions(oof_pred, oof_indices, model_dir / "oof_predictions.csv")
        joblib.dump(estimator_final, model_dir / f"model_{model_name}.joblib")
        LOGGER.info("Stored trained baseline model in %s", model_dir)
        _plot_cv_curves(fold_roc_curves, fold_pr_curves, model_name, model_dir)
        brier = _plot_calibration(y.to_numpy(), oof_pred, model_name, model_dir)
        best_threshold, _, _ = _plot_confusion_matrices(
            y.to_numpy(), oof_pred, model_name, model_dir
        )
        metrics_df.loc["mean", "brier"] = brier
        metrics_df.loc["mean", "youden_threshold"] = best_threshold
        if feature_importance is not None:
            _plot_feature_importance(feature_importance, model_name, model_dir)

    _plot_cv_boxplot(metric_tables, output_dir / "cv_boxplot_baselines.png")
    return results


# =============================================================================
# Hyperparameter tuning for the advanced models
# =============================================================================

def _objective_factory(
    estimator_cls: Any,
    param_space: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
) -> Any:
    """Build an Optuna objective tailored to the given estimator and space."""

    def objective(trial: "optuna.trial.Trial") -> float:
        params = {}
        for key, sampler in param_space.items():
            params[key] = sampler(trial)
        estimator = estimator_cls(random_state=RANDOM_STATE, **params)
        scores: List[float] = []
        for train_idx, valid_idx in cv.split(X, y):
            est = clone(estimator)
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            fit_params = {}
            if hasattr(est, "eval_set"):
                fit_params["eval_set"] = [(X_valid, y_valid)]
                fit_params["verbose"] = False
                fit_params["early_stopping_rounds"] = 50
            est.fit(X_train, y_train, **fit_params)
            proba = est.predict_proba(X_valid)[:, 1]
            scores.append(roc_auc_score(y_valid, proba))
        mean_score = float(np.mean(scores))
        LOGGER.info("Trial score %.4f with parameters %s", mean_score, params)
        return mean_score

    return objective


def _grid_search(
    estimator: BaseEstimator,
    param_grid: Dict[str, Sequence[Any]],
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    from itertools import product

    best_score = -np.inf
    best_params = {}
    keys = list(param_grid.keys())
    for values in product(*param_grid.values()):
        params = dict(zip(keys, values))
        scores: List[float] = []
        for train_idx, valid_idx in cv.split(X, y):
            est = clone(estimator)
            est.set_params(**params)
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            fit_params = {}
            if hasattr(est, "eval_set"):
                fit_params["eval_set"] = [(X_valid, y_valid)]
                fit_params["verbose"] = False
                fit_params["early_stopping_rounds"] = 50
            est.fit(X_train, y_train, **fit_params)
            proba = est.predict_proba(X_valid)[:, 1]
            scores.append(roc_auc_score(y_valid, proba))
        score = float(np.mean(scores))
        LOGGER.info("Grid-search score %.4f with parameters %s", score, params)
        if score > best_score:
            best_score = score
            best_params = params
    best_estimator = clone(estimator).set_params(**best_params)
    best_estimator.fit(X, y)
    return best_estimator, best_params


def tune_models(
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    output_dir: Path,
    use_optuna: bool = True,
    n_trials: int = 40,
    models: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Tune the advanced models and report the best parameters found."""

    ensure_dir(output_dir)
    models = models or ["xgboost", "lightgbm", "catboost"]
    tuning_results: Dict[str, Dict[str, Any]] = {}

    for model_name in models:
        LOGGER.info("Running hyperparameter search for %s", model_name)
        if model_name == "xgboost" and XGBClassifier is not None:
            estimator_cls = XGBClassifier
            default_params = {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "tree_method": "hist",
                "use_label_encoder": False,
                "random_state": RANDOM_STATE,
            }
            optuna_space = {
                "n_estimators": lambda t: t.suggest_int("n_estimators", 200, 800),
                "max_depth": lambda t: t.suggest_int("max_depth", 3, 10),
                "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": lambda t: t.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": lambda t: t.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": lambda t: t.suggest_float("min_child_weight", 1.0, 10.0),
                "reg_alpha": lambda t: t.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": lambda t: t.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }
            grid_space = {
                "n_estimators": [300, 500, 700],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.03, 0.1, 0.2],
            }
        elif model_name == "lightgbm" and LGBMClassifier is not None:
            estimator_cls = LGBMClassifier
            default_params = {
                "objective": "binary",
                "metric": "auc",
                "verbosity": -1,
                "random_state": RANDOM_STATE,
                "n_jobs": -1,
            }
            optuna_space = {
                "n_estimators": lambda t: t.suggest_int("n_estimators", 200, 1000),
                "num_leaves": lambda t: t.suggest_int("num_leaves", 16, 128),
                "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": lambda t: t.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": lambda t: t.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_samples": lambda t: t.suggest_int("min_child_samples", 10, 100),
                "reg_alpha": lambda t: t.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
                "reg_lambda": lambda t: t.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
            }
            grid_space = {
                "n_estimators": [300, 600, 900],
                "num_leaves": [31, 63, 95],
                "learning_rate": [0.03, 0.1, 0.2],
            }
        elif model_name == "catboost" and CatBoostClassifier is not None:
            estimator_cls = CatBoostClassifier
            default_params = {
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "random_seed": RANDOM_STATE,
                "verbose": 0,
                "thread_count": -1,
            }
            optuna_space = {
                "iterations": lambda t: t.suggest_int("iterations", 300, 800),
                "depth": lambda t: t.suggest_int("depth", 4, 10),
                "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg": lambda t: t.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "subsample": lambda t: t.suggest_float("subsample", 0.6, 1.0),
                "border_count": lambda t: t.suggest_int("border_count", 32, 255),
            }
            grid_space = {
                "iterations": [400, 600, 800],
                "depth": [5, 7, 9],
                "learning_rate": [0.03, 0.1, 0.2],
            }
        else:
            LOGGER.warning("%s is not available — skipping tuning", model_name)
            continue

        model_dir = output_dir / model_name
        ensure_dir(model_dir)

        if use_optuna and OPTUNA_AVAILABLE:
            objective = _objective_factory(estimator_cls, optuna_space, X, y, cv)
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            best_params = study.best_params
            LOGGER.info("Best Optuna parameters for %s: %s", model_name, best_params)
            with open(model_dir / "study_summary.json", "w", encoding="utf-8") as f:
                json.dump({"best_value": study.best_value, "best_params": best_params}, f, indent=2)
        else:
            LOGGER.info("Optuna not available or disabled — falling back to grid search.")
            estimator = estimator_cls(**default_params)
            best_estimator, best_params = _grid_search(estimator, grid_space, X, y, cv)
            joblib.dump(best_estimator, model_dir / "best_estimator_grid.joblib")

        if use_optuna and OPTUNA_AVAILABLE:
            best_estimator = estimator_cls(**default_params, **best_params)
        else:
            best_estimator = estimator_cls(**default_params, **best_params)
        best_estimator.fit(X, y)
        joblib.dump(best_estimator, model_dir / f"model_{model_name}.joblib")
        LOGGER.info("Saved tuned model to %s", model_dir)
        tuning_results[model_name] = {
            "best_params": best_params,
            "model_path": str(model_dir / f"model_{model_name}.joblib"),
        }

    return tuning_results


# =============================================================================
# Final training and evaluation
# =============================================================================

def _train_with_params(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    model_name: str,
    output_dir: Path,
) -> CVResult:
    estimator = clone(estimator)
    fold_metrics: List[Dict[str, float]] = []
    oof_pred = np.zeros(len(y), dtype=float)
    oof_indices = np.zeros(len(y), dtype=int)
    feature_importance = None
    fold_roc_curves: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    fold_pr_curves: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        LOGGER.info("[%s] Fold %s/%s", model_name, fold, cv.n_splits)
        est = clone(estimator)
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        fit_params = {}
        if isinstance(est, (RandomForestClassifier,)):
            est.set_params(n_jobs=-1)
        if hasattr(est, "eval_set"):
            fit_params["eval_set"] = [(X_valid, y_valid)]
            fit_params["verbose"] = False
            fit_params["early_stopping_rounds"] = 50
        if hasattr(est, "fit"):
            est.fit(X_train, y_train, **fit_params)
        proba = est.predict_proba(X_valid)[:, 1]
        oof_pred[valid_idx] = proba
        oof_indices[valid_idx] = valid_idx
        fold_metrics.append(_compute_fold_metrics(y_valid, proba))
        fpr, tpr, roc_thr = roc_curve(y_valid, proba)
        precision, recall, pr_thr = precision_recall_curve(y_valid, proba)
        fold_roc_curves.append((fpr, tpr, roc_thr))
        fold_pr_curves.append((recall, precision, pr_thr))

    estimator.fit(X, y)
    if hasattr(estimator, "feature_importances_"):
        feature_importance = pd.Series(estimator.feature_importances_, index=X.columns)
    elif hasattr(estimator, "coef_"):
        coefs = estimator.coef_[0] if estimator.coef_.ndim > 1 else estimator.coef_
        feature_importance = pd.Series(coefs, index=X.columns)

    metrics_df = _aggregate_metrics(fold_metrics)
    ensure_dir(output_dir)
    _save_metrics(metrics_df, output_dir / f"metrics_{model_name}.csv")
    _save_oof_predictions(oof_pred, oof_indices, output_dir / "oof_predictions.csv")
    joblib.dump(estimator, output_dir / f"model_{model_name}.joblib")

    _plot_cv_curves(fold_roc_curves, fold_pr_curves, model_name, output_dir)
    brier = _plot_calibration(y.to_numpy(), oof_pred, model_name, output_dir)
    best_threshold, _, _ = _plot_confusion_matrices(y.to_numpy(), oof_pred, model_name, output_dir)
    metrics_df.loc["mean", "brier"] = brier
    metrics_df.loc["mean", "youden_threshold"] = best_threshold
    if feature_importance is not None:
        _plot_feature_importance(feature_importance, model_name, output_dir)

    return CVResult(
        model_name=model_name,
        oof_predictions=oof_pred,
        oof_indices=oof_indices,
        fold_metrics=metrics_df,
        final_estimator=estimator,
        feature_importances=feature_importance,
        roc_curves=fold_roc_curves,
        pr_curves=fold_pr_curves,
    )


def fit_advanced_models(
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    tuning_info: Dict[str, Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, CVResult]:
    results: Dict[str, CVResult] = {}
    for model_name, info in tuning_info.items():
        model_path = Path(info["model_path"])
        if not model_path.exists():
            LOGGER.warning("Model file %s not found — skipping", model_path)
            continue
        estimator = joblib.load(model_path)
        model_dir = output_dir / model_name
        ensure_dir(model_dir)
        result = _train_with_params(estimator, X, y, cv, model_name, model_dir)
        results[model_name] = result
    return results


# =============================================================================
# Ensemble
# =============================================================================

def fit_ensemble(
    results: Dict[str, CVResult],
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    output_dir: Path,
    top_k: int = 3,
) -> Optional[CVResult]:
    if len(results) < top_k:
        LOGGER.warning("Not enough models available for an ensemble — skipping")
        return None

    sorted_models = sorted(
        results.values(),
        key=lambda res: res.fold_metrics.loc["mean", "roc_auc"],
        reverse=True,
    )
    selected = sorted_models[:top_k]
    LOGGER.info("Ensemble will use the following models: %s", [res.model_name for res in selected])

    estimators = []
    weights = []
    for res in selected:
        estimators.append((res.model_name, res.final_estimator))
        weights.append(res.fold_metrics.loc["mean", "roc_auc"])

    ensemble = VotingClassifier(
        estimators=estimators,
        voting="soft",
        weights=weights,
        n_jobs=-1,
    )

    result = _train_with_params(ensemble, X, y, cv, "ensemble", output_dir)
    with open(output_dir / "ensemble_members.json", "w", encoding="utf-8") as f:
        json.dump({"members": [res.model_name for res in selected], "weights": weights}, f, indent=2)
    LOGGER.info("Stored ensemble artifacts.")
    return result


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_models(
    results: Dict[str, CVResult],
    y: pd.Series,
    output_dir: Path,
) -> pd.DataFrame:
    ensure_dir(output_dir)
    summary_rows = []
    metrics_path = ensure_dir(output_dir / "metrics")
    plots_path = ensure_dir(output_dir / "plots")
    y_true = y.to_numpy()

    for model_name, res in results.items():
        model_metrics_path = metrics_path / f"metrics_{model_name}.csv"
        res.fold_metrics.to_csv(model_metrics_path)
        LOGGER.info("Saved metrics again to %s", model_metrics_path)
        model_plot_dir = ensure_dir(plots_path / model_name)
        _plot_cv_curves(res.roc_curves or [], res.pr_curves or [], model_name, model_plot_dir)
        _plot_calibration(y_true, res.oof_predictions, model_name, model_plot_dir)
        best_thr, _, _ = _plot_confusion_matrices(y_true, res.oof_predictions, model_name, model_plot_dir)
        if res.feature_importances is not None:
            _plot_feature_importance(res.feature_importances, model_name, model_plot_dir)
        summary_rows.append({
            "model": model_name,
            "roc_auc": res.fold_metrics.loc["mean", "roc_auc"],
            "pr_auc": res.fold_metrics.loc["mean", "pr_auc"],
            "f1": res.fold_metrics.loc["mean", "f1"],
            "brier": res.fold_metrics.loc["mean", "brier"],
            "youden_threshold": best_thr,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("roc_auc", ascending=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    LOGGER.info("Stored overall evaluation summary at %s", output_dir / "summary.csv")
    return summary_df


# =============================================================================
# SHAP and interpretability helpers
# =============================================================================

def interpret_models(
    results: Dict[str, CVResult],
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    sample_size: int = 2000,
) -> None:
    if shap is None:
        LOGGER.warning("SHAP is not available — skipping interpretability exports")
        return

    ensure_dir(output_dir)
    sample_indices = np.random.choice(len(X), size=min(sample_size, len(X)), replace=False)
    X_sample = X.iloc[sample_indices]
    shap_summary = []

    for model_name, res in results.items():
        LOGGER.info("Computing SHAP values for %s", model_name)
        model_dir = ensure_dir(output_dir / model_name)
        estimator = res.final_estimator
        try:
            if hasattr(estimator, "get_booster") or "xgb" in model_name:
                explainer = shap.TreeExplainer(estimator)
            elif hasattr(estimator, "predict_proba") and hasattr(estimator, "feature_importances_"):
                explainer = shap.TreeExplainer(estimator)
            elif isinstance(estimator, LogisticRegression):
                explainer = shap.LinearExplainer(estimator, X_sample)
            else:
                LOGGER.info("Falling back to KernelExplainer for %s — this may take a while", model_name)
                background = shap.sample(X, min(200, len(X)))
                explainer = shap.KernelExplainer(estimator.predict_proba, background)

            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            res.shap_values = shap_values
            expected_value = explainer.expected_value
            if isinstance(expected_value, (list, tuple, np.ndarray)):
                expected_value = expected_value[1]
            res.shap_expected_value = float(expected_value)

            shap.summary_plot(shap_values, X_sample, show=False)
            plt.tight_layout()
            plt.savefig(model_dir / f"shap_summary_{model_name}.png")
            plt.close()

            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(model_dir / f"shap_bar_{model_name}.png")
            plt.close()

            mean_abs = np.abs(shap_values).mean(axis=0)
            shap_importance = pd.Series(mean_abs, index=X_sample.columns).sort_values(ascending=False)
            shap_importance.head(20).to_csv(model_dir / f"shap_top_features_{model_name}.csv")
            shap_summary.append((model_name, shap_importance))

            # Local explanation snippets -----------------------------------------
            sample_pred = estimator.predict_proba(X_sample)[:, 1]
            candidates = {
                "highest_positive": int(np.argmax(sample_pred)),
                "lowest_negative": int(np.argmin(sample_pred)),
                "median": int(np.argsort(sample_pred)[len(sample_pred) // 2]),
                "near_threshold": int(np.argmin(np.abs(sample_pred - 0.5))),
                "random": int(np.random.choice(len(sample_pred))),
            }
            for label, idx in candidates.items():
                shap.force_plot(
                    res.shap_expected_value,
                    shap_values[idx],
                    X_sample.iloc[idx],
                    matplotlib=True,
                )
                plt.tight_layout()
                plt.savefig(model_dir / f"shap_force_{label}.png", bbox_inches="tight")
                plt.close()
        except Exception as exc:  # pragma: no cover - fallback
            LOGGER.warning("SHAP evaluation for %s failed: %s", model_name, exc)
            continue

    # Interpretation Notes ----------------------------------------------------
    notes_path = output_dir / "interpretation_notes.md"
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write("# Interpretation Notes\n\n")
        f.write("TODO: Drop selected figures into the paper or slide deck.\n\n")
        f.write(
            "The top features are ranked by the mean absolute SHAP values from the strongest models.\n"
        )
        for model_name, shap_importance in shap_summary:
            top_features = ", ".join(shap_importance.head(5).index)
            f.write(f"- {model_name}: {top_features}\n")
        f.write(
            "\nHypothesis check: Height (higher = more risk), liver enzymes (higher), and HDL (lower) should appear among the drivers."
        )
        f.write(
            " Please watch out for gender-specific biases or confounding introduced by synthetic data."  # noqa: E501
        )
    LOGGER.info("Saved interpretation notes to %s", notes_path)


# =============================================================================
# Persistence helpers
# =============================================================================

def save_artifacts(
    baseline_results: Dict[str, CVResult],
    advanced_results: Dict[str, CVResult],
    ensemble_result: Optional[CVResult],
    output_dir: Path,
    preprocessing: Optional[PreprocessingArtifacts] = None,
) -> None:
    ensure_dir(output_dir)
    if preprocessing is not None:
        prep_dir = ensure_dir(Path("artifacts") / "preprocessing")
        joblib.dump(preprocessing.cleaner, prep_dir / "data_cleaning.joblib")
        joblib.dump(preprocessing.preprocessor, prep_dir / "feature_preprocessor.joblib")
        with open(prep_dir / "feature_names.json", "w", encoding="utf-8") as fh:
            json.dump({"feature_names": preprocessing.feature_names}, fh, indent=2)
        LOGGER.info("Saved preprocessing artifacts to %s", prep_dir)

    summary_records = []
    for res in list(baseline_results.values()) + list(advanced_results.values()):
        summary_records.append(
            {
                "model": res.model_name,
                "roc_auc_mean": res.fold_metrics.loc["mean", "roc_auc"],
                "roc_auc_std": res.fold_metrics.loc["std", "roc_auc"],
                "pr_auc_mean": res.fold_metrics.loc["mean", "pr_auc"],
                "f1_mean": res.fold_metrics.loc["mean", "f1"],
                "brier_mean": res.fold_metrics.loc["mean", "brier"],
            }
        )
    if ensemble_result is not None:
        summary_records.append(
            {
                "model": ensemble_result.model_name,
                "roc_auc_mean": ensemble_result.fold_metrics.loc["mean", "roc_auc"],
                "roc_auc_std": ensemble_result.fold_metrics.loc["std", "roc_auc"],
                "pr_auc_mean": ensemble_result.fold_metrics.loc["mean", "pr_auc"],
                "f1_mean": ensemble_result.fold_metrics.loc["mean", "f1"],
                "brier_mean": ensemble_result.fold_metrics.loc["mean", "brier"],
            }
        )
    summary_df = pd.DataFrame(summary_records)
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    LOGGER.info("Stored consolidated summary at %s", summary_path)


# =============================================================================
# CLI und Hauptablauf
# =============================================================================

def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoking Status ML-Pipeline")
    parser.add_argument("--data-path", type=Path, default=Path("train_dataset.csv"))
    parser.add_argument("--target", type=str, default="smoking")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--use-optuna", action="store_true")
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=["logreg", "rf", "xgboost", "lightgbm", "catboost"],
        help="Which models to train",
    )
    parser.add_argument("--do-ensemble", action="store_true")
    return parser.parse_args(args)


def main(cli_args: Optional[Sequence[str]] = None) -> None:
    args = parse_args(cli_args)
    np.random.seed(args.seed)

    LOGGER.info("Starting pipeline with arguments: %s", args)
    X, y, preprocessing = load_data(args.data_path, args.target)
    cv = get_cv(args.cv_folds, args.seed)

    baseline_models = [m for m in args.models if m in {"logreg", "rf"}]
    advanced_models = [m for m in args.models if m in {"xgboost", "lightgbm", "catboost"}]

    baseline_results = train_baselines(
        X,
        y,
        cv,
        Path("artifacts/baseline"),
        models=baseline_models,
    )

    tuning_info = tune_models(
        X,
        y,
        cv,
        Path("artifacts/tuning"),
        use_optuna=args.use_optuna,
        n_trials=args.n_trials,
        models=advanced_models,
    )

    advanced_results = fit_advanced_models(
        X,
        y,
        cv,
        tuning_info,
        Path("artifacts/advanced"),
    )

    ensemble_result = None
    if args.do_ensemble:
        combined_results = {**baseline_results, **advanced_results}
        ensemble_result = fit_ensemble(
            combined_results,
            X,
            y,
            cv,
            Path("artifacts/ensemble"),
        )
        if ensemble_result:
            combined_results["ensemble"] = ensemble_result
    else:
        combined_results = {**baseline_results, **advanced_results}

    evaluate_models(combined_results, y, Path("results"))
    interpret_models(combined_results, X, y, Path("results/interpretation"))
    save_artifacts(
        baseline_results,
        advanced_results,
        ensemble_result,
        Path("results"),
        preprocessing=preprocessing,
    )

    # Console summary ---------------------------------------------------------
    summary_records = []
    for model_name, res in combined_results.items():
        summary_records.append(
            (
                model_name,
                res.fold_metrics.loc["mean", "roc_auc"],
                res.fold_metrics.loc["std", "roc_auc"],
                res.fold_metrics.loc["mean", "pr_auc"],
                res.fold_metrics.loc["mean", "f1"],
                res.fold_metrics.loc["mean", "brier"],
            )
        )
    summary_sorted = sorted(summary_records, key=lambda x: x[1], reverse=True)
    LOGGER.info("\nRanking by cross-validated ROC-AUC:")
    for rank, record in enumerate(summary_sorted, start=1):
        LOGGER.info(
            "%d) %s — ROC-AUC: %.3f ± %.3f | PR-AUC: %.3f | F1: %.3f | Brier: %.3f",
            rank,
            record[0],
            record[1],
            record[2],
            record[3],
            record[4],
            record[5],
        )

    LOGGER.info("Artifacts are stored under artifacts/, results under results/.")
    LOGGER.info("TODO: Promote selected plots from results/plots into the report or slides.")


if __name__ == "__main__":
    main()
