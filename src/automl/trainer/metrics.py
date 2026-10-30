"""Evaluation metrics computation for classification and regression tasks."""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error

    def root_mean_squared_error(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

from sklearn.metrics import roc_auc_score

from automl.profiler.schemas import TaskType
from automl.trainer.schemas import MetricName


def compute_metrics(
    task_type: TaskType,
    requested_metrics: List[MetricName],
    y_true: Any,
    y_pred: Any,
    y_proba: Optional[Any] = None,
) -> Dict[str, float]:
    """Compute evaluated metric scores for a given set of true and predicted targets.

    Args:
        task_type: TaskType (binary_classification, multiclass_classification, regression).
        requested_metrics: List of metrics to compute.
        y_true: Ground truth target array.
        y_pred: Predicted target array.
        y_proba: Predicted probability array (optional, used for ROC-AUC).

    Returns:
        Dict[str, float]: Dictionary mapping metric name to calculated float score.
    """
    scores: Dict[str, float] = {}

    for metric in requested_metrics:
        # Classification
        if metric == MetricName.ACCURACY:
            scores[metric.value] = round(float(accuracy_score(y_true, y_pred)), 4)

        elif metric == MetricName.PRECISION_MACRO:
            scores[metric.value] = round(
                float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4
            )

        elif metric == MetricName.RECALL_MACRO:
            scores[metric.value] = round(
                float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4
            )

        elif metric == MetricName.F1_MACRO:
            scores[metric.value] = round(
                float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4
            )

        elif metric == MetricName.ROC_AUC_OVR:
            if y_proba is not None:
                try:
                    if task_type == TaskType.BINARY_CLASSIFICATION:
                        # Extract positive class probability
                        pos_proba = y_proba[:, 1] if y_proba.ndim == 2 and y_proba.shape[1] == 2 else y_proba
                        score = roc_auc_score(y_true, pos_proba)
                    else:
                        score = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                    scores[metric.value] = round(float(score), 4)
                except Exception:
                    # e.g., if one fold does not contain all classes
                    scores[metric.value] = 0.5
            else:
                scores[metric.value] = 0.5

        # Regression
        elif metric == MetricName.RMSE:
            scores[metric.value] = round(float(root_mean_squared_error(y_true, y_pred)), 4)

        elif metric == MetricName.MAE:
            scores[metric.value] = round(float(mean_absolute_error(y_true, y_pred)), 4)

        elif metric == MetricName.R2:
            scores[metric.value] = round(float(r2_score(y_true, y_pred)), 4)

    return scores
