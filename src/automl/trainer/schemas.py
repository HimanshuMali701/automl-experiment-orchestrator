"""Experiment configuration and result schemas for the deterministic ML engine."""

from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field, field_validator, model_validator

from automl.profiler.schemas import TaskType


class MetricName(str, Enum):
    """Supported evaluation metrics."""
    # Classification metrics
    ACCURACY = "accuracy"
    PRECISION_MACRO = "precision_macro"
    RECALL_MACRO = "recall_macro"
    F1_MACRO = "f1_macro"
    ROC_AUC_OVR = "roc_auc_ovr"

    # Regression metrics
    RMSE = "rmse"
    MAE = "mae"
    R2 = "r2"


CLASSIFICATION_METRICS = {
    MetricName.ACCURACY,
    MetricName.PRECISION_MACRO,
    MetricName.RECALL_MACRO,
    MetricName.F1_MACRO,
    MetricName.ROC_AUC_OVR,
}

REGRESSION_METRICS = {
    MetricName.RMSE,
    MetricName.MAE,
    MetricName.R2,
}

# Metrics where higher value is better
GREATER_IS_BETTER: Dict[MetricName, bool] = {
    MetricName.ACCURACY: True,
    MetricName.PRECISION_MACRO: True,
    MetricName.RECALL_MACRO: True,
    MetricName.F1_MACRO: True,
    MetricName.ROC_AUC_OVR: True,
    MetricName.RMSE: False,
    MetricName.MAE: False,
    MetricName.R2: True,
}


class ModelFamily(str, Enum):
    """Allowed model families."""
    # Baselines
    DUMMY_CLASSIFIER = "dummy_classifier"
    DUMMY_REGRESSOR = "dummy_regressor"

    # Classification
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST_CLASSIFIER = "random_forest_classifier"
    GRADIENT_BOOSTING_CLASSIFIER = "gradient_boosting_classifier"

    # Regression
    RIDGE_REGRESSION = "ridge_regression"
    RANDOM_FOREST_REGRESSOR = "random_forest_regressor"
    GRADIENT_BOOSTING_REGRESSOR = "gradient_boosting_regressor"


CLASSIFICATION_MODELS = {
    ModelFamily.DUMMY_CLASSIFIER,
    ModelFamily.LOGISTIC_REGRESSION,
    ModelFamily.RANDOM_FOREST_CLASSIFIER,
    ModelFamily.GRADIENT_BOOSTING_CLASSIFIER,
}

REGRESSION_MODELS = {
    ModelFamily.DUMMY_REGRESSOR,
    ModelFamily.RIDGE_REGRESSION,
    ModelFamily.RANDOM_FOREST_REGRESSOR,
    ModelFamily.GRADIENT_BOOSTING_REGRESSOR,
}


class ImputationStrategy(str, Enum):
    """Strategy for missing value imputation."""
    MEAN = "mean"
    MEDIAN = "median"
    MOST_FREQUENT = "most_frequent"
    CONSTANT = "constant"


class ScalingStrategy(str, Enum):
    """Scaling strategy for numeric features."""
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    NONE = "none"


class PreprocessingConfig(BaseModel):
    """Data preprocessing options for scikit-learn Pipeline and ColumnTransformer."""
    numeric_imputer: ImputationStrategy = ImputationStrategy.MEDIAN
    categorical_imputer: ImputationStrategy = ImputationStrategy.MOST_FREQUENT
    scaling: ScalingStrategy = ScalingStrategy.STANDARD
    handle_unknown_categories: str = "ignore"


class ModelCandidateConfig(BaseModel):
    """Configuration for a single model candidate."""
    name: str
    model_family: ModelFamily
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    is_baseline: bool = False


class RunStatus(str, Enum):
    """Execution status of an individual candidate run."""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class FoldResult(BaseModel):
    """Validation metrics for a single CV fold."""
    fold_index: int
    metrics: Dict[str, float]
    fit_time_seconds: float


class CandidateRunResult(BaseModel):
    """Execution results, metrics, and parameters for an evaluated model candidate."""
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    candidate_name: str
    model_family: ModelFamily
    is_baseline: bool = False
    status: RunStatus
    parameters: Dict[str, Any] = Field(default_factory=dict)
    cv_mean_metrics: Dict[str, float] = Field(default_factory=dict)
    cv_std_metrics: Dict[str, float] = Field(default_factory=dict)
    fold_results: List[FoldResult] = Field(default_factory=list)
    test_metrics: Optional[Dict[str, float]] = None
    fit_time_seconds: float = 0.0
    error_message: Optional[str] = None


class ExperimentConfig(BaseModel):
    """Complete specification for a deterministic ML experiment."""
    target_column: str
    task_type: TaskType
    primary_metric: MetricName
    secondary_metrics: List[MetricName] = Field(default_factory=list)
    test_size: float = Field(default=0.2, gt=0.0, lt=0.5)
    cv_folds: int = Field(default=5, ge=2, le=10)
    random_seed: int = 42
    candidates: List[ModelCandidateConfig] = Field(default_factory=list)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    max_total_time_seconds: int = Field(default=300, gt=0)

    @model_validator(mode="after")
    def validate_metrics_and_models(self) -> "ExperimentConfig":
        """Validate task consistency with metrics and candidate models."""
        # 1. Validate primary metric
        if self.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
            if self.primary_metric not in CLASSIFICATION_METRICS:
                raise ValueError(
                    f"Primary metric '{self.primary_metric.value}' is invalid for classification task. "
                    f"Choose from {[m.value for m in CLASSIFICATION_METRICS]}."
                )
            for m in self.secondary_metrics:
                if m not in CLASSIFICATION_METRICS:
                    raise ValueError(f"Secondary metric '{m.value}' is invalid for classification task.")
        elif self.task_type == TaskType.REGRESSION:
            if self.primary_metric not in REGRESSION_METRICS:
                raise ValueError(
                    f"Primary metric '{self.primary_metric.value}' is invalid for regression task. "
                    f"Choose from {[m.value for m in REGRESSION_METRICS]}."
                )
            for m in self.secondary_metrics:
                if m not in REGRESSION_METRICS:
                    raise ValueError(f"Secondary metric '{m.value}' is invalid for regression task.")

        # 2. Validate candidate models
        if len(self.candidates) > 10:
            raise ValueError(f"Number of candidates ({len(self.candidates)}) exceeds maximum limit of 10.")

        for cand in self.candidates:
            if self.task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
                if cand.model_family not in CLASSIFICATION_MODELS:
                    raise ValueError(
                        f"Candidate '{cand.name}' uses model '{cand.model_family}' which is invalid for classification."
                    )
            elif self.task_type == TaskType.REGRESSION:
                if cand.model_family not in REGRESSION_MODELS:
                    raise ValueError(
                        f"Candidate '{cand.name}' uses model '{cand.model_family}' which is invalid for regression."
                    )

        return self


class ExperimentResult(BaseModel):
    """Aggregate experiment results including ranked leaderboard and test set evaluation."""
    experiment_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_type: TaskType
    target_column: str
    primary_metric: MetricName
    best_candidate_name: Optional[str] = None
    leaderboard: List[CandidateRunResult] = Field(default_factory=list)
    total_execution_time_seconds: float = 0.0
    train_rows: int
    test_rows: int
    warnings: List[str] = Field(default_factory=list)
