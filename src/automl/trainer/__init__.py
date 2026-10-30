"""Leakage-free ML training pipelines, preprocessing, model evaluations, and experiment engine."""

from automl.trainer.engine import ExperimentEngine, build_default_experiment_config
from automl.trainer.metrics import compute_metrics
from automl.trainer.models import instantiate_model
from automl.trainer.preprocessing import build_preprocessor, extract_feature_columns
from automl.trainer.schemas import (
    CandidateRunResult,
    ExperimentConfig,
    ExperimentResult,
    FoldResult,
    ImputationStrategy,
    MetricName,
    ModelCandidateConfig,
    ModelFamily,
    PreprocessingConfig,
    RunStatus,
    ScalingStrategy,
)

__all__ = [
    "ExperimentEngine",
    "build_default_experiment_config",
    "build_preprocessor",
    "extract_feature_columns",
    "instantiate_model",
    "compute_metrics",
    "ExperimentConfig",
    "ExperimentResult",
    "CandidateRunResult",
    "FoldResult",
    "ModelCandidateConfig",
    "PreprocessingConfig",
    "ModelFamily",
    "MetricName",
    "RunStatus",
    "ImputationStrategy",
    "ScalingStrategy",
]
