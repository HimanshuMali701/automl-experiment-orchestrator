"""Dataset profiling, schema inference, validation, and task detection module."""

from automl.profiler.loader import load_dataset
from automl.profiler.profiler import profile_dataset
from automl.profiler.schemas import (
    CategoricalSummary,
    ColumnProfile,
    DatasetProfile,
    DataType,
    NumericalSummary,
    ProfileWarning,
    TargetProfile,
    TaskType,
    WarningType,
)
from automl.profiler.task_detector import detect_target_profile
from automl.profiler.validator import validate_dataset

__all__ = [
    "load_dataset",
    "validate_dataset",
    "detect_target_profile",
    "profile_dataset",
    "DatasetProfile",
    "ColumnProfile",
    "TargetProfile",
    "NumericalSummary",
    "CategoricalSummary",
    "ProfileWarning",
    "TaskType",
    "DataType",
    "WarningType",
]
