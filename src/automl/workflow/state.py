"""Shared typed state definitions and error classifications for LangGraph workflow."""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from typing_extensions import TypedDict

from automl.profiler.schemas import DatasetProfile, TaskType
from automl.trainer.schemas import (
    CandidateRunResult,
    ExperimentConfig,
    ExperimentResult,
    MetricName,
)


class WorkflowStatus(str, Enum):
    """Lifecycle status of the experiment workflow."""
    INITIALIZED = "initialized"
    PROFILING = "profiling"
    PLANNING = "planning"
    RUNNING = "running"
    EVALUATING = "evaluating"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"


class ErrorCategory(str, Enum):
    """Structured taxonomy of errors occurring during graph execution."""
    NONE = "none"
    INVALID_DATA = "invalid_data"
    INVALID_PLAN = "invalid_plan"
    RECOVERABLE_RUNTIME = "recoverable_runtime"
    FATAL_RUNTIME = "fatal_runtime"


class UserConstraints(TypedDict, total=False):
    """User-specified constraints and overrides for experiment planning and training."""
    primary_metric: Optional[Union[MetricName, str]]
    cv_folds: Optional[int]
    test_size: Optional[float]
    random_seed: Optional[int]
    max_models: Optional[int]


class ExperimentWorkflowState(TypedDict):
    """Shared state container passed between LangGraph nodes.

    Large datasets and fitted models are kept out of state copies;
    references, filepaths, and compact summaries are used instead.
    """
    # Dataset references & metadata
    dataset_ref: str
    target_column: Optional[str]
    task_type_override: Optional[TaskType]
    dataset_metadata: Dict[str, Any]

    # User-specified constraints & overrides
    user_constraints: Optional[Dict[str, Any]]

    # Analysis & planning artifacts
    profile: Optional[DatasetProfile]
    plan: Optional[ExperimentConfig]
    plan_metadata: Dict[str, Any]
    use_llm_planner: bool

    # Phase 2: Reflection & Plan Repair
    validation_errors: List[str]
    plan_retry_count: int
    max_plan_retries: int
    current_proposal: Optional[Dict[str, Any]]
    llm_ref: Optional[str]
    llm_instance: Optional[Any]

    # Execution & results
    candidate_runs: List[CandidateRunResult]
    experiment_result: Optional[ExperimentResult]
    best_candidate_name: Optional[str]
    result_explanation: Optional[Any]

    # Diagnostics, error taxonomy & loop control
    workflow_status: WorkflowStatus
    error_category: ErrorCategory
    error_message: Optional[str]
    retry_count: int
    max_retries: int
    is_recoverable: bool

    # Execution audit trail
    execution_trace: List[str]
