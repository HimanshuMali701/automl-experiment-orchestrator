"""Shared typed state definitions and error classifications for LangGraph workflow."""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from typing_extensions import TypedDict

from automl.profiler.schemas import DatasetProfile, TaskType
from automl.recovery.schemas import FailureCategory
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
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    RUNNING = "running"
    EVALUATING = "evaluating"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    REJECTED = "rejected"
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

    # Phase 4: Human-in-the-Loop Approval Checkpoint
    requires_approval: bool
    approval_status: Optional[str]
    approval_feedback: Optional[str]
    approved_plan: Optional[ExperimentConfig]
    current_node: Optional[str]

    # Execution & results
    candidate_runs: List[CandidateRunResult]
    experiment_result: Optional[ExperimentResult]
    best_candidate_name: Optional[str]
    result_explanation: Optional[Any]

    # Diagnostics, error taxonomy & loop control
    thread_id: Optional[str]
    workflow_status: WorkflowStatus
    error_category: ErrorCategory
    error_message: Optional[str]
    retry_count: int
    max_retries: int
    is_recoverable: bool

    # Phase 3: Failure Diagnosis & Adaptive Recovery
    failure_category: Optional[FailureCategory]
    failure_diagnosis: Optional[Dict[str, Any]]
    recovery_actions: List[Dict[str, Any]]
    recovery_attempt_count: int

    # Execution audit trail
    execution_trace: List[str]
