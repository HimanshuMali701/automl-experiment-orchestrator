"""Pydantic request, response, and error schemas for the AutoML REST API."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from automl.profiler.schemas import TaskType
from automl.trainer.schemas import MetricName


class JobStatus(str, Enum):
    """Execution status states for background experiment jobs."""
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"


class HealthResponse(BaseModel):
    """API health and service status response."""
    status: str = Field(default="healthy", description="Service health indicator.")
    app_name: str = Field(..., description="Application name.")
    version: str = Field(..., description="Application version.")
    active_jobs: int = Field(default=0, description="Currently running experiment jobs.")
    queued_jobs: int = Field(default=0, description="Jobs waiting in execution queue.")
    completed_jobs: int = Field(default=0, description="Total completed jobs.")


class DatasetValidationResponse(BaseModel):
    """Structured response for dataset profiling and validation."""
    is_valid: bool = Field(default=True, description="Whether the dataset is valid for ML training.")
    row_count: int = Field(..., description="Total number of rows.")
    column_count: int = Field(..., description="Total number of columns.")
    target_column: Optional[str] = Field(default=None, description="Analyzed target column name.")
    detected_task_type: Optional[str] = Field(default=None, description="Inferred or specified task type.")
    recommended_primary_metric: Optional[str] = Field(default=None, description="Recommended evaluation metric.")
    data_quality_warnings: List[str] = Field(default_factory=list, description="Diagnostic warnings.")
    column_types: Dict[str, str] = Field(default_factory=dict, description="Inferred types per column.")
    missing_value_counts: Dict[str, int] = Field(default_factory=dict, description="Missing count per column.")
    numerical_columns: List[str] = Field(default_factory=list, description="Identified numerical feature columns.")
    categorical_columns: List[str] = Field(default_factory=list, description="Identified categorical feature columns.")
    duplicate_rows: int = Field(default=0, description="Count of duplicate rows.")


class ExperimentJobResponse(BaseModel):
    """Metadata response describing the state and progress of an experiment job."""
    experiment_id: str = Field(..., description="Unique UUID identifier for the experiment job.")
    status: JobStatus = Field(..., description="Current job lifecycle status.")
    target_column: str = Field(..., description="Prediction target column name.")
    task_type: Optional[str] = Field(default=None, description="Task type (classification or regression).")
    created_at: str = Field(..., description="ISO 8601 creation timestamp.")
    started_at: Optional[str] = Field(default=None, description="ISO 8601 execution start timestamp.")
    completed_at: Optional[str] = Field(default=None, description="ISO 8601 completion or failure timestamp.")
    execution_time_seconds: Optional[float] = Field(default=None, description="Total execution duration in seconds.")
    error_message: Optional[str] = Field(default=None, description="Sanitized failure description if failed.")
    status_url: str = Field(..., description="URL to poll for job progress.")
    results_url: str = Field(..., description="URL to retrieve results once completed.")
    user_constraints: Optional[Dict[str, Any]] = Field(default=None, description="User-specified experiment constraints.")
    effective_config: Optional[Dict[str, Any]] = Field(default=None, description="Effective parameters applied during execution.")
    requires_approval: bool = Field(default=False, description="Whether this job pauses for human approval.")
    approval_status: Optional[str] = Field(default=None, description="Current approval status: pending, approved, or rejected.")
    approval_url: Optional[str] = Field(default=None, description="URL to view and act on pending plan approval.")

    # Phase 5 Observability & Inspector fields
    current_stage: Optional[str] = Field(default=None, description="Human-readable current workflow stage.")
    execution_trace: List[str] = Field(default_factory=list, description="Ordered trace of executed workflow steps.")
    current_node: Optional[str] = Field(default=None, description="Currently active LangGraph node.")
    completed_nodes: List[str] = Field(default_factory=list, description="List of completed node names in execution order.")
    pending_approval: bool = Field(default=False, description="Whether the workflow is currently blocked waiting for approval.")


class CandidatePlanView(BaseModel):
    """Configuration view for an individual candidate model in a proposed plan."""
    candidate_name: str = Field(..., description="Unique name of the candidate model.")
    model_family: str = Field(..., description="Model family identifier.")
    is_baseline: bool = Field(..., description="Whether this model serves as the baseline benchmark.")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Configured hyperparameters.")
    description: Optional[str] = Field(default=None, description="Model candidate description or rationale.")


class PlanApprovalView(BaseModel):
    """Detailed experiment plan view exposed to human reviewers for approval or overrides."""
    experiment_id: str = Field(..., description="Unique experiment ID.")
    status: JobStatus = Field(..., description="Current job lifecycle status.")
    approval_status: str = Field(..., description="Approval status ('pending', 'approved', 'rejected').")
    task_type: str = Field(..., description="Detected ML task type.")
    target_column: str = Field(..., description="Target column name.")
    primary_metric: str = Field(..., description="Primary ranking metric.")
    cv_folds: int = Field(..., description="Number of cross-validation folds.")
    test_size: float = Field(..., description="Held-out test split ratio.")
    random_seed: int = Field(..., description="Random seed for reproducibility.")
    candidates: List[CandidatePlanView] = Field(default_factory=list, description="Proposed candidate models.")
    validation_summary: Dict[str, Any] = Field(default_factory=dict, description="Validation report and rationale.")
    user_constraints: Optional[Dict[str, Any]] = Field(default=None, description="User-specified constraints.")
    approval_feedback: Optional[str] = Field(default=None, description="Reviewer feedback or notes.")


class ApprovalRequest(BaseModel):
    """Payload for human approval submission with optional hyperparameter overrides."""
    feedback: Optional[str] = Field(default=None, description="Optional reviewer feedback or notes.")
    hyperparameter_overrides: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Optional candidate-specific hyperparameter overrides, e.g. {'logistic_regression': {'C': 0.5}}.",
    )


class RejectionRequest(BaseModel):
    """Payload for rejecting an experiment plan."""
    feedback: Optional[str] = Field(default=None, description="Reason or feedback for plan rejection.")


class CandidateSummaryResponse(BaseModel):
    """Benchmark performance summary for a single model candidate."""
    candidate_name: str = Field(..., description="User-friendly name of the candidate model.")
    model_family: str = Field(..., description="Model family identifier.")
    is_baseline: bool = Field(..., description="Whether this candidate is a baseline.")
    is_champion: bool = Field(default=False, description="Whether this candidate was chosen as champion.")
    status: str = Field(..., description="Training status (success or failed).")
    cv_mean_metrics: Dict[str, float] = Field(default_factory=dict, description="Mean CV metrics.")
    cv_std_metrics: Dict[str, float] = Field(default_factory=dict, description="Standard deviation CV metrics.")
    test_metrics: Optional[Dict[str, float]] = Field(default=None, description="Held-out test metrics (champion only).")
    fit_time_seconds: float = Field(..., description="Training duration in seconds.")


class ExperimentResultResponse(BaseModel):
    """Complete structured benchmark results and leaderboard."""
    experiment_id: str = Field(..., description="Unique experiment identifier.")
    status: JobStatus = Field(default=JobStatus.COMPLETED, description="Job status.")
    target_column: str = Field(..., description="Target column name.")
    task_type: str = Field(..., description="Task type.")
    primary_metric: str = Field(..., description="Primary ranking metric.")
    best_candidate_name: Optional[str] = Field(default=None, description="Champion model name.")
    champion_test_score: Optional[float] = Field(default=None, description="Held-out test score for primary metric.")
    train_rows: int = Field(..., description="Training partition row count.")
    test_rows: int = Field(..., description="Held-out test partition row count.")
    total_execution_time_seconds: float = Field(..., description="Total execution duration in seconds.")
    mlflow_run_id: Optional[str] = Field(default=None, description="Parent MLflow run ID.")
    mlflow_experiment_id: Optional[str] = Field(default=None, description="MLflow experiment ID.")
    leaderboard: List[CandidateSummaryResponse] = Field(default_factory=list, description="Ranked model leaderboard.")
    result_explanation: Optional[Dict[str, Any]] = Field(default=None, description="Structured explanation.")
    warnings: List[str] = Field(default_factory=list, description="Execution or data warnings.")
    effective_config: Optional[Dict[str, Any]] = Field(default=None, description="Effective parameters applied during execution.")


class ErrorResponse(BaseModel):
    """Standardized API error response envelope."""
    detail: str = Field(..., description="Human-readable error explanation.")
    error_type: str = Field(..., description="Machine-readable error type identifier.")


class WorkflowEventView(BaseModel):
    """Structured view of a single LangGraph execution event."""
    event_id: str = Field(..., description="Unique event identifier.")
    experiment_id: str = Field(..., description="Experiment UUID.")
    timestamp: str = Field(..., description="ISO 8601 event timestamp.")
    event_type: str = Field(..., description="Event type category.")
    node: Optional[str] = Field(default=None, description="Graph node name.")
    stage: Optional[str] = Field(default=None, description="Human-readable workflow stage.")
    status: Optional[str] = Field(default=None, description="Execution status.")
    duration_seconds: Optional[float] = Field(default=None, description="Node execution duration in seconds.")
    summary: str = Field(..., description="Sanitized human-readable event summary.")
    details: Dict[str, Any] = Field(default_factory=dict, description="Sanitized event details.")


class NodeTimelineItem(BaseModel):
    """Timing and status summary for an individual workflow node."""
    node: str = Field(..., description="Node name.")
    stage: str = Field(..., description="Human-readable stage.")
    status: str = Field(..., description="Status: completed, running, failed, waiting_for_approval, pending, or skipped.")
    start_time: Optional[str] = Field(default=None, description="ISO 8601 start timestamp.")
    end_time: Optional[str] = Field(default=None, description="ISO 8601 completion timestamp.")
    duration_seconds: Optional[float] = Field(default=None, description="Execution duration in seconds.")
    summary: Optional[str] = Field(default=None, description="Execution summary.")


class WorkflowInspectorResponse(BaseModel):
    """Comprehensive observability view for the Agent Workflow Inspector."""
    experiment_id: str = Field(..., description="Experiment UUID.")
    status: JobStatus = Field(..., description="Current job status.")
    current_node: Optional[str] = Field(default=None, description="Currently active node.")
    current_stage: Optional[str] = Field(default=None, description="Currently active stage.")
    completed_nodes: List[str] = Field(default_factory=list, description="Completed nodes in order.")
    pending_approval: bool = Field(default=False, description="Whether currently blocked waiting for approval.")
    execution_trace: List[str] = Field(default_factory=list, description="Ordered execution trace.")
    events: List[WorkflowEventView] = Field(default_factory=list, description="All workflow events chronologically.")
    timeline: List[NodeTimelineItem] = Field(default_factory=list, description="Aggregated node execution timeline.")
    routing_decisions: List[Dict[str, Any]] = Field(default_factory=list, description="Evaluated conditional routing decisions.")
    llm_interactions: List[Dict[str, Any]] = Field(default_factory=list, description="LLM proposals and repair reflections.")
    recovery_history: List[Dict[str, Any]] = Field(default_factory=list, description="Failure diagnosis and recovery history.")


