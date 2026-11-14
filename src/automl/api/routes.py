"""FastAPI route handlers for dataset validation, experiment submission, and status queries."""

from typing import Optional
from fastapi import APIRouter, Body, File, Form, HTTPException, Request, UploadFile, status
from pydantic import ValidationError

from automl.api.jobs import get_job_manager
from automl.api.schemas import (
    ApprovalRequest,
    DatasetValidationResponse,
    ErrorResponse,
    ExperimentJobResponse,
    ExperimentResultResponse,
    HealthResponse,
    JobStatus,
    NodeTimelineItem,
    PlanApprovalView,
    RejectionRequest,
    WorkflowEventView,
    WorkflowInspectorResponse,
)
from automl.workflow.events import get_event_manager
from automl.core.config import get_settings
from automl.core.exceptions import (
    AutoMLError,
    DatasetValidationError,
    ProfilingError,
    ResourceExhaustionError,
)
from automl.core.logging import get_logger
from automl.profiler.loader import load_dataset
from automl.profiler.profiler import profile_dataset
from automl.profiler.schemas import DataType, TaskType
from automl.trainer.schemas import MetricName

logger = get_logger("api.routes")
router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    description="Check the operational health of the AutoML API and view background job counts.",
)
def get_health() -> HealthResponse:
    """Return application name, version, and active job capacity."""
    settings = get_settings()
    manager = get_job_manager()
    running, queued, completed = manager.get_job_counts()

    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        version=settings.app_version,
        active_jobs=running,
        queued_jobs=queued,
        completed_jobs=completed,
    )


@router.post(
    "/datasets/validate",
    response_model=DatasetValidationResponse,
    summary="Validate and Profile Dataset",
    description="Upload a CSV dataset to inspect column types, missingness, data quality warnings, and infer the ML task type.",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file format or empty data"},
        413: {"model": ErrorResponse, "description": "File exceeds maximum upload size"},
        422: {"model": ErrorResponse, "description": "Validation constraint violation"},
    },
)
async def validate_dataset(
    file: UploadFile = File(..., description="Tabular dataset in CSV format."),
    target_column: Optional[str] = Form(None, description="Optional target column to evaluate."),
    task_type_override: Optional[str] = Form(None, description="Optional task type override (e.g. 'binary_classification', 'regression')."),
) -> DatasetValidationResponse:
    """Validate uploaded dataset and produce non-destructive profile."""
    settings = get_settings()

    # 1. Validate file extension
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Only standard tabular CSV files (.csv) are accepted.",
        )

    # 2. Read bytes and enforce size bounds
    content = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file size ({len(content) / (1024*1024):.1f} MB) exceeds maximum allowed limit of {settings.max_upload_size_mb} MB.",
        )

    if len(content.strip()) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded CSV file is empty.",
        )

    # 3. Parse task override if provided
    parsed_override: Optional[TaskType] = None
    if task_type_override:
        try:
            parsed_override = TaskType(task_type_override.strip().lower())
        except ValueError:
            valid_tasks = [t.value for t in TaskType]
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid task_type_override '{task_type_override}'. Must be one of: {valid_tasks}",
            )

    # 4. Profile dataset using Phase 1 engine
    try:
        profile = profile_dataset(
            content,
            target_column=target_column.strip() if target_column else None,
            task_type_override=parsed_override,
        )

        detected_task = None
        rec_metric = None
        if profile.target_profile:
            detected_task = profile.target_profile.detected_task_type.value
            # Recommended metrics
            if profile.target_profile.detected_task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
                rec_metric = "f1_macro"
            else:
                rec_metric = "rmse"

        numeric_cols = [c for c, p in profile.columns.items() if p.inferred_type == DataType.NUMERIC]
        cat_cols = [c for c, p in profile.columns.items() if p.inferred_type in (DataType.CATEGORICAL, DataType.BOOLEAN, DataType.TEXT)]

        return DatasetValidationResponse(
            is_valid=True,
            row_count=profile.row_count,
            column_count=profile.column_count,
            target_column=target_column,
            detected_task_type=detected_task,
            recommended_primary_metric=rec_metric,
            data_quality_warnings=[w.message if hasattr(w, "message") else str(w) for w in profile.warnings],
            column_types={c: p.inferred_type.value for c, p in profile.columns.items()},
            missing_value_counts={c: p.missing_count for c, p in profile.columns.items()},
            numerical_columns=numeric_cols,
            categorical_columns=cat_cols,
            duplicate_rows=profile.duplicate_rows,
        )

    except (DatasetValidationError, ProfilingError) as e:
        logger.warning(f"Validation failed for uploaded file '{filename}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected profiling error on file '{filename}': {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while profiling the dataset.",
        )


@router.post(
    "/experiments",
    response_model=ExperimentJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Launch Asynchronous AutoML Experiment",
    description="Enqueue an automated ML experiment to train and evaluate baseline and candidate models in the background.",
    responses={
        202: {"model": ExperimentJobResponse, "description": "Experiment job accepted and queued"},
        400: {"model": ErrorResponse, "description": "Invalid input parameters or CSV"},
        413: {"model": ErrorResponse, "description": "File exceeds size limits"},
        422: {"model": ErrorResponse, "description": "Validation constraint violation"},
        429: {"model": ErrorResponse, "description": "Server at maximum concurrent job capacity"},
    },
)
async def create_experiment(
    request: Request,
    file: UploadFile = File(..., description="Training dataset in CSV format."),
    target_column: str = Form(..., description="Target column name to predict."),
    use_llm_planner: bool = Form(False, description="Whether to use LLM-based plan proposer (defaults to deterministic plan)."),
    primary_metric: Optional[str] = Form(None, description="Optional primary ranking metric (e.g. 'f1_macro', 'rmse')."),
    task_type_override: Optional[str] = Form(None, description="Optional task type override."),
    cv_folds: int = Form(5, description="Number of cross-validation folds (2 to 10)."),
    test_size: float = Form(0.2, description="Held-out test split fraction (0.05 to 0.50)."),
    random_seed: int = Form(42, description="Random seed for reproducibility."),
    requires_approval: bool = Form(False, description="Whether to pause workflow for human approval before model training."),
) -> ExperimentJobResponse:
    """Validate parameters, load dataset, and launch experiment in background thread."""
    settings = get_settings()

    # 1. Parameter bounds validation
    if not target_column or not target_column.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Field 'target_column' must not be empty.",
        )
    target_column = target_column.strip()

    if cv_folds < 2 or cv_folds > 10:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"cv_folds must be between 2 and 10 (got {cv_folds}).",
        )

    if test_size < 0.05 or test_size > 0.50:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"test_size must be between 0.05 and 0.50 (got {test_size}).",
        )

    # 2. Metric and Task Type parsing
    parsed_metric: Optional[MetricName] = None
    if primary_metric:
        try:
            parsed_metric = MetricName(primary_metric.strip().lower())
        except ValueError:
            valid_metrics = [m.value for m in MetricName]
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid primary_metric '{primary_metric}'. Must be one of: {valid_metrics}",
            )

    parsed_task: Optional[TaskType] = None
    if task_type_override:
        try:
            parsed_task = TaskType(task_type_override.strip().lower())
        except ValueError:
            valid_tasks = [t.value for t in TaskType]
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid task_type_override '{task_type_override}'. Must be one of: {valid_tasks}",
            )

    # 3. File validation
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Only CSV files (.csv) are supported.",
        )

    content = await file.read()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file exceeds {settings.max_upload_size_mb} MB limit.",
        )

    # 4. Load dataset
    try:
        df = load_dataset(content)
    except DatasetValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to load dataset: {str(e)}",
        )

    if target_column not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Target column '{target_column}' does not exist in dataset. Available columns: {list(df.columns)}",
        )

    # 5. Enqueue Job
    manager = get_job_manager()
    try:
        job = manager.submit_job(
            dataset_df=df,
            target_column=target_column,
            use_llm_planner=use_llm_planner,
            primary_metric=parsed_metric,
            task_type_override=parsed_task,
            cv_folds=cv_folds,
            test_size=test_size,
            random_seed=random_seed,
            requires_approval=requires_approval,
        )
    except ResourceExhaustionError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e),
        )

    base_url = str(request.base_url).rstrip("/")
    status_url = f"{base_url}/experiments/{job.experiment_id}"
    results_url = f"{base_url}/experiments/{job.experiment_id}/results"
    approval_url = f"{base_url}/experiments/{job.experiment_id}/approval" if job.requires_approval else None

    return ExperimentJobResponse(
        experiment_id=job.experiment_id,
        status=job.status,
        target_column=job.target_column,
        task_type=job.task_type,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        execution_time_seconds=job.execution_time_seconds,
        error_message=job.error_message,
        status_url=status_url,
        results_url=results_url,
        user_constraints=job.user_constraints,
        effective_config=job.effective_config,
        requires_approval=job.requires_approval,
        approval_status=job.approval_status,
        approval_url=approval_url,
        current_stage=job.current_stage,
        execution_trace=job.execution_trace,
        current_node=job.current_node,
        completed_nodes=job.completed_nodes,
        pending_approval=job.pending_approval,
    )


@router.get(
    "/experiments/{experiment_id}",
    response_model=ExperimentJobResponse,
    summary="Get Experiment Status",
    description="Check the current execution status, timings, and progress of a background experiment job.",
    responses={
        404: {"model": ErrorResponse, "description": "Experiment ID not found"},
    },
)
def get_experiment_status(experiment_id: str, request: Request) -> ExperimentJobResponse:
    """Retrieve job metadata and lifecycle status."""
    manager = get_job_manager()
    job = manager.get_job(experiment_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment '{experiment_id}' was not found.",
        )

    base_url = str(request.base_url).rstrip("/")
    status_url = f"{base_url}/experiments/{job.experiment_id}"
    results_url = f"{base_url}/experiments/{job.experiment_id}/results"
    approval_url = f"{base_url}/experiments/{job.experiment_id}/approval" if job.requires_approval else None

    return ExperimentJobResponse(
        experiment_id=job.experiment_id,
        status=job.status,
        target_column=job.target_column,
        task_type=job.task_type,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        execution_time_seconds=job.execution_time_seconds,
        error_message=job.error_message,
        status_url=status_url,
        results_url=results_url,
        user_constraints=job.user_constraints,
        effective_config=job.effective_config,
        requires_approval=job.requires_approval,
        approval_status=job.approval_status,
        approval_url=approval_url,
        current_stage=job.current_stage,
        execution_trace=job.execution_trace,
        current_node=job.current_node,
        completed_nodes=job.completed_nodes,
        pending_approval=job.pending_approval,
    )


@router.get(
    "/experiments/{experiment_id}/results",
    response_model=ExperimentResultResponse,
    summary="Get Experiment Results",
    description="Retrieve the complete ranked model leaderboard, cross-validation scores, held-out test metrics, and MLflow tracking run IDs.",
    responses={
        200: {"model": ExperimentResultResponse, "description": "Completed experiment benchmark results"},
        404: {"model": ErrorResponse, "description": "Experiment ID not found"},
        409: {"model": ErrorResponse, "description": "Job is still queued or running"},
        422: {"model": ErrorResponse, "description": "Job execution failed"},
    },
)
def get_experiment_results(experiment_id: str) -> ExperimentResultResponse:
    """Retrieve final results and model benchmark leaderboard."""
    manager = get_job_manager()
    job = manager.get_job(experiment_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment '{experiment_id}' was not found.",
        )

    if job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Experiment '{experiment_id}' is currently in state '{job.status.value}'. Please poll GET /experiments/{experiment_id} until completed.",
        )

    if job.status == JobStatus.FAILED:
        err = job.error_message or "Experiment execution failed."
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Experiment '{experiment_id}' failed: {err}",
        )

    if not job.result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Experiment '{experiment_id}' marked completed but result payload is missing.",
        )

    return job.result


@router.get(
    "/experiments/{experiment_id}/approval",
    response_model=PlanApprovalView,
    summary="Get Pending Experiment Plan for Review",
    description="Retrieve the proposed experiment plan, model candidates, hyperparameters, and validation warnings when an experiment is awaiting human approval.",
    responses={
        200: {"model": PlanApprovalView, "description": "Proposed experiment plan pending approval"},
        404: {"model": ErrorResponse, "description": "Experiment ID not found"},
        409: {"model": ErrorResponse, "description": "Job is not in waiting_for_approval status"},
    },
)
def get_pending_plan_for_approval(experiment_id: str) -> PlanApprovalView:
    """Retrieve proposed plan configuration and validation summary awaiting human review."""
    manager = get_job_manager()
    job = manager.get_job(experiment_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment '{experiment_id}' was not found.",
        )

    try:
        return manager.get_pending_approval(experiment_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )


@router.post(
    "/experiments/{experiment_id}/approve",
    response_model=ExperimentJobResponse,
    summary="Approve Experiment Plan",
    description="Approve the proposed plan to resume workflow and model execution. Optionally supply validated hyperparameter overrides.",
    responses={
        200: {"model": ExperimentJobResponse, "description": "Experiment plan approved; execution resumed"},
        404: {"model": ErrorResponse, "description": "Experiment ID not found"},
        409: {"model": ErrorResponse, "description": "Job is not in waiting_for_approval status or already processed"},
        422: {"model": ErrorResponse, "description": "Disallowed or invalid hyperparameter overrides"},
    },
)
def approve_experiment_plan(
    experiment_id: str,
    request: Request,
    payload: Optional[ApprovalRequest] = Body(None),
) -> ExperimentJobResponse:
    """Approve a pending experiment plan, validate any overrides against catalogue allowlists, and resume execution."""
    manager = get_job_manager()
    job = manager.get_job(experiment_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment '{experiment_id}' was not found.",
        )

    overrides = payload.hyperparameter_overrides if payload else None
    feedback = payload.feedback if payload else None

    try:
        updated_job = manager.approve_job(
            experiment_id=experiment_id,
            overrides=overrides,
            feedback=feedback,
        )
    except ValueError as e:
        err_msg = str(e)
        if "not found" in err_msg.lower():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=err_msg)
        elif "cannot be approved" in err_msg.lower() or "status" in err_msg.lower():
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=err_msg)
        else:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=err_msg)

    base_url = str(request.base_url).rstrip("/")
    status_url = f"{base_url}/experiments/{updated_job.experiment_id}"
    results_url = f"{base_url}/experiments/{updated_job.experiment_id}/results"
    approval_url = f"{base_url}/experiments/{updated_job.experiment_id}/approval"

    return ExperimentJobResponse(
        experiment_id=updated_job.experiment_id,
        status=updated_job.status,
        target_column=updated_job.target_column,
        task_type=updated_job.task_type,
        created_at=updated_job.created_at,
        started_at=updated_job.started_at,
        completed_at=updated_job.completed_at,
        execution_time_seconds=updated_job.execution_time_seconds,
        error_message=updated_job.error_message,
        status_url=status_url,
        results_url=results_url,
        user_constraints=updated_job.user_constraints,
        effective_config=updated_job.effective_config,
        requires_approval=updated_job.requires_approval,
        approval_status=updated_job.approval_status,
        approval_url=approval_url,
        current_stage=updated_job.current_stage,
        execution_trace=updated_job.execution_trace,
        current_node=updated_job.current_node,
        completed_nodes=updated_job.completed_nodes,
        pending_approval=updated_job.pending_approval,
    )


@router.post(
    "/experiments/{experiment_id}/reject",
    response_model=ExperimentJobResponse,
    summary="Reject Experiment Plan",
    description="Reject the proposed plan and stop the experiment workflow without executing candidate models.",
    responses={
        200: {"model": ExperimentJobResponse, "description": "Experiment plan rejected; job terminated"},
        404: {"model": ErrorResponse, "description": "Experiment ID not found"},
        409: {"model": ErrorResponse, "description": "Job is not in waiting_for_approval status"},
    },
)
def reject_experiment_plan(
    experiment_id: str,
    request: Request,
    payload: Optional[RejectionRequest] = Body(None),
) -> ExperimentJobResponse:
    """Reject a pending experiment plan, preventing candidate model execution and marking job as rejected."""
    manager = get_job_manager()
    job = manager.get_job(experiment_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment '{experiment_id}' was not found.",
        )

    feedback = payload.feedback if payload else None

    try:
        updated_job = manager.reject_job(
            experiment_id=experiment_id,
            feedback=feedback,
        )
    except ValueError as e:
        err_msg = str(e)
        if "not found" in err_msg.lower():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=err_msg)
        else:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=err_msg)

    base_url = str(request.base_url).rstrip("/")
    status_url = f"{base_url}/experiments/{updated_job.experiment_id}"
    results_url = f"{base_url}/experiments/{updated_job.experiment_id}/results"
    approval_url = f"{base_url}/experiments/{updated_job.experiment_id}/approval"

    return ExperimentJobResponse(
        experiment_id=updated_job.experiment_id,
        status=updated_job.status,
        target_column=updated_job.target_column,
        task_type=updated_job.task_type,
        created_at=updated_job.created_at,
        started_at=updated_job.started_at,
        completed_at=updated_job.completed_at,
        execution_time_seconds=updated_job.execution_time_seconds,
        error_message=updated_job.error_message,
        status_url=status_url,
        results_url=results_url,
        user_constraints=updated_job.user_constraints,
        effective_config=updated_job.effective_config,
        requires_approval=updated_job.requires_approval,
        approval_status=updated_job.approval_status,
        approval_url=approval_url,
        current_stage=updated_job.current_stage,
        execution_trace=updated_job.execution_trace,
        current_node=updated_job.current_node,
        completed_nodes=updated_job.completed_nodes,
        pending_approval=updated_job.pending_approval,
    )


@router.get(
    "/experiments/{experiment_id}/events",
    response_model=list[WorkflowEventView],
    summary="Get Workflow Events",
    description="Retrieve the chronological sequence of fine-grained LangGraph execution events for an experiment.",
    responses={
        200: {"description": "List of recorded workflow execution events"},
        404: {"model": ErrorResponse, "description": "Experiment ID not found"},
    },
)
def get_workflow_events(experiment_id: str) -> list[WorkflowEventView]:
    """Retrieve incremental real-time events for an active or completed experiment."""
    manager = get_job_manager()
    job = manager.get_job(experiment_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment '{experiment_id}' was not found.",
        )

    event_mgr = get_event_manager()
    events = event_mgr.get_events(experiment_id)
    return [
        WorkflowEventView(
            event_id=ev.event_id,
            experiment_id=ev.experiment_id,
            timestamp=ev.timestamp,
            event_type=ev.event_type.value,
            node=ev.node,
            stage=ev.stage,
            status=ev.status,
            duration_seconds=ev.duration_seconds,
            summary=ev.summary,
            details=ev.details,
        )
        for ev in events
    ]


@router.get(
    "/experiments/{experiment_id}/inspector",
    response_model=WorkflowInspectorResponse,
    summary="Get Workflow Inspector Data",
    description="Retrieve comprehensive telemetry, node timelines, routing decisions, LLM reflections, and failure recovery history.",
    responses={
        200: {"model": WorkflowInspectorResponse, "description": "Aggregated workflow inspector data"},
        404: {"model": ErrorResponse, "description": "Experiment ID not found"},
    },
)
def get_workflow_inspector(experiment_id: str) -> WorkflowInspectorResponse:
    """Retrieve aggregated telemetry data for the Agent Workflow Inspector."""
    manager = get_job_manager()
    job = manager.get_job(experiment_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment '{experiment_id}' was not found.",
        )

    event_mgr = get_event_manager()
    raw_events = event_mgr.get_events(experiment_id)
    events_view = [
        WorkflowEventView(
            event_id=ev.event_id,
            experiment_id=ev.experiment_id,
            timestamp=ev.timestamp,
            event_type=ev.event_type.value,
            node=ev.node,
            stage=ev.stage,
            status=ev.status,
            duration_seconds=ev.duration_seconds,
            summary=ev.summary,
            details=ev.details,
        )
        for ev in raw_events
    ]

    timeline_data = event_mgr.get_timeline(experiment_id)
    timeline_view = [
        NodeTimelineItem(
            node=item["node"],
            stage=item["stage"],
            status=item["status"],
            start_time=item.get("start_time"),
            end_time=item.get("end_time"),
            duration_seconds=item.get("duration_seconds"),
            summary=item.get("summary"),
        )
        for item in timeline_data
    ]

    routing_decisions = event_mgr.get_routing_decisions(experiment_id)
    llm_interactions = event_mgr.get_llm_interactions(experiment_id)
    recovery_history = event_mgr.get_recovery_history(experiment_id)

    return WorkflowInspectorResponse(
        experiment_id=job.experiment_id,
        status=job.status,
        current_node=job.current_node,
        current_stage=job.current_stage,
        completed_nodes=job.completed_nodes,
        pending_approval=job.pending_approval,
        execution_trace=job.execution_trace,
        events=events_view,
        timeline=timeline_view,
        routing_decisions=routing_decisions,
        llm_interactions=llm_interactions,
        recovery_history=recovery_history,
    )
