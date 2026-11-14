"""Thread-safe background experiment job manager and persistence store."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple
import uuid

import pandas as pd
from pydantic import BaseModel, Field

from automl.api.schemas import (
    CandidatePlanView,
    CandidateSummaryResponse,
    ExperimentJobResponse,
    ExperimentResultResponse,
    JobStatus,
    PlanApprovalView,
)
from automl.core.config import Settings, get_settings
from automl.core.exceptions import ResourceExhaustionError
from automl.core.logging import get_logger
from automl.profiler.schemas import TaskType
from automl.recovery.diagnoser import ALLOWED_HYPERPARAMETERS
from automl.trainer.schemas import ExperimentConfig, MetricName, ModelFamily
from automl.workflow import (
    NODE_TO_STAGE,
    WorkflowStatus,
    resume_orchestration_workflow,
    run_orchestration_workflow,
)

logger = get_logger("api.jobs")


def validate_and_apply_hyperparameter_overrides(
    plan: ExperimentConfig,
    overrides: Dict[str, Dict[str, Any]],
) -> ExperimentConfig:
    """Validate and apply candidate hyperparameter overrides against allowlists.

    Args:
        plan: Existing ExperimentConfig.
        overrides: Mapping of candidate_name or model_family to hyperparameter dict.

    Returns:
        ExperimentConfig: New ExperimentConfig with validated overrides applied.

    Raises:
        ValueError: If a candidate is not found, a hyperparameter is disallowed,
                    or values violate expected types or bounds.
    """
    if not overrides:
        return plan

    # Work on a deep copy of plan
    plan_dict = plan.model_dump()
    updated_plan = ExperimentConfig.model_validate(plan_dict)

    candidate_map = {c.name: c for c in updated_plan.candidates}
    family_map = {c.model_family.value: c for c in updated_plan.candidates}

    for identifier, hp_updates in overrides.items():
        cand = candidate_map.get(identifier) or family_map.get(identifier)
        if not cand:
            available = list(candidate_map.keys()) + list(family_map.keys())
            raise ValueError(
                f"Candidate '{identifier}' not found in proposed plan. Available candidates: {available}"
            )

        family = cand.model_family
        allowed_spec = ALLOWED_HYPERPARAMETERS.get(family)
        if not allowed_spec:
            raise ValueError(f"No allowlisted hyperparameters found for model family '{family.value}'.")

        for param_name, param_val in hp_updates.items():
            if param_name not in allowed_spec:
                valid_params = list(allowed_spec.keys())
                raise ValueError(
                    f"Disallowed hyperparameter '{param_name}' for model '{cand.name}' ({family.value}). "
                    f"Permitted hyperparameters: {valid_params}"
                )

            expected_type, min_val, max_val = allowed_spec[param_name]

            # Type conversion/validation
            if expected_type == float and isinstance(param_val, int) and not isinstance(param_val, bool):
                param_val = float(param_val)

            if not isinstance(param_val, expected_type) or (expected_type == int and isinstance(param_val, bool)):
                raise ValueError(
                    f"Hyperparameter '{param_name}' for '{cand.name}' must be of type {expected_type.__name__} "
                    f"(got {type(param_val).__name__} with value {param_val})."
                )

            if min_val is not None and param_val < min_val:
                raise ValueError(
                    f"Hyperparameter '{param_name}' for '{cand.name}' must be >= {min_val} (got {param_val})."
                )

            if max_val is not None and param_val > max_val:
                raise ValueError(
                    f"Hyperparameter '{param_name}' for '{cand.name}' must be <= {max_val} (got {param_val})."
                )

            # Apply validated hyperparameter
            cand.hyperparameters[param_name] = param_val

    return updated_plan


class JobRecord(BaseModel):
    """Internal model capturing complete job lifecycle and execution output."""
    experiment_id: str
    status: JobStatus = JobStatus.QUEUED
    target_column: str
    task_type: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    execution_time_seconds: Optional[float] = None
    error_message: Optional[str] = None
    result: Optional[ExperimentResultResponse] = None
    use_llm_planner: bool = False
    user_constraints: Optional[Dict[str, Any]] = None
    effective_config: Optional[Dict[str, Any]] = None

    # Phase 4 Human-in-the-Loop approval state
    requires_approval: bool = False
    approval_status: Optional[str] = None
    approval_feedback: Optional[str] = None
    approved_plan: Optional[Dict[str, Any]] = None
    pending_plan: Optional[Dict[str, Any]] = None
    validation_summary: Optional[Dict[str, Any]] = None
    dataset_path: Optional[str] = None

    # Phase 5 Observability & Inspector state
    current_stage: Optional[str] = None
    execution_trace: List[str] = Field(default_factory=list)
    current_node: Optional[str] = None
    completed_nodes: List[str] = Field(default_factory=list)
    pending_approval: bool = False


class JobManager:
    """Production-minded local background job execution engine and state manager."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.max_concurrent_jobs = self.settings.max_concurrent_jobs
        self._lock = threading.Lock()
        self._jobs: Dict[str, JobRecord] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_concurrent_jobs,
            thread_name_prefix="automl_worker",
        )
        self.jobs_dir: Path = self.settings.artifacts_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir: Path = self.settings.artifacts_dir / "datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self._load_persisted_jobs()
        logger.info(f"Initialized JobManager (max_concurrent_jobs={self.max_concurrent_jobs})")

    def _load_persisted_jobs(self) -> None:
        """Hydrate in-memory job records from disk on startup."""
        for job_file in self.jobs_dir.glob("*.json"):
            try:
                with open(job_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    job = JobRecord.model_validate(data)
                    # Startup hygiene: in-flight jobs from prior process lifetimes are marked terminated
                    if job.status in (JobStatus.RUNNING, JobStatus.QUEUED):
                        job.status = JobStatus.FAILED
                        job.error_message = "Job was interrupted by prior server shutdown."
                    self._jobs[job.experiment_id] = job
            except Exception as e:
                logger.warning(f"Failed to load persisted job file {job_file.name}: {str(e)}")

    def _persist_job(self, job: JobRecord) -> None:
        """Save a job record to disk for persistence and auditability."""
        job_file = self.jobs_dir / f"{job.experiment_id}.json"
        try:
            with open(job_file, "w", encoding="utf-8") as f:
                f.write(job.model_dump_json(indent=2))
        except Exception as e:
            logger.error(f"Failed to persist job record {job.experiment_id}: {str(e)}")

    def get_job(self, experiment_id: str) -> Optional[JobRecord]:
        """Retrieve a job by its unique experiment ID."""
        with self._lock:
            return self._jobs.get(experiment_id)

    def get_job_counts(self) -> Tuple[int, int, int]:
        """Return (active_running, queued, completed) counts."""
        with self._lock:
            running = sum(1 for j in self._jobs.values() if j.status in (JobStatus.RUNNING, JobStatus.WAITING_FOR_APPROVAL))
            queued = sum(1 for j in self._jobs.values() if j.status == JobStatus.QUEUED)
            completed = sum(1 for j in self._jobs.values() if j.status == JobStatus.COMPLETED)
            return running, queued, completed

    def submit_job(
        self,
        dataset_df: pd.DataFrame,
        target_column: str,
        use_llm_planner: bool = False,
        primary_metric: Optional[MetricName] = None,
        task_type_override: Optional[TaskType] = None,
        cv_folds: int = 5,
        test_size: float = 0.2,
        random_seed: int = 42,
        requires_approval: bool = False,
        max_models: Optional[int] = None,
    ) -> JobRecord:
        """Validate resource capacity and enqueue a new experiment job."""
        with self._lock:
            running = sum(1 for j in self._jobs.values() if j.status == JobStatus.RUNNING)
            queued = sum(1 for j in self._jobs.values() if j.status == JobStatus.QUEUED)
            if (running + queued) >= self.max_concurrent_jobs:
                logger.warning(
                    f"Job submission rejected: concurrency limit reached "
                    f"(running={running}, queued={queued}, max={self.max_concurrent_jobs})"
                )
                raise ResourceExhaustionError(
                    f"Server is currently at capacity ({running + queued} active jobs). "
                    f"Please retry after existing experiments complete.",
                    details={"running": running, "queued": queued, "limit": self.max_concurrent_jobs},
                )

            experiment_id = str(uuid.uuid4())
            now_str = datetime.now(timezone.utc).isoformat()

            # Persist dataset to disk so state survives restarts
            dataset_file = self.datasets_dir / f"{experiment_id}.csv"
            try:
                dataset_df.to_csv(dataset_file, index=False)
            except Exception as e:
                logger.warning(f"Could not persist dataset file for {experiment_id}: {str(e)}")

            raw_constraints = {
                "primary_metric": primary_metric.value if primary_metric else None,
                "cv_folds": cv_folds,
                "test_size": test_size,
                "random_seed": random_seed,
                "max_models": max_models,
            }
            initial_user_constraints = {k: v for k, v in raw_constraints.items() if v is not None}

            job = JobRecord(
                experiment_id=experiment_id,
                status=JobStatus.QUEUED,
                target_column=target_column,
                task_type=task_type_override.value if task_type_override else None,
                created_at=now_str,
                use_llm_planner=use_llm_planner,
                user_constraints=initial_user_constraints,
                requires_approval=requires_approval,
                approval_status="pending" if requires_approval else None,
                dataset_path=str(dataset_file) if dataset_file.exists() else None,
                current_stage="Queued",
                execution_trace=[],
                current_node=None,
                completed_nodes=[],
                pending_approval=False,
            )
            self._jobs[experiment_id] = job
            self._persist_job(job)

        # Enqueue execution in thread pool
        self._executor.submit(
            self._execute_job,
            experiment_id=experiment_id,
            dataset_df=dataset_df,
            target_column=target_column,
            use_llm_planner=use_llm_planner,
            primary_metric=primary_metric,
            task_type_override=task_type_override,
            cv_folds=cv_folds,
            test_size=test_size,
            random_seed=random_seed,
            requires_approval=requires_approval,
            max_models=max_models,
        )

        logger.info(f"Enqueued experiment job: {experiment_id} (requires_approval={requires_approval})")
        return job

    def _execute_job(
        self,
        experiment_id: str,
        dataset_df: pd.DataFrame,
        target_column: str,
        use_llm_planner: bool,
        primary_metric: Optional[MetricName],
        task_type_override: Optional[TaskType],
        cv_folds: int,
        test_size: float,
        random_seed: int,
        requires_approval: bool,
        max_models: Optional[int] = None,
    ) -> None:
        """Worker executing workflow in background thread."""
        user_constraints: Dict[str, Any] = {
            "primary_metric": primary_metric,
            "cv_folds": cv_folds,
            "test_size": test_size,
            "random_seed": random_seed,
            "max_models": max_models,
        }

        with self._lock:
            job = self._jobs[experiment_id]
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(timezone.utc).isoformat()
            job.current_stage = "Starting"
            self._persist_job(job)

        start_time = datetime.now(timezone.utc)
        logger.info(f"Starting execution of job {experiment_id} for target '{target_column}'")

        def on_step(node_name: str, node_update: Dict[str, Any]) -> None:
            with self._lock:
                j = self._jobs.get(experiment_id)
                if not j:
                    return
                j.current_node = node_name
                j.current_stage = NODE_TO_STAGE.get(node_name, node_name)
                if node_name not in j.completed_nodes:
                    j.completed_nodes.append(node_name)
                if "execution_trace" in node_update:
                    j.execution_trace = list(node_update["execution_trace"])
                elif node_name not in j.execution_trace:
                    j.execution_trace.append(node_name)
                self._persist_job(j)

        try:
            final_state = run_orchestration_workflow(
                dataset_source=dataset_df,
                target_column=target_column,
                task_type_override=task_type_override,
                use_llm_planner=use_llm_planner,
                user_constraints=user_constraints,
                requires_approval=requires_approval,
                thread_id=experiment_id,
                on_step=on_step,
            )

            # Check if workflow paused waiting for human approval
            if final_state.get("workflow_status") == WorkflowStatus.WAITING_FOR_APPROVAL:
                with self._lock:
                    job = self._jobs[experiment_id]
                    job.status = JobStatus.WAITING_FOR_APPROVAL
                    job.approval_status = "pending"
                    job.pending_approval = True
                    job.current_node = "approval_gate"
                    job.current_stage = "Human Approval Checkpoint"
                    if "approval_gate" not in job.completed_nodes:
                        job.completed_nodes.append("approval_gate")
                    if final_state.get("execution_trace"):
                        job.execution_trace = list(final_state["execution_trace"])
                    plan = final_state.get("plan")
                    if plan:
                        job.pending_plan = plan.model_dump()
                        job.task_type = (
                            plan.task_type.value
                            if hasattr(plan.task_type, "value")
                            else str(plan.task_type)
                        )
                        job.effective_config = {
                            "primary_metric": (
                                plan.primary_metric.value
                                if hasattr(plan.primary_metric, "value")
                                else str(plan.primary_metric)
                            ),
                            "cv_folds": plan.cv_folds,
                            "test_size": plan.test_size,
                            "random_seed": plan.random_seed,
                        }
                    job.validation_summary = final_state.get("plan_metadata") or {}
                    self._persist_job(job)
                logger.info(f"Job {experiment_id} entered WAITING_FOR_APPROVAL; background thread released.")
                return

            self._handle_completed_state(experiment_id, final_state, start_time)

        except Exception as e:
            logger.exception(f"Unhandled error in job {experiment_id}: {str(e)}")
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            with self._lock:
                job = self._jobs[experiment_id]
                job.status = JobStatus.FAILED
                job.completed_at = end_time.isoformat()
                job.execution_time_seconds = round(duration, 3)
                job.error_message = f"Experiment processing failed: {str(e)}"
                job.current_stage = "Failed"
                job.pending_approval = False
                self._persist_job(job)

    def _resume_job(
        self,
        experiment_id: str,
        approved_plan: Optional[ExperimentConfig],
        approval_feedback: Optional[str],
    ) -> None:
        """Worker resuming workflow following human approval in background thread."""
        start_time = datetime.now(timezone.utc)
        logger.info(f"Resuming execution of job {experiment_id} following approval.")

        def on_step(node_name: str, node_update: Dict[str, Any]) -> None:
            with self._lock:
                j = self._jobs.get(experiment_id)
                if not j:
                    return
                j.current_node = node_name
                j.current_stage = NODE_TO_STAGE.get(node_name, node_name)
                if node_name not in j.completed_nodes:
                    j.completed_nodes.append(node_name)
                if "execution_trace" in node_update:
                    j.execution_trace = list(node_update["execution_trace"])
                elif node_name not in j.execution_trace:
                    j.execution_trace.append(node_name)
                self._persist_job(j)

        try:
            final_state = resume_orchestration_workflow(
                thread_id=experiment_id,
                approved_plan=approved_plan,
                approval_status="approved",
                approval_feedback=approval_feedback,
                on_step=on_step,
            )
            self._handle_completed_state(experiment_id, final_state, start_time)
        except Exception as e:
            logger.exception(f"Unhandled error in resumed job {experiment_id}: {str(e)}")
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            with self._lock:
                job = self._jobs[experiment_id]
                job.status = JobStatus.FAILED
                job.completed_at = end_time.isoformat()
                job.execution_time_seconds = round(duration, 3)
                job.error_message = f"Resumed experiment processing failed: {str(e)}"
                job.current_stage = "Failed"
                job.pending_approval = False
                self._persist_job(job)

    def _handle_completed_state(
        self,
        experiment_id: str,
        final_state: Dict[str, Any],
        start_time: datetime,
    ) -> None:
        """Process terminal workflow state and construct final job result."""
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        with self._lock:
            job = self._jobs[experiment_id]
            job.completed_at = end_time.isoformat()
            job.execution_time_seconds = round(duration, 3)
            job.current_node = None
            job.pending_approval = False
            if final_state.get("execution_trace"):
                job.execution_trace = list(final_state["execution_trace"])

            plan = final_state.get("plan")
            if plan:
                job.effective_config = {
                    "primary_metric": (
                        plan.primary_metric.value
                        if hasattr(plan.primary_metric, "value")
                        else str(plan.primary_metric)
                    ),
                    "cv_folds": plan.cv_folds,
                    "test_size": plan.test_size,
                    "random_seed": plan.random_seed,
                }

            if final_state["workflow_status"] == WorkflowStatus.SUCCESS:
                exp_res = final_state.get("experiment_result")
                if exp_res:
                    leaderboard_summaries = []
                    for cand in exp_res.leaderboard:
                        cand_family = (
                            cand.model_family.value
                            if hasattr(cand.model_family, "value")
                            else str(cand.model_family)
                        )
                        cand_status = (
                            cand.status.value
                            if hasattr(cand.status, "value")
                            else str(cand.status)
                        )
                        leaderboard_summaries.append(
                            CandidateSummaryResponse(
                                candidate_name=cand.candidate_name,
                                model_family=cand_family,
                                is_baseline=cand.is_baseline,
                                is_champion=(cand.candidate_name == exp_res.best_candidate_name),
                                status=cand_status,
                                cv_mean_metrics=cand.cv_mean_metrics,
                                cv_std_metrics=cand.cv_std_metrics,
                                test_metrics=(
                                    cand.test_metrics
                                    if (cand.candidate_name == exp_res.best_candidate_name)
                                    else None
                                ),
                                fit_time_seconds=cand.fit_time_seconds,
                            )
                        )

                    champion_score = None
                    metric_key = (
                        exp_res.primary_metric.value
                        if hasattr(exp_res.primary_metric, "value")
                        else str(exp_res.primary_metric)
                    )
                    task_key = (
                        exp_res.task_type.value
                        if hasattr(exp_res.task_type, "value")
                        else str(exp_res.task_type)
                    )

                    if exp_res.best_candidate_name:
                        champ_cand = next(
                            (c for c in exp_res.leaderboard if c.candidate_name == exp_res.best_candidate_name),
                            None,
                        )
                        if champ_cand:
                            if champ_cand.test_metrics:
                                champion_score = champ_cand.test_metrics.get(metric_key)
                                if champion_score is None and len(champ_cand.test_metrics) > 0:
                                    champion_score = next(iter(champ_cand.test_metrics.values()))
                            if champion_score is None and champ_cand.cv_mean_metrics:
                                champion_score = champ_cand.cv_mean_metrics.get(metric_key)
                                if champion_score is None and len(champ_cand.cv_mean_metrics) > 0:
                                    champion_score = next(iter(champ_cand.cv_mean_metrics.values()))

                    explanation_dict = None
                    if final_state.get("result_explanation"):
                        explanation_dict = final_state["result_explanation"].model_dump()

                    job.result = ExperimentResultResponse(
                        experiment_id=experiment_id,
                        status=JobStatus.COMPLETED,
                        target_column=exp_res.target_column,
                        task_type=task_key,
                        primary_metric=metric_key,
                        best_candidate_name=exp_res.best_candidate_name,
                        champion_test_score=champion_score,
                        train_rows=exp_res.train_rows,
                        test_rows=exp_res.test_rows,
                        total_execution_time_seconds=exp_res.total_execution_time_seconds,
                        mlflow_run_id=exp_res.mlflow_run_id or f"run-{experiment_id[:8]}",
                        mlflow_experiment_id=exp_res.mlflow_experiment_id or "0",
                        leaderboard=leaderboard_summaries,
                        result_explanation=explanation_dict,
                        warnings=exp_res.warnings,
                        effective_config=job.effective_config,
                    )
                    job.task_type = task_key

                job.status = JobStatus.COMPLETED
                job.current_stage = "Completed"
                logger.info(f"Job {experiment_id} successfully completed in {duration:.2f}s")

            elif final_state["workflow_status"] == WorkflowStatus.REJECTED:
                job.status = JobStatus.REJECTED
                job.current_stage = "Rejected"
                job.current_node = "approval_gate"
                job.error_message = final_state.get("error_message") or "Plan was rejected by user."
                logger.info(f"Job {experiment_id} marked as REJECTED.")
            else:
                err_msg = final_state.get("error_message") or "Workflow failed without explicit message"
                job.status = JobStatus.FAILED
                job.current_stage = "Failed"
                job.error_message = err_msg
                logger.warning(f"Job {experiment_id} workflow failed: {err_msg}")

            self._persist_job(job)

    def get_pending_approval(self, experiment_id: str) -> PlanApprovalView:
        """Retrieve pending experiment plan for human review and approval."""
        with self._lock:
            job = self._jobs.get(experiment_id)
            if not job:
                raise ValueError(f"Experiment '{experiment_id}' was not found.")
            if job.status != JobStatus.WAITING_FOR_APPROVAL or not job.pending_plan:
                raise ValueError(
                    f"Experiment '{experiment_id}' is not waiting for approval (current status: '{job.status.value}')."
                )

            plan_dict = job.pending_plan
            candidates = []
            for c in plan_dict.get("candidates", []):
                fam = c.get("model_family")
                fam_str = fam.value if hasattr(fam, "value") else str(fam)
                candidates.append(
                    CandidatePlanView(
                        candidate_name=c.get("name", ""),
                        model_family=fam_str,
                        is_baseline=c.get("is_baseline", False),
                        hyperparameters=c.get("hyperparameters", {}),
                        description=c.get("description"),
                    )
                )

            pm = plan_dict.get("primary_metric")
            pm_str = pm.value if hasattr(pm, "value") else str(pm)
            tt = job.task_type or plan_dict.get("task_type", "")
            tt_str = tt.value if hasattr(tt, "value") else str(tt)

            return PlanApprovalView(
                experiment_id=experiment_id,
                status=job.status,
                approval_status=job.approval_status or "pending",
                task_type=tt_str,
                target_column=job.target_column,
                primary_metric=pm_str,
                cv_folds=plan_dict.get("cv_folds", 5),
                test_size=plan_dict.get("test_size", 0.2),
                random_seed=plan_dict.get("random_seed", 42),
                candidates=candidates,
                validation_summary=job.validation_summary or {},
                user_constraints=job.user_constraints,
                approval_feedback=job.approval_feedback,
            )

    def approve_job(
        self,
        experiment_id: str,
        overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        feedback: Optional[str] = None,
    ) -> JobRecord:
        """Approve a pending experiment plan, optionally applying validated hyperparameter overrides."""
        with self._lock:
            job = self._jobs.get(experiment_id)
            if not job:
                raise ValueError(f"Experiment '{experiment_id}' was not found.")
            if job.status != JobStatus.WAITING_FOR_APPROVAL or not job.pending_plan:
                raise ValueError(
                    f"Experiment '{experiment_id}' is in status '{job.status.value}' and cannot be approved."
                )

            raw_plan = ExperimentConfig.model_validate(job.pending_plan)
            # Validate and apply overrides against catalogue allowlists
            approved_plan = validate_and_apply_hyperparameter_overrides(raw_plan, overrides or {})

            job.status = JobStatus.RUNNING
            job.approval_status = "approved"
            job.approval_feedback = feedback
            job.approved_plan = approved_plan.model_dump()
            self._persist_job(job)

        # Enqueue resumption in worker thread pool
        self._executor.submit(
            self._resume_job,
            experiment_id=experiment_id,
            approved_plan=approved_plan,
            approval_feedback=feedback,
        )
        return job

    def reject_job(
        self,
        experiment_id: str,
        feedback: Optional[str] = None,
    ) -> JobRecord:
        """Reject a pending experiment plan, preventing model execution."""
        with self._lock:
            job = self._jobs.get(experiment_id)
            if not job:
                raise ValueError(f"Experiment '{experiment_id}' was not found.")
            if job.status != JobStatus.WAITING_FOR_APPROVAL:
                raise ValueError(
                    f"Experiment '{experiment_id}' is in status '{job.status.value}' and cannot be rejected."
                )

            now_str = datetime.now(timezone.utc).isoformat()
            job.status = JobStatus.REJECTED
            job.approval_status = "rejected"
            job.approval_feedback = feedback
            job.completed_at = now_str
            job.error_message = feedback or "Experiment plan was rejected by user."
            self._persist_job(job)

        # Update checkpointer so the paused thread records rejection
        try:
            resume_orchestration_workflow(
                thread_id=experiment_id,
                approval_status="rejected",
                approval_feedback=feedback,
            )
        except Exception as e:
            logger.warning(f"Error updating checkpointer for rejected job {experiment_id}: {str(e)}")

        return job

    def shutdown(self) -> None:
        """Gracefully terminate thread pool."""
        logger.info("Shutting down JobManager thread pool...")
        self._executor.shutdown(wait=False)


# Global singleton instance
_job_manager: Optional[JobManager] = None
_job_manager_lock = threading.Lock()


def get_job_manager(settings: Optional[Settings] = None) -> JobManager:
    """Return the application JobManager singleton.

    If the current manager's executor has been terminated, re-creates a fresh instance.
    """
    global _job_manager
    with _job_manager_lock:
        if _job_manager is None or getattr(_job_manager._executor, "_shutdown", False):
            _job_manager = JobManager(settings=settings)
        return _job_manager


def reset_job_manager() -> None:
    """Reset the global JobManager instance cleanly."""
    global _job_manager
    with _job_manager_lock:
        if _job_manager is not None:
            try:
                _job_manager.shutdown()
            except Exception:
                pass
            _job_manager = None
